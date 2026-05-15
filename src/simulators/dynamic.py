"""固定边 + 动态边 simulator.

实现 spec ``data/specs/dynamic_simulator.md``:

* Fixed-edge (FE): X_{t+1} = σ(Ã X_t W + U a_t 1ᵀ) + ξ_t
* Dynamic-edge (DE): A_{t+1} = TopK(σ_edge(X_t X_tᵀ Q))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..graph_generators.base import GraphSample


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class SimulatorTrace:
    """Rollout 轨迹."""

    X: np.ndarray                                        # (T+1, N, D) float32, INCLUDES X_0
    A: Optional[np.ndarray]                              # (T+1, N, N) for DE; None for FE
    actions: np.ndarray                                  # (T, D_a) float32
    perturbations: List[Optional[np.ndarray]]            # len T
    W: np.ndarray                                        # (D, D) ground-truth dynamics
    U: np.ndarray                                        # (D, D_a)
    Q: Optional[np.ndarray]                              # (N, N) for DE only
    config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 工具: 谱归一化矩阵采样
# ---------------------------------------------------------------------------

def _sample_spec_normalized(shape: Tuple[int, ...], target_norm: float, rng: np.random.Generator) -> np.ndarray:
    """采样 G_ij ~ N(0,1) / sqrt(fan_in), 然后 spectral-norm 缩放至 target_norm."""
    fan_in = shape[0]
    M = rng.standard_normal(shape).astype(np.float32) / np.sqrt(max(fan_in, 1))
    # 用最大奇异值 (spectral norm) 缩放
    if M.ndim == 2:
        s = np.linalg.svd(M, compute_uv=False)
        s_max = float(s[0]) if s.size else 1.0
    else:
        # 仅 2D 用; fallback Frobenius
        s_max = float(np.linalg.norm(M))
    if s_max < 1e-8:
        return M
    return (M / s_max * target_norm).astype(np.float32)


def _apply_sigma(z: np.ndarray, sigma: str) -> np.ndarray:
    if sigma == "tanh":
        return np.tanh(z)
    if sigma == "relu":
        return np.maximum(z, 0.0)
    if sigma == "leaky_relu":
        return np.where(z >= 0, z, 0.1 * z)
    raise ValueError(f"unknown sigma: {sigma}")


# ---------------------------------------------------------------------------
# 工具: 动态边 TopK 对称化
# ---------------------------------------------------------------------------

def _topk_symmetric(P: np.ndarray, k: int) -> np.ndarray:
    """行 top-k 取并集 (与转置 OR) 强制对称.

    P : (N, N) 相似度矩阵 (已经做过 softmax 或 sigmoid).
    返回 0/1 对称矩阵 (无自环).

    实现: A_r[i,j] = 1 iff j 在 P[i,:] 的 top-k → A_sym = A_r OR A_r.T.
    这等价于 spec 的 "row top-k ⊕ col top-k" union, 因为 A_r.T[i,j] = A_r[j,i]
    = 1 iff i ∈ top-k of row j = 1 iff j 接收 i 的 top-k message.
    保证 bit-exact 对称, 且 EDA 验证可通过.
    """
    N = P.shape[0]
    if k >= N - 1:
        A = np.ones((N, N), dtype=np.float32)
        np.fill_diagonal(A, 0)
        return A
    # P 不能含 NaN
    P_clean = np.where(np.isnan(P), -np.inf, P)
    # 行 top-k 索引 (取最大 k 个)
    row_topk = np.argpartition(-P_clean, kth=k, axis=1)[:, :k]
    A_row = np.zeros((N, N), dtype=np.float32)
    rows = np.repeat(np.arange(N), k)
    A_row[rows, row_topk.ravel()] = 1.0
    # 对称化
    A_sym = ((A_row + A_row.T) > 0).astype(np.float32)
    np.fill_diagonal(A_sym, 0)
    return A_sym


def _softmax_rows(S: np.ndarray) -> np.ndarray:
    """数值稳定行 softmax."""
    s = S - S.max(axis=1, keepdims=True)
    e = np.exp(s)
    return (e / (e.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def _normalize_adj_self(A: np.ndarray) -> np.ndarray:
    """Ã = D^{-1/2}(A+I)D^{-1/2}, float32."""
    N = A.shape[0]
    A_self = A + np.eye(N, dtype=np.float32)
    d = A_self.sum(axis=1)
    d_safe = np.where(d > 0, d, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(d_safe)
    D_inv_sqrt = np.diag(d_inv_sqrt).astype(np.float32)
    return (D_inv_sqrt @ A_self @ D_inv_sqrt).astype(np.float32)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def rollout(
    graph: GraphSample,
    T: int,
    *,
    mode: str = "fixed_edge",
    action_mode: str = "broadcast",
    sigma: str = "tanh",
    sigma_noise: float = 0.01,
    D: int = 8,
    D_a: int = 4,
    target_w_norm: float = 0.9,
    target_u_norm: float = 0.5,
    target_q_norm: float = 1.0,
    injection_schedule: Optional[List[Tuple[int, Dict[str, Any]]]] = None,
    injection_node_id: Optional[int] = None,
    action_seq_mode: str = "random_walk",
    k_dyn: Optional[int] = None,
    seed: int = 0,
    env_seed: Optional[int] = None,
    W_override: Optional[np.ndarray] = None,
    U_override: Optional[np.ndarray] = None,
    Q_override: Optional[np.ndarray] = None,
) -> SimulatorTrace:
    """对给定 graph 做 T 步 rollout.

    Parameters
    ----------
    graph : 已生成的 GraphSample.
    T : rollout 步数 (X 长度 T+1, action 长度 T).
    mode : "fixed_edge" | "dynamic_edge".
    action_mode : "broadcast" | "action_nodes_only" | "single_node" | "zero".
    sigma : 非线性, "tanh" / "relu" / "leaky_relu".
    sigma_noise : dynamics noise σ.
    D, D_a : 特征维 / action 维.
    target_w_norm, target_u_norm, target_q_norm : 矩阵谱范数目标.
    injection_schedule : list of (t, kwargs) — 在第 t 步对生成器侧加 perturbation.
        kwargs 接受:
          "perturbation": (N, D) ndarray  额外加性 δ
          "action_mode": str  临时切换
          "injection_node_id": int
    injection_node_id : 若 action_mode == "single_node" 时默认作用节点.
    action_seq_mode : "random_walk" | "piecewise_constant" | "zero".
    k_dyn : DE 模式 TopK 的 k; 默认取 graph 平均度的 round.
    seed : 主 seed (派生 X0/actions/noise; 若 env_seed=None 也派生 W/U/Q).
    env_seed : 若给定, 用 env_seed 派生 W/U/Q (per spec dataset_layout §3.1:
        W/U 由 (topology, N, outer_seed) 决定, 不由 inner_seed 决定).
    W_override / U_override / Q_override : 直接覆盖 ground-truth 矩阵 (用于训练
        集合内 trajectory 间共享 W/U 的场景).

    Returns
    -------
    SimulatorTrace
    """
    if mode not in ("fixed_edge", "dynamic_edge"):
        raise ValueError(mode)
    N = graph.N

    # ---------------- env seed (决定 W/U/Q) ----------------
    if env_seed is None:
        # backward-compatible: 用 (graph.topology, graph.N, graph.seed) 派生 env_seed,
        # 这样同 GraphSample 的 W/U/Q 在不同 inner seed 下保持 bit-exact (符合 spec §2.1).
        env_seed = hash(("env", graph.topology, graph.N, graph.seed)) % (2 ** 31)
    env_rng = np.random.default_rng(int(env_seed))
    seed_W_env = env_rng.integers(0, 2 ** 31 - 1)
    seed_U_env = env_rng.integers(0, 2 ** 31 - 1)
    seed_Q_env = env_rng.integers(0, 2 ** 31 - 1)

    # ---------------- inner seed (决定 X0/actions/noise) ----------------
    base = hash(("rollout", graph.topology, graph.N, graph.seed, seed)) % (2 ** 31)
    rng_main = np.random.default_rng(base)
    seed_X0 = rng_main.integers(0, 2 ** 31 - 1)
    seed_a = rng_main.integers(0, 2 ** 31 - 1)
    seed_xi = rng_main.integers(0, 2 ** 31 - 1)

    # ---------------- ground-truth 矩阵 ----------------
    if W_override is not None:
        W = W_override.astype(np.float32)
    else:
        rng_W = np.random.default_rng(int(seed_W_env))
        W = _sample_spec_normalized((D, D), target_w_norm, rng_W)
    if U_override is not None:
        U = U_override.astype(np.float32)
    else:
        rng_U = np.random.default_rng(int(seed_U_env))
        U = _sample_spec_normalized((D, D_a), target_u_norm, rng_U)
    Q: Optional[np.ndarray] = None
    if mode == "dynamic_edge":
        if Q_override is not None:
            Q = Q_override.astype(np.float32)
        else:
            rng_Q = np.random.default_rng(int(seed_Q_env))
            Q = _sample_spec_normalized((N, N), target_q_norm, rng_Q)

    # ---------------- X_0 ----------------
    rng_X0 = np.random.default_rng(int(seed_X0))
    X0 = rng_X0.standard_normal((N, D)).astype(np.float32)

    # ---------------- 动作序列 ----------------
    rng_a = np.random.default_rng(int(seed_a))
    if action_seq_mode == "zero":
        actions = np.zeros((T, D_a), dtype=np.float32)
    elif action_seq_mode == "piecewise_constant":
        n_segments = max(1, T // 5)
        seg = rng_a.standard_normal((n_segments, D_a)).astype(np.float32)
        actions = np.zeros((T, D_a), dtype=np.float32)
        for i in range(T):
            actions[i] = seg[min(i // 5, n_segments - 1)]
    else:  # random_walk
        actions = np.zeros((T, D_a), dtype=np.float32)
        if T > 0:
            actions[0] = rng_a.standard_normal(D_a)
            for t in range(1, T):
                actions[t] = actions[t - 1] + 0.1 * rng_a.standard_normal(D_a)
        actions = actions.astype(np.float32)

    # ---------------- 噪声预采 ----------------
    rng_xi = np.random.default_rng(int(seed_xi))
    if mode == "fixed_edge":
        noise_traj = (rng_xi.standard_normal((T, N, D)).astype(np.float32) * sigma_noise)
    else:
        # DE 模式 N 可能变 (虽然此实现保持 N 不变), 直接用同尺寸
        noise_traj = (rng_xi.standard_normal((T, N, D)).astype(np.float32) * sigma_noise)

    # ---------------- action_mode → action 注入 mask ----------------
    def _build_action_inject(action_mode_t: str, inj_node: Optional[int]) -> np.ndarray:
        """返回 (N,) 0/1 mask, 用于把 U·a_t 广播到部分节点."""
        mask = np.zeros(N, dtype=np.float32)
        if action_mode_t == "broadcast":
            mask[:] = 1.0
        elif action_mode_t == "zero":
            pass
        elif action_mode_t == "action_nodes_only":
            for nid in graph.critical_roles.get("action", []):
                if 0 <= nid < N:
                    mask[nid] = 1.0
        elif action_mode_t == "single_node":
            if inj_node is None:
                raise ValueError("single_node mode requires injection_node_id")
            mask[inj_node] = 1.0
        else:
            raise ValueError(action_mode_t)
        return mask

    # ---------------- 注入 schedule 转 dict ----------------
    inj_map: Dict[int, Dict[str, Any]] = {}
    if injection_schedule is not None:
        for (t, kw) in injection_schedule:
            inj_map[int(t)] = dict(kw)

    # ---------------- 主循环 ----------------
    X_traj = np.zeros((T + 1, N, D), dtype=np.float32)
    X_traj[0] = X0
    if mode == "dynamic_edge":
        A_traj = np.zeros((T + 1, N, N), dtype=np.float32)
        A_traj[0] = graph.A_dense
    else:
        A_traj = None

    perturbations: List[Optional[np.ndarray]] = []

    # 默认 fixed 邻接归一化
    A_norm_static = graph.A_norm

    # DE: 计算 k_dyn
    if mode == "dynamic_edge":
        if k_dyn is None:
            avg_deg = float(graph.A_dense.sum() / max(N, 1))
            k_dyn = max(1, int(round(avg_deg)))
        # 当前邻接 (动态更新)
        A_curr = graph.A_dense.copy()
        A_norm_curr = _normalize_adj_self(A_curr)
    else:
        A_norm_curr = A_norm_static
        A_curr = graph.A_dense

    for t in range(T):
        # 本步 action mode (可能被 injection_schedule 改写)
        kw_t = inj_map.get(t, {})
        mode_t = kw_t.get("action_mode", action_mode)
        inj_node_t = kw_t.get("injection_node_id", injection_node_id)
        pert_t: Optional[np.ndarray] = kw_t.get("perturbation", None)

        mask_node = _build_action_inject(mode_t, inj_node_t)

        # action 项: U·a_t   -> (D,) , 广播到 mask 节点
        Ua = U @ actions[t]                              # (D,)
        action_term = np.outer(mask_node, Ua).astype(np.float32)  # (N, D)

        # message passing
        message = A_norm_curr @ X_traj[t] @ W            # (N, D)
        z = message + action_term
        X_next = _apply_sigma(z, sigma).astype(np.float32) + noise_traj[t]

        if pert_t is not None:
            pert_arr = np.asarray(pert_t, dtype=np.float32)
            if pert_arr.shape != (N, D):
                raise ValueError(f"perturbation shape {pert_arr.shape} != ({N},{D})")
            X_next = X_next + pert_arr
            perturbations.append(pert_arr.copy())
        else:
            perturbations.append(None)

        X_traj[t + 1] = X_next

        # 动态边更新
        if mode == "dynamic_edge":
            S = X_next @ X_next.T                        # (N, N)
            P = _softmax_rows(S @ Q)                     # softmax 行
            A_new = _topk_symmetric(P, k=k_dyn)
            # 兜底连通: 若不连通, 添加最小路径恢复
            # (避免昂贵: 仅当 N 小或随意采样时检查)
            A_curr = A_new
            A_norm_curr = _normalize_adj_self(A_curr)
            A_traj[t + 1] = A_curr

    config = {
        "mode": mode,
        "action_mode": action_mode,
        "sigma": sigma,
        "sigma_noise": sigma_noise,
        "D": D,
        "D_a": D_a,
        "target_w_norm": target_w_norm,
        "target_u_norm": target_u_norm,
        "target_q_norm": target_q_norm,
        "action_seq_mode": action_seq_mode,
        "k_dyn": k_dyn,
        "seed": seed,
        "T": T,
        "N": N,
        "topology": graph.topology,
        "graph_seed": graph.seed,
    }

    return SimulatorTrace(
        X=X_traj,
        A=A_traj,
        actions=actions,
        perturbations=perturbations,
        W=W,
        U=U,
        Q=Q,
        config=config,
    )
