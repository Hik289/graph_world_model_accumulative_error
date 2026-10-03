from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..graph_generators.base import GraphSample
from ..utils.seeding import stable_seed



@dataclass
class SimulatorTrace:

    X: np.ndarray
    A: Optional[np.ndarray]
    actions: np.ndarray
    perturbations: List[Optional[np.ndarray]]
    W: np.ndarray
    U: np.ndarray
    Q: Optional[np.ndarray]
    config: Dict[str, Any] = field(default_factory=dict)



def _sample_spec_normalized(shape: Tuple[int, ...], target_norm: float, rng: np.random.Generator) -> np.ndarray:
    fan_in = shape[0]
    M = rng.standard_normal(shape).astype(np.float32) / np.sqrt(max(fan_in, 1))
    if M.ndim == 2:
        s = np.linalg.svd(M, compute_uv=False)
        s_max = float(s[0]) if s.size else 1.0
    else:
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



def _topk_symmetric(P: np.ndarray, k: int) -> np.ndarray:
    N = P.shape[0]
    if k >= N - 1:
        A = np.ones((N, N), dtype=np.float32)
        np.fill_diagonal(A, 0)
        return A
    P_clean = np.where(np.isnan(P), -np.inf, P)
    row_topk = np.argpartition(-P_clean, kth=k, axis=1)[:, :k]
    A_row = np.zeros((N, N), dtype=np.float32)
    rows = np.repeat(np.arange(N), k)
    A_row[rows, row_topk.ravel()] = 1.0
    A_sym = ((A_row + A_row.T) > 0).astype(np.float32)
    np.fill_diagonal(A_sym, 0)
    return A_sym


def _softmax_rows(S: np.ndarray) -> np.ndarray:
    s = S - S.max(axis=1, keepdims=True)
    e = np.exp(s)
    return (e / (e.sum(axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def _normalize_adj_self(A: np.ndarray) -> np.ndarray:
    N = A.shape[0]
    A_self = A + np.eye(N, dtype=np.float32)
    d = A_self.sum(axis=1)
    d_safe = np.where(d > 0, d, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(d_safe)
    D_inv_sqrt = np.diag(d_inv_sqrt).astype(np.float32)
    return (D_inv_sqrt @ A_self @ D_inv_sqrt).astype(np.float32)



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
    if mode not in ("fixed_edge", "dynamic_edge"):
        raise ValueError(mode)
    N = graph.N

    if env_seed is None:
        env_seed = stable_seed("env", graph.topology, graph.N, graph.seed)
    env_rng = np.random.default_rng(int(env_seed))
    seed_W_env = env_rng.integers(0, 2 ** 31 - 1)
    seed_U_env = env_rng.integers(0, 2 ** 31 - 1)
    seed_Q_env = env_rng.integers(0, 2 ** 31 - 1)

    base = stable_seed("rollout", graph.topology, graph.N, graph.seed, seed)
    rng_main = np.random.default_rng(base)
    seed_X0 = rng_main.integers(0, 2 ** 31 - 1)
    seed_a = rng_main.integers(0, 2 ** 31 - 1)
    seed_xi = rng_main.integers(0, 2 ** 31 - 1)

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

    rng_X0 = np.random.default_rng(int(seed_X0))
    X0 = rng_X0.standard_normal((N, D)).astype(np.float32)

    rng_a = np.random.default_rng(int(seed_a))
    if action_seq_mode == "zero":
        actions = np.zeros((T, D_a), dtype=np.float32)
    elif action_seq_mode == "piecewise_constant":
        n_segments = max(1, T // 5)
        seg = rng_a.standard_normal((n_segments, D_a)).astype(np.float32)
        actions = np.zeros((T, D_a), dtype=np.float32)
        for i in range(T):
            actions[i] = seg[min(i // 5, n_segments - 1)]
    else:
        actions = np.zeros((T, D_a), dtype=np.float32)
        if T > 0:
            actions[0] = rng_a.standard_normal(D_a)
            for t in range(1, T):
                actions[t] = actions[t - 1] + 0.1 * rng_a.standard_normal(D_a)
        actions = actions.astype(np.float32)

    rng_xi = np.random.default_rng(int(seed_xi))
    if mode == "fixed_edge":
        noise_traj = (rng_xi.standard_normal((T, N, D)).astype(np.float32) * sigma_noise)
    else:
        noise_traj = (rng_xi.standard_normal((T, N, D)).astype(np.float32) * sigma_noise)

    def _build_action_inject(action_mode_t: str, inj_node: Optional[int]) -> np.ndarray:
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

    inj_map: Dict[int, Dict[str, Any]] = {}
    if injection_schedule is not None:
        for (t, kw) in injection_schedule:
            inj_map[int(t)] = dict(kw)

    X_traj = np.zeros((T + 1, N, D), dtype=np.float32)
    X_traj[0] = X0
    if mode == "dynamic_edge":
        A_traj = np.zeros((T + 1, N, N), dtype=np.float32)
        A_traj[0] = graph.A_dense
    else:
        A_traj = None

    perturbations: List[Optional[np.ndarray]] = []

    A_norm_static = graph.A_norm

    if mode == "dynamic_edge":
        if k_dyn is None:
            avg_deg = float(graph.A_dense.sum() / max(N, 1))
            k_dyn = max(1, int(round(avg_deg)))
        A_curr = graph.A_dense.copy()
        A_norm_curr = _normalize_adj_self(A_curr)
    else:
        A_norm_curr = A_norm_static
        A_curr = graph.A_dense

    for t in range(T):
        kw_t = inj_map.get(t, {})
        mode_t = kw_t.get("action_mode", action_mode)
        inj_node_t = kw_t.get("injection_node_id", injection_node_id)
        pert_t: Optional[np.ndarray] = kw_t.get("perturbation", None)

        mask_node = _build_action_inject(mode_t, inj_node_t)

        Ua = U @ actions[t]
        action_term = np.outer(mask_node, Ua).astype(np.float32)

        message = A_norm_curr @ X_traj[t] @ W
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

        if mode == "dynamic_edge":
            S = X_next @ X_next.T
            P = _softmax_rows(S @ Q)
            A_new = _topk_symmetric(P, k=k_dyn)
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
