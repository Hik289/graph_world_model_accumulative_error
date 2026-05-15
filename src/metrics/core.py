"""核心 metrics: NodeMSE / EdgeF1 / GraphDist / AffectedNodes / GrowthSlope.

对接 ``data/specs/metrics.md`` §1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import warnings


# ---------------------------------------------------------------------------
# 输入数据接口
# ---------------------------------------------------------------------------

@dataclass
class RolloutPrediction:
    """Metrics 输入: 一条 ground-truth + 一条 prediction trajectory.

    数据形状:
      X_true / X_pred : (T+1, N, D)
      A_true / A_pred : (T+1, N, N) 或 (N, N) 若 fixed-edge
    """

    X_true: np.ndarray
    A_true: np.ndarray
    X_pred: np.ndarray
    A_pred: np.ndarray
    horizons: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    is_fixed_edge: bool = True
    edge_types_true: Optional[np.ndarray] = None         # (T+1, N, N) int8 多分类
    edge_types_pred: Optional[np.ndarray] = None         # (T+1, N, N) int8 OR (T+1, N, N, K) soft
    node_types: Optional[np.ndarray] = None              # (N,) int8
    rewards_true: Optional[np.ndarray] = None
    rewards_pred: Optional[np.ndarray] = None
    actions_true: Optional[np.ndarray] = None
    actions_pred: Optional[np.ndarray] = None
    J_oracle: Optional[float] = None
    J_pred_policy: Optional[float] = None


# ---------------------------------------------------------------------------
# NodeMSE
# ---------------------------------------------------------------------------

def node_mse(pred: RolloutPrediction, H: int) -> float:
    """(1/(N·D)) · ‖X_pred[H] − X_true[H]‖_F²."""
    T_plus_1 = pred.X_true.shape[0]
    if H >= T_plus_1:
        warnings.warn(f"node_mse: H={H} >= T+1={T_plus_1}; returning NaN")
        return float("nan")
    if H == 0:
        # sanity check
        return float(np.mean((pred.X_pred[0] - pred.X_true[0]) ** 2))
    diff = pred.X_pred[H] - pred.X_true[H]
    return float(np.mean(diff ** 2))


# ---------------------------------------------------------------------------
# EdgeF1 — binary
# ---------------------------------------------------------------------------

def _get_A_at_H(A: np.ndarray, H: int) -> np.ndarray:
    if A.ndim == 2:
        return A
    return A[H]


def edge_f1_binary(pred: RolloutPrediction, H: int, threshold: float = 0.5) -> float:
    A_true = _get_A_at_H(pred.A_true, H)
    A_pred = _get_A_at_H(pred.A_pred, H)
    N = A_true.shape[0]
    # binarize prediction
    A_p = (A_pred >= threshold).astype(np.int8)
    A_t = (A_true >= threshold).astype(np.int8)
    # 无向 → upper triangle
    # 通过 symmetry 检测 directedness (粗略): 这里默认对所有 case 用 full
    iu = np.triu_indices(N, k=1)
    p = A_p[iu]
    t = A_t[iu]
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if prec + rec < 1e-12:
        return 0.0
    return float(2 * prec * rec / (prec + rec + 1e-12))


def edge_f1_multiclass(pred: RolloutPrediction, H: int) -> Dict[str, float]:
    """Multiclass edge type F1 (macro / micro)."""
    if pred.edge_types_true is None or pred.edge_types_pred is None:
        return {"macro_f1": float("nan"), "micro_f1": float("nan")}
    T_plus_1 = pred.edge_types_true.shape[0]
    if H >= T_plus_1:
        return {"macro_f1": float("nan"), "micro_f1": float("nan")}
    e_true = pred.edge_types_true[H]
    e_pred = pred.edge_types_pred[H]
    if e_pred.ndim == 4 - 1:  # already int
        e_pred_hard = e_pred
    elif e_pred.ndim == 4:    # softmax (N,N,K)
        e_pred_hard = e_pred.argmax(axis=-1)
    elif e_pred.ndim == 3 and e_true.ndim == 2:
        e_pred_hard = e_pred.argmax(axis=-1)
    else:
        e_pred_hard = e_pred
    K = int(max(e_true.max(), e_pred_hard.max())) + 1
    f1_per_class: List[float] = []
    tp_all = 0
    fp_all = 0
    fn_all = 0
    for c in range(K):
        tp = int(((e_pred_hard == c) & (e_true == c)).sum())
        fp = int(((e_pred_hard == c) & (e_true != c)).sum())
        fn = int(((e_pred_hard != c) & (e_true == c)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        if prec + rec < 1e-12:
            f1_per_class.append(0.0)
        else:
            f1_per_class.append(2 * prec * rec / (prec + rec))
        tp_all += tp
        fp_all += fp
        fn_all += fn
    macro = float(np.mean(f1_per_class)) if f1_per_class else float("nan")
    micro_prec = tp_all / max(tp_all + fp_all, 1)
    micro_rec = tp_all / max(tp_all + fn_all, 1)
    if micro_prec + micro_rec < 1e-12:
        micro = 0.0
    else:
        micro = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)
    return {"macro_f1": macro, "micro_f1": float(micro),
            "per_class_f1": f1_per_class}


# ---------------------------------------------------------------------------
# GraphDist
# ---------------------------------------------------------------------------

def graph_dist(pred: RolloutPrediction, H: int, *,
               alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.5,
               topk_eigvals: int = 20) -> float:
    """节点距离 + 谱距离 + 1-EdgeF1 加权."""
    A_true = _get_A_at_H(pred.A_true, H)
    A_pred = _get_A_at_H(pred.A_pred, H)
    if pred.X_true.shape[0] <= H:
        return float("nan")
    X_true = pred.X_true[H]
    X_pred = pred.X_pred[H]
    N, D = X_true.shape
    node_term = alpha * float(np.linalg.norm(X_pred - X_true) / np.sqrt(N * D))
    # 谱距离
    if N <= 200:
        try:
            ev_t = np.sort(np.real(np.linalg.eigvals(A_true.astype(np.float64))))
            ev_p = np.sort(np.real(np.linalg.eigvals(A_pred.astype(np.float64))))
            spec_term = beta * float(np.linalg.norm(ev_t - ev_p) / np.sqrt(N))
        except Exception:
            spec_term = 0.0
    else:
        # top-k eigvals fallback
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
            k = min(topk_eigvals, N - 2)
            ev_t = spla.eigs(sp.csr_matrix(A_true.astype(np.float64)),
                             k=k, which="LM", return_eigenvectors=False)
            ev_p = spla.eigs(sp.csr_matrix(A_pred.astype(np.float64)),
                             k=k, which="LM", return_eigenvectors=False)
            ev_t = np.sort(np.abs(ev_t))
            ev_p = np.sort(np.abs(ev_p))
            spec_term = beta * float(np.linalg.norm(ev_t - ev_p) / np.sqrt(N))
        except Exception:
            spec_term = 0.0
    # 1 - EdgeF1
    f1 = edge_f1_binary(pred, H, threshold=0.5)
    edit_term = gamma * (1.0 - f1)
    return node_term + spec_term + edit_term


# ---------------------------------------------------------------------------
# AffectedNodes
# ---------------------------------------------------------------------------

def affected_nodes(pred: RolloutPrediction, H: int, tau_aff: float = 0.5,
                   sigma_signal: float = 1.0) -> float:
    """(1/N)·#{ v : ‖X_pred[H,v] − X_true[H,v]‖₂ > tau·σ_signal }."""
    if pred.X_true.shape[0] <= H:
        return float("nan")
    diff = pred.X_pred[H] - pred.X_true[H]
    node_norm = np.linalg.norm(diff, axis=-1)
    thresh = tau_aff * sigma_signal
    return float((node_norm > thresh).mean())


# ---------------------------------------------------------------------------
# GrowthSlope
# ---------------------------------------------------------------------------

def growth_slope(pred: RolloutPrediction, H1: int = 4, H2: int = 32) -> float:
    mse_1 = node_mse(pred, H1)
    mse_2 = node_mse(pred, H2)
    if mse_1 < 1e-8 or mse_2 < 1e-8 or np.isnan(mse_1) or np.isnan(mse_2):
        warnings.warn(f"growth_slope: degenerate MSE (H1={mse_1}, H2={mse_2})")
        return 0.0
    return float((np.log(mse_2) - np.log(mse_1)) / (H2 - H1))


# ---------------------------------------------------------------------------
# Planning metrics
# ---------------------------------------------------------------------------

def return_error(pred: RolloutPrediction, H: int, gamma: float = 0.95) -> float:
    """|Σ γ^k (r_pred(k) − r_true(k))|, k = 0..H-1."""
    if pred.rewards_true is None or pred.rewards_pred is None:
        return float("nan")
    H_eff = min(H, pred.rewards_true.shape[0], pred.rewards_pred.shape[0])
    if H_eff <= 0:
        return 0.0
    discounts = np.array([gamma ** k for k in range(H_eff)], dtype=np.float64)
    diff = (pred.rewards_pred[:H_eff] - pred.rewards_true[:H_eff]).astype(np.float64)
    return float(abs(float((discounts * diff).sum())))


def regret(pred: RolloutPrediction, normalize: bool = False, eps: float = 1e-6) -> float:
    if pred.J_oracle is None or pred.J_pred_policy is None:
        return float("nan")
    r = float(pred.J_oracle - pred.J_pred_policy)
    if normalize:
        return r / max(abs(pred.J_oracle), eps)
    return r


def action_mismatch(pred: RolloutPrediction, H: int) -> float:
    if pred.actions_true is None or pred.actions_pred is None:
        return float("nan")
    H_eff = min(H, pred.actions_true.shape[0], pred.actions_pred.shape[0])
    if H_eff <= 0:
        return 0.0
    if pred.actions_true.ndim == 1:
        # 硬 ID match
        mismatch = (pred.actions_true[:H_eff] != pred.actions_pred[:H_eff]).astype(np.float64)
        return float(mismatch.mean())
    # 连续 fallback: ‖a_pred − a_true‖₂ 求均值
    diff = pred.actions_pred[:H_eff] - pred.actions_true[:H_eff]
    return float(np.linalg.norm(diff, axis=-1).mean())


# ---------------------------------------------------------------------------
# Agent-system metrics
# ---------------------------------------------------------------------------

def task_success_rate(success_flags: np.ndarray) -> Dict[str, float]:
    """SR mean + 95% bootstrap CI (1000 resamples)."""
    success_flags = np.asarray(success_flags).astype(np.float64)
    M = success_flags.shape[0]
    if M == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(0)
    bsamples = rng.choice(success_flags, size=(1000, M), replace=True).mean(axis=1)
    return {
        "mean": float(success_flags.mean()),
        "ci_low": float(np.percentile(bsamples, 2.5)),
        "ci_high": float(np.percentile(bsamples, 97.5)),
    }


def failure_propagation_depth(A: np.ndarray, v_inject: int,
                              error_flags_final: np.ndarray) -> int:
    """从 v_inject 出发, BFS, 找最大可达 error_flag=1 节点距离.

    A : (N, N) 邻接.
    error_flags_final : (N,) {0, 1}.
    """
    N = A.shape[0]
    if v_inject < 0 or v_inject >= N:
        return 0
    visited = np.zeros(N, dtype=bool)
    dist = -np.ones(N, dtype=int)
    queue = [v_inject]
    visited[v_inject] = True
    dist[v_inject] = 0
    max_d = 0
    while queue:
        u = queue.pop(0)
        # 邻居 (无向, 也 OK 对 directed; 用 row)
        neigh = np.where(A[u] > 0)[0]
        for w in neigh:
            if not visited[w]:
                visited[w] = True
                dist[w] = dist[u] + 1
                if error_flags_final[w] > 0.5:
                    if dist[w] > max_d:
                        max_d = int(dist[w])
                queue.append(int(w))
    return int(max_d)


def cost_latency(X_traj: np.ndarray, cost_dim: int = 2, latency_dim: int = 3,
                 exec_thresh: float = 0.5, success_dim: int = 0) -> Dict[str, float]:
    """Cost = Σ exp(cost(v))·1[executed]; Latency 同理.

    X_traj : (T+1, N, D).
    Executed = success_prob > exec_thresh 在某个 t.
    """
    T_plus_1, N, D = X_traj.shape
    success_prob = X_traj[:, :, success_dim]
    executed = (success_prob > exec_thresh).any(axis=0)
    cost = float(np.exp(X_traj[:, :, cost_dim]).sum(axis=0)[executed].sum())
    latency = float(np.exp(X_traj[:, :, latency_dim]).sum(axis=0)[executed].sum())
    return {"cost": cost, "latency": latency, "n_executed": int(executed.sum())}
