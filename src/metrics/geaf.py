"""GEAF (Graph Error Amplification Factor) 估计与 correction 接口.

实现 ``data/specs/metrics.md`` §1.6 + §2.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx


def _spectral_radius(A: np.ndarray, sparse_threshold: int = 200) -> float:
    """ρ(A) = max |λ(A)|."""
    N = A.shape[0]
    if N <= sparse_threshold:
        vals = np.linalg.eigvals(A.astype(np.float64))
        return float(np.max(np.abs(vals)))
    A_sp = sp.csr_matrix(A.astype(np.float64))
    try:
        vals = spla.eigs(A_sp, k=1, which="LM", return_eigenvectors=False, maxiter=2000)
        return float(np.max(np.abs(vals)))
    except spla.ArpackNoConvergence:
        vals = np.linalg.eigvals(A.astype(np.float64))
        return float(np.max(np.abs(vals)))


def geaf_global(A: np.ndarray, model_W: List[np.ndarray]) -> float:
    """ρ(A) · ∏ ‖W_ℓ‖_2."""
    rho = _spectral_radius(A)
    prod = 1.0
    for W in model_W:
        s = np.linalg.svd(W, compute_uv=False)
        prod *= float(s[0])
    return float(rho * prod)


def _pagerank_numpy(A: np.ndarray, alpha: float = 0.85, max_iter: int = 100,
                    tol: float = 1e-6) -> np.ndarray:
    """简易 power iteration PageRank (无向图也 OK)."""
    N = A.shape[0]
    out_deg = A.sum(axis=1)
    out_deg_safe = np.where(out_deg > 0, out_deg, 1.0)
    P = (A / out_deg_safe[:, None]).T   # 列归一化 (转移概率)
    v = np.ones(N) / N
    for _ in range(max_iter):
        v_new = alpha * P @ v + (1 - alpha) / N
        if np.linalg.norm(v_new - v, ord=1) < tol:
            v = v_new
            break
        v = v_new
    return v


def _betweenness_nx(A: np.ndarray) -> np.ndarray:
    N = A.shape[0]
    G = nx.from_numpy_array(A)
    bw = nx.betweenness_centrality(G)
    return np.array([bw[i] for i in range(N)], dtype=np.float64)


def geaf_local(
    A: np.ndarray, model_W: List[np.ndarray], *,
    kind: str = "degree",
) -> np.ndarray:
    """局部 GEAF_hat(v) = c(v) · ∏ ‖W_ℓ‖_2.

    c(v) ∈ {degree, pagerank, betweenness}.
    """
    prod = 1.0
    for W in model_W:
        s = np.linalg.svd(W, compute_uv=False)
        prod *= float(s[0])
    if kind == "degree":
        c = A.sum(axis=1).astype(np.float64)
    elif kind == "pagerank":
        c = _pagerank_numpy(A, alpha=0.85)
    elif kind == "betweenness":
        c = _betweenness_nx(A)
    else:
        raise ValueError(f"unknown kind: {kind}")
    return c * prod


def coupled_B_operator(
    A: np.ndarray,
    model_W: Union[List[np.ndarray], np.ndarray],
    X: Optional[np.ndarray] = None,
    *,
    sigma_lipschitz: float = 1.0,
    Q: Optional[np.ndarray] = None,
    dynamic_edge: bool = False,
    L_g: Optional[float] = None,
    R_X: Optional[float] = None,
    R_A: Optional[float] = None,
) -> np.ndarray:
    """构造 joint node-edge 误差耦合算子 B = [[L_X, L_A], [M_X, M_A]] (2×2).

    按 analysis/insight.md Definition 3 / analysis/b_operator_patch_planned.md 的 tight 版本:

      L_X = L_σ · ‖A‖_2 · ∏_ℓ ‖W_ℓ‖_2
      L_A = L_σ · ∏_ℓ ‖W_ℓ‖_2 · R_X
      M_X = L_g · R_A           (dynamic_edge only; else 0)
      M_A = L_g · R_X           (dynamic_edge only; else 0)

    其中:
      R_X  : 节点状态紧致性 (assumption A1), 默认 ‖X‖_F (或 1.0).
      R_A  : 邻接紧致性, 默认 ‖A‖_F.
      L_g  : edge-update Lipschitz, 默认 L_σ · ‖Q‖_2 (TopK Jacobian surrogate).

    Backward-compat: `model_W` 接受 single ndarray (会包一层 list).
    """
    if isinstance(model_W, np.ndarray):
        model_W_list = [model_W]
    else:
        model_W_list = list(model_W)
    A_norm_2 = float(np.linalg.norm(A.astype(np.float64), ord=2))
    prod_W = 1.0
    for W in model_W_list:
        s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
        prod_W *= float(s[0]) if s.size else 1.0
    if R_X is None:
        if X is not None:
            R_X = float(np.linalg.norm(X.astype(np.float64)))
        else:
            R_X = 1.0
    if R_A is None:
        R_A = float(np.linalg.norm(A.astype(np.float64)))
    L_X = sigma_lipschitz * A_norm_2 * prod_W
    L_A = sigma_lipschitz * prod_W * R_X
    if dynamic_edge and Q is not None:
        if L_g is None:
            L_g = sigma_lipschitz * float(np.linalg.norm(Q.astype(np.float64), ord=2))
        M_X = L_g * R_A
        M_A = L_g * R_X
    else:
        L_g = 0.0
        M_X = 0.0
        M_A = 0.0
    B = np.array([[L_X, L_A], [M_X, M_A]], dtype=np.float64)
    return B


def rho_B_closed_form(B: np.ndarray) -> float:
    """ρ(B) 闭式: (L_X + M_A + sqrt(Δ)) / 2, Δ = (L_X − M_A)² + 4·L_A·M_X.

    适用 2×2 非负 B 矩阵 (theorist 推导).
    """
    L_X, L_A = float(B[0, 0]), float(B[0, 1])
    M_X, M_A = float(B[1, 0]), float(B[1, 1])
    delta = (L_X - M_A) ** 2 + 4.0 * L_A * M_X
    return float((L_X + M_A + np.sqrt(max(delta, 0.0))) / 2.0)


def theory_constants(
    A: np.ndarray,
    model_W: List[np.ndarray],
    X: Optional[np.ndarray] = None,
    *,
    Q: Optional[np.ndarray] = None,
    dynamic_edge: bool = False,
    sigma_lipschitz: float = 1.0,
    L_g: Optional[float] = None,
    R_X: Optional[float] = None,
    R_A: Optional[float] = None,
) -> Dict[str, float]:
    """Compute the theory scalars recorded after P2 training.

    Returns
    -------
    dict with keys:
        L_X, L_A, M_X, M_A : 耦合算子 4 entries
        rho_B              : closed form ρ(B)
        rho_B_eig          : numpy 直接特征值 (sanity 检查)
        GEAF_hat           : ρ(A) · ∏‖W_ℓ‖_2
        rho_A_raw          : raw 邻接谱半径
        rho_A_norm         : 归一化邻接谱半径 (默认 ≈ 1, 仅当 A 已为 A_norm 时给出)
        L_g, R_X, R_A      : 输入 / 推断的 compactness 常量 (备记录)
    """
    B = coupled_B_operator(
        A, model_W, X=X, sigma_lipschitz=sigma_lipschitz,
        Q=Q, dynamic_edge=dynamic_edge,
        L_g=L_g, R_X=R_X, R_A=R_A,
    )
    L_X, L_A = float(B[0, 0]), float(B[0, 1])
    M_X, M_A = float(B[1, 0]), float(B[1, 1])
    rho_b_cf = rho_B_closed_form(B)
    rho_b_eig = float(np.max(np.abs(np.linalg.eigvals(B))))
    geaf = geaf_global(A, model_W if isinstance(model_W, list) else [model_W])
    rho_A_raw = _spectral_radius(A)
    # 推断的 compactness
    R_X_out = R_X if R_X is not None else (float(np.linalg.norm(X)) if X is not None else 1.0)
    R_A_out = R_A if R_A is not None else float(np.linalg.norm(A))
    if dynamic_edge and Q is not None and L_g is None:
        L_g_out = sigma_lipschitz * float(np.linalg.norm(Q, ord=2))
    else:
        L_g_out = L_g if L_g is not None else 0.0
    return {
        "L_X": L_X, "L_A": L_A, "M_X": M_X, "M_A": M_A,
        "rho_B": rho_b_cf,
        "rho_B_eig": rho_b_eig,
        "GEAF_hat": float(geaf),
        "rho_A_raw": float(rho_A_raw),
        "rho_A_norm": None,        # caller 应单独传入 A_norm 算
        "L_g": float(L_g_out),
        "R_X": float(R_X_out),
        "R_A": float(R_A_out),
    }


def rho_B(B: np.ndarray) -> float:
    vals = np.linalg.eigvals(B)
    return float(np.max(np.abs(vals)))


def geaf_correction_score(
    A: np.ndarray, model_W: List[np.ndarray], X_pred: np.ndarray,
    *, uncertainty: Optional[np.ndarray] = None, combine: str = "multiplicative",
    alpha: float = 1.0, kind: str = "degree",
) -> np.ndarray:
    """每节点 correction 优先级分数 (Exp 10/16).

    P1 stub: score = geaf_local · uncertainty^alpha (可选).
    """
    base = geaf_local(A, model_W, kind=kind)
    if uncertainty is None:
        return base
    if combine == "multiplicative":
        return base * (uncertainty ** alpha)
    if combine == "additive":
        return base + alpha * uncertainty
    raise ValueError(combine)
