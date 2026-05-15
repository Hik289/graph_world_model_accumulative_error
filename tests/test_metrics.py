"""Metrics 单元测试 (对接 metrics.md §5)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from src.metrics import (
    RolloutPrediction, node_mse, edge_f1_binary, growth_slope,
    geaf_global, geaf_local, task_success_rate, failure_propagation_depth,
    regret, action_mismatch, return_error, correlation_topology_error,
)
from src.metrics.geaf import _spectral_radius


def _make_pred(N=10, T=10, D=4, *, perfect=False, seed=0):
    rng = np.random.default_rng(seed)
    X_true = rng.standard_normal((T + 1, N, D)).astype(np.float32)
    A_true = (rng.random((N, N)) > 0.7).astype(np.float32)
    A_true = ((A_true + A_true.T) > 0).astype(np.float32)
    np.fill_diagonal(A_true, 0)
    if perfect:
        X_pred = X_true.copy()
        A_pred = A_true.copy()
    else:
        X_pred = X_true + rng.standard_normal(X_true.shape).astype(np.float32) * 0.1
        A_pred = A_true.copy()
        # 翻转 5% 边
        mask = rng.random(A_true.shape) < 0.05
        A_pred = np.where(mask, 1 - A_pred, A_pred).astype(np.float32)
        A_pred = ((A_pred + A_pred.T) > 0).astype(np.float32)
        np.fill_diagonal(A_pred, 0)
    return RolloutPrediction(X_true=X_true, A_true=A_true,
                              X_pred=X_pred, A_pred=A_pred,
                              horizons=[1, 2, 4, 8], is_fixed_edge=True)


def test_T1_node_mse_zero_when_perfect():
    pred = _make_pred(perfect=True)
    assert node_mse(pred, 4) < 1e-8


def test_T2_edge_f1():
    pred = _make_pred(perfect=True)
    assert abs(edge_f1_binary(pred, 4) - 1.0) < 1e-6
    # 全错
    pred.A_pred = 1.0 - pred.A_true
    np.fill_diagonal(pred.A_pred, 0)
    assert edge_f1_binary(pred, 4) < 0.1


def test_T3_geaf_star():
    """GEAF(star N=10) = sqrt(9) · ‖W‖_2."""
    import networkx as nx
    A = nx.to_numpy_array(nx.star_graph(9)).astype(np.float32)
    W = np.eye(4, dtype=np.float32) * 2.0  # ‖W‖_2 = 2.0
    g = geaf_global(A, [W])
    expected = math.sqrt(9) * 2.0
    assert abs(g - expected) < 1e-3, f"got {g}, expected {expected}"


def test_T4_rho_complete():
    import networkx as nx
    A = nx.to_numpy_array(nx.complete_graph(10)).astype(np.float32)
    rho = _spectral_radius(A)
    assert abs(rho - 9.0) < 1e-3


def test_T5_regret_nonneg():
    pred = _make_pred()
    pred.J_oracle = 1.0
    pred.J_pred_policy = 0.7
    assert regret(pred) >= 0


def test_T6_fpd_isolated():
    A = np.zeros((10, 10), dtype=np.float32)
    err = np.zeros(10)
    err[5] = 1
    fpd = failure_propagation_depth(A, v_inject=0, error_flags_final=err)
    assert fpd == 0


def test_T7_sr_bootstrap():
    flags = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    out = task_success_rate(flags)
    assert abs(out["mean"] - 0.7) < 1e-8
    assert out["ci_low"] < out["mean"] < out["ci_high"]


def test_T8_correlation_linear():
    """correlation 在已知线性关系上 ≈ 1.0."""
    import pandas as pd
    stats = pd.DataFrame({"topology": ["a"] * 10, "seed": list(range(10)),
                          "rho": list(range(10))})
    errors = pd.DataFrame({"topology": ["a"] * 10, "seed": list(range(10)),
                           "mse": [2 * i + 0.5 for i in range(10)]})
    df = correlation_topology_error(stats, errors, method="pearson")
    row = df[(df["stat"] == "rho") & (df["error_metric"] == "mse")].iloc[0]
    assert row["r"] > 0.99


def test_T9_reproducibility():
    pred1 = _make_pred(seed=42)
    pred2 = _make_pred(seed=42)
    assert np.allclose(pred1.X_pred, pred2.X_pred)
    assert node_mse(pred1, 4) == node_mse(pred2, 4)


def test_growth_slope():
    pred = _make_pred()
    s = growth_slope(pred, H1=2, H2=8)
    # 仅检查 finite
    assert np.isfinite(s) or s == 0.0


def test_geaf_local_kinds():
    import networkx as nx
    A = nx.to_numpy_array(nx.star_graph(9)).astype(np.float32)
    W = [np.eye(4, dtype=np.float32)]
    deg = geaf_local(A, W, kind="degree")
    pr = geaf_local(A, W, kind="pagerank")
    btw = geaf_local(A, W, kind="betweenness")
    # hub 节点 (0) 在三种 kind 上应该都是 max
    assert deg.argmax() == 0
    assert pr.argmax() == 0
    assert btw.argmax() == 0


# ---------------------------------------------------------------------------
# B operator patch tests (per analysis/b_operator_patch_planned.md §5)
# ---------------------------------------------------------------------------

def test_coupled_B_two_layers():
    """model_W=[W1, W2] returns 2×2 matrix."""
    import networkx as nx
    from src.metrics.geaf import coupled_B_operator
    A = nx.to_numpy_array(nx.complete_graph(10)).astype(np.float32)
    W1 = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
    W2 = np.random.default_rng(1).standard_normal((8, 8)).astype(np.float32)
    B = coupled_B_operator(A, [W1, W2])
    assert B.shape == (2, 2)
    # L_X = σ · ‖A‖_2 · ‖W1‖·‖W2‖
    s1 = np.linalg.svd(W1, compute_uv=False)[0]
    s2 = np.linalg.svd(W2, compute_uv=False)[0]
    A_norm_2 = np.linalg.norm(A, ord=2)
    expected_L_X = 1.0 * A_norm_2 * s1 * s2
    assert abs(B[0, 0] - expected_L_X) < 1e-3 * expected_L_X


def test_coupled_B_backward_compat_single_W():
    """Single ndarray W still works (wrapped into list)."""
    import networkx as nx
    from src.metrics.geaf import coupled_B_operator
    A = nx.to_numpy_array(nx.complete_graph(10)).astype(np.float32)
    W = np.eye(8, dtype=np.float32) * 2.0  # ‖W‖_2 = 2
    B = coupled_B_operator(A, W)
    A_norm_2 = float(np.linalg.norm(A, ord=2))
    assert abs(B[0, 0] - A_norm_2 * 2.0) < 1e-3


def test_coupled_B_fixed_edge_block_diag():
    """dynamic_edge=False → M_X = M_A = 0."""
    from src.metrics.geaf import coupled_B_operator
    A = np.eye(5, dtype=np.float32)
    W = np.eye(8, dtype=np.float32)
    B = coupled_B_operator(A, [W], dynamic_edge=False)
    assert B[1, 0] == 0.0
    assert B[1, 1] == 0.0


def test_coupled_B_dynamic_edge_nonzero_M():
    """dynamic_edge=True, Q!=None → M_X, M_A 非零."""
    from src.metrics.geaf import coupled_B_operator
    A = np.eye(5, dtype=np.float32) + np.diag(np.ones(4), k=1)
    A = A + A.T
    A = (A > 0).astype(np.float32) - np.eye(5, dtype=np.float32)
    A = np.clip(A, 0, 1)
    W = np.eye(8, dtype=np.float32)
    Q = np.random.default_rng(0).standard_normal((5, 5)).astype(np.float32)
    X = np.random.default_rng(1).standard_normal((5, 8)).astype(np.float32)
    B = coupled_B_operator(A, [W], X=X, Q=Q, dynamic_edge=True)
    assert B[1, 0] > 0
    assert B[1, 1] > 0


def test_rho_B_closed_form_matches_eigvals():
    """ρ(B) closed form vs numpy eigvals on random 2×2 non-neg matrices."""
    from src.metrics.geaf import rho_B_closed_form, rho_B
    rng = np.random.default_rng(0)
    for _ in range(50):
        B = np.abs(rng.standard_normal((2, 2))).astype(np.float64)
        cf = rho_B_closed_form(B)
        eg = rho_B(B)
        assert abs(cf - eg) < 1e-8 * max(eg, 1.0), f"closed form {cf} != eigvals {eg}"


def test_theory_constants_keys():
    """落盘字典完整含 8 个 mandatory keys."""
    import networkx as nx
    from src.metrics.geaf import theory_constants
    A = nx.to_numpy_array(nx.scale_free_graph(10).to_undirected()).astype(np.float32)
    A = (A > 0).astype(np.float32)
    np.fill_diagonal(A, 0)
    W = np.eye(8, dtype=np.float32)
    tc = theory_constants(A, [W], X=np.zeros((10, 8), dtype=np.float32))
    for key in ["L_X", "L_A", "M_X", "M_A", "rho_B", "GEAF_hat", "rho_A_raw"]:
        assert key in tc, f"missing {key}"


def test_geaf_rho_B_ordering_7_topologies():
    """Proposition T3.2: ρ(B) ordering matches GEAF_hat ordering across 7 topologies."""
    import networkx as nx
    from src.graph_generators import generate
    from src.metrics.geaf import geaf_global, theory_constants
    W = [np.eye(8, dtype=np.float32) * 1.5]
    rho_b_list = []
    geaf_list = []
    for top in ["chain", "tree", "grid", "small_world", "scale_free", "star", "complete"]:
        g = generate(top, N=20, seed=1)
        tc = theory_constants(g.A_dense, W)
        rho_b_list.append(tc["rho_B"])
        geaf_list.append(tc["GEAF_hat"])
    # Spearman: 完全单调
    from scipy.stats import spearmanr
    rho_r = spearmanr(rho_b_list, geaf_list)[0]
    assert rho_r >= 0.95, f"Spearman ρ(rho_B, GEAF) = {rho_r}, expected ≥ 0.95"
