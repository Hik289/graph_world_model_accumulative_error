"""Simulator (dynamic FE/DE + agent + skill) 单元测试."""
from __future__ import annotations

import numpy as np
import pytest

from src.graph_generators import generate
from src.simulators import (
    rollout, generate_calling_tree, simulate_calling_tree,
    generate_skill_graph, simulate_skill_graph,
)


def test_fe_bounded():
    """sanity 1: σ_noise=0, action=zero → ‖X_t‖_F 不爆炸 (T=64)."""
    g = generate("complete", N=50, seed=1)
    trace = rollout(g, T=64, mode="fixed_edge", sigma_noise=0.0,
                    action_seq_mode="zero", seed=0)
    x0_norm = float(np.linalg.norm(trace.X[0]))
    xT_norm = float(np.linalg.norm(trace.X[-1]))
    assert xT_norm / max(x0_norm, 1e-8) <= 2.5, f"X grew: {xT_norm}/{x0_norm}"


def test_fe_reproducibility():
    g = generate("scale_free", N=50, seed=1)
    t1 = rollout(g, T=20, mode="fixed_edge", seed=42)
    t2 = rollout(g, T=20, mode="fixed_edge", seed=42)
    assert np.allclose(t1.X, t2.X)


def test_fe_action_sensitivity():
    g = generate("scale_free", N=50, seed=1)
    t_zero = rollout(g, T=20, mode="fixed_edge", action_seq_mode="zero",
                     sigma_noise=0.0, seed=0)
    t_bc = rollout(g, T=20, mode="fixed_edge", action_seq_mode="random_walk",
                   sigma_noise=0.0, seed=0)
    diff = np.linalg.norm(t_bc.X[10] - t_zero.X[10])
    base = np.linalg.norm(t_zero.X[10])
    ratio = diff / max(base, 1e-8)
    assert ratio > 0.05, f"action 不敏感: ratio={ratio}"


def test_de_runs():
    g = generate("scale_free", N=20, seed=1)
    trace = rollout(g, T=10, mode="dynamic_edge", seed=0)
    assert trace.A is not None
    assert trace.A.shape == (11, 20, 20)
    # 邻接对称 (无向)
    for t in range(11):
        assert np.allclose(trace.A[t], trace.A[t].T)


def test_de_connectivity_check():
    """DE 模式邻接不可全 0 (即至少有边)."""
    g = generate("scale_free", N=20, seed=1)
    trace = rollout(g, T=10, mode="dynamic_edge", seed=0)
    for t in range(11):
        assert trace.A[t].sum() > 0, f"empty A at t={t}"


def test_agent_calling_tree_generates():
    s = generate_calling_tree(N_target=50, seed=0)
    assert s.features_init.shape[1] == 8
    assert s.oracle_answer_node is not None
    # planner 至少 2 出边
    from src.simulators.agent_calling_tree import EDGE_TYPES
    planner_ids = s.critical_roles["planner"]
    for pid in planner_ids:
        calls_out = sum(1 for e in s.edges
                        if e["src"] == pid and e["type"] == EDGE_TYPES["calls"])
        assert calls_out >= 2


def test_agent_simulator_runs():
    s = generate_calling_tree(N_target=30, seed=0)
    trace = simulate_calling_tree(s, T=10, seed=0)
    assert trace.X.shape == (11, s.features_init.shape[0], 8)
    assert 0 <= trace.final_sr <= 1 or np.isnan(trace.final_sr)


# ---------------------------------------------------------------------------
# Spec §3 plan B 4 项 acceptance criteria
# ---------------------------------------------------------------------------

def test_agent_sp_in_unit_interval():
    """∀ t, success_prob ∈ [0, 1] (无负值, 无 > 1)."""
    for seed in range(5):
        s = generate_calling_tree(N_target=50, seed=seed)
        trace = simulate_calling_tree(s, T=32, seed=seed, action_policy="random")
        sp = trace.X[:, :, 0]
        assert sp.min() >= 0.0, f"seed={seed}: sp min = {sp.min()}"
        assert sp.max() <= 1.0, f"seed={seed}: sp max = {sp.max()}"


def test_agent_error_flag_binary():
    """∀ t, error_flag ∈ {0, 1}."""
    for seed in range(5):
        s = generate_calling_tree(N_target=50, seed=seed)
        trace = simulate_calling_tree(s, T=32, seed=seed, action_policy="random")
        ef = trace.X[:, :, 5]
        # 严格 {0, 1}
        unique = np.unique(ef)
        bad = unique[(unique != 0.0) & (unique != 1.0)]
        assert len(bad) == 0, f"seed={seed}: error_flag 非 {{0,1}}: {bad}"


def test_agent_injection_planner_fpd_ge_2():
    """注入 planner 节点 sp=0, error_flag=1 → 至少 2 跳之外有 error 传播.

    跨多 seed 平均 (因 Bernoulli 有随机性, 单 seed 偶尔达不到).
    """
    fpds = []
    for seed in range(20):
        s = generate_calling_tree(N_target=50, seed=seed)
        planner_ids = s.critical_roles.get("planner", [])
        if not planner_ids:
            continue
        pid = planner_ids[0]
        trace = simulate_calling_tree(s, T=32, seed=seed, action_policy="random",
                                       injection_node_id=pid)
        # BFS from pid on directed graph, 找最远 error_flag=1 节点
        N = trace.X.shape[1]
        A_dir = np.zeros((N, N), dtype=np.float32)
        for src, dst in zip(s.edge_index[0].tolist(), s.edge_index[1].tolist()):
            A_dir[src, dst] = 1.0
        err_final = trace.X[-1, :, 5]
        from src.metrics.core import failure_propagation_depth
        fpd = failure_propagation_depth(A_dir, pid, err_final)
        fpds.append(fpd)
    median_fpd = float(np.median(fpds))
    assert median_fpd >= 2.0, f"injected-planner FPD median = {median_fpd}, expected >= 2"


def test_agent_random_sr_in_window():
    """T=32, random policy, 多 instance 平均 SR mean ∈ (0.2, 0.85).

    跨 30 instance × 1 seed each.
    """
    srs = []
    for seed in range(30):
        s = generate_calling_tree(N_target=50, seed=seed)
        trace = simulate_calling_tree(s, T=32, seed=seed, action_policy="random")
        srs.append(trace.final_sr)
    srs = [x for x in srs if not np.isnan(x)]
    sr_mean = float(np.mean(srs))
    assert 0.2 < sr_mean < 0.85, f"SR mean = {sr_mean}, expected in (0.2, 0.85)"


def test_skill_graph_generates():
    s = generate_skill_graph(N_target=50, seed=0)
    from src.simulators.platform_skill_graph import SKILL_NODE_TYPES
    type_counts = s.meta["type_counts"]
    assert type_counts["skill"] >= 5
    assert type_counts["task"] >= 1
    assert type_counts["patch"] >= 1


def test_skill_simulator_runs():
    s = generate_skill_graph(N_target=30, seed=0)
    trace = simulate_skill_graph(s, T=10, seed=0)
    assert trace.X.shape == (11, s.features_init.shape[0], 8)
