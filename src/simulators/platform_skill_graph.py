"""OpenClaw skill-graph simulator.

实现 ``data/specs/platform_skill_graph.md``: 8 类节点 × 8 类边 × 8 维特征 +
动作集 + failure propagation + repair + oracle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from .agent_calling_tree import HeteroGraphSample, HeteroTrace, D_FEAT, D_ACTION


SKILL_NODE_TYPES = {
    "skill": 0, "validator": 1, "adapter": 2, "artifact": 3,
    "failure_case": 4, "patch": 5, "task": 6, "tool": 7,
}
SKILL_NODE_TYPE_NAMES = {v: k for k, v in SKILL_NODE_TYPES.items()}

SKILL_EDGE_TYPES = {
    "requires": 0, "produces": 1, "validates": 2, "repairs": 3,
    "fails_on": 4, "compatible_with": 5, "replaces": 6, "calls": 7,
}
SKILL_EDGE_TYPE_NAMES = {v: k for k, v in SKILL_EDGE_TYPES.items()}

# 允许的 (src, dst) 类型 pair (按 spec §1.2)
_ALLOWED_PAIRS: Dict[str, List[Tuple[str, str]]] = {
    "requires": [("skill", "skill"), ("skill", "tool"), ("skill", "artifact")],
    "produces": [("skill", "artifact")],
    "validates": [("validator", "skill"), ("validator", "artifact")],
    "repairs": [("patch", "skill"), ("patch", "failure_case")],
    "fails_on": [("skill", "failure_case")],
    "compatible_with": [("skill", "skill"), ("skill", "adapter")],
    "replaces": [("skill", "skill")],
    "calls": [("skill", "task"), ("skill", "tool")],
}

SKILL_ACTIONS = {
    "add_skill": 0, "add_validator": 1, "add_adapter": 2, "repair_skill": 3,
    "remove_edge": 4, "add_compat_edge": 5, "reroute_task": 6, "no_op": 7,
}
N_SKILL_ACTIONS = len(SKILL_ACTIONS)


def generate_skill_graph(
    N_target: int = 100,
    seed: int = 0,
    *,
    m_BA: int = 2,
    edge_density: float = 0.05,
    type_probs: Optional[Dict[str, float]] = None,
) -> HeteroGraphSample:
    """生成 typed skill ecosystem graph.

    步骤:
      1. BA scale-free 拓扑 (m_BA, N_target).
      2. 节点类型采样 (按 type_probs).
      3. 按允许 pair + density 添加 typed edges.
      4. 标 critical_roles + oracle_repair_paths.
    """
    if type_probs is None:
        type_probs = {
            "skill": 0.50, "validator": 0.15, "adapter": 0.10, "artifact": 0.10,
            "failure_case": 0.05, "patch": 0.05, "task": 0.03, "tool": 0.02,
        }
    rng = np.random.default_rng(seed)

    # 1. BA 拓扑
    G_ba = nx.barabasi_albert_graph(N_target, m=m_BA, seed=seed)
    N = G_ba.number_of_nodes()

    # 2. 类型分配 (按 probs)
    type_names = list(type_probs.keys())
    type_probs_arr = np.array([type_probs[t] for t in type_names], dtype=np.float32)
    type_probs_arr = type_probs_arr / type_probs_arr.sum()
    sampled_types = rng.choice(type_names, size=N, p=type_probs_arr)

    # 保证至少 5 个 skill / 1 个 task
    skill_count = (sampled_types == "skill").sum()
    task_count = (sampled_types == "task").sum()
    if skill_count < 5:
        # 随机选 5 个非 skill 改为 skill
        non_skill = np.where(sampled_types != "skill")[0]
        n_swap = 5 - int(skill_count)
        swap = rng.choice(non_skill, size=min(n_swap, len(non_skill)), replace=False)
        sampled_types[swap] = "skill"
    if task_count < 1:
        non_task = np.where(sampled_types != "task")[0]
        sampled_types[rng.choice(non_task)] = "task"

    # 保证 ≥ N·0.05 patch
    n_patch_min = max(1, int(N_target * 0.05))
    if (sampled_types == "patch").sum() < n_patch_min:
        # 把一些 adapter 改为 patch
        candidates = np.where(sampled_types != "patch")[0]
        n_need = n_patch_min - int((sampled_types == "patch").sum())
        swap = rng.choice(candidates, size=min(n_need, len(candidates)), replace=False)
        sampled_types[swap] = "patch"

    node_types_arr = np.array([SKILL_NODE_TYPES[t] for t in sampled_types], dtype=np.int8)

    # 节点 dict
    nodes: List[Dict[str, Any]] = []
    for nid in range(N):
        nodes.append({"id": nid, "type": int(node_types_arr[nid]),
                      "type_name": str(sampled_types[nid])})

    # 3. 加 typed edges
    edges: List[Dict[str, Any]] = []
    used_edges = set()  # 避免重复 (src, dst, et)

    # Step 3a: BA 邻接 → 主拓扑骨架 (按 produces/requires/calls 分配)
    for u, v in G_ba.edges():
        ut, vt = sampled_types[u], sampled_types[v]
        # 找一对 allowed (et, src, dst) 给该 BA 边
        chosen = None
        for et_name in ("requires", "produces", "calls", "compatible_with"):
            for (a, b) in _ALLOWED_PAIRS[et_name]:
                if a == ut and b == vt:
                    chosen = (et_name, u, v)
                    break
                if a == vt and b == ut:
                    chosen = (et_name, v, u)
                    break
            if chosen:
                break
        if chosen is None:
            # 退化: 跳过 (BA 边在类型不匹配下不强制)
            continue
        et_name, s, d = chosen
        et = SKILL_EDGE_TYPES[et_name]
        if (s, d, et) in used_edges:
            continue
        edges.append({"src": int(s), "dst": int(d), "type": et})
        used_edges.add((s, d, et))

    # Step 3b: 按 edge_density 随机加 typed edges
    target_E = int(edge_density * N * N)
    attempts = 0
    while len(edges) < target_E and attempts < target_E * 10:
        attempts += 1
        et_name = rng.choice(list(_ALLOWED_PAIRS.keys()))
        allowed_pairs = _ALLOWED_PAIRS[et_name]
        a, b = allowed_pairs[rng.integers(0, len(allowed_pairs))]
        srcs = np.where(sampled_types == a)[0]
        dsts = np.where(sampled_types == b)[0]
        if len(srcs) == 0 or len(dsts) == 0:
            continue
        s = int(srcs[rng.integers(0, len(srcs))])
        d = int(dsts[rng.integers(0, len(dsts))])
        if s == d:
            continue
        et = SKILL_EDGE_TYPES[et_name]
        if (s, d, et) in used_edges:
            continue
        edges.append({"src": s, "dst": d, "type": et})
        used_edges.add((s, d, et))

    # 4. numpy arrays
    if edges:
        edge_index = np.array([[e["src"] for e in edges], [e["dst"] for e in edges]], dtype=np.int64)
        edge_type = np.array([e["type"] for e in edges], dtype=np.int8)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int8)

    # 初始特征 (8 维, skill-graph 版本)
    features_init = _init_skill_features(node_types_arr, rng).astype(np.float32)

    # 5. critical roles
    critical_roles = _annotate_skill_roles(nodes, edges, edge_index, node_types_arr, N)

    # 6. oracle_repair_paths: failure_case → patch → skill
    oracle_repair_paths: List[List[int]] = []
    fc_ids = [i for i, t in enumerate(node_types_arr) if t == SKILL_NODE_TYPES["failure_case"]]
    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(range(N))
    for e in edges:
        G_dir.add_edge(e["src"], e["dst"])
    for fc in fc_ids:
        # 找 patch nodes 能 reach 它 (repairs edge)
        patch_repair = [e["src"] for e in edges
                        if e["type"] == SKILL_EDGE_TYPES["repairs"] and e["dst"] == fc]
        for p in patch_repair:
            oracle_repair_paths.append([int(p), int(fc)])

    sample = HeteroGraphSample(
        nodes=nodes,
        edges=edges,
        node_types=node_types_arr,
        edge_index=edge_index,
        edge_type=edge_type,
        features_init=features_init,
        critical_roles=critical_roles,
        oracle_path=[],            # skill-graph 没单一 path
        oracle_answer_node=None,
        meta={
            "N": N,
            "N_target": N_target,
            "seed": seed,
            "m_BA": m_BA,
            "edge_density": edge_density,
            "type_probs": type_probs,
            "type_counts": {t: int((sampled_types == t).sum()) for t in type_names},
            "oracle_repair_paths": oracle_repair_paths,
            "graph_kind": "platform_skill_graph",
        },
    )
    return sample


def _init_skill_features(node_types_arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """skill-graph 8 维特征: success_rate, version_age, complexity, maturity,
    usage_freq, error_flag, repair_pending, compat_breadth.
    """
    N = node_types_arr.shape[0]
    feats = rng.standard_normal((N, D_FEAT)).astype(np.float32) * 0.3
    base_success = {
        SKILL_NODE_TYPES["skill"]: 0.7,
        SKILL_NODE_TYPES["validator"]: 0.8,
        SKILL_NODE_TYPES["adapter"]: 0.7,
        SKILL_NODE_TYPES["artifact"]: 0.8,
        SKILL_NODE_TYPES["failure_case"]: 0.0,
        SKILL_NODE_TYPES["patch"]: 0.9,
        SKILL_NODE_TYPES["task"]: 0.5,
        SKILL_NODE_TYPES["tool"]: 0.85,
    }
    for nid in range(N):
        feats[nid, 0] = base_success.get(int(node_types_arr[nid]), 0.5) + 0.05 * rng.standard_normal()
    feats[:, 5] = 0.0   # error_flag
    feats[:, 6] = 0.0   # repair_pending
    return feats


def _annotate_skill_roles(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
    edge_index: np.ndarray, node_types_arr: np.ndarray, N: int,
) -> Dict[str, List[int]]:
    skill_mask = (node_types_arr == SKILL_NODE_TYPES["skill"])
    skill_ids = np.where(skill_mask)[0]

    # hub_skill: degree top-5% in skill 子集
    degrees = np.zeros(N, dtype=int)
    for e in edges:
        degrees[e["src"]] += 1
        degrees[e["dst"]] += 1
    if len(skill_ids):
        skill_deg = degrees[skill_ids]
        n_top = max(1, math.ceil(0.05 * len(skill_ids)))
        order = np.argsort(skill_deg)
        hub_skill_ids = list(skill_ids[order[-n_top:][::-1]].astype(int))
    else:
        hub_skill_ids = []

    # failure_hub: 连接 failure_case 最多的 skill (前 5%)
    fc_mask = (node_types_arr == SKILL_NODE_TYPES["failure_case"])
    failure_link_cnt = np.zeros(N, dtype=int)
    for e in edges:
        if e["type"] == SKILL_EDGE_TYPES["fails_on"] and fc_mask[e["dst"]]:
            failure_link_cnt[e["src"]] += 1
    n_top = max(1, math.ceil(0.05 * max(len(skill_ids), 1)))
    fhub_order = np.argsort(failure_link_cnt)
    failure_hub_ids = list(fhub_order[-n_top:][::-1].astype(int))

    # patch_critical: patch 节点 (简化: 所有 patch)
    patch_critical = [int(i) for i in range(N)
                      if node_types_arr[i] == SKILL_NODE_TYPES["patch"]]

    # bridge: edge_betweenness top-5%
    bridge_ids: List[int] = []
    try:
        G_undir = nx.Graph()
        G_undir.add_nodes_from(range(N))
        for s, d in edge_index.T.tolist():
            G_undir.add_edge(s, d)
        eb = nx.edge_betweenness_centrality(G_undir)
        node_score = np.zeros(N)
        for (u, v), s in eb.items():
            node_score[u] += s
            node_score[v] += s
        n_top = max(1, math.ceil(0.05 * N))
        bridge_ids = list(np.argsort(node_score)[-n_top:][::-1].astype(int))
    except Exception:
        pass

    return {
        "hub_skill": [int(x) for x in hub_skill_ids],
        "failure_hub": [int(x) for x in failure_hub_ids],
        "patch_critical": [int(x) for x in patch_critical],
        "bridge": [int(x) for x in bridge_ids],
    }


# ---------------------------------------------------------------------------
# Simulator (与 agent_calling_tree.simulate 共享接口, 跑 typed dynamics)
# ---------------------------------------------------------------------------

def simulate_skill_graph(
    sample: HeteroGraphSample,
    T: int = 32,
    *,
    seed: int = 0,
    action_policy: str = "random",
    sigma_noise: float = 0.05,
    decay_factor: float = 0.05,
    repair_success_prob: float = 0.6,
    propagate_prob: float = 0.5,
) -> HeteroTrace:
    """skill-graph 动力学 rollout.

    与 agent_calling_tree.simulate_calling_tree 结构类似, 但用 8 类 edge / action.
    """
    rng = np.random.default_rng(seed)
    N = sample.features_init.shape[0]
    D = D_FEAT

    rng_W = np.random.default_rng(seed + 101)
    W_self = rng_W.standard_normal((D, D)).astype(np.float32) / np.sqrt(D)
    s = np.linalg.svd(W_self, compute_uv=False)[0]
    W_self = (W_self / max(s, 1e-6) * 0.6).astype(np.float32)

    n_etypes = len(SKILL_EDGE_TYPES)
    W_edge = np.zeros((n_etypes, D, D), dtype=np.float32)
    for k in range(n_etypes):
        M = rng_W.standard_normal((D, D)).astype(np.float32) / np.sqrt(D)
        s_M = np.linalg.svd(M, compute_uv=False)[0]
        W_edge[k] = (M / max(s_M, 1e-6) * 0.3).astype(np.float32)

    U = rng_W.standard_normal((D, D_ACTION)).astype(np.float32) / np.sqrt(D_ACTION)
    s_U = np.linalg.svd(U, compute_uv=False)[0]
    U = (U / max(s_U, 1e-6) * 0.5).astype(np.float32)

    # 反向邻接按 edge type
    parents_by_type: Dict[int, Dict[int, List[int]]] = {k: {} for k in range(n_etypes)}
    repairers_of: Dict[int, List[int]] = {}
    for src, dst, et in zip(sample.edge_index[0].tolist(), sample.edge_index[1].tolist(), sample.edge_type.tolist()):
        parents_by_type[et].setdefault(dst, []).append(src)
        if et == SKILL_EDGE_TYPES["repairs"]:
            repairers_of.setdefault(dst, []).append(src)

    X_traj = np.zeros((T + 1, N, D), dtype=np.float32)
    X_traj[0] = sample.features_init.copy()
    actions_oh = np.zeros((T, D_ACTION), dtype=np.float32)
    action_ids = np.zeros((T,), dtype=np.int8)

    propagate_etypes = {SKILL_EDGE_TYPES["produces"], SKILL_EDGE_TYPES["compatible_with"], SKILL_EDGE_TYPES["calls"]}

    for t in range(T):
        Xt = X_traj[t]

        if action_policy == "oracle":
            err = np.where(Xt[:, 5] > 0.5)[0]
            act_id = SKILL_ACTIONS["repair_skill"] if len(err) > 0 else SKILL_ACTIONS["no_op"]
        elif action_policy == "noop":
            act_id = SKILL_ACTIONS["no_op"]
        else:
            act_id = int(rng.integers(0, N_SKILL_ACTIONS))
        action_ids[t] = act_id
        a_vec = np.zeros(D_ACTION, dtype=np.float32)
        a_vec[act_id % D_ACTION] = 1.0
        actions_oh[t] = a_vec

        msg = np.zeros((N, D), dtype=np.float32)
        for et, parents in parents_by_type.items():
            if not parents:
                continue
            W_k = W_edge[et]
            for dst, srcs in parents.items():
                msg[dst] += (Xt[srcs].mean(axis=0) @ W_k)

        z = Xt @ W_self.T + msg
        X_next = np.tanh(z).astype(np.float32) + (rng.standard_normal((N, D)).astype(np.float32) * sigma_noise)

        # success-rate decay
        base = 0.5 * np.ones(N, dtype=np.float32)
        for dst in parents_by_type[SKILL_EDGE_TYPES["validates"]]:
            if 0 <= dst < N:
                base[dst] = min(1.0, base[dst] + 0.2)
        # fails_on 边的 skill src 上 base 降低
        for src, dst, et in zip(sample.edge_index[0].tolist(),
                                sample.edge_index[1].tolist(),
                                sample.edge_type.tolist()):
            if et == SKILL_EDGE_TYPES["fails_on"]:
                base[src] = max(0.0, base[src] - 0.2)
        # success_rate ∈ [0,1]; tanh 输出 ∈ [-1,1] → sigmoid 化
        sp_pred = 1.0 / (1.0 + np.exp(-X_next[:, 0]))
        sp_next = (sp_pred * (1 - decay_factor) + base * decay_factor).astype(np.float32)
        sp_next = np.clip(sp_next, 0.0, 1.0)

        # Bernoulli failure
        fail = rng.random(N) > sp_next
        err_flag = fail.astype(np.float32)

        # repair action effect (oracle / repair_skill)
        if SKILL_ACTIONS["repair_skill"] == act_id:
            for v in np.where(err_flag > 0.5)[0]:
                if rng.random() < repair_success_prob:
                    err_flag[v] = 0.0

        # 误差沿 propagate_etypes 传染
        e_src = sample.edge_index[0]
        e_dst = sample.edge_index[1]
        for ei in range(e_src.shape[0]):
            if int(sample.edge_type[ei]) not in propagate_etypes:
                continue
            s_id, d_id = int(e_src[ei]), int(e_dst[ei])
            if err_flag[s_id] > 0.5 and rng.random() < propagate_prob:
                err_flag[d_id] = 1.0

        # 自然 repair (有 repairer 父)
        for v in range(N):
            if err_flag[v] > 0.5 and v in repairers_of:
                if rng.random() < repair_success_prob:
                    err_flag[v] = 0.0

        X_next[:, 5] = err_flag
        X_next[:, 0] = np.where(err_flag > 0.5, sp_next * 0.5, sp_next).astype(np.float32)
        X_traj[t + 1] = X_next

    # 全图 skill 节点 success_rate 均值作 final_sr
    skill_mask = (sample.node_types == SKILL_NODE_TYPES["skill"])
    final_sr = float(X_traj[T, skill_mask, 0].mean()) if skill_mask.any() else float("nan")
    return HeteroTrace(
        X=X_traj,
        edge_index=sample.edge_index,
        edge_type=sample.edge_type,
        actions=actions_oh,
        action_ids=action_ids,
        success_path=[],
        final_sr=final_sr,
        config={"T": T, "policy": action_policy, "seed": seed,
                "kind": "platform_skill_graph"},
    )
