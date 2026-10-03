from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx



NODE_TYPES = {
    "user_query": 0, "planner": 1, "retriever": 2, "tool": 3, "agent": 4,
    "validator": 5, "repairer": 6, "memory": 7, "artifact": 8,
}
NODE_TYPE_NAMES = {v: k for k, v in NODE_TYPES.items()}

EDGE_TYPES = {
    "calls": 0, "depends_on": 1, "validates": 2, "repairs": 3,
    "retrieves": 4, "produces": 5,
}
EDGE_TYPE_NAMES = {v: k for k, v in EDGE_TYPES.items()}

D_FEAT = 8
D_ACTION = 4

ACTIONS = {
    "select_agent": 0, "select_tool": 1, "add_validator": 2,
    "repair_failed_node": 3, "reroute_dependency": 4, "stop_execution": 5,
}
N_ACTIONS = len(ACTIONS)



@dataclass
class HeteroGraphSample:

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    node_types: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    features_init: np.ndarray
    critical_roles: Dict[str, List[int]] = field(default_factory=dict)
    oracle_path: List[int] = field(default_factory=list)
    oracle_answer_node: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[Any] = None



def generate_calling_tree(
    N_target: int = 50,
    seed: int = 0,
    *,
    max_depth: int = 4,
    p_branch: float = 0.5,
    p_validator: float = 0.7,
    p_repair: float = 0.4,
    branching_factor_planner: Tuple[int, int] = (2, 4),
) -> HeteroGraphSample:
    rng = np.random.default_rng(seed)
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    def add_node(type_name: str, **attrs: Any) -> int:
        nid = len(nodes)
        nodes.append({"id": nid, "type": NODE_TYPES[type_name], "type_name": type_name, **attrs})
        return nid

    def add_edge(src: int, dst: int, etype: str) -> None:
        edges.append({"src": src, "dst": dst, "type": EDGE_TYPES[etype]})

    def n_total() -> int:
        return len(nodes)

    user_id = add_node("user_query")
    planner_id = add_node("planner")
    add_edge(user_id, planner_id, "calls")
    planners = [planner_id]

    mem_id = add_node("memory")

    frontier: List[Tuple[int, int]] = [(planner_id, 1)]
    sink_artifacts: List[int] = []
    while frontier and n_total() < N_target:
        parent_id, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        parent_type = nodes[parent_id]["type"]

        if parent_type == NODE_TYPES["planner"]:
            bf = int(rng.integers(branching_factor_planner[0], branching_factor_planner[1] + 1))
        else:
            if rng.random() > p_branch:
                bf = 0
            else:
                bf = int(rng.integers(1, 3))

        for _ in range(bf):
            if n_total() >= N_target:
                break
            child_kind = rng.choice(
                ["agent", "tool", "retriever"], p=[0.45, 0.40, 0.15]
            )
            child_id = add_node(child_kind)
            add_edge(parent_id, child_id, "calls")
            if child_kind == "retriever":
                add_edge(child_id, mem_id, "retrieves")
            if n_total() < N_target:
                art_id = add_node("artifact", is_terminal=False)
                add_edge(child_id, art_id, "produces")
                sink_artifacts.append(art_id)
            if rng.random() < p_validator and n_total() < N_target:
                val_id = add_node("validator")
                add_edge(val_id, child_id, "validates")
                if rng.random() < p_repair and n_total() < N_target:
                    rep_id = add_node("repairer")
                    add_edge(rep_id, child_id, "repairs")
            if child_kind in ("agent", "tool") and depth + 1 <= max_depth:
                frontier.append((child_id, depth + 1))

    if not sink_artifacts:
        final_id = add_node("artifact", is_terminal=True)
        add_edge(planner_id, final_id, "produces")
        oracle_answer_node = final_id
    else:
        final_id = add_node("artifact", is_terminal=True)
        for aid in sink_artifacts:
            add_edge(aid, final_id, "depends_on")
        oracle_answer_node = final_id

    N = len(nodes)
    node_types = np.array([n["type"] for n in nodes], dtype=np.int8)
    if not edges:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int8)
    else:
        edge_index = np.array([[e["src"] for e in edges], [e["dst"] for e in edges]], dtype=np.int64)
        edge_type = np.array([e["type"] for e in edges], dtype=np.int8)

    features_init = _init_features(node_types, rng).astype(np.float32)

    critical_roles = _annotate_agent_roles(nodes, edges, edge_index, N)

    oracle_path: List[int] = []
    try:
        G_simple = nx.DiGraph()
        G_simple.add_nodes_from(range(N))
        for s, d in edge_index.T.tolist():
            G_simple.add_edge(s, d)
        if nx.has_path(G_simple, planner_id, oracle_answer_node):
            oracle_path = nx.shortest_path(G_simple, planner_id, oracle_answer_node)
    except Exception:
        oracle_path = []

    sample = HeteroGraphSample(
        nodes=nodes,
        edges=edges,
        node_types=node_types,
        edge_index=edge_index,
        edge_type=edge_type,
        features_init=features_init,
        critical_roles=critical_roles,
        oracle_path=[int(x) for x in oracle_path],
        oracle_answer_node=int(oracle_answer_node),
        meta={
            "N": N,
            "N_target": N_target,
            "seed": seed,
            "user_query_id": user_id,
            "planner_id": planner_id,
            "memory_id": mem_id,
            "max_depth": max_depth,
        },
    )
    return sample


def _init_features(node_types: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = node_types.shape[0]
    feats = rng.standard_normal((N, D_FEAT)).astype(np.float32) * 0.5
    base_sp = {
        NODE_TYPES["user_query"]: 1.0,
        NODE_TYPES["planner"]: 0.8,
        NODE_TYPES["retriever"]: 0.6,
        NODE_TYPES["tool"]: 0.5,
        NODE_TYPES["agent"]: 0.6,
        NODE_TYPES["validator"]: 0.7,
        NODE_TYPES["repairer"]: 0.6,
        NODE_TYPES["memory"]: 1.0,
        NODE_TYPES["artifact"]: 0.5,
    }
    for nid in range(N):
        feats[nid, 0] = base_sp.get(int(node_types[nid]), 0.5) + 0.1 * rng.standard_normal()
    feats[:, 5] = 0.0
    feats[:, 6] = 1.0
    return feats


def _annotate_agent_roles(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
    edge_index: np.ndarray, N: int,
) -> Dict[str, List[int]]:
    planner_ids = [n["id"] for n in nodes if n["type"] == NODE_TYPES["planner"]]
    validator_ids = [n["id"] for n in nodes if n["type"] == NODE_TYPES["validator"]]
    action_ids: List[int] = []
    for e in edges:
        if e["type"] == EDGE_TYPES["calls"] and e["src"] in planner_ids:
            action_ids.append(e["dst"])
    action_ids = sorted(set(action_ids))

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

    degrees = np.zeros(N, dtype=int)
    for e in edges:
        degrees[e["src"]] += 1
        degrees[e["dst"]] += 1
    n_top = max(1, math.ceil(0.05 * N))
    hub_ids = list(np.argsort(degrees)[-n_top:][::-1].astype(int))

    return {
        "planner": [int(x) for x in planner_ids],
        "validator": [int(x) for x in validator_ids],
        "action": [int(x) for x in action_ids],
        "bridge": [int(x) for x in bridge_ids],
        "hub": [int(x) for x in hub_ids],
    }



@dataclass
class HeteroTrace:
    X: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    actions: np.ndarray
    action_ids: np.ndarray
    success_path: List[int]
    final_sr: float
    config: Dict[str, Any] = field(default_factory=dict)


CONTINUOUS_DIMS: List[int] = [1, 2, 3, 4, 6, 7]
DISCRETE_DIMS: List[int] = [0, 5]
D_CONT = len(CONTINUOUS_DIMS)
DIM_QUALITY = 1
DIM_CONFIDENCE = 4


def simulate_calling_tree(
    sample: HeteroGraphSample,
    T: int = 32,
    *,
    seed: int = 0,
    action_policy: str = "random",
    sigma_noise: float = 0.05,
    decay_factor: float = 0.3,
    repair_success_prob: float = 0.6,
    failure_shock_factor: float = 0.7,
    sp_noise_std: float = 0.02,
    injection_node_id: Optional[int] = None,
) -> HeteroTrace:
    rng = np.random.default_rng(seed)
    N = sample.features_init.shape[0]
    D = D_FEAT

    rng_W = np.random.default_rng(seed + 101)
    W_self = rng_W.standard_normal((D_CONT, D_CONT)).astype(np.float32) / np.sqrt(D_CONT)
    s = np.linalg.svd(W_self, compute_uv=False)[0]
    W_self = (W_self / max(s, 1e-6) * 0.6).astype(np.float32)

    n_edge_types = len(EDGE_TYPES)
    W_edge = np.zeros((n_edge_types, D_CONT, D_CONT), dtype=np.float32)
    for k in range(n_edge_types):
        M = rng_W.standard_normal((D_CONT, D_CONT)).astype(np.float32) / np.sqrt(D_CONT)
        s_M = np.linalg.svd(M, compute_uv=False)[0]
        W_edge[k] = (M / max(s_M, 1e-6) * 0.3).astype(np.float32)

    U = rng_W.standard_normal((D_CONT, D_ACTION)).astype(np.float32) / np.sqrt(D_ACTION)
    s_U = np.linalg.svd(U, compute_uv=False)[0]
    U = (U / max(s_U, 1e-6) * 0.5).astype(np.float32)

    parents_by_type: Dict[int, Dict[int, List[int]]] = {k: {} for k in range(n_edge_types)}
    all_parents: Dict[int, List[int]] = {}
    repairers_of: Dict[int, List[int]] = {}
    children_by_type: Dict[int, Dict[int, List[int]]] = {k: {} for k in range(n_edge_types)}
    for src, dst, et in zip(sample.edge_index[0].tolist(),
                            sample.edge_index[1].tolist(),
                            sample.edge_type.tolist()):
        parents_by_type[et].setdefault(dst, []).append(src)
        children_by_type[et].setdefault(src, []).append(dst)
        all_parents.setdefault(dst, []).append(src)
        if et == EDGE_TYPES["repairs"]:
            repairers_of.setdefault(dst, []).append(src)
    has_validator: np.ndarray = np.zeros(N, dtype=bool)
    for dst in parents_by_type[EDGE_TYPES["validates"]]:
        if 0 <= dst < N:
            has_validator[dst] = True

    X_traj = np.zeros((T + 1, N, D), dtype=np.float32)
    X_traj[0] = sample.features_init.copy()
    X_traj[0, :, 0] = np.clip(X_traj[0, :, 0], 0.0, 1.0)
    X_traj[0, :, 5] = (X_traj[0, :, 5] > 0.5).astype(np.float32)
    if injection_node_id is not None and 0 <= injection_node_id < N:
        X_traj[0, injection_node_id, 0] = 0.0
        X_traj[0, injection_node_id, 5] = 1.0

    actions_oh = np.zeros((T, D_ACTION), dtype=np.float32)
    action_ids = np.zeros((T,), dtype=np.int8)
    action_set = set(sample.critical_roles.get("action", []))

    for t in range(T):
        Xt = X_traj[t]
        err_prev = Xt[:, 5]
        sp_prev = Xt[:, 0]

        if action_policy == "oracle":
            err_nodes = np.where(err_prev > 0.5)[0]
            act_id = ACTIONS["repair_failed_node"] if len(err_nodes) > 0 else ACTIONS["select_agent"]
        elif action_policy == "noop":
            act_id = ACTIONS["select_agent"]
        else:
            act_id = int(rng.integers(0, N_ACTIONS))
        action_ids[t] = act_id
        a_vec = np.zeros(D_ACTION, dtype=np.float32)
        a_vec[act_id % D_ACTION] = 1.0
        actions_oh[t] = a_vec

        Xt_cont_shocked = Xt[:, CONTINUOUS_DIMS].copy()
        shock_mask = (err_prev > 0.5)
        if shock_mask.any():
            q_idx = CONTINUOUS_DIMS.index(DIM_QUALITY)
            c_idx = CONTINUOUS_DIMS.index(DIM_CONFIDENCE)
            Xt_cont_shocked[shock_mask, q_idx] *= failure_shock_factor
            Xt_cont_shocked[shock_mask, c_idx] *= failure_shock_factor

        msg = np.zeros((N, D_CONT), dtype=np.float32)
        for et, parents in parents_by_type.items():
            if not parents:
                continue
            W_k = W_edge[et]
            for dst, srcs in parents.items():
                msg[dst] += (Xt_cont_shocked[srcs].mean(axis=0) @ W_k)

        act_term_cont = np.zeros((N, D_CONT), dtype=np.float32)
        if action_set:
            Ua = U @ a_vec
            for v in action_set:
                if 0 <= v < N:
                    act_term_cont[v] = Ua

        z = Xt_cont_shocked @ W_self.T + msg + act_term_cont
        cont_next = np.tanh(z).astype(np.float32) + \
            (rng.standard_normal((N, D_CONT)).astype(np.float32) * sigma_noise)

        parent_has_err = np.zeros(N, dtype=bool)
        for v, parents in all_parents.items():
            if any(err_prev[p] > 0.5 for p in parents):
                parent_has_err[v] = True
        base_rate = np.full(N, 0.7, dtype=np.float32)
        base_rate[has_validator & ~parent_has_err] = 0.9
        base_rate[has_validator & parent_has_err] = 0.7
        base_rate[~has_validator & ~parent_has_err] = 0.7
        base_rate[~has_validator & parent_has_err] = 0.4

        sp_noise = rng.standard_normal(N).astype(np.float32) * sp_noise_std
        sp_next = (1 - decay_factor) * sp_prev + decay_factor * base_rate + sp_noise
        sp_next = np.clip(sp_next, 0.0, 1.0).astype(np.float32)

        ef_draw = (rng.random(N) < (1.0 - sp_next)).astype(np.float32)

        for v in range(N):
            if ef_draw[v] > 0.5 and v in repairers_of:
                if rng.random() < repair_success_prob:
                    ef_draw[v] = 0.0
                    sp_next[v] = min(1.0, sp_next[v] + 0.2)

        if injection_node_id is not None and 0 <= injection_node_id < N:
            sp_next[injection_node_id] = 0.0
            ef_draw[injection_node_id] = 1.0

        X_next = np.zeros((N, D), dtype=np.float32)
        X_next[:, CONTINUOUS_DIMS] = cont_next
        X_next[:, 0] = sp_next
        X_next[:, 5] = ef_draw
        X_traj[t + 1] = X_next

    if sample.oracle_answer_node is not None and 0 <= sample.oracle_answer_node < N:
        final_sr = float(X_traj[T, sample.oracle_answer_node, 0])
    else:
        final_sr = float("nan")

    return HeteroTrace(
        X=X_traj,
        edge_index=sample.edge_index,
        edge_type=sample.edge_type,
        actions=actions_oh,
        action_ids=action_ids,
        success_path=sample.oracle_path,
        final_sr=final_sr,
        config={"T": T, "policy": action_policy, "seed": seed,
                "decay_factor": decay_factor,
                "repair_success_prob": repair_success_prob,
                "failure_shock_factor": failure_shock_factor,
                "sigma_noise": sigma_noise,
                "sp_noise_std": sp_noise_std,
                "injection_node_id": injection_node_id,
                "scheme": "plan_B_separated"},
    )


def oracle_return(
    sample: HeteroGraphSample,
    T: int = 32,
    n_mc: int = 200,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    best = -1.0
    for i in range(n_mc):
        trace = simulate_calling_tree(sample, T=T, seed=int(rng.integers(0, 2 ** 31 - 1)),
                                      action_policy="random")
        best = max(best, trace.final_sr)
    return float(best)
