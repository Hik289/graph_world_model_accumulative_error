"""Agent calling-tree simulator.

实现 ``data/specs/agent_calling_tree.md``: 9 类节点 × 6 类边 × 8 维特征 + 动作集 +
Bernoulli failure + repair + (MCTS 风格) oracle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# 节点 / 边类型 常量
# ---------------------------------------------------------------------------

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

D_FEAT = 8  # success_prob, quality, cost, latency, confidence, error_flag, budget, context_usage
D_ACTION = 4

# action set
ACTIONS = {
    "select_agent": 0, "select_tool": 1, "add_validator": 2,
    "repair_failed_node": 3, "reroute_dependency": 4, "stop_execution": 5,
}
N_ACTIONS = len(ACTIONS)


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class HeteroGraphSample:
    """Heterogeneous graph (typed nodes/edges) sample."""

    nodes: List[Dict[str, Any]]                  # [{id, type, features_init}]
    edges: List[Dict[str, Any]]                  # [{src, dst, type}]
    node_types: np.ndarray                       # (N,) int8
    edge_index: np.ndarray                       # (2, E) int64
    edge_type: np.ndarray                        # (E,) int8
    features_init: np.ndarray                    # (N, D) float32
    critical_roles: Dict[str, List[int]] = field(default_factory=dict)
    oracle_path: List[int] = field(default_factory=list)
    oracle_answer_node: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[Any] = None                  # 之后由 simulate 填充


# ---------------------------------------------------------------------------
# 生成 calling tree
# ---------------------------------------------------------------------------

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
    """生成一棵 typed calling tree (DAG).

    实现 spec §2 流程; 一旦节点数 >= N_target 即截断.
    """
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

    # 1. user_query
    user_id = add_node("user_query")
    # 2. planner
    planner_id = add_node("planner")
    add_edge(user_id, planner_id, "calls")
    planners = [planner_id]

    # 3. 共享 memory 节点
    mem_id = add_node("memory")

    # 4. BFS 展开
    #    存 (parent_id, depth) frontier; 仅 planner / agent / tool / retriever 会进一步展开
    frontier: List[Tuple[int, int]] = [(planner_id, 1)]
    sink_artifacts: List[int] = []  # 用于最后挑 final_answer
    while frontier and n_total() < N_target:
        parent_id, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        parent_type = nodes[parent_id]["type"]

        # planner 必定分支; 其他按 p_branch 决定
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
            # 子节点类型权重: 越深越偏 agent/tool 直接执行
            child_kind = rng.choice(
                ["agent", "tool", "retriever"], p=[0.45, 0.40, 0.15]
            )
            child_id = add_node(child_kind)
            add_edge(parent_id, child_id, "calls")
            # retriever → memory
            if child_kind == "retriever":
                add_edge(child_id, mem_id, "retrieves")
            # 产出 artifact
            if n_total() < N_target:
                art_id = add_node("artifact", is_terminal=False)
                add_edge(child_id, art_id, "produces")
                sink_artifacts.append(art_id)
            # validator
            if rng.random() < p_validator and n_total() < N_target:
                val_id = add_node("validator")
                add_edge(val_id, child_id, "validates")
                # repairer
                if rng.random() < p_repair and n_total() < N_target:
                    rep_id = add_node("repairer")
                    add_edge(rep_id, child_id, "repairs")
            # 子节点继续展开 (agent / tool 类才推进)
            if child_kind in ("agent", "tool") and depth + 1 <= max_depth:
                frontier.append((child_id, depth + 1))

    # 5. 终态 final_answer (artifact + is_terminal=True)
    if not sink_artifacts:
        final_id = add_node("artifact", is_terminal=True)
        add_edge(planner_id, final_id, "produces")
        oracle_answer_node = final_id
    else:
        final_id = add_node("artifact", is_terminal=True)
        # depends_on 所有现有 artifact (汇聚)
        for aid in sink_artifacts:
            add_edge(aid, final_id, "depends_on")
        oracle_answer_node = final_id

    # ---- 转 numpy ----
    N = len(nodes)
    node_types = np.array([n["type"] for n in nodes], dtype=np.int8)
    if not edges:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int8)
    else:
        edge_index = np.array([[e["src"] for e in edges], [e["dst"] for e in edges]], dtype=np.int64)
        edge_type = np.array([e["type"] for e in edges], dtype=np.int8)

    # ---- 初始特征 ----
    features_init = _init_features(node_types, rng).astype(np.float32)

    # ---- critical roles ----
    critical_roles = _annotate_agent_roles(nodes, edges, edge_index, N)

    # ---- oracle path: BFS from planner 到 final_answer 的最短路 (作为 success path) ----
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
    """初始 8 维节点特征.

    dims: success_prob, quality, cost, latency, confidence, error_flag, budget, context_usage.
    简单按 type 给出 prior, 然后加 N(0, 0.05) 抖动. 这是 P1 stub, EDA 后再调.
    """
    N = node_types.shape[0]
    # 默认 ~ N(0, 0.5) 量级 (已"标准化"后 abstract)
    feats = rng.standard_normal((N, D_FEAT)).astype(np.float32) * 0.5
    # 把 success_prob (dim 0) 设为 logit-style, 大部分 type 偏正
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
    # error_flag (dim 5) 初始 0
    feats[:, 5] = 0.0
    # budget (dim 6) 初始 1.0
    feats[:, 6] = 1.0
    return feats


def _annotate_agent_roles(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
    edge_index: np.ndarray, N: int,
) -> Dict[str, List[int]]:
    """spec §2.2 critical_roles."""
    planner_ids = [n["id"] for n in nodes if n["type"] == NODE_TYPES["planner"]]
    validator_ids = [n["id"] for n in nodes if n["type"] == NODE_TYPES["validator"]]
    # action = planner 的 calls 出边目标
    action_ids: List[int] = []
    for e in edges:
        if e["type"] == EDGE_TYPES["calls"] and e["src"] in planner_ids:
            action_ids.append(e["dst"])
    action_ids = sorted(set(action_ids))

    # bridge: edge_betweenness top-5%
    bridge_ids: List[int] = []
    try:
        G_undir = nx.Graph()
        G_undir.add_nodes_from(range(N))
        for s, d in edge_index.T.tolist():
            G_undir.add_edge(s, d)
        eb = nx.edge_betweenness_centrality(G_undir)
        # 把 edge 度量上汇到端点
        node_score = np.zeros(N)
        for (u, v), s in eb.items():
            node_score[u] += s
            node_score[v] += s
        n_top = max(1, math.ceil(0.05 * N))
        bridge_ids = list(np.argsort(node_score)[-n_top:][::-1].astype(int))
    except Exception:
        pass

    # hub: degree top-5%
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


# ---------------------------------------------------------------------------
# 动力学 rollout
# ---------------------------------------------------------------------------

@dataclass
class HeteroTrace:
    X: np.ndarray                     # (T+1, N, D_FEAT)
    edge_index: np.ndarray            # (2, E)
    edge_type: np.ndarray             # (E,)
    actions: np.ndarray               # (T, D_ACTION)
    action_ids: np.ndarray            # (T,) int8 -- 离散 action id
    success_path: List[int]           # oracle path
    final_sr: float                   # final success_rate at sink
    config: Dict[str, Any] = field(default_factory=dict)


# spec §3 plan B 分离演化:
# - 6 维 continuous (dims 1=quality, 2=cost, 3=latency, 4=confidence, 6=budget, 7=context_usage)
#   走 tanh dynamics ∈ [-1, 1]
# - 2 维离散 (dim 0=success_prob ∈ [0,1], dim 5=error_flag ∈ {0,1}) 走 Bernoulli
# - 耦合: (a) error_flag=1 → quality/confidence 衰减 30% (failure shock);
#          (b) base_rate 依赖 validator-in + parent-error 状态
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
    action_policy: str = "random",      # "random" | "oracle" | "noop"
    sigma_noise: float = 0.05,
    decay_factor: float = 0.3,           # spec §3.1 组 B step 2
    repair_success_prob: float = 0.6,
    failure_shock_factor: float = 0.7,   # = 1 - 0.3 (quality/confidence 衰减 30%)
    sp_noise_std: float = 0.02,
    injection_node_id: Optional[int] = None,    # Exp 15: 强制该节点 sp=0, ef=1
) -> HeteroTrace:
    """Agent calling-tree dynamics (spec §3 plan B 分离演化).

    Parameters
    ----------
    sample : HeteroGraphSample (generate_calling_tree 产出).
    T : rollout 步数.
    seed : 派生 W_self / W_edge / U / 噪声 / Bernoulli 的主 seed.
    action_policy : "random" | "oracle" | "noop".
    sigma_noise : continuous 演化高斯噪声 σ (默认 0.05).
    decay_factor : success_prob → base_rate 的 decay 比例 (默认 0.3).
    repair_success_prob : repairer-in 节点 repair 成功概率 (默认 0.6).
    failure_shock_factor : error_flag=1 时 quality/confidence 的衰减乘数
        (默认 0.7 = 衰减 30%).
    sp_noise_std : success_prob 高斯噪声 std (默认 0.02).
    injection_node_id : 若给定, 每步强制该节点 sp=0, error_flag=1 (Exp 15).
    """
    rng = np.random.default_rng(seed)
    N = sample.features_init.shape[0]
    D = D_FEAT

    # ---------- ground-truth 矩阵 (仅作用 continuous dims) ----------
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

    # ---------- 反向邻接 (parents) 按 edge type 索引 ----------
    parents_by_type: Dict[int, Dict[int, List[int]]] = {k: {} for k in range(n_edge_types)}
    # 所有 parents (跨 edge type 合并, 用于 Bernoulli 子系统的 "parent.error" 判断)
    all_parents: Dict[int, List[int]] = {}
    repairers_of: Dict[int, List[int]] = {}
    # 用于 failure 显式传播 (Exp 15 FPD ≥ 2 验收要求)
    children_by_type: Dict[int, Dict[int, List[int]]] = {k: {} for k in range(n_edge_types)}
    for src, dst, et in zip(sample.edge_index[0].tolist(),
                            sample.edge_index[1].tolist(),
                            sample.edge_type.tolist()):
        parents_by_type[et].setdefault(dst, []).append(src)
        children_by_type[et].setdefault(src, []).append(dst)
        all_parents.setdefault(dst, []).append(src)
        if et == EDGE_TYPES["repairs"]:
            repairers_of.setdefault(dst, []).append(src)
    # validator-in nodes
    has_validator: np.ndarray = np.zeros(N, dtype=bool)
    for dst in parents_by_type[EDGE_TYPES["validates"]]:
        if 0 <= dst < N:
            has_validator[dst] = True

    # ---------- 状态 ----------
    X_traj = np.zeros((T + 1, N, D), dtype=np.float32)
    X_traj[0] = sample.features_init.copy()
    # 强制初始 sp/ef 在合理 range
    X_traj[0, :, 0] = np.clip(X_traj[0, :, 0], 0.0, 1.0)
    X_traj[0, :, 5] = (X_traj[0, :, 5] > 0.5).astype(np.float32)
    # injection 起步
    if injection_node_id is not None and 0 <= injection_node_id < N:
        X_traj[0, injection_node_id, 0] = 0.0
        X_traj[0, injection_node_id, 5] = 1.0

    actions_oh = np.zeros((T, D_ACTION), dtype=np.float32)
    action_ids = np.zeros((T,), dtype=np.int8)
    action_set = set(sample.critical_roles.get("action", []))

    for t in range(T):
        Xt = X_traj[t]
        err_prev = Xt[:, 5]              # error_flag(v, t)
        sp_prev = Xt[:, 0]               # success_prob(v, t)

        # ---------- 选 action ----------
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

        # ---------- (a) failure shock: 应用到 Xt 的 continuous dims 之 quality/confidence ----------
        # 给定 error_flag(v, t) = 1, 把 quality 与 confidence 各乘 failure_shock_factor (= 0.7)
        Xt_cont_shocked = Xt[:, CONTINUOUS_DIMS].copy()
        shock_mask = (err_prev > 0.5)
        if shock_mask.any():
            # CONTINUOUS_DIMS 中 dim_quality = 1 对应 idx 0; dim_confidence = 4 对应 idx 3
            q_idx = CONTINUOUS_DIMS.index(DIM_QUALITY)
            c_idx = CONTINUOUS_DIMS.index(DIM_CONFIDENCE)
            Xt_cont_shocked[shock_mask, q_idx] *= failure_shock_factor
            Xt_cont_shocked[shock_mask, c_idx] *= failure_shock_factor

        # ---------- (b) continuous dims tanh dynamics ----------
        # message_v = Σ_τ W_τ · mean_{u→v with type τ}(x_u[continuous_dims])
        msg = np.zeros((N, D_CONT), dtype=np.float32)
        for et, parents in parents_by_type.items():
            if not parents:
                continue
            W_k = W_edge[et]
            for dst, srcs in parents.items():
                msg[dst] += (Xt_cont_shocked[srcs].mean(axis=0) @ W_k)

        # action 注入: 仅作用 continuous dims, action_set 节点
        act_term_cont = np.zeros((N, D_CONT), dtype=np.float32)
        if action_set:
            Ua = U @ a_vec                                                       # (D_CONT,)
            for v in action_set:
                if 0 <= v < N:
                    act_term_cont[v] = Ua

        z = Xt_cont_shocked @ W_self.T + msg + act_term_cont
        cont_next = np.tanh(z).astype(np.float32) + \
            (rng.standard_normal((N, D_CONT)).astype(np.float32) * sigma_noise)
        # tanh 输出已 ∈ [-1, 1]; 加噪声后稍出界, 不再 clip (latent state 允许小偏)

        # ---------- (c) discrete dims Bernoulli evolution ----------
        # base_rate(v, t): 0.9 / 0.7 / 0.7 / 0.4 按 (validator-in, parent-error) 4 种组合
        parent_has_err = np.zeros(N, dtype=bool)
        for v, parents in all_parents.items():
            if any(err_prev[p] > 0.5 for p in parents):
                parent_has_err[v] = True
        base_rate = np.full(N, 0.7, dtype=np.float32)
        base_rate[has_validator & ~parent_has_err] = 0.9
        base_rate[has_validator & parent_has_err] = 0.7
        base_rate[~has_validator & ~parent_has_err] = 0.7
        base_rate[~has_validator & parent_has_err] = 0.4

        # success_prob update with small Gaussian noise + clip
        sp_noise = rng.standard_normal(N).astype(np.float32) * sp_noise_std
        sp_next = (1 - decay_factor) * sp_prev + decay_factor * base_rate + sp_noise
        sp_next = np.clip(sp_next, 0.0, 1.0).astype(np.float32)

        # Bernoulli error_flag draw
        ef_draw = (rng.random(N) < (1.0 - sp_next)).astype(np.float32)

        # repair: 若 error_flag=1 且有 repairer-in → 60% 概率 reset, sp +0.2
        for v in range(N):
            if ef_draw[v] > 0.5 and v in repairers_of:
                if rng.random() < repair_success_prob:
                    ef_draw[v] = 0.0
                    sp_next[v] = min(1.0, sp_next[v] + 0.2)

        # ---------- injection (Exp 15): 强制 sp=0, ef=1 ----------
        if injection_node_id is not None and 0 <= injection_node_id < N:
            sp_next[injection_node_id] = 0.0
            ef_draw[injection_node_id] = 1.0

        # ---------- 拼回 (T+1, N, D) ----------
        X_next = np.zeros((N, D), dtype=np.float32)
        X_next[:, CONTINUOUS_DIMS] = cont_next
        X_next[:, 0] = sp_next
        X_next[:, 5] = ef_draw
        X_traj[t + 1] = X_next

    # final_sr = success_prob(sink, T)
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
    """MCTS-style oracle: N_mc random rollouts, 取最大 final_sr 作 J*."""
    rng = np.random.default_rng(seed)
    best = -1.0
    for i in range(n_mc):
        trace = simulate_calling_tree(sample, T=T, seed=int(rng.integers(0, 2 ** 31 - 1)),
                                      action_policy="random")
        best = max(best, trace.final_sr)
    return float(best)
