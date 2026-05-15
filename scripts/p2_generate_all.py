"""P2 数据全量落盘 (按 data/specs/dataset_layout.md).

生成:
import os as _os; PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
- synthetic_graphs/      7 拓扑 × 3 outer seed × {N=50 默认 + sweep} + ER × p
- synthetic_rollouts/    FE rollout T=50 (训练) + T=32 (评估), 各 seed
- injection_data/        Exp 3/4 注入轨迹: node-only / edge-only / +bridge / +hub
- agent_calling_tree/    750 instance (500 train + 100 val + 100 test + 50 ood N=100)
- platform_skill_graph/ 450 instance (300 + 60 + 60 + 30 ood N=200)

每子目录附 manifest.json + 每文件 sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import scipy
import networkx as nx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.graph_generators import generate, compute_all
from src.simulators import (
    rollout, generate_calling_tree, simulate_calling_tree,
    generate_skill_graph, simulate_skill_graph,
)

JST = timezone(timedelta(hours=9))
DATA_ROOT_DEFAULT = "PROJECT_ROOT/data"
CODE_VERSION = "1"  # 与 dataset_layout.md §2 manifest.version 对齐


def now_jst() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def _git_commit_hash() -> str:
    """读 git commit hash (P1 仓库未 git init, fallback timestamp)."""
    try:
        import subprocess
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return f"no-git-{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _save_torch(obj: Dict[str, Any], path: str) -> str:
    """torch.save + 返回该文件 sha256."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)
    with open(path, "rb") as f:
        sha = _sha256_bytes(f.read())
    return sha


def _env_meta() -> Dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "networkx_version": nx.__version__,
        "torch_version": torch.__version__,
        "generated_at_jst": now_jst(),
        "generator_commit_hash": _git_commit_hash(),
    }


# ---------------------------------------------------------------------------
# Section 1: synthetic_graphs/ — 7 拓扑 + sweep + ER
# ---------------------------------------------------------------------------

TOP_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "chain": {},
    "tree": {"variant": "balanced_binary"},
    "grid": {"shape": "auto"},
    "small_world": {"k": 4, "p": 0.1},
    "scale_free": {"m": 2},
    "star": {},
    "complete": {},
}

OUTER_SEEDS = [1, 2, 3]
N_DEFAULT = 50
# scaling sweep (Exp 21); star/complete @ N=500 跳过 (per spec §4.1)
N_SWEEP = [20, 50, 100, 200, 500]
# small_world k 扫 (Exp 22)
SW_K_SWEEP = [2, 4, 6, 8]
# scale_free m 扫 (sparsity)
SF_M_SWEEP = [1, 2, 3]
# ER density 扫 (Exp 22)
ER_P_SWEEP = [0.02, 0.05, 0.10, 0.20]
ER_N = 50


def _serialize_graph_sample(g, stats: Dict[str, float], extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GraphSample -> torch.save 友好 dict (per dataset_layout §4.1)."""
    payload = {
        "version": int(CODE_VERSION),
        "topology": g.topology,
        "N": int(g.N),
        "D": 8,                               # 默认 D=8
        "seed": int(g.seed),
        "params": {**g.params, **(extra_params or {})},
        "A_dense": g.A_dense,
        "A_sparse": g.A_sparse,
        "A_norm": g.A_norm,
        "stats": stats,
        "critical_roles": g.critical_roles,
        "is_directed": bool(g.is_directed),
    }
    return payload


def gen_synthetic_graphs(data_root: str) -> Dict[str, Any]:
    """生成 synthetic_graphs/ 全部 .pt + manifest.json."""
    out_dir = os.path.join(data_root, "synthetic_graphs")
    os.makedirs(out_dir, exist_ok=True)
    files: List[Dict[str, Any]] = []
    t0 = time.time()

    def _do(topology: str, N: int, seed: int, extra: Dict[str, Any], filename: str):
        params = {**TOP_DEFAULTS.get(topology, {}), **extra}
        g = generate(topology, N=N, seed=seed, compute_stats=False, **params)
        stats = compute_all(g)
        payload = _serialize_graph_sample(g, stats, extra_params=extra)
        path = os.path.join(out_dir, filename)
        sha = _save_torch(payload, path)
        size = os.path.getsize(path)
        files.append({
            "path": filename, "topology": topology, "N": int(g.N),
            "D": 8, "seed": int(seed), "params": payload["params"],
            "stats": stats, "sha256": sha, "size_bytes": int(size),
        })

    # (1) default suite: 7 拓扑 × 3 outer seed × N=50
    for top in TOP_DEFAULTS:
        for seed in OUTER_SEEDS:
            _do(top, N=N_DEFAULT, seed=seed, extra={},
                filename=f"{top}_N{N_DEFAULT}_seed{seed}.pt")

    # (2) scaling sweep N ∈ {20, 100, 200, 500} (N=50 已在 default)
    for top in TOP_DEFAULTS:
        for N in N_SWEEP:
            if N == N_DEFAULT:
                continue
            if N == 500 and top in ("star", "complete"):
                # spec §4.1: N=500 上 complete/star ρ 过大, 跳过
                continue
            for seed in OUTER_SEEDS:
                _do(top, N=N, seed=seed, extra={},
                    filename=f"{top}_N{N}_seed{seed}.pt")

    # (3) small_world k 扫 (Exp 22)
    for k in SW_K_SWEEP:
        if k == 4:  # already in default
            continue
        for seed in OUTER_SEEDS:
            _do("small_world", N=N_DEFAULT, seed=seed,
                extra={"k": k, "p": 0.1},
                filename=f"small_world_N{N_DEFAULT}_k{k}_p0.10_seed{seed}.pt")

    # (4) scale_free m 扫
    for m in SF_M_SWEEP:
        if m == 2:
            continue
        for seed in OUTER_SEEDS:
            _do("scale_free", N=N_DEFAULT, seed=seed,
                extra={"m": m},
                filename=f"scale_free_N{N_DEFAULT}_m{m}_seed{seed}.pt")

    # (5) ER (Erdős–Rényi) p 扫 — 用 networkx 直接生成, 用 chain 名字 + 特殊 param
    # ER 不在 7 主拓扑里, 单独 path 处理
    for p in ER_P_SWEEP:
        for seed in OUTER_SEEDS:
            # 把 ER 当作 "small_world" 的退化 (p=1) 不准确, 我们直接 manual 生成
            G_er = nx.erdos_renyi_graph(ER_N, p=p, seed=seed)
            # 确保连通
            if not nx.is_connected(G_er):
                # 补一棵 spanning tree
                cc = list(nx.connected_components(G_er))
                for i in range(len(cc) - 1):
                    a = next(iter(cc[i]))
                    b = next(iter(cc[i + 1]))
                    G_er.add_edge(a, b)
            A = nx.to_numpy_array(G_er, dtype=np.float32)
            np.fill_diagonal(A, 0.0)
            # 包装成 GraphSample-ish
            import scipy.sparse as sp
            from src.graph_generators.base import GraphSample, _normalize_adj, _annotate_critical_roles
            A_norm = _normalize_adj(A)
            A_sparse = sp.csr_matrix(A)
            g = GraphSample(
                A_dense=A, A_sparse=A_sparse, A_norm=A_norm,
                is_directed=False, N=ER_N, topology="er",
                params={"p": float(p)}, seed=seed,
            )
            g.critical_roles = _annotate_critical_roles(A, "er", seed)
            stats = compute_all(g)
            payload = _serialize_graph_sample(g, stats, extra_params={"p": p})
            filename = f"er_N{ER_N}_p{p:.2f}_seed{seed}.pt"
            path = os.path.join(out_dir, filename)
            sha = _save_torch(payload, path)
            files.append({
                "path": filename, "topology": "er", "N": ER_N, "D": 8, "seed": int(seed),
                "params": {"p": float(p)}, "stats": stats,
                "sha256": sha, "size_bytes": int(os.path.getsize(path)),
            })

    manifest = {
        "version": 1, "subdir": "synthetic_graphs",
        "n_files": len(files), "generated_at_jst": now_jst(),
        "env": _env_meta(), "files": files,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    # 计算 manifest sha256
    with open(manifest_path, "rb") as f:
        manifest_sha = _sha256_bytes(f.read())
    print(f"  ✓ synthetic_graphs: {len(files)} files in {time.time()-t0:.1f}s, manifest sha256 {manifest_sha[:16]}")
    return {"subdir": "synthetic_graphs", "n_files": len(files), "manifest_sha256_16": manifest_sha[:16]}


# ---------------------------------------------------------------------------
# Section 2: synthetic_rollouts/ — FE rollout 预生成
# ---------------------------------------------------------------------------

def gen_synthetic_rollouts(data_root: str) -> Dict[str, Any]:
    """对 default suite (7 拓扑 × N=50 × 3 outer seed) 生成 FE rollout T=50 训练 + T=32 评估.

    每条 trajectory 不同 X_0 (内 seed_init), 共 100 train + 20 val + 20 test = 140 / (top, seed).
    7×3×140 = 2940 trajectories.

    存格式: 一个 (top, seed) 一个文件, 包含所有 140 条 trajectory.
    """
    out_dir = os.path.join(data_root, "synthetic_rollouts")
    os.makedirs(out_dir, exist_ok=True)
    files = []
    t0 = time.time()
    T_train = 50

    for top in TOP_DEFAULTS:
        for outer_seed in OUTER_SEEDS:
            params = TOP_DEFAULTS.get(top, {})
            g = generate(top, N=N_DEFAULT, seed=outer_seed, **params)
            # W/U 由 env_seed=(top, N, outer_seed) 共享, inner seed 控制 X0/actions/noise
            # (per dataset_layout.md §3.1)
            train_X, val_X, test_X = [], [], []
            train_actions, val_actions, test_actions = [], [], []
            W_shared = None
            U_shared = None
            for split_name, n_traj, start in [("train", 100, 0), ("val", 20, 100), ("test", 20, 120)]:
                for inner_idx in range(n_traj):
                    inner_seed = start + inner_idx
                    tr = rollout(g, T=T_train, mode="fixed_edge",
                                 sigma_noise=0.01, seed=inner_seed)
                    if W_shared is None:
                        W_shared = tr.W.copy()
                        U_shared = tr.U.copy()
                    # rollout 默认 env_seed=hash(top, N, graph.seed), inner trajectory 间 W/U 自动一致
                    if split_name == "train":
                        train_X.append(tr.X); train_actions.append(tr.actions)
                    elif split_name == "val":
                        val_X.append(tr.X); val_actions.append(tr.actions)
                    else:
                        test_X.append(tr.X); test_actions.append(tr.actions)
            payload = {
                "version": 1, "topology": top, "N": N_DEFAULT, "D": 8,
                "outer_seed": outer_seed,
                "T_train": T_train,
                "W": W_shared, "U": U_shared,
                "train_X": np.stack(train_X),               # (100, T+1, N, D)
                "train_actions": np.stack(train_actions),   # (100, T, D_a)
                "val_X": np.stack(val_X),                   # (20, T+1, N, D)
                "val_actions": np.stack(val_actions),
                "test_X": np.stack(test_X),
                "test_actions": np.stack(test_actions),
                "config": {"mode": "fixed_edge", "sigma_noise": 0.01,
                           "target_w_norm": 0.9, "target_u_norm": 0.5},
            }
            filename = f"fe_{top}_N{N_DEFAULT}_seed{outer_seed}_T{T_train}.pt"
            path = os.path.join(out_dir, filename)
            sha = _save_torch(payload, path)
            size = os.path.getsize(path)
            files.append({"path": filename, "topology": top, "N": N_DEFAULT,
                          "outer_seed": int(outer_seed),
                          "n_train": 100, "n_val": 20, "n_test": 20,
                          "T": T_train, "sha256": sha, "size_bytes": int(size)})

    manifest = {
        "version": 1, "subdir": "synthetic_rollouts",
        "n_files": len(files), "generated_at_jst": now_jst(),
        "env": _env_meta(), "files": files,
    }
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(mpath, "rb") as f:
        manifest_sha = _sha256_bytes(f.read())
    print(f"  ✓ synthetic_rollouts: {len(files)} files in {time.time()-t0:.1f}s, manifest sha256 {manifest_sha[:16]}")
    return {"subdir": "synthetic_rollouts", "n_files": len(files), "manifest_sha256_16": manifest_sha[:16]}


# ---------------------------------------------------------------------------
# Section 3: injection_data/ — Exp 3/4/5/6 注入轨迹
# ---------------------------------------------------------------------------

INJECTION_KINDS = ["node_hub", "node_bridge", "node_leaf", "edge_random", "node_action"]


def gen_injection_data(data_root: str) -> Dict[str, Any]:
    """对 default suite (7 拓扑 × 3 outer seed) 跑 5 种注入, 每种 5 inner seed."""
    out_dir = os.path.join(data_root, "injection_data")
    os.makedirs(out_dir, exist_ok=True)
    files = []
    t0 = time.time()
    T_inj = 32                  # Exp 3/4 default horizon
    rng_per_kind = np.random.default_rng(0)

    for top in TOP_DEFAULTS:
        for outer_seed in OUTER_SEEDS:
            params = TOP_DEFAULTS.get(top, {})
            g = generate(top, N=N_DEFAULT, seed=outer_seed, **params)
            crit = g.critical_roles
            for kind in INJECTION_KINDS:
                # 选注入节点 / 边 + perturbation 步
                if kind == "node_hub":
                    inj_nodes = crit.get("hub", [])[:1] or [0]
                elif kind == "node_bridge":
                    inj_nodes = crit.get("bridge", [])[:1] or [0]
                elif kind == "node_leaf":
                    inj_nodes = crit.get("leaf", [])[:1] or [N_DEFAULT - 1]
                elif kind == "node_action":
                    inj_nodes = crit.get("action", [])[:1] or [0]
                elif kind == "edge_random":
                    inj_nodes = []
                else:
                    inj_nodes = []

                clean_X_list = []
                pert_X_list = []
                inner_seeds = list(range(5))
                for inner in inner_seeds:
                    # clean rollout
                    tr_clean = rollout(g, T=T_inj, mode="fixed_edge",
                                       sigma_noise=0.01, seed=inner)
                    if kind.startswith("node_") and inj_nodes:
                        # node injection: 在 t = T_inj // 4 加一个 (N, D) perturbation, 只对 inj_nodes 非零
                        pert = np.zeros((N_DEFAULT, 8), dtype=np.float32)
                        for nid in inj_nodes:
                            if 0 <= nid < N_DEFAULT:
                                pert[nid] = np.random.default_rng(seed=hash((kind, inner, nid)) % (2**31)) \
                                    .standard_normal(8).astype(np.float32) * 0.5
                        schedule = [(T_inj // 4, {"perturbation": pert})]
                    elif kind == "edge_random":
                        # edge perturbation: 给整张 X 加噪声 (近似)
                        pert = np.random.default_rng(seed=hash((kind, inner, "all")) % (2**31)) \
                            .standard_normal((N_DEFAULT, 8)).astype(np.float32) * 0.1
                        schedule = [(T_inj // 4, {"perturbation": pert})]
                    else:
                        schedule = []
                    tr_pert = rollout(g, T=T_inj, mode="fixed_edge",
                                      sigma_noise=0.01,
                                      injection_schedule=schedule, seed=inner)
                    clean_X_list.append(tr_clean.X)
                    pert_X_list.append(tr_pert.X)

                payload = {
                    "version": 1, "topology": top, "N": N_DEFAULT,
                    "outer_seed": int(outer_seed),
                    "injection_kind": kind,
                    "injection_nodes": [int(x) for x in inj_nodes],
                    "T": T_inj,
                    "inner_seeds": inner_seeds,
                    "clean_X": np.stack(clean_X_list),       # (n_inner, T+1, N, D)
                    "pert_X": np.stack(pert_X_list),
                }
                filename = f"{kind}_{top}_N{N_DEFAULT}_seed{outer_seed}.pt"
                path = os.path.join(out_dir, filename)
                sha = _save_torch(payload, path)
                files.append({"path": filename, "topology": top, "N": N_DEFAULT,
                              "outer_seed": int(outer_seed), "injection_kind": kind,
                              "n_inner_seeds": len(inner_seeds), "T": T_inj,
                              "sha256": sha, "size_bytes": int(os.path.getsize(path))})

    manifest = {
        "version": 1, "subdir": "injection_data",
        "n_files": len(files), "generated_at_jst": now_jst(),
        "env": _env_meta(), "files": files,
    }
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(mpath, "rb") as f:
        sha = _sha256_bytes(f.read())
    print(f"  ✓ injection_data: {len(files)} files in {time.time()-t0:.1f}s, manifest sha256 {sha[:16]}")
    return {"subdir": "injection_data", "n_files": len(files), "manifest_sha256_16": sha[:16]}


# ---------------------------------------------------------------------------
# Section 4: agent_calling_tree/ — 500 + 100 + 100 + 50 instance
# ---------------------------------------------------------------------------

def _save_hetero(sample, trace, path: str) -> Tuple[str, int]:
    payload = {
        "version": 1,
        "kind": "agent_calling_tree",
        "nodes": sample.nodes,
        "edges": sample.edges,
        "node_types": sample.node_types,
        "edge_index": sample.edge_index,
        "edge_type": sample.edge_type,
        "features_init": sample.features_init,
        "critical_roles": sample.critical_roles,
        "oracle_path": sample.oracle_path,
        "oracle_answer_node": sample.oracle_answer_node,
        "meta": sample.meta,
        "trace_X": trace.X,
        "trace_actions": trace.actions,
        "trace_action_ids": trace.action_ids,
        "trace_final_sr": trace.final_sr,
        "trace_config": trace.config,
    }
    sha = _save_torch(payload, path)
    return sha, os.path.getsize(path)


def gen_agent_calling_tree(data_root: str) -> Dict[str, Any]:
    """生成 500 train + 100 val + 100 test + 50 ood (N=100)."""
    out_dir = os.path.join(data_root, "agent_calling_tree")
    splits = [("train", 500, 0, 50), ("val", 100, 500, 50),
              ("test", 100, 600, 50), ("ood_test", 50, 700, 100)]
    files = []
    t0 = time.time()
    for split, n, base, N_target in splits:
        sub = os.path.join(out_dir, split)
        os.makedirs(sub, exist_ok=True)
        for i in range(n):
            seed = base + i
            sample = generate_calling_tree(N_target=N_target, seed=seed)
            trace = simulate_calling_tree(sample, T=32, seed=seed,
                                          action_policy="random")
            filename = f"instance_{i:04d}.pt"
            path = os.path.join(sub, filename)
            sha, size = _save_hetero(sample, trace, path)
            files.append({"path": f"{split}/{filename}", "split": split,
                          "instance_id": i, "seed": seed, "N": sample.features_init.shape[0],
                          "T": 32, "final_sr": float(trace.final_sr),
                          "sha256": sha, "size_bytes": int(size)})

    manifest = {
        "version": 1, "subdir": "agent_calling_tree",
        "n_files": len(files), "generated_at_jst": now_jst(),
        "env": _env_meta(),
        "splits": {s: n for s, n, *_ in splits},
        "files": files,
    }
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(mpath, "rb") as f:
        sha = _sha256_bytes(f.read())
    print(f"  ✓ agent_calling_tree: {len(files)} files in {time.time()-t0:.1f}s, manifest sha256 {sha[:16]}")
    return {"subdir": "agent_calling_tree", "n_files": len(files), "manifest_sha256_16": sha[:16]}


# ---------------------------------------------------------------------------
# Section 5: ocplatform_skill_graph/ — 300 + 60 + 60 + 30 instance
# ---------------------------------------------------------------------------

def _save_skill_hetero(sample, trace, path: str) -> Tuple[str, int]:
    payload = {
        "version": 1,
        "kind": "platform_skill_graph",
        "nodes": sample.nodes,
        "edges": sample.edges,
        "node_types": sample.node_types,
        "edge_index": sample.edge_index,
        "edge_type": sample.edge_type,
        "features_init": sample.features_init,
        "critical_roles": sample.critical_roles,
        "meta": sample.meta,
        "trace_X": trace.X,
        "trace_actions": trace.actions,
        "trace_action_ids": trace.action_ids,
        "trace_final_sr": trace.final_sr,
        "trace_config": trace.config,
    }
    sha = _save_torch(payload, path)
    return sha, os.path.getsize(path)


def gen_skill_graph(data_root: str) -> Dict[str, Any]:
    out_dir = os.path.join(data_root, "platform_skill_graph")
    splits = [("train", 300, 0, 100), ("val", 60, 300, 100),
              ("test", 60, 360, 100), ("ood_test", 30, 420, 200)]
    files = []
    t0 = time.time()
    for split, n, base, N_target in splits:
        sub = os.path.join(out_dir, split)
        os.makedirs(sub, exist_ok=True)
        for i in range(n):
            seed = base + i
            sample = generate_skill_graph(N_target=N_target, seed=seed)
            trace = simulate_skill_graph(sample, T=32, seed=seed,
                                         action_policy="random")
            filename = f"instance_{i:04d}.pt"
            path = os.path.join(sub, filename)
            sha, size = _save_skill_hetero(sample, trace, path)
            files.append({"path": f"{split}/{filename}", "split": split,
                          "instance_id": i, "seed": seed, "N": sample.features_init.shape[0],
                          "T": 32, "final_skill_sr": float(trace.final_sr),
                          "sha256": sha, "size_bytes": int(size)})

    manifest = {
        "version": 1, "subdir": "platform_skill_graph",
        "n_files": len(files), "generated_at_jst": now_jst(),
        "env": _env_meta(),
        "splits": {s: n for s, n, *_ in splits},
        "files": files,
    }
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(mpath, "rb") as f:
        sha = _sha256_bytes(f.read())
    print(f"  ✓ platform_skill_graph: {len(files)} files in {time.time()-t0:.1f}s, manifest sha256 {sha[:16]}")
    return {"subdir": "platform_skill_graph", "n_files": len(files), "manifest_sha256_16": sha[:16]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="space-separated section names to skip")
    args = parser.parse_args()
    os.makedirs(args.data_root, exist_ok=True)
    print(f"[{now_jst()}] P2 全量数据生成启动; data_root={args.data_root}")
    print(f"  skip: {args.skip}")

    summary = []
    if "synthetic_graphs" not in args.skip:
        summary.append(gen_synthetic_graphs(args.data_root))
    if "synthetic_rollouts" not in args.skip:
        summary.append(gen_synthetic_rollouts(args.data_root))
    if "injection_data" not in args.skip:
        summary.append(gen_injection_data(args.data_root))
    if "agent_calling_tree" not in args.skip:
        summary.append(gen_agent_calling_tree(args.data_root))
    if "platform_skill_graph" not in args.skip:
        summary.append(gen_skill_graph(args.data_root))

    # 写顶层 summary
    overview = {
        "timestamp_jst": now_jst(),
        "data_root": args.data_root,
        "env": _env_meta(),
        "summary": summary,
    }
    overview_path = os.path.join(REPO_ROOT, "results", "p2_data_landed.json")
    os.makedirs(os.path.dirname(overview_path), exist_ok=True)
    with open(overview_path, "w") as f:
        json.dump(overview, f, indent=2)
    print(f"\n[{now_jst()}] 完成; 写入 {overview_path}")
    for s in summary:
        print(f"  {s['subdir']:30s} {s['n_files']:6d} files  manifest_sha {s['manifest_sha256_16']}")


if __name__ == "__main__":
    main()
