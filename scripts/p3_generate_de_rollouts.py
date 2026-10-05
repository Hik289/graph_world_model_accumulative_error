from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import numpy as np
import scipy
import networkx as nx
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.graph_generators import generate
from src.simulators import rollout, generate_skill_graph

JST = timezone(timedelta(hours=9))
DATA_ROOT_DEFAULT = os.path.join(REPO_ROOT, 'data')

DE_TOPOS = ["chain", "tree", "grid", "small_world", "scale_free", "star"]
TOP_DEFAULTS = {
    "chain": {}, "tree": {"variant": "balanced_binary"},
    "grid": {"shape": "auto"}, "small_world": {"k": 4, "p": 0.1},
    "scale_free": {"m": 2}, "star": {},
}
OUTER_SEEDS = [1, 2, 3]
N_DEFAULT = 50
T_DE = 32
Q_SWEEP = [0.5, 1.0, 1.5, 2.0]


def now_jst() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def _git_hash() -> str:
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass

    return "unknown"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _env_meta(commit: str) -> Dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "networkx_version": nx.__version__,
        "torch_version": torch.__version__,
        "build_ts_jst": now_jst(),
        "generator_commit_hash": commit,
    }


def _save(obj, path) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)
    with open(path, "rb") as f:
        return _sha256(f.read())


def gen_de_synthetic(data_root: str, commit: str) -> Dict[str, Any]:

    out_dir = os.path.join(data_root, "de_synthetic")
    os.makedirs(out_dir, exist_ok=True)
    files = []
    t0 = time.time()


    for top in DE_TOPOS:
        for outer_seed in OUTER_SEEDS:
            params = TOP_DEFAULTS.get(top, {})
            g = generate(top, N=N_DEFAULT, seed=outer_seed, **params)

            train_X, train_A, train_actions = [], [], []
            val_X, val_A, val_actions = [], [], []
            test_X, test_A, test_actions = [], [], []
            W_shared, U_shared, Q_shared = None, None, None
            for split, n_traj, start in [("train", 80, 0), ("val", 10, 80), ("test", 10, 90)]:
                for idx in range(n_traj):
                    inner = start + idx
                    tr = rollout(g, T=T_DE, mode="dynamic_edge",
                                 sigma_noise=0.01, target_q_norm=1.0,
                                 seed=inner)
                    if W_shared is None:
                        W_shared = tr.W.copy(); U_shared = tr.U.copy(); Q_shared = tr.Q.copy()
                    if split == "train":
                        train_X.append(tr.X); train_A.append(tr.A); train_actions.append(tr.actions)
                    elif split == "val":
                        val_X.append(tr.X); val_A.append(tr.A); val_actions.append(tr.actions)
                    else:
                        test_X.append(tr.X); test_A.append(tr.A); test_actions.append(tr.actions)
            payload = {
                "version": 1, "topology": top, "N": N_DEFAULT, "D": 8,
                "outer_seed": int(outer_seed), "T": T_DE, "mode": "dynamic_edge",
                "Q_norm": 1.0,
                "W": W_shared, "U": U_shared, "Q": Q_shared,
                "train_X": np.stack(train_X),
                "train_A": np.stack(train_A),
                "train_actions": np.stack(train_actions),
                "val_X": np.stack(val_X), "val_A": np.stack(val_A),
                "val_actions": np.stack(val_actions),
                "test_X": np.stack(test_X), "test_A": np.stack(test_A),
                "test_actions": np.stack(test_actions),
                "config": {"mode": "dynamic_edge", "sigma_noise": 0.01,
                           "target_w_norm": 0.9, "target_u_norm": 0.5,
                           "target_q_norm": 1.0},
            }
            filename = f"de_{top}_N{N_DEFAULT}_seed{outer_seed}_T{T_DE}.pt"
            path = os.path.join(out_dir, filename)
            sha = _save(payload, path)
            size = os.path.getsize(path)
            files.append({"path": filename, "topology": top, "N": N_DEFAULT,
                          "outer_seed": int(outer_seed),
                          "Q_norm": 1.0,
                          "n_train": 80, "n_val": 10, "n_test": 10,
                          "T": T_DE, "sha256": sha, "size_bytes": int(size)})


    for q_norm in Q_SWEEP:
        if abs(q_norm - 1.0) < 1e-6:
            continue
        for outer_seed in OUTER_SEEDS:
            g = generate("scale_free", N=N_DEFAULT, seed=outer_seed, m=2)
            train_X, train_A, train_actions = [], [], []
            val_X, val_A, val_actions = [], [], []
            test_X, test_A, test_actions = [], [], []
            W_shared, U_shared, Q_shared = None, None, None
            for split, n_traj, start in [("train", 80, 0), ("val", 10, 80), ("test", 10, 90)]:
                for idx in range(n_traj):
                    inner = start + idx
                    tr = rollout(g, T=T_DE, mode="dynamic_edge",
                                 sigma_noise=0.01, target_q_norm=q_norm,
                                 seed=inner)
                    if W_shared is None:
                        W_shared = tr.W.copy(); U_shared = tr.U.copy(); Q_shared = tr.Q.copy()
                    if split == "train":
                        train_X.append(tr.X); train_A.append(tr.A); train_actions.append(tr.actions)
                    elif split == "val":
                        val_X.append(tr.X); val_A.append(tr.A); val_actions.append(tr.actions)
                    else:
                        test_X.append(tr.X); test_A.append(tr.A); test_actions.append(tr.actions)
            payload = {
                "version": 1, "topology": "scale_free", "N": N_DEFAULT, "D": 8,
                "outer_seed": int(outer_seed), "T": T_DE, "mode": "dynamic_edge",
                "Q_norm": float(q_norm),
                "W": W_shared, "U": U_shared, "Q": Q_shared,
                "train_X": np.stack(train_X), "train_A": np.stack(train_A),
                "train_actions": np.stack(train_actions),
                "val_X": np.stack(val_X), "val_A": np.stack(val_A),
                "val_actions": np.stack(val_actions),
                "test_X": np.stack(test_X), "test_A": np.stack(test_A),
                "test_actions": np.stack(test_actions),
                "config": {"mode": "dynamic_edge", "sigma_noise": 0.01,
                           "target_w_norm": 0.9, "target_u_norm": 0.5,
                           "target_q_norm": float(q_norm)},
            }
            filename = f"de_scale_free_N{N_DEFAULT}_q{q_norm:.1f}_seed{outer_seed}_T{T_DE}.pt"
            path = os.path.join(out_dir, filename)
            sha = _save(payload, path)
            size = os.path.getsize(path)
            files.append({"path": filename, "topology": "scale_free", "N": N_DEFAULT,
                          "outer_seed": int(outer_seed),
                          "Q_norm": float(q_norm),
                          "n_train": 80, "n_val": 10, "n_test": 10,
                          "T": T_DE, "sha256": sha, "size_bytes": int(size)})

    manifest = {
        "version": 1, "subdir": "de_synthetic",
        "commit_hash": commit, "build_ts_jst": now_jst(),
        "n_files": len(files), "env": _env_meta(commit),
        "files": files,
    }
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(mpath, "rb") as f:
        sha = _sha256(f.read())
    print(f"  ✓ de_synthetic: {len(files)} files in {time.time()-t0:.1f}s, manifest sha256 {sha[:16]}")
    return {"subdir": "de_synthetic", "n_files": len(files), "manifest_sha256_16": sha[:16]}


def gen_de_skill_graph(data_root: str, commit: str, n_per_seed: int = 30) -> Dict[str, Any]:


    from src.simulators import simulate_skill_graph
    out_dir = os.path.join(data_root, "de_skill_graph")
    os.makedirs(out_dir, exist_ok=True)
    files = []
    t0 = time.time()

    for outer_seed in [0, 1, 2]:
        for inst in range(n_per_seed):
            seed = outer_seed * 1000 + inst
            sample = generate_skill_graph(N_target=100, seed=seed)
            trace = simulate_skill_graph(sample, T=T_DE, seed=seed, action_policy="random")
            payload = {
                "version": 1, "kind": "platform_skill_graph_de",
                "outer_seed": int(outer_seed), "instance_id": int(inst), "seed": int(seed),
                "T": T_DE,
                "nodes": sample.nodes, "edges": sample.edges,
                "node_types": sample.node_types,
                "edge_index": sample.edge_index, "edge_type": sample.edge_type,
                "features_init": sample.features_init,
                "critical_roles": sample.critical_roles, "meta": sample.meta,
                "trace_X": trace.X, "trace_actions": trace.actions,
                "trace_action_ids": trace.action_ids,
                "trace_final_sr": trace.final_sr, "trace_config": trace.config,
            }
            filename = f"seed{outer_seed}/instance_{inst:04d}.pt"
            path = os.path.join(out_dir, filename)
            sha = _save(payload, path)
            size = os.path.getsize(path)
            files.append({"path": filename, "outer_seed": int(outer_seed),
                          "instance_id": int(inst), "seed": int(seed),
                          "N": int(sample.features_init.shape[0]),
                          "T": T_DE, "sha256": sha, "size_bytes": int(size)})

    manifest = {
        "version": 1, "subdir": "de_skill_graph",
        "commit_hash": commit, "build_ts_jst": now_jst(),
        "n_files": len(files), "env": _env_meta(commit),
        "splits": {"seed0": n_per_seed, "seed1": n_per_seed, "seed2": n_per_seed},
        "files": files,
    }
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(mpath, "rb") as f:
        sha = _sha256(f.read())
    print(f"  ✓ de_skill_graph: {len(files)} files in {time.time()-t0:.1f}s, manifest sha256 {sha[:16]}")
    return {"subdir": "de_skill_graph", "n_files": len(files), "manifest_sha256_16": sha[:16]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--summary_path", default=os.path.join(REPO_ROOT, "results", "p3_de_rollouts_landed.json"))
    args = parser.parse_args()
    commit = _git_hash()
    print(f"[{now_jst()}] DE rollouts 补落盘启动")
    print(f"  commit: {commit}")
    summary = []
    if "de_synthetic" not in args.skip:
        summary.append(gen_de_synthetic(args.data_root, commit))
    if "de_skill_graph" not in args.skip:
        summary.append(gen_de_skill_graph(args.data_root, commit))
    overview = {
        "timestamp_jst": now_jst(),
        "data_root": args.data_root,
        "commit_hash": commit,
        "summary": summary,
    }
    overview_path = os.path.abspath(args.summary_path)
    os.makedirs(os.path.dirname(overview_path), exist_ok=True)
    with open(overview_path, "w") as f:
        json.dump(overview, f, indent=2)
    print(f"\n[{now_jst()}] 完成; 写入 {overview_path}")
    for s in summary:
        print(f"  {s['subdir']:30s} {s['n_files']:6d} files  manifest_sha {s['manifest_sha256_16']}")


if __name__ == "__main__":
    main()
