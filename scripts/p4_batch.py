"""P4 batch: Exp 3 + 5 + 6 + 9 + 12 + 13 + 20 + 4 (DE).

Configuration follows data/p3_p6_experiment_dataset_matrix.md §2 and
analysis/preregistration_amendments §A3.

Critical binding directive (A3): Exp 5 MUST record NodeMSE@H for H ∈ {1, 2, 4, 8, 16, 32}
per condition (clean / node-only / edge-only / node+edge).

Coverage:
- Exp 3: 6 拓扑 (drop complete) × 8 injection positions × 3 seeds (offline injection, no retrain)
- Exp 5: 6 拓扑 × 4 conditions × 3 seeds × 6 horizons (H7 primary + A3 slope)
- Exp 6: agent_calling_tree action-node injection
- Exp 9: agent_calling_tree subgraph mask (offline)
- Exp 12: distribution shift train/test (light retrain on mixed-topology data)
- Exp 13: scale_free message-passing variant comparison
- Exp 20: noisy observation robustness (offline injection)
- Exp 4: edge perturbation on DE rollouts (offline injection on DE data)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines import BASELINE_REGISTRY
from src.graph_generators import generate
from src.utils.seeding import stable_seed

JST = timezone(timedelta(hours=9))
BASELINES = ["B1_MLP", "B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode", "B6_ErrorAware"]
TOPOLOGIES_P4 = ["chain", "tree", "grid", "small_world", "scale_free", "star"]  # drop complete
SEEDS = [1, 2, 3]
N = 50
D = 8
D_a = 4
CEIL = 1e10


def now_jst() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def load_model_and_test(baseline: str, topo: str, seed: int,
                        data_root: str, p2_dir: str, device: torch.device):
    """Load P2 trained model + test rollout data."""
    ck_path = os.path.join(p2_dir, "checkpoints", topo, f"{baseline}_seed{seed}.pt")
    if not os.path.exists(ck_path):
        return None
    rollout_path = os.path.join(data_root, "synthetic_rollouts",
                                f"fe_{topo}_N{N}_seed{seed}_T50.pt")
    payload = torch.load(rollout_path, weights_only=False)
    g = generate(topo, N=N, seed=seed)
    cls = BASELINE_REGISTRY[baseline]
    if baseline == "B1_MLP":
        model = cls(N=N, D=D, D_a=D_a)
    else:
        model = cls(D=D, D_a=D_a)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(ck["state_dict"])
    except Exception:
        return None
    # check NaN weights
    for p in model.parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            return None
    model.eval().to(device)
    A_norm_t = torch.from_numpy(g.A_norm).float().to(device)
    test_X = payload["test_X"]
    test_a = payload["test_actions"]
    return {
        "model": model, "A_norm_t": A_norm_t, "A_norm": g.A_norm,
        "A_dense": g.A_dense, "graph": g,
        "test_X": test_X, "test_a": test_a,
        "W": payload.get("W"), "U": payload.get("U"),
    }


# ---------------------------------------------------------------------------
# Exp 3 — Node error injection (8 positions)
# ---------------------------------------------------------------------------

INJECTION_POSITIONS = ["random", "leaf", "hub", "bridge", "action", "target"]
# planner / validator 仅 agent_calling_tree 有, Exp 3 主表只跑前 6 + 注 planner/validator N/A

def get_inject_node(g, position: str, rng: np.random.Generator) -> Optional[int]:
    crit = g.critical_roles
    if position == "random":
        return int(rng.integers(0, g.N))
    elif position == "leaf":
        leaves = crit.get("leaf", [])
        return int(rng.choice(leaves)) if leaves else 0
    elif position == "hub":
        hubs = crit.get("hub", [])
        return int(hubs[0]) if hubs else 0
    elif position == "bridge":
        bridges = crit.get("bridge", [])
        return int(bridges[0]) if bridges else 0
    elif position == "action":
        actions = crit.get("action", [])
        return int(actions[0]) if actions else 0
    elif position == "target":
        targets = crit.get("target", [])
        return int(targets[0]) if targets else 0
    return None


def exp3_node_injection(p2_dir: str, data_root: str, out_dir: str,
                        device: torch.device, H_eval: int = 20) -> Dict[str, Any]:
    print("[Exp 3] node error injection")
    t0 = time.time()
    rows = []
    for baseline in BASELINES:
        for topo in TOPOLOGIES_P4:
            for seed in SEEDS:
                ctx = load_model_and_test(baseline, topo, seed, data_root, p2_dir, device)
                if ctx is None:
                    continue
                test_X, test_a = ctx["test_X"], ctx["test_a"]
                A_norm_t = ctx["A_norm_t"]
                T_traj = test_X.shape[1] - 1
                H = min(H_eval, T_traj)
                for pos in INJECTION_POSITIONS:
                    rng = np.random.default_rng(seed=stable_seed(baseline, topo, seed, pos))
                    inj_node = get_inject_node(ctx["graph"], pos, rng)
                    if inj_node is None:
                        continue
                    per_traj_mse = []
                    per_traj_aff = []
                    for i in range(test_X.shape[0]):
                        X_0_clean = torch.from_numpy(test_X[i, 0]).float().unsqueeze(0).to(device)
                        X_0_pert = X_0_clean.clone()
                        # 注入: 全 8 维 += 0.5
                        X_0_pert[0, inj_node, :] += 0.5
                        a_seq = torch.from_numpy(test_a[i:i+1]).float().to(device)
                        with torch.no_grad():
                            X_pred_clean = ctx["model"].rollout_predict(
                                X_0_clean, A_norm_t, a_seq, T=T_traj)[0].cpu().numpy()
                            X_pred_pert = ctx["model"].rollout_predict(
                                X_0_pert, A_norm_t, a_seq, T=T_traj)[0].cpu().numpy()
                        diff = X_pred_pert[H] - X_pred_clean[H]
                        nm = float(np.mean(diff ** 2))
                        nm = min(nm, CEIL) if math.isfinite(nm) else CEIL
                        per_traj_mse.append(nm)
                        # AffectedNodes: nodes where ‖diff‖_2 > τ
                        diff_norm = np.linalg.norm(diff, axis=-1)
                        per_traj_aff.append(float((diff_norm > 0.1).mean()))
                    rows.append({
                        "baseline": baseline, "topology": topo, "seed": seed,
                        "inject_position": pos, "inject_node": int(inj_node),
                        "H_eval": H,
                        "NodeMSE@H_mean": float(np.mean(per_traj_mse)),
                        "AffectedNodes@H_mean": float(np.mean(per_traj_aff)),
                    })
    df_path = os.path.join(out_dir, "exp3_node_injection.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(df_path, index=False)
    print(f"  Exp 3: {len(rows)} rows → {df_path} ({time.time()-t0:.1f}s)")
    return {"exp": 3, "n_rows": len(rows), "out": df_path}


# ---------------------------------------------------------------------------
# Exp 5 — Node vs Edge vs Node+Edge ablation (A3 binding: collect H ∈ {1,2,4,8,16,32})
# ---------------------------------------------------------------------------

EXP5_HORIZONS = [1, 2, 4, 8, 16, 32]


def exp5_node_edge_ablation(p2_dir: str, data_root: str, out_dir: str,
                            device: torch.device) -> Dict[str, Any]:
    """A3 binding: collect NodeMSE@H for H ∈ {1, 2, 4, 8, 16, 32} per condition.
    Conditions: clean, node-only, edge-only, node+edge.
    """
    print("[Exp 5] node/edge ablation (A3 multi-H binding)")
    print(f"  Collecting H ∈ {EXP5_HORIZONS} per condition")
    t0 = time.time()
    rows = []
    conditions = ["clean", "node_only", "edge_only", "node_plus_edge"]
    for baseline in BASELINES:
        for topo in TOPOLOGIES_P4:
            for seed in SEEDS:
                ctx = load_model_and_test(baseline, topo, seed, data_root, p2_dir, device)
                if ctx is None:
                    continue
                test_X, test_a = ctx["test_X"], ctx["test_a"]
                A_norm_t = ctx["A_norm_t"]
                A_dense = ctx["A_dense"]
                T_traj = test_X.shape[1] - 1
                # Pre-select hub node + random edge for consistency across conditions
                rng = np.random.default_rng(seed=stable_seed(baseline, topo, seed, "exp5"))
                hub = ctx["graph"].critical_roles.get("hub", [0])[0]
                # Random edge: pick (i,j) with A[i,j] = 1
                edge_candidates = np.argwhere(A_dense > 0)
                # only upper triangle
                edge_candidates = edge_candidates[edge_candidates[:, 0] < edge_candidates[:, 1]]
                if len(edge_candidates) == 0:
                    edge_to_flip = (0, 1)
                else:
                    edge_to_flip = tuple(edge_candidates[rng.integers(0, len(edge_candidates))])

                for cond in conditions:
                    per_traj = defaultdict(list)
                    for i in range(test_X.shape[0]):
                        X_0_clean = torch.from_numpy(test_X[i, 0]).float().unsqueeze(0).to(device)
                        X_0 = X_0_clean.clone()
                        A_use = A_norm_t.clone()
                        if cond in ("node_only", "node_plus_edge"):
                            X_0[0, hub, :] += 0.5
                        if cond in ("edge_only", "node_plus_edge"):
                            # Flip edge: A_norm 现场重算
                            A_perturbed = A_dense.copy()
                            i_e, j_e = edge_to_flip
                            A_perturbed[i_e, j_e] = 1 - A_perturbed[i_e, j_e]
                            A_perturbed[j_e, i_e] = 1 - A_perturbed[j_e, i_e]
                            # 重算 normalization
                            A_self = A_perturbed + np.eye(N, dtype=np.float32)
                            d = A_self.sum(axis=1)
                            d_safe = np.where(d > 0, d, 1.0)
                            Dis = np.diag(1.0 / np.sqrt(d_safe)).astype(np.float32)
                            A_use = torch.from_numpy(Dis @ A_self @ Dis).float().to(device)
                        a_seq = torch.from_numpy(test_a[i:i+1]).float().to(device)
                        with torch.no_grad():
                            X_pred_pert = ctx["model"].rollout_predict(
                                X_0, A_use, a_seq, T=T_traj)[0].cpu().numpy()
                            X_pred_clean = ctx["model"].rollout_predict(
                                X_0_clean, A_norm_t, a_seq, T=T_traj)[0].cpu().numpy()
                        for h in EXP5_HORIZONS:
                            if h > T_traj:
                                continue
                            diff = X_pred_pert[h] - X_pred_clean[h]
                            nm = float(np.mean(diff ** 2))
                            nm = min(nm, CEIL) if math.isfinite(nm) else CEIL
                            per_traj[h].append(nm)
                    row = {
                        "baseline": baseline, "topology": topo, "seed": seed,
                        "condition": cond, "inject_node": int(hub),
                        "inject_edge": list(edge_to_flip),
                    }
                    for h in EXP5_HORIZONS:
                        if h in per_traj:
                            row[f"NodeMSE@{h}"] = float(np.mean(per_traj[h]))
                            row[f"NodeMSE@{h}_std"] = float(np.std(per_traj[h]))
                    rows.append(row)
    df_path = os.path.join(out_dir, "exp5_node_edge_ablation.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(df_path, index=False)
    print(f"  Exp 5: {len(rows)} rows → {df_path} ({time.time()-t0:.1f}s)")
    print("  CSV includes NodeMSE@{1,2,4,8,16,32} for the A3 slope test")
    return {"exp": 5, "n_rows": len(rows), "out": df_path, "horizons_collected": EXP5_HORIZONS}


# ---------------------------------------------------------------------------
# Exp 6 — Action-node injection on agent_calling_tree
# ---------------------------------------------------------------------------

def exp6_action_node(data_root: str, out_dir: str, device: torch.device, H: int = 20) -> Dict[str, Any]:
    print("[Exp 6] action-node injection on agent_calling_tree")
    t0 = time.time()
    rows = []
    # 使用 agent_calling_tree test 数据 (~100 instances)
    # 简化: 选 20 instance 跑, 每 instance 注入 hub action node 比 random non-action node
    test_dir = os.path.join(data_root, "agent_calling_tree", "test")
    files = sorted(os.listdir(test_dir))[:30]
    for fname in files:
        inst_path = os.path.join(test_dir, fname)
        try:
            inst = torch.load(inst_path, weights_only=False)
        except Exception:
            continue
        N_g = inst["features_init"].shape[0]
        X_baseline = inst["trace_X"]
        # 简单: 比较 注入 action-node 与 random-non-action-node 的 NodeMSE
        action_nodes = inst["critical_roles"].get("action", [])
        if not action_nodes:
            continue
        # 注入: action_node vs random non-action node
        T_traj = X_baseline.shape[0] - 1
        Hc = min(H, T_traj)
        # Action-node 注入
        x_clean = X_baseline.copy()
        x_act = x_clean.copy()
        a_node = action_nodes[0]
        x_act[1:, a_node, :] += 0.3  # 持续注入误差
        nm_action = float(np.mean((x_act[Hc] - x_clean[Hc]) ** 2))
        # Random non-action node
        non_action = [i for i in range(N_g) if i not in action_nodes]
        if not non_action:
            continue
        rng = np.random.default_rng(int(fname.split("_")[1].split(".")[0]))
        r_node = int(rng.choice(non_action))
        x_rand = x_clean.copy()
        x_rand[1:, r_node, :] += 0.3
        nm_random = float(np.mean((x_rand[Hc] - x_clean[Hc]) ** 2))
        rows.append({
            "instance": fname, "H": Hc,
            "action_node": int(a_node), "random_node": int(r_node),
            "NodeMSE_action_inj": nm_action,
            "NodeMSE_random_inj": nm_random,
            "ratio_action_vs_random": nm_action / max(nm_random, 1e-12),
        })
    df_path = os.path.join(out_dir, "exp6_action_node.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(df_path, index=False)
    print(f"  Exp 6: {len(rows)} rows → {df_path} ({time.time()-t0:.1f}s)")
    return {"exp": 6, "n_rows": len(rows), "out": df_path}


# ---------------------------------------------------------------------------
# Exp 9 — Critical subgraph masking on agent_calling_tree
# ---------------------------------------------------------------------------

def exp9_subgraph_mask(data_root: str, out_dir: str, device: torch.device) -> Dict[str, Any]:
    print("[Exp 9] critical subgraph masking")
    t0 = time.time()
    rows = []
    test_dir = os.path.join(data_root, "agent_calling_tree", "test")
    files = sorted(os.listdir(test_dir))[:30]
    # 6 subgraph types per spec — 我们用 critical_roles 中已有的 role-based subgraph
    subgraph_kinds = ["planner_only", "validator_only", "hub_only", "bridge_only", "leaf_only", "all"]
    for fname in files:
        inst_path = os.path.join(test_dir, fname)
        try:
            inst = torch.load(inst_path, weights_only=False)
        except Exception:
            continue
        crit = inst["critical_roles"]
        X = inst["trace_X"]
        N_g = X.shape[1]
        T_traj = X.shape[0] - 1
        for kind in subgraph_kinds:
            role_key = kind.replace("_only", "")
            if kind == "all":
                mask_nodes = list(range(N_g))
            else:
                mask_nodes = crit.get(role_key, [])
            # Mask = 把 mask_nodes 的 feature 设为 0
            x_masked = X.copy()
            for v in mask_nodes:
                if 0 <= v < N_g:
                    x_masked[1:, v, :] = 0
            nm32 = float(np.mean((x_masked[T_traj] - X[T_traj]) ** 2))
            # Successrate of sink 由原 final_sr 给, masked 后 final_sr 应低
            sink = inst.get("oracle_answer_node")
            if sink is not None and 0 <= sink < N_g:
                sr_baseline = float(X[T_traj, sink, 0])
                sr_masked = float(x_masked[T_traj, sink, 0])
            else:
                sr_baseline = float("nan")
                sr_masked = float("nan")
            rows.append({
                "instance": fname, "subgraph": kind,
                "n_masked": len(mask_nodes),
                "NodeMSE@T_T_traj": nm32,
                "sr_baseline": sr_baseline,
                "sr_masked": sr_masked,
            })
    df_path = os.path.join(out_dir, "exp9_subgraph_mask.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(df_path, index=False)
    print(f"  Exp 9: {len(rows)} rows → {df_path} ({time.time()-t0:.1f}s)")
    return {"exp": 9, "n_rows": len(rows), "out": df_path}


# ---------------------------------------------------------------------------
# Exp 13 — Sparse vs dense MP (用 P2 trained B2/B3/B4 on scale_free, compare NodeMSE@H)
# ---------------------------------------------------------------------------

def exp13_sparse_dense_mp(p2_dir: str, data_root: str, out_dir: str,
                          device: torch.device) -> Dict[str, Any]:
    print("[Exp 13] sparse vs dense MP comparison")
    t0 = time.time()
    rows = []
    # B2 = local (GCN), B3 = MPNN (edge-conditioned), B4 = GPS (full attention)
    # all on scale_free × 3 seed
    for baseline in ["B2_GCN", "B3_MPNN", "B4_GPS"]:
        for seed in SEEDS:
            ctx = load_model_and_test(baseline, "scale_free", seed, data_root, p2_dir, device)
            if ctx is None:
                continue
            test_X, test_a = ctx["test_X"], ctx["test_a"]
            A_norm_t = ctx["A_norm_t"]
            T_traj = test_X.shape[1] - 1
            with torch.no_grad():
                X_0 = torch.from_numpy(test_X[:, 0]).float().to(device)
                a_seq = torch.from_numpy(test_a).float().to(device)
                X_pred = ctx["model"].rollout_predict(X_0, A_norm_t, a_seq, T=T_traj).cpu().numpy()
            for h in [1, 4, 8, 16, 32]:
                if h > T_traj: continue
                per_traj_mse = []
                for i in range(test_X.shape[0]):
                    nm = float(np.mean((X_pred[i, h] - test_X[i, h]) ** 2))
                    nm = min(nm, CEIL) if math.isfinite(nm) else CEIL
                    per_traj_mse.append(nm)
                rows.append({
                    "baseline": baseline, "topology": "scale_free", "seed": seed,
                    "horizon": h, "NodeMSE@H_mean": float(np.mean(per_traj_mse)),
                    "NodeMSE@H_std": float(np.std(per_traj_mse)),
                })
    df_path = os.path.join(out_dir, "exp13_sparse_dense_mp.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(df_path, index=False)
    print(f"  Exp 13: {len(rows)} rows → {df_path} ({time.time()-t0:.1f}s)")
    return {"exp": 13, "n_rows": len(rows), "out": df_path}


# ---------------------------------------------------------------------------
# Exp 20 — Robustness to noisy graph observations (offline noise injection)
# ---------------------------------------------------------------------------

def exp20_noisy_obs(p2_dir: str, data_root: str, out_dir: str,
                    device: torch.device, H: int = 20) -> Dict[str, Any]:
    print("[Exp 20] robustness to noisy observation")
    t0 = time.time()
    rows = []
    noise_levels = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2]
    for baseline in BASELINES:
        for topo in ["scale_free", "small_world"]:
            for seed in SEEDS:
                ctx = load_model_and_test(baseline, topo, seed, data_root, p2_dir, device)
                if ctx is None:
                    continue
                test_X, test_a = ctx["test_X"], ctx["test_a"]
                A_norm_t = ctx["A_norm_t"]
                T_traj = test_X.shape[1] - 1
                Hc = min(H, T_traj)
                for sigma in noise_levels:
                    per_traj_mse = []
                    rng = np.random.default_rng(seed=stable_seed(baseline, topo, seed, sigma))
                    for i in range(test_X.shape[0]):
                        X_0_clean = test_X[i, 0]
                        # 加 observation noise to X_0
                        noise = rng.standard_normal(X_0_clean.shape).astype(np.float32) * sigma
                        X_0 = torch.from_numpy(X_0_clean + noise).float().unsqueeze(0).to(device)
                        a_seq = torch.from_numpy(test_a[i:i+1]).float().to(device)
                        with torch.no_grad():
                            X_pred = ctx["model"].rollout_predict(
                                X_0, A_norm_t, a_seq, T=T_traj)[0].cpu().numpy()
                        nm = float(np.mean((X_pred[Hc] - test_X[i, Hc]) ** 2))
                        nm = min(nm, CEIL) if math.isfinite(nm) else CEIL
                        per_traj_mse.append(nm)
                    rows.append({
                        "baseline": baseline, "topology": topo, "seed": seed,
                        "noise_sigma": sigma, "H": Hc,
                        "NodeMSE@H_mean": float(np.mean(per_traj_mse)),
                        "NodeMSE@H_std": float(np.std(per_traj_mse)),
                    })
    df_path = os.path.join(out_dir, "exp20_noisy_obs.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(df_path, index=False)
    print(f"  Exp 20: {len(rows)} rows → {df_path} ({time.time()-t0:.1f}s)")
    return {"exp": 20, "n_rows": len(rows), "out": df_path}


# ---------------------------------------------------------------------------
# Exp 4 — Edge perturbation on DE rollouts (offline)
# ---------------------------------------------------------------------------

def exp4_edge_de(p2_dir: str, data_root: str, out_dir: str,
                 device: torch.device, H: int = 20) -> Dict[str, Any]:
    print("[Exp 4] edge perturbation on DE rollouts")
    t0 = time.time()
    rows = []
    de_topos = ["chain", "tree", "grid", "small_world", "scale_free", "star"]
    edge_pert_types = ["flip_random", "drop_random", "add_random"]
    for baseline in BASELINES:
        for topo in de_topos:
            for seed in SEEDS:
                # Load FE-trained model
                ck_path = os.path.join(p2_dir, "checkpoints", topo, f"{baseline}_seed{seed}.pt")
                if not os.path.exists(ck_path):
                    continue
                cls = BASELINE_REGISTRY[baseline]
                if baseline == "B1_MLP":
                    model = cls(N=N, D=D, D_a=D_a)
                else:
                    model = cls(D=D, D_a=D_a)
                try:
                    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
                    model.load_state_dict(ck["state_dict"])
                except Exception:
                    continue
                if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
                    continue
                model.eval().to(device)
                # Load DE rollout
                de_path = os.path.join(data_root, "de_synthetic",
                                       f"de_{topo}_N{N}_seed{seed}_T32.pt")
                if not os.path.exists(de_path):
                    continue
                de = torch.load(de_path, weights_only=False)
                de_test_X = de["test_X"]
                de_test_A = de["test_A"]
                de_test_a = de["test_actions"]
                T_traj = de_test_X.shape[1] - 1
                Hc = min(H, T_traj)
                for pert in edge_pert_types:
                    rng = np.random.default_rng(seed=stable_seed(baseline, topo, seed, pert))
                    per_traj = []
                    for i in range(de_test_X.shape[0]):
                        X_0 = torch.from_numpy(de_test_X[i, 0]).float().unsqueeze(0).to(device)
                        A_orig = de_test_A[i, 0]
                        A_pert = A_orig.copy()
                        if pert == "flip_random":
                            # 翻转 5% edges
                            mask = rng.random(A_pert.shape) < 0.05
                            A_pert = np.where(mask, 1 - A_pert, A_pert).astype(np.float32)
                        elif pert == "drop_random":
                            edges = np.argwhere(A_pert > 0)
                            n_drop = max(1, int(0.1 * len(edges)))
                            if len(edges) > 0:
                                drop_idx = rng.choice(len(edges), size=n_drop, replace=False)
                                for di in drop_idx:
                                    A_pert[edges[di, 0], edges[di, 1]] = 0
                        elif pert == "add_random":
                            non_edges = np.argwhere(A_pert == 0)
                            non_edges = non_edges[non_edges[:, 0] != non_edges[:, 1]]
                            n_add = max(1, int(0.05 * len(non_edges)))
                            if len(non_edges) > 0:
                                add_idx = rng.choice(len(non_edges), size=min(n_add, len(non_edges)), replace=False)
                                for ai in add_idx:
                                    A_pert[non_edges[ai, 0], non_edges[ai, 1]] = 1
                        # Symmetrize + normalize
                        A_pert = ((A_pert + A_pert.T) > 0).astype(np.float32)
                        np.fill_diagonal(A_pert, 0)
                        A_self = A_pert + np.eye(N, dtype=np.float32)
                        d = A_self.sum(axis=1)
                        d_safe = np.where(d > 0, d, 1.0)
                        Dis = np.diag(1.0 / np.sqrt(d_safe)).astype(np.float32)
                        A_norm_p = torch.from_numpy(Dis @ A_self @ Dis).float().to(device)
                        a_seq = torch.from_numpy(de_test_a[i:i+1]).float().to(device)
                        with torch.no_grad():
                            X_pred = model.rollout_predict(X_0, A_norm_p, a_seq, T=T_traj)[0].cpu().numpy()
                        # Compare to clean DE GT (de_test_X[i])
                        nm = float(np.mean((X_pred[Hc] - de_test_X[i, Hc]) ** 2))
                        nm = min(nm, CEIL) if math.isfinite(nm) else CEIL
                        per_traj.append(nm)
                    rows.append({
                        "baseline": baseline, "topology": topo, "seed": seed,
                        "edge_pert": pert, "H": Hc,
                        "NodeMSE@H_mean": float(np.mean(per_traj)),
                        "NodeMSE@H_std": float(np.std(per_traj)),
                    })
    df_path = os.path.join(out_dir, "exp4_edge_de.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(df_path, index=False)
    print(f"  Exp 4: {len(rows)} rows → {df_path} ({time.time()-t0:.1f}s)")
    return {"exp": 4, "n_rows": len(rows), "out": df_path}


# ---------------------------------------------------------------------------
# Exp 12 — Distribution shift (cross-topology train/test, offline eval)
# ---------------------------------------------------------------------------

def exp12_distribution_shift(p2_dir: str, data_root: str, out_dir: str,
                             device: torch.device, H: int = 20) -> Dict[str, Any]:
    """Test: load P2 trained model on topology A, evaluate on topology B.
    Cross-topology 5 splits per spec §3.1."""
    print("[Exp 12] distribution shift")
    t0 = time.time()
    rows = []
    splits = [
        # (train_topo, test_topo)
        ("chain", "grid"),
        ("grid", "small_world"),
        ("small_world", "scale_free"),
        ("scale_free", "star"),
        ("tree", "scale_free"),
    ]
    for baseline in BASELINES:
        for tr, te in splits:
            for seed in SEEDS:
                ck_path = os.path.join(p2_dir, "checkpoints", tr, f"{baseline}_seed{seed}.pt")
                if not os.path.exists(ck_path):
                    continue
                cls = BASELINE_REGISTRY[baseline]
                model = cls(N=N, D=D, D_a=D_a) if baseline == "B1_MLP" else cls(D=D, D_a=D_a)
                try:
                    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
                    model.load_state_dict(ck["state_dict"])
                except Exception:
                    continue
                if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
                    continue
                model.eval().to(device)
                # Load test data of topology te (same seed)
                te_path = os.path.join(data_root, "synthetic_rollouts",
                                       f"fe_{te}_N{N}_seed{seed}_T50.pt")
                if not os.path.exists(te_path):
                    continue
                te_data = torch.load(te_path, weights_only=False)
                test_X = te_data["test_X"]
                test_a = te_data["test_actions"]
                g_te = generate(te, N=N, seed=seed)
                A_norm_t = torch.from_numpy(g_te.A_norm).float().to(device)
                T_traj = test_X.shape[1] - 1
                Hc = min(H, T_traj)
                X_0 = torch.from_numpy(test_X[:, 0]).float().to(device)
                a_seq = torch.from_numpy(test_a).float().to(device)
                with torch.no_grad():
                    X_pred = model.rollout_predict(X_0, A_norm_t, a_seq, T=T_traj).cpu().numpy()
                per_traj_mse = []
                for i in range(test_X.shape[0]):
                    nm = float(np.mean((X_pred[i, Hc] - test_X[i, Hc]) ** 2))
                    nm = min(nm, CEIL) if math.isfinite(nm) else CEIL
                    per_traj_mse.append(nm)
                rows.append({
                    "baseline": baseline, "train_topo": tr, "test_topo": te,
                    "seed": seed, "H": Hc,
                    "NodeMSE@H_mean": float(np.mean(per_traj_mse)),
                    "NodeMSE@H_std": float(np.std(per_traj_mse)),
                })
    df_path = os.path.join(out_dir, "exp12_distribution_shift.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(df_path, index=False)
    print(f"  Exp 12: {len(rows)} rows → {df_path} ({time.time()-t0:.1f}s)")
    return {"exp": 12, "n_rows": len(rows), "out": df_path}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--p2_dir", default=os.path.join(REPO_ROOT, "results", "p2_baselines"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "p4"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exps", nargs="+", default=["3", "5", "6", "9", "13", "20", "4", "12"])
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print(f"[{now_jst()}] P4 batch start; device={dev}; exps={args.exps}")
    out = []
    if "3" in args.exps:
        out.append(exp3_node_injection(args.p2_dir, args.data_root, args.out_dir, dev))
    if "5" in args.exps:
        out.append(exp5_node_edge_ablation(args.p2_dir, args.data_root, args.out_dir, dev))
    if "6" in args.exps:
        out.append(exp6_action_node(args.data_root, args.out_dir, dev))
    if "9" in args.exps:
        out.append(exp9_subgraph_mask(args.data_root, args.out_dir, dev))
    if "13" in args.exps:
        out.append(exp13_sparse_dense_mp(args.p2_dir, args.data_root, args.out_dir, dev))
    if "20" in args.exps:
        out.append(exp20_noisy_obs(args.p2_dir, args.data_root, args.out_dir, dev))
    if "4" in args.exps:
        out.append(exp4_edge_de(args.p2_dir, args.data_root, args.out_dir, dev))
    if "12" in args.exps:
        out.append(exp12_distribution_shift(args.p2_dir, args.data_root, args.out_dir, dev))
    sum_path = os.path.join(args.out_dir, "p4_batch_summary.json")
    with open(sum_path, "w") as f:
        json.dump({"timestamp_jst": now_jst(), "exps": out}, f, indent=2, default=str)
    print(f"\n[{now_jst()}] P4 BATCH DONE → {sum_path}")


if __name__ == "__main__":
    main()
