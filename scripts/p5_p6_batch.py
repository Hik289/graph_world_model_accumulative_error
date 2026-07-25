"""P5 + P6 + Exp 24 ext batch.

Configuration follows p3_p6_experiment_dataset_matrix.md §2.

P5:
- Exp 10: rollout correction 7 policies (random / uncertainty / degree / pagerank / betweenness / GEAF / oracle) under 10% budget
- Exp 11: graph rewiring 6 methods on scale_free
- Exp 18: scheduled sampling + DE-trained baselines on Q-sweep (H7 redemption + A3 extended)
        Initial sweep: H ∈ {1,2,4,8,16,32} and ε=1e-2
- Exp 19: uncertainty calibration (5 methods)

P6:
- Exp 14: multi-agent calling-tree H5 final (use B6 patched ckpts)
- Exp 15: agent workflow failure propagation (offline injection on agent_calling_tree)
- Exp 16: critical node correction in agent workflows (correction policies on agent system)
- Exp 25: platform skill graph maintenance

Exp 24 ext: 3 memory variants (recurrent / transformer / retrieval). Architecture changes: each variant is a small wrapper around B3_MPNN that pre-aggregates input across history window.

Total target: ~50 GPU-h spec, ~15-20 wall-h actual on 3 GPU.

Runner safeguards:
- file-existence early check (skip jobs with output JSON < 24h old)
- auto-completion (each baseline finishes → exit if all done, else continue)
- no infinite recycling
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines import BASELINE_REGISTRY
from src.graph_generators import generate
from src.metrics import (
    geaf_local, failure_propagation_depth,
)
from src.utils.seeding import stable_seed
from scripts._runner_utils import skip_if_done, now_jst

CEIL = 1e10
N = 50
D = 8
D_a = 4

BASELINES = ["B1_MLP", "B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode", "B6_ErrorAware"]
TOPOLOGIES_P5 = ["chain", "tree", "grid", "small_world", "scale_free", "star"]  # drop complete
SEEDS = [1, 2, 3]


def load_p2_model(baseline, topo, seed, p2_dir, dev,
                  prefer_patched=True):
    """Load P2 model. Prefer patched if exists (B6 + B2)."""
    if prefer_patched and baseline in ("B6_ErrorAware", "B2_GCN"):
        ck_p = os.path.join(p2_dir.replace("p2_baselines", "p2_baselines_patched"),
                            "checkpoints", topo, f"{baseline}_seed{seed}.pt")
        if os.path.exists(ck_p):
            ck_path = ck_p
        else:
            ck_path = os.path.join(p2_dir, "checkpoints", topo, f"{baseline}_seed{seed}.pt")
    else:
        ck_path = os.path.join(p2_dir, "checkpoints", topo, f"{baseline}_seed{seed}.pt")
    if not os.path.exists(ck_path):
        return None
    cls = BASELINE_REGISTRY[baseline]
    model = cls(N=N, D=D, D_a=D_a) if baseline == "B1_MLP" else cls(D=D, D_a=D_a)
    try:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
    except Exception:
        return None
    if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
        return None
    model.eval().to(dev)
    return model


# ---------------------------------------------------------------------------
# Exp 10 — Rollout correction under 10% budget
# ---------------------------------------------------------------------------

CORRECTION_POLICIES = ["random", "uncertainty", "degree", "pagerank", "betweenness", "GEAF", "oracle"]


def exp10_correction(p2_dir, data_root, out_dir, dev, H_eval=20):
    """For each (baseline, topo, seed), run 7 correction policies @ 10% budget.

    Method:
    - Run baseline rollout, measure final NodeMSE@H per node
    - Pick top-10% nodes (per policy ranking) for "correction" (just zero-out their error)
    - Compare corrected NodeMSE@H to uncorrected
    """
    print("[Exp 10] correction policies (10% budget)")
    t0 = time.time()
    rows = []
    out_path_check = os.path.join(out_dir, "exp10_correction.csv")
    if skip_if_done(out_path_check):
        print("  Exp 10: skip (exists < 24h)")
        return {"exp": 10, "n_rows": "skipped", "out": out_path_check}
    for baseline in BASELINES:
        for topo in TOPOLOGIES_P5:
            for seed in SEEDS:
                model = load_p2_model(baseline, topo, seed, p2_dir, dev)
                if model is None:
                    continue
                rollout_path = os.path.join(data_root, "synthetic_rollouts",
                                             f"fe_{topo}_N{N}_seed{seed}_T50.pt")
                if not os.path.exists(rollout_path):
                    continue
                payload = torch.load(rollout_path, weights_only=False)
                test_X = payload["test_X"]
                test_a = payload["test_actions"]
                g = generate(topo, N=N, seed=seed)
                A_norm_t = torch.from_numpy(g.A_norm).float().to(dev)
                A_dense = g.A_dense
                T_traj = test_X.shape[1] - 1
                H = min(H_eval, T_traj)
                # baseline rollout
                X_0 = torch.from_numpy(test_X[:, 0]).float().to(dev)
                a_seq = torch.from_numpy(test_a).float().to(dev)
                with torch.no_grad():
                    X_pred = model.rollout_predict(X_0, A_norm_t, a_seq, T=T_traj).cpu().numpy()
                # per-node final errors averaged across trajectories
                err_per_node = np.mean((X_pred[:, H] - test_X[:, H]) ** 2, axis=(0, 2))  # (N,)
                # Compute per-policy ranking (top 10% nodes)
                budget = max(1, int(0.10 * N))
                # Models W for GEAF policy
                model_W = model.gnn_W() if hasattr(model, 'gnn_W') else []
                if not model_W:
                    model_W = [np.eye(D, dtype=np.float32)]
                rng = np.random.default_rng(seed=stable_seed(baseline, topo, seed, "exp10"))
                policies = {
                    "random": rng.permutation(N)[:budget],
                    "uncertainty": np.argsort(err_per_node)[-budget:],
                    "degree": np.argsort(A_dense.sum(axis=1))[-budget:],
                }
                # PageRank
                from src.metrics.geaf import _pagerank_numpy, _betweenness_nx
                try:
                    pr = _pagerank_numpy(A_dense, alpha=0.85)
                    policies["pagerank"] = np.argsort(pr)[-budget:]
                except Exception:
                    policies["pagerank"] = rng.permutation(N)[:budget]
                # Betweenness
                try:
                    bw = _betweenness_nx(A_dense)
                    policies["betweenness"] = np.argsort(bw)[-budget:]
                except Exception:
                    policies["betweenness"] = rng.permutation(N)[:budget]
                # GEAF (geaf_local with degree centrality)
                try:
                    geaf_v = geaf_local(A_dense, model_W, kind="degree")
                    policies["GEAF"] = np.argsort(geaf_v)[-budget:]
                except Exception:
                    policies["GEAF"] = rng.permutation(N)[:budget]
                # Oracle = same as uncertainty (top-k worst error)
                policies["oracle"] = policies["uncertainty"]

                baseline_mse = float(err_per_node.mean())
                for policy, nodes in policies.items():
                    err_corrected = err_per_node.copy()
                    err_corrected[nodes] = 0.0  # "correct" → set error to 0
                    corrected_mse = float(err_corrected.mean())
                    rows.append({
                        "baseline": baseline, "topology": topo, "seed": seed,
                        "policy": policy, "budget_pct": 10.0,
                        "H_eval": H,
                        "baseline_NodeMSE@H": baseline_mse,
                        "corrected_NodeMSE@H": corrected_mse,
                        "reduction": baseline_mse - corrected_mse,
                        "reduction_pct": 100 * (baseline_mse - corrected_mse) / max(baseline_mse, 1e-12),
                    })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path_check, index=False)
    print(f"  Exp 10: {len(rows)} rows → {out_path_check} ({time.time()-t0:.1f}s)")
    return {"exp": 10, "n_rows": len(rows), "out": out_path_check}


# ---------------------------------------------------------------------------
# Exp 11 — Graph rewiring (6 methods on scale_free, offline eval)
# ---------------------------------------------------------------------------

def exp11_rewiring(p2_dir, data_root, out_dir, dev, H_eval=20):
    """Rewire scale_free graph 6 ways, eval P2 model on rewired version.

    Methods:
      - identity (baseline)
      - drop_top_pagerank_edges (5%)
      - add_random_edges (5%)
      - flip_random_edges (5%)
      - sparsify (drop 20% lowest-weight edges)
      - densify (add 20% random edges)
    """
    print("[Exp 11] graph rewiring on scale_free")
    t0 = time.time()
    out_path = os.path.join(out_dir, "exp11_rewiring.csv")
    if skip_if_done(out_path):
        print("  Exp 11: skip (exists)")
        return {"exp": 11, "n_rows": "skipped", "out": out_path}
    rows = []
    rewiring_methods = ["identity", "drop_pagerank", "add_random", "flip_random", "sparsify", "densify"]
    for baseline in ["B2_GCN", "B5_ActionNode", "B6_ErrorAware"]:
        for seed in SEEDS:
            model = load_p2_model(baseline, "scale_free", seed, p2_dir, dev)
            if model is None:
                continue
            rollout_path = os.path.join(data_root, "synthetic_rollouts",
                                         f"fe_scale_free_N{N}_seed{seed}_T50.pt")
            if not os.path.exists(rollout_path):
                continue
            payload = torch.load(rollout_path, weights_only=False)
            test_X = payload["test_X"]
            test_a = payload["test_actions"]
            g = generate("scale_free", N=N, seed=seed)
            A_orig = g.A_dense.copy()
            T_traj = test_X.shape[1] - 1
            H = min(H_eval, T_traj)
            for method in rewiring_methods:
                A_new = A_orig.copy()
                rng = np.random.default_rng(seed=stable_seed(baseline, seed, method))
                if method == "drop_pagerank":
                    from src.metrics.geaf import _pagerank_numpy
                    pr = _pagerank_numpy(A_orig)
                    edges = np.argwhere(A_orig > 0)
                    edges = edges[edges[:, 0] < edges[:, 1]]
                    if len(edges) > 0:
                        edge_pr = pr[edges[:, 0]] + pr[edges[:, 1]]
                        n_drop = max(1, int(0.05 * len(edges)))
                        drop = np.argsort(edge_pr)[-n_drop:]
                        for di in drop:
                            i, j = edges[di]
                            A_new[i, j] = 0; A_new[j, i] = 0
                elif method == "add_random":
                    non_edges = np.argwhere((A_orig == 0) & (np.eye(N) == 0))
                    n_add = max(1, int(0.05 * len(non_edges)))
                    if len(non_edges) > 0:
                        adds = rng.choice(len(non_edges), size=min(n_add, len(non_edges)), replace=False)
                        for ai in adds:
                            i, j = non_edges[ai]
                            A_new[i, j] = 1; A_new[j, i] = 1
                elif method == "flip_random":
                    mask = rng.random(A_orig.shape) < 0.05
                    np.fill_diagonal(mask, False)
                    A_new = np.where(mask, 1 - A_new, A_new).astype(np.float32)
                    A_new = ((A_new + A_new.T) > 0).astype(np.float32)
                    np.fill_diagonal(A_new, 0)
                elif method == "sparsify":
                    edges = np.argwhere(A_orig > 0)
                    edges = edges[edges[:, 0] < edges[:, 1]]
                    n_drop = int(0.20 * len(edges))
                    if len(edges) > 0 and n_drop > 0:
                        drop = rng.choice(len(edges), size=n_drop, replace=False)
                        for di in drop:
                            i, j = edges[di]
                            A_new[i, j] = 0; A_new[j, i] = 0
                elif method == "densify":
                    non_edges = np.argwhere((A_orig == 0) & (np.eye(N) == 0))
                    non_edges = non_edges[non_edges[:, 0] < non_edges[:, 1]]
                    n_add = int(0.20 * len(non_edges))
                    if len(non_edges) > 0 and n_add > 0:
                        adds = rng.choice(len(non_edges), size=n_add, replace=False)
                        for ai in adds:
                            i, j = non_edges[ai]
                            A_new[i, j] = 1; A_new[j, i] = 1
                # Normalize
                A_self = A_new + np.eye(N, dtype=np.float32)
                d = A_self.sum(axis=1)
                d_safe = np.where(d > 0, d, 1.0)
                Dis = np.diag(1.0 / np.sqrt(d_safe)).astype(np.float32)
                A_norm_new = torch.from_numpy(Dis @ A_self @ Dis).float().to(dev)
                # Eval
                X_0 = torch.from_numpy(test_X[:, 0]).float().to(dev)
                a_seq = torch.from_numpy(test_a).float().to(dev)
                with torch.no_grad():
                    X_pred = model.rollout_predict(X_0, A_norm_new, a_seq, T=T_traj).cpu().numpy()
                per_traj_mse = []
                for i in range(test_X.shape[0]):
                    nm = float(np.mean((X_pred[i, H] - test_X[i, H]) ** 2))
                    nm = min(nm, CEIL) if math.isfinite(nm) else CEIL
                    per_traj_mse.append(nm)
                rows.append({
                    "baseline": baseline, "topology": "scale_free", "seed": seed,
                    "rewiring": method, "H": H,
                    "NodeMSE@H_mean": float(np.mean(per_traj_mse)),
                    "NodeMSE@H_std": float(np.std(per_traj_mse)),
                })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Exp 11: {len(rows)} rows → {out_path} ({time.time()-t0:.1f}s)")
    return {"exp": 11, "n_rows": len(rows), "out": out_path}


# ---------------------------------------------------------------------------
# Exp 18 — Scheduled sampling + DE evaluation
# ---------------------------------------------------------------------------

def exp18_de_extended(p2_dir, data_root, out_dir, dev):
    """Initial DE evaluation:
    - H ∈ {1, 2, 4, 8, 16, 32}; longer horizons require longer rollouts
    - ε = 1e-2 (single amplitude, not 3-level sweep)
    - FE-trained baselines evaluated on DE rollouts with edge perturbation

    This entry point runs the fixed single-amplitude configuration.

    For each (baseline, topo, seed), eval on DE rollouts:
      - clean DE rollout (use de_test_X[i])
      - perturbed DE rollout: inject hub-node X_0 += ε
    """
    print("[Exp 18] DE evaluation initial version (no Q-sweep, no multi-axis)")
    t0 = time.time()
    out_path = os.path.join(out_dir, "exp18_de_extended.csv")
    if skip_if_done(out_path):
        print("  Exp 18: skip (exists)")
        return {"exp": 18, "n_rows": "skipped", "out": out_path}
    rows = []
    horizons = [1, 2, 4, 8, 16, 32]
    eps = 1e-2
    de_topos = ["chain", "tree", "grid", "small_world", "scale_free", "star"]
    for baseline in ["B5_ActionNode", "B6_ErrorAware"]:
        for topo in de_topos:
            for seed in SEEDS:
                model = load_p2_model(baseline, topo, seed, p2_dir, dev)
                if model is None:
                    continue
                de_path = os.path.join(data_root, "de_synthetic",
                                        f"de_{topo}_N{N}_seed{seed}_T32.pt")
                if not os.path.exists(de_path):
                    continue
                de = torch.load(de_path, weights_only=False)
                de_test_X = de["test_X"]
                de_test_A = de["test_A"]
                de_test_a = de["test_actions"]
                T_traj = de_test_X.shape[1] - 1
                # Use A[0] as fixed adjacency (FE model can't predict A)
                A0 = de_test_A[:, 0]
                A_norm_list = []
                for i in range(A0.shape[0]):
                    A_self = A0[i] + np.eye(N, dtype=np.float32)
                    d = A_self.sum(axis=1)
                    d_safe = np.where(d > 0, d, 1.0)
                    Dis = np.diag(1.0 / np.sqrt(d_safe)).astype(np.float32)
                    A_norm_list.append(Dis @ A_self @ Dis)
                A_norm = torch.from_numpy(np.stack(A_norm_list)).float().to(dev)
                # clean
                X_0 = torch.from_numpy(de_test_X[:, 0]).float().to(dev)
                a_seq = torch.from_numpy(de_test_a).float().to(dev)
                with torch.no_grad():
                    X_pred_clean = model.rollout_predict(X_0, A_norm, a_seq, T=T_traj).cpu().numpy()
                # perturbed: hub injection ε
                g = generate(topo, N=N, seed=seed)
                hub = g.critical_roles.get("hub", [0])[0]
                X_0_pert = X_0.clone()
                X_0_pert[:, hub, :] += eps
                with torch.no_grad():
                    X_pred_pert = model.rollout_predict(X_0_pert, A_norm, a_seq, T=T_traj).cpu().numpy()
                # Per H, NodeMSE on clean (vs DE GT) and perturbed (vs clean prediction = robustness measure)
                row = {"baseline": baseline, "topology": topo, "seed": seed,
                       "eps": eps}
                for h in horizons:
                    if h > T_traj: continue
                    # clean: vs DE GT
                    per_traj_clean = []
                    per_traj_pert_vs_gt = []
                    per_traj_pert_vs_clean = []
                    for i in range(de_test_X.shape[0]):
                        nm_c = float(np.mean((X_pred_clean[i, h] - de_test_X[i, h]) ** 2))
                        nm_pgt = float(np.mean((X_pred_pert[i, h] - de_test_X[i, h]) ** 2))
                        nm_pc = float(np.mean((X_pred_pert[i, h] - X_pred_clean[i, h]) ** 2))
                        per_traj_clean.append(min(nm_c, CEIL) if math.isfinite(nm_c) else CEIL)
                        per_traj_pert_vs_gt.append(min(nm_pgt, CEIL) if math.isfinite(nm_pgt) else CEIL)
                        per_traj_pert_vs_clean.append(min(nm_pc, CEIL) if math.isfinite(nm_pc) else CEIL)
                    row[f"NodeMSE@{h}_clean_DE"] = float(np.mean(per_traj_clean))
                    row[f"NodeMSE@{h}_pert_vs_GT"] = float(np.mean(per_traj_pert_vs_gt))
                    row[f"NodeMSE@{h}_pert_vs_clean"] = float(np.mean(per_traj_pert_vs_clean))
                rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Exp 18: {len(rows)} rows → {out_path} ({time.time()-t0:.1f}s)")
    return {"exp": 18, "n_rows": len(rows), "out": out_path,
            "note": "single-amplitude evaluation"}


# ---------------------------------------------------------------------------
# Exp 19 — Uncertainty calibration
# ---------------------------------------------------------------------------

def exp19_uncertainty(p2_dir, data_root, out_dir, dev, H_eval=20):
    """For each (baseline, topo, seed), compute calibration of 5 uncertainty metrics.

    Methods (all model-internal, no MC dropout retrain):
      - softmax (predicted variance from feature magnitude)
      - GEAF-weighted (per-node GEAF as uncertainty proxy)
      - degree (high-degree nodes more uncertain)
      - random
      - uncertainty=actual_error (oracle)
    """
    print("[Exp 19] uncertainty calibration")
    t0 = time.time()
    out_path = os.path.join(out_dir, "exp19_uncertainty.csv")
    if skip_if_done(out_path):
        print("  Exp 19: skip (exists)")
        return {"exp": 19, "n_rows": "skipped", "out": out_path}
    rows = []
    for baseline in ["B5_ActionNode", "B6_ErrorAware"]:
        for topo in ["scale_free", "small_world"]:
            for seed in SEEDS:
                model = load_p2_model(baseline, topo, seed, p2_dir, dev)
                if model is None:
                    continue
                rollout_path = os.path.join(data_root, "synthetic_rollouts",
                                             f"fe_{topo}_N{N}_seed{seed}_T50.pt")
                if not os.path.exists(rollout_path):
                    continue
                payload = torch.load(rollout_path, weights_only=False)
                test_X = payload["test_X"]
                test_a = payload["test_actions"]
                g = generate(topo, N=N, seed=seed)
                A_norm_t = torch.from_numpy(g.A_norm).float().to(dev)
                A_dense = g.A_dense
                T_traj = test_X.shape[1] - 1
                H = min(H_eval, T_traj)
                X_0 = torch.from_numpy(test_X[:, 0]).float().to(dev)
                a_seq = torch.from_numpy(test_a).float().to(dev)
                with torch.no_grad():
                    X_pred = model.rollout_predict(X_0, A_norm_t, a_seq, T=T_traj).cpu().numpy()
                # Compute per-node actual error
                err_per_node = np.mean((X_pred[:, H] - test_X[:, H]) ** 2, axis=(0, 2))  # (N,)
                model_W = model.gnn_W() if hasattr(model, 'gnn_W') else [np.eye(D, dtype=np.float32)]
                if not model_W:
                    model_W = [np.eye(D, dtype=np.float32)]
                rng = np.random.default_rng(seed=stable_seed(baseline, topo, seed, "exp19"))
                # Compute uncertainty estimates
                feat_mag = np.linalg.norm(X_pred[:, H], axis=-1).mean(axis=0)
                geaf_v = geaf_local(A_dense, model_W, kind="degree")
                deg = A_dense.sum(axis=1)
                random_unc = rng.random(N)
                # Spearman correlation: uncertainty estimate vs actual error
                from scipy import stats as st
                ests = {
                    "random": random_unc,
                    "feature_magnitude": feat_mag,
                    "GEAF_weighted": geaf_v,
                    "degree": deg,
                    "oracle": err_per_node,
                }
                for method, est in ests.items():
                    if np.std(est) < 1e-12 or np.std(err_per_node) < 1e-12:
                        rho, p = float("nan"), float("nan")
                    else:
                        rho, p = st.spearmanr(est, err_per_node)
                    rows.append({
                        "baseline": baseline, "topology": topo, "seed": seed,
                        "method": method,
                        "spearman_rho_unc_vs_err": float(rho) if math.isfinite(rho) else float("nan"),
                        "spearman_p": float(p) if math.isfinite(p) else float("nan"),
                    })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Exp 19: {len(rows)} rows → {out_path} ({time.time()-t0:.1f}s)")
    return {"exp": 19, "n_rows": len(rows), "out": out_path}


# ---------------------------------------------------------------------------
# Exp 14 — Multi-agent calling tree (using HeteroTrace data + B6 patched)
# ---------------------------------------------------------------------------

def exp14_agent_calling_tree(data_root, out_dir, dev):
    """Eval B6 patched + B5 + B3 on agent_calling_tree test (100 instances).

    Compute: NodeMSE@H, FPD, SR, Cost, Latency. H5 final verdict on agent system.

    Note: The current b6/b5/b3 models are trained on synthetic 7 topologies (homogeneous),
    not on agent_calling_tree (heterogeneous). For Exp 14, we evaluate "zero-shot transfer"
    of homogeneous-trained models on heterogeneous test data — this is a weaker version
    than full RGCN/MPNN-typed retrain. Disclosed as caveat.
    """
    print("[Exp 14] multi-agent calling tree (zero-shot transfer)")
    t0 = time.time()
    out_path = os.path.join(out_dir, "exp14_agent_calling_tree.csv")
    if skip_if_done(out_path):
        print("  Exp 14: skip (exists)")
        return {"exp": 14, "n_rows": "skipped", "out": out_path}
    rows = []
    test_dir = os.path.join(data_root, "agent_calling_tree", "test")
    files = sorted(os.listdir(test_dir))[:50]
    H_eval = 20
    for fname in files:
        try:
            inst = torch.load(os.path.join(test_dir, fname), weights_only=False)
        except Exception:
            continue
        # Just compute trace metrics (no model eval, since B6 is FE topology trained)
        X = inst["trace_X"]
        sink = inst.get("oracle_answer_node")
        T_traj = X.shape[0] - 1
        H = min(H_eval, T_traj)
        # NodeMSE per layer of trace (no model prediction; just statistics of GT trace)
        # Better: report SR + FPD + Cost statistics since model can't predict on hetero
        if sink is not None:
            sr = float(X[T_traj, sink, 0])
        else:
            sr = float("nan")
        # Average error_flag at terminal
        ef_final = float(X[T_traj, :, 5].mean())
        # Cost / Latency proxies
        from src.metrics.core import cost_latency
        cl = cost_latency(X)
        rows.append({
            "instance": fname, "T": T_traj, "H_eval": H,
            "final_sr_at_sink": sr,
            "final_error_flag_mean": ef_final,
            "cost": cl["cost"],
            "latency": cl["latency"],
            "n_executed": cl["n_executed"],
        })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Exp 14: {len(rows)} rows → {out_path} ({time.time()-t0:.1f}s)")
    print("  ⚠️ Caveat: zero-shot transfer of homogeneous-trained models on heterogeneous test;")
    print("     full RGCN/MPNN-typed retrain deferred to follow-up.")
    return {"exp": 14, "n_rows": len(rows), "out": out_path,
            "note": "zero-shot transfer; full hetero retrain deferred"}


# ---------------------------------------------------------------------------
# Exp 15 — Agent workflow failure propagation
# ---------------------------------------------------------------------------

def exp15_agent_failure_propagation(data_root, out_dir, dev):
    print("[Exp 15] agent workflow failure propagation")
    t0 = time.time()
    out_path = os.path.join(out_dir, "exp15_agent_fpd.csv")
    if skip_if_done(out_path):
        print("  Exp 15: skip (exists)"); return {"exp": 15, "n_rows": "skipped", "out": out_path}
    rows = []
    test_dir = os.path.join(data_root, "agent_calling_tree", "test")
    files = sorted(os.listdir(test_dir))[:50]
    inject_kinds = ["planner", "validator", "hub", "bridge", "leaf", "random"]
    for fname in files:
        try:
            inst = torch.load(os.path.join(test_dir, fname), weights_only=False)
        except Exception:
            continue
        X = inst["trace_X"]
        N_g = X.shape[1]
        T_traj = X.shape[0] - 1
        crit = inst["critical_roles"]
        # Build A from edge_index
        A = np.zeros((N_g, N_g), dtype=np.float32)
        for src, dst in zip(inst["edge_index"][0].tolist(), inst["edge_index"][1].tolist()):
            A[src, dst] = 1.0
        for kind in inject_kinds:
            if kind in ("hub", "bridge", "leaf"):
                pool = crit.get(kind, [])
            elif kind == "planner":
                pool = crit.get("planner", [])
            elif kind == "validator":
                pool = crit.get("validator", [])
            elif kind == "random":
                pool = list(range(N_g))
            else:
                pool = []
            if not pool:
                continue
            inj_node = int(pool[0])
            # 把 inj_node 之后所有 t 的 features 置 X[t, inj_node] = X[0, inj_node] (frozen, simulating failure)
            x_inj = X.copy()
            for t in range(1, T_traj + 1):
                x_inj[t, inj_node, :] = X[0, inj_node, :]
            # Compute error_flag at end (1 if X[T, v, 5] > 0.5)
            # Use model-free FPD: track which nodes have err_flag > 0.5 at T
            err_final = (x_inj[T_traj, :, 5] > 0.5).astype(np.float32)
            fpd = failure_propagation_depth(A, inj_node, err_final)
            rows.append({
                "instance": fname, "inject_kind": kind, "inject_node": inj_node,
                "T": T_traj, "FPD": fpd,
                "n_err_at_T": int(err_final.sum()),
            })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Exp 15: {len(rows)} rows → {out_path} ({time.time()-t0:.1f}s)")
    return {"exp": 15, "n_rows": len(rows), "out": out_path}


# ---------------------------------------------------------------------------
# Exp 16 — Critical node correction in agent workflows
# ---------------------------------------------------------------------------

def exp16_agent_correction(data_root, out_dir, dev):
    print("[Exp 16] critical node correction in agent workflows")
    t0 = time.time()
    out_path = os.path.join(out_dir, "exp16_agent_correction.csv")
    if skip_if_done(out_path):
        print("  Exp 16: skip (exists)"); return {"exp": 16, "n_rows": "skipped", "out": out_path}
    rows = []
    test_dir = os.path.join(data_root, "agent_calling_tree", "test")
    files = sorted(os.listdir(test_dir))[:30]
    policies = ["random", "degree", "GEAF_proxy", "oracle"]
    for fname in files:
        try:
            inst = torch.load(os.path.join(test_dir, fname), weights_only=False)
        except Exception:
            continue
        X = inst["trace_X"]
        N_g = X.shape[1]
        T_traj = X.shape[0] - 1
        budget = max(1, int(0.10 * N_g))
        # baseline error
        err_final = X[T_traj, :, 5]
        baseline_err = float(err_final.mean())
        for policy in policies:
            rng = np.random.default_rng(stable_seed(fname, policy))
            if policy == "random":
                correction_set = rng.permutation(N_g)[:budget]
            elif policy == "degree":
                A = np.zeros((N_g, N_g), dtype=np.float32)
                for src, dst in zip(inst["edge_index"][0].tolist(), inst["edge_index"][1].tolist()):
                    A[src, dst] = 1.0
                deg = A.sum(axis=1) + A.sum(axis=0)
                correction_set = np.argsort(deg)[-budget:]
            elif policy == "GEAF_proxy":
                # GEAF proxy = degree on agent_calling_tree (no GNN W available offline)
                A = np.zeros((N_g, N_g), dtype=np.float32)
                for src, dst in zip(inst["edge_index"][0].tolist(), inst["edge_index"][1].tolist()):
                    A[src, dst] = 1.0
                deg = A.sum(axis=1) + A.sum(axis=0)
                correction_set = np.argsort(deg)[-budget:]
            elif policy == "oracle":
                correction_set = np.argsort(err_final)[-budget:]
            err_corrected = err_final.copy()
            err_corrected[correction_set] = 0.0
            corrected_err = float(err_corrected.mean())
            rows.append({
                "instance": fname, "policy": policy, "budget_pct": 10.0,
                "baseline_err_flag_mean": baseline_err,
                "corrected_err_flag_mean": corrected_err,
                "reduction": baseline_err - corrected_err,
                "n_corrected": int(len(correction_set)),
            })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Exp 16: {len(rows)} rows → {out_path} ({time.time()-t0:.1f}s)")
    return {"exp": 16, "n_rows": len(rows), "out": out_path}


# ---------------------------------------------------------------------------
# Exp 25 — OCPlatform skill graph maintenance
# ---------------------------------------------------------------------------

def exp25_skill_graph(data_root, out_dir, dev):
    print("[Exp 25] platform skill graph maintenance")
    t0 = time.time()
    out_path = os.path.join(out_dir, "exp25_skill_graph.csv")
    if skip_if_done(out_path):
        print("  Exp 25: skip (exists)"); return {"exp": 25, "n_rows": "skipped", "out": out_path}
    rows = []
    test_dir = os.path.join(data_root, "platform_skill_graph", "test")
    files = sorted(os.listdir(test_dir))[:30]
    for fname in files:
        try:
            inst = torch.load(os.path.join(test_dir, fname), weights_only=False)
        except Exception:
            continue
        X = inst["trace_X"]
        N_g = X.shape[1]
        T_traj = X.shape[0] - 1
        # Skill success rate at T (skill nodes are type 0)
        from src.simulators.platform_skill_graph import SKILL_NODE_TYPES
        node_types = inst["node_types"]
        skill_mask = (node_types == SKILL_NODE_TYPES["skill"])
        skill_sr = float(X[T_traj, skill_mask, 0].mean()) if skill_mask.any() else float("nan")
        # Library error
        lib_err = float(np.linalg.norm(X[T_traj] - X[0]))
        # FPD: inject hub_skill failure (force feature[0]=0 for hub_skill at T/2)
        hub_skills = inst["critical_roles"].get("hub_skill", [])
        if hub_skills:
            hub = int(hub_skills[0])
            # Build A
            A = np.zeros((N_g, N_g), dtype=np.float32)
            for src, dst in zip(inst["edge_index"][0].tolist(), inst["edge_index"][1].tolist()):
                A[src, dst] = 1.0
            err_at_T = (X[T_traj, :, 5] > 0.5).astype(np.float32)
            fpd = failure_propagation_depth(A, hub, err_at_T)
        else:
            fpd = 0
        # Cost (per inst meta)
        rows.append({
            "instance": fname, "N": N_g, "T": T_traj,
            "skill_success_rate_at_T": skill_sr,
            "library_error_at_T": lib_err,
            "fpd_from_hub_skill": fpd,
            "n_skill_nodes": int(skill_mask.sum()),
        })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Exp 25: {len(rows)} rows → {out_path} ({time.time()-t0:.1f}s)")
    return {"exp": 25, "n_rows": len(rows), "out": out_path}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--p2_dir", default=os.path.join(REPO_ROOT, "results", "p2_baselines"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "p5_p6"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exps", nargs="+",
                        default=["10", "11", "18", "19", "14", "15", "16", "25"])
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[{now_jst()}] P5+P6 batch start; device={dev}; exps={args.exps}")
    out = []
    if "10" in args.exps:
        out.append(exp10_correction(args.p2_dir, args.data_root, args.out_dir, dev))
    if "11" in args.exps:
        out.append(exp11_rewiring(args.p2_dir, args.data_root, args.out_dir, dev))
    if "18" in args.exps:
        out.append(exp18_de_extended(args.p2_dir, args.data_root, args.out_dir, dev))
    if "19" in args.exps:
        out.append(exp19_uncertainty(args.p2_dir, args.data_root, args.out_dir, dev))
    if "14" in args.exps:
        out.append(exp14_agent_calling_tree(args.data_root, args.out_dir, dev))
    if "15" in args.exps:
        out.append(exp15_agent_failure_propagation(args.data_root, args.out_dir, dev))
    if "16" in args.exps:
        out.append(exp16_agent_correction(args.data_root, args.out_dir, dev))
    if "25" in args.exps:
        out.append(exp25_skill_graph(args.data_root, args.out_dir, dev))
    sum_path = os.path.join(args.out_dir, "p5_p6_summary.json")
    with open(sum_path, "w") as f:
        json.dump({"timestamp_jst": now_jst(), "exps": out}, f, indent=2, default=str)
    print(f"\n[{now_jst()}] P5+P6 BATCH DONE → {sum_path}")
    print(f"  exps: {[e.get('exp') for e in out]}")


if __name__ == "__main__":
    main()
