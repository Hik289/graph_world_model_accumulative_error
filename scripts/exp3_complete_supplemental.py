from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines import BASELINE_REGISTRY
from src.graph_generators import generate
from src.metrics import failure_propagation_depth
from src.utils.seeding import stable_seed

JST = timezone(timedelta(hours=9))
BASELINES = ["B1_MLP", "B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode", "B6_ErrorAware"]
SEEDS = [1, 2, 3]
N = 50
D = 8
D_a = 4
H_EVAL = 20
CEIL = 1e10
INJECTION_POSITIONS = ["random", "leaf", "hub", "bridge", "action", "target"]


def now_jst():
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def get_inject_node(g, position: str, rng: np.random.Generator):
    crit = g.critical_roles
    if position == "random":
        return int(rng.integers(0, g.N))
    elif position == "leaf":
        leaves = crit.get("leaf", [])

        return int(rng.choice(leaves)) if leaves else int(rng.integers(0, g.N))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, 'data'))
    parser.add_argument("--p2_dir", default=os.path.join(REPO_ROOT, 'results', 'p2_baselines'))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, 'results', 'p4'))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(args.device)
    else:
        dev = torch.device("cpu")
    print(f"[{now_jst()}] Exp 3 supplemental (complete graph) start; device={dev}")
    t0 = time.time()
    rows = []
    topo = "complete"

    for baseline in BASELINES:
        for seed in SEEDS:
            ck_path = os.path.join(args.p2_dir, "checkpoints", topo,
                                    f"{baseline}_seed{seed}.pt")
            rollout_path = os.path.join(args.data_root, "synthetic_rollouts",
                                         f"fe_{topo}_N{N}_seed{seed}_T50.pt")
            if not os.path.exists(ck_path) or not os.path.exists(rollout_path):
                print(f"  SKIP {baseline} {topo} seed{seed}: missing ckpt or rollout")
                continue
            try:
                ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"  SKIP {baseline} {topo} seed{seed}: load_error {e}")
                continue
            cls = BASELINE_REGISTRY[baseline]
            model = cls(N=N, D=D, D_a=D_a) if baseline == "B1_MLP" else cls(D=D, D_a=D_a)
            try:
                model.load_state_dict(ck["state_dict"])
            except Exception as e:
                print(f"  SKIP {baseline} {topo} seed{seed}: state_dict_load_error {e}")
                continue
            if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
                print(f"  SKIP {baseline} {topo} seed{seed}: NaN weights")
                continue
            model.eval().to(dev)
            g = generate(topo, N=N, seed=seed)
            A_norm_t = torch.from_numpy(g.A_norm).float().to(dev)
            A_dense = g.A_dense
            payload = torch.load(rollout_path, weights_only=False)
            test_X = payload["test_X"]
            test_a = payload["test_actions"]
            T_traj = test_X.shape[1] - 1
            H = min(H_EVAL, T_traj)

            for pos in INJECTION_POSITIONS:
                rng = np.random.default_rng(
                    seed=stable_seed(baseline, topo, seed, pos))
                inj_node = get_inject_node(g, pos, rng)
                if inj_node is None:
                    continue
                per_traj_mse = []
                per_traj_aff = []
                per_traj_fpd = []
                for i in range(test_X.shape[0]):
                    X_0_clean = torch.from_numpy(test_X[i, 0]).float().unsqueeze(0).to(dev)
                    X_0_pert = X_0_clean.clone()
                    X_0_pert[0, inj_node, :] += 0.5
                    a_seq = torch.from_numpy(test_a[i:i + 1]).float().to(dev)
                    with torch.no_grad():
                        X_pred_clean = model.rollout_predict(
                            X_0_clean, A_norm_t, a_seq, T=T_traj)[0].cpu().numpy()
                        X_pred_pert = model.rollout_predict(
                            X_0_pert, A_norm_t, a_seq, T=T_traj)[0].cpu().numpy()
                    diff = X_pred_pert[H] - X_pred_clean[H]
                    nm = float(np.mean(diff ** 2))
                    nm = min(nm, CEIL) if math.isfinite(nm) else CEIL
                    per_traj_mse.append(nm)
                    diff_norm = np.linalg.norm(diff, axis=-1)
                    per_traj_aff.append(float((diff_norm > 0.1).mean()))

                    err_flags = (diff_norm > 0.1).astype(np.float32)
                    fpd = failure_propagation_depth(A_dense, inj_node, err_flags)
                    per_traj_fpd.append(fpd)
                rows.append({
                    "baseline": baseline, "topology": topo, "seed": seed,
                    "inject_position": pos, "inject_node": int(inj_node),
                    "H_eval": H,
                    "NodeMSE@H_mean": float(np.mean(per_traj_mse)),
                    "AffectedNodes@H_mean": float(np.mean(per_traj_aff)),
                    "FPD_mean": float(np.mean(per_traj_fpd)),
                    "FPD_median": float(np.median(per_traj_fpd)),
                })
                print(f"  {baseline:14s} {topo} seed{seed} {pos:8s}: "
                      f"NodeMSE@H={np.mean(per_traj_mse):.3e} "
                      f"AffNodes={np.mean(per_traj_aff):.3f} "
                      f"FPD={np.median(per_traj_fpd):.1f}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(args.out_dir, "exp3_complete_supplemental.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[{now_jst()}] DONE: {len(rows)} rows → {out_path} ({time.time()-t0:.1f}s)")

    print("\n=== Per-baseline NodeMSE@H median (complete graph, position-wise) ===")
    print(df.groupby(["baseline", "inject_position"])["NodeMSE@H_mean"].median()
          .unstack().to_string())


if __name__ == "__main__":
    main()
