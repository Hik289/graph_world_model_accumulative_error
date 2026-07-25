"""Stream C: H3 out-of-distribution retest from Amendment §A4.

Experiment specification:
- Train set: chain ∪ tree (use P2 chain checkpoints as conservative proxy)
- Test set: scale_free (OOD)
- Inject 6 positions × δ ∈ {0.1, 0.5, 1.0, 2.0} × H ∈ {1,2,4,8,16,20,32}
- Zero-shot eval from P2 baseline chain checkpoints (per A4 spec)
- 6 baselines × 3 seeds × 6 positions × 4 δ × 7 H = 3,024 rows

Output: results/h3_ood_amplitude_sweep.csv
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines import BASELINE_REGISTRY
from src.graph_generators import generate
from src.utils.seeding import stable_seed
from scripts._runner_utils import now_jst

CEIL = 1e10
N = 50
D = 8
D_a = 4

BASELINES = ["B1_MLP", "B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode", "B6_ErrorAware"]
SEEDS = [1, 2, 3]
POSITIONS = ["random", "leaf", "hub", "bridge", "action", "target"]
DELTAS = [0.1, 0.5, 1.0, 2.0]
HORIZONS = [1, 2, 4, 8, 16, 20, 32]
TRAIN_TOPO = "chain"  # OOD training proxy: chain (low ρ); could also use tree
TEST_TOPO = "scale_free"


def load_chain_checkpoint(baseline: str, seed: int, p2_dir: str, dev: torch.device):
    """Load baseline checkpoint trained on chain (conservative low-ρ training).

    For B6/B2 prefer patched checkpoints if available.
    """
    if baseline in ("B6_ErrorAware", "B2_GCN"):
        patched_ck = os.path.join(p2_dir.replace("p2_baselines", "p2_baselines_patched"),
                                   "checkpoints", TRAIN_TOPO, f"{baseline}_seed{seed}.pt")
        if os.path.exists(patched_ck):
            ck_path = patched_ck
        else:
            ck_path = os.path.join(p2_dir, "checkpoints", TRAIN_TOPO,
                                    f"{baseline}_seed{seed}.pt")
    else:
        ck_path = os.path.join(p2_dir, "checkpoints", TRAIN_TOPO,
                                f"{baseline}_seed{seed}.pt")
    if not os.path.exists(ck_path):
        return None
    cls = BASELINE_REGISTRY[baseline]
    model = cls(N=N, D=D, D_a=D_a) if baseline == "B1_MLP" else cls(D=D, D_a=D_a)
    try:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
    except Exception:
        return None
    # Skip if any NaN weights
    for p in model.parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            return None
    model.eval().to(dev)
    return model


def get_inject_node(g, position: str, rng: np.random.Generator) -> Optional[int]:
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
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--p2_dir", default=os.path.join(REPO_ROOT, "results", "p2_baselines"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "h3_ood_retest"))
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[{now_jst()}] Stream C: H3 OOD retest start; device={dev}")
    print(f"  Train topology: {TRAIN_TOPO} (load P2 / patched checkpoints)")
    print(f"  Test topology: {TEST_TOPO}")
    print(f"  Coverage: {len(BASELINES)} baselines × {len(SEEDS)} seeds × {len(POSITIONS)} positions"
          f" × {len(DELTAS)} deltas × {len(HORIZONS)} horizons = {len(BASELINES)*len(SEEDS)*len(POSITIONS)*len(DELTAS)*len(HORIZONS)} rows")
    t0 = time.time()
    rows = []

    for baseline in BASELINES:
        for seed in SEEDS:
            # Load model trained on chain (OOD train set proxy)
            model = load_chain_checkpoint(baseline, seed, args.p2_dir, dev)
            if model is None:
                print(f"  SKIP {baseline} seed{seed} (no chain checkpoint or NaN)")
                continue
            # Load scale_free test graph (same outer_seed for consistency)
            g_test = generate(TEST_TOPO, N=N, seed=seed, m=2)
            A_norm_t = torch.from_numpy(g_test.A_norm).float().to(dev)
            # Load scale_free test trajectories (for clean baseline rollout)
            rollout_path = os.path.join(args.data_root, "synthetic_rollouts",
                                         f"fe_{TEST_TOPO}_N{N}_seed{seed}_T50.pt")
            if not os.path.exists(rollout_path):
                print(f"  SKIP {baseline} seed{seed} (no scale_free rollout)")
                continue
            payload = torch.load(rollout_path, weights_only=False)
            test_X = payload["test_X"]
            test_a = payload["test_actions"]
            T_traj = test_X.shape[1] - 1

            # Clean rollout (no perturbation)
            X_0 = torch.from_numpy(test_X[:, 0]).float().to(dev)
            a_seq = torch.from_numpy(test_a).float().to(dev)
            with torch.no_grad():
                X_pred_clean = model.rollout_predict(X_0, A_norm_t, a_seq, T=T_traj).cpu().numpy()

            for position in POSITIONS:
                rng = np.random.default_rng(
                    seed=stable_seed(baseline, seed, position, "h3_ood"))
                inj_node = get_inject_node(g_test, position, rng)
                if inj_node is None:
                    continue
                for delta in DELTAS:
                    # Inject perturbation at X_0[inj_node, :] += δ
                    X_0_pert = X_0.clone()
                    X_0_pert[:, inj_node, :] += float(delta)
                    with torch.no_grad():
                        X_pred_pert = model.rollout_predict(
                            X_0_pert, A_norm_t, a_seq, T=T_traj).cpu().numpy()
                    # For each H, compute NodeMSE and AffectedNodes
                    for H in HORIZONS:
                        if H > T_traj:
                            continue
                        per_traj_mse = []
                        per_traj_aff = []
                        for i in range(test_X.shape[0]):
                            diff = X_pred_pert[i, H] - X_pred_clean[i, H]
                            nm = float(np.mean(diff ** 2))
                            nm = min(nm, CEIL) if math.isfinite(nm) else CEIL
                            per_traj_mse.append(nm)
                            diff_norm = np.linalg.norm(diff, axis=-1)
                            per_traj_aff.append(float((diff_norm > 0.1).mean()))
                        rows.append({
                            "baseline": baseline,
                            "seed": int(seed),
                            "train_topo": TRAIN_TOPO,
                            "test_topo": TEST_TOPO,
                            "injection_position": position,
                            "inject_node": int(inj_node),
                            "delta": float(delta),
                            "H": int(H),
                            "NodeMSE@H_mean": float(np.mean(per_traj_mse)),
                            "NodeMSE@H_std": float(np.std(per_traj_mse)),
                            "AffectedNodes@H_mean": float(np.mean(per_traj_aff)),
                            "AffectedNodes@H_std": float(np.std(per_traj_aff)),
                        })
            elapsed = time.time() - t0
            print(f"  {baseline:14s} seed{seed} done. Elapsed: {elapsed:.0f}s, rows: {len(rows)}")

    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = os.path.join(args.out_dir, "h3_ood_amplitude_sweep.csv")
    df.to_csv(out_path, index=False)
    elapsed = time.time() - t0
    print(f"\n[{now_jst()}] Stream C DONE: {len(df)} rows → {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
