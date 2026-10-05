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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines import BASELINE_REGISTRY
from src.graph_generators import generate
from src.simulators import rollout
from scripts._runner_utils import skip_if_done, now_jst

CEIL = 1e10
N = 50
D = 8
D_a = 4

DE_TOPOS = ["chain", "tree", "grid", "small_world", "scale_free", "star"]
SEEDS = [1, 2, 3]
H_LIST = [1, 2, 4, 8, 16, 32, 64, 128]
EPS_LIST = [1e-3, 1e-2, 1e-1]


def gen_de_t128(data_root: str):

    out_dir = os.path.join(data_root, "de_synthetic_t128")
    os.makedirs(out_dir, exist_ok=True)
    generated = 0
    for top in DE_TOPOS:
        for seed in SEEDS:
            fname = f"de_{top}_N{N}_seed{seed}_T128.pt"
            path = os.path.join(out_dir, fname)
            if os.path.exists(path):
                continue
            params = {"chain": {}, "tree": {"variant": "balanced_binary"},
                      "grid": {"shape": "auto"},
                      "small_world": {"k": 4, "p": 0.1},
                      "scale_free": {"m": 2}, "star": {}}.get(top, {})
            g = generate(top, N=N, seed=seed, **params)
            train_X, train_A, train_a = [], [], []
            val_X, val_A, val_a = [], [], []
            test_X, test_A, test_a = [], [], []
            W_shared = U_shared = Q_shared = None
            for split, n_traj, start in [("train", 20, 0), ("val", 5, 100), ("test", 10, 200)]:
                for idx in range(n_traj):
                    inner = start + idx
                    tr = rollout(g, T=128, mode="dynamic_edge",
                                 sigma_noise=0.01, target_q_norm=1.0,
                                 seed=inner)
                    if W_shared is None:
                        W_shared, U_shared, Q_shared = tr.W.copy(), tr.U.copy(), tr.Q.copy()
                    if split == "train":
                        train_X.append(tr.X); train_A.append(tr.A); train_a.append(tr.actions)
                    elif split == "val":
                        val_X.append(tr.X); val_A.append(tr.A); val_a.append(tr.actions)
                    else:
                        test_X.append(tr.X); test_A.append(tr.A); test_a.append(tr.actions)
            payload = {
                "version": 1, "topology": top, "N": N, "D": D,
                "outer_seed": int(seed), "T": 128, "mode": "dynamic_edge",
                "Q_norm": 1.0,
                "W": W_shared, "U": U_shared, "Q": Q_shared,
                "train_X": np.stack(train_X), "train_A": np.stack(train_A),
                "train_actions": np.stack(train_a),
                "val_X": np.stack(val_X), "val_A": np.stack(val_A),
                "val_actions": np.stack(val_a),
                "test_X": np.stack(test_X), "test_A": np.stack(test_A),
                "test_actions": np.stack(test_a),
            }
            torch.save(payload, path)
            generated += 1
    print(f"  generated {generated} T=128 DE files")
    return generated


def load_p2_model(baseline, topo, seed, dev,
                  p2_dir=os.path.join(REPO_ROOT, 'results', 'p2_baselines'),
                  prefer_patched=True):
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
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    if any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters()):
        return None
    model.eval().to(dev)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, 'data'))
    parser.add_argument("--p2_dir", default=os.path.join(REPO_ROOT, 'results', 'p2_baselines'))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, 'results', 'p5_exp18_full'))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_gen", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    print(f"[{now_jst()}] Exp 18 multi-axis start; device={dev}")


    if not args.skip_gen:
        print("[Step 1] generating T=128 DE rollouts...")
        t0 = time.time()
        n_gen = gen_de_t128(args.data_root)
        print(f"  generated {n_gen} files in {time.time()-t0:.1f}s")


    out_path = os.path.join(args.out_dir, "exp18_multi_axis.csv")
    if skip_if_done(out_path):
        print("Exp 18 multi-axis: skip (exists)")
        return
    rows = []
    t0 = time.time()
    for baseline in ["B5_ActionNode", "B6_ErrorAware"]:
        for topo in DE_TOPOS:
            for seed in SEEDS:
                model = load_p2_model(baseline, topo, seed, dev, p2_dir=args.p2_dir)
                if model is None:
                    continue
                de_path = os.path.join(args.data_root, "de_synthetic_t128",
                                        f"de_{topo}_N{N}_seed{seed}_T128.pt")
                if not os.path.exists(de_path):
                    print(f"  missing {de_path}")
                    continue
                de = torch.load(de_path, weights_only=False)
                de_test_X = de["test_X"]
                de_test_A = de["test_A"]
                de_test_a = de["test_actions"]
                T_traj = de_test_X.shape[1] - 1

                A0 = de_test_A[:, 0]
                A_norm_list = []
                for i in range(A0.shape[0]):
                    A_self = A0[i] + np.eye(N, dtype=np.float32)
                    d = A_self.sum(axis=1)
                    d_safe = np.where(d > 0, d, 1.0)
                    Dis = np.diag(1.0 / np.sqrt(d_safe)).astype(np.float32)
                    A_norm_list.append(Dis @ A_self @ Dis)
                A_norm = torch.from_numpy(np.stack(A_norm_list)).float().to(dev)
                X_0 = torch.from_numpy(de_test_X[:, 0]).float().to(dev)
                a_seq = torch.from_numpy(de_test_a).float().to(dev)

                with torch.no_grad():
                    X_pred_clean = model.rollout_predict(X_0, A_norm, a_seq, T=T_traj).cpu().numpy()

                g = generate(topo, N=N, seed=seed,
                             **{"chain": {}, "tree": {"variant": "balanced_binary"},
                                "grid": {"shape": "auto"},
                                "small_world": {"k": 4, "p": 0.1},
                                "scale_free": {"m": 2}, "star": {}}.get(topo, {}))
                hub = g.critical_roles.get("hub", [0])[0]

                for eps in EPS_LIST:
                    X_0_pert = X_0.clone()
                    X_0_pert[:, hub, :] += eps
                    with torch.no_grad():
                        X_pred_pert = model.rollout_predict(X_0_pert, A_norm, a_seq,
                                                             T=T_traj).cpu().numpy()
                    row = {"baseline": baseline, "topology": topo, "seed": seed,
                           "eps": eps, "inject_node": hub}
                    for h in H_LIST:
                        if h > T_traj: continue
                        per_clean = []
                        per_pgt = []
                        per_pc = []
                        for i in range(de_test_X.shape[0]):
                            nm_c = float(np.mean((X_pred_clean[i, h] - de_test_X[i, h]) ** 2))
                            nm_pgt = float(np.mean((X_pred_pert[i, h] - de_test_X[i, h]) ** 2))
                            nm_pc = float(np.mean((X_pred_pert[i, h] - X_pred_clean[i, h]) ** 2))
                            per_clean.append(min(nm_c, CEIL) if math.isfinite(nm_c) else CEIL)
                            per_pgt.append(min(nm_pgt, CEIL) if math.isfinite(nm_pgt) else CEIL)
                            per_pc.append(min(nm_pc, CEIL) if math.isfinite(nm_pc) else CEIL)
                        row[f"NodeMSE@{h}_clean"] = float(np.mean(per_clean))
                        row[f"NodeMSE@{h}_pert_vs_GT"] = float(np.mean(per_pgt))
                        row[f"NodeMSE@{h}_pert_vs_clean"] = float(np.mean(per_pc))
                    rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[{now_jst()}] Exp 18 multi-axis DONE: {len(rows)} rows → {out_path} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
