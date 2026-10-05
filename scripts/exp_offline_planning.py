from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines import BASELINE_REGISTRY
from src.graph_generators import generate, compute_all
from src.metrics import (
    RolloutPrediction, node_mse, return_error, action_mismatch,
    failure_propagation_depth,
)

JST = timezone(timedelta(hours=9))

BASELINES = ["B1_MLP", "B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode", "B6_ErrorAware"]
TOPOLOGIES = ["chain", "tree", "grid", "small_world", "scale_free", "star", "complete"]
SEEDS = [1, 2, 3]
N_DEFAULT = 50

CEIL = 1e10


def now_jst() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def load_model_and_data(baseline: str, topo: str, seed: int, *,
                        data_root: str, p2_dir: str, device: torch.device,
                        N: int = N_DEFAULT) -> Optional[Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    ck_path = os.path.join(p2_dir, "checkpoints", topo, f"{baseline}_seed{seed}.pt")
    if not os.path.exists(ck_path):
        return None
    rollout_path = os.path.join(data_root, "synthetic_rollouts",
                                f"fe_{topo}_N{N}_seed{seed}_T50.pt")
    payload = torch.load(rollout_path, weights_only=False)
    g = generate(topo, N=N, seed=seed)
    test_X = torch.from_numpy(payload["test_X"]).float()
    test_a = torch.from_numpy(payload["test_actions"]).float()
    cls = BASELINE_REGISTRY[baseline]
    if baseline == "B1_MLP":
        model = cls(N=N)
    else:
        model = cls()
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    A_norm_t = torch.from_numpy(g.A_norm).float().to(device)
    return (model, A_norm_t, g.A_norm, g.A_dense,
            test_X.numpy(), test_a.numpy(), payload.get("W"))


def exp7_multi_hop(data_root: str, p2_dir: str, out_dir: str, device: torch.device) -> Dict[str, Any]:


    print(f"[Exp 7] multi-hop dependency error 启动")
    t0 = time.time()
    rows = []
    inject_depths = [1, 2, 4, 6, 8, 10]
    H_eval = 20
    for baseline in BASELINES:
        for topo in ["chain", "scale_free"]:
            for seed in SEEDS:
                loaded = load_model_and_data(baseline, topo, seed,
                                              data_root=data_root, p2_dir=p2_dir,
                                              device=device)
                if loaded is None:
                    continue
                model, A_norm_t, A_norm, A_dense, test_X, test_a, _ = loaded

                for d in inject_depths:

                    if d >= N_DEFAULT:
                        continue
                    per_traj_mse = []
                    per_traj_fpd = []
                    for i in range(test_X.shape[0]):
                        X_0 = torch.from_numpy(test_X[i, 0]).float().unsqueeze(0).to(device)

                        X_0_pert = X_0.clone()
                        X_0_pert[0, d, :4] += 1.0
                        a_seq = torch.from_numpy(test_a[i:i+1]).float().to(device)
                        T_traj = test_X.shape[1] - 1
                        with torch.no_grad():
                            X_pred_clean = model.rollout_predict(X_0, A_norm_t, a_seq, T=T_traj)
                            X_pred_pert = model.rollout_predict(X_0_pert, A_norm_t, a_seq, T=T_traj)
                        X_pred_clean = X_pred_clean[0].cpu().numpy()
                        X_pred_pert = X_pred_pert[0].cpu().numpy()

                        H = min(H_eval, T_traj)
                        out_idx = N_DEFAULT - 1
                        nm = float(np.mean((X_pred_pert[H, out_idx] - X_pred_clean[H, out_idx]) ** 2))
                        per_traj_mse.append(nm)

                        diff_norm = np.linalg.norm(X_pred_pert[H] - X_pred_clean[H], axis=-1)
                        err_flags = (diff_norm > 0.1).astype(np.float32)
                        fpd = failure_propagation_depth(A_dense, d, err_flags)
                        per_traj_fpd.append(fpd)
                    rows.append({
                        "baseline": baseline, "topology": topo, "seed": seed,
                        "inject_depth": d, "H_eval": H_eval,
                        "NodeMSE@H_output_node_mean": float(np.mean(per_traj_mse)),
                        "FPD_mean": float(np.mean(per_traj_fpd)),
                        "FPD_median": float(np.median(per_traj_fpd)),
                    })
    df_path = os.path.join(out_dir, "exp7_multi_hop.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(df_path, index=False)
    print(f"  Exp 7: {len(rows)} rows → {df_path} ({time.time()-t0:.1f}s)")
    return {"exp": 7, "n_rows": len(rows), "out": df_path}


def exp8_planning_regret(data_root: str, p2_dir: str, out_dir: str, device: torch.device) -> Dict[str, Any]:


    print(f"[Exp 8] planning regret 启动")
    t0 = time.time()
    rows = []
    horizons = [1, 2, 4, 8, 16, 32]
    n_actions = 4
    for baseline in BASELINES:
        for topo in TOPOLOGIES:
            for seed in SEEDS:
                loaded = load_model_and_data(baseline, topo, seed,
                                              data_root=data_root, p2_dir=p2_dir,
                                              device=device)
                if loaded is None:
                    continue
                model, A_norm_t, A_norm, A_dense, test_X, test_a, _ = loaded
                T_traj = test_X.shape[1] - 1

                for H in horizons:
                    if H > T_traj:
                        continue
                    per_nm = []
                    per_re = []
                    per_action_mm = []
                    per_regret = []
                    for i in range(test_X.shape[0]):
                        X_true_traj = test_X[i]
                        a_true_seq = test_a[i]
                        X_0 = torch.from_numpy(X_true_traj[0]).float().unsqueeze(0).to(device)
                        a_t = torch.from_numpy(a_true_seq).float().unsqueeze(0).to(device)
                        with torch.no_grad():
                            X_pred_traj = model.rollout_predict(X_0, A_norm_t, a_t, T=T_traj)[0].cpu().numpy()

                        nm = float(np.mean((X_pred_traj[H] - X_true_traj[H]) ** 2))

                        r_true = np.linalg.norm(X_true_traj[1:H+1].reshape(H, -1), axis=1)
                        r_pred = np.linalg.norm(X_pred_traj[1:H+1].reshape(H, -1), axis=1)
                        gamma = 0.95
                        disc = np.array([gamma ** k for k in range(H)])
                        re = float(abs(float((disc * (r_pred - r_true)).sum())))

                        a_true_id = np.argmax(a_true_seq[:H], axis=-1).astype(np.int64)


                        a_pred_id_list = []
                        for t in range(H):
                            x_avg = X_pred_traj[t+1, :, :n_actions].mean(0)
                            a_pred_id_list.append(int(np.argmax(x_avg)))
                        a_pred_id = np.array(a_pred_id_list, dtype=np.int64)
                        am = float(np.mean(a_true_id != a_pred_id))


                        regret = re * (1.0 + am)
                        per_nm.append(nm)
                        per_re.append(re)
                        per_action_mm.append(am)
                        per_regret.append(regret)
                    rows.append({
                        "baseline": baseline, "topology": topo, "seed": seed,
                        "horizon": H,
                        "NodeMSE@H_mean": float(np.mean(per_nm)),
                        "ReturnError@H_mean": float(np.mean(per_re)),
                        "ActionMismatch@H_mean": float(np.mean(per_action_mm)),
                        "Regret@H_mean": float(np.mean(per_regret)),
                    })
    df_path = os.path.join(out_dir, "exp8_planning.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(df_path, index=False)
    print(f"  Exp 8: {len(rows)} rows → {df_path} ({time.time()-t0:.1f}s)")
    return {"exp": 8, "n_rows": len(rows), "out": df_path}


def exp17_rollout_length(data_root: str, p2_dir: str, out_dir: str, device: torch.device) -> Dict[str, Any]:


    print(f"[Exp 17] rollout length 启动")
    t0 = time.time()
    rows = []
    horizons = [1, 2, 4, 8, 16, 32, 48]
    for baseline in BASELINES:
        for topo in TOPOLOGIES:
            for seed in SEEDS:
                loaded = load_model_and_data(baseline, topo, seed,
                                              data_root=data_root, p2_dir=p2_dir,
                                              device=device)
                if loaded is None:
                    continue
                model, A_norm_t, A_norm, A_dense, test_X, test_a, _ = loaded
                T_traj = test_X.shape[1] - 1

                X_0 = torch.from_numpy(test_X[:, 0]).float().to(device)
                a_seq = torch.from_numpy(test_a).float().to(device)
                with torch.no_grad():
                    X_pred = model.rollout_predict(X_0, A_norm_t, a_seq, T=T_traj).cpu().numpy()

                for H in horizons:
                    if H > T_traj:
                        continue
                    per_nm = []
                    per_re = []
                    for i in range(test_X.shape[0]):
                        nm = float(np.mean((X_pred[i, H] - test_X[i, H]) ** 2))
                        r_true = np.linalg.norm(test_X[i, 1:H+1].reshape(H, -1), axis=1)
                        r_pred = np.linalg.norm(X_pred[i, 1:H+1].reshape(H, -1), axis=1)
                        gamma = 0.95
                        disc = np.array([gamma ** k for k in range(H)])
                        re = float(abs(float((disc * (r_pred - r_true)).sum())))
                        per_nm.append(nm)
                        per_re.append(re)
                    rows.append({
                        "baseline": baseline, "topology": topo, "seed": seed,
                        "horizon": H,
                        "NodeMSE@H_mean": float(np.mean(per_nm)),
                        "NodeMSE@H_capped": float(min(np.mean(per_nm), CEIL)) if np.isfinite(np.mean(per_nm)) else CEIL,
                        "ReturnError@H_mean": float(np.mean(per_re)),
                    })
    df_path = os.path.join(out_dir, "exp17_rollout_length.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(df_path, index=False)
    print(f"  Exp 17: {len(rows)} rows → {df_path} ({time.time()-t0:.1f}s)")
    return {"exp": 17, "n_rows": len(rows), "out": df_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, 'data'))
    parser.add_argument("--p2_dir", default=os.path.join(REPO_ROOT, 'results', 'p2_baselines'))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, 'results'))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only", nargs="+", default=["7", "8", "17"])
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    print(f"[{now_jst()}] Offline exp 7/8/17 启动; device={dev}")

    out = []
    if "7" in args.only:
        out.append(exp7_multi_hop(args.data_root, args.p2_dir, args.out_dir, dev))
    if "8" in args.only:
        out.append(exp8_planning_regret(args.data_root, args.p2_dir, args.out_dir, dev))
    if "17" in args.only:
        out.append(exp17_rollout_length(args.data_root, args.p2_dir, args.out_dir, dev))

    sum_path = os.path.join(args.out_dir, "exp_offline_planning_summary.json")
    with open(sum_path, "w") as f:
        json.dump({"timestamp_jst": now_jst(), "exps": out}, f, indent=2)
    print(f"\n[{now_jst()}] DONE → {sum_path}")


if __name__ == "__main__":
    main()
