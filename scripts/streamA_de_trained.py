"""Stream A: DE-trained baselines with edge-prediction head.

Per Director 2026-05-14 21:02 UTC: 6 topo (drop complete) × 6 baselines × 3 seeds = 108 jobs.
import os as _os; PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

Loss: L_total = L_X (node MSE) + λ_e * BCE(Â, A_true), default λ_e = 1.0

Each run落盘:
- NodeMSE@H and EdgeF1@H for H ∈ {1,2,4,8,16,32}
- theory_constants with L_g (edge Lipschitz, M_X derived)
- final_train_loss, final_train_loss_node, final_train_loss_edge

Output: results/de_trained/{topo}/{baseline}_seed{S}.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue as queue_mod
import sys
import time
import multiprocessing as mp
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines import BASELINE_REGISTRY
from src.baselines.edge_head import WorldModelWithEdgeHead
from src.graph_generators import generate, compute_all
from src.metrics import RolloutPrediction, node_mse, edge_f1_binary, theory_constants
from src.metrics.geaf import _spectral_radius
from scripts._runner_utils import skip_if_done, now_jst

JST = timezone(timedelta(hours=9))
BASELINES = ["B1_MLP", "B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode", "B6_ErrorAware"]
TOPOLOGIES = ["chain", "tree", "grid", "small_world", "scale_free", "star"]  # drop complete
SEEDS = [1, 2, 3]
N = 50
D = 8
D_a = 4
HORIZONS = [1, 2, 4, 8, 16, 32]
CEIL = 1e10


def build_node_model(baseline: str, N_g: int = 50):
    """Build the node-level world model (without edge head)."""
    cls = BASELINE_REGISTRY[baseline]
    if baseline == "B1_MLP":
        return cls(N=N_g, D=D, D_a=D_a)
    else:
        return cls(D=D, D_a=D_a)


def train_one_de(
    baseline: str, topo: str, seed: int, *,
    data_root: str, out_dir: str,
    device: torch.device,
    epochs: int = 80, batch_size: int = 8,
    lr: float = 1e-3,
    lam_edge: float = 1.0,
    save_ckpt: bool = True,
) -> Dict[str, Any]:
    """Train baseline + edge head on DE rollout data."""
    t0 = time.time()
    # Seeding
    py_seed = seed * 1000 + 7
    np.random.seed(py_seed)
    torch.manual_seed(py_seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(py_seed + 2)

    # Load DE rollouts (default Q_norm=1.0)
    de_path = os.path.join(data_root, "de_synthetic",
                            f"de_{topo}_N{N}_seed{seed}_T32.pt")
    if not os.path.exists(de_path):
        return {"skipped": True, "reason": f"DE file missing: {de_path}"}
    payload = torch.load(de_path, weights_only=False)
    train_X = payload["train_X"]      # (80, T+1, N, D)
    train_A = payload["train_A"]      # (80, T+1, N, N)
    train_a = payload["train_actions"]
    val_X = payload["val_X"]
    val_A = payload["val_A"]
    val_a = payload["val_actions"]
    test_X = payload["test_X"]
    test_A = payload["test_A"]
    test_a = payload["test_actions"]
    W_gt = payload.get("W")
    U_gt = payload.get("U")
    Q_gt = payload.get("Q")

    # Build model
    node_model = build_node_model(baseline)
    model = WorldModelWithEdgeHead(node_model, D=D, hidden=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())

    # Construct 1-step pairs (X_t, A_t_normalized, a_t) -> (X_{t+1}, A_{t+1})
    def normalize_A_batch(A: np.ndarray) -> np.ndarray:
        # A shape (..., N, N). Symmetric normalize with self-loop.
        N_g = A.shape[-1]
        A_self = A + np.eye(N_g, dtype=np.float32)
        d = A_self.sum(axis=-1, keepdims=True)
        d_safe = np.where(d > 0, d, 1.0)
        d_inv_sqrt = 1.0 / np.sqrt(d_safe)
        # Sym norm: D^-1/2 (A+I) D^-1/2
        A_norm = A_self * d_inv_sqrt * np.swapaxes(d_inv_sqrt, -1, -2)
        return A_norm.astype(np.float32)

    # Flatten to 1-step pairs: (n_traj * T, N, D), with corresponding A
    def make_pairs(X, A, a):
        n_traj, T_plus_1, N_g, _ = X.shape
        T = T_plus_1 - 1
        # X_t, X_{t+1}: (n_traj * T, N, D)
        X_in = X[:, :-1].reshape(-1, N_g, D)
        X_target = X[:, 1:].reshape(-1, N_g, D)
        A_in = A[:, :-1].reshape(-1, N_g, N_g)
        A_target = A[:, 1:].reshape(-1, N_g, N_g)
        a_in = a.reshape(-1, D_a)
        return X_in, X_target, A_in, A_target, a_in

    tr_Xin, tr_Xt, tr_Ain, tr_At, tr_ain = make_pairs(train_X, train_A, train_a)
    va_Xin, va_Xt, va_Ain, va_At, va_ain = make_pairs(val_X, val_A, val_a)
    # Normalize A
    tr_Anorm = normalize_A_batch(tr_Ain)
    va_Anorm = normalize_A_batch(va_Ain)
    # To tensors
    tr_Xin_t = torch.from_numpy(tr_Xin).float().to(device)
    tr_Xt_t = torch.from_numpy(tr_Xt).float().to(device)
    tr_Anorm_t = torch.from_numpy(tr_Anorm).float().to(device)
    tr_At_t = torch.from_numpy(tr_At).float().to(device)
    tr_ain_t = torch.from_numpy(tr_ain).float().to(device)
    va_Xin_t = torch.from_numpy(va_Xin).float().to(device)
    va_Xt_t = torch.from_numpy(va_Xt).float().to(device)
    va_Anorm_t = torch.from_numpy(va_Anorm).float().to(device)
    va_At_t = torch.from_numpy(va_At).float().to(device)
    va_ain_t = torch.from_numpy(va_ain).float().to(device)

    n_train = tr_Xin_t.shape[0]
    train_loss_curve = []
    val_loss_curve = []
    best_val = float("inf")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        loss_node_sum = 0; loss_edge_sum = 0; nb = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            X_pred, A_pred_logits = model.forward_step(
                tr_Xin_t[idx], tr_Anorm_t[idx], tr_ain_t[idx])
            loss_node = F.mse_loss(X_pred, tr_Xt_t[idx])
            # Edge BCE: predict A_{t+1} from X_{t+1}
            loss_edge = F.binary_cross_entropy_with_logits(
                A_pred_logits, tr_At_t[idx])
            loss = loss_node + lam_edge * loss_edge
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_node_sum += float(loss_node.item())
            loss_edge_sum += float(loss_edge.item())
            nb += 1
        train_loss_curve.append({
            "epoch": ep,
            "loss_node": loss_node_sum / max(nb, 1),
            "loss_edge": loss_edge_sum / max(nb, 1),
        })
        # val
        model.eval()
        with torch.no_grad():
            Xp, Ap = model.forward_step(va_Xin_t, va_Anorm_t, va_ain_t)
            vl_node = float(F.mse_loss(Xp, va_Xt_t).item())
            vl_edge = float(F.binary_cross_entropy_with_logits(Ap, va_At_t).item())
            vl = vl_node + lam_edge * vl_edge
        val_loss_curve.append({"epoch": ep, "loss_node": vl_node, "loss_edge": vl_edge, "total": vl})
        if vl < best_val:
            best_val = vl

    # ---- Eval: rollout on test set (T=32) ----
    model.eval()
    with torch.no_grad():
        X0 = torch.from_numpy(test_X[:, 0]).float().to(device)
        # Initial A (un-normalized then we normalize inside rollout)
        A0_normed = torch.from_numpy(normalize_A_batch(test_A[:, 0])).float().to(device)
        actions = torch.from_numpy(test_a).float().to(device)
        T_test = test_X.shape[1] - 1
        X_pred_traj, A_logits_traj = model.rollout_predict(
            X0, A0_normed, actions, T=T_test, return_edges=True)
        X_pred_np = X_pred_traj.cpu().numpy().astype(np.float32)
        A_pred_logits_np = A_logits_traj.cpu().numpy() if A_logits_traj is not None else None

    # Metrics per H
    test_X_np = test_X
    test_A_np = test_A
    metrics = {}
    for h in HORIZONS:
        if h > T_test:
            continue
        # NodeMSE
        per_traj_mse = []
        per_traj_f1 = []
        for i in range(test_X_np.shape[0]):
            diff = X_pred_np[i, h] - test_X_np[i, h]
            nm = float(np.mean(diff ** 2))
            nm = min(nm, CEIL) if math.isfinite(nm) else CEIL
            per_traj_mse.append(nm)
            # EdgeF1: sigmoid(logits) > 0.5 vs A_true
            if A_pred_logits_np is not None and h <= A_pred_logits_np.shape[1]:
                A_pred_h = 1.0 / (1.0 + np.exp(-A_pred_logits_np[i, h - 1]))
                A_pred_h_hard = (A_pred_h > 0.5).astype(np.float32)
                A_true_h = test_A_np[i, h]
                # F1
                tp = float(((A_pred_h_hard == 1) & (A_true_h == 1)).sum())
                fp = float(((A_pred_h_hard == 1) & (A_true_h == 0)).sum())
                fn = float(((A_pred_h_hard == 0) & (A_true_h == 1)).sum())
                if tp + fp > 0:
                    prec = tp / (tp + fp)
                else:
                    prec = 0.0
                if tp + fn > 0:
                    rec = tp / (tp + fn)
                else:
                    rec = 0.0
                if prec + rec > 0:
                    f1 = 2 * prec * rec / (prec + rec)
                else:
                    f1 = 0.0
                per_traj_f1.append(float(f1))
        metrics[f"NodeMSE@{h}"] = float(np.mean(per_traj_mse))
        metrics[f"NodeMSE@{h}_std"] = float(np.std(per_traj_mse))
        if per_traj_f1:
            metrics[f"EdgeF1@{h}"] = float(np.mean(per_traj_f1))
            metrics[f"EdgeF1@{h}_std"] = float(np.std(per_traj_f1))

    # ---- theory_constants with L_g ----
    g = generate(topo, N=N, seed=seed)
    model_W = model.gnn_W()
    if not model_W:
        model_W = [np.eye(D, dtype=np.float32)]
    L_g = model.get_L_g()
    tc = theory_constants(
        g.A_dense, model_W,
        X=test_X_np[0, 0],
        sigma_lipschitz=1.0,
        Q=Q_gt if Q_gt is not None else None,
        dynamic_edge=True,
        L_g=L_g,
    )
    tc["rho_A_norm"] = float(compute_all(g)["rho_A_norm"])
    tc["L_g"] = float(L_g)
    tc["lam_edge"] = float(lam_edge)

    elapsed = time.time() - t0
    result = {
        "meta": {
            "baseline": baseline, "topology": topo, "seed": int(seed),
            "N": N, "D": D, "epochs": epochs,
            "lr": lr, "lam_edge": lam_edge,
            "n_params": int(n_params),
            "train_time_sec": float(elapsed),
            "device": str(device),
            "timestamp_jst": now_jst(),
            "stream": "A_DE_trained",
        },
        "train_loss_curve": train_loss_curve,
        "val_loss_curve": val_loss_curve,
        "best_val_loss": float(best_val),
        "test_metrics": metrics,
        "theory_constants": tc,
        "graph_stats": compute_all(g),
    }
    # Save
    out_path = os.path.join(out_dir, topo, f"{baseline}_seed{seed}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    if save_ckpt:
        ck_dir = os.path.join(out_dir, "checkpoints", topo)
        os.makedirs(ck_dir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "config": {"baseline": baseline, "topo": topo, "seed": seed}},
                   os.path.join(ck_dir, f"{baseline}_seed{seed}.pt"))
    return result


def worker_fn(gpu_id, job_q, res_q, data_root, out_dir, log_path, epochs):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, REPO_ROOT)
    import torch
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda")
    while True:
        try:
            job = job_q.get(timeout=10)
        except queue_mod.Empty:
            continue
        if job is None:
            res_q.put(("done", gpu_id, None))
            return
        baseline, topo, seed = job
        # Skip if exists
        out_path = os.path.join(out_dir, topo, f"{baseline}_seed{seed}.json")
        if skip_if_done(out_path):
            res_q.put(("skipped", gpu_id, {"baseline": baseline, "topology": topo,
                                            "seed": seed, "elapsed_sec": 0,
                                            "status": "skipped"}))
            continue
        t0 = time.time()
        try:
            result = train_one_de(baseline, topo, seed,
                                   data_root=data_root, out_dir=out_dir,
                                   device=dev, epochs=epochs)
            status = "skipped" if result.get("skipped") else "ok"
        except Exception as e:
            import traceback
            with open(log_path, "a") as f:
                f.write(f"[{now_jst()}] [GPU{gpu_id}] EXCEPTION on ({baseline},{topo},seed{seed}):\n")
                f.write(traceback.format_exc())
                f.write("\n")
            status = "error"
        elapsed = time.time() - t0
        res_q.put((status, gpu_id, {
            "baseline": baseline, "topology": topo, "seed": seed,
            "elapsed_sec": elapsed, "status": status, "ts": now_jst(),
        }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="PROJECT_ROOT/data")
    parser.add_argument("--out_dir", default="PROJECT_ROOT/results/de_trained")
    parser.add_argument("--log_path", default="PROJECT_ROOT/logs/de_trained.log")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--filter_baselines", nargs="+", default=None)
    args = parser.parse_args()
    print(f"[{now_jst()}] Stream A: DE-trained baselines start; gpus={args.gpu_ids}")

    bs = args.filter_baselines or BASELINES
    jobs = []
    for bl in bs:
        for top in TOPOLOGIES:
            for s in SEEDS:
                jobs.append((bl, top, s))
    n_total = len(jobs)
    print(f"  total jobs: {n_total}")
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, "a") as f:
        f.write(f"[{now_jst()}] Stream A START n_jobs={n_total} gpus={args.gpu_ids}\n")

    job_q = mp.Queue()
    res_q = mp.Queue()
    for j in jobs:
        job_q.put(j)
    for _ in args.gpu_ids:
        job_q.put(None)
    procs = []
    for gpu in args.gpu_ids:
        p = mp.Process(target=worker_fn,
                       args=(gpu, job_q, res_q, args.data_root, args.out_dir,
                             args.log_path, args.epochs))
        p.start()
        procs.append(p)
    n_done = n_ok = n_skipped = n_err = 0
    summary = []
    left = len(args.gpu_ids)
    t_start = time.time()
    while left > 0:
        try:
            status, gpu, info = res_q.get(timeout=3600)
        except queue_mod.Empty:
            break
        if status == "done":
            left -= 1; continue
        n_done += 1
        if status == "ok": n_ok += 1
        elif status == "skipped": n_skipped += 1
        else: n_err += 1
        summary.append(info)
        msg = (f"[{now_jst()}] [{n_done}/{n_total}] GPU{gpu} {info['baseline']:14s} "
               f"{info['topology']:12s} seed{info['seed']} {status} ({info['elapsed_sec']:.0f}s)")
        print(msg)
        with open(args.log_path, "a") as f:
            f.write(msg + "\n")
    for p in procs:
        p.join(timeout=10)
        if p.is_alive(): p.terminate()
    elapsed = time.time() - t_start
    sum_path = os.path.join(REPO_ROOT, "results", "stream_a_summary.json")
    os.makedirs(os.path.dirname(sum_path), exist_ok=True)
    with open(sum_path, "w") as f:
        json.dump({"timestamp_jst": now_jst(), "n_total": n_total,
                   "n_ok": n_ok, "n_skipped": n_skipped, "n_err": n_err,
                   "elapsed_sec": elapsed, "jobs": summary}, f, indent=2, default=str)
    print(f"\n[{now_jst()}] STREAM A DONE {n_ok} ok / {n_skipped} skipped / {n_err} err, "
          f"{elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
