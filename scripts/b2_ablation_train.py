from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import multiprocessing as mp
import queue as queue_mod
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines import BASELINE_REGISTRY
from src.baselines.b2_variants import B2GCN_specproj
from src.graph_generators import generate, compute_all
from src.metrics import (
    RolloutPrediction, node_mse, edge_f1_binary, growth_slope,
    theory_constants,
)
from scripts._runner_utils import skip_if_done, now_jst

CEIL = 1e10
N = 50
D = 8
D_a = 4

VARIANTS = ["B2_wd", "B2_clip", "B2_specproj"]
TOPOLOGIES = ["chain", "tree", "grid", "small_world", "scale_free", "star", "complete"]
SEEDS = [1, 2, 3]


def train_one_variant(
    variant: str, top: str, outer_seed: int, *,
    data_root: str, out_dir: str, device: torch.device,
    epochs: int = 100, batch_size: int = 16,
) -> Dict[str, Any]:

    t0 = time.time()

    py_seed = outer_seed * 1000 + 7
    np.random.seed(py_seed)
    torch.manual_seed(py_seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(py_seed + 2)


    if variant == "B2_wd":
        cls = BASELINE_REGISTRY["B2_wd"]
        lr, weight_decay, grad_clip = 1e-3, 0.01, 1.0
        use_specproj = False
    elif variant == "B2_clip":
        cls = BASELINE_REGISTRY["B2_clip"]
        lr, weight_decay, grad_clip = 1e-3, 0.0, 0.1
        use_specproj = False
    elif variant == "B2_specproj":
        cls = BASELINE_REGISTRY["B2_specproj"]
        lr, weight_decay, grad_clip = 1e-3, 0.0, 1.0
        use_specproj = True
    else:
        raise ValueError(variant)


    rollout_path = os.path.join(data_root, "synthetic_rollouts",
                                f"fe_{top}_N{N}_seed{outer_seed}_T50.pt")
    payload = torch.load(rollout_path, weights_only=False)
    train_X = torch.from_numpy(payload["train_X"]).float()
    val_X = torch.from_numpy(payload["val_X"]).float()
    test_X = torch.from_numpy(payload["test_X"]).float()
    train_a = torch.from_numpy(payload["train_actions"]).float()
    val_a = torch.from_numpy(payload["val_actions"]).float()
    test_a = torch.from_numpy(payload["test_actions"]).float()
    W_gt = payload["W"]

    g = generate(top, N=N, seed=outer_seed)
    A_norm_t = torch.from_numpy(g.A_norm).float().to(device)
    A_dense = g.A_dense
    stats = compute_all(g)
    rho_A_norm = stats["rho_A_norm"]

    model = cls(D=D, D_a=D_a).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in model.parameters())


    def make_pairs(X, A):
        N_g = X.shape[2]
        Xt = X[:, :-1].reshape(-1, N_g, D)
        Xtp1 = X[:, 1:].reshape(-1, N_g, D)
        At = A.reshape(-1, A.shape[-1])
        return Xt, Xtp1, At

    train_Xt, train_Xtp1, train_At = make_pairs(train_X, train_a)
    val_Xt, val_Xtp1, val_At = make_pairs(val_X, val_a)
    train_Xt, train_Xtp1, train_At = train_Xt.to(device), train_Xtp1.to(device), train_At.to(device)
    val_Xt, val_Xtp1, val_At = val_Xt.to(device), val_Xtp1.to(device), val_At.to(device)
    n_train = train_Xt.shape[0]

    train_losses, val_losses = [], []
    best_val = float("inf")
    n_param_updates = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        loss_sum = 0.0
        n_batch = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            X_in = train_Xt[idx]
            X_target = train_Xtp1[idx]
            a_in = train_At[idx]
            X_pred = model.forward_step(X_in, A_norm_t, a_in)
            loss = F.mse_loss(X_pred, X_target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if use_specproj and isinstance(model, B2GCN_specproj):
                model.project_spectral_norm(target_spec=1.0)
            loss_sum += float(loss.item())
            n_batch += 1
            n_param_updates += 1
        train_loss = loss_sum / max(n_batch, 1)
        train_losses.append(train_loss)
        model.eval()
        with torch.no_grad():
            X_pred = model.forward_step(val_Xt, A_norm_t, val_At)
            vl = float(F.mse_loss(X_pred, val_Xtp1).item())
        val_losses.append(vl)
        if vl < best_val:
            best_val = vl


    model.eval()
    with torch.no_grad():
        X_0 = test_X[:, 0].to(device)
        actions_test = test_a.to(device)
        T_test = test_X.shape[1] - 1
        X_pred_traj = model.rollout_predict(X_0, A_norm_t, actions_test, T=T_test)
        X_pred_traj = X_pred_traj.cpu().numpy().astype(np.float32)
    test_X_np = test_X.numpy()
    eval_horizons = [1, 2, 4, 8, 16, 32]
    metrics = {}
    for h in eval_horizons:
        if h > T_test:
            continue
        per_traj_mse = []
        for i in range(test_X_np.shape[0]):
            pred = RolloutPrediction(
                X_true=test_X_np[i], A_true=A_dense,
                X_pred=X_pred_traj[i], A_pred=A_dense,
                is_fixed_edge=True,
            )
            per_traj_mse.append(node_mse(pred, h))
        metrics[f"NodeMSE@{h}"] = float(np.mean(per_traj_mse))
        metrics[f"NodeMSE@{h}_std"] = float(np.std(per_traj_mse))


    slopes = []
    for i in range(test_X_np.shape[0]):
        pred = RolloutPrediction(
            X_true=test_X_np[i], A_true=A_dense,
            X_pred=X_pred_traj[i], A_pred=A_dense,
            is_fixed_edge=True,
        )
        slopes.append(growth_slope(pred, 4, min(32, T_test)))
    metrics["GrowthSlope_4_32"] = float(np.mean(slopes))


    nm32 = metrics.get("NodeMSE@32", float("nan"))
    diverged = (
        not math.isfinite(nm32) or nm32 > 1e3 or
        not math.isfinite(train_losses[-1])
    )


    model_W_np = model.gnn_W() if hasattr(model, "gnn_W") else [W_gt.astype(np.float32)]
    if not model_W_np:
        model_W_np = [W_gt.astype(np.float32)]
    tc = theory_constants(A_dense, model_W_np, X=test_X_np[0, 0],
                          sigma_lipschitz=1.0, Q=None, dynamic_edge=False)
    tc["rho_A_norm"] = float(rho_A_norm)

    elapsed = time.time() - t0
    result = {
        "meta": {
            "variant": variant, "topology": top, "outer_seed": int(outer_seed),
            "N": N, "D": D, "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "grad_clip_norm": grad_clip,
            "use_specproj": use_specproj,
            "n_params": int(n_params),
            "n_param_updates": int(n_param_updates),
            "device": str(device),
            "train_time_sec": float(elapsed),
            "timestamp_jst": now_jst(),
        },
        "train_loss_curve": train_losses,
        "val_loss_curve": val_losses,
        "best_val_loss": float(best_val),
        "test_metrics": metrics,
        "diverged": bool(diverged),
        "theory_constants": tc,
        "graph_stats": stats,
    }
    out_path = os.path.join(out_dir, variant, top, f"seed{outer_seed}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    return result


def worker_fn(gpu_id: int, job_q: mp.Queue, res_q: mp.Queue,
              data_root: str, out_dir: str, log_path: str, epochs: int):
    sys.path.insert(0, REPO_ROOT)
    import torch
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(gpu_id)
    except Exception as exc:
        res_q.put(("unavailable", gpu_id, {"error": str(exc)}))
        return
    while True:
        try:
            job = job_q.get(timeout=10)
        except queue_mod.Empty:
            continue
        if job is None:
            res_q.put(("done", gpu_id, None)); return
        variant, top, seed = job
        out_path = os.path.join(out_dir, variant, top, f"seed{seed}.json")
        if skip_if_done(out_path):
            res_q.put(("skipped", gpu_id, {"variant": variant, "topology": top,
                                            "seed": seed, "status": "skipped"}))
            continue
        t0 = time.time()
        try:
            result = train_one_variant(
                variant, top, seed,
                data_root=data_root, out_dir=out_dir,
                device=torch.device(f"cuda:{gpu_id}"), epochs=epochs,
            )
            status = "ok"
            if result["diverged"]:
                status = "ok_diverged"
        except Exception as e:
            import traceback
            with open(log_path, "a") as f:
                f.write(f"[{now_jst()}] [GPU{gpu_id}] EXCEPTION on ({variant},{top},seed{seed}):\n")
                f.write(traceback.format_exc())
                f.write("\n")
            status = "error"
            result = {"error": str(e)[:200]}
        elapsed = time.time() - t0
        res_q.put((status, gpu_id, {
            "variant": variant, "topology": top, "seed": seed,
            "elapsed_sec": elapsed, "status": status, "ts": now_jst(),
        }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, 'data'))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, 'results', 'p2_b2_ablation'))
    parser.add_argument("--log_path", default=os.path.join(REPO_ROOT, 'logs', 'b2_ablation.log'))
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--summary_path", default=None)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this batch runner")
    if len(set(args.gpu_ids)) != len(args.gpu_ids):
        parser.error("--gpu_ids must contain distinct device indices")
    if any(gpu_id < 0 or gpu_id >= torch.cuda.device_count() for gpu_id in args.gpu_ids):
        parser.error("--gpu_ids contains an unavailable device index")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    args.summary_path = args.summary_path or os.path.join(args.out_dir, "summary.json")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[{now_jst()}] B2 ablation start; gpus={args.gpu_ids} epochs={args.epochs}")


    jobs = []
    for variant in VARIANTS:
        for top in TOPOLOGIES:
            for seed in SEEDS:
                jobs.append((variant, top, seed))
    n_total = len(jobs)
    print(f"  total jobs: {n_total} ({len(VARIANTS)} variants × {len(TOPOLOGIES)} topos × {len(SEEDS)} seeds)")

    os.makedirs(os.path.dirname(os.path.abspath(args.log_path)), exist_ok=True)
    with open(args.log_path, "a") as f:
        f.write(f"[{now_jst()}] B2 ABLATION START n_jobs={n_total} gpus={args.gpu_ids}\n")

    ctx = mp.get_context("spawn")
    job_q = ctx.Queue()
    res_q = ctx.Queue()
    for j in jobs:
        job_q.put(j)
    for _ in args.gpu_ids:
        job_q.put(None)
    procs = []
    for gpu_id in args.gpu_ids:
        p = ctx.Process(target=worker_fn,
                       args=(gpu_id, job_q, res_q,
                             args.data_root, args.out_dir,
                             args.log_path, args.epochs))
        p.start()
        procs.append(p)

    n_done = n_ok = n_skipped = n_err = n_div = 0
    summary = []
    left = len(args.gpu_ids)
    t_start = time.time()
    worker_errors = []
    try:
        while left > 0:
            try:
                status, gpu_id, info = res_q.get(timeout=5)
            except queue_mod.Empty:
                failed = [(args.gpu_ids[i], proc.exitcode) for i, proc in enumerate(procs)
                          if proc.exitcode not in (None, 0)]
                if failed or all(not proc.is_alive() for proc in procs):
                    worker_errors.append({"error": "Workers exited before completion", "failed": failed})
                    break
                continue
            if status == "unavailable":
                worker_errors.append({"gpu_id": gpu_id, **info})
                break
            if status == "done":
                left -= 1; continue
            n_done += 1
            if status == "ok": n_ok += 1
            elif status == "ok_diverged":
                n_ok += 1; n_div += 1
            elif status == "skipped": n_skipped += 1
            else: n_err += 1
            summary.append(info)
            msg = (f"[{now_jst()}] [{n_done}/{n_total}] GPU{gpu_id} "
                   f"{info.get('variant', '?'):14s} {info.get('topology', '?'):12s} "
                   f"seed{info.get('seed', '?')} {status} "
                   f"({info.get('elapsed_sec', 0):.0f}s)")
            print(msg)
            with open(args.log_path, "a") as f:
                f.write(msg + "\n")
    finally:
        for proc in procs:
            if worker_errors and proc.is_alive():
                proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
                worker_errors.append({"error": "Worker did not exit", "pid": proc.pid})
        job_q.cancel_join_thread()
        job_q.close()
        res_q.close()
    if n_done != n_total:
        worker_errors.append({"error": "Incomplete batch", "n_done": n_done, "n_total": n_total})
    failed_exits = [proc.exitcode for proc in procs if proc.exitcode not in (0,)]
    if failed_exits:
        worker_errors.append({"error": "Abnormal worker exit", "exit_codes": failed_exits})
    elapsed = time.time() - t_start
    sum_path = args.summary_path
    os.makedirs(os.path.dirname(os.path.abspath(sum_path)), exist_ok=True)
    with open(sum_path, "w") as f:
        json.dump({
            "timestamp_jst": now_jst(),
            "n_total": n_total, "n_ok": n_ok, "n_div": n_div,
            "n_skipped": n_skipped, "n_err": n_err,
            "elapsed_sec": elapsed,
            "jobs": summary, "worker_errors": worker_errors,
        }, f, indent=2, default=str)
    if worker_errors or n_err:
        raise SystemExit(f"Batch failed; details: {sum_path}")
    print(f"\n[{now_jst()}] BATCH DONE: {n_ok} ok ({n_div} diverged) / "
          f"{n_skipped} skip / {n_err} err, {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
