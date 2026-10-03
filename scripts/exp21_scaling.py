from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.graph_generators import generate
from src.simulators import rollout

JST = timezone(timedelta(hours=9))

BASELINES = ["B1_MLP", "B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode", "B6_ErrorAware"]
TOPOLOGIES_SCALING = ["chain", "tree", "grid", "small_world", "scale_free"]
SEEDS = [1, 2, 3]
N_VALUES = [20, 50, 100, 200, 500]
TOP_DEFAULTS = {
    "chain": {}, "tree": {"variant": "balanced_binary"}, "grid": {"shape": "auto"},
    "small_world": {"k": 4, "p": 0.1}, "scale_free": {"m": 2},
}


def now_jst() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def gen_rollouts_for_N(data_root: str, N: int, force: bool = False) -> int:
    if N == 50:
        return 0
    n_done = 0
    out_dir = os.path.join(data_root, "synthetic_rollouts")
    os.makedirs(out_dir, exist_ok=True)
    for top in TOPOLOGIES_SCALING:
        for s in SEEDS:
            fname = f"fe_{top}_N{N}_seed{s}_T50.pt"
            path = os.path.join(out_dir, fname)
            if os.path.exists(path) and not force:
                continue
            params = TOP_DEFAULTS.get(top, {})
            g = generate(top, N=N, seed=s, **params)
            train_X, val_X, test_X = [], [], []
            train_a, val_a, test_a = [], [], []
            W_shared = U_shared = None
            n_train, n_val, n_test = (30, 5, 5) if N >= 200 else (50, 10, 10)
            for split_name, n_traj, start in [("train", n_train, 0),
                                              ("val", n_val, 100),
                                              ("test", n_test, 200)]:
                for idx in range(n_traj):
                    inner = start + idx
                    tr = rollout(g, T=50, mode="fixed_edge", sigma_noise=0.01, seed=inner)
                    if W_shared is None:
                        W_shared, U_shared = tr.W.copy(), tr.U.copy()
                    if split_name == "train":
                        train_X.append(tr.X); train_a.append(tr.actions)
                    elif split_name == "val":
                        val_X.append(tr.X); val_a.append(tr.actions)
                    else:
                        test_X.append(tr.X); test_a.append(tr.actions)
            payload = {
                "version": 1, "topology": top, "N": N, "D": 8,
                "outer_seed": int(s), "T_train": 50,
                "W": W_shared, "U": U_shared,
                "train_X": np.stack(train_X), "train_actions": np.stack(train_a),
                "val_X": np.stack(val_X), "val_actions": np.stack(val_a),
                "test_X": np.stack(test_X), "test_actions": np.stack(test_a),
                "config": {"mode": "fixed_edge", "sigma_noise": 0.01,
                           "target_w_norm": 0.9, "target_u_norm": 0.5},
            }
            torch.save(payload, path)
            n_done += 1
    return n_done


def worker_fn(gpu_id: int, job_q: "mp.Queue", res_q: "mp.Queue",
              data_root: str, out_dir: str, log_path: str, epochs: int = 60):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, REPO_ROOT)
    from scripts.train_one_baseline import train_one
    import torch
    if not torch.cuda.is_available():
        print(f"[worker GPU{gpu_id}] CUDA unavailable, abort"); return
    while True:
        try:
            job = job_q.get(timeout=10)
        except queue_mod.Empty:
            continue
        if job is None:
            res_q.put(("done", gpu_id, None))
            return
        baseline, top, seed, N = job
        t0 = time.time()
        try:
            sub_out = os.path.join(out_dir, f"N{N}")
            result = train_one(
                baseline, top, seed,
                data_root=data_root, out_dir=sub_out,
                N=N, epochs=epochs, device="cuda",
            )
            status = "skipped" if result.get("skipped") else "ok"
        except Exception as e:
            import traceback
            with open(log_path, "a") as f:
                f.write(f"[{now_jst()}] [GPU{gpu_id}] EXCEPTION on ({baseline},{top},seed{seed},N{N}):\n")
                f.write(traceback.format_exc())
                f.write("\n")
            status = "error"
            result = {"error": str(e)[:200]}
        elapsed = time.time() - t0
        res_q.put((status, gpu_id, {
            "baseline": baseline, "topology": top, "seed": seed, "N": N,
            "elapsed_sec": elapsed, "status": status, "ts": now_jst(),
        }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "exp21_scaling"))
    parser.add_argument("--log_path", default=os.path.join(REPO_ROOT, "logs", "exp21.log"))
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--skip_data_gen", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    print(f"[{now_jst()}] Exp 21 SCALING 启动; gpus={args.gpu_ids} epochs={args.epochs}")

    if not args.skip_data_gen:
        for N in N_VALUES:
            t0 = time.time()
            n_gen = gen_rollouts_for_N(args.data_root, N)
            print(f"  rollouts N={N}: {n_gen} new files in {time.time()-t0:.1f}s")

    jobs = []
    for bl in BASELINES:
        for N in N_VALUES:
            if bl == "B1_MLP" and N > 50:
                continue
            topos = TOPOLOGIES_SCALING
            for top in topos:
                for s in SEEDS:
                    if N == 50 and bl == "B1_MLP":
                        continue
                    if args.skip_existing:
                        path = os.path.join(args.out_dir, f"N{N}", top, f"{bl}_seed{s}.json")
                        if os.path.exists(path):
                            continue
                    jobs.append((bl, top, s, N))
    n_total = len(jobs)
    print(f"  total jobs: {n_total}")
    with open(args.log_path, "w") as f:
        f.write(f"[{now_jst()}] EXP 21 BATCH START n_jobs={n_total}\n")

    job_q: "mp.Queue" = mp.Queue()
    res_q: "mp.Queue" = mp.Queue()
    for j in jobs:
        job_q.put(j)
    for _ in args.gpu_ids:
        job_q.put(None)

    procs = []
    for gpu_id in args.gpu_ids:
        p = mp.Process(target=worker_fn, args=(gpu_id, job_q, res_q,
                                                args.data_root, args.out_dir,
                                                args.log_path, args.epochs))
        p.start()
        procs.append(p)

    n_done = n_ok = n_skipped = n_err = 0
    results_summary = []
    workers_left = len(args.gpu_ids)
    t_start = time.time()
    while workers_left > 0:
        try:
            status, gpu_id, info = res_q.get(timeout=3600)
        except queue_mod.Empty:
            print(f"[{now_jst()}] queue timeout, exit")
            break
        if status == "done":
            workers_left -= 1
            continue
        n_done += 1
        if status == "ok":
            n_ok += 1
        elif status == "skipped":
            n_skipped += 1
        else:
            n_err += 1
        results_summary.append(info)
        msg = (f"[{now_jst()}] [{n_done}/{n_total}] GPU{gpu_id} "
               f"{info['baseline']:14s} {info['topology']:12s} seed{info['seed']} "
               f"N={info['N']:>3d} {status} ({info['elapsed_sec']:.0f}s)")
        print(msg)
        with open(args.log_path, "a") as f:
            f.write(msg + "\n")

    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    total_elapsed = time.time() - t_start
    summary = {
        "timestamp_jst": now_jst(),
        "n_total_jobs": n_total, "n_done": n_done,
        "n_ok": n_ok, "n_skipped": n_skipped, "n_err": n_err,
        "total_elapsed_sec": total_elapsed,
        "gpu_ids": args.gpu_ids, "epochs": args.epochs,
        "jobs": results_summary,
    }
    sum_path = os.path.join(REPO_ROOT, "results", "exp21_summary.json")
    os.makedirs(os.path.dirname(sum_path), exist_ok=True)
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[{now_jst()}] EXP 21 DONE: {n_ok} ok, {n_skipped} skip, {n_err} err; "
          f"total {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
