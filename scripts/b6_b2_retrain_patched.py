from __future__ import annotations

import argparse
import json
import os
import sys
import time
import multiprocessing as mp
import queue as queue_mod
from datetime import datetime, timezone, timedelta

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

JST = timezone(timedelta(hours=9))
TOPOLOGIES = ["chain", "tree", "grid", "small_world", "scale_free", "star", "complete"]
SEEDS = [1, 2, 3]


def now_jst():
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def worker_fn(gpu_id, job_q, res_q, data_root, out_dir, log_path):
    sys.path.insert(0, REPO_ROOT)
    from scripts.train_one_baseline import train_one
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
            res_q.put(("done", gpu_id, None))
            return
        baseline, top, seed, epochs = job
        t0 = time.time()
        try:
            result = train_one(
                baseline, top, seed,
                data_root=data_root, out_dir=out_dir,
                N=50, epochs=epochs, device=f"cuda:{gpu_id}",
            )
            status = "ok" if not result.get("skipped") else "skipped"
        except Exception as e:
            import traceback
            with open(log_path, "a") as f:
                f.write(f"[{now_jst()}] [GPU{gpu_id}] EXCEPTION ({baseline},{top},seed{seed}):\n")
                f.write(traceback.format_exc())
                f.write("\n")
            status = "error"
            result = {"error": str(e)[:200]}
        elapsed = time.time() - t0
        res_q.put((status, gpu_id, {
            "baseline": baseline, "topology": top, "seed": seed,
            "elapsed_sec": elapsed, "status": status, "ts": now_jst(),
        }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, 'data'))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, 'results', 'p2_baselines_patched'))
    parser.add_argument("--log_path", default=os.path.join(REPO_ROOT, 'logs', 'b6_b2_retrain.log'))
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs_b6", type=int, default=100)
    parser.add_argument("--epochs_b2", type=int, default=100)
    parser.add_argument("--summary_path", default=None)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this batch runner")
    if len(set(args.gpu_ids)) != len(args.gpu_ids):
        parser.error("--gpu_ids must contain distinct device indices")
    if any(gpu_id < 0 or gpu_id >= torch.cuda.device_count() for gpu_id in args.gpu_ids):
        parser.error("--gpu_ids contains an unavailable device index")
    if args.epochs_b6 <= 0 or args.epochs_b2 <= 0:
        parser.error("--epochs_b6 and --epochs_b2 must be positive")
    args.summary_path = args.summary_path or os.path.join(args.out_dir, "summary.json")
    print(f"[{now_jst()}] B6+B2 retrain start; gpus={args.gpu_ids}")
    print(f"  epochs B6={args.epochs_b6}, B2={args.epochs_b2}")

    jobs = []
    for top in TOPOLOGIES:
        for s in SEEDS:
            jobs.append(("B6_ErrorAware", top, s, args.epochs_b6))


    b2_diverged_cells = [
        ("grid", 1), ("grid", 2), ("grid", 3),
        ("small_world", 1),
        ("scale_free", 1), ("scale_free", 2), ("scale_free", 3),
        ("star", 1), ("star", 2), ("star", 3),
    ]
    for top, s in b2_diverged_cells:
        jobs.append(("B2_GCN", top, s, args.epochs_b2))
    n_total = len(jobs)
    print(f"  total jobs: {n_total} (21 B6 + 10 B2)")
    os.makedirs(os.path.dirname(os.path.abspath(args.log_path)), exist_ok=True)
    with open(args.log_path, "a") as f:
        f.write(f"[{now_jst()}] B6+B2 RETRAIN START n_jobs={n_total} gpus={args.gpu_ids}\n")

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
                             args.data_root, args.out_dir, args.log_path))
        p.start()
        procs.append(p)
    n_done = n_ok = n_err = 0
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
            else: n_err += 1
            summary.append(info)
            msg = (f"[{now_jst()}] [{n_done}/{n_total}] GPU{gpu_id} "
                   f"{info['baseline']:14s} {info['topology']:12s} seed{info['seed']} "
                   f"{status} ({info['elapsed_sec']:.0f}s)")
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
        json.dump({"timestamp_jst": now_jst(), "n_total": n_total,
                   "n_ok": n_ok, "n_err": n_err,
                   "elapsed_sec": elapsed,
                   "jobs": summary, "worker_errors": worker_errors}, f, indent=2, default=str)
    if worker_errors or n_err:
        raise SystemExit(f"Batch failed; details: {sum_path}")
    print(f"\n[{now_jst()}] RETRAIN DONE: {n_ok} ok / {n_err} err, {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
