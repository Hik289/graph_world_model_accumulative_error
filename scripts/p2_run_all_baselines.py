"""P2 baselines 4-GPU parallel scheduler.

调度 6 baseline × 7 topology × 3 outer_seed = 126 jobs, 4 GPU 并行.
import os as _os; PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
逻辑:
- 4 个 worker (one per GPU); 每 worker pop job 从 queue, 串行训练
- 每 job 完成自动写 results/p2_baselines/{top}/{baseline}_seed{S}.json
- 顶层 progress 写到 logs/p2_progress.log
- 所有完成后写 results/p2_all_baselines_summary.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time
from datetime import datetime, timezone, timedelta

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

JST = timezone(timedelta(hours=9))

BASELINES = ["B1_MLP", "B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode", "B6_ErrorAware"]
TOPOLOGIES = ["chain", "tree", "grid", "small_world", "scale_free", "star", "complete"]
SEEDS = [1, 2, 3]


def now_jst() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def worker_fn(gpu_id: int, job_q: "mp.Queue", res_q: "mp.Queue",
              data_root: str, out_dir: str, log_path: str):
    """单 GPU worker. 持续从 queue 取 job 直到 sentinel (None)."""
    # 限制本 worker 只能看到一张卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # ⚠️ import torch 必须在 set env 之后
    sys.path.insert(0, REPO_ROOT)
    from scripts.train_one_baseline import train_one
    import torch
    if not torch.cuda.is_available():
        print(f"[worker GPU{gpu_id}] CUDA 不可用, 跳过")
        return
    while True:
        try:
            job = job_q.get(timeout=10)
        except queue_mod.Empty:
            continue
        if job is None:
            res_q.put(("done", gpu_id, None))
            return
        baseline, top, seed = job
        t0 = time.time()
        try:
            # 每个 worker 内部只看一张卡, 用 cuda:0
            result = train_one(
                baseline, top, seed,
                data_root=data_root, out_dir=out_dir,
                device="cuda",
            )
            status = "ok"
            if result.get("skipped"):
                status = "skipped"
        except Exception as e:
            import traceback
            with open(log_path, "a") as f:
                f.write(f"[{now_jst()}] [GPU{gpu_id}] EXCEPTION on ({baseline},{top},seed{seed}):\n")
                f.write(traceback.format_exc())
                f.write("\n")
            status = "error"
            result = {"error": str(e)[:500]}
        elapsed = time.time() - t0
        res_q.put((status, gpu_id, {
            "baseline": baseline, "topology": top, "seed": seed,
            "elapsed_sec": elapsed,
            "status": status,
            "ts": now_jst(),
        }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="PROJECT_ROOT/data")
    parser.add_argument("--out_dir", default="PROJECT_ROOT/results/p2_baselines")
    parser.add_argument("--log_path", default="PROJECT_ROOT/logs/p2_baselines.log")
    parser.add_argument("--n_gpus", type=int, default=3,
                        help="并发 GPU 数 (默认 3, 因 dltank 全空闲但可能他人会用)")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=None,
                        help="具体 GPU id 列表 (覆盖 n_gpus)")
    parser.add_argument("--filter_baselines", nargs="+", default=None)
    parser.add_argument("--filter_topologies", nargs="+", default=None)
    parser.add_argument("--filter_seeds", nargs="+", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true",
                        help="若 result JSON 已存在则跳过")
    args = parser.parse_args()

    # 检查 GPU 数
    if args.gpu_ids is None:
        args.gpu_ids = list(range(args.n_gpus))
    print(f"[{now_jst()}] 启动 P2 baseline 训练; gpus={args.gpu_ids}")

    # 构造 job list
    blines = args.filter_baselines or BASELINES
    tops = args.filter_topologies or TOPOLOGIES
    seeds = args.filter_seeds or SEEDS
    jobs = []
    for bl in blines:
        for top in tops:
            for s in seeds:
                # skip existing
                if args.skip_existing:
                    out_path = os.path.join(args.out_dir, top, f"{bl}_seed{s}.json")
                    if os.path.exists(out_path):
                        continue
                jobs.append((bl, top, s))
    n_total = len(jobs)
    print(f"  total jobs: {n_total} ({len(blines)} baselines × {len(tops)} topos × {len(seeds)} seeds)")

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, "a") as f:
        f.write(f"[{now_jst()}] P2 BATCH START n_jobs={n_total} gpus={args.gpu_ids}\n")

    # mp queues
    job_q: "mp.Queue" = mp.Queue()
    res_q: "mp.Queue" = mp.Queue()
    for j in jobs:
        job_q.put(j)
    for _ in args.gpu_ids:
        job_q.put(None)

    procs = []
    for gpu_id in args.gpu_ids:
        p = mp.Process(target=worker_fn,
                       args=(gpu_id, job_q, res_q,
                             args.data_root, args.out_dir, args.log_path))
        p.start()
        procs.append(p)

    # 收 result
    n_done = 0
    n_ok = n_skipped = n_err = 0
    results_summary = []
    workers_left = len(args.gpu_ids)
    t_start = time.time()
    while workers_left > 0:
        try:
            status, gpu_id, info = res_q.get(timeout=600)
        except queue_mod.Empty:
            print(f"[{now_jst()}] 等待超时 600s, 退出")
            break
        if status == "done":
            workers_left -= 1
            with open(args.log_path, "a") as f:
                f.write(f"[{now_jst()}] worker GPU{gpu_id} exited\n")
            continue
        n_done += 1
        if status == "ok":
            n_ok += 1
        elif status == "skipped":
            n_skipped += 1
        else:
            n_err += 1
        results_summary.append(info)
        msg = f"[{now_jst()}] [{n_done}/{n_total}] GPU{gpu_id} {info['baseline']:14s} {info['topology']:12s} seed{info['seed']} {status} ({info['elapsed_sec']:.0f}s)"
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
        "n_total_jobs": n_total,
        "n_done": n_done,
        "n_ok": n_ok, "n_skipped": n_skipped, "n_err": n_err,
        "total_elapsed_sec": total_elapsed,
        "gpu_ids": args.gpu_ids,
        "jobs": results_summary,
    }
    sum_path = os.path.join(REPO_ROOT, "results", "p2_all_baselines_summary.json")
    os.makedirs(os.path.dirname(sum_path), exist_ok=True)
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[{now_jst()}] BATCH DONE: {n_ok} ok, {n_skipped} skipped, {n_err} errors; "
          f"total {total_elapsed/60:.1f} min")
    print(f"summary: {sum_path}")


if __name__ == "__main__":
    main()
