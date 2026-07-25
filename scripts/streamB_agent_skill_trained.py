"""Stream B: End-to-end training on agent calling and platform skill graphs.

This replaces the P5 zero-shot evaluation of homogeneous models on
heterogeneous data.

Spec:
- 5 baselines (B1 MLP / B2 GCN / B3 MPNN / B4 GPS / B5 ActionNode) + B6 patched + R-GCN-Hetero  
- Trained 50 epochs × 3 seeds × 2 testbeds = ~36 jobs total
- Use R-GCN-Hetero implementation (already exists, has edge_type/node_type embeddings)
- For homogeneous baselines, use edge_type=0 (ignore types) or skip homogeneous on hetero (commonly we use RGCN only)

All variants use the R-GCN-style edge embeddings implemented by
``RGCNHetero``, with the number of edge types set per testbed.

Outputs: results/agent_skill_trained/{testbed}/{baseline}_seed{S}.json
"""
from __future__ import annotations

import argparse
import json
import os
import queue as queue_mod
import sys
import time
import multiprocessing as mp
from datetime import timezone, timedelta
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines.rgcn_hetero import RGCNHetero
from src.simulators.agent_calling_tree import NODE_TYPES, EDGE_TYPES, D_FEAT, D_ACTION
from src.simulators.platform_skill_graph import SKILL_NODE_TYPES, SKILL_EDGE_TYPES
from src.metrics import failure_propagation_depth
from src.metrics.core import cost_latency
from scripts._runner_utils import skip_if_done, now_jst

JST = timezone(timedelta(hours=9))
CEIL = 1e10
SEEDS = [1, 2, 3]
# For Stream B, we use R-GCN-Hetero as the universal model class (per agent_calling_tree spec
# uses 9 node types, 6 edge types; skill_graph 8/8)
# 5 "baselines" map to 5 hidden dims / depths / layer-counts to provide minimal architectural variation
BASELINE_CONFIGS = {
    "B1_RGCN_small":  {"hidden": 32, "n_layers": 2},   # like B1
    "B2_RGCN_med":    {"hidden": 64, "n_layers": 2},   # like B2 GCN
    "B3_RGCN_deep":   {"hidden": 64, "n_layers": 3},   # like B3 MPNN (more layers)
    "B4_RGCN_wide":   {"hidden": 128, "n_layers": 2},  # like B4 GPS (wider)
    "B5_RGCN_actn":   {"hidden": 64, "n_layers": 2},   # like B5 ActionNode
    "B6_RGCN_critical": {"hidden": 64, "n_layers": 2, "use_critical_loss": True},  # B6 with R_critical
}


def train_one_hetero(
    baseline: str, testbed: str, seed: int, *,
    data_root: str, out_dir: str,
    device: torch.device,
    epochs: int = 50, lr: float = 1e-3,
    n_train: int = 100,
    save_ckpt: bool = True,
) -> Dict[str, Any]:
    """Train R-GCN-based baseline on hetero testbed."""
    t0 = time.time()
    py_seed = seed * 1000 + 7
    np.random.seed(py_seed); torch.manual_seed(py_seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(py_seed + 2)

    # Determine node/edge type counts
    if testbed == "agent_calling_tree":
        n_node_types = len(NODE_TYPES); n_edge_types = len(EDGE_TYPES)
    elif testbed == "platform_skill_graph":
        n_node_types = len(SKILL_NODE_TYPES); n_edge_types = len(SKILL_EDGE_TYPES)
    else:
        raise ValueError(testbed)
    cfg = BASELINE_CONFIGS[baseline]
    use_critical_loss = cfg.get("use_critical_loss", False)
    lam_critical = 0.1

    model = RGCNHetero(D=D_FEAT, D_a=D_ACTION,
                       n_node_types=n_node_types,
                       n_edge_types=n_edge_types,
                       hidden=cfg["hidden"], n_layers=cfg["n_layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Load training instances
    train_dir = os.path.join(data_root, testbed, "train")
    train_files = sorted(os.listdir(train_dir))[:n_train]
    train_loss_curve = []
    for ep in range(epochs):
        model.train()
        loss_sum = 0; nb = 0
        np.random.shuffle(train_files)
        for fname in train_files:
            try:
                inst = torch.load(os.path.join(train_dir, fname), weights_only=False)
            except Exception:
                continue
            X_traj = torch.from_numpy(inst["trace_X"]).float().to(device)
            actions = torch.from_numpy(inst["trace_actions"]).float().to(device)
            edge_index = torch.from_numpy(np.array(inst["edge_index"], dtype=np.int64)).to(device)
            edge_type = torch.from_numpy(np.array(inst["edge_type"], dtype=np.int64)).to(device)
            node_types = torch.from_numpy(np.array(inst["node_types"], dtype=np.int64)).to(device)
            T_traj = X_traj.shape[0] - 1
            # T-batched forward (per speedup pattern from P6)
            X_in = X_traj[:-1]  # (T, N, D)
            X_target = X_traj[1:]
            a_in = actions
            X_pred = model.forward_step(X_in, edge_index, edge_type, a_in, node_types=node_types)
            loss = F.mse_loss(X_pred, X_target)
            if use_critical_loss:
                # R_critical: degree-weighted MSE
                N_g = X_in.shape[1]
                A_dense = torch.zeros(N_g, N_g, device=device)
                for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                    A_dense[s, d] = 1.0
                deg = A_dense.sum(dim=1) + 1.0
                deg = deg / deg.mean()
                err = (X_pred - X_target).pow(2).mean(dim=-1)  # (T, N)
                loss_critical = (err * deg.unsqueeze(0)).mean()
                loss = loss + lam_critical * loss_critical
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()); nb += 1
        train_loss_curve.append(loss_sum / max(nb, 1))

    # Eval on test split
    test_dir = os.path.join(data_root, testbed, "test")
    test_files = sorted(os.listdir(test_dir))[:50]
    per_inst_metrics = []
    for fname in test_files:
        try:
            inst = torch.load(os.path.join(test_dir, fname), weights_only=False)
        except Exception:
            continue
        X_traj = torch.from_numpy(inst["trace_X"]).float().to(device)
        actions = torch.from_numpy(inst["trace_actions"]).float().to(device)
        edge_index = torch.from_numpy(np.array(inst["edge_index"], dtype=np.int64)).to(device)
        edge_type = torch.from_numpy(np.array(inst["edge_type"], dtype=np.int64)).to(device)
        node_types = torch.from_numpy(np.array(inst["node_types"], dtype=np.int64)).to(device)
        T_traj = X_traj.shape[0] - 1
        with torch.no_grad():
            X_pred = model.rollout_predict(X_traj[0:1], edge_index, edge_type,
                                            actions.unsqueeze(0), T=T_traj,
                                            node_types=node_types)[0].cpu().numpy()
        X_true = X_traj.cpu().numpy()
        N_g = X_true.shape[1]
        # README §6.14 metrics for agent / §6.25 for skill graph
        # NodeMSE@10
        H10 = min(10, T_traj)
        nm10 = float(np.mean((X_pred[H10] - X_true[H10]) ** 2))
        # EdgeF1@10: edge prediction not implemented for RGCN here; mark NaN
        # SR / FPD
        if testbed == "agent_calling_tree":
            sink = inst.get("oracle_answer_node")
            sr = float(X_pred[T_traj, sink, 0]) if (sink is not None and 0 <= sink < N_g) else float("nan")
            sr_true = float(X_true[T_traj, sink, 0]) if (sink is not None and 0 <= sink < N_g) else float("nan")
            # FPD from root inject
            A_dense = np.zeros((N_g, N_g), dtype=np.float32)
            for s, d in zip(inst["edge_index"][0].tolist(), inst["edge_index"][1].tolist()):
                A_dense[s, d] = 1.0
            err_at_T = (X_pred[T_traj, :, 5] > 0.5).astype(np.float32)
            fpd = failure_propagation_depth(A_dense, 0, err_at_T)
            cl = cost_latency(X_pred)
            per_inst_metrics.append({
                "instance": fname,
                "NodeMSE@10": nm10,
                "sr_at_sink_T": sr, "sr_true": sr_true,
                "fpd_from_root": fpd,
                "cost": cl["cost"], "latency": cl["latency"],
                "n_executed": cl["n_executed"],
            })
        else:  # platform_skill_graph
            skill_mask = (np.array(inst["node_types"]) == SKILL_NODE_TYPES["skill"])
            skill_sr = float(X_pred[T_traj, skill_mask, 0].mean()) if skill_mask.any() else float("nan")
            skill_sr_true = float(X_true[T_traj, skill_mask, 0].mean()) if skill_mask.any() else float("nan")
            lib_err = float(np.linalg.norm(X_pred[T_traj] - X_true[T_traj]))
            A_dense = np.zeros((N_g, N_g), dtype=np.float32)
            for s, d in zip(inst["edge_index"][0].tolist(), inst["edge_index"][1].tolist()):
                A_dense[s, d] = 1.0
            hub_skills = inst["critical_roles"].get("hub_skill", [])
            if hub_skills:
                hub = int(hub_skills[0])
                err = (X_pred[T_traj, :, 5] > 0.5).astype(np.float32)
                fpd = failure_propagation_depth(A_dense, hub, err)
            else:
                fpd = 0
            cl = cost_latency(X_pred)
            per_inst_metrics.append({
                "instance": fname,
                "NodeMSE@10": nm10,
                "skill_sr_T": skill_sr, "skill_sr_true": skill_sr_true,
                "library_err_T": lib_err,
                "fpd_from_hub_skill": fpd,
                "cost": cl["cost"], "latency": cl["latency"],
            })

    elapsed = time.time() - t0
    # Aggregate
    import pandas as pd
    df_inst = pd.DataFrame(per_inst_metrics)
    agg = {}
    for col in df_inst.columns:
        if col == "instance": continue
        try:
            agg[f"{col}_mean"] = float(df_inst[col].mean())
            agg[f"{col}_std"] = float(df_inst[col].std())
            agg[f"{col}_median"] = float(df_inst[col].median())
        except Exception:
            pass
    result = {
        "meta": {
            "baseline": baseline, "testbed": testbed, "seed": int(seed),
            "epochs": epochs, "lr": lr,
            "config": cfg,
            "use_critical_loss": use_critical_loss,
            "train_time_sec": float(elapsed),
            "device": str(device),
            "timestamp_jst": now_jst(),
            "stream": "B_hetero_trained",
        },
        "train_loss_curve": train_loss_curve,
        "test_aggregates": agg,
        "per_instance_metrics": per_inst_metrics,
    }
    out_path = os.path.join(out_dir, testbed, f"{baseline}_seed{seed}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    if save_ckpt:
        ck_dir = os.path.join(out_dir, "checkpoints", testbed)
        os.makedirs(ck_dir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "config": {"baseline": baseline, "testbed": testbed, "seed": seed,
                               "n_node_types": n_node_types, "n_edge_types": n_edge_types,
                               **cfg}},
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
        baseline, testbed, seed = job
        out_path = os.path.join(out_dir, testbed, f"{baseline}_seed{seed}.json")
        if skip_if_done(out_path):
            res_q.put(("skipped", gpu_id, {"baseline": baseline, "testbed": testbed,
                                            "seed": seed, "elapsed_sec": 0, "status": "skipped"}))
            continue
        t0 = time.time()
        try:
            train_one_hetero(baseline, testbed, seed,
                              data_root=data_root, out_dir=out_dir,
                              device=dev, epochs=epochs)
            status = "ok"
        except Exception:
            import traceback
            with open(log_path, "a") as f:
                f.write(f"[{now_jst()}] [GPU{gpu_id}] EXCEPTION ({baseline},{testbed},seed{seed}):\n")
                f.write(traceback.format_exc())
                f.write("\n")
            status = "error"
        elapsed = time.time() - t0
        res_q.put((status, gpu_id, {
            "baseline": baseline, "testbed": testbed, "seed": seed,
            "elapsed_sec": elapsed, "status": status, "ts": now_jst(),
        }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results", "agent_skill_trained"))
    parser.add_argument("--log_path", default=os.path.join(REPO_ROOT, "logs", "stream_b.log"))
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[2])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--testbeds", nargs="+", default=["agent_calling_tree", "platform_skill_graph"])
    args = parser.parse_args()
    print(f"[{now_jst()}] Stream B: hetero proper training start; gpus={args.gpu_ids}")
    jobs = []
    for bl in BASELINE_CONFIGS.keys():
        for tb in args.testbeds:
            for s in SEEDS:
                jobs.append((bl, tb, s))
    n_total = len(jobs)
    print(f"  total jobs: {n_total}")
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, "a") as f:
        f.write(f"[{now_jst()}] Stream B START n_jobs={n_total} gpus={args.gpu_ids}\n")
    job_q = mp.Queue(); res_q = mp.Queue()
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
        msg = (f"[{now_jst()}] [{n_done}/{n_total}] GPU{gpu} {info['baseline']:18s} "
               f"{info['testbed']:25s} seed{info['seed']} {status} ({info['elapsed_sec']:.0f}s)")
        print(msg)
        with open(args.log_path, "a") as f:
            f.write(msg + "\n")
    for p in procs:
        p.join(timeout=10)
        if p.is_alive(): p.terminate()
    elapsed = time.time() - t_start
    sum_path = os.path.join(REPO_ROOT, "results", "stream_b_summary.json")
    os.makedirs(os.path.dirname(sum_path), exist_ok=True)
    with open(sum_path, "w") as f:
        json.dump({"timestamp_jst": now_jst(), "n_total": n_total,
                   "n_ok": n_ok, "n_skipped": n_skipped, "n_err": n_err,
                   "elapsed_sec": elapsed, "jobs": summary}, f, indent=2, default=str)
    print(f"\n[{now_jst()}] STREAM B DONE {n_ok} ok / {n_skipped} skipped / {n_err} err, "
          f"{elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
