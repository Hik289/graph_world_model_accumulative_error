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
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines import BASELINE_REGISTRY
from src.graph_generators import generate, compute_all
from src.simulators import rollout
from src.metrics import (
    RolloutPrediction, node_mse, return_error, geaf_global, theory_constants,
)

JST = timezone(timedelta(hours=9))
BASELINES_P3 = ["B2_GCN", "B5_ActionNode", "B6_ErrorAware"]
SEEDS = [1, 2, 3]
N = 50
D = 8
D_a = 4


def now_jst():
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def _gen_or_load_er(N: int, p: float, seed: int):

    import networkx as nx
    from src.graph_generators.base import GraphSample, _normalize_adj, _annotate_critical_roles
    import scipy.sparse as sp
    G = nx.erdos_renyi_graph(N, p=p, seed=seed)
    if not nx.is_connected(G):
        cc = list(nx.connected_components(G))
        for i in range(len(cc) - 1):
            a = next(iter(cc[i]))
            b = next(iter(cc[i + 1]))
            G.add_edge(a, b)
    A = nx.to_numpy_array(G, dtype=np.float32)
    np.fill_diagonal(A, 0.0)
    A_norm = _normalize_adj(A)
    sample = GraphSample(
        A_dense=A, A_sparse=sp.csr_matrix(A), A_norm=A_norm,
        is_directed=False, N=N, topology="er",
        params={"p": float(p)}, seed=seed,
    )
    sample.critical_roles = _annotate_critical_roles(A, "er", seed)
    return sample


def _train_one_unified(
    baseline: str, graph_sample, variant_name: str,
    *, epochs: int, device: torch.device, out_path: str,
    T_train: int = 50, sigma_noise: float = 0.01,
    is_directed: bool = False,
):


    t0 = time.time()
    g = graph_sample
    N_g = g.N
    A_norm_t = torch.from_numpy(g.A_norm).float().to(device)
    A_dense = g.A_dense

    train_X, train_a = [], []
    val_X, val_a = [], []
    test_X, test_a = [], []
    W_shared = None
    U_shared = None
    for split, n_traj, start in [("train", 50, 0), ("val", 10, 100), ("test", 10, 120)]:
        for idx in range(n_traj):
            inner = start + idx
            tr = rollout(g, T=T_train, mode="fixed_edge",
                         sigma_noise=sigma_noise, seed=inner)
            if W_shared is None:
                W_shared, U_shared = tr.W.copy(), tr.U.copy()
            if split == "train":
                train_X.append(tr.X); train_a.append(tr.actions)
            elif split == "val":
                val_X.append(tr.X); val_a.append(tr.actions)
            else:
                test_X.append(tr.X); test_a.append(tr.actions)
    train_X = np.stack(train_X); train_a = np.stack(train_a)
    val_X = np.stack(val_X); val_a = np.stack(val_a)
    test_X = np.stack(test_X); test_a = np.stack(test_a)

    cls = BASELINE_REGISTRY[baseline]
    if baseline == "B1_MLP":
        model = cls(N=N_g)
    else:
        model = cls()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters())


    Xt = torch.from_numpy(train_X[:, :-1]).float().reshape(-1, N_g, D).to(device)
    Xtp1 = torch.from_numpy(train_X[:, 1:]).float().reshape(-1, N_g, D).to(device)
    At = torch.from_numpy(train_a).float().reshape(-1, D_a).to(device)
    valXt = torch.from_numpy(val_X[:, :-1]).float().reshape(-1, N_g, D).to(device)
    valXtp1 = torch.from_numpy(val_X[:, 1:]).float().reshape(-1, N_g, D).to(device)
    valAt = torch.from_numpy(val_a).float().reshape(-1, D_a).to(device)
    n_pairs = Xt.shape[0]
    bs = 16
    train_curve, val_curve = [], []
    best_val = float("inf")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_pairs, device=device)
        loss_sum = 0; nb = 0
        for i in range(0, n_pairs, bs):
            idx = perm[i:i+bs]
            X_pred = model.forward_step(Xt[idx], A_norm_t, At[idx])
            loss = F.mse_loss(X_pred, Xtp1[idx])
            if baseline == "B6_ErrorAware":
                R_spec = model.spectral_reg(target_spec=1.0)
                loss = loss + 0.01 * R_spec
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()); nb += 1
        train_curve.append(loss_sum / max(nb, 1))
        model.eval()
        with torch.no_grad():
            vP = model.forward_step(valXt, A_norm_t, valAt)
            vl = float(F.mse_loss(vP, valXtp1).item())
        val_curve.append(vl)
        if vl < best_val: best_val = vl


    model.eval()
    with torch.no_grad():
        X0 = torch.from_numpy(test_X[:, 0]).float().to(device)
        a_seq = torch.from_numpy(test_a).float().to(device)
        T_test = test_X.shape[1] - 1
        X_pred = model.rollout_predict(X0, A_norm_t, a_seq, T=T_test).cpu().numpy()
    eval_h = [1, 2, 4, 8, 16, 32]
    metrics = {}
    for h in eval_h:
        if h > T_test: continue
        per_nm = []
        for i in range(test_X.shape[0]):
            per_nm.append(float(np.mean((X_pred[i, h] - test_X[i, h]) ** 2)))
        metrics[f"NodeMSE@{h}"] = float(np.mean(per_nm))
        metrics[f"NodeMSE@{h}_std"] = float(np.std(per_nm))

    model_W_np = model.gnn_W()
    if not model_W_np:
        model_W_np = [W_shared.astype(np.float32)]
    tc = theory_constants(A_dense, model_W_np, X=test_X[0, 0],
                          sigma_lipschitz=1.0, Q=None, dynamic_edge=False)
    tc["rho_A_norm"] = float(compute_all(g)["rho_A_norm"])

    result = {
        "meta": {
            "baseline": baseline, "variant": variant_name,
            "seed": int(g.seed), "N": int(N_g), "D": D,
            "epochs": epochs, "n_params": int(n_params),
            "is_directed": bool(is_directed),
            "device": str(device),
            "train_time_sec": float(time.time() - t0),
            "timestamp_jst": now_jst(),
        },
        "train_loss_curve": train_curve,
        "val_loss_curve": val_curve,
        "best_val_loss": float(best_val),
        "test_metrics": metrics,
        "theory_constants": tc,
        "graph_stats": compute_all(g),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    return result


def build_exp22_jobs() -> List[Tuple[str, str, int, Dict[str, Any]]]:

    jobs = []

    for p in [0.02, 0.05, 0.10, 0.20]:
        for bl in BASELINES_P3:
            for s in SEEDS:
                jobs.append((bl, f"er_p{p:.2f}", s,
                             {"kind": "er", "p": p, "topology": "er"}))

    for k in [2, 4, 6, 8]:
        for bl in BASELINES_P3:
            for s in SEEDS:
                jobs.append((bl, f"sw_k{k}", s,
                             {"kind": "small_world", "k": k, "p": 0.1,
                              "topology": "small_world"}))
    return jobs


def build_exp23_jobs() -> List[Tuple[str, str, int, Dict[str, Any]]]:
    jobs = []
    for top in ["chain", "tree", "scale_free"]:
        for directed in [False, True]:
            for bl in BASELINES_P3:
                for s in SEEDS:
                    suffix = "directed" if directed else "undirected"
                    jobs.append((bl, f"{top}_{suffix}", s,
                                 {"kind": top, "directed": directed,
                                  "topology": top}))
    return jobs


def build_exp24_jobs() -> List[Tuple[str, str, int, Dict[str, Any]]]:


    jobs = []
    for top in ["scale_free", "small_world"]:
        for mem in ["markov", "last2", "last4"]:
            for s in SEEDS:
                jobs.append(("B3_MPNN", f"{top}_mem_{mem}", s,
                             {"kind": top, "memory": mem, "topology": top}))
    return jobs


def _make_graph_from_params(params: Dict[str, Any], seed: int):
    if params["kind"] == "er":
        return _gen_or_load_er(N=N, p=params["p"], seed=seed)
    elif params["kind"] == "small_world":
        return generate("small_world", N=N, seed=seed,
                        k=params.get("k", 4), p=params.get("p", 0.1))
    elif params["kind"] in ["chain", "tree", "scale_free"]:
        directed = params.get("directed", False)
        kwargs = {}
        if params["kind"] == "tree":
            kwargs["variant"] = "balanced_binary"
        return generate(params["kind"], N=N, seed=seed, directed=directed, **kwargs)
    else:
        raise ValueError(params)


def worker_fn(gpu_id: int, job_q: mp.Queue, res_q: mp.Queue,
              out_dir: str, log_path: str, epochs: int):
    sys.path.insert(0, REPO_ROOT)
    import torch
    if not torch.cuda.is_available():
        res_q.put(("unavailable", gpu_id, None))
        return
    dev = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(dev)
    while True:
        try:
            job = job_q.get(timeout=10)
        except queue_mod.Empty:
            continue
        if job is None:
            res_q.put(("done", gpu_id, None))
            return
        exp_id, baseline, variant, seed, params = job
        t0 = time.time()
        try:
            g = _make_graph_from_params(params, seed=seed)
            out_path = os.path.join(out_dir, f"exp{exp_id}",
                                    variant, f"{baseline}_seed{seed}.json")
            res = _train_one_unified(
                baseline, g, variant,
                epochs=epochs, device=dev, out_path=out_path,
                is_directed=params.get("directed", False),
            )
            status = "ok"
        except Exception as e:
            import traceback
            with open(log_path, "a") as f:
                f.write(f"[{now_jst()}] [GPU{gpu_id}] EXCEPTION exp{exp_id} {baseline} {variant} seed{seed}:\n")
                f.write(traceback.format_exc())
                f.write("\n")
            status = "error"
        elapsed = time.time() - t0
        res_q.put((status, gpu_id, {
            "exp": exp_id, "baseline": baseline, "variant": variant,
            "seed": seed, "elapsed_sec": elapsed,
            "status": status, "ts": now_jst(),
        }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, 'results', 'p3_training'))
    parser.add_argument("--log_path", default=os.path.join(REPO_ROOT, 'logs', 'p3_training.log'))
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exps", nargs="+", choices=["22", "23", "24"], default=["22", "23", "24"])
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this training batch.")
    if len(set(args.gpu_ids)) != len(args.gpu_ids):
        parser.error("--gpu_ids must contain distinct CUDA device indices.")
    if any(gpu_id < 0 or gpu_id >= torch.cuda.device_count() for gpu_id in args.gpu_ids):
        parser.error("--gpu_ids contains an unavailable CUDA device index.")
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.log_path)), exist_ok=True)
    print(f"[{now_jst()}] P3 training batch 启动; exps={args.exps} gpus={args.gpu_ids}")

    jobs = []
    if "22" in args.exps:
        for j in build_exp22_jobs():
            jobs.append(("22", *j))
    if "23" in args.exps:
        for j in build_exp23_jobs():
            jobs.append(("23", *j))
    if "24" in args.exps:
        for j in build_exp24_jobs():
            jobs.append(("24", *j))


    if args.skip_existing:
        filtered = []
        for j in jobs:
            exp_id, bl, var, sd, _ = j
            path = os.path.join(args.out_dir, f"exp{exp_id}", var, f"{bl}_seed{sd}.json")
            if os.path.exists(path):
                continue
            filtered.append(j)
        jobs = filtered
    n_total = len(jobs)
    print(f"  total jobs: {n_total}")

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
                             args.out_dir, args.log_path, args.epochs))
        p.start()
        procs.append(p)

    n_done = n_ok = n_err = 0
    summary = []
    pending_gpus = set(args.gpu_ids)
    t_start = time.time()
    try:
        while pending_gpus:
            failed = [(gpu_id, proc.exitcode) for gpu_id, proc in zip(args.gpu_ids, procs)
                      if proc.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(f"Training workers exited unsuccessfully: {failed}")
            try:
                status, gpu_id, info = res_q.get(timeout=5)
            except queue_mod.Empty:
                missing = [gpu_id for gpu_id, proc in zip(args.gpu_ids, procs)
                           if gpu_id in pending_gpus and not proc.is_alive()]
                if missing:
                    raise RuntimeError(f"Training workers exited without completion: {missing}")
                continue
            if status == "unavailable":
                raise RuntimeError(f"CUDA device {gpu_id} is unavailable to the worker.")
            if status == "done":
                pending_gpus.remove(gpu_id)
                continue
            if status not in ("ok", "error") or info is None:
                raise RuntimeError(f"Unexpected worker response: {status!r}")
            n_done += 1
            if status == "ok": n_ok += 1
            else: n_err += 1
            summary.append(info)
            msg = (f"[{now_jst()}] [{n_done}/{n_total}] GPU{gpu_id} "
                   f"exp{info['exp']} {info['baseline']:14s} {info['variant']:20s} "
                   f"seed{info['seed']} {status} ({info['elapsed_sec']:.0f}s)")
            print(msg)
            with open(args.log_path, "a") as f:
                f.write(msg + "\n")
        for gpu_id, proc in zip(args.gpu_ids, procs):
            proc.join(timeout=10)
            if proc.is_alive() or proc.exitcode != 0:
                raise RuntimeError(f"Training worker {gpu_id} failed to exit cleanly: {proc.exitcode}")
        if n_done != n_total:
            raise RuntimeError(f"Incomplete training batch: received {n_done} of {n_total} job results.")
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=10)
        job_q.cancel_join_thread()
        job_q.close()
        res_q.close()

    print(f"\n[{now_jst()}] BATCH FINISHED {n_ok} ok / {n_err} err, "
          f"{(time.time()-t_start)/60:.1f} min")
    sum_path = os.path.join(args.out_dir, "p3_training_summary.json")
    os.makedirs(os.path.dirname(sum_path), exist_ok=True)
    with open(sum_path, "w") as f:
        json.dump({"timestamp_jst": now_jst(), "n_total": n_total,
                   "n_ok": n_ok, "n_err": n_err,
                   "jobs": summary}, f, indent=2, default=str)
    if n_err:
        raise RuntimeError(f"{n_err} training jobs failed; see {args.log_path}.")


if __name__ == "__main__":
    main()
