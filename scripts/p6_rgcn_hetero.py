from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines.rgcn_hetero import RGCNHetero
from src.simulators.agent_calling_tree import NODE_TYPES, EDGE_TYPES, D_FEAT, D_ACTION
from src.simulators.platform_skill_graph import SKILL_NODE_TYPES, SKILL_EDGE_TYPES
from src.metrics import failure_propagation_depth
from src.utils.seeding import stable_seed
from scripts._runner_utils import skip_if_done, now_jst

CEIL = 1e10


def train_rgcn(testbed: str, data_root: str, out_dir: str,
               dev: torch.device, n_seeds: int = 3, epochs: int = 50,
               n_train: int = 100):


    if testbed == "agent_calling_tree":
        n_node_types = len(NODE_TYPES)
        n_edge_types = len(EDGE_TYPES)
    elif testbed == "platform_skill_graph":
        n_node_types = len(SKILL_NODE_TYPES)
        n_edge_types = len(SKILL_EDGE_TYPES)
    else:
        raise ValueError(f"Unknown testbed={repr(testbed)}")
    print(f"[train_rgcn] {testbed}: n_node_types={n_node_types}, n_edge_types={n_edge_types}")
    train_dir = os.path.join(data_root, testbed, "train")
    files = sorted(name for name in os.listdir(train_dir) if name.endswith(".pt"))[:n_train]
    print(f"  using {len(files)} train instances")

    for seed in range(1, n_seeds + 1):
        out_path = os.path.join(out_dir, "checkpoints", testbed, f"rgcn_seed{seed}.pt")
        if skip_if_done(out_path):
            print(f"  skip seed{seed} (ckpt exists)")
            continue
        torch.manual_seed(seed * 1000 + 7)
        np.random.seed(seed * 1000 + 8)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 1000 + 9)
        model = RGCNHetero(D=D_FEAT, D_a=D_ACTION,
                           n_node_types=n_node_types,
                           n_edge_types=n_edge_types,
                           hidden=64, n_layers=2).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_curve = []
        for ep in range(epochs):
            model.train()
            loss_sum = 0; n_batch = 0

            np.random.shuffle(files)
            for fname in files:
                try:
                    inst = torch.load(os.path.join(train_dir, fname), weights_only=False)
                except Exception:
                    continue
                X_traj = torch.from_numpy(inst["trace_X"]).float()
                actions = torch.from_numpy(inst["trace_actions"]).float()
                edge_index = torch.from_numpy(np.array(inst["edge_index"], dtype=np.int64))
                edge_type = torch.from_numpy(np.array(inst["edge_type"], dtype=np.int64))
                node_types = torch.from_numpy(np.array(inst["node_types"], dtype=np.int64))
                T_traj = X_traj.shape[0] - 1

                X_traj = X_traj.to(dev)
                actions = actions.to(dev)
                edge_index = edge_index.to(dev)
                edge_type = edge_type.to(dev)
                node_types = node_types.to(dev)

                X_in_batch = X_traj[:-1]
                X_target_batch = X_traj[1:]
                a_in_batch = actions
                X_pred = model.forward_step(
                    X_in_batch, edge_index, edge_type, a_in_batch,
                    node_types=node_types,
                )
                loss = F.mse_loss(X_pred, X_target_batch)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_sum += float(loss.item()); n_batch += 1
            train_curve.append(loss_sum / max(n_batch, 1))
            if ep % 10 == 0:
                print(f"  seed{seed} ep{ep}: train_loss={train_curve[-1]:.4e}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "config": {"D": D_FEAT, "D_a": D_ACTION,
                       "n_node_types": n_node_types,
                       "n_edge_types": n_edge_types,
                       "hidden": 64, "n_layers": 2,
                       "testbed": testbed, "seed": seed,
                       "n_train": len(files), "epochs": epochs},
            "train_loss_curve": train_curve,
        }, out_path)
        print(f"  saved {out_path}")


def eval_exp14_model(data_root: str, ckpt_dir: str, out_dir: str,
                     dev: torch.device, n_test: int = 50, H_eval: int = 20,
                     n_seeds: int = 3):
    print(f"[Exp 14 model] R-GCN-Agent on agent_calling_tree test")
    out_path = os.path.join(out_dir, "exp14_model_rollout.csv")
    if skip_if_done(out_path):
        print(f"  Exp 14 model: skip (exists)"); return out_path
    test_dir = os.path.join(data_root, "agent_calling_tree", "test")
    files = sorted(name for name in os.listdir(test_dir) if name.endswith(".pt"))[:n_test]
    rows = []
    for seed in range(1, n_seeds + 1):
        ckpt_path = os.path.join(ckpt_dir, "checkpoints", "agent_calling_tree",
                                  f"rgcn_seed{seed}.pt")
        if not os.path.exists(ckpt_path):
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        rgcn = RGCNHetero(D=cfg["D"], D_a=cfg["D_a"],
                          n_node_types=cfg["n_node_types"],
                          n_edge_types=cfg["n_edge_types"],
                          hidden=cfg["hidden"], n_layers=cfg["n_layers"]).to(dev)
        rgcn.load_state_dict(ck["state_dict"])
        rgcn.eval()


        for fname in files:
            try:
                inst = torch.load(os.path.join(test_dir, fname), weights_only=False)
            except Exception:
                continue
            X_traj = torch.from_numpy(inst["trace_X"]).float().to(dev)
            actions = torch.from_numpy(inst["trace_actions"]).float().to(dev)
            edge_index = torch.from_numpy(np.array(inst["edge_index"], dtype=np.int64)).to(dev)
            edge_type = torch.from_numpy(np.array(inst["edge_type"], dtype=np.int64)).to(dev)
            node_types = torch.from_numpy(np.array(inst["node_types"], dtype=np.int64)).to(dev)
            T_traj = X_traj.shape[0] - 1
            H = min(H_eval, T_traj)

            with torch.no_grad():
                X0 = X_traj[0:1]
                a_in = actions.unsqueeze(0)
                X_pred_rgcn = rgcn.rollout_predict(X0, edge_index, edge_type, a_in,
                                                    T=T_traj, node_types=node_types)
                X_pred_rgcn = X_pred_rgcn[0].cpu().numpy()
            X_true = X_traj.cpu().numpy()
            nm_h_rgcn = float(np.mean((X_pred_rgcn[H] - X_true[H]) ** 2))
            sink = inst.get("oracle_answer_node")
            sr_rgcn = float(X_pred_rgcn[T_traj, sink, 0]) if sink is not None and 0 <= sink < X_traj.shape[1] else float("nan")
            sr_true = float(X_true[T_traj, sink, 0]) if sink is not None and 0 <= sink < X_traj.shape[1] else float("nan")

            err_rgcn = (X_pred_rgcn[T_traj, :, 5] > 0.5).astype(np.float32)
            err_true = (X_true[T_traj, :, 5] > 0.5).astype(np.float32)

            N_g = X_traj.shape[1]
            A_dense = np.zeros((N_g, N_g), dtype=np.float32)
            for s, d in zip(inst["edge_index"][0].tolist(), inst["edge_index"][1].tolist()):
                A_dense[s, d] = 1.0
            inj_node = 0
            fpd_rgcn = failure_propagation_depth(A_dense, inj_node, err_rgcn)
            fpd_true = failure_propagation_depth(A_dense, inj_node, err_true)
            rows.append({
                "instance": fname, "rgcn_seed": seed,
                "T": T_traj, "H_eval": H, "N": N_g,
                "rgcn_NodeMSE@H": nm_h_rgcn,
                "rgcn_sr_at_sink_T": sr_rgcn,
                "true_sr_at_sink_T": sr_true,
                "rgcn_fpd": fpd_rgcn,
                "true_fpd": fpd_true,
                "rgcn_n_err_T": int(err_rgcn.sum()),
                "true_n_err_T": int(err_true.sum()),
            })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Exp 14 model: {len(rows)} rows → {out_path}")
    return out_path


def eval_exp25_model(data_root: str, ckpt_dir: str, out_dir: str,
                     dev: torch.device, n_test: int = 30, n_seeds: int = 3):
    print(f"[Exp 25 model] R-GCN-SkillGraph on skill_graph test")
    out_path = os.path.join(out_dir, "exp25_model_rollout.csv")
    if skip_if_done(out_path):
        print(f"  Exp 25 model: skip (exists)"); return out_path
    test_dir = os.path.join(data_root, "platform_skill_graph", "test")
    files = sorted(name for name in os.listdir(test_dir) if name.endswith(".pt"))[:n_test]
    rows = []
    for seed in range(1, n_seeds + 1):
        ckpt_path = os.path.join(ckpt_dir, "checkpoints", "platform_skill_graph",
                                  f"rgcn_seed{seed}.pt")
        if not os.path.exists(ckpt_path):
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        rgcn = RGCNHetero(D=cfg["D"], D_a=cfg["D_a"],
                          n_node_types=cfg["n_node_types"],
                          n_edge_types=cfg["n_edge_types"],
                          hidden=cfg["hidden"], n_layers=cfg["n_layers"]).to(dev)
        rgcn.load_state_dict(ck["state_dict"])
        rgcn.eval()
        for fname in files:
            try:
                inst = torch.load(os.path.join(test_dir, fname), weights_only=False)
            except Exception:
                continue
            X_traj = torch.from_numpy(inst["trace_X"]).float().to(dev)
            actions = torch.from_numpy(inst["trace_actions"]).float().to(dev)
            edge_index = torch.from_numpy(np.array(inst["edge_index"], dtype=np.int64)).to(dev)
            edge_type = torch.from_numpy(np.array(inst["edge_type"], dtype=np.int64)).to(dev)
            node_types = torch.from_numpy(np.array(inst["node_types"], dtype=np.int64)).to(dev)
            T_traj = X_traj.shape[0] - 1
            with torch.no_grad():
                X0 = X_traj[0:1]
                a_in = actions.unsqueeze(0)
                X_pred = rgcn.rollout_predict(X0, edge_index, edge_type, a_in,
                                               T=T_traj, node_types=node_types)
                X_pred = X_pred[0].cpu().numpy()
            X_true = X_traj.cpu().numpy()

            skill_mask = (np.array(inst["node_types"]) == SKILL_NODE_TYPES["skill"])
            skill_sr_rgcn = float(X_pred[T_traj, skill_mask, 0].mean()) if skill_mask.any() else float("nan")
            skill_sr_true = float(X_true[T_traj, skill_mask, 0].mean()) if skill_mask.any() else float("nan")
            lib_err_rgcn = float(np.linalg.norm(X_pred[T_traj] - X_true[T_traj]))

            hub_skills = inst["critical_roles"].get("hub_skill", [])
            if hub_skills:
                hub = int(hub_skills[0])
                N_g = X_traj.shape[1]
                A_dense = np.zeros((N_g, N_g), dtype=np.float32)
                for s, d in zip(inst["edge_index"][0].tolist(), inst["edge_index"][1].tolist()):
                    A_dense[s, d] = 1.0
                err = (X_pred[T_traj, :, 5] > 0.5).astype(np.float32)
                fpd = failure_propagation_depth(A_dense, hub, err)
            else:
                fpd = 0
            rows.append({
                "instance": fname, "rgcn_seed": seed,
                "T": T_traj, "N": X_traj.shape[1],
                "rgcn_skill_sr_T": skill_sr_rgcn,
                "true_skill_sr_T": skill_sr_true,
                "rgcn_lib_err_T": lib_err_rgcn,
                "rgcn_fpd_hub": fpd,
                "n_skill_nodes": int(skill_mask.sum()),
            })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Exp 25 model: {len(rows)} rows → {out_path}")
    return out_path


def eval_exp16_model(data_root: str, ckpt_dir: str, out_dir: str,
                     dev: torch.device, n_test: int = 30, n_seeds: int = 3):
    print(f"[Exp 16 model] correction policies via R-GCN-Agent rollout")
    out_path = os.path.join(out_dir, "exp16_model_correction.csv")
    if skip_if_done(out_path):
        print(f"  Exp 16 model: skip (exists)"); return out_path
    test_dir = os.path.join(data_root, "agent_calling_tree", "test")
    files = sorted(name for name in os.listdir(test_dir) if name.endswith(".pt"))[:n_test]
    rows = []
    policies = ["random", "degree", "GEAF_proxy", "oracle"]
    for seed in range(1, n_seeds + 1):
        ckpt_path = os.path.join(ckpt_dir, "checkpoints", "agent_calling_tree",
                                  f"rgcn_seed{seed}.pt")
        if not os.path.exists(ckpt_path):
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        rgcn = RGCNHetero(D=cfg["D"], D_a=cfg["D_a"],
                          n_node_types=cfg["n_node_types"],
                          n_edge_types=cfg["n_edge_types"],
                          hidden=cfg["hidden"], n_layers=cfg["n_layers"]).to(dev)
        rgcn.load_state_dict(ck["state_dict"])
        rgcn.eval()
        for fname in files:
            try:
                inst = torch.load(os.path.join(test_dir, fname), weights_only=False)
            except Exception:
                continue
            X_traj = torch.from_numpy(inst["trace_X"]).float().to(dev)
            actions = torch.from_numpy(inst["trace_actions"]).float().to(dev)
            edge_index = torch.from_numpy(np.array(inst["edge_index"], dtype=np.int64)).to(dev)
            edge_type = torch.from_numpy(np.array(inst["edge_type"], dtype=np.int64)).to(dev)
            node_types = torch.from_numpy(np.array(inst["node_types"], dtype=np.int64)).to(dev)
            T_traj = X_traj.shape[0] - 1
            with torch.no_grad():
                X_pred = rgcn.rollout_predict(X_traj[0:1], edge_index, edge_type,
                                               actions.unsqueeze(0), T=T_traj,
                                               node_types=node_types)[0].cpu().numpy()
            X_true = X_traj.cpu().numpy()
            err_per_node = np.mean((X_pred[T_traj] - X_true[T_traj]) ** 2, axis=-1)
            N_g = X_traj.shape[1]
            budget = max(1, int(0.10 * N_g))
            A_dense = np.zeros((N_g, N_g), dtype=np.float32)
            for s, d in zip(inst["edge_index"][0].tolist(), inst["edge_index"][1].tolist()):
                A_dense[s, d] = 1.0
            deg = A_dense.sum(axis=1) + A_dense.sum(axis=0)
            rng = np.random.default_rng(stable_seed(fname, seed))
            policy_sets = {
                "random": rng.permutation(N_g)[:budget],
                "degree": np.argsort(deg)[-budget:],
                "GEAF_proxy": np.argsort(deg)[-budget:],
                "oracle": np.argsort(err_per_node)[-budget:],
            }
            baseline_err = float(err_per_node.mean())
            for policy, nodes in policy_sets.items():
                err_corrected = err_per_node.copy()
                err_corrected[nodes] = 0.0
                corrected_err = float(err_corrected.mean())
                rows.append({
                    "instance": fname, "rgcn_seed": seed, "policy": policy,
                    "budget_pct": 10.0,
                    "baseline_err": baseline_err,
                    "corrected_err": corrected_err,
                    "reduction": baseline_err - corrected_err,
                    "reduction_pct": 100 * (baseline_err - corrected_err) / max(baseline_err, 1e-12),
                })
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Exp 16 model: {len(rows)} rows → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, 'data'))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, 'results', 'p6_rgcn_hetero'))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase", default="all", choices=["train", "eval", "all"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_train", type=int, default=100)
    args = parser.parse_args()
    if args.epochs <= 0 or args.n_seeds <= 0 or args.n_train <= 0:
        parser.error("--epochs, --n_seeds, and --n_train must be positive")
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    print(f"[{now_jst()}] R-GCN hetero pipeline start; device={dev}; phase={args.phase}")

    if args.phase in ("train", "all"):
        for testbed in ("agent_calling_tree", "platform_skill_graph"):
            train_rgcn(testbed, args.data_root, args.out_dir, dev,
                       n_seeds=args.n_seeds, epochs=args.epochs,
                       n_train=args.n_train)

    if args.phase in ("eval", "all"):
        eval_exp14_model(args.data_root, args.out_dir, args.out_dir, dev,
                         n_test=50, H_eval=20, n_seeds=args.n_seeds)
        eval_exp25_model(args.data_root, args.out_dir, args.out_dir, dev,
                         n_test=30, n_seeds=args.n_seeds)
        eval_exp16_model(args.data_root, args.out_dir, args.out_dir, dev,
                         n_test=30, n_seeds=args.n_seeds)

    print(f"\n[{now_jst()}] R-GCN hetero pipeline DONE")


if __name__ == "__main__":
    main()
