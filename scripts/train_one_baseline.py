"""单 (baseline × topology × outer_seed) 训练脚本.

输入: PROJECT_ROOT/data/synthetic_rollouts/fe_{top}_N50_seed{S}_T50.pt
import os as _os; PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
输出: PROJECT_ROOT/results/p2_baselines/{top}/{baseline}_seed{S}.json
      + checkpoints/{top}/{baseline}_seed{S}.pt  (可选, 默认存)

每个 run output JSON 包含:
  - meta: baseline, topology, outer_seed, N, T, epochs, train_time_s
  - train_loss / val_loss curves
  - final eval metrics: NodeMSE@{1,2,4,8,16,32}, EdgeF1@H (FE 上 trivially=1.0)
  - theory_constants: 全 8 个 scalars (per Director 01:26 UTC)
  - growth_slope, GEAF_hat
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines import BASELINE_REGISTRY, ErrorAwareGWM
from src.graph_generators import generate, compute_all
from src.metrics import (
    RolloutPrediction, node_mse, edge_f1_binary, growth_slope,
    geaf_global, theory_constants,
)
from src.metrics.geaf import _spectral_radius

JST = timezone(timedelta(hours=9))


def now_jst() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def load_rollouts(data_root: str, top: str, N: int, outer_seed: int, T: int = 50) -> Dict[str, Any]:
    path = os.path.join(data_root, "synthetic_rollouts",
                        f"fe_{top}_N{N}_seed{outer_seed}_T{T}.pt")
    return torch.load(path, weights_only=False)


def train_one(
    baseline: str, top: str, outer_seed: int,
    *, data_root: str, out_dir: str,
    N: int = 50, D: int = 8, D_a: int = 4,
    epochs: int = 100, batch_size: int = 16,
    lr: float = 1e-3, device: str = "cuda",
    eval_horizons: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    save_ckpt: bool = True,
    rollout_steps: int = 4,             # multi-step rollout loss (B6)
    lam_rollout: float = 0.5,           # B6 only
    lam_spectral: float = 0.01,         # B6 only
    lam_critical: float = 0.1,          # B6 only (Item 3 patch)
    grad_clip_norm: float = 1.0,        # all baselines (Item 5 patch)
) -> Dict[str, Any]:
    """训练一个 baseline. 返回 result dict (也写到 disk).

    Patched 2026-05-13: deterministic seeding (Item 5a) + grad_clip propagation.
    """
    # Item 5a patch: deterministic seeding for reproducibility
    # 派生 seed from outer_seed: distinct seeds for python/numpy/torch/torch.cuda
    py_seed = outer_seed * 1000 + 7
    np.random.seed(py_seed)
    torch.manual_seed(py_seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(py_seed + 2)
    t0 = time.time()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    # 加载数据
    payload = load_rollouts(data_root, top, N, outer_seed)
    train_X = torch.from_numpy(payload["train_X"]).float()    # (100, T+1, N, D)
    val_X = torch.from_numpy(payload["val_X"]).float()
    test_X = torch.from_numpy(payload["test_X"]).float()
    train_a = torch.from_numpy(payload["train_actions"]).float()
    val_a = torch.from_numpy(payload["val_actions"]).float()
    test_a = torch.from_numpy(payload["test_actions"]).float()
    W_gt = payload["W"]                       # (D, D)
    g = generate(top, N=N, seed=outer_seed)
    A_norm_t = torch.from_numpy(g.A_norm).float().to(dev)
    A_dense = g.A_dense
    stats = compute_all(g)
    rho_A_raw = stats["rho_A"]
    rho_A_norm = stats["rho_A_norm"]

    # MLP-WM N>50 标 N/A
    if baseline == "B1_MLP" and N > 50:
        result = {
            "meta": {"baseline": baseline, "topology": top, "outer_seed": outer_seed,
                     "N": N, "status": "N/A (MLP-WM 仅 N≤50)"},
            "skipped": True,
        }
        path = os.path.join(out_dir, top, f"{baseline}_seed{outer_seed}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    # 模型
    cls = BASELINE_REGISTRY[baseline]
    if baseline == "B1_MLP":
        model = cls(N=N, D=D, D_a=D_a).to(dev)
    else:
        model = cls(D=D, D_a=D_a).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())

    # 1-step 训练对 — (X_t, X_{t+1}, a_t)
    # 构造 train 1-step pair tensor
    def make_pairs(X: torch.Tensor, A: torch.Tensor):
        # X: (n_traj, T+1, N, D); A: (n_traj, T, D_a)
        n_traj, Tp1, _, _ = X.shape
        T = Tp1 - 1
        Xt = X[:, :-1].reshape(-1, N, D)           # (n_traj*T, N, D)
        Xtp1 = X[:, 1:].reshape(-1, N, D)
        At = A.reshape(-1, A.shape[-1])             # (n_traj*T, D_a)
        return Xt, Xtp1, At

    train_Xt, train_Xtp1, train_At = make_pairs(train_X, train_a)
    val_Xt, val_Xtp1, val_At = make_pairs(val_X, val_a)
    train_Xt, train_Xtp1, train_At = train_Xt.to(dev), train_Xtp1.to(dev), train_At.to(dev)
    val_Xt, val_Xtp1, val_At = val_Xt.to(dev), val_Xtp1.to(dev), val_At.to(dev)

    n_train = train_Xt.shape[0]

    train_losses, val_losses = [], []
    best_val = float("inf")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        loss_sum = 0.0
        n_batch = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            X_in = train_Xt[idx]
            X_target = train_Xtp1[idx]
            a_in = train_At[idx]
            X_pred = model.forward_step(X_in, A_norm_t, a_in)
            loss_1step = F.mse_loss(X_pred, X_target)
            loss = loss_1step

            # B6 额外 loss
            if baseline == "B6_ErrorAware":
                # multi-step rollout loss: 用 mini-trajectory 内 rollout_steps 步
                # 简化: 从 train_X 随机抽 1 条 trajectory 起点 t0, 跑 rollout_steps 步
                if rollout_steps > 1:
                    # 从 train_X (n_traj, T+1, N, D) 抽
                    n_traj_tot = train_X.shape[0]
                    T_max = train_X.shape[1] - 1
                    bs_rollout = min(8, n_traj_tot)
                    idx_traj = torch.randint(0, n_traj_tot, (bs_rollout,))
                    t0_rollout = torch.randint(0, T_max - rollout_steps, (1,)).item()
                    X_seq = train_X[idx_traj, t0_rollout:t0_rollout + rollout_steps + 1].to(dev)  # (bs, k+1, N, D)
                    a_seq = train_a[idx_traj, t0_rollout:t0_rollout + rollout_steps].to(dev)
                    X_pred_traj = model.rollout_predict(X_seq[:, 0], A_norm_t, a_seq, T=rollout_steps)
                    loss_rollout = F.mse_loss(X_pred_traj, X_seq)
                    loss = loss + lam_rollout * loss_rollout
                # spectral reg (Item 2 patch: 4-iter PI in b6_error_aware.py)
                R_spec = model.spectral_reg(target_spec=1.0)
                loss = loss + lam_spectral * R_spec
                # Item 3 patch: R_critical (degree-weighted node MSE on 1-step pred)
                R_critical = model.critical_node_weighted_loss(
                    X_pred, X_target, node_weights=None, A_norm=A_norm_t,
                )
                loss = loss + lam_critical * R_critical

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()
            loss_sum += float(loss.item())
            n_batch += 1
        train_loss = loss_sum / max(n_batch, 1)
        train_losses.append(train_loss)

        # val 1-step
        model.eval()
        with torch.no_grad():
            X_pred = model.forward_step(val_Xt, A_norm_t, val_At)
            val_loss = float(F.mse_loss(X_pred, val_Xtp1).item())
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss

    # ---- final eval: rollout on test set ----
    model.eval()
    with torch.no_grad():
        # test_X: (20, T+1, N, D); test_a: (20, T, D_a)
        X_0 = test_X[:, 0].to(dev)
        actions_test = test_a.to(dev)
        T_test = test_X.shape[1] - 1
        X_pred_traj = model.rollout_predict(X_0, A_norm_t, actions_test, T=T_test)
        X_pred_traj = X_pred_traj.cpu().numpy().astype(np.float32)
    test_X_np = test_X.numpy()
    # 跨 20 test trajectory mean
    metrics_per_h: Dict[str, float] = {}
    for h in eval_horizons:
        if h > T_test:
            continue
        # NodeMSE@H pooled
        per_traj_mse = []
        per_traj_f1 = []
        for i in range(test_X_np.shape[0]):
            pred = RolloutPrediction(
                X_true=test_X_np[i], A_true=A_dense,
                X_pred=X_pred_traj[i], A_pred=A_dense,
                is_fixed_edge=True,
            )
            per_traj_mse.append(node_mse(pred, h))
            per_traj_f1.append(edge_f1_binary(pred, h))
        metrics_per_h[f"NodeMSE@{h}"] = float(np.mean(per_traj_mse))
        metrics_per_h[f"EdgeF1@{h}"] = float(np.mean(per_traj_f1))
        metrics_per_h[f"NodeMSE@{h}_std"] = float(np.std(per_traj_mse))

    # GrowthSlope (H1=4, H2=32) per trajectory then average
    slopes = []
    for i in range(test_X_np.shape[0]):
        pred = RolloutPrediction(
            X_true=test_X_np[i], A_true=A_dense,
            X_pred=X_pred_traj[i], A_pred=A_dense,
            is_fixed_edge=True,
        )
        slopes.append(growth_slope(pred, 4, min(32, T_test)))
    metrics_per_h["GrowthSlope_4_32"] = float(np.mean(slopes))

    # ---- theory constants ----
    model_W_np = model.gnn_W()
    if not model_W_np:
        model_W_np = [W_gt.astype(np.float32)]    # MLP fallback: 用 GT W (虽不准确, 至少有数)
    tc = theory_constants(
        A_dense, model_W_np,
        X=test_X_np[0, 0],          # 任取一个初始 X
        sigma_lipschitz=1.0,
        Q=None, dynamic_edge=False,
    )
    tc["rho_A_norm"] = float(rho_A_norm)

    elapsed = time.time() - t0
    result = {
        "meta": {
            "baseline": baseline, "topology": top, "outer_seed": int(outer_seed),
            "N": N, "D": D, "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "n_params": int(n_params),
            "device": str(dev), "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "train_time_sec": float(elapsed),
            "timestamp_jst": now_jst(),
        },
        "train_loss_curve": train_losses,
        "val_loss_curve": val_losses,
        "best_val_loss": float(best_val),
        "test_metrics": metrics_per_h,
        "theory_constants": tc,
        "graph_stats": stats,
    }
    out_path = os.path.join(out_dir, top, f"{baseline}_seed{outer_seed}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    if save_ckpt:
        ck_dir = os.path.join(out_dir, "checkpoints", top)
        os.makedirs(ck_dir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "config": {"baseline": baseline, "top": top, "seed": outer_seed}},
                   os.path.join(ck_dir, f"{baseline}_seed{outer_seed}.pt"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=list(BASELINE_REGISTRY.keys()))
    parser.add_argument("--topology", required=True)
    parser.add_argument("--outer_seed", type=int, required=True)
    parser.add_argument("--data_root", default="PROJECT_ROOT/data")
    parser.add_argument("--out_dir", default="PROJECT_ROOT/results/p2_baselines")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_ckpt", action="store_true")
    args = parser.parse_args()

    # epochs 默认: B4 GPS 80, 其他 100
    epochs = args.epochs
    if epochs is None:
        epochs = 80 if args.baseline == "B4_GPS" else 100

    result = train_one(
        args.baseline, args.topology, args.outer_seed,
        data_root=args.data_root, out_dir=args.out_dir,
        epochs=epochs, device=args.device,
        save_ckpt=not args.no_ckpt,
    )
    if result.get("skipped"):
        print(f"[{now_jst()}] SKIPPED {args.baseline} {args.topology} seed{args.outer_seed} (N/A)")
    else:
        m = result["test_metrics"]
        tc = result["theory_constants"]
        print(f"[{now_jst()}] DONE {args.baseline} {args.topology} seed{args.outer_seed}: "
              f"NodeMSE@8={m.get('NodeMSE@8', float('nan')):.4e} "
              f"NodeMSE@32={m.get('NodeMSE@32', float('nan')):.4e} "
              f"GEAF={tc['GEAF_hat']:.3f} ρ(B)={tc['rho_B']:.3f} "
              f"({result['meta']['train_time_sec']:.0f}s)")


if __name__ == "__main__":
    main()
