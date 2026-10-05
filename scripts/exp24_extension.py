from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines import BASELINE_REGISTRY
from src.graph_generators import generate
from scripts._runner_utils import skip_if_done, now_jst

CEIL = 1e10
N = 50
D = 8
D_a = 4


class HistoryAggregator(nn.Module):


    def __init__(self, kind: str, D: int = 8, hidden: int = 16, k: int = 4):
        super().__init__()
        self.kind = kind
        self.k = k
        if kind == "recurrent":
            self.gru = nn.GRU(D, hidden, num_layers=1, batch_first=True)
            self.proj = nn.Linear(hidden, D)
        elif kind == "transformer":
            self.attn = nn.MultiheadAttention(D, num_heads=2, batch_first=True)
        elif kind == "retrieval":

            pass

    def forward(self, history: torch.Tensor) -> torch.Tensor:


        B, k, N_, D_ = history.shape
        if self.kind == "recurrent":

            h_in = history.permute(0, 2, 1, 3).reshape(B * N_, k, D_)
            out, _ = self.gru(h_in)
            x_agg = self.proj(out[:, -1, :])
            return x_agg.reshape(B, N_, D_)
        elif self.kind == "transformer":

            h_in = history.permute(0, 2, 1, 3).reshape(B * N_, k, D_)
            q = h_in[:, -1:, :]
            attn_out, _ = self.attn(q, h_in, h_in)
            return attn_out.squeeze(1).reshape(B, N_, D_)
        elif self.kind == "retrieval":

            return history.mean(dim=1)
        else:
            raise ValueError(self.kind)


def exp24_ext(out_dir, dev, data_root=os.path.join(REPO_ROOT, 'data'),
              p2_dir=os.path.join(REPO_ROOT, 'results', 'p2_baselines')):
    print(f"[{now_jst()}] Exp 24 ext start; device={dev}")
    out_path = os.path.join(out_dir, "exp24_ext.csv")
    if skip_if_done(out_path):
        print(f"  Exp 24 ext: skip (exists)")
        return {"exp": "24_ext", "n_rows": "skipped", "out": out_path}
    rows = []
    horizons = [1, 2, 4, 8, 16, 32]
    variants = ["recurrent", "transformer", "retrieval"]
    topos = ["scale_free", "small_world"]
    seeds = [1, 2, 3]
    history_k = 4

    for variant in variants:
        agg = HistoryAggregator(variant, D=D, hidden=16, k=history_k).to(dev).eval()
        for topo in topos:
            for seed in seeds:

                ck_path = os.path.join(p2_dir, "checkpoints", topo,
                                        f"B3_MPNN_seed{seed}.pt")
                if not os.path.exists(ck_path):
                    continue
                cls = BASELINE_REGISTRY["B3_MPNN"]
                model = cls(D=D, D_a=D_a).to(dev)
                try:
                    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
                    model.load_state_dict(ck["state_dict"])
                except Exception:
                    continue
                model.eval()

                rollout_path = os.path.join(data_root, "synthetic_rollouts",
                                             f"fe_{topo}_N{N}_seed{seed}_T50.pt")
                if not os.path.exists(rollout_path):
                    continue
                payload = torch.load(rollout_path, weights_only=False)
                test_X = payload["test_X"]
                test_a = payload["test_actions"]
                g = generate(topo, N=N, seed=seed)
                A_norm_t = torch.from_numpy(g.A_norm).float().to(dev)
                T_traj = test_X.shape[1] - 1


                per_h_mse = {h: [] for h in horizons}
                for i in range(test_X.shape[0]):
                    X_traj_pred = [torch.from_numpy(test_X[i, 0]).float().to(dev)]
                    a_seq = torch.from_numpy(test_a[i]).float().to(dev)
                    for t in range(T_traj):

                        if len(X_traj_pred) < history_k:

                            pad = [X_traj_pred[0]] * (history_k - len(X_traj_pred))
                            window = pad + X_traj_pred
                        else:
                            window = X_traj_pred[-history_k:]
                        history = torch.stack(window, dim=0).unsqueeze(0)
                        with torch.no_grad():
                            X_tilde = agg(history)
                            X_next = model.forward_step(X_tilde, A_norm_t, a_seq[t:t+1])
                        X_traj_pred.append(X_next.squeeze(0))
                    X_pred = torch.stack(X_traj_pred, dim=0).cpu().numpy()
                    for h in horizons:
                        if h > T_traj: continue
                        nm = float(np.mean((X_pred[h] - test_X[i, h]) ** 2))
                        nm = min(nm, CEIL) if math.isfinite(nm) else CEIL
                        per_h_mse[h].append(nm)
                row = {"variant": variant, "topology": topo, "seed": seed,
                       "history_k": history_k}
                for h in horizons:
                    if per_h_mse[h]:
                        row[f"NodeMSE@{h}_mean"] = float(np.mean(per_h_mse[h]))
                        row[f"NodeMSE@{h}_std"] = float(np.std(per_h_mse[h]))
                rows.append(row)
                print(f"  {variant:12s} {topo:12s} seed{seed}: NodeMSE@8={row.get('NodeMSE@8_mean', 0):.3e} NodeMSE@32={row.get('NodeMSE@32_mean', 0):.3e}")
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n[{now_jst()}] Exp 24 ext DONE: {len(rows)} rows → {out_path}")
    return {"exp": "24_ext", "n_rows": len(rows), "out": out_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, 'results', 'p5_p6'))
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--p2_dir", default=os.path.join(REPO_ROOT, "results", "p2_baselines"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    exp24_ext(args.out_dir, dev, data_root=args.data_root, p2_dir=args.p2_dir)


if __name__ == "__main__":
    main()
