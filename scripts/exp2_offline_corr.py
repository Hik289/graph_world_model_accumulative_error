"""Exp 2: Correlation between graph statistics and error growth.

Configuration follows README §6.2 and the P3–P6 matrix §2.
- 8 个 graph stats × 3 error metrics (NodeMSE@20, ReturnError@20, Regret@20)
- 跨 (baseline, topology, seed) ok-or-cap diverged
- Pearson + Spearman + 95% bootstrap CI (Fisher-z)
- 用 stat_test_spec §0.4 small-sample fallback 一致的 ceiling assignment

Inputs (从 P2 已 trained 产物):
  - .../results/p2_baselines/{topo}/{baseline}_seed{s}.json (含 theory_constants)
  - .../results/p2_baselines/checkpoints/{topo}/{baseline}_seed{s}.pt (model weights)
  - .../data/synthetic_graphs/{topo}_N50_seed{s}.pt (graph stats)
  - .../data/synthetic_rollouts/fe_{topo}_N50_seed{s}_T50.pt (test data)

Outputs:
  - results/exp2_offline_raw.csv (每个 baseline × topo × seed × metric 一行)
  - results/exp2_correlation_table.csv (stat × metric × baseline-group → r, p, CI)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.baselines import BASELINE_REGISTRY
from src.graph_generators import generate, compute_all
from src.metrics import (
    RolloutPrediction, node_mse, return_error,
)

BASELINES = ["B1_MLP", "B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode", "B6_ErrorAware"]
TOPOLOGIES = ["chain", "tree", "grid", "small_world", "scale_free", "star", "complete"]
SEEDS = [1, 2, 3]
N_DEFAULT = 50

# 8 graph statistics per Exp 2:
# 7 拓扑-only stats + GEAF_hat (来自 P2 trained model 的 theory_constants)
GRAPH_STATS = ["rho_A", "GEAF_hat", "avg_degree", "degree_variance",
               "diameter", "clustering", "betweenness_concentration",
               "pagerank_concentration"]

# Error metric pool
ERROR_METRICS = ["NodeMSE@20", "ReturnError@20", "GrowthSlope_4_20"]

# Diverged ceiling (per stat_test_spec)
CEIL_NODE_MSE = 1e10


def load_graph_stats(data_root: str, topo: str, seed: int) -> Dict[str, float]:
    """从 synthetic_graphs 加载图统计量."""
    g = generate(topo, N=N_DEFAULT, seed=seed)
    stats_dict = compute_all(g)
    return stats_dict


def load_p2_theory_constants(p2_dir: str, baseline: str, topo: str, seed: int) -> Dict[str, float]:
    """从 P2 run JSON 读 theory_constants (含 GEAF_hat 等 model-side scalars)."""
    fp = os.path.join(p2_dir, topo, f"{baseline}_seed{seed}.json")
    if not os.path.exists(fp):
        return {}
    try:
        with open(fp) as f:
            d = json.load(f)
        tc = d.get("theory_constants", {})
        return {f"tc_{k}": v for k, v in tc.items()
                if isinstance(v, (int, float)) or v is None}
    except Exception:
        return {}


def eval_one_run(
    baseline: str, topo: str, seed: int, *,
    data_root: str, p2_dir: str,
    device: str = "cpu",
    H_eval: int = 20,
) -> Dict[str, Any]:
    """加载 checkpoint, 跑 rollout, 计算 NodeMSE@H_eval / ReturnError@H_eval / GrowthSlope_4_H_eval."""
    ck_path = os.path.join(p2_dir, "checkpoints", topo,
                           f"{baseline}_seed{seed}.pt")
    if not os.path.exists(ck_path):
        return {"status": "missing_ck"}
    rollout_path = os.path.join(data_root, "synthetic_rollouts",
                                f"fe_{topo}_N{N_DEFAULT}_seed{seed}_T50.pt")
    payload = torch.load(rollout_path, weights_only=False)
    test_X = torch.from_numpy(payload["test_X"]).float()
    test_a = torch.from_numpy(payload["test_actions"]).float()
    g = generate(topo, N=N_DEFAULT, seed=seed)
    A_norm_t = torch.from_numpy(g.A_norm).float()
    A_dense = g.A_dense
    # 模型
    cls = BASELINE_REGISTRY[baseline]
    if baseline == "B1_MLP":
        model = cls(N=N_DEFAULT)
    else:
        model = cls()
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    A_norm_t = A_norm_t.to(dev)
    X_0 = test_X[:, 0].to(dev)
    actions_test = test_a.to(dev)
    T_test = test_X.shape[1] - 1
    H = min(H_eval, T_test)
    with torch.no_grad():
        X_pred_traj = model.rollout_predict(X_0, A_norm_t, actions_test, T=T_test)
        X_pred_traj = X_pred_traj.cpu().numpy().astype(np.float32)
    test_X_np = test_X.numpy()
    per_traj_mse = []
    per_traj_re = []
    per_traj_mse_4 = []
    n_traj = test_X_np.shape[0]
    for i in range(n_traj):
        pred = RolloutPrediction(
            X_true=test_X_np[i], A_true=A_dense,
            X_pred=X_pred_traj[i], A_pred=A_dense,
            is_fixed_edge=True,
        )
        # Reward proxy: ‖X_t‖_F per step, normalized
        T_traj = test_X_np[i].shape[0] - 1
        r_true = np.linalg.norm(test_X_np[i][1:].reshape(T_traj, -1), axis=1)
        r_pred = np.linalg.norm(X_pred_traj[i][1:].reshape(T_traj, -1), axis=1)
        pred.rewards_true = r_true
        pred.rewards_pred = r_pred
        per_traj_mse.append(node_mse(pred, H))
        per_traj_re.append(return_error(pred, H, gamma=0.95))
        per_traj_mse_4.append(node_mse(pred, 4))
    mse_h = float(np.mean(per_traj_mse))
    re_h = float(np.mean(per_traj_re))
    mse_4 = float(np.mean(per_traj_mse_4))
    # GrowthSlope_4_H = (log mse_H - log mse_4) / (H - 4)
    if mse_h > 1e-15 and mse_4 > 1e-15:
        gs = (math.log(mse_h) - math.log(mse_4)) / (H - 4)
    else:
        gs = float("nan")
    diverged = (not math.isfinite(mse_h)) or mse_h > 1e3
    # Apply ceiling for diverged
    mse_h_capped = min(mse_h, CEIL_NODE_MSE) if math.isfinite(mse_h) else CEIL_NODE_MSE
    re_h_capped = min(abs(re_h), CEIL_NODE_MSE) if math.isfinite(re_h) else CEIL_NODE_MSE
    return {
        "status": "diverged" if diverged else "ok",
        "NodeMSE@20": mse_h,
        "NodeMSE@20_capped": mse_h_capped,
        "ReturnError@20": re_h,
        "ReturnError@20_capped": re_h_capped,
        "NodeMSE@4_proxy": mse_4,
        "GrowthSlope_4_20": gs,
    }


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Fisher z-transform → 95% CI for Pearson r."""
    if n < 4 or not math.isfinite(r) or abs(r) >= 1.0 - 1e-12:
        return (float("nan"), float("nan"))
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = math.tanh(z - z_crit * se)
    hi = math.tanh(z + z_crit * se)
    return (float(lo), float(hi))


def compute_correlations(df: pd.DataFrame, baseline_group: str, baselines_in_group: List[str],
                         stat_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    rows = []
    sub = df[df["baseline"].isin(baselines_in_group)].copy()
    for stat in stat_cols:
        for metric in metric_cols:
            x = sub[stat].astype(float).to_numpy()
            y = sub[metric].astype(float).to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            n = int(m.sum())
            if n < 4:
                rows.append({"baseline_group": baseline_group, "n": n, "stat": stat,
                             "metric": metric, "r": float("nan"), "p": float("nan"),
                             "ci_low": float("nan"), "ci_high": float("nan"),
                             "spearman_rho": float("nan"), "spearman_p": float("nan")})
                continue
            r, p = stats.pearsonr(x[m], y[m])
            rs, ps = stats.spearmanr(x[m], y[m])
            lo, hi = fisher_ci(r, n)
            rows.append({"baseline_group": baseline_group, "n": n, "stat": stat,
                         "metric": metric, "r": float(r), "p": float(p),
                         "ci_low": lo, "ci_high": hi,
                         "spearman_rho": float(rs), "spearman_p": float(ps)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--p2_dir", default=os.path.join(REPO_ROOT, "results", "p2_baselines"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("[Exp 2] Offline correlation analysis")
    t0 = time.time()
    rows = []
    for bl in BASELINES:
        for topo in TOPOLOGIES:
            for s in SEEDS:
                gs = load_graph_stats(args.data_root, topo, s)
                tc = load_p2_theory_constants(args.p2_dir, bl, topo, s)
                try:
                    r = eval_one_run(bl, topo, s,
                                     data_root=args.data_root, p2_dir=args.p2_dir,
                                     device=args.device, H_eval=20)
                except Exception as e:
                    r = {"status": f"error: {str(e)[:120]}"}
                # 让 model-side GEAF_hat 与 graph-only rho_A 都进 row
                row = {"baseline": bl, "topology": topo, "seed": s, **gs, **tc, **r}
                # 别名: tc_GEAF_hat → GEAF_hat (供 GRAPH_STATS 使用)
                if "tc_GEAF_hat" in tc:
                    row["GEAF_hat"] = tc["tc_GEAF_hat"]
                rows.append(row)
                print(f"  {bl:14s} {topo:12s} seed{s} status={r.get('status')}"
                      f" NodeMSE@20={r.get('NodeMSE@20', float('nan')):.3e}"
                      f" ReturnError@20={r.get('ReturnError@20', float('nan')):.3e}")
    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "exp2_offline_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"Raw: {raw_path}  ({len(df)} rows)")

    # Per-baseline-group correlation
    # (a) pooled all 6 baselines (用 capped metric)
    df_capped = df.copy()
    df_capped["NodeMSE@20"] = df["NodeMSE@20_capped"]
    df_capped["ReturnError@20"] = df["ReturnError@20_capped"]
    err_cols = ["NodeMSE@20", "ReturnError@20", "GrowthSlope_4_20"]
    corr_all = []
    corr_all.append(compute_correlations(df_capped, "all_6", BASELINES, GRAPH_STATS, err_cols))
    # (b) excl B6 (per stat_test_spec H1.1)
    corr_all.append(compute_correlations(df_capped, "B2_B3_B4_B5", ["B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode"], GRAPH_STATS, err_cols))
    # (c) per-baseline
    for bl in BASELINES:
        corr_all.append(compute_correlations(df_capped, bl, [bl], GRAPH_STATS, err_cols))
    corr_df = pd.concat(corr_all, ignore_index=True)
    corr_path = os.path.join(args.out_dir, "exp2_correlation_table.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"Correlation: {corr_path}  ({len(corr_df)} rows)")

    # 高亮: 跨 (rho_A, GEAF_hat, degree_variance, betweenness_concentration) × all 6 baselines
    print("\n=== H1 Headline numbers (all 6 baselines pooled, capped) ===")
    h1_stats = ["rho_A_raw", "GEAF_hat", "degree_variance", "betweenness_concentration", "pagerank_concentration"]
    headline = corr_df[(corr_df["baseline_group"] == "all_6") &
                       (corr_df["stat"].isin(h1_stats))]
    print(headline.to_string(index=False))

    print(f"\n[Exp 2] DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
