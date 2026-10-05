from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINES = ["B1_MLP", "B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode", "B6_ErrorAware"]
TOPOLOGIES = ["chain", "tree", "grid", "small_world", "scale_free", "star", "complete"]
SEEDS = [1, 2, 3]


def load_results(out_dir: str) -> pd.DataFrame:
    rows = []
    for bl in BASELINES:
        for top in TOPOLOGIES:
            for s in SEEDS:
                fp = os.path.join(out_dir, top, f"{bl}_seed{s}.json")
                if not os.path.exists(fp):
                    rows.append({"baseline": bl, "topology": top, "seed": s,
                                 "status": "missing"})
                    continue
                try:
                    with open(fp) as f:
                        d = json.load(f)
                    if d.get("skipped"):
                        rows.append({"baseline": bl, "topology": top, "seed": s,
                                     "status": "N/A"})
                        continue
                    m = d["test_metrics"]
                    tc = d["theory_constants"]
                    final_train_loss = float(d["train_loss_curve"][-1]) if d["train_loss_curve"] else float("nan")
                    val_loss = float(d.get("best_val_loss", float("nan")))

                    nm32 = m.get("NodeMSE@32", float("nan"))
                    diverged = (
                        not math.isfinite(nm32) or nm32 > 1e3 or
                        not math.isfinite(final_train_loss)
                    )
                    status = "diverged" if diverged else "ok"
                    rows.append({
                        "baseline": bl, "topology": top, "seed": s,
                        "status": status,
                        "NodeMSE@1": m.get("NodeMSE@1"),
                        "NodeMSE@4": m.get("NodeMSE@4"),
                        "NodeMSE@8": m.get("NodeMSE@8"),
                        "NodeMSE@16": m.get("NodeMSE@16"),
                        "NodeMSE@32": m.get("NodeMSE@32"),
                        "EdgeF1@8": m.get("EdgeF1@8"),
                        "EdgeF1@32": m.get("EdgeF1@32"),
                        "GrowthSlope_4_32": m.get("GrowthSlope_4_32"),
                        "rho_A_raw": tc.get("rho_A_raw"),
                        "rho_A_norm": tc.get("rho_A_norm"),
                        "GEAF_hat": tc.get("GEAF_hat"),
                        "rho_B": tc.get("rho_B"),
                        "rho_B_eig": tc.get("rho_B_eig"),
                        "L_X": tc.get("L_X"),
                        "L_A": tc.get("L_A"),
                        "M_X": tc.get("M_X"),
                        "M_A": tc.get("M_A"),
                        "final_train_loss": final_train_loss,
                        "best_val_loss": val_loss,
                        "train_time_sec": float(d["meta"].get("train_time_sec", float("nan"))),
                        "n_params": int(d["meta"].get("n_params", 0)),
                    })
                except Exception as e:
                    rows.append({"baseline": bl, "topology": top, "seed": s,
                                 "status": f"parse_error: {str(e)[:80]}"})
    return pd.DataFrame(rows)


def aggregate_mean_std(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    agg = ok.reindex(columns=group_cols + metric_cols).groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    return agg


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for bl in BASELINES:
        sub = df[(df["baseline"] == bl) & (df["status"] == "ok")]
        if len(sub) < 3:
            continue
        rho_b = sub["rho_B"].astype(float).to_numpy()
        geaf = sub["GEAF_hat"].astype(float).to_numpy()
        mask = np.isfinite(rho_b) & np.isfinite(geaf)
        if mask.sum() < 3:
            continue
        r, p = stats.pearsonr(rho_b[mask], geaf[mask])
        rs, ps = stats.spearmanr(rho_b[mask], geaf[mask])
        out.append({
            "baseline": bl, "n": int(mask.sum()),
            "pearson_rho_B_vs_GEAF": float(r),
            "pearson_p": float(p),
            "spearman_rho_B_vs_GEAF": float(rs),
            "spearman_p": float(ps),
        })
    return pd.DataFrame(out)


def h1_primary_signal(df: pd.DataFrame) -> Dict[str, Any]:
    sub = df[(df["status"] == "ok") &
             (df["baseline"].isin(["B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode"]))]
    if len(sub) < 3:
        return {"n": int(len(sub)), "pearson_r": float("nan")}
    geaf = sub["GEAF_hat"].astype(float).to_numpy()
    nm = sub["NodeMSE@32"].astype(float).to_numpy()
    mask = np.isfinite(geaf) & np.isfinite(nm) & (nm > 0)
    if mask.sum() < 5:
        return {"n": int(mask.sum()), "pearson_r": float("nan")}
    log_geaf = np.log(geaf[mask])
    log_nm = np.log(nm[mask])
    r, p = stats.pearsonr(log_geaf, log_nm)
    rs, ps = stats.spearmanr(geaf[mask], nm[mask])
    return {
        "n_pairs": int(mask.sum()),
        "pearson_log_GEAF_log_NodeMSE32": float(r),
        "pearson_p": float(p),
        "spearman_GEAF_NodeMSE32": float(rs),
        "spearman_p": float(ps),
        "baselines_pooled": ["B2_GCN", "B3_MPNN", "B4_GPS", "B5_ActionNode"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, 'results', 'p2_baselines'))
    parser.add_argument("--report_dir", default=os.path.join(REPO_ROOT, 'results'))
    args = parser.parse_args()
    df = load_results(args.out_dir)
    os.makedirs(args.report_dir, exist_ok=True)
    csv_path = os.path.join(args.report_dir, "p2_baselines_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"Raw: {csv_path}  ({len(df)} rows)")

    status_summary = df.groupby("status").size().to_dict()
    print("Status:", status_summary)

    metrics = ["NodeMSE@8", "NodeMSE@32", "GEAF_hat", "rho_B", "final_train_loss"]
    agg = aggregate_mean_std(df, ["baseline", "topology"], metrics)
    agg_path = os.path.join(args.report_dir, "p2_baselines_agg.csv")
    agg.to_csv(agg_path, index=False)

    corr = correlation_table(df)
    corr_path = os.path.join(args.report_dir, "p2_correlation.csv")
    corr.to_csv(corr_path, index=False)
    print("\nPer-baseline rho_B vs GEAF correlation:")
    print(corr.to_string(index=False))

    h1 = h1_primary_signal(df)
    print("\nH1 primary signal (pooled log-log Pearson):")
    print(json.dumps(h1, indent=2))

    summary = {
        "n_total": int(len(df)),
        "status_counts": {k: int(v) for k, v in status_summary.items()},
        "h1_primary_signal": h1,
        "per_baseline_correlation": corr.to_dict(orient="records"),
    }
    sum_path = os.path.join(args.report_dir, "p2_baselines_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {sum_path}")


if __name__ == "__main__":
    main()
