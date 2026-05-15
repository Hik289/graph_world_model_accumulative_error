"""Topology stats × error metrics correlation (Exp 2/11/19)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def correlation_topology_error(
    stats_df: pd.DataFrame, errors_df: pd.DataFrame,
    *, method: str = "pearson",
    join_keys: List[str] = None,
) -> pd.DataFrame:
    """跨 (topology × seed) 计算 stat × error_metric 相关系数 + 95% bootstrap CI.

    Parameters
    ----------
    stats_df : 列至少含 'topology', 'seed', + 各 stat 列.
    errors_df : 列至少含 'topology', 'seed', + 各 error metric 列.
    method : "pearson" | "spearman".
    join_keys : merge 键 (默认 ['topology', 'seed']).
    """
    if join_keys is None:
        join_keys = ["topology", "seed"]
    df = stats_df.merge(errors_df, on=join_keys, how="inner")
    stat_cols = [c for c in stats_df.columns if c not in join_keys]
    err_cols = [c for c in errors_df.columns if c not in join_keys]

    rows = []
    for s_col in stat_cols:
        for e_col in err_cols:
            x = df[s_col].astype(np.float64).to_numpy()
            y = df[e_col].astype(np.float64).to_numpy()
            mask = (~np.isnan(x)) & (~np.isnan(y))
            if mask.sum() < 3:
                rows.append({"stat": s_col, "error_metric": e_col,
                             "r": float("nan"), "p_value": float("nan"),
                             "ci_low": float("nan"), "ci_high": float("nan"),
                             "n_pairs": int(mask.sum())})
                continue
            xm, ym = x[mask], y[mask]
            if method == "pearson":
                r, p = stats.pearsonr(xm, ym)
            elif method == "spearman":
                r, p = stats.spearmanr(xm, ym)
            else:
                raise ValueError(method)
            # 95% bootstrap CI on r
            rng = np.random.default_rng(0)
            idx = rng.choice(len(xm), size=(1000, len(xm)), replace=True)
            rs = []
            for k in range(1000):
                xs = xm[idx[k]]
                ys = ym[idx[k]]
                if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
                    continue
                r_b = np.corrcoef(xs, ys)[0, 1] if method == "pearson" else stats.spearmanr(xs, ys)[0]
                rs.append(r_b)
            if rs:
                ci_low = float(np.percentile(rs, 2.5))
                ci_high = float(np.percentile(rs, 97.5))
            else:
                ci_low = ci_high = float("nan")
            rows.append({"stat": s_col, "error_metric": e_col,
                         "r": float(r), "p_value": float(p),
                         "ci_low": ci_low, "ci_high": ci_high,
                         "n_pairs": int(mask.sum())})
    return pd.DataFrame(rows)
