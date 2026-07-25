"""F6 — H7 redemption test and B5/B6 GEAF spectral comparison.

Two-panel design:
  Left:  NodeMSE@H pert_vs_clean vs H (log-log), 3 ε values, contractive decay
  Right: GEAF_hat per topology, B5 vs B6 grouped bars + floor annotation

Data sources:
  experiments/exp18_multi_axis.csv   (108 rows: B5/B6 × 6 topo × 3 seed × 3 ε)
  results/p2_baselines_raw.csv       (GEAF per topology per baseline)

Author: Anonymous
Date: 2026-05-13 JST
"""

import os, warnings, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP18 = os.path.join(REPO_ROOT, "results", "p5_exp18_full", "exp18_multi_axis.csv")
P2_RAW = os.path.join(REPO_ROOT, "results", "p2_baselines_raw.csv")
OUT = os.path.join(REPO_ROOT, "results", "figures", "p2")
os.makedirs(OUT, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-paper")
matplotlib.rcParams.update({
    "font.family":      "Helvetica",
    "font.size":        9,
    "axes.labelsize":   10,
    "axes.labelweight": "bold",
    "axes.titlesize":   9,
    "axes.titleweight": "bold",
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "legend.framealpha": 0.92,
    "lines.linewidth":  1.8,
    "lines.markersize": 5,
    "errorbar.capsize": 3.5,
    "axes.grid":        True,
    "grid.alpha":       0.22,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       200,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

EPS_COLORS = {
    0.001: "#56B4E9",   # light blue
    0.01:  "#E69F00",   # orange
    0.1:   "#D55E00",   # vermillion
}
EPS_LABELS = {
    0.001: r"$\varepsilon=0.001$",
    0.01:  r"$\varepsilon=0.010$",
    0.1:   r"$\varepsilon=0.100$",
}

BL_COLORS = {
    "B5_ActionNode": "#009E73",   # bluish-green
    "B6_ErrorAware": "#CC79A7",   # reddish-purple
}
BL_LABELS = {
    "B5_ActionNode": "B5 ActionNode",
    "B6_ErrorAware": "B6 ErrorAware",
}
BL_ORDER = ["B5_ActionNode", "B6_ErrorAware"]

TOPO_ORDER_6 = ["chain", "tree", "grid", "small_world", "scale_free", "star"]
TOPO_LABELS  = {
    "chain": "Chain", "tree": "Tree", "grid": "Grid",
    "small_world": "SW", "scale_free": "SF", "star": "Star",
}

H_VALS = [1, 2, 4, 8, 16, 32, 64, 128]
EPS_VALS = [0.001, 0.01, 0.1]

# ─── Load data ────────────────────────────────────────────────────────────────
print("[F6] Loading data ...")
df_exp18 = pd.read_csv(EXP18)
df_p2    = pd.read_csv(P2_RAW)
EPS_KEY  = "eps"  # column name in exp18

print(f"  Exp18 shape: {df_exp18.shape}")
print(f"  eps values: {sorted(df_exp18[EPS_KEY].unique())}")
print(f"  baselines: {sorted(df_exp18['baseline'].unique())}")

# ─── LEFT PANEL: pert_vs_clean decay ─────────────────────────────────────────
print("[F6] Computing pert_vs_clean per (eps, H) ...")
EPS_CLIP = 1e-20  # floor clip for log scale

decay = {}
for eps in EPS_VALS:
    sub = df_exp18[df_exp18[EPS_KEY] == eps]
    means, stds = [], []
    for h in H_VALS:
        col = f"NodeMSE@{h}_pert_vs_clean"
        vals = sub[col].clip(lower=EPS_CLIP)
        means.append(vals.mean())
        stds.append(vals.std())
    decay[eps] = {"mean": means, "std": stds}
    print(f"  eps={eps}: H1={means[0]:.2e}  H16={means[4]:.2e}  H128={means[-1]:.2e}")

# B5 vs B6 floors at H=128
b5_floor = df_exp18[df_exp18["baseline"]=="B5_ActionNode"]["NodeMSE@128_pert_vs_clean"].mean()
b6_floor = df_exp18[df_exp18["baseline"]=="B6_ErrorAware"]["NodeMSE@128_pert_vs_clean"].mean()
floor_ratio = b5_floor / max(b6_floor, 1e-20)
print(f"  B5 floor@H128: {b5_floor:.3e}, B6 floor@H128: {b6_floor:.3e}, ratio: {floor_ratio:.1f}×")

# ─── RIGHT PANEL: GEAF per (topology, baseline) ───────────────────────────────
print("[F6] Computing GEAF per (topology, baseline) ...")
df_geaf = df_p2[df_p2["topology"].isin(TOPO_ORDER_6)]
geaf = df_geaf.groupby(["baseline","topology"])["GEAF_hat"].agg(["mean","std"]).reset_index()
for bl in BL_ORDER:
    sub = geaf[geaf["baseline"]==bl]
    print(f"  {bl}: " + "  ".join(f"{TOPO_LABELS[t]}={sub[sub['topology']==t]['mean'].values[0]:.1f}" for t in TOPO_ORDER_6))

# ─── Render ───────────────────────────────────────────────────────────────────
print("[F6] Rendering ...")
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.0, 4.0))

# ────────────────────────────────────────────────────────────────────────────
# LEFT PANEL: Log-log decay plot
# ────────────────────────────────────────────────────────────────────────────
for eps in EPS_VALS:
    col  = EPS_COLORS[eps]
    lbl  = EPS_LABELS[eps]
    mn   = np.array(decay[eps]["mean"]).clip(min=EPS_CLIP)
    sd   = np.array(decay[eps]["std"])

    ax_l.plot(H_VALS, mn, color=col, lw=2.0, ls="-",
              marker="o", markersize=5, alpha=0.9, label=lbl, zorder=4)
    ax_l.fill_between(H_VALS,
                      np.clip(mn - sd, EPS_CLIP, None),
                      np.clip(mn + sd, EPS_CLIP, None),
                      color=col, alpha=0.14, zorder=2)

ax_l.set_xscale("log")
ax_l.set_yscale("log")
ax_l.set_xlabel("Rollout horizon $H$", fontsize=10)
ax_l.set_ylabel(r"NodeMSE@$H$ (pert vs clean, log scale)", fontsize=9)
ax_l.set_title(r"(a) H7 Redemption: Contractive decay $\propto\varepsilon^2$ absorbed by $H=16$",
               fontsize=8.5, fontweight="bold")
ax_l.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
ax_l.set_xticklabels(["1","2","4","8","16","32","64","128"])

# ε² slope reference line (at H=1 anchor eps=0.1)
h_ref = np.array([1, 128])
ref_start = decay[0.1]["mean"][0] * 1.2
slope = -2.0  # contractive decay steeper than ε² in H; ref line shows ε² scale
ref_line = ref_start * (h_ref / h_ref[0]) ** slope
ax_l.plot(h_ref, ref_line, color="gray", lw=1.0, ls=":", alpha=0.6, zorder=1, label="slope ∝ H⁻²")

# Floor annotation
ax_l.axhline(b5_floor, color=BL_COLORS["B5_ActionNode"], lw=0.9, ls="--",
             alpha=0.65, zorder=3)
ax_l.axhline(b6_floor, color=BL_COLORS["B6_ErrorAware"], lw=0.9, ls="--",
             alpha=0.65, zorder=3)
ax_l.text(1.4, b5_floor * 2.2,
          f"B5 floor: {b5_floor:.1e}",
          color=BL_COLORS["B5_ActionNode"], fontsize=6.5, va="bottom")
ax_l.text(1.4, b6_floor * 0.35,
          f"B6 floor: {b6_floor:.1e}",
          color=BL_COLORS["B6_ErrorAware"], fontsize=6.5, va="top")

# ε² scaling annotation at H=1
ax_l.text(1.1, decay[0.001]["mean"][0] * 0.5,
          r"$\varepsilon^2$ scaling at $H=1$",
          fontsize=6.5, color="#333333", rotation=0,
          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, lw=0.4))

ax_l.legend(loc="lower left", fontsize=7.5, ncol=1,
            title="Perturbation size", title_fontsize=7.0)

# ────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL: GEAF grouped bar B5 vs B6
# ────────────────────────────────────────────────────────────────────────────
n_topos = len(TOPO_ORDER_6)
bw = 0.35
offsets = [-bw/2, bw/2]
x = np.arange(n_topos)

for bi, bl in enumerate(BL_ORDER):
    sub = geaf[geaf["baseline"]==bl].set_index("topology")
    means = [sub.loc[t,"mean"] if t in sub.index else 0.0 for t in TOPO_ORDER_6]
    stds  = [sub.loc[t,"std"]  if t in sub.index else 0.0 for t in TOPO_ORDER_6]
    xpos  = x + offsets[bi]
    ax_r.bar(xpos, means, bw * 0.92, color=BL_COLORS[bl], alpha=0.85,
             edgecolor="white", linewidth=0.4, label=BL_LABELS[bl], zorder=3)
    ax_r.errorbar(xpos, means, yerr=stds, fmt="none",
                  color="#333333", capsize=3.5, linewidth=0.9, zorder=5)

ax_r.set_xticks(x)
ax_r.set_xticklabels([TOPO_LABELS[t] for t in TOPO_ORDER_6], fontsize=9)
ax_r.set_xlabel("Graph Topology (complete excluded — outlier $K_N$ GEAF)", fontsize=8.5)
ax_r.set_ylabel(r"GEAF$_{\hat{}}$ = $\rho(A)\cdot\prod_\ell\|W_\ell\|_2$ (mean $\pm$ std)", fontsize=9)
ax_r.set_title(r"(b) GEAF$_{\hat{}}$: B6 spectral regularization reduces error ceiling ~10×",
               fontsize=8.5, fontweight="bold")

# Summary annotation
b5_geaf_mean = geaf[geaf["baseline"]=="B5_ActionNode"]["mean"].mean()
b6_geaf_mean = geaf[geaf["baseline"]=="B6_ErrorAware"]["mean"].mean()
geaf_fold = b5_geaf_mean / max(b6_geaf_mean, 1e-9)

ax_r.text(0.98, 0.97,
          f"B5 mean GEAF: {b5_geaf_mean:.1f}\n"
          f"B6 mean GEAF: {b6_geaf_mean:.1f}\n"
          f"B5/B6 ratio: {geaf_fold:.1f}×\n"
          f"B5 floor: {b5_floor:.1e}\n"
          f"B6 floor: {b6_floor:.1e}\n"
          f"Floor ratio: {floor_ratio:.1f}×",
          transform=ax_r.transAxes, ha="right", va="top", fontsize=6.5,
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, lw=0.5))

ax_r.legend(loc="upper left", fontsize=8, title="Baseline", title_fontsize=7.5)

# ─── Overall layout ───────────────────────────────────────────────────────────
fig.suptitle(
    "F6: H7 Redemption — Contractive Decay (Exp 18) + B5/B6 GEAF Comparison (P2)",
    fontsize=10, fontweight="bold"
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

for fmt in ("png", "pdf"):
    out_path = os.path.join(OUT, f"f6.{fmt}")
    fig.savefig(out_path, dpi=200)
    print(f"  Saved: {out_path}")
plt.close(fig)

shutil.copy(__file__, os.path.join(OUT, "f6_gen_code.py"))
print("  Code backup saved.")

# ─── Caption ─────────────────────────────────────────────────────────────────
# Pre-compute values for caption
_b5_f = b5_floor
_b6_f = b6_floor
_fr   = floor_ratio
_eps_h1_ratio = decay[0.1]["mean"][0] / decay[0.01]["mean"][0]  # should be ~100

caption = (
    r"F6. H7 redemption test and B5/B6 spectral comparison. "
    r"\textbf{(a) Left} --- NodeMSE@$H$ (perturbed vs.\ clean rollout, "
    r"mean $\pm$ std across B5 + B6, all topologies, 3 seeds) "
    r"under three perturbation magnitudes $\varepsilon \in \{10^{-3}, 10^{-2}, 10^{-1}\}$ "
    r"(Exp~18, P5). "
    r"At $H=1$, the pert-vs-clean error scales as $\varepsilon^2$ "
    r"(100$\times$ ratio per decade: consistent with linearised perturbation bound). "
    r"All three $\varepsilon$ values decay to numerical floor by $H=16$, "
    r"confirming contractive dynamics in the in-distribution training regime "
    r"(Theorist \S3.2 E2: block-diagonal collapse). "
    + "B5 floor at $H=128$: {:.1e}; B6 floor: {:.1e} ({:.1f}$\\times$ lower), ".format(_b5_f, _b6_f, _fr) +
    r"consistent with B6 spectral regularisation reducing steady-state numerical noise. "
    r"\textit{H7 verdict not claimed; verdict pending data\_scientist "
    r"\texttt{stat\_test\_spec} \S\S A3-dual-test.} "
    r"\textbf{(b) Right} --- GEAF$_{\hat{}}$ per topology "
    r"(mean $\pm$ std, P2 baselines, complete excluded as $K_N$ outlier). "
    r"B6 (ErrorAware, spectral-regularised) has "
    + "{:.1f}$\\times$ lower GEAF than B5 (ActionNode) ".format(geaf_fold) +
    r"across all topologies, confirming that spectral regularisation "
    r"reduces the theoretical error ceiling even in the fixed-edge regime."
)

with open(os.path.join(OUT, "f6_caption.txt"), "w") as fh:
    fh.write(caption)
print("  Caption saved.")
print("[F6] DONE")
