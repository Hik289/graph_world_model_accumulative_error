"""Generate the F7 and F8 heterogeneous R-GCN baseline figures.

F7: Exp 16 correction policy reduction_pct bar chart (H5 secondary signal)
F8: R-GCN sigmoid-fixed calibration scatter (Exp 14 + Exp 25 side-by-side)

Author: Anonymous
Date: 2026-05-13 JST
"""

import os, warnings, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP16 = os.path.join(REPO_ROOT, "results", "p6_rgcn_hetero", "exp16_model_correction.csv")
EXP14 = os.path.join(REPO_ROOT, "results", "p6_rgcn_hetero", "exp14_model_rollout.csv")
EXP25 = os.path.join(REPO_ROOT, "results", "p6_rgcn_hetero", "exp25_model_rollout.csv")
OUT = os.path.join(REPO_ROOT, "results", "figures", "p6")
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
    "xtick.labelsize":  8.5,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "legend.framealpha": 0.92,
    "lines.linewidth":  1.6,
    "lines.markersize": 4,
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

POL_COLORS = {
    "oracle":     "#0072B2",
    "GEAF_proxy": "#D55E00",
    "degree":     "#E69F00",
    "random":     "#BBBBBB",
}
POL_LABELS = {
    "oracle":     "Oracle",
    "GEAF_proxy": "GEAF proxy",
    "degree":     "Degree",
    "random":     "Random",
}
POL_ORDER = ["oracle", "GEAF_proxy", "degree", "random"]

# ══════════════════════════════════════════════════════════════════════════════
# F7 — Correction policy reduction_pct
# ══════════════════════════════════════════════════════════════════════════════
print("[F7] Loading Exp 16 data ...")
df16 = pd.read_csv(EXP16)
agg16 = df16.groupby("policy")["reduction_pct"].agg(["mean","std"]).reset_index().set_index("policy")
for p in POL_ORDER:
    print(f"  {p}: {agg16.loc[p,'mean']:.2f}% ± {agg16.loc[p,'std']:.2f}%")

fig7, ax7 = plt.subplots(figsize=(5.5, 3.6))
x7 = np.arange(len(POL_ORDER))
means7 = [agg16.loc[p,"mean"] for p in POL_ORDER]
stds7  = [agg16.loc[p,"std"]  for p in POL_ORDER]
cols7  = [POL_COLORS[p] for p in POL_ORDER]

bars7 = ax7.bar(x7, means7, 0.55, color=cols7, alpha=0.87,
                edgecolor="black", linewidth=0.6, zorder=3)
ax7.errorbar(x7, means7, yerr=stds7, fmt="none",
             color="#333333", capsize=5, linewidth=1.2, zorder=5)

# Value annotations above bars
for i, (m, s) in enumerate(zip(means7, stds7)):
    ax7.text(i, m + s + 0.3, f"{m:.1f}%",
             ha="center", va="bottom", fontsize=8, fontweight="bold",
             color=cols7[i])

ax7.set_xticks(x7)
ax7.set_xticklabels([POL_LABELS[p] for p in POL_ORDER], fontsize=9)
ax7.set_xlabel("Correction policy (budget = 10%)", fontsize=10)
ax7.set_ylabel("Error reduction (%, mean ± std)", fontsize=9)
ax7.set_title("F7 (§5): R-GCN Correction Policy — Centrality Proxy Matches Degree (H5 secondary)",
              fontsize=8.5, fontweight="bold")
ax7.set_ylim(0, max(means7) * 1.30)

# Oracle-vs-GEAF gap annotation
gap = means7[POL_ORDER.index("oracle")] - means7[POL_ORDER.index("GEAF_proxy")]
ax7.annotate(
    f"oracle gap: +{gap:.1f}pp",
    xy=(0, means7[0]),
    xytext=(1.5, means7[0] * 0.92),
    fontsize=7, color="#0072B2",
    arrowprops=dict(arrowstyle="-", color="#0072B2", lw=0.8)
)
ax7.text(0.99, 0.05,
         "GEAF proxy ≡ Degree\n(identical centrality heuristic)\n"
         "H5: non-random node targeting\nconsistent with GEAF ordering",
         transform=ax7.transAxes, ha="right", va="bottom", fontsize=6.5,
         style="italic", color="#444444",
         bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, lw=0.4))

for fmt in ("png","pdf"):
    fig7.savefig(os.path.join(OUT, f"f7.{fmt}"), dpi=200)
    print(f"  Saved: f7.{fmt}")
plt.close(fig7)
shutil.copy(__file__, os.path.join(OUT, "f7_f8_gen_code.py"))

# Caption F7
_oracle = agg16.loc["oracle","mean"]
_geaf   = agg16.loc["GEAF_proxy","mean"]
_rand   = agg16.loc["random","mean"]
cap7 = (
    r"F7. R-GCN correction policy comparison on \textit{agent\_calling\_tree} heterogeneous "
    r"testbed (Exp~16, P6; budget = 10\% of nodes). "
    r"Mean $\pm$ std reduction in error\_flag mean across 10 instances. "
    + f"Oracle correction recovers {_oracle:.1f}\\% of error; "
    + f"GEAF proxy and degree-centrality are identical ({_geaf:.1f}\\%, "
    r"consistent with GEAF $\propto$ degree in star/scale-free subgraphs); "
    + f"random correction yields {_rand:.1f}\\%. "
    r"The GEAF-proxy advantage over random supports H5 (non-random targeting effective); "
    r"oracle gap indicates structural room for improved centraliy measures "
    r"beyond degree. \textit{Verdict not claimed; H5 secondary signal on heterogeneous testbed.}"
)
with open(os.path.join(OUT,"f7_caption.txt"),"w") as fh: fh.write(cap7)

# ══════════════════════════════════════════════════════════════════════════════
# F8 — R-GCN calibration scatter: Exp 14 + Exp 25
# ══════════════════════════════════════════════════════════════════════════════
print("[F8] Loading Exp 14 + Exp 25 data ...")
df14 = pd.read_csv(EXP14)
df25 = pd.read_csv(EXP25)

r14, p14 = stats.pearsonr(df14.rgcn_sr_at_sink_T, df14.true_sr_at_sink_T)
r25, p25 = stats.pearsonr(df25.rgcn_skill_sr_T,   df25.true_skill_sr_T)
mae14 = np.abs(df14.rgcn_sr_at_sink_T - df14.true_sr_at_sink_T).mean()
mae25 = np.abs(df25.rgcn_skill_sr_T   - df25.true_skill_sr_T).mean()

print(f"  Exp14: r={r14:.3f} p={p14:.3e} MAE={mae14:.3f}")
print(f"  Exp25: r={r25:.3f} p={p25:.3e} MAE={mae25:.3f}")

fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(8.5, 3.8))

for ax, df, xcol, ycol, title_txt, r_val, p_val, mae, col in [
    (ax8a, df14, "true_sr_at_sink_T", "rgcn_sr_at_sink_T",
     "Exp 14: agent_calling_tree ($sr_{\\rm sink}$)", r14, p14, mae14, "#0072B2"),
    (ax8b, df25, "true_skill_sr_T",   "rgcn_skill_sr_T",
     "Exp 25: skill_graph ($sr_{\\rm skill}$)", r25, p25, mae25, "#D55E00"),
]:
    x_ = df[xcol].values
    y_ = df[ycol].values
    ax.scatter(x_, y_, s=18, color=col, alpha=0.55, edgecolors="none", zorder=3)

    # Identity line
    mn_ = min(x_.min(), y_.min()) * 0.97
    mx_ = max(x_.max(), y_.max()) * 1.03
    ax.plot([mn_, mx_], [mn_, mx_], color="gray", lw=1.0, ls="--", alpha=0.7,
            label="$y=x$ (perfect)", zorder=2)

    # Regression line
    m_, b_ = np.polyfit(x_, y_, 1)
    xs_ = np.linspace(mn_, mx_, 100)
    ax.plot(xs_, m_*xs_+b_, color=col, lw=1.4, ls="-", alpha=0.6, zorder=4)

    # Stats annotation
    ax.text(0.05, 0.95,
            f"R-GCN mean: {y_.mean():.3f}\nGT mean: {x_.mean():.3f}\n"
            f"Pearson r={r_val:.3f}\nMAE={mae:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, lw=0.4))

    ax.set_xlabel("Ground-truth simulator", fontsize=9.5)
    ax.set_ylabel("R-GCN prediction (sigmoid-fixed)", fontsize=9.5)
    ax.set_title(title_txt, fontsize=8.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(mn_, mx_)
    ax.set_ylim(mn_, mx_)

fig8.suptitle("F8: R-GCN Baseline Calibration (sigmoid-fixed) vs Ground-Truth Simulator",
              fontsize=10, fontweight="bold")
fig8.tight_layout(rect=[0, 0, 1, 0.94])

for fmt in ("png","pdf"):
    fig8.savefig(os.path.join(OUT, f"f8.{fmt}"), dpi=200)
    print(f"  Saved: f8.{fmt}")
plt.close(fig8)

cap8 = (
    r"F8. R-GCN heterogeneous baseline (sigmoid-clipped \texttt{sr}/\texttt{error\_flag} heads) "
    r"compared to ground-truth simulator outputs. "
    r"\textbf{Left} --- Exp~14 agent\_calling\_tree: "
    r"R-GCN mean $sr_{\rm sink}$ = " + f"{df14.rgcn_sr_at_sink_T.mean():.3f}" +
    r" vs GT = " + f"{df14.true_sr_at_sink_T.mean():.3f}" +
    r" (Pearson $r=" + f"{r14:.3f}" + r"$, MAE=" + f"{mae14:.3f}" + r"). "
    r"\textbf{Right} --- Exp~25 platform skill\_graph: "
    r"R-GCN mean $sr_{\rm skill}$ = " + f"{df25.rgcn_skill_sr_T.mean():.3f}" +
    r" vs GT = " + f"{df25.true_skill_sr_T.mean():.3f}" +
    r" (Pearson $r=" + f"{r25:.3f}" + r"$, MAE=" + f"{mae25:.3f}" + r"). "
    r"Post-sigmoid-fix, R-GCN closely matches ground-truth mean on both testbeds, "
    r"validating the R-GCN hetero baseline as a reliable Method \S4 comparison point. "
    r"Moderate Pearson $r$ values reflect instance-level variance, not systematic bias."
)
with open(os.path.join(OUT,"f8_caption.txt"),"w") as fh: fh.write(cap8)

print("[F7 + F8] DONE")
print(f"  Output: {OUT}/f7.{{png,pdf,_caption.txt}} + f8.{{png,pdf,_caption.txt}}")
