# Graph World Model Accumulative Error (gwmerror)

Official code release for the NeurIPS 2026 paper:

> **"Topology-Aware Growth-Rate Prediction and Spectral-Regularization Stability for Graph World Model Rollouts"**

---

## Overview

![Framework Overview](figures/fig_main.png)

This repository provides the full benchmark suite for studying **accumulative error in Graph World Models (GWMs)**. A GWM autoregressively predicts a graph-structured environment $G_t = (V, E_t, X_t, A_t)$ and rolls out future states for multi-step planning. Small prediction errors in node features or edges compound over rollout horizons, producing **planning regret** — yet this problem is entirely unstudied in the graph-structured case.

We introduce:
- A **2×2 joint node-edge error propagation operator B** whose spectral radius ρ(B) governs geometric error growth
- The **Graph Error Amplification Factor (GEAF)** = ρ(A)·∏‖W_ℓ‖₂ as a topology-aware upper bound on ρ(B)
- A **Regime-Conditional Framework**: Fixed-Edge (FE) regime collapses B to block-diagonal; Dynamic-Edge (DE) regime activates full cross-coupling

---

## Rollout Pipeline

![Rollout Pipeline](figures/fig_pipeline.png)

The pipeline contrasts two regimes:
- **Fixed-Edge (FE)**: edges are fixed during rollout; B = [[L_X, 0], [0, 0]], ρ(B) ≡ L_X = 2–50
- **Dynamic-Edge (DE)**: edges are also predicted at each step; B is fully coupled, ρ(B) = 132–368 (5–10× higher)

---

## Key Results

| Claim | Result |
|---|---|
| **H4** Planning regret super-linear in horizon | Median Regret@32/Regret@1 ≥ 4.4× (all 7 topologies, strict pass) |
| **H6** DE-trained beats FE-trained on DE envs | Geomean 11× (Cohen d_z=5.08, p=1.5e-11, n=36 paired) |
| **H7 E1** Cross-coupling activated in DE regime | ρ(B) > max(L_X, M_A) in 108/108 cells, mean excess ≈155 |
| **H5 partial** B6 unique stability+accuracy | 0/21 diverged + NodeMSE@32 1690× lower than closest B2 variant |

---

## Installation

```bash
pip install -r requirements.txt
```

**Python**: 3.11+ recommended. **CUDA**: 11.8+ for GPU training.

## Quick Start

```bash
# Step 1: Generate synthetic graph data (7 topologies + agent/skill simulators)
python scripts/p2_generate_all.py --out_dir ./data

# Step 2: Train all baselines — Fixed-Edge (FE) mode
python scripts/p2_run_all_baselines.py --data_root ./data --out_dir ./results

# Step 3: Train Dynamic-Edge (DE) baselines (adds edge-prediction head)
python scripts/streamA_de_trained.py --data_root ./data --out_dir ./results/de_trained

# Step 4: Run error injection + ablation experiments (Exp 3–13, 17–23)
python scripts/p4_batch.py --data_root ./data --p2_dir ./results/p2_baselines --out_dir ./results/p4

# Step 5: Run correction, rewiring, agent workflow experiments (Exp 10–11, 14–16, 25)
python scripts/p5_p6_batch.py --data_root ./data --p2_dir ./results/p2_baselines --out_dir ./results/p5

# Step 6: Generate figures
python scripts/gen_p2_figures.py --out_dir ./results/figures
```

## Repository Structure

```
gwmerror/
├── figures/                   ← Paper figures (main + pipeline)
├── src/
│   ├── graph_generators/      # 7 topology generators (chain/tree/grid/SW/SF/star/complete)
│   ├── simulators/            # Dynamic graph, agent calling tree, platform skill graph
│   ├── metrics/               # NodeMSE, EdgeF1, GEAF, ReturnError, Regret, FPD, ...
│   └── baselines/             # B1–B6 world model implementations + DE edge head
├── scripts/                   # Training, evaluation, and figure generation scripts
├── tests/                     # Unit tests (pytest)
└── requirements.txt
```

## Experiments

All 25 experiments from the paper are reproducible:

| Phase | Experiments | Script |
|---|---|---|
| P2 | Baseline training (6 models × 7 topos × 3 seeds) | `p2_run_all_baselines.py` |
| P3 | Rollout error vs topology, horizon, scaling | `p4_batch.py` |
| P4 | Error injection (node/edge/position) | `p4_batch.py` |
| P5 | Correction, rewiring, scheduled sampling | `p5_p6_batch.py` |
| P6 | Agent calling tree, platform skill graph | `p5_p6_batch.py` |
| DE | Dynamic-edge training (H6/H7) | `streamA_de_trained.py` |

## Citation

```bibtex
@inproceedings{gwmerror2026,
  title     = {Topology-Aware Growth-Rate Prediction and Spectral-Regularization Stability
               for Graph World Model Rollouts},
  author    = {Anonymous Authors},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT License. See LICENSE.
