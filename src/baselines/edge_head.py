r"""Edge-prediction head wrapper for DE-trained baselines.

Design notes, 2026-05-14 21:02 UTC:
- 加 independent Bernoulli sigmoid output for $\hat A_{t+1}$
- Loss: L_total = L_X (node MSE) + λ_e * BCE(Â, A_true)
- theory_constants must include L_g (edge-prediction Lipschitz) > 0
  enabling T2 super-additivity in M_X > 0 regime (analysis note §3.4)

Implementation: separate edge_proj head computes pairwise features → sigmoid logits for adjacency.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import WorldModelBase, init_weight


class EdgePredictionHead(nn.Module):
    """Predict next-step adjacency via pairwise sigmoid logits.

    Input: node features (B, N, D)
    Output: edge logits (B, N, N), apply sigmoid for Â ∈ [0,1]

    Architecture: low-rank bilinear scoring
      Â_{ij} = sigmoid(h_i^T W_edge h_j + b)
    """

    def __init__(self, D: int = 8, hidden: int = 16):
        super().__init__()
        # Project nodes to edge-feature space
        self.node_proj = nn.Linear(D, hidden)
        # Bilinear weight Q ∈ R^{hidden x hidden}
        self.Q = nn.Parameter(torch.randn(hidden, hidden) / np.sqrt(hidden))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.node_proj.weight)
        nn.init.zeros_(self.node_proj.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, N, D) → A_pred_logits (B, N, N)."""
        h = self.node_proj(X)  # (B, N, hidden)
        # Bilinear: h @ Q @ h.T
        logits = torch.einsum('bnd,de,bme->bnm', h, self.Q, h) + self.bias
        return logits

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Hard adjacency prediction."""
        with torch.no_grad():
            logits = self.forward(X)
            return (torch.sigmoid(logits) > threshold).float()

    def get_lipschitz(self) -> float:
        """L_g = spectral norm of effective edge-prediction map.

        For bilinear h^T Q h + b form, L_g ≈ 2 * ‖node_proj.weight‖_2 * ‖Q‖_2 * R_X
        (derivative w.r.t. one node h scales with the other; symmetric bilinear → factor 2).
        Returns scalar (post-sigmoid Lipschitz upper bound = (1/4) * pre-sigmoid).
        """
        with torch.no_grad():
            W_in = self.node_proj.weight  # (hidden, D)
            s_W = torch.linalg.svd(W_in, full_matrices=False).S[0].item()
            s_Q = torch.linalg.svd(self.Q, full_matrices=False).S[0].item()
            # 0.25 = derivative of sigmoid at 0
            return 0.25 * 2.0 * s_W * s_Q


class WorldModelWithEdgeHead(nn.Module):
    """Wrap any WorldModelBase + add edge-prediction head.

    Forward returns (X_next, A_next_logits). The original model.forward_step provides X_next;
    edge_head independently predicts A from X_next (using post-update features).
    """

    def __init__(self, node_model: WorldModelBase, D: int = 8, hidden: int = 16):
        super().__init__()
        self.node_model = node_model
        self.edge_head = EdgePredictionHead(D=D, hidden=hidden)
        self.D = D

    def forward_step(self, X_t: torch.Tensor, A_norm: torch.Tensor,
                     a_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X_next = self.node_model.forward_step(X_t, A_norm, a_t)
        A_next_logits = self.edge_head(X_next)
        return X_next, A_next_logits

    def rollout_predict(self, X_0: torch.Tensor, A_0_norm: torch.Tensor,
                        actions: torch.Tensor, T: int,
                        return_edges: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Teacher-free rollout with dynamic edge update.

        At each step:
        1. X_next = node_model(X, A, a)
        2. A_next_logits = edge_head(X_next); A_next = sigmoid(logits) (soft) or threshold (hard)
        3. Re-normalize A_next for next step's GNN message passing

        Returns (X_traj, A_logits_traj):
          X_traj: (B, T+1, N, D)
          A_logits_traj: (B, T, N, N) or None
        """
        X_traj = [X_0]
        A_logits_traj = [] if return_edges else None
        A_curr = A_0_norm
        for t in range(T):
            a_t = actions[:, t]
            X_next, A_logits = self.forward_step(X_traj[-1], A_curr, a_t)
            X_traj.append(X_next)
            if return_edges:
                A_logits_traj.append(A_logits)
            # Update adjacency for next step: sigmoid + soft norm
            A_soft = torch.sigmoid(A_logits)
            # Use mean to symmetrize + add identity diag for normalization
            if A_soft.dim() == 3:
                A_sym = 0.5 * (A_soft + A_soft.transpose(-1, -2))
                # Add self-loop
                N = A_sym.shape[-1]
                I = torch.eye(N, device=A_sym.device).unsqueeze(0)
                A_with_self = A_sym + I
                # Symmetric normalize
                d = A_with_self.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                d_inv_sqrt = d.pow(-0.5)
                A_curr = A_with_self * d_inv_sqrt * d_inv_sqrt.transpose(-1, -2)
        X_traj = torch.stack(X_traj, dim=1)
        if return_edges and len(A_logits_traj) > 0:
            A_logits_traj = torch.stack(A_logits_traj, dim=1)
        return X_traj, A_logits_traj

    def gnn_W(self) -> List[np.ndarray]:
        """Delegate to underlying node model."""
        return self.node_model.gnn_W()

    def get_L_g(self) -> float:
        """Edge head Lipschitz for theory_constants."""
        return self.edge_head.get_lipschitz()
