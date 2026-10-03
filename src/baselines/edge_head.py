from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import WorldModelBase, init_weight


class EdgePredictionHead(nn.Module):

    def __init__(self, D: int = 8, hidden: int = 16):
        super().__init__()
        self.node_proj = nn.Linear(D, hidden)
        self.Q = nn.Parameter(torch.randn(hidden, hidden) / np.sqrt(hidden))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.node_proj.weight)
        nn.init.zeros_(self.node_proj.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.node_proj(X)
        logits = torch.einsum('bnd,de,bme->bnm', h, self.Q, h) + self.bias
        return logits

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(X)
            return (torch.sigmoid(logits) > threshold).float()

    def get_lipschitz(self) -> float:
        with torch.no_grad():
            W_in = self.node_proj.weight
            s_W = torch.linalg.svd(W_in, full_matrices=False).S[0].item()
            s_Q = torch.linalg.svd(self.Q, full_matrices=False).S[0].item()
            return 0.25 * 2.0 * s_W * s_Q


class WorldModelWithEdgeHead(nn.Module):

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
        X_traj = [X_0]
        A_logits_traj = [] if return_edges else None
        A_curr = A_0_norm
        for t in range(T):
            a_t = actions[:, t]
            X_next, A_logits = self.forward_step(X_traj[-1], A_curr, a_t)
            X_traj.append(X_next)
            if return_edges:
                A_logits_traj.append(A_logits)
            A_soft = torch.sigmoid(A_logits)
            if A_soft.dim() == 3:
                A_sym = 0.5 * (A_soft + A_soft.transpose(-1, -2))
                N = A_sym.shape[-1]
                I = torch.eye(N, device=A_sym.device).unsqueeze(0)
                A_with_self = A_sym + I
                d = A_with_self.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                d_inv_sqrt = d.pow(-0.5)
                A_curr = A_with_self * d_inv_sqrt * d_inv_sqrt.transpose(-1, -2)
        X_traj = torch.stack(X_traj, dim=1)
        if return_edges and len(A_logits_traj) > 0:
            A_logits_traj = torch.stack(A_logits_traj, dim=1)
        return X_traj, A_logits_traj

    def gnn_W(self) -> List[np.ndarray]:
        return self.node_model.gnn_W()

    def get_L_g(self) -> float:
        return self.edge_head.get_lipschitz()
