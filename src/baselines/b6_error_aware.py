from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from ._common import WorldModelBase, init_weight


class ErrorAwareGWM(WorldModelBase):

    def __init__(self, D: int = 8, D_a: int = 4, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.D = D
        self.hidden = hidden
        self.n_layers = n_layers
        self.input_proj = nn.Linear(D, hidden)
        self.gnn_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden, D)
        self.action_proj = nn.Linear(D_a, hidden)
        self.apply(init_weight)
        for li, layer in enumerate(self.gnn_layers):
            u = torch.randn(layer.weight.shape[1])
            u = u / (u.norm() + 1e-8)
            self.register_buffer(f"_spec_u_{li}", u, persistent=False)

    def forward_step(self, X_t, A_norm, a_t):
        h = self.input_proj(X_t)
        for layer in self.gnn_layers:
            h2 = layer(h)
            if A_norm.dim() == 2:
                h = torch.relu(torch.einsum("ij,bjd->bid", A_norm, h2)) + h
            else:
                h = torch.relu(torch.einsum("bij,bjd->bid", A_norm, h2)) + h
        a_h = self.action_proj(a_t).unsqueeze(1)
        h = h + a_h
        return X_t + self.out_proj(h)

    def spectral_reg(self, target_spec: float = 1.0, n_iter: int = 4) -> torch.Tensor:
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for li, layer in enumerate(self.gnn_layers):
            W = layer.weight
            buf_name = f"_spec_u_{li}"
            u = getattr(self, buf_name).to(W.device)
            for _ in range(n_iter):
                v = W @ u
                v = v / (v.norm() + 1e-8)
                u_new = W.T @ v
                u_new = u_new / (u_new.norm() + 1e-8)
                u = u_new
            with torch.no_grad():
                setattr(self, buf_name, u.detach().clone())
            sigma = (W @ u).norm()
            reg = reg + (sigma - target_spec) ** 2
        return reg

    def critical_node_weighted_loss(
        self,
        X_pred: torch.Tensor,
        X_true: torch.Tensor,
        node_weights: Optional[torch.Tensor] = None,
        A_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        err = (X_pred - X_true).pow(2).mean(dim=-1)
        if node_weights is None:
            if A_norm is None:
                return err.mean()
            if A_norm.dim() == 2:
                c = A_norm.sum(dim=-1)
            else:
                c = A_norm.sum(dim=-1).mean(dim=0)
            c = c / (c.mean() + 1e-8)
            node_weights = c.to(err.device)
        if node_weights.dim() == 1:
            while node_weights.dim() < err.dim():
                node_weights = node_weights.unsqueeze(0)
        return (err * node_weights).mean()

    def gnn_W(self) -> List[np.ndarray]:
        Ws = []
        for layer in self.gnn_layers:
            W = layer.weight.detach().cpu().numpy().astype(np.float32)
            d = min(W.shape)
            if W.shape[0] == W.shape[1]:
                Ws.append(W)
            else:
                try:
                    U, s, Vh = np.linalg.svd(W, full_matrices=False)
                    k = min(d, len(s))
                    W_reduced = (U[:, :k] * s[:k]) @ Vh[:k]
                    if W_reduced.shape[0] >= d and W_reduced.shape[1] >= d:
                        W_square = (U[:d, :d] if U.shape[0] >= d else U) * s[:d]
                        if W_square.ndim == 1:
                            W_square = np.diag(W_square)
                        elif W_square.shape != (d, d):
                            W_square = W_reduced[:d, :d]
                        Ws.append(W_square.astype(np.float32))
                    else:
                        Ws.append(W_reduced.astype(np.float32))
                except Exception:
                    Ws.append(W[:d, :d])
        return Ws
