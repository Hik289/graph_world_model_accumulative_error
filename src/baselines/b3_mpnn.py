"""B3 MPNN-WM: edge-conditioned message passing (with edge features = A weights)."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from ._common import WorldModelBase, init_weight


class MPNNWorldModel(WorldModelBase):
    def __init__(self, D: int = 8, D_a: int = 4, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.D = D
        self.n_layers = n_layers
        self.input_proj = nn.Linear(D, hidden)
        # message MLPs (一个 per layer)
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_layers)
        ])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden, D)
        self.action_proj = nn.Linear(D_a, hidden)
        self.apply(init_weight)

    def forward_step(self, X_t: torch.Tensor, A_norm: torch.Tensor,
                     a_t: torch.Tensor) -> torch.Tensor:
        B, N, _ = X_t.shape
        h = self.input_proj(X_t)                       # (B, N, H)
        for li in range(self.n_layers):
            # message: m_{ij} = MLP([h_i, h_j])
            h_i = h.unsqueeze(2).expand(B, N, N, -1)   # (B, N, N, H)
            h_j = h.unsqueeze(1).expand(B, N, N, -1)
            msg_in = torch.cat([h_i, h_j], dim=-1)     # (B, N, N, 2H)
            m_all = self.msg_mlps[li](msg_in)          # (B, N, N, H)
            # 加权聚合 (A_norm 作为边权)
            if A_norm.dim() == 2:
                A_w = A_norm.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
            else:
                A_w = A_norm.unsqueeze(-1)
            m_agg = (m_all * A_w).sum(dim=2)           # (B, N, H)
            upd_in = torch.cat([h, m_agg], dim=-1)
            h = h + self.update_mlps[li](upd_in)
        a_h = self.action_proj(a_t).unsqueeze(1)
        h = h + a_h
        out = self.out_proj(h)
        return X_t + out

    def gnn_W(self) -> List[np.ndarray]:
        Ws = []
        # 各 layer 的最后线性层 weight 作 spectral surrogate
        for layer in self.msg_mlps:
            last = list(layer.children())[-1]
            W = last.weight.detach().cpu().numpy().astype(np.float32)
            d = min(W.shape)
            Ws.append(W[:d, :d])
        return Ws
