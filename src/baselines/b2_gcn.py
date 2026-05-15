"""B2 GCN-WM: 2-layer GCN with action injection."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from ._common import WorldModelBase, init_weight


class GCNWorldModel(WorldModelBase):
    def __init__(self, D: int = 8, D_a: int = 4, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.D = D
        self.D_a = D_a
        self.n_layers = n_layers
        layers = []
        in_d = D
        for _ in range(n_layers):
            layers.append(nn.Linear(in_d, hidden))
            in_d = hidden
        self.gcn_layers = nn.ModuleList(layers)
        self.out_proj = nn.Linear(hidden, D)
        self.action_proj = nn.Linear(D_a, hidden)
        self.apply(init_weight)

    def forward_step(self, X_t: torch.Tensor, A_norm: torch.Tensor,
                     a_t: torch.Tensor) -> torch.Tensor:
        # X_t (B, N, D), A_norm (N, N) or (B, N, N), a_t (B, D_a)
        h = X_t
        for layer in self.gcn_layers:
            h = layer(h)                                           # (B, N, hidden)
            if A_norm.dim() == 2:
                h = torch.einsum("ij,bjd->bid", A_norm, h)
            else:
                h = torch.einsum("bij,bjd->bid", A_norm, h)
            h = torch.relu(h)
        # action 广播
        a_h = self.action_proj(a_t).unsqueeze(1)                  # (B, 1, hidden)
        h = h + a_h
        out = self.out_proj(h)                                     # (B, N, D)
        return X_t + out  # residual

    def gnn_W(self) -> List[np.ndarray]:
        """每层 GCN 的 Linear weight (D × D / D × hidden / hidden × D)."""
        Ws = []
        for layer in self.gcn_layers:
            W = layer.weight.detach().cpu().numpy().astype(np.float32)
            # 取 (in, in) 最大方阵子矩阵作 spectral norm 估计
            d = min(W.shape)
            Ws.append(W[:d, :d])
        return Ws
