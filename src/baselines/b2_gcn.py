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
        h = X_t
        for layer in self.gcn_layers:
            h = layer(h)
            if A_norm.dim() == 2:
                h = torch.einsum("ij,bjd->bid", A_norm, h)
            else:
                h = torch.einsum("bij,bjd->bid", A_norm, h)
            h = torch.relu(h)
        a_h = self.action_proj(a_t).unsqueeze(1)
        h = h + a_h
        out = self.out_proj(h)
        return X_t + out

    def gnn_W(self) -> List[np.ndarray]:
        Ws = []
        for layer in self.gcn_layers:
            W = layer.weight.detach().cpu().numpy().astype(np.float32)
            d = min(W.shape)
            Ws.append(W[:d, :d])
        return Ws
