from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from ._common import WorldModelBase, init_weight


class ActionNodeGWM(WorldModelBase):

    def __init__(self, D: int = 8, D_a: int = 4, hidden: int = 64,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.D = D
        self.input_proj = nn.Linear(D, hidden)
        self.action_proj = nn.Linear(D_a, hidden)
        self.gnn_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, D)
        self.apply(init_weight)

    def forward_step(self, X_t, A_norm, a_t):
        B, N, _ = X_t.shape
        h = self.input_proj(X_t)
        for layer in self.gnn_layers:
            h2 = layer(h)
            if A_norm.dim() == 2:
                h = torch.relu(torch.einsum("ij,bjd->bid", A_norm, h2)) + h
            else:
                h = torch.relu(torch.einsum("bij,bjd->bid", A_norm, h2)) + h
        a_h = self.action_proj(a_t).unsqueeze(1)
        h_attn, _ = self.cross_attn(query=h, key=a_h, value=a_h)
        h = self.ln1(h + h_attn)
        return X_t + self.out_proj(h)

    def gnn_W(self) -> List[np.ndarray]:
        Ws = []
        for layer in self.gnn_layers:
            W = layer.weight.detach().cpu().numpy().astype(np.float32)
            d = min(W.shape)
            Ws.append(W[:d, :d])
        return Ws
