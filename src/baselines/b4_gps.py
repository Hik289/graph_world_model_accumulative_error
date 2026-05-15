"""B4 GPS-WM: Graph Transformer (full attention + local MP residual)."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from ._common import WorldModelBase, init_weight


class GPSBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.local = nn.Linear(hidden, hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                nn.Linear(hidden * 2, hidden))
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ln3 = nn.LayerNorm(hidden)

    def forward(self, h, A_norm):
        # global attention
        h_attn, _ = self.attn(h, h, h)
        h = self.ln1(h + h_attn)
        # local MP (GCN-like)
        if A_norm.dim() == 2:
            h_local = torch.einsum("ij,bjd->bid", A_norm, self.local(h))
        else:
            h_local = torch.einsum("bij,bjd->bid", A_norm, self.local(h))
        h = self.ln2(h + h_local)
        # FFN
        h = self.ln3(h + self.ff(h))
        return h


class GraphTransformerWorldModel(WorldModelBase):
    def __init__(self, D: int = 8, D_a: int = 4, hidden: int = 64,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.D = D
        self.input_proj = nn.Linear(D, hidden)
        self.blocks = nn.ModuleList([GPSBlock(hidden, n_heads) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden, D)
        self.action_proj = nn.Linear(D_a, hidden)
        self.apply(init_weight)

    def forward_step(self, X_t, A_norm, a_t):
        h = self.input_proj(X_t)
        for block in self.blocks:
            h = block(h, A_norm)
        a_h = self.action_proj(a_t).unsqueeze(1)
        h = h + a_h
        return X_t + self.out_proj(h)

    def gnn_W(self) -> List[np.ndarray]:
        Ws = []
        for block in self.blocks:
            W = block.local.weight.detach().cpu().numpy().astype(np.float32)
            d = min(W.shape)
            Ws.append(W[:d, :d])
        return Ws
