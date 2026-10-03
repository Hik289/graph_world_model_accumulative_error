from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from ._common import WorldModelBase, init_weight


class MLPWorldModel(WorldModelBase):
    def __init__(self, N: int, D: int = 8, D_a: int = 4, hidden: int = 256):
        super().__init__()
        if N > 50:
            pass
        self.N = N
        self.D = D
        self.D_a = D_a
        in_dim = N * D + D_a
        out_dim = N * D
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.apply(init_weight)

    def forward_step(self, X_t: torch.Tensor, A_norm: torch.Tensor,
                     a_t: torch.Tensor) -> torch.Tensor:
        B = X_t.shape[0]
        flat = torch.cat([X_t.reshape(B, -1), a_t], dim=-1)
        out = self.net(flat).reshape(B, self.N, self.D)
        return out

    def gnn_W(self) -> List[np.ndarray]:
        last = self.net[-1]
        W = last.weight.detach().cpu().numpy().astype(np.float32)
        D = self.D
        W_sub = W[:D, :D] if W.shape[0] >= D and W.shape[1] >= D else W
        return [W_sub]
