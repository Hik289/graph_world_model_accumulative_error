"""B1 MLP-WM: flatten input → MLP, 无图结构. N>50 上 N/A."""
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
            # spec: N>50 上 MLP-WM 标 N/A
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
        # X_t: (B, N, D), a_t: (B, D_a) → flatten → (B, N·D + D_a) → out (B, N·D) → (B, N, D)
        B = X_t.shape[0]
        flat = torch.cat([X_t.reshape(B, -1), a_t], dim=-1)
        out = self.net(flat).reshape(B, self.N, self.D)
        return out

    def gnn_W(self) -> List[np.ndarray]:
        """MLP 无图 W; 用最后线性层的 weight 作为 surrogate."""
        last = self.net[-1]
        W = last.weight.detach().cpu().numpy().astype(np.float32)
        # 截到 (D, D) 大小用作 spectral norm
        D = self.D
        W_sub = W[:D, :D] if W.shape[0] >= D and W.shape[1] >= D else W
        return [W_sub]
