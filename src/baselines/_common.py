from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class WorldModelBase(nn.Module):

    def forward_step(self, X_t: torch.Tensor, A_norm: torch.Tensor,
                     a_t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def rollout_predict(self, X_0: torch.Tensor, A_norm: torch.Tensor,
                        actions: torch.Tensor, T: int) -> torch.Tensor:
        B = X_0.shape[0]
        T_traj = T
        traj = [X_0]
        X = X_0
        for t in range(T_traj):
            a_t = actions[:, t]
            X = self.forward_step(X, A_norm, a_t)
            traj.append(X)
        return torch.stack(traj, dim=1)

    def gnn_W(self) -> List[np.ndarray]:
        return []


def init_weight(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
