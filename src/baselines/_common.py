"""Baseline 共用工具.

每个 baseline 都要暴露:
  - .forward_step(X_t, A_norm, a_t) -> X_{t+1}_pred
  - .rollout_predict(X_0, A_norm, actions, T) -> X_traj_pred  (T+1, N, D)
  - .gnn_W() -> List[np.ndarray]   # 用于 GEAF/theory_constants 落盘的权重矩阵列表
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class WorldModelBase(nn.Module):
    """Abstract base. 子类必须实现 forward_step + gnn_W."""

    def forward_step(self, X_t: torch.Tensor, A_norm: torch.Tensor,
                     a_t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def rollout_predict(self, X_0: torch.Tensor, A_norm: torch.Tensor,
                        actions: torch.Tensor, T: int) -> torch.Tensor:
        """Teacher-free rollout: 以自身预测作为下一步输入.

        X_0     : (B, N, D)
        A_norm  : (N, N) 或 (B, N, N)
        actions : (B, T, D_a)
        Returns : (B, T+1, N, D)
        """
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
        """返回用于 GEAF/B-operator 的 W 矩阵列表 (CPU, float32 ndarray)."""
        return []


def init_weight(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
