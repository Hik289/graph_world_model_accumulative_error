"""B6 Error-Aware GWM (ours): GCN backbone + multi-step rollout loss + spectral reg
+ critical-node weighted correction (during training).

Implementation details:
  - ``R_critical`` uses ``critical_node_weighted_loss``.
  - Spectral regularization uses four power iterations and a warm-start buffer.
  - ``gnn_W()`` uses an SVD-based top-d×d reduction.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from ._common import WorldModelBase, init_weight


class ErrorAwareGWM(WorldModelBase):
    """B2 GCN backbone with audit-patched error-aware loss components.

    Forward signature unchanged. Loss components computed externally by trainer:

      total = L_1step
            + lam_rollout  * L_rollout       (k-step teacher-free MSE)
            + lam_spectral * R_spectral      (spectral norm push toward 1.0)
            + lam_critical * R_critical      (degree-weighted node MSE)
    """

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
        # Power-iteration warm-start buffers (Item 2 patch): one per GNN layer
        # 持久化 u 跨 forward 调用, 让 spec norm 估计稳定 + 更快收敛
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
        """Σ_ℓ (‖W_ℓ‖_2 − target_spec)² — differentiable spectral norm push.

        Item 2 patch: 2→4 power iter + register_buffer warm-start `u` for variance↓.
        """
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
            # 更新 buffer (detach 避免追踪 graph 跨 step)
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
        """R_critical = Σ_v c(v) · ‖x̂_v − x_v‖²  (per README §4.5).

        Item 3 patch (Critical): 缺失的 R_critical 现已实现.

        c(v) = degree centrality, derived from A_norm 若未显式提供.
        Returns 标量 loss term, ready to be weighted by λ_critical.

        Parameters
        ----------
        X_pred, X_true : (B, N, D) or (B, T, N, D); MSE 在 feature dim 上做.
        node_weights : (N,) or (B, N) — 若 None, 从 A_norm 自动算 normalized degree.
        A_norm : (N, N) or (B, N, N) — 用于自动算 degree weight.
        """
        # 计算 per-node squared error: shape (..., N)
        err = (X_pred - X_true).pow(2).mean(dim=-1)   # (..., N)
        if node_weights is None:
            if A_norm is None:
                # fallback: uniform (退化为 mean)
                return err.mean()
            # degree centrality: A_norm.sum(dim=-1) — 因 A_norm 已 sym normalized, sum
            # 大致表示 node-level "graph weight". normalize 到 mean=1 避免 scale 漂移.
            if A_norm.dim() == 2:
                c = A_norm.sum(dim=-1)
            else:
                c = A_norm.sum(dim=-1).mean(dim=0)
            c = c / (c.mean() + 1e-8)
            node_weights = c.to(err.device)
        # broadcast node_weights to err.shape
        if node_weights.dim() == 1:
            # (N,) → expand
            while node_weights.dim() < err.dim():
                node_weights = node_weights.unsqueeze(0)
        return (err * node_weights).mean()

    def gnn_W(self) -> List[np.ndarray]:
        """Item 4 patch: return top-d×d via SVD-based reduction (replace naive square slice).

        Spectral norm of original W is preserved when reducing to d×d via top-d singular
        components: W_reduced = U[:, :d] @ diag(s[:d]) @ Vh[:d, :].
        """
        Ws = []
        for layer in self.gnn_layers:
            W = layer.weight.detach().cpu().numpy().astype(np.float32)
            d = min(W.shape)
            if W.shape[0] == W.shape[1]:
                Ws.append(W)
            else:
                # SVD-based reduction (preserves spectral norm of top-d components)
                try:
                    U, s, Vh = np.linalg.svd(W, full_matrices=False)
                    # take top d (full svd already gives min(M,N) cnt)
                    k = min(d, len(s))
                    W_reduced = (U[:, :k] * s[:k]) @ Vh[:k]
                    # Force square: take min dim
                    if W_reduced.shape[0] >= d and W_reduced.shape[1] >= d:
                        # 保留 top-d × top-d 等价 spectral norm
                        # (top-d singular: SVD truncation)
                        W_square = (U[:d, :d] if U.shape[0] >= d else U) * s[:d]
                        if W_square.ndim == 1:
                            W_square = np.diag(W_square)
                        elif W_square.shape != (d, d):
                            # final fallback: pad / truncate
                            W_square = W_reduced[:d, :d]
                        Ws.append(W_square.astype(np.float32))
                    else:
                        Ws.append(W_reduced.astype(np.float32))
                except Exception:
                    Ws.append(W[:d, :d])
        return Ws
