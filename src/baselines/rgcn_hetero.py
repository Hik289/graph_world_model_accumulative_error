"""R-GCN baseline for heterogeneous graphs (agent_calling_tree, platform_skill_graph).

Per data/specs/baseline_dataset_matrix.md §2: B2 → R-GCN replacement on hetero testbeds.

Architecture (Schlichtkrull ESWC 2018):
  h_v^{l+1} = σ( Σ_τ Σ_{u∈N_τ(v)} W_τ^{(l)} h_u^{(l)} + W_self^{(l)} h_v^{(l)} )

Implementation:
- node features (B, N, D), edge_index (2, E), edge_type (E,)
- Per edge type τ: separate W_τ; W_self for self-loop
- Aggregate via scatter_mean per type
- Forward signature: (X_t, edge_index, edge_type, a_t) → X_{t+1}
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNHetero(nn.Module):
    """R-GCN baseline with action injection.

    Note: edge_index/edge_type are per-instance (not batched) — so forward处理 1 instance at a time
    or batch with same graph. For simplicity, batch=1 per forward call (instance-wise).
    """

    def __init__(self, D: int = 8, D_a: int = 4, n_node_types: int = 9,
                 n_edge_types: int = 6, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.D = D
        self.n_layers = n_layers
        self.n_edge_types = n_edge_types
        # Input projection (with optional node-type embedding addition)
        self.input_proj = nn.Linear(D, hidden)
        self.node_type_emb = nn.Embedding(n_node_types, hidden)
        # Per-edge-type W matrices for each layer (n_layers × n_edge_types × Linear)
        self.W_edge_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(hidden, hidden, bias=False)
                           for _ in range(n_edge_types)])
            for _ in range(n_layers)
        ])
        self.W_self_layers = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden, D)
        self.action_proj = nn.Linear(D_a, hidden)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_step(self, X_t: torch.Tensor, edge_index: torch.Tensor,
                     edge_type: torch.Tensor, a_t: torch.Tensor,
                     node_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """X_t: (B, N, D); edge_index: (2, E); edge_type: (E,); a_t: (B, D_a).

        Returns X_{t+1}: (B, N, D).

        ⚠️ Per data_scientist Q2 decision (2026-05-13 06:43 UTC) + Director 13:25 UTC:
        - dim 0 (success_prob) ∈ [0,1]: sigmoid head
        - dim 5 (error_flag) ∈ {0,1}: sigmoid then 0/1 binarization done in eval (not training)
        - dims 1-4, 6-7 (continuous): tanh-style residual (free range)
        """
        B, N, _ = X_t.shape
        h = self.input_proj(X_t)  # (B, N, hidden)
        if node_types is not None:
            type_emb = self.node_type_emb(node_types.long())  # (N, hidden)
            h = h + type_emb.unsqueeze(0)  # broadcast over batch
        for li in range(self.n_layers):
            # Self
            h_self = self.W_self_layers[li](h)
            # Per edge type
            msg = torch.zeros_like(h)
            src = edge_index[0]  # (E,)
            dst = edge_index[1]  # (E,)
            for tau in range(self.n_edge_types):
                mask = (edge_type == tau)
                if not mask.any():
                    continue
                src_t = src[mask]
                dst_t = dst[mask]
                W_t = self.W_edge_layers[li][tau]
                h_src = h[:, src_t]  # (B, E_tau, hidden)
                m = W_t(h_src)  # (B, E_tau, hidden)
                msg.index_add_(1, dst_t, m)
            h = F.relu(h_self + msg)
        # action injection (broadcast)
        a_h = self.action_proj(a_t).unsqueeze(1)  # (B, 1, hidden)
        h = h + a_h
        # Output projection
        delta = self.out_proj(h)  # (B, N, D)
        X_next = X_t + delta
        # ⚠️ Issue 1 fix: clip sr (dim 0) and error_flag (dim 5) to [0,1] via sigmoid
        # Use sigmoid on the predicted sr / error_flag absolute values, not residuals,
        # to keep them in valid Bernoulli-parameter range.
        # We sigmoid the FULL X_next[:,:,[0,5]] (model's prediction of probability).
        sp = torch.sigmoid(X_next[..., 0:1])
        ef = torch.sigmoid(X_next[..., 5:6])
        # Re-assemble: dims [0]=sp, [1..4]=tanh-bounded continuous, [5]=ef, [6..7]=continuous
        # tanh on continuous dims to keep them bounded [-1,1] consistent with simulator output range
        cont_1_4 = torch.tanh(X_next[..., 1:5])
        cont_6_7 = torch.tanh(X_next[..., 6:8])
        X_next = torch.cat([sp, cont_1_4, ef, cont_6_7], dim=-1)
        return X_next

    def rollout_predict(self, X_0: torch.Tensor, edge_index: torch.Tensor,
                        edge_type: torch.Tensor, actions: torch.Tensor,
                        T: int, node_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Teacher-free rollout: (B, N, D) initial state → (B, T+1, N, D) trajectory."""
        traj = [X_0]
        X = X_0
        for t in range(T):
            a_t = actions[:, t]
            X = self.forward_step(X, edge_index, edge_type, a_t, node_types=node_types)
            traj.append(X)
        return torch.stack(traj, dim=1)

    def gnn_W(self) -> List[np.ndarray]:
        """Return W_self matrices per layer for theory_constants."""
        Ws = []
        for layer in self.W_self_layers:
            W = layer.weight.detach().cpu().numpy().astype(np.float32)
            d = min(W.shape)
            Ws.append(W[:d, :d])
        return Ws
