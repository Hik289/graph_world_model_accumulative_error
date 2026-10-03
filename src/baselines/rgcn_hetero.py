from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNHetero(nn.Module):

    def __init__(self, D: int = 8, D_a: int = 4, n_node_types: int = 9,
                 n_edge_types: int = 6, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.D = D
        self.n_layers = n_layers
        self.n_edge_types = n_edge_types
        self.input_proj = nn.Linear(D, hidden)
        self.node_type_emb = nn.Embedding(n_node_types, hidden)
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
        B, N, _ = X_t.shape
        h = self.input_proj(X_t)
        if node_types is not None:
            type_emb = self.node_type_emb(node_types.long())
            h = h + type_emb.unsqueeze(0)
        for li in range(self.n_layers):
            h_self = self.W_self_layers[li](h)
            msg = torch.zeros_like(h)
            src = edge_index[0]
            dst = edge_index[1]
            for tau in range(self.n_edge_types):
                mask = (edge_type == tau)
                if not mask.any():
                    continue
                src_t = src[mask]
                dst_t = dst[mask]
                W_t = self.W_edge_layers[li][tau]
                h_src = h[:, src_t]
                m = W_t(h_src)
                msg.index_add_(1, dst_t, m)
            h = F.relu(h_self + msg)
        a_h = self.action_proj(a_t).unsqueeze(1)
        h = h + a_h
        delta = self.out_proj(h)
        X_next = X_t + delta
        sp = torch.sigmoid(X_next[..., 0:1])
        ef = torch.sigmoid(X_next[..., 5:6])
        cont_1_4 = torch.tanh(X_next[..., 1:5])
        cont_6_7 = torch.tanh(X_next[..., 6:8])
        X_next = torch.cat([sp, cont_1_4, ef, cont_6_7], dim=-1)
        return X_next

    def rollout_predict(self, X_0: torch.Tensor, edge_index: torch.Tensor,
                        edge_type: torch.Tensor, actions: torch.Tensor,
                        T: int, node_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        traj = [X_0]
        X = X_0
        for t in range(T):
            a_t = actions[:, t]
            X = self.forward_step(X, edge_index, edge_type, a_t, node_types=node_types)
            traj.append(X)
        return torch.stack(traj, dim=1)

    def gnn_W(self) -> List[np.ndarray]:
        Ws = []
        for layer in self.W_self_layers:
            W = layer.weight.detach().cpu().numpy().astype(np.float32)
            d = min(W.shape)
            Ws.append(W[:d, :d])
        return Ws
