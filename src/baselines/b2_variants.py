"""B2 GCN variants used in the B6 ablation comparison.

Variants:
- B2GCN_wd: B2 GCN backbone (regularization at optimizer side via Adam weight_decay)
- B2GCN_clip: B2 GCN backbone (regularization at trainer side via tighter clip)
- B2GCN_specproj: B2 GCN backbone + post-step spectral-norm projection
"""
from __future__ import annotations

import torch

from .b2_gcn import GCNWorldModel


class B2GCN_wd(GCNWorldModel):
    """B2 GCN with weight_decay applied via optimizer."""
    pass


class B2GCN_clip(GCNWorldModel):
    """B2 GCN with tighter clip_grad_norm via trainer."""
    pass


class B2GCN_specproj(GCNWorldModel):
    """B2 GCN with post-step spectral-norm projection on GCN layer weights."""

    @torch.no_grad()
    def project_spectral_norm(self, target_spec: float = 1.0):
        """Project each GCN layer weight to spectral norm ≤ target_spec via SVD truncation."""
        for layer in self.gcn_layers:
            W = layer.weight.data
            try:
                U, s, Vh = torch.linalg.svd(W, full_matrices=False)
                if s[0] > target_spec:
                    s_capped = torch.clamp(s, max=target_spec)
                    W_proj = U @ torch.diag(s_capped) @ Vh
                    layer.weight.data.copy_(W_proj)
            except Exception:
                pass
