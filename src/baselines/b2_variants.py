from __future__ import annotations

import torch

from .b2_gcn import GCNWorldModel


class B2GCN_wd(GCNWorldModel):
    pass


class B2GCN_clip(GCNWorldModel):
    pass


class B2GCN_specproj(GCNWorldModel):

    @torch.no_grad()
    def project_spectral_norm(self, target_spec: float = 1.0):
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
