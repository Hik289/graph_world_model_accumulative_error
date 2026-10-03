from __future__ import annotations

import numpy as np
import torch

from src.baselines.b6_error_aware import ErrorAwareGWM


def test_R_critical_returns_scalar():
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    X_pred = torch.randn(4, 10, 8)
    X_true = torch.randn(4, 10, 8)
    A = torch.eye(10) + torch.diag(torch.ones(9), diagonal=1)
    A = A + A.T
    A = torch.clamp(A, max=1.0) - torch.eye(10)
    A = torch.clamp(A, min=0.0)
    loss = m.critical_node_weighted_loss(X_pred, X_true, node_weights=None, A_norm=A)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_R_critical_weighted_by_degree():
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    N = 5
    A = torch.zeros(N, N)
    for i in range(1, N):
        A[0, i] = 1
        A[i, 0] = 1
    X_pred = torch.zeros(1, N, 8)
    X_true = torch.zeros(1, N, 8)
    X_pred_hub = X_pred.clone()
    X_pred_hub[0, 0, :] = 1.0
    loss_hub = m.critical_node_weighted_loss(X_pred_hub, X_true, A_norm=A)
    X_pred_leaf = X_pred.clone()
    X_pred_leaf[0, 1, :] = 1.0
    loss_leaf = m.critical_node_weighted_loss(X_pred_leaf, X_true, A_norm=A)
    assert loss_hub.item() > loss_leaf.item(), \
        f"hub_loss={loss_hub.item()} should > leaf_loss={loss_leaf.item()}"


def test_R_critical_passes_grad():
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    X_pred = torch.randn(4, 10, 8, requires_grad=True)
    X_true = torch.randn(4, 10, 8)
    A = torch.eye(10)
    loss = m.critical_node_weighted_loss(X_pred, X_true, A_norm=A)
    loss.backward()
    assert X_pred.grad is not None
    assert (X_pred.grad != 0).any()


def test_spectral_reg_4_iter():
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    reg = m.spectral_reg(target_spec=1.0, n_iter=4)
    assert reg.dim() == 0
    assert reg.item() >= 0


def test_spectral_reg_buffer_persists():
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    _ = m.spectral_reg(target_spec=1.0)
    u_after = m._spec_u_0.clone()
    assert abs(u_after.norm().item() - 1.0) < 1e-5


def test_spectral_reg_converges_to_sigma_max():
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    for _ in range(3):
        _ = m.spectral_reg(target_spec=1.0, n_iter=4)
    for li, layer in enumerate(m.gnn_layers):
        W = layer.weight.detach().cpu().numpy()
        true_sigma = np.linalg.svd(W, compute_uv=False)[0]
        u = getattr(m, f"_spec_u_{li}").detach().cpu().numpy()
        v = W @ u
        est_sigma = np.linalg.norm(v)
        rel_err = abs(est_sigma - true_sigma) / true_sigma
        assert rel_err < 0.05, f"layer {li}: est={est_sigma}, true={true_sigma}, rel_err={rel_err}"


def test_gnn_W_returns_square_matrices():
    m = ErrorAwareGWM(D=8, hidden=64, n_layers=2)
    Ws = m.gnn_W()
    assert len(Ws) == 2
    for W in Ws:
        assert W.shape[0] == W.shape[1], f"non-square W shape {W.shape}"


def test_gnn_W_preserves_top_spec_norm():
    m = ErrorAwareGWM(D=8, hidden=64, n_layers=2)
    Ws = m.gnn_W()
    for li, W_reduced in enumerate(Ws):
        W_full = m.gnn_layers[li].weight.detach().cpu().numpy()
        sigma_full = np.linalg.svd(W_full, compute_uv=False)[0]
        sigma_reduced = np.linalg.svd(W_reduced, compute_uv=False)[0]
        rel_err = abs(sigma_full - sigma_reduced) / sigma_full
        assert rel_err < 0.10, f"layer {li}: reduction lost spec norm: {sigma_full} → {sigma_reduced}"


def test_seeding_reproducibility():
    def make(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        return ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    m1 = make(42)
    m2 = make(42)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)
    m3 = make(123)
    diff = any(not torch.allclose(p1, p3) for p1, p3 in zip(m1.parameters(), m3.parameters()))
    assert diff


def test_b6_forward_unchanged():
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    X = torch.randn(2, 5, 8)
    A_norm = torch.eye(5)
    a = torch.randn(2, 4)
    Y = m.forward_step(X, A_norm, a)
    assert Y.shape == X.shape


def test_b6_rollout_predict_unchanged():
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    X0 = torch.randn(2, 5, 8)
    A_norm = torch.eye(5)
    actions = torch.randn(2, 10, 4)
    traj = m.rollout_predict(X0, A_norm, actions, T=10)
    assert traj.shape == (2, 11, 5, 8)
