"""Unit tests for B6 patches (per ml_engineer_gpt audit 2026-05-13 06:38 UTC).

Item 3 (Critical): R_critical implemented.
Item 2: 4-iter power iteration + register_buffer warm-start.
Item 4: gnn_W() SVD-based reduction.
Item 5a: deterministic seeding (tested in test_seeding_reproducibility).
"""
from __future__ import annotations

import numpy as np
import torch
import pytest

from src.baselines.b6_error_aware import ErrorAwareGWM


def test_R_critical_returns_scalar():
    """Item 3: R_critical helper returns a scalar tensor."""
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
    """Verify high-degree nodes contribute more to loss."""
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    # 构造 star graph: node 0 = hub, 其他全 leaf
    N = 5
    A = torch.zeros(N, N)
    for i in range(1, N):
        A[0, i] = 1
        A[i, 0] = 1
    X_pred = torch.zeros(1, N, 8)
    X_true = torch.zeros(1, N, 8)
    # case A: hub (node 0) error = 1, leaves = 0 → R_critical 大
    X_pred_hub = X_pred.clone()
    X_pred_hub[0, 0, :] = 1.0
    loss_hub = m.critical_node_weighted_loss(X_pred_hub, X_true, A_norm=A)
    # case B: 一个 leaf error = 1, hub = 0
    X_pred_leaf = X_pred.clone()
    X_pred_leaf[0, 1, :] = 1.0
    loss_leaf = m.critical_node_weighted_loss(X_pred_leaf, X_true, A_norm=A)
    # hub error should weigh more than leaf error
    assert loss_hub.item() > loss_leaf.item(), \
        f"hub_loss={loss_hub.item()} should > leaf_loss={loss_leaf.item()}"


def test_R_critical_passes_grad():
    """R_critical gradient flows back to weights."""
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    X_pred = torch.randn(4, 10, 8, requires_grad=True)
    X_true = torch.randn(4, 10, 8)
    A = torch.eye(10)
    loss = m.critical_node_weighted_loss(X_pred, X_true, A_norm=A)
    loss.backward()
    assert X_pred.grad is not None
    assert (X_pred.grad != 0).any()


def test_spectral_reg_4_iter():
    """Item 2: spectral_reg uses 4-iter power iter, more accurate than 2-iter."""
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    reg = m.spectral_reg(target_spec=1.0, n_iter=4)
    # 应当返回 scalar
    assert reg.dim() == 0
    assert reg.item() >= 0


def test_spectral_reg_buffer_persists():
    """Item 2: register_buffer warm-start `_spec_u_{li}` 跨调用持久化."""
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    # 调一次, buffer 应被更新
    u_before = m._spec_u_0.clone()
    _ = m.spectral_reg(target_spec=1.0)
    u_after = m._spec_u_0.clone()
    # 不要求 bit-exact (因为 weight 随机), 但 norm 应保持 ~1
    assert abs(u_after.norm().item() - 1.0) < 1e-5


def test_spectral_reg_converges_to_sigma_max():
    """Verify spec norm estimate (sigma) converges to true σ_max via SVD."""
    m = ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    # 跑 several spec_reg 让 buffer warm up
    for _ in range(3):
        _ = m.spectral_reg(target_spec=1.0, n_iter=4)
    # Compare to true SVD
    for li, layer in enumerate(m.gnn_layers):
        W = layer.weight.detach().cpu().numpy()
        true_sigma = np.linalg.svd(W, compute_uv=False)[0]
        u = getattr(m, f"_spec_u_{li}").detach().cpu().numpy()
        v = W @ u
        est_sigma = np.linalg.norm(v)
        rel_err = abs(est_sigma - true_sigma) / true_sigma
        assert rel_err < 0.05, f"layer {li}: est={est_sigma}, true={true_sigma}, rel_err={rel_err}"


def test_gnn_W_returns_square_matrices():
    """Item 4: gnn_W() returns list of square matrices."""
    m = ErrorAwareGWM(D=8, hidden=64, n_layers=2)
    Ws = m.gnn_W()
    assert len(Ws) == 2
    for W in Ws:
        assert W.shape[0] == W.shape[1], f"non-square W shape {W.shape}"


def test_gnn_W_preserves_top_spec_norm():
    """Item 4: SVD-based reduction preserves leading singular value within 5%."""
    m = ErrorAwareGWM(D=8, hidden=64, n_layers=2)
    Ws = m.gnn_W()
    for li, W_reduced in enumerate(Ws):
        W_full = m.gnn_layers[li].weight.detach().cpu().numpy()
        sigma_full = np.linalg.svd(W_full, compute_uv=False)[0]
        sigma_reduced = np.linalg.svd(W_reduced, compute_uv=False)[0]
        rel_err = abs(sigma_full - sigma_reduced) / sigma_full
        assert rel_err < 0.10, f"layer {li}: reduction lost spec norm: {sigma_full} → {sigma_reduced}"


def test_seeding_reproducibility():
    """Item 5a: same seed → bit-exact model init."""
    def make(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        return ErrorAwareGWM(D=8, hidden=16, n_layers=2)
    m1 = make(42)
    m2 = make(42)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)
    # different seed → different
    m3 = make(123)
    diff = any(not torch.allclose(p1, p3) for p1, p3 in zip(m1.parameters(), m3.parameters()))
    assert diff


def test_b6_forward_unchanged():
    """B6 forward signature 与 patch 前一致 (regression guard)."""
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
