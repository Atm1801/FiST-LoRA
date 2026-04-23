"""Truncated SVD, gradient projection and magnitude control of R_init."""

import pytest
import torch

from fist_lora.calibration.fisher import fisher_weighted_matrix
from fist_lora.calibration.inner_init import (
    gaussian_R,
    gradient_projected_R,
    normalize_to_scale,
    projected_gradient,
    sigma_diag_R,
)
from fist_lora.calibration.subspace import truncated_svd

SHAPES = [(64, 32, 4), (32, 64, 8), (48, 48, 16), (40, 17, 17)]


@pytest.mark.parametrize("d,k,r", SHAPES)
def test_svd_shapes_orthonormality_and_truncation(d, k, r):
    W = torch.randn(d, k, generator=torch.Generator().manual_seed(d * k))
    svd = truncated_svd(W, r)
    assert svd.U.shape == (d, r) and svd.S.shape == (r,) and svd.Vh.shape == (r, k)
    assert torch.allclose(svd.U.T @ svd.U, torch.eye(r), atol=1e-5)  # B^T B = I
    assert torch.allclose(svd.Vh @ svd.Vh.T, torch.eye(r), atol=1e-5)  # A A^T = I
    assert torch.all(svd.S[:-1] >= svd.S[1:])
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    assert torch.allclose(svd.S.double(), S[:r], rtol=1e-4)
    # Same rank-r reconstruction (Eckart-Young) as the full SVD truncated by hand.
    assert torch.allclose((svd.U * svd.S) @ svd.Vh, ((U[:, :r] * S[:r]) @ Vh[:r]).float(), atol=1e-4)
    small = svd.truncate(max(1, r // 2))
    assert torch.equal(small.U, svd.U[:, : max(1, r // 2)])


def test_svd_rejects_bad_input():
    with pytest.raises(ValueError):
        truncated_svd(torch.randn(4, 5), 6)
    with pytest.raises(ValueError):
        truncated_svd(torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), 1)


def test_svd_accepts_bf16_input_via_fp32():
    W = torch.randn(16, 12).bfloat16()
    svd = truncated_svd(W, 4)
    assert svd.U.dtype == torch.float32


def test_uniform_fisher_recovers_plain_subspace():
    W = torch.randn(30, 20, generator=torch.Generator().manual_seed(3))
    plain = truncated_svd(W, 5)
    weighted = truncated_svd(fisher_weighted_matrix(W, torch.full_like(W, 0.7)), 5)
    # Projectors onto the column spaces coincide.
    assert torch.allclose(plain.U @ plain.U.T, weighted.U @ weighted.U.T, atol=1e-4)


def test_fisher_weighting_changes_subspace():
    W = torch.randn(30, 20, generator=torch.Generator().manual_seed(4))
    F = torch.rand(30, 20, generator=torch.Generator().manual_seed(5)) ** 6
    plain, weighted = truncated_svd(W, 3), truncated_svd(fisher_weighted_matrix(W, F), 3)
    assert not torch.allclose(plain.U @ plain.U.T, weighted.U @ weighted.U.T, atol=1e-2)


def test_projection_formula_on_hand_built_matrices():
    B = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # d=3, r=2
    A = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])  # r=2, k=4
    G = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    # B^T G A^T picks rows {0,1} and columns {1,3} of G.
    assert torch.equal(projected_gradient(B, G, A), torch.tensor([[1.0, 3.0], [5.0, 7.0]]))
    R = gradient_projected_R(B, G, A, init_scale=0.01)
    assert torch.linalg.norm(R).item() == pytest.approx(0.01, rel=1e-6)
    assert torch.allclose(R / torch.linalg.norm(R), projected_gradient(B, G, A) / torch.linalg.norm(projected_gradient(B, G, A)))


@pytest.mark.parametrize("d,k,r", SHAPES)
@pytest.mark.parametrize("alpha,gamma", [(16, 0.01), (32, 0.01), (16, 0.1)])
def test_initial_perturbation_is_exactly_alpha_over_r_gamma(d, k, r, alpha, gamma):
    """||(alpha/r) B R_init A||_F = (alpha/r) gamma exactly for orthonormal B, A."""
    g = torch.Generator().manual_seed(r)
    svd = truncated_svd(torch.randn(d, k, generator=g), r)
    G = torch.randn(d, k, generator=g)
    R = gradient_projected_R(svd.U, G, svd.Vh, gamma)
    delta = (alpha / r) * svd.U @ R @ svd.Vh
    assert torch.linalg.norm(delta).item() == pytest.approx(alpha / r * gamma, rel=1e-4)


def test_update_invariant_to_svd_sign_flips():
    g = torch.Generator().manual_seed(7)
    W, G = torch.randn(20, 12, generator=g), torch.randn(20, 12, generator=g)
    svd = truncated_svd(W, 4)
    flips = torch.tensor([1.0, -1.0, -1.0, 1.0])
    U2, Vh2 = svd.U * flips, svd.Vh * flips[:, None]  # joint flips (u_i, v_i) -> (-u_i, -v_i)
    d1 = svd.U @ gradient_projected_R(svd.U, G, svd.Vh, 0.01) @ svd.Vh
    d2 = U2 @ gradient_projected_R(U2, G, Vh2, 0.01) @ Vh2
    assert torch.allclose(d1, d2, atol=1e-7)


def test_zero_or_nonfinite_projection_rejected():
    with pytest.raises(ValueError, match="zero Frobenius norm"):
        normalize_to_scale(torch.zeros(3, 3), 0.01)
    with pytest.raises(ValueError):
        normalize_to_scale(torch.tensor([[float("inf")]]), 0.01)


def test_sigma_diag_and_gaussian_inits():
    R = sigma_diag_R(torch.tensor([3.0, 2.0, 1.0]), 0.01)
    assert torch.equal(R, torch.diag(torch.diagonal(R)))
    assert torch.linalg.norm(R).item() == pytest.approx(0.01)
    a = gaussian_R(64, 1e-5, torch.Generator().manual_seed(0))
    b = gaussian_R(64, 1e-5, torch.Generator().manual_seed(0))
    assert torch.equal(a, b)
    assert a.std().item() == pytest.approx(1e-5, rel=0.05)
