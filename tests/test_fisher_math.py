"""Quantile, clipping, normalisation and Fisher re-weighting."""

import numpy as np
import pytest
import torch

from fist_lora.calibration.fisher import clip_and_normalize, fisher_weighted_matrix, quantile_linear


@pytest.mark.parametrize("n", [1, 2, 7, 100, 1001])
@pytest.mark.parametrize("q", [0.0, 0.05, 0.5, 0.95, 1.0])
def test_quantile_matches_numpy(n, q):
    x = torch.randn(n, generator=torch.Generator().manual_seed(n))
    assert quantile_linear(x, q).item() == pytest.approx(float(np.quantile(x.numpy().astype(np.float64), q)), rel=1e-6, abs=1e-7)


def test_quantile_matches_torch_quantile_on_matrix():
    x = torch.rand(64, 48, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(quantile_linear(x, 0.95), torch.quantile(x, 0.95))


def test_quantile_beyond_torch_quantile_size_limit():
    # torch.quantile refuses > 2**24 elements; every 7B MLP weight is larger than that.
    n = 2**24 + 3
    x = torch.arange(n, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        torch.quantile(x, 0.95)
    expected = 0.95 * (n - 1)  # linear interpolation on 0..n-1
    assert quantile_linear(x, 0.95).item() == pytest.approx(expected, rel=1e-7)


def test_clip_and_normalize_closed_form():
    F = torch.rand(20, 30, generator=torch.Generator().manual_seed(1)) ** 4  # heavy-ish tail
    q95 = np.quantile(F.numpy().astype(np.float64), 0.95)
    clipped = np.minimum(F.numpy(), q95)
    expected = clipped / (clipped.mean() + 1e-8)
    assert np.allclose(clip_and_normalize(F).numpy(), expected, rtol=1e-5, atol=1e-7)


def test_normalized_fisher_is_centered_at_one_and_bounded():
    F = torch.distributions.Pareto(1.0, 1.1).sample((50, 40))  # heavy tail
    Fb = clip_and_normalize(F)
    assert Fb.mean().item() == pytest.approx(1.0, rel=1e-5)
    q = quantile_linear(F, 0.95)
    assert Fb.max().item() == pytest.approx((q / torch.minimum(F, q).mean()).item(), rel=1e-5)
    assert (Fb > 0).all()


def test_zero_and_constant_fisher():
    zero = clip_and_normalize(torch.zeros(4, 5))
    assert torch.equal(zero, torch.zeros(4, 5))
    const = clip_and_normalize(torch.full((4, 5), 3.0))
    assert torch.allclose(const, torch.ones(4, 5), atol=1e-7)


def test_fisher_weighted_matrix_eps_inside_sqrt():
    W = torch.randn(6, 5)
    F = torch.rand(6, 5)
    Fb = clip_and_normalize(F, 0.95, 1e-8)
    assert torch.allclose(fisher_weighted_matrix(W, F), torch.sqrt(Fb + 1e-8) * W)
    # Zero Fisher: W~ = sqrt(eps) W - a rescaling that leaves the singular subspaces unchanged.
    assert torch.allclose(fisher_weighted_matrix(W, torch.zeros(6, 5)), W * 1e-4)


@pytest.mark.parametrize("bad", [torch.tensor([[1.0, float("nan")]]), torch.tensor([[1.0, float("inf")]]), torch.tensor([[1.0, -1.0]])])
def test_invalid_fisher_rejected(bad):
    with pytest.raises(ValueError):
        clip_and_normalize(bad)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        fisher_weighted_matrix(torch.randn(3, 4), torch.rand(4, 3))
