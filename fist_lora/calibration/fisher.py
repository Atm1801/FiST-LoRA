"""Fisher clipping, normalisation and re-weighting.

    q95 = Quantile(F, 0.95)
    F_bar = min(F, q95) / (mean(min(F, q95)) + eps)
    W_tilde = sqrt(F_bar + eps) (.) W0

All statistics are global over the entries of one (d, k) matrix; no row/column-wise
normalisation is involved.
"""

from __future__ import annotations

import torch


def quantile_linear(x: torch.Tensor, q: float) -> torch.Tensor:
    """Quantile with linear interpolation, identical to ``torch.quantile``/``numpy.quantile``.

    ``torch.quantile`` refuses inputs with more than 2**24 elements, and every MLP matrix
    of a 7B model is larger (4096 x 11008 = 45M), so the order statistics are taken
    with ``kthvalue`` instead.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    flat = x.reshape(-1).to(torch.float64)
    n = flat.numel()
    if n == 0:
        raise ValueError("quantile of an empty tensor")
    pos = q * (n - 1)
    lo = int(pos)  # floor for pos >= 0
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    v_lo = torch.kthvalue(flat, lo + 1).values
    v_hi = v_lo if hi == lo else torch.kthvalue(flat, hi + 1).values
    return (v_lo + frac * (v_hi - v_lo)).to(x.dtype)


def clip_and_normalize(fisher: torch.Tensor, quantile: float = 0.95, eps: float = 1e-8) -> torch.Tensor:
    """Clip at the ``quantile`` and divide by the mean of the clipped matrix (plus eps)."""
    if not torch.isfinite(fisher).all():
        raise ValueError("Fisher estimate contains non-finite values")
    if (fisher < 0).any():
        raise ValueError("Fisher estimate must be non-negative")
    f = fisher.float()
    q = quantile_linear(f, quantile)
    clipped = torch.minimum(f, q)
    return clipped / (clipped.mean() + eps)


def fisher_weighted_matrix(
    weight: torch.Tensor, fisher: torch.Tensor, quantile: float = 0.95, eps: float = 1e-8
) -> torch.Tensor:
    """sqrt(F_bar + eps) (.) W0 (element-wise; eps inside the square root)."""
    if weight.shape != fisher.shape:
        raise ValueError(f"weight {tuple(weight.shape)} and Fisher {tuple(fisher.shape)} differ in shape")
    f_bar = clip_and_normalize(fisher, quantile, eps)
    return torch.sqrt(f_bar + eps) * weight.float()
