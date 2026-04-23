"""Initialisations of the trainable inner matrix R (r x r)."""

from __future__ import annotations

import torch


def projected_gradient(B: torch.Tensor, G: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """R_proj = B^T G A^T, i.e. the gradient expressed in the frozen (B, A) basis."""
    return B.T.float() @ G.float() @ A.T.float()


def normalize_to_scale(R_proj: torch.Tensor, init_scale: float) -> torch.Tensor:
    """R_init = gamma * R_proj / ||R_proj||_F.

    With orthonormal B and A (B^T B = I, A A^T = I) this fixes ||s B R_init A||_F = s * gamma
    exactly; the gradient contributes the direction only.  A zero projection has no
    direction, so it is an error rather than a silent fallback.
    """
    if not torch.isfinite(R_proj).all():
        raise ValueError("projected gradient contains non-finite values")
    norm = torch.linalg.norm(R_proj)
    if norm.item() == 0.0:
        raise ValueError("projected gradient has zero Frobenius norm; its direction is undefined")
    return init_scale * R_proj / norm


def gradient_projected_R(B: torch.Tensor, G: torch.Tensor, A: torch.Tensor, init_scale: float) -> torch.Tensor:
    return normalize_to_scale(projected_gradient(B, G, A), init_scale)


def zero_R(rank: int) -> torch.Tensor:
    return torch.zeros(rank, rank)


def gaussian_R(rank: int, std: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """R ~ N(0, std^2), the original LoRA-XS initialisation (std = 1e-5)."""
    return torch.randn(rank, rank, generator=generator) * std


def sigma_diag_R(singular_values: torch.Tensor, init_scale: float) -> torch.Tensor:
    """Ablation ('diagonal-Sigma'): diag(S_r) rescaled to Frobenius norm gamma."""
    return normalize_to_scale(torch.diag(singular_values.float()), init_scale)
