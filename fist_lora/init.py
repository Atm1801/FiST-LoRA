"""
Initialization functions for FiST-LoRA.

Provides:
- fisher_weighted_svd: Fisher-information-weighted SVD with outlier clipping
- plain_svd: Standard SVD baseline (LoRA-XS)
- gradient_projected_R: Gradient-projected R initialization
- zero_R: Zero initialization for R (LoRA-XS baseline)
- sigma_R: Diagonal singular-value initialization for R (ablation)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional


def fisher_weighted_svd(
    weight: torch.Tensor,       # (d, k)
    fisher_diag: torch.Tensor,  # (d, k)
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fisher-information-weighted SVD with outlier clipping.

    [PoC-VALIDATED] The clipping and normalization steps are ESSENTIAL.
    Without them, extreme Fisher values dominate the SVD and select a
    degenerate subspace that causes training to collapse to random chance
    on most tasks.

    Steps:
    1. Clip Fisher at 95th percentile: F_clipped = fisher_diag.clamp(max=quantile_95)
    2. Normalize by mean: F_norm = F_clipped / (F_clipped.mean() + 1e-8)
    3. Scale weight: W_tilde = sqrt(F_norm + 1e-8) * weight
    4. SVD: U, S, Vt = torch.linalg.svd(W_tilde, full_matrices=False)
    5. Return B=U[:,:rank], S[:rank], A=Vt[:rank,:]

    DO NOT skip the clipping. DO NOT use raw Fisher values.

    Args:
        weight: Pretrained weight matrix (d, k).
        fisher_diag: Diagonal Fisher information (d, k), same shape as weight.
        rank: Target rank for truncation.

    Returns:
        B (d, r), S (r,), A (r, k) — the truncated SVD components.
    """
    q95 = torch.quantile(fisher_diag.float(), 0.95)
    F_clipped = fisher_diag.clamp(max=q95)
    F_norm = F_clipped / (F_clipped.mean() + 1e-8)

    W_tilde = (F_norm + 1e-8).sqrt() * weight
    U, S, Vt = torch.linalg.svd(W_tilde, full_matrices=False)
    return U[:, :rank].clone(), S[:rank].clone(), Vt[:rank, :].clone()


def plain_svd(
    weight: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard SVD of weight matrix. Used for LoRA-XS baseline and ablations.

    Args:
        weight: Pretrained weight matrix (d, k).
        rank: Target rank.

    Returns:
        B (d, r), S (r,), A (r, k).
    """
    U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
    return U[:, :rank].clone(), S[:rank].clone(), Vt[:rank, :].clone()


def gradient_projected_R(
    model: nn.Module,
    dataloader: DataLoader,
    frozen_BA: Dict[str, Tuple[torch.Tensor, torch.Tensor]],  # {name: (B, A)}
    target_module_names: List[str],
    num_samples: int = 256,
    alpha: float = 32.0,
    rank: int = 32,
    init_scale: float = 0.01,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Initialize R by projecting the average gradient into the frozen (B, A) subspace.

    For each module:
    1. G = mean(dL/dW) over calibration samples
    2. R_init = B^T @ G @ A^T  (r x r)
    3. Normalize: R_init = R_init * (init_scale / ||R_init||_F)

    [PoC-VALIDATED] The init_scale parameter is CRITICAL.
    - init_scale=0.01 gives ||DeltaW|| ~ (alpha/r) * 0.01 ~ 0.04 (works)
    - The old formula alpha/(norm*sqrt(r)) gave ||DeltaW|| ~ 45 (catastrophic)
    - Direction is preserved from gradient projection; only magnitude controlled.

    Args:
        model: Model with warmed-up head.
        dataloader: Calibration data loader.
        frozen_BA: Dict mapping module name to (B, A) tensor tuples.
        target_module_names: Suffixes to match.
        num_samples: Number of calibration samples.
        alpha: LoRA scaling factor (used only for logging context).
        rank: Target rank.
        init_scale: Frobenius norm target for R_init.
        device: Device override.

    Returns:
        Dict mapping module names to R_init tensors (r, r), on CPU.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Identify target modules
    target_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(t) for t in target_module_names
        ):
            if name in frozen_BA:
                target_modules[name] = module

    if not target_modules:
        raise ValueError("No target modules found for gradient projection.")

    # Enable grad on target weight matrices temporarily
    orig_requires_grad = {}
    for name, module in target_modules.items():
        orig_requires_grad[name] = module.weight.requires_grad
        module.weight.requires_grad_(True)

    grad_accum = {
        name: torch.zeros_like(module.weight, device=device)
        for name, module in target_modules.items()
    }

    samples_seen = 0
    for batch in dataloader:
        if samples_seen >= num_samples:
            break

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k != "length"
        }

        try:
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None:
                loss = outputs.logits.sum()
            loss.backward()
        except Exception as e:
            print(f"[GradR] Warning: forward/backward failed on batch - {e}")
            model.zero_grad()
            continue

        for name, module in target_modules.items():
            if module.weight.grad is not None:
                grad_accum[name] += module.weight.grad.detach()

        model.zero_grad()
        samples_seen += batch[next(iter(batch))].size(0)

    if samples_seen == 0:
        print("[GradR] Warning: saw zero samples; falling back to zero R init.")
        return {name: zero_R(rank) for name in frozen_BA}

    # Restore requires_grad
    for name, module in target_modules.items():
        module.weight.requires_grad_(orig_requires_grad[name])

    R_dict = {}
    for name, (B, A) in frozen_BA.items():
        G = grad_accum[name] / max(samples_seen, 1)  # (d, k)

        B_dev = B.to(device)
        A_dev = A.to(device)
        G_dev = G.to(device)

        # Project: R_init = B^T G A^T  -> (r, d) @ (d, k) @ (k, r) = (r, r)
        R_init = B_dev.T @ G_dev @ A_dev.T

        # Check for numerical issues
        if not torch.isfinite(R_init).all():
            print(f"[GradR] Warning: non-finite R_init for {name}; using zero init.")
            R_dict[name] = zero_R(rank)
            continue

        # Normalize to unit direction, then scale to init_scale.
        norm = R_init.norm()
        if norm > 1e-12:
            R_init = R_init * (init_scale / norm)
        else:
            print(f"[GradR] Near-zero R_init for {name}; using zero init.")
            R_dict[name] = zero_R(rank)
            continue

        R_dict[name] = R_init.cpu()

    return R_dict


def zero_R(rank: int) -> torch.Tensor:
    """Return zeros(rank, rank). LoRA-XS default init."""
    return torch.zeros(rank, rank)


def sigma_R(singular_values: torch.Tensor, rank: int, init_scale: float = 0.01) -> torch.Tensor:
    """
    Initialize R as a scaled diagonal of singular values (ablation variant).

    R = diag(S[:rank]) normalized to init_scale Frobenius norm.

    Args:
        singular_values: Singular values from SVD (at least rank entries).
        rank: Target rank.
        init_scale: Target Frobenius norm.

    Returns:
        R tensor (rank, rank).
    """
    R = torch.diag(singular_values[:rank].float())
    norm = R.norm()
    if norm > 1e-12:
        R = R * (init_scale / norm)
    return R
