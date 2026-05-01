"""
Diagonal Fisher information computation for FiST-LoRA.

Computes F[i,j] = E[(dL/dW[i,j])^2] for each target linear layer,
used to weight the SVD that determines the frozen outer subspace.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional


def compute_diagonal_fisher(
    model: nn.Module,
    dataloader: DataLoader,
    target_module_names: List[str],
    num_samples: int = 256,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute diagonal Fisher information for each target linear layer.

    Algorithm:
    1. model.eval()
    2. Identify target modules by name matching (name.endswith(t) for t in target_module_names)
    3. Temporarily enable requires_grad on target weight matrices
    4. For each batch (up to num_samples):
       a. Forward pass -> loss
       b. loss.backward()
       c. Accumulate module.weight.grad.pow(2).detach()
       d. model.zero_grad()
    5. Divide by samples_seen
    6. Restore requires_grad to False
    7. Return {module_name: fisher_diagonal_tensor}

    The fisher diagonal has shape (d, k), same as the weight matrix.
    No epsilon or clipping here -- that happens in fisher_weighted_svd.

    Args:
        model: The model (should have a warmed-up head for meaningful gradients).
        dataloader: Calibration data loader.
        target_module_names: Suffixes to match (e.g., ["query", "key", "value"]).
        num_samples: Maximum number of samples to use.
        device: Device override. If None, inferred from model parameters.

    Returns:
        Dictionary mapping full module names to Fisher diagonal tensors.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Identify target modules by full name
    target_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(t) for t in target_module_names
        ):
            target_modules[name] = module

    if not target_modules:
        raise ValueError(
            f"No Linear modules found matching target_module_names: {target_module_names}"
        )

    # Accumulate squared gradients
    fisher_accum = {
        name: torch.zeros_like(module.weight, device=device)
        for name, module in target_modules.items()
    }

    # Temporarily enable grad for weight matrices
    orig_requires_grad = {}
    for name, module in target_modules.items():
        orig_requires_grad[name] = module.weight.requires_grad
        module.weight.requires_grad_(True)

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
            print(f"[Fisher] Warning: forward/backward failed on batch - {e}")
            model.zero_grad()
            continue

        for name, module in target_modules.items():
            if module.weight.grad is not None:
                fisher_accum[name] += module.weight.grad.pow(2).detach()

        model.zero_grad()
        samples_seen += batch[next(iter(batch))].size(0)

    if samples_seen == 0:
        raise RuntimeError("Fisher computation saw zero samples. Check dataloader.")

    # Average over samples
    for name in fisher_accum:
        fisher_accum[name] /= max(samples_seen, 1)

    # Restore requires_grad
    for name, module in target_modules.items():
        module.weight.requires_grad_(orig_requires_grad[name])

    return fisher_accum


def compute_diagonal_fisher_with_hooks(
    model: nn.Module,
    dataloader: DataLoader,
    target_module_names: List[str],
    num_samples: int = 256,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient Fisher computation using backward hooks.

    For large models (7B+), this avoids storing full gradients for all
    layers simultaneously. Instead, it uses hooks to accumulate squared
    gradients incrementally during the backward pass.

    Args:
        model: The model.
        dataloader: Calibration data loader.
        target_module_names: Suffixes to match.
        num_samples: Maximum number of samples.
        device: Device override.

    Returns:
        Dictionary mapping full module names to Fisher diagonal tensors.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    target_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(t) for t in target_module_names
        ):
            target_modules[name] = module

    if not target_modules:
        raise ValueError(
            f"No Linear modules found matching target_module_names: {target_module_names}"
        )

    fisher_accum = {
        name: torch.zeros_like(module.weight, device=device)
        for name, module in target_modules.items()
    }

    # Register hooks to accumulate grad^2 without storing full grad tensors
    hooks = []
    for name, module in target_modules.items():
        module.weight.requires_grad_(True)

        def make_hook(mod_name):
            def hook_fn(grad):
                fisher_accum[mod_name] += grad.pow(2).detach()
                return grad
            return hook_fn

        h = module.weight.register_hook(make_hook(name))
        hooks.append(h)

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
            print(f"[Fisher] Warning: forward/backward failed - {e}")
            model.zero_grad()
            continue

        model.zero_grad()
        samples_seen += batch[next(iter(batch))].size(0)

    # Clean up hooks
    for h in hooks:
        h.remove()

    if samples_seen == 0:
        raise RuntimeError("Fisher computation saw zero samples.")

    for name in fisher_accum:
        fisher_accum[name] /= max(samples_seen, 1)

    for module in target_modules.values():
        module.weight.requires_grad_(False)

    return fisher_accum
