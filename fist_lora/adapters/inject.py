"""Replacing target layers with frozen-outer adapters and setting the trainable set."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from fist_lora.adapters.frozen_outer import FrozenOuterLinear
from fist_lora.modeling.targets import get_parent, head_parameters


@dataclass
class AdapterFactors:
    """Frozen outer factors and inner initialisation for one target module."""

    B: torch.Tensor  # (d, r)
    A: torch.Tensor  # (r, k)
    R: torch.Tensor  # (r, r)


def inject_frozen_outer(
    model: nn.Module, factors: dict[str, AdapterFactors], scaling: float
) -> dict[str, FrozenOuterLinear]:
    """Wrap every module named in ``factors`` with a :class:`FrozenOuterLinear`."""
    adapters = {}
    for name, f in factors.items():
        parent, child = get_parent(model, name)
        base = getattr(parent, child)
        if isinstance(base, FrozenOuterLinear):
            raise ValueError(f"{name} already carries an adapter")
        adapter = FrozenOuterLinear(base, f.B, f.A, f.R, scaling)
        setattr(parent, child, adapter)
        adapters[name] = adapter
    return adapters


def set_trainable(model: nn.Module, head_modules: list[str], train_head: bool) -> None:
    """Freeze everything, then unfreeze every adapter ``R`` and (optionally) the task head."""
    for p in model.parameters():
        p.requires_grad_(False)
    n_adapters = 0
    for m in model.modules():
        if isinstance(m, FrozenOuterLinear):
            m.R.requires_grad_(True)
            n_adapters += 1
    if n_adapters == 0:
        raise RuntimeError("set_trainable found no FrozenOuterLinear adapters")
    if train_head:
        for p in head_parameters(model, head_modules).values():
            p.requires_grad_(True)


def count_parameters(model: nn.Module, head_modules: list[str]) -> dict[str, int]:
    """Trainable parameter counts, split into adapter and head (adapter counts exclude the head)."""
    head_names = set(head_parameters(model, head_modules))
    head = adapter = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in head_names:
            head += p.numel()
        else:
            adapter += p.numel()
    total = sum(p.numel() for p in model.parameters())
    return {"adapter_params": adapter, "head_params": head, "total_params": total}
