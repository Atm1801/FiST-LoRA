"""Resolution of adapted modules by full-path regex.

Suffix matching (``name.endswith("dense")``) silently over-matches in RoBERTa, where
``attention.output.dense``, ``intermediate.dense``, ``output.dense`` and
``classifier.dense`` all end in ``dense``.  Targets are therefore regexes matched with
``re.fullmatch`` against the dotted module path, and the number of matches is asserted.
"""

from __future__ import annotations

import re

import torch.nn as nn


def is_linear_like(module: nn.Module) -> bool:
    """nn.Linear and its subclasses (bitsandbytes ``Linear4bit`` subclasses nn.Linear)."""
    return isinstance(module, nn.Linear)


def resolve_targets(
    model: nn.Module, patterns: list[str], expected: int | None = None
) -> dict[str, nn.Module]:
    compiled = [re.compile(p) for p in patterns]
    found = {
        name: module
        for name, module in model.named_modules()
        if is_linear_like(module) and any(p.fullmatch(name) for p in compiled)
    }
    if not found:
        raise ValueError(f"no linear modules match target patterns {patterns}")
    if expected is not None and len(found) != expected:
        raise ValueError(
            f"target patterns {patterns} matched {len(found)} modules, expected {expected}: "
            f"{sorted(found)[:6]}..."
        )
    return found


def get_parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), child


def head_parameters(model: nn.Module, head_modules: list[str]) -> dict[str, nn.Parameter]:
    """Parameters of the task head, matched by module-name component (e.g. ``classifier``)."""
    params = {
        name: p
        for name, p in model.named_parameters()
        if any(part in head_modules for part in name.split("."))
    }
    if not params:
        raise ValueError(f"no parameters found for head modules {head_modules}")
    return params
