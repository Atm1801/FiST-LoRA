"""
Model injection for FiST-LoRA.

Replaces target nn.Linear modules with FiSTLoRALinear adapters,
then freezes everything except R matrices and the task head.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

from fist_lora.layers import FiSTLoRALinear, StandardLoRALinear


def _get_parent_and_attr(model: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    """
    Given a dot-separated module name, return (parent_module, child_attr_name).
    E.g. "roberta.encoder.layer.0.attention.self.query"
         -> (roberta.encoder.layer.0.attention.self, "query")
    """
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_fist_lora(
    model: nn.Module,
    target_module_names: List[str],
    B_dict: Dict[str, torch.Tensor],
    A_dict: Dict[str, torch.Tensor],
    R_dict: Dict[str, torch.Tensor],
    alpha: float = 32.0,
    rank: int = 32,
    head_keywords: Optional[List[str]] = None,
) -> nn.Module:
    """
    Replace each target nn.Linear in the model with FiSTLoRALinear.

    Matching rule: a module is replaced if its full dotted name ends with
    any string in target_module_names AND it is an nn.Linear.

    After injection:
      1. ALL parameters frozen (requires_grad=False)
      2. R matrices unfrozen
      3. Classification/LM head unfrozen

    [PoC-VALIDATED] Head unfreezing keywords:
    - For classification: ("classifier", "score", "qa_outputs")
    - For LLM generation: ("lm_head",)
    - Do NOT include "pooler" -- it's pretrained, not random, and adds ~786K
      params that inflate the count unfairly vs PEFT LoRA.

    Args:
        model: The pretrained model.
        target_module_names: Suffixes to match (e.g., ["query", "key", "value"]).
        B_dict: {full_module_name: B tensor (d, r)}.
        A_dict: {full_module_name: A tensor (r, k)}.
        R_dict: {full_module_name: R tensor (r, r)}.
        alpha: LoRA scaling numerator.
        rank: LoRA rank.
        head_keywords: Keywords for unfreezing the task head. If None,
            defaults to ("classifier", "score", "qa_outputs").

    Returns:
        The modified model (in-place, but also returned for convenience).
    """
    if head_keywords is None:
        head_keywords = ["classifier", "score", "qa_outputs"]

    # Collect replacements first to avoid mutating model while iterating
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(t) for t in target_module_names
        ):
            if name not in B_dict:
                print(f"[inject] Warning: {name} matched but not in B_dict; skipping.")
                continue
            replacements.append(name)

    for name in replacements:
        parent, attr = _get_parent_and_attr(model, name)
        original_linear = getattr(parent, attr)

        fist_layer = FiSTLoRALinear(
            original_linear=original_linear,
            B=B_dict[name],
            A=A_dict[name],
            R_init=R_dict[name],
            alpha=alpha,
            rank=rank,
        )
        setattr(parent, attr, fist_layer)

    # Freeze everything, then selectively unfreeze
    for param in model.parameters():
        param.requires_grad_(False)

    # Unfreeze trainable R matrices in every adapter layer
    for name, module in model.named_modules():
        if isinstance(module, FiSTLoRALinear):
            module.R.requires_grad_(True)

    # Unfreeze the classification/LM head
    for name, param in model.named_parameters():
        if any(kw in name for kw in head_keywords):
            param.requires_grad_(True)

    return model


def inject_standard_lora(
    model: nn.Module,
    target_module_names: List[str],
    alpha: float = 32.0,
    rank: int = 8,
    head_keywords: Optional[List[str]] = None,
) -> nn.Module:
    """
    Replace target nn.Linear modules with StandardLoRALinear (manual LoRA).
    Avoids PEFT's wrappers entirely, so gradient flow is straightforward.
    Trainable: lora_A, lora_B per module + the task head.
    """
    if head_keywords is None:
        head_keywords = ["classifier", "score", "qa_outputs"]

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(t) for t in target_module_names
        ):
            replacements.append(name)

    for name in replacements:
        parent, attr = _get_parent_and_attr(model, name)
        original_linear = getattr(parent, attr)
        setattr(parent, attr, StandardLoRALinear(original_linear, alpha=alpha, rank=rank))

    for param in model.parameters():
        param.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, StandardLoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)

    for name, param in model.named_parameters():
        if any(kw in name for kw in head_keywords):
            param.requires_grad_(True)

    return model


def count_trainable_params(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    """Return total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def collect_plain_svd(
    model: nn.Module,
    target_module_names: List[str],
    rank: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Collect plain SVD decompositions for all target modules.

    Returns:
        Dict mapping module name to (B, S, A) tuples.
    """
    from fist_lora.init import plain_svd

    svd_results = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(t) for t in target_module_names
        ):
            B, S, A = plain_svd(module.weight.data.float(), rank)
            svd_results[name] = (B, S, A)
    return svd_results


def collect_fisher_svd(
    model: nn.Module,
    fisher_diags: Dict[str, torch.Tensor],
    target_module_names: List[str],
    rank: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Collect Fisher-weighted SVD decompositions for all target modules.

    Falls back to plain SVD if Fisher diagonal is missing for a module.

    Returns:
        Dict mapping module name to (B, S, A) tuples.
    """
    from fist_lora.init import fisher_weighted_svd, plain_svd

    svd_results = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(t) for t in target_module_names
        ):
            if name in fisher_diags:
                B, S, A = fisher_weighted_svd(
                    module.weight.data.float(),
                    fisher_diags[name].float(),
                    rank,
                )
            else:
                print(f"[SVD] Warning: no Fisher diag for {name}; using plain SVD.")
                B, S, A = plain_svd(module.weight.data.float(), rank)
            svd_results[name] = (B, S, A)
    return svd_results
