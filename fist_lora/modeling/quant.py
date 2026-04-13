"""4-bit NF4 helpers (NF4, double quantization, bf16 compute) for the 7B backbones.

bitsandbytes is only imported lazily so the CPU code path and tests never need it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def nf4_quantization_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def is_4bit_linear(module: nn.Module) -> bool:
    return type(module).__name__ == "Linear4bit"


@torch.no_grad()
def effective_weight(module: nn.Module) -> torch.Tensor:
    """The (d, k) weight W0 actually used in the forward pass, as fp32.

    For NF4 layers this is the dequantized weight: the adapter perturbs exactly this
    matrix, so the Fisher statistic, the SVD and the gradient all refer to it.
    """
    if is_4bit_linear(module):
        import bitsandbytes.functional as bnbf

        w = bnbf.dequantize_4bit(module.weight.data, module.weight.quant_state)
    else:
        w = module.weight
    w = w.detach().float()
    expected = (module.out_features, module.in_features)
    if tuple(w.shape) != expected:
        raise RuntimeError(f"effective weight has shape {tuple(w.shape)}, expected {expected}")
    return w
