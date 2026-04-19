"""
FiST-LoRA: Fisher-Informed Subspace Training for Low-Rank Adaptation.

In the LoRA-XS architecture (DeltaW = s * B * R * A, where B and A are frozen,
R is a small r x r trainable matrix), we replace the standard SVD-based
initialization of the frozen outer matrices with Fisher-information-weighted SVD,
and initialize the inner R matrix using gradient projection into the
Fisher-optimized subspace.
"""

from fist_lora.config import FiSTLoRAConfig
from fist_lora.fisher import compute_diagonal_fisher
from fist_lora.init import fisher_weighted_svd, plain_svd, gradient_projected_R, zero_R
from fist_lora.layers import FiSTLoRALinear
from fist_lora.model import inject_fist_lora, count_trainable_params, count_total_params

__all__ = [
    "FiSTLoRAConfig",
    "compute_diagonal_fisher",
    "fisher_weighted_svd",
    "plain_svd",
    "gradient_projected_R",
    "zero_R",
    "FiSTLoRALinear",
    "inject_fist_lora",
    "count_trainable_params",
    "count_total_params",
]
