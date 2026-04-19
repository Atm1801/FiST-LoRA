"""
FiSTLoRALinear: drop-in nn.Linear replacement with FiST-LoRA adapter.

Architecture: h = F.linear(x, W, bias) + (alpha/r) * x @ A^T @ R^T @ B^T

Storage:
- self.weight: frozen pretrained weight (nn.Parameter, requires_grad=False)
- self.bias: frozen pretrained bias or None
- self.B: frozen left outer (d, r) -- registered as BUFFER
- self.A: frozen right outer (r, k) -- registered as BUFFER
- self.R: trainable inner (r, r) -- nn.Parameter, requires_grad=True
- self.scaling: alpha/r scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiSTLoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a FiST-LoRA adapter.

    Forward computation order (left-to-right, never materializes d x k):
        step1 = x @ A^T          # (..., k) -> (..., r)
        step2 = step1 @ R^T      # (..., r) -> (..., r)
        step3 = step2 @ B^T      # (..., r) -> (..., d)
        out = base + scaling * step3
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        B: torch.Tensor,       # (d, r)
        A: torch.Tensor,       # (r, k)
        R_init: torch.Tensor,  # (r, r)
        alpha: float,
        rank: int,
    ):
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen original weight
        self.weight = nn.Parameter(
            original_linear.weight.data.clone(), requires_grad=False
        )
        if original_linear.bias is not None:
            self.bias = nn.Parameter(
                original_linear.bias.data.clone(), requires_grad=False
            )
        else:
            self.bias = None

        # Frozen outer matrices -- buffers so they move with .to(device)
        self.register_buffer("B", B.clone().to(self.weight.dtype))  # (d, r)
        self.register_buffer("A", A.clone().to(self.weight.dtype))  # (r, k)

        # Trainable inner matrix
        self.R = nn.Parameter(R_init.clone().to(self.weight.dtype))  # (r, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear (uses frozen weight)
        out = F.linear(x, self.weight, self.bias)

        # LoRA path: x @ A^T @ R^T @ B^T  scaled by alpha/rank
        step1 = x @ self.A.T           # (..., r)
        step2 = step1 @ self.R.T       # (..., r)
        step3 = step2 @ self.B.T       # (..., d)
        out = out + self.scaling * step3

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.4f}"
        )
