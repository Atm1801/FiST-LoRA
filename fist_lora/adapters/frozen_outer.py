"""Frozen-outer low-rank adapter shared by LoRA-XS, LoRA-SB and FiST-LoRA.

    h = base(x) + s * B (R (A x))

* ``base`` is the original layer, kept by composition (it may be an fp32/bf16
  ``nn.Linear`` or a bitsandbytes NF4 ``Linear4bit``; its weight is never copied).
* ``B`` (d x r) and ``A`` (r x k) are frozen fp32 buffers; ``R`` (r x r) is the only
  trainable tensor, stored in fp32 so optimiser updates are not rounded to bf16.
* The product is evaluated right-to-left on activations, costing O(kr + r^2 + dr) per
  token; the d x k update ``s * B R A`` is never materialised in ``forward``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenOuterLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        B: torch.Tensor,
        A: torch.Tensor,
        R: torch.Tensor,
        scaling: float,
    ) -> None:
        super().__init__()
        d, k = base_layer.out_features, base_layer.in_features
        r = R.shape[0]
        if B.shape != (d, r) or A.shape != (r, k) or R.shape != (r, r):
            raise ValueError(
                f"shape mismatch: base (d={d}, k={k}), B {tuple(B.shape)}, "
                f"A {tuple(A.shape)}, R {tuple(R.shape)}"
            )
        for name, t in (("B", B), ("A", A), ("R", R)):
            if not torch.isfinite(t).all():
                raise ValueError(f"{name} contains non-finite values")
        self.base_layer = base_layer
        self.in_features, self.out_features, self.rank = k, d, r
        self.scaling = float(scaling)
        device = _device_of(base_layer)
        self.register_buffer("B", B.detach().to(device=device, dtype=torch.float32).contiguous())
        self.register_buffer("A", A.detach().to(device=device, dtype=torch.float32).contiguous())
        self.R = nn.Parameter(R.detach().to(device=device, dtype=torch.float32).clone())

    @property
    def weight(self) -> torch.Tensor:  # some HF code paths inspect `.weight` (dtype/device)
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    def adapter_forward(self, x: torch.Tensor) -> torch.Tensor:
        """s * B(R(Ax)) in row-vector form: ((x A^T) R^T) B^T."""
        if torch.is_autocast_enabled():
            # Under autocast F.linear runs in the autocast dtype; the fp32 master copy of R
            # still receives the gradient through the implicit cast.
            z = F.linear(F.linear(F.linear(x, self.A), self.R), self.B)
        else:
            z = F.linear(F.linear(F.linear(x.to(self.R.dtype), self.A), self.R), self.B)
        return self.scaling * z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        return out + self.adapter_forward(x).to(out.dtype)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        """Materialised update s * B R A (analysis/tests only; never used in forward)."""
        return self.scaling * (self.B @ self.R @ self.A)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, scaling={self.scaling:g}"


def _device_of(module: nn.Module) -> torch.device:
    for p in module.parameters():
        return p.device
    return torch.device("cpu")
