"""Truncated SVD of (re-weighted) weight or gradient matrices.

Convention: for a matrix M = U S V^T of shape (d, k) (d = out_features, k = in_features,
as stored in ``nn.Linear.weight``), the rank-r factors are U_r (d x r), S_r (r,),
Vh_r = V_r^T (r x k).  The frozen factors are then B = U_r (left/output side) and
A = Vh_r (right/input side), matching h = W0 x + s B R A x.

Singular vectors are only defined up to a joint sign flip (u_i, v_i) -> (-u_i, -v_i);
every quantity used downstream (B R A with R = B^T G A^T, or with R diagonal) is
invariant to it, so no sign canonicalisation is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


@dataclass
class TruncatedSVD:
    U: torch.Tensor  # (d, r_max)
    S: torch.Tensor  # (r_max,)
    Vh: torch.Tensor  # (r_max, k)

    def truncate(self, r: int) -> TruncatedSVD:
        if r > self.S.numel():
            raise ValueError(f"requested rank {r} > stored rank {self.S.numel()}")
        return TruncatedSVD(self.U[:, :r], self.S[:r], self.Vh[:r])

    def to(self, device: torch.device | str) -> TruncatedSVD:
        return TruncatedSVD(self.U.to(device), self.S.to(device), self.Vh.to(device))


def truncated_svd(matrix: torch.Tensor, rank: int, dtype: str = "float32") -> TruncatedSVD:
    """Exact SVD (``torch.linalg.svd``, full_matrices=False) truncated to ``rank``.

    bf16/fp16 are not supported by the LAPACK/cuSOLVER kernels, so the input is cast to
    ``dtype`` (fp32 by default, fp64 available for numerical checks).
    """
    if matrix.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    if rank < 1 or rank > min(matrix.shape):
        raise ValueError(f"rank {rank} invalid for a matrix of shape {tuple(matrix.shape)}")
    m = matrix.to(_DTYPES[dtype])
    if not torch.isfinite(m).all():
        raise ValueError("SVD input contains non-finite values")
    U, S, Vh = torch.linalg.svd(m, full_matrices=False)
    return TruncatedSVD(
        U[:, :rank].float().contiguous(), S[:rank].float().contiguous(), Vh[:rank].float().contiguous()
    )
