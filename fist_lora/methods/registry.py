"""Method definitions.

Every method is a combination of

* ``outer``   - how the frozen factors (B, A) are chosen,
* ``inner``   - how the trainable R is initialised,
* ``scaling`` - the adapter scale s in h = W0 x + s B R A x,
* ``calibration`` - which data-dependent statistics it needs.

========================  ==============================  ====================  ============  ===========
method                    outer (B, A)                    inner R_init          scale s       calibration
========================  ==============================  ====================  ============  ===========
lora_xs (original)        SVD(W0): U_r, S_r V_r^T         N(0, 1e-5^2)          alpha / r     none
lora_sb (original)        SVD(-eta sign(sum grad))        S_r                   1             per seed
fist_no_fisher            SVD(W0): U_r, V_r^T             gamma-normalised      alpha / r     FiST (task)
                                                          projected gradient
fist                      SVD(sqrt(F_bar+eps).W0)         gamma-normalised      alpha / r     FiST (task)
                                                          projected gradient
svd_zero (ablation)       SVD(W0): U_r, V_r^T             0                     alpha / r     none
fisher_zero (ablation)    Fisher-weighted SVD             0                     alpha / r     FiST (task)
svd_sigma (ablation)      SVD(W0): U_r, V_r^T             gamma diag(S_r)/|.|   alpha / r     none
fisher_sigma (ablation)   Fisher-weighted SVD             gamma diag(S_r)/|.|   alpha / r     FiST (task)
========================  ==============================  ====================  ============  ===========

``svd_zero`` (plain-SVD subspace, zero R) differs from ``fist_no_fisher`` only in the inner
initialisation and from ``fisher_zero`` only in the subspace, so the chain
svd_zero -> fist_no_fisher -> fist separates the two FiST components.

Full-rank-budget methods: ``full_ft`` (all weights), ``lora`` and ``pissa`` (PEFT).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fist_lora.adapters.inject import AdapterFactors
from fist_lora.calibration.fist import ModuleCalibration
from fist_lora.calibration.inner_init import (
    gaussian_R,
    normalize_to_scale,
    sigma_diag_R,
    zero_R,
)
from fist_lora.calibration.subspace import TruncatedSVD


@dataclass(frozen=True)
class MethodSpec:
    kind: str  # "full" | "peft" | "frozen_outer"
    peft_init: str | None = None
    outer: str | None = None  # "svd" | "svd_sigma_input" | "fisher_svd" | "lora_sb"
    inner: str | None = None  # "gradient_projected" | "zero" | "gaussian" | "sigma_diag" | "lora_sb"
    calibration: str | None = None  # None | "fist" | "lora_sb"
    scaling: str = "alpha_over_r"  # "alpha_over_r" | "one"
    lr_group: str = "r2"  # "full_rank" | "r2" | "full_ft"

    @property
    def uses_rank(self) -> bool:
        return self.kind != "full"

    @property
    def needs_plain_svd(self) -> bool:
        return self.kind == "frozen_outer" and self.outer in ("svd", "svd_sigma_input") and self.calibration is None


METHODS: dict[str, MethodSpec] = {
    "full_ft": MethodSpec(kind="full", lr_group="full_ft"),
    "lora": MethodSpec(kind="peft", peft_init="lora", lr_group="full_rank"),
    "pissa": MethodSpec(kind="peft", peft_init="pissa", lr_group="full_rank"),
    "lora_xs": MethodSpec(kind="frozen_outer", outer="svd_sigma_input", inner="gaussian"),
    "lora_sb": MethodSpec(kind="frozen_outer", outer="lora_sb", inner="lora_sb", calibration="lora_sb", scaling="one"),
    "fist_no_fisher": MethodSpec(kind="frozen_outer", outer="svd", inner="gradient_projected", calibration="fist"),
    "fist": MethodSpec(kind="frozen_outer", outer="fisher_svd", inner="gradient_projected", calibration="fist"),
    "svd_zero": MethodSpec(kind="frozen_outer", outer="svd", inner="zero"),
    "fisher_zero": MethodSpec(kind="frozen_outer", outer="fisher_svd", inner="zero", calibration="fist"),
    "svd_sigma": MethodSpec(kind="frozen_outer", outer="svd", inner="sigma_diag"),
    "fisher_sigma": MethodSpec(kind="frozen_outer", outer="fisher_svd", inner="sigma_diag", calibration="fist"),
}

# Hyper-parameters of the original LoRA-XS (Bałazy et al. 2024; official code
# github.com/MohammadrezaBanaei/LoRA-XS @ e50b1a82b5d7b7e87a0c46bc62e07e54f6648d31).
LORA_XS_R_STD = 1e-5


def get_method(key: str) -> MethodSpec:
    try:
        return METHODS[key]
    except KeyError:
        raise KeyError(f"unknown method {key!r}; known: {sorted(METHODS)}") from None


def adapter_scaling(spec: MethodSpec, alpha: float, rank: int) -> float:
    return alpha / rank if spec.scaling == "alpha_over_r" else 1.0


def frozen_outer_factors(
    spec: MethodSpec,
    rank: int,
    init_scale: float,
    calibration: dict[str, ModuleCalibration] | None = None,
    plain_svd: dict[str, TruncatedSVD] | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, AdapterFactors]:
    """(B, A, R_init) for every target module, for all frozen-outer methods except LoRA-SB."""
    if spec.kind != "frozen_outer" or spec.outer == "lora_sb":
        raise ValueError("frozen_outer_factors handles the SVD-of-weight methods only")
    if spec.calibration == "fist":
        if calibration is None:
            raise ValueError("this method needs the FiST calibration")
        names = sorted(calibration)
    else:
        if plain_svd is None:
            raise ValueError("this method needs the plain SVD of the target weights")
        names = sorted(plain_svd)

    factors = {}
    for name in names:
        if spec.outer == "fisher_svd":
            svd, proj = calibration[name].fisher, calibration[name].proj_fisher
        elif calibration is not None:
            svd, proj = calibration[name].plain, calibration[name].proj_plain
        else:
            svd, proj = plain_svd[name], None
        svd = svd.truncate(rank)

        B = svd.U
        # LoRA-XS absorbs the singular values into the input-side frozen factor:
        # official code runs SVD on W^T and sets lora_A = (U' S)^T, which in the
        # h = W0 x convention is A = S_r V_r^T (LoRA-XS row-vector notation "A = U_r S_r").
        A = torch.diag(svd.S) @ svd.Vh if spec.outer == "svd_sigma_input" else svd.Vh

        if spec.inner == "gradient_projected":
            R = normalize_to_scale(proj[:rank, :rank], init_scale)  # gamma * B^T G A^T / ||.||_F
        elif spec.inner == "zero":
            R = zero_R(rank)
        elif spec.inner == "gaussian":
            R = gaussian_R(rank, LORA_XS_R_STD, generator)
        elif spec.inner == "sigma_diag":
            R = sigma_diag_R(svd.S, init_scale)
        else:
            raise ValueError(f"unknown inner init {spec.inner!r}")
        factors[name] = AdapterFactors(B=B, A=A, R=R)
    return factors
