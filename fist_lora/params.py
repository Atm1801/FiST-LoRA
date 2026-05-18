"""Analytic adapter parameter counts, storage arithmetic and initial perturbation size.

Counts are computed from the architecture dimensions in ``configs/models/*.yaml``, which
are checked against the real modules when a model is loaded (:func:`verify_architecture`).
Trainable adapter parameters exclude the task head.
"""

from __future__ import annotations

from fist_lora.config.schema import ArchitectureSpec

BYTES_FP32 = 4
MB = 1e6  # decimal megabytes (e.g. 19,988,480 fp32 parameters = 79.95 MB)
GB_PER_MB = 1024


def module_shapes(arch: ArchitectureSpec, family: str) -> list[tuple[str, int, int]]:
    """(name, d_out, k_in) of the M adapted modules of one layer."""
    h, i = arch.hidden_size, arch.intermediate_size
    kv = h * arch.num_key_value_heads // arch.num_attention_heads
    if family == "roberta":
        return [("query", h, h), ("key", h, h), ("value", h, h), ("attention.output.dense", h, h)]
    if family in ("llama", "mistral"):
        return [
            ("q_proj", h, h), ("k_proj", kv, h), ("v_proj", kv, h), ("o_proj", h, h),
            ("gate_proj", i, h), ("up_proj", i, h), ("down_proj", h, i),
        ]
    raise ValueError(f"unknown model family {family!r}")


def r2_params(arch: ArchitectureSpec, family: str, rank: int) -> int:
    """Frozen-outer adapters: P(r) = L * M * r^2 (layers x adapted modules x r^2), independent of d and k."""
    return arch.num_layers * len(module_shapes(arch, family)) * rank**2


def lora_params(arch: ArchitectureSpec, family: str, rank: int) -> int:
    """LoRA / PiSSA: r (d + k) per module."""
    return arch.num_layers * sum(rank * (d + k) for _, d, k in module_shapes(arch, family))


def storage_mb(num_params: int) -> tuple[float, float]:
    """(fp32 checkpoint MB, Adam moment MB) for ``num_params`` trainable parameters."""
    params_mb = num_params * BYTES_FP32 / MB
    return params_mb, 2 * params_mb


def initial_perturbation(alpha: float, rank: int, init_scale: float) -> float:
    """||Delta W_init||_F = (alpha / r) * gamma exactly, since B and A have orthonormal columns/rows."""
    return alpha / rank * init_scale


ARCHS = {
    "roberta": ArchitectureSpec(24, 1024, 4096, 16, 16),
    "llama": ArchitectureSpec(32, 4096, 11008, 32, 32),
    "mistral": ArchitectureSpec(32, 4096, 14336, 32, 8),
}


def verify_architecture(targets: dict, arch: ArchitectureSpec) -> None:
    """Check the YAML ``architecture`` against the loaded target modules (shapes and count)."""
    family = "llama" if any(n.endswith("q_proj") for n in targets) else "roberta"
    expected = module_shapes(arch, family)
    if len(targets) != arch.num_layers * len(expected):
        raise ValueError(f"{len(targets)} targets, architecture implies {arch.num_layers * len(expected)}")
    for name, module in targets.items():
        match = [(d, k) for suffix, d, k in expected if name.endswith(suffix)]
        if not match:
            raise ValueError(f"target {name} matches no module of the {family} architecture spec")
        actual = (module.out_features, module.in_features)
        if actual != match[0]:
            raise ValueError(f"{name}: shape {actual} != architecture spec {match[0]}")
