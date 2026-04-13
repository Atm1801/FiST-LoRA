"""Adapter forward pass, trainable set, targeting and parameter counts."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import ROBERTA_TARGETS, tiny_llama, tiny_roberta
from fist_lora.adapters.frozen_outer import FrozenOuterLinear
from fist_lora.adapters.inject import (
    AdapterFactors,
    count_parameters,
    inject_frozen_outer,
    set_trainable,
)
from fist_lora.calibration.subspace import truncated_svd
from fist_lora.modeling.targets import resolve_targets


def make_adapter(d=12, k=9, r=3, R=None, bias=True, scaling=2.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = nn.Linear(k, d, bias=bias)
    svd = truncated_svd(torch.randn(d, k, generator=g), r)
    R = torch.randn(r, r, generator=g) if R is None else R
    return base, FrozenOuterLinear(base, svd.U, svd.Vh, R, scaling)


@pytest.mark.parametrize("shape", [(5, 9), (2, 7, 9)])
def test_forward_equals_closed_form(shape):
    base, ad = make_adapter()
    x = torch.randn(*shape)
    expected = F.linear(x, base.weight, base.bias) + 2.0 * x @ (ad.B @ ad.R @ ad.A).T
    assert torch.allclose(ad(x), expected, atol=1e-5)
    assert torch.allclose(ad.delta_weight(), 2.0 * ad.B @ ad.R @ ad.A)


def test_zero_R_is_identity_to_base():
    base, ad = make_adapter(R=torch.zeros(3, 3), bias=False)
    x = torch.randn(4, 9)
    assert torch.equal(ad(x), base(x))


def test_adapter_path_never_materialises_d_by_k(monkeypatch):
    base, ad = make_adapter(d=12, k=9, r=3)
    shapes = []
    real_linear = F.linear

    def spy(inp, weight, bias=None):
        shapes.append(tuple(weight.shape))
        return real_linear(inp, weight, bias)

    monkeypatch.setattr(F, "linear", spy)
    ad.adapter_forward(torch.randn(4, 9))
    assert shapes == [(3, 9), (3, 3), (12, 3)]  # A, R, B - never (12, 9)


def test_only_R_receives_gradient_and_is_fp32():
    base, ad = make_adapter()
    base.requires_grad_(False)
    ad(torch.randn(4, 9)).sum().backward()
    assert ad.R.grad is not None and ad.R.dtype == torch.float32
    assert base.weight.grad is None
    assert not ad.B.requires_grad and not ad.A.requires_grad
    assert {n for n, _ in ad.named_buffers()} == {"B", "A"}


def test_autocast_bf16_path_keeps_fp32_master_R():
    base, ad = make_adapter()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = ad(torch.randn(4, 9))
    out.float().sum().backward()
    assert ad.R.dtype == torch.float32 and ad.R.grad.dtype == torch.float32


def test_shape_and_finiteness_validation():
    base = nn.Linear(9, 12)
    with pytest.raises(ValueError, match="shape mismatch"):
        FrozenOuterLinear(base, torch.randn(12, 3), torch.randn(3, 8), torch.zeros(3, 3), 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        FrozenOuterLinear(base, torch.randn(12, 3), torch.randn(3, 9), torch.full((3, 3), float("nan")), 1.0)


def test_roberta_targets_are_exactly_q_k_v_o():
    model = tiny_roberta()
    targets = resolve_targets(model, ROBERTA_TARGETS, expected=8)
    suffixes = sorted({n.split("layer.")[1].split(".", 1)[1] for n in targets})
    assert suffixes == ["attention.output.dense", "attention.self.key", "attention.self.query", "attention.self.value"]
    assert not any("intermediate" in n or n.endswith("output.dense") and "attention" not in n for n in targets)
    with pytest.raises(ValueError, match="expected 9"):
        resolve_targets(model, ROBERTA_TARGETS, expected=9)


def test_inject_trainable_set_and_counts():
    model = tiny_roberta()
    targets = resolve_targets(model, ROBERTA_TARGETS, expected=8)
    r = 4
    factors = {}
    for name, m in targets.items():
        svd = truncated_svd(m.weight, r)
        factors[name] = AdapterFactors(svd.U, svd.Vh, torch.zeros(r, r))
    x = {"input_ids": torch.randint(3, 100, (2, 6))}
    before = model(**x).logits
    inject_frozen_outer(model, factors, scaling=16 / r)
    set_trainable(model, ["classifier"], train_head=True)
    assert torch.allclose(model(**x).logits, before, atol=1e-6)  # R = 0: unchanged function
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(n.endswith(".R") or n.startswith("classifier.") for n in trainable)
    counts = count_parameters(model, ["classifier"])
    assert counts["adapter_params"] == 2 * 4 * r**2  # L * M * r^2
    head = sum(p.numel() for n, p in model.named_parameters() if n.startswith("classifier."))
    assert counts["head_params"] == head
    with pytest.raises(ValueError, match="already carries"):
        inject_frozen_outer(model, factors, 1.0)


def test_causal_lm_head_frozen_when_configured():
    model = tiny_llama()
    targets = resolve_targets(model, [r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)", r".*\.mlp\.(gate_proj|up_proj|down_proj)"], 14)
    factors = {n: AdapterFactors(*[t for t in (truncated_svd(m.weight, 2).U, truncated_svd(m.weight, 2).Vh)], torch.zeros(2, 2)) for n, m in targets.items()}
    inject_frozen_outer(model, factors, 16.0)
    set_trainable(model, ["lm_head"], train_head=False)
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {f"{n}.R" for n in targets}
