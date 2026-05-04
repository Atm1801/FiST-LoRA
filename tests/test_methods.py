"""Method definitions: FiST variants, original LoRA-XS / LoRA-SB, PEFT LoRA / PiSSA."""

import math

import pytest
import torch
from datasets import Dataset

from conftest import (
    ROBERTA_TARGETS,
    PadCollator,
    cls_rows,
    pad_batch,
    roberta_model_cfg,
    tiny_roberta,
)
from fist_lora.adapters.peft_methods import build_peft_model
from fist_lora.calibration.fist import ModuleCalibration
from fist_lora.calibration.grad_stats import GradientRecorder, activation_gradients
from fist_lora.calibration.lora_sb import effective_lr, estimate_update, lora_sb_factors
from fist_lora.calibration.subspace import truncated_svd
from fist_lora.config.schema import LoraSBConfig
from fist_lora.methods.registry import (
    LORA_XS_R_STD,
    METHODS,
    adapter_scaling,
    frozen_outer_factors,
    get_method,
)
from fist_lora.modeling.targets import resolve_targets


def fake_calibration(d=10, k=8, rmax=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    W, G = torch.randn(d, k, generator=g), torch.randn(d, k, generator=g)
    plain = truncated_svd(W, rmax)
    fisher = truncated_svd(W * torch.rand(d, k, generator=g), rmax)
    return W, G, {
        "m": ModuleCalibration(plain, fisher, plain.U.T @ G @ plain.Vh.T, fisher.U.T @ G @ fisher.Vh.T, {})
    }


def test_registry_is_consistent():
    for key, spec in METHODS.items():
        assert spec.kind in ("full", "peft", "frozen_outer"), key
        if spec.kind == "frozen_outer":
            assert spec.outer and spec.inner, key
    assert adapter_scaling(get_method("fist"), 16, 8) == 2.0
    assert adapter_scaling(get_method("lora_sb"), 16, 8) == 1.0  # original LoRA-SB: s = 1
    with pytest.raises(KeyError):
        get_method("nope")


@pytest.mark.parametrize("rank", [1, 2, 4])
def test_fist_uses_fisher_subspace_and_projected_gradient(rank):
    _, G, cal = fake_calibration()
    f = frozen_outer_factors(get_method("fist"), rank, 0.01, calibration=cal)["m"]
    fisher = cal["m"].fisher.truncate(rank)
    assert torch.equal(f.B, fisher.U) and torch.equal(f.A, fisher.Vh)
    expected = fisher.U.T @ G @ fisher.Vh.T  # B^T G A^T at this rank = leading block of M
    assert torch.allclose(f.R, 0.01 * expected / torch.linalg.norm(expected), atol=1e-7)


def test_fist_no_fisher_uses_plain_subspace():
    _, G, cal = fake_calibration()
    f = frozen_outer_factors(get_method("fist_no_fisher"), 3, 0.01, calibration=cal)["m"]
    plain = cal["m"].plain.truncate(3)
    assert torch.equal(f.B, plain.U) and torch.equal(f.A, plain.Vh)
    assert torch.linalg.norm(f.R).item() == pytest.approx(0.01)


def test_lora_xs_original_absorbs_sigma_into_input_factor():
    W, _, _ = fake_calibration()
    plain = {"m": truncated_svd(W, 4)}
    f = frozen_outer_factors(get_method("lora_xs"), 4, 0.01, plain_svd=plain, generator=torch.Generator().manual_seed(0))["m"]
    svd = plain["m"]
    assert torch.equal(f.B, svd.U)
    assert torch.allclose(f.A, torch.diag(svd.S) @ svd.Vh)
    assert torch.allclose(f.B @ f.A, (svd.U * svd.S) @ svd.Vh)  # B A = best rank-r approximation of W0
    assert f.R.abs().max() < 10 * LORA_XS_R_STD and f.R.abs().sum() > 0


def test_ablation_cells():
    _, _, cal = fake_calibration()
    W = torch.randn(10, 8)
    assert torch.equal(frozen_outer_factors(get_method("fisher_zero"), 2, 0.01, calibration=cal)["m"].R, torch.zeros(2, 2))
    sig = frozen_outer_factors(get_method("fisher_sigma"), 2, 0.01, calibration=cal)["m"].R
    assert torch.equal(sig, torch.diag(torch.diagonal(sig))) and torch.linalg.norm(sig).item() == pytest.approx(0.01)
    z = frozen_outer_factors(get_method("svd_zero"), 2, 0.01, plain_svd={"m": truncated_svd(W, 2)})["m"]
    assert torch.equal(z.R, torch.zeros(2, 2))
    with pytest.raises(ValueError):
        frozen_outer_factors(get_method("fist"), 2, 0.01)


def test_lora_sb_effective_lr_formula():
    # official train_glue.py: lr / (warmup_ratio * len(loader) * epochs)
    assert effective_lr(1e-3, 0.06, 2490, 128, 50) == pytest.approx(1e-3 / (0.06 * math.ceil(2490 / 128) * 50))


def test_lora_sb_estimate_matches_definition():
    model = tiny_roberta().train()
    targets = resolve_targets(model, ROBERTA_TARGETS, 8)
    for p in model.parameters():
        p.requires_grad_(False)
    ds = Dataset.from_list(cls_rows(20))
    cfg = LoraSBConfig(num_samples=6, estimation_batch_size=4)
    eta = 1e-3
    upd = estimate_update(model, targets, ds, PadCollator("seq_cls"), cfg, eta, seed=5, device=torch.device("cpu"))
    # Reference: same loader order and dropout masks, batch loss / examples taken, summed, signed.
    torch.manual_seed(0)
    from torch.utils.data import DataLoader

    from fist_lora.reproducibility import make_generator

    # estimate_update replays the RNG state at its entry; reproduce it.
    model2 = tiny_roberta().train()
    for p in model2.parameters():
        p.requires_grad_(False)
    t2 = resolve_targets(model2, ROBERTA_TARGETS, 8)
    loader = DataLoader(ds, batch_size=4, shuffle=True, generator=make_generator(5), collate_fn=PadCollator("seq_cls"))
    torch.manual_seed(0)
    upd_again = estimate_update(model2, t2, ds, PadCollator("seq_cls"), cfg, eta, seed=5, device=torch.device("cpu"))
    torch.manual_seed(0)
    with activation_gradients(model2), GradientRecorder(t2, per_example=False) as rec:
        seen = 0
        for batch in loader:
            take = min(4, 6 - seen)
            if take <= 0:
                break
            (model2(**{k: v[:take] for k, v in batch.items()}).loss / take).backward()
            seen += take
    for name in targets:
        ref = -eta * torch.sign(rec.grad_sum[name])
        assert torch.equal(upd_again[name], ref), name
        assert torch.all(torch.isin(upd[name], torch.tensor([-eta, 0.0, eta], dtype=torch.float32)))
    fac = lora_sb_factors(upd_again, rank=3)
    name = next(iter(fac))
    svd = truncated_svd(upd_again[name], 3)
    assert torch.allclose(fac[name].R, torch.diag(svd.S))  # R_init = S_r / s, s = 1
    assert torch.allclose(fac[name].B @ fac[name].R @ fac[name].A, (svd.U * svd.S) @ svd.Vh, atol=1e-6)


def test_lora_sb_estimate_rejects_too_few_examples():
    model = tiny_roberta()
    targets = resolve_targets(model, ROBERTA_TARGETS, 8)
    ds = Dataset.from_list(cls_rows(3))
    with pytest.raises(RuntimeError, match="expected 5"):
        estimate_update(model, targets, ds, PadCollator("seq_cls"), LoraSBConfig(5, 2), 1e-3, 0, torch.device("cpu"))


def test_peft_lora_init_and_trainable_head():
    model = tiny_roberta()
    x = pad_batch(cls_rows(3), "seq_cls")
    base_logits = model(**{k: v for k, v in x.items() if k != "labels"}).logits
    peft = build_peft_model(model, roberta_model_cfg(), rank=4, init="lora")
    peft.eval()
    assert torch.allclose(peft(**{k: v for k, v in x.items() if k != "labels"}).logits, base_logits, atol=1e-6)  # B = 0
    trainable = {n: p for n, p in peft.named_parameters() if p.requires_grad}
    lora = sum(p.numel() for n, p in trainable.items() if "lora_" in n)
    assert lora == 2 * 4 * 4 * (16 + 16)  # L * M * r (d + k)
    assert any("modules_to_save" in n and "classifier" in n for n in trainable)
    # The trained head copy (modules_to_save) is the one used in forward.
    peft.train()
    opt = torch.optim.SGD([p for p in trainable.values()], lr=1.0)
    head_before = {n: p.detach().clone() for n, p in trainable.items() if "classifier" in n}
    peft(**x).loss.backward()
    opt.step()
    peft.eval()
    assert any(not torch.equal(p, head_before[n]) for n, p in trainable.items() if "classifier" in n)
    assert not torch.allclose(peft(**{k: v for k, v in x.items() if k != "labels"}).logits, base_logits)


def test_pissa_preserves_function_at_init():
    model = tiny_roberta()  # PEFT's PiSSA accepts fp32/fp16/bf16 weights only
    x = {k: v for k, v in pad_batch(cls_rows(3), "seq_cls").items() if k != "labels"}
    base_logits = model(**x).logits
    peft = build_peft_model(model, roberta_model_cfg(), rank=4, init="pissa").eval()
    assert torch.allclose(peft(**x).logits, base_logits, atol=1e-5)
    layer = peft.base_model.model.roberta.encoder.layer[0].attention.self.query
    assert layer.lora_B["default"].weight.abs().sum() > 0  # principal components, not zero init
    with pytest.raises(NotImplementedError):
        build_peft_model(tiny_roberta(), roberta_model_cfg(precision="nf4"), rank=4, init="pissa")
