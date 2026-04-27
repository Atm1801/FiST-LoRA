"""Calibration lifecycle: warm-up, subset sampling, statistics, caching."""

import pytest
import torch
from datasets import Dataset

import fist_lora.calibration.fist as fist_mod
from conftest import ROBERTA_TARGETS, PadCollator, cls_rows, roberta_model_cfg, tiny_roberta
from fist_lora.calibration.subspace import truncated_svd
from fist_lora.calibration.warmup import warmup_head
from fist_lora.config.schema import CalibrationConfig, TaskConfig
from fist_lora.data.calibration import sample_subset
from fist_lora.modeling.targets import resolve_targets

TASK = TaskConfig(name="toy", kind="glue", epochs=1, max_length=16, metric="accuracy", num_labels=2)
CALIB = CalibrationConfig(num_samples=10, seed=42, warmup_steps=5, warmup_batch_size=8, microbatch_size=4)


@pytest.fixture
def patched_loader(monkeypatch):
    monkeypatch.setattr(fist_mod, "load_model", lambda cfg, task, device: tiny_roberta())


def test_warmup_trains_only_the_head_for_exactly_the_requested_steps():
    model = tiny_roberta()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ds = Dataset.from_list(cls_rows(10))  # 10 examples < 7 steps x 4: must cycle, not stop early
    losses = warmup_head(model, ["classifier"], ds, PadCollator("seq_cls"), steps=7, batch_size=4,
                         lr=1e-3, seed=0, device=torch.device("cpu"))
    assert len(losses) == 7
    for n, p in model.named_parameters():
        changed = not torch.equal(p, before[n])
        assert changed == n.startswith("classifier."), n
        assert not p.requires_grad
    assert not model.training


def test_subset_is_seeded_and_without_replacement():
    ds = Dataset.from_list([{"i": i} for i in range(100)])
    a, b = sample_subset(ds, 20, 42), sample_subset(ds, 20, 42)
    assert a["i"] == b["i"] and len(set(a["i"])) == 20
    assert a["i"] != sample_subset(ds, 20, 43)["i"]
    with pytest.raises(ValueError):
        sample_subset(ds, 101, 0)


def test_calibration_outputs_and_determinism(patched_loader):
    ds = Dataset.from_list(cls_rows(30))
    cfg = roberta_model_cfg()
    res1, meta = fist_mod.compute_fist_calibration(cfg, TASK, CALIB, ds, PadCollator("seq_cls"), 4, torch.device("cpu"))
    res2, _ = fist_mod.compute_fist_calibration(cfg, TASK, CALIB, ds, PadCollator("seq_cls"), 4, torch.device("cpu"))
    assert len(res1) == 8 and meta["num_examples"] == 10
    fresh_targets = resolve_targets(tiny_roberta(), ROBERTA_TARGETS, 8)
    for name, m in res1.items():
        assert torch.equal(m.fisher.U, res2[name].fisher.U) and torch.equal(m.proj_fisher, res2[name].proj_fisher)
        assert m.proj_plain.shape == (4, 4) and m.fisher.U.shape == (16, 4) and m.fisher.Vh.shape == (4, 16)
        # Warm-up touches only the head, so the plain SVD is that of the pretrained W0.
        ref = truncated_svd(fresh_targets[name].weight, 4)
        assert torch.allclose(m.plain.S, ref.S, atol=1e-5)


def test_calibration_cache_roundtrip_and_rng_restoration(patched_loader, tmp_path):
    ds = Dataset.from_list(cls_rows(30))
    cfg = roberta_model_cfg()
    torch.manual_seed(123)
    state = torch.get_rng_state()
    res, meta = fist_mod.get_fist_calibration(cfg, TASK, CALIB, ds, PadCollator("seq_cls"), 4, torch.device("cpu"), tmp_path)
    assert torch.equal(torch.get_rng_state(), state)  # calibration does not perturb the run's RNG
    cached, meta2 = fist_mod.get_fist_calibration(cfg, TASK, CALIB, ds, PadCollator("seq_cls"), 4, torch.device("cpu"), tmp_path)
    assert meta["key"] == meta2["key"]
    for name in res:
        assert torch.equal(res[name].proj_fisher, cached[name].proj_fisher)
        assert torch.equal(res[name].fisher.Vh, cached[name].fisher.Vh)
    other = fist_mod.calibration_key(cfg, TASK, CalibrationConfig(num_samples=11), 4)
    assert other != meta["key"]
    # Bookkeeping that does not change the statistics does not change the key.
    assert fist_mod.calibration_key(cfg, TASK, CalibrationConfig(**{**CALIB.__dict__, "cache_dir": "x"}), 4) == meta["key"]
