"""Parameter counts and experiment configuration integrity."""

from pathlib import Path

import pytest
import torch

from conftest import LLAMA_TARGETS, tiny_llama
from fist_lora.config import expand_runs, load_experiment, stable_hash
from fist_lora.config.load import CONFIG_ROOT, apply_overrides, load_yaml_tree
from fist_lora.config.schema import ArchitectureSpec
from fist_lora.methods.registry import METHODS
from fist_lora.modeling.targets import resolve_targets
from fist_lora.params import (
    ARCHS,
    lora_params,
    r2_params,
    storage_mb,
    verify_architecture,
)

EXPERIMENTS = sorted((CONFIG_ROOT / "experiments").glob("*.yaml"))


def test_table5_counts():
    assert [r2_params(ARCHS["roberta"], "roberta", r) for r in (8, 16, 24)] == [6144, 24576, 55296]
    assert [r2_params(ARCHS["llama"], "llama", r) for r in (8, 16, 24)] == [14336, 57344, 129024]
    assert [r2_params(ARCHS["mistral"], "mistral", r) for r in (8, 16, 24)] == [14336, 57344, 129024]


def test_lora_counts():
    assert lora_params(ARCHS["roberta"], "roberta", 8) == 96 * 8 * 2048
    assert lora_params(ARCHS["llama"], "llama", 8) == 19_988_480
    assert lora_params(ARCHS["mistral"], "mistral", 8) == 20_971_520  # GQA k/v 1024, MLP 14336
    assert storage_mb(19_988_480) == pytest.approx((79.95392, 159.90784))


def test_analytic_counts_match_a_real_gqa_model():
    model = tiny_llama()  # 2 layers, hidden 16, 4 heads / 2 kv heads, intermediate 24
    arch = ArchitectureSpec(num_layers=2, hidden_size=16, intermediate_size=24, num_attention_heads=4, num_key_value_heads=2)
    targets = resolve_targets(model, LLAMA_TARGETS, 14)
    verify_architecture(targets, arch)
    assert lora_params(arch, "llama", 4) == sum(4 * (m.in_features + m.out_features) for m in targets.values())
    with pytest.raises(ValueError):
        verify_architecture(targets, ArchitectureSpec(2, 16, 32, 4, 2))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_every_experiment_config_builds(path: Path):
    exp = load_experiment(path)
    for m in exp.methods:
        assert m.registry_key in METHODS, m.name
        uses_rank = METHODS[m.registry_key].uses_rank
        assert all((r is not None) == uses_rank for r in m.ranks), m.name
    assert len({m.name for m in exp.methods}) == len(exp.methods)


def test_experiment_grids():
    t2 = load_experiment(CONFIG_ROOT / "experiments/glue.yaml")
    assert [t.name for t in t2.tasks] == ["cola", "rte", "mrpc", "stsb", "qnli", "sst2"]
    assert len(expand_runs(t2)) == 6 * (1 + 1 + 4 * 3) * 3 == 252
    assert t2.model.alpha == 16 and t2.seeds == [42, 123, 456] and t2.model.expected_num_targets == 96
    assert {t.name: (t.epochs, t.max_length) for t in t2.tasks} == {
        "cola": (30, 128), "rte": (50, 256), "mrpc": (30, 256), "stsb": (30, 256), "qnli": (10, 256), "sst2": (20, 128)}
    c = t2.calibration
    assert (c.num_samples, c.warmup_steps, c.warmup_batch_size, c.warmup_lr, c.clip_quantile, c.eps, c.init_scale) == (
        256, 100, 32, 1e-3, 0.95, 1e-8, 0.01)
    lrs = {m.name: m.lr for m in t2.methods}
    assert lrs["lora"] == 4e-4 and all(lrs[k] == 1e-3 for k in ("lora_xs", "lora_sb", "fist_no_fisher", "fist"))
    for name, batch in (("commonsense", 64), ("math", 128)):
        e = load_experiment(CONFIG_ROOT / f"experiments/{name}.yaml")
        assert len(expand_runs(e)) == 13 * 3
        assert e.training.per_device_train_batch_size * e.training.gradient_accumulation_steps == batch
        assert e.model.alpha == 32 and e.model.precision == "nf4" and e.model.expected_num_targets == 224
        assert e.calibration.warmup_batch_size == 4 and not e.model.train_head


def test_overrides_and_hash():
    tree = load_yaml_tree(CONFIG_ROOT / "experiments/glue.yaml")
    t2 = apply_overrides(tree, ["training.logging_steps=50", "seeds=[1]"])
    assert t2["training"]["logging_steps"] == 50 and t2["seeds"] == [1]
    assert stable_hash({"a": 1, "output_dir": "x"}) == stable_hash({"a": 1, "output_dir": "y"})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
    with pytest.raises(ValueError):
        apply_overrides(tree, ["no_equals_sign"])


def test_targets_regexes_compile_and_select_attention_output_only():
    import re

    pats = load_experiment(CONFIG_ROOT / "experiments/glue.yaml").model.target_modules
    names = ["roberta.encoder.layer.3.attention.output.dense", "roberta.encoder.layer.3.output.dense",
             "roberta.encoder.layer.3.intermediate.dense", "classifier.dense", "roberta.encoder.layer.0.attention.self.key"]
    hits = [n for n in names if any(re.fullmatch(p, n) for p in pats)]
    assert hits == [names[0], names[4]]
    assert torch.__version__.startswith("2.4")
