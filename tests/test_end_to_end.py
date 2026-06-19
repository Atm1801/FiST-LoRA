"""End-to-end runs on tiny hub models (needs network once; run with `pytest -m network`)."""

import json

import pytest
import torch

from fist_lora.config import load_experiment
from fist_lora.config.load import CONFIG_ROOT
from fist_lora.config.schema import RunSpec
from fist_lora.training.run import run_dir, run_is_complete, run_single

pytestmark = pytest.mark.network


def spec(config: str, tmp_path, task: str, method: str, rank, seed: int, extra=()):
    exp = load_experiment(CONFIG_ROOT / f"experiments/{config}.yaml",
                          [f"output_dir={tmp_path}/out", f"calibration.cache_dir={tmp_path}/cache", *extra])
    return RunSpec(exp, exp.task(task), exp.method(method), rank, seed)


def test_glue_run_is_deterministic_resumable_and_trains_R(tmp_path):
    s1 = spec("smoke_glue", tmp_path / "a", "pair", "fist", 4, 42)
    m1 = run_single(s1, torch.device("cpu"))
    assert run_is_complete(s1)
    assert m1["adapter_params"] == 20 * 16 and len(m1["per_epoch"]) == 2
    assert m1["best"] == max(p["value"] for p in m1["per_epoch"])
    s2 = spec("smoke_glue", tmp_path / "b", "pair", "fist", 4, 42)
    m2 = run_single(s2, torch.device("cpu"))
    assert m1["per_epoch"] == m2["per_epoch"] and m1["train_loss"] == m2["train_loss"]
    state = torch.load(run_dir(s1) / "adapter.pt")
    Rs = [v for k, v in state.items() if k.endswith(".R")]
    assert len(Rs) == 20 and all(abs(torch.linalg.norm(R).item() - 0.01) > 1e-6 for R in Rs)  # moved from R_init
    changed = spec("smoke_glue", tmp_path / "a", "pair", "fist", 4, 42, ["training.logging_steps=2"])
    assert not run_is_complete(changed)


def test_math_run_with_evaluation(tmp_path):
    extra = [
        "evaluation.enabled=true",
        "evaluation.gsm8k={data_files: tests/fixtures/gsm8k_mini.jsonl, max_new_tokens: 8}",
        "evaluation.math={data_files: tests/fixtures/math_mini.jsonl, max_new_tokens: 8}",
    ]
    s = spec("smoke_causal", tmp_path, "metamath_mini", "fist", 4, 42, extra)
    m = run_single(s, torch.device("cpu"))
    for bench in ("gsm8k", "math"):
        res = m["benchmarks"][bench]
        assert res["num_examples"] == 6 and 0.0 <= res["exact_match"] <= 1.0
        with open(run_dir(s) / f"predictions_{bench}.jsonl") as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == 6 and {"prediction", "reference", "correct", "generation"} <= set(rows[0])


def test_batched_greedy_equals_unbatched():
    from transformers import AutoModelForCausalLM

    from fist_lora.config.schema import ModelConfig
    from fist_lora.evaluation.math import greedy_generate
    from fist_lora.modeling.load import load_tokenizer

    cfg = load_experiment(CONFIG_ROOT / "experiments/smoke_causal.yaml").model
    tok = load_tokenizer(cfg)
    model = AutoModelForCausalLM.from_pretrained(cfg.name, revision=cfg.revision).eval()
    prompts = ["### Question:\nWhat is 2 plus 2?\n\n### Answer:\n", "### Question:\nA much longer question about many apples and pears?\n\n### Answer:\n", "Hi"]
    batched = greedy_generate(model, tok, prompts, max_new_tokens=6, batch_size=3)
    single = [greedy_generate(model, tok, [p], max_new_tokens=6, batch_size=1)[0] for p in prompts]
    assert batched == single
    assert isinstance(cfg, ModelConfig)


def test_lm_eval_wrapper_on_tiny_model():
    pytest.importorskip("lm_eval")
    from transformers import AutoModelForCausalLM

    from fist_lora.config.schema import EvaluationConfig
    from fist_lora.evaluation.commonsense import evaluate_commonsense
    from fist_lora.modeling.load import load_tokenizer

    cfg = load_experiment(CONFIG_ROOT / "experiments/smoke_causal.yaml").model
    model = AutoModelForCausalLM.from_pretrained(cfg.name, revision=cfg.revision)
    res = evaluate_commonsense(model, load_tokenizer(cfg), EvaluationConfig(lm_eval_tasks=["arc_easy"], limit=4, batch_size=2))
    assert res["arc_easy"]["lm_eval_task"] == "arc_easy" and 0.0 <= res["arc_easy"]["accuracy"] <= 1.0


@pytest.mark.parametrize("method", ["fist", "lora_sb", "lora", "pissa"])
def test_restored_adapter_reproduces_final_validation_metric(tmp_path, method):
    import numpy as np

    from fist_lora.data.collate import Collator
    from fist_lora.data.loading import load_task_data
    from fist_lora.evaluation.glue import glue_metrics
    from fist_lora.modeling.load import load_tokenizer
    from fist_lora.training.restore import restore_trained_model

    s = spec("smoke_glue", tmp_path, "pair", method, 4, 123)
    m = run_single(s, torch.device("cpu"))
    model = restore_trained_model(s, torch.device("cpu"))
    tok = load_tokenizer(s.experiment.model)
    _, eval_ds = load_task_data(s.task, tok)
    batch = Collator(tok, "seq_cls")([eval_ds[i] for i in range(len(eval_ds))])
    with torch.no_grad():
        out = model(**batch)
    f1 = glue_metrics(out.logits.numpy(), batch["labels"].numpy(), "f1")["f1"]
    assert np.isclose(f1, m["final"])
    # Continuous check: the Trainer's final-epoch validation loss (one eval batch of 32).
    with open(run_dir(s) / "log_history.jsonl") as f:
        history = [json.loads(line) for line in f]
    final_eval_loss = [h["eval_loss"] for h in history if "eval_loss" in h][-1]
    assert out.loss.item() == pytest.approx(final_eval_loss, rel=1e-5)
