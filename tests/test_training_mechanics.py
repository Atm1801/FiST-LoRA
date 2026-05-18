"""Gradient flow under gradient checkpointing, and head pairing across methods."""

import pytest
import torch

from conftest import LLAMA_TARGETS, llama_model_cfg, lm_rows, pad_batch, tiny_llama, tiny_roberta
from fist_lora.adapters.frozen_outer import FrozenOuterLinear
from fist_lora.adapters.inject import AdapterFactors, inject_frozen_outer, set_trainable
from fist_lora.calibration.subspace import truncated_svd
from fist_lora.config.schema import MethodEntry, RunSpec, TaskConfig
from fist_lora.modeling.targets import resolve_targets
from fist_lora.training.trainer import adapter_state_dict, prepare_for_training


def fist_like_llama(R_scale=0.01):
    model = tiny_llama()
    targets = resolve_targets(model, LLAMA_TARGETS, 14)
    factors = {}
    for n, m in targets.items():
        svd = truncated_svd(m.weight, 2)
        factors[n] = AdapterFactors(svd.U, svd.Vh, R_scale * torch.eye(2))
    inject_frozen_outer(model, factors, 16.0)
    set_trainable(model, ["lm_head"], train_head=False)
    return model


def spec_for(cfg):
    task = TaskConfig(name="t", kind="metamath", epochs=1, max_length=32)
    from fist_lora.config.schema import (
        CalibrationConfig,
        EvaluationConfig,
        ExperimentConfig,
        LoraSBConfig,
        TrainingConfig,
    )

    exp = ExperimentConfig("e", cfg, [task], [MethodEntry("fist", 1e-4, [2])], [0], CalibrationConfig(),
                           TrainingConfig(), EvaluationConfig(), LoraSBConfig(), "results/x")
    return RunSpec(exp, task, exp.methods[0], 2, 0)


def test_every_R_gets_gradient_with_gradient_checkpointing():
    model = fist_like_llama()
    prepare_for_training(model, spec_for(llama_model_cfg(gradient_checkpointing=True)))
    model.train()
    model(**pad_batch(lm_rows(3), "causal_lm")).loss.backward()
    adapters = [m for m in model.modules() if isinstance(m, FrozenOuterLinear)]
    assert len(adapters) == 14
    for m in adapters:
        assert m.R.grad is not None and m.R.grad.abs().sum() > 0


def test_reentrant_checkpointing_without_input_grads_starves_adapters():
    """The failure mode prepare_for_training prevents: reentrant checkpointing with a frozen backbone."""
    model = fist_like_llama()
    model.lm_head.weight.requires_grad_(True)  # a trainable head keeps the loss differentiable
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.train()
    with pytest.warns(UserWarning):
        model(**pad_batch(lm_rows(3), "causal_lm")).loss.backward()
    assert model.lm_head.weight.grad is not None
    grads = [m.R.grad for m in model.modules() if isinstance(m, FrozenOuterLinear)]
    assert all(g is None for g in grads)  # training would silently update nothing but the head


def test_head_initialisation_is_identical_across_methods_for_a_seed():
    """Runs are paired by seed: the head exists before any adapter consumes randomness."""
    from conftest import roberta_model_cfg
    from fist_lora.adapters.peft_methods import build_peft_model
    from fist_lora.calibration.fist import plain_svd_factors
    from fist_lora.methods.registry import frozen_outer_factors, get_method
    from fist_lora.reproducibility import make_generator

    def head_after(attach):
        torch.manual_seed(7)
        model = tiny_roberta(seed=7)  # builds the random classifier under the run seed
        model = attach(model) or model
        return {".".join(n.split(".")[-2:]): p.detach().clone() for n, p in model.named_parameters()
                if "classifier" in n and "original_module" not in n}

    def lora(model):
        return build_peft_model(model, roberta_model_cfg(), 4, "lora")

    def lora_xs(model):
        targets = resolve_targets(model, [r".*\.attention\.self\.(query|key|value)", r".*\.attention\.output\.dense"], 8)
        f = frozen_outer_factors(get_method("lora_xs"), 4, 0.01, plain_svd=plain_svd_factors(targets, 4),
                                 generator=make_generator(7))
        inject_frozen_outer(model, f, 4.0)

    h1, h2 = head_after(lora), head_after(lora_xs)
    assert h1.keys() == h2.keys() and len(h1) == 4
    for k in h1:
        assert torch.equal(h1[k], h2[k]), k


def test_adapter_state_dict_contains_factors():
    model = fist_like_llama()
    sd = adapter_state_dict(model)
    assert sum(k.endswith(".R") for k in sd) == 14
    assert sum(k.endswith(".B") for k in sd) == 14 and sum(k.endswith(".scaling") for k in sd) == 14
