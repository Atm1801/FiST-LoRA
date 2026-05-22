"""4-bit NF4 path. Needs CUDA + bitsandbytes: `pytest -m gpu`.

Skipped on CPU-only machines.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

LLAMA_TARGETS = [r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)", r".*\.mlp\.(gate_proj|up_proj|down_proj)"]


@pytest.fixture
def nf4_model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pytest.importorskip("bitsandbytes")
    from transformers import AutoModelForCausalLM

    from fist_lora.modeling.quant import nf4_quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", revision="9fb191250dd56d0ba7ec9785a025ed29c03d5998",
        quantization_config=nf4_quantization_config(), torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    return model


def test_nf4_calibration_statistics_and_training_step(nf4_model):
    from fist_lora.adapters.frozen_outer import FrozenOuterLinear
    from fist_lora.adapters.inject import AdapterFactors, inject_frozen_outer, set_trainable
    from fist_lora.calibration.grad_stats import fisher_and_mean_gradient
    from fist_lora.calibration.subspace import truncated_svd
    from fist_lora.modeling.quant import effective_weight, is_4bit_linear
    from fist_lora.modeling.targets import resolve_targets

    model = nf4_model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    targets = resolve_targets(model, LLAMA_TARGETS, 14)
    assert all(is_4bit_linear(m) for m in targets.values())
    ids = torch.randint(3, 1000, (2, 12), device="cuda")
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}
    F, G, n = fisher_and_mean_gradient(model, [batch], targets, "causal_lm", torch.device("cuda"))
    factors = {}
    for name, m in targets.items():
        W = effective_weight(m)
        assert F[name].shape == W.shape and torch.isfinite(F[name]).all()
        svd = truncated_svd(W, 2)
        factors[name] = AdapterFactors(svd.U, svd.Vh, 0.01 * torch.eye(2))
    inject_frozen_outer(model, factors, 16.0)
    set_trainable(model, ["lm_head"], train_head=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()
    for m in model.modules():
        if isinstance(m, FrozenOuterLinear):
            assert m.R.grad is not None and m.R.grad.abs().sum() > 0


def test_trainer_accepts_frozen_outer_on_quantized_model(nf4_model, tmp_path):
    from transformers import Trainer, TrainingArguments

    from fist_lora.adapters.inject import AdapterFactors, inject_frozen_outer, set_trainable
    from fist_lora.calibration.subspace import truncated_svd
    from fist_lora.modeling.quant import effective_weight
    from fist_lora.modeling.targets import resolve_targets

    model = nf4_model
    targets = resolve_targets(model, LLAMA_TARGETS, 14)
    factors = {n: AdapterFactors(truncated_svd(effective_weight(m), 2).U, truncated_svd(effective_weight(m), 2).Vh,
                                 torch.zeros(2, 2)) for n, m in targets.items()}
    inject_frozen_outer(model, factors, 16.0)
    set_trainable(model, ["lm_head"], train_head=False)
    model._hf_peft_config_loaded = True  # what fist_lora.training.trainer.prepare_for_training sets
    Trainer(model=model, args=TrainingArguments(output_dir=str(tmp_path), report_to="none"))
