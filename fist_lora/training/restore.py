"""Rebuild a trained model from a run directory (base checkpoint + saved adapter)."""

from __future__ import annotations

import torch

from fist_lora.adapters.inject import AdapterFactors, inject_frozen_outer, set_trainable
from fist_lora.adapters.peft_methods import build_peft_model
from fist_lora.config.schema import RunSpec
from fist_lora.methods.registry import get_method
from fist_lora.modeling.load import load_model
from fist_lora.training.run import run_dir


def restore_trained_model(spec: RunSpec, device: torch.device):
    """Base model + trained adapter of ``spec``; every saved tensor must be consumed."""
    cfg = spec.experiment.model
    method = get_method(spec.method.registry_key)
    if method.kind == "full":
        raise ValueError("full fine-tuning runs do not save weights")
    state = torch.load(run_dir(spec) / "adapter.pt", map_location="cpu", weights_only=True)
    model = load_model(cfg, spec.task, device)
    if method.kind == "frozen_outer":
        names = [k[: -len(".R")] for k in state if k.endswith(".R")]
        factors = {n: AdapterFactors(state.pop(f"{n}.B"), state.pop(f"{n}.A"), state.pop(f"{n}.R")) for n in names}
        scalings = {state.pop(f"{n}.scaling").item() for n in names}
        if len(scalings) != 1:
            raise ValueError(f"inconsistent adapter scalings {scalings}")
        inject_frozen_outer(model, factors, scalings.pop())
        set_trainable(model, cfg.head_modules, cfg.train_head)
    else:
        # PiSSA re-derives the same residual base weights from the same exact SVD.
        model = build_peft_model(model, cfg, spec.rank, method.peft_init)
    params = dict(model.named_parameters())
    missing = [k for k in state if k not in params]
    if missing:
        raise KeyError(f"saved tensors without a matching parameter: {missing[:5]}")
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(params[k].device, params[k].dtype))
    return model.eval()
