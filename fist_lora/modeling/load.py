"""Model and tokenizer loading at pinned hub revisions."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from fist_lora.config.schema import ModelConfig, TaskConfig
from fist_lora.modeling.quant import nf4_quantization_config


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(cfg: ModelConfig):
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.name, revision=cfg.revision)
    if cfg.pad_token == "unk":
        # LLM-Adapters convention for LLaMA: pad with <unk> (id 0) so padding differs from EOS.
        tok.pad_token = tok.unk_token
    elif cfg.pad_token == "eos" or (tok.pad_token is None and cfg.task_type == "causal_lm"):
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: ModelConfig, task: TaskConfig, device: torch.device | None = None):
    """Load a fresh pretrained model.

    For sequence classification the head is freshly (randomly) initialised by
    ``from_pretrained`` using the *current* global RNG state, which is why callers seed
    immediately before loading (identical head init for every method at a given seed).
    """
    device = device or default_device()
    if cfg.precision == "nf4":
        if cfg.task_type != "causal_lm":
            raise ValueError("NF4 loading is only used for the 7B causal-LM experiments")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.name,
            revision=cfg.revision,
            quantization_config=nf4_quantization_config(),
            torch_dtype=torch.bfloat16,
            device_map={"": device.index or 0} if device.type == "cuda" else None,
        )
    elif cfg.precision == "bf16-mixed":
        if cfg.task_type == "seq_cls":
            # ignore_mismatched_sizes: a checkpoint that already carries a classifier with a
            # different label count gets that layer re-initialised (roberta-large has none).
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg.name, revision=cfg.revision, num_labels=task.num_labels, ignore_mismatched_sizes=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg.name, revision=cfg.revision)
        model.to(device)
    else:
        raise ValueError(f"unknown precision {cfg.precision!r}")
    return model
