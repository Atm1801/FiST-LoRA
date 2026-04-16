"""Full-rank-budget baselines built with Hugging Face PEFT.

* LoRA (Hu et al. 2022): A ~ kaiming-uniform(a=sqrt(5)), B = 0, scale alpha/r
  (PEFT ``init_lora_weights=True``).
* PiSSA (Meng et al. 2024): PEFT ``init_lora_weights="pissa"`` is the PiSSA authors'
  upstreamed implementation: exact SVD W = U S V^T, B = U_r sqrt(S_r)/sqrt(s),
  A = sqrt(S_r) V_r^T / sqrt(s), and the base weight is replaced by the residual
  W - s * B A, so the model output is unchanged at initialisation.
"""

from __future__ import annotations

import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

from fist_lora.config.schema import ModelConfig


def _target_regex(patterns: list[str]) -> str:
    # PEFT treats a string target_modules as a regex matched with re.fullmatch.
    return "|".join(f"(?:{p})" for p in patterns)


def build_peft_model(model: nn.Module, cfg: ModelConfig, rank: int, init: str) -> nn.Module:
    if init not in ("lora", "pissa"):
        raise ValueError(f"unknown PEFT init {init!r}")
    if init == "pissa" and cfg.precision == "nf4":
        # PEFT refuses PiSSA on quantised weights: QPiSSA requires the SVD of the
        # full-precision weight and re-quantising the residual, a separate pipeline.
        raise NotImplementedError(
            "PiSSA on an NF4 backbone needs the QPiSSA residual-quantisation pipeline, "
            "which is not implemented; run PiSSA with precision=bf16-mixed."
        )
    task_type = TaskType.SEQ_CLS if cfg.task_type == "seq_cls" else TaskType.CAUSAL_LM
    peft_cfg = LoraConfig(
        task_type=task_type,
        r=rank,
        lora_alpha=cfg.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=_target_regex(cfg.target_modules),
        modules_to_save=list(cfg.head_modules) if cfg.train_head else None,
        init_lora_weights=True if init == "lora" else "pissa",
    )
    peft_model = get_peft_model(model, peft_cfg)
    if cfg.task_type == "seq_cls" and not cfg.train_head:
        raise ValueError("sequence classification requires a trainable head")
    return peft_model
