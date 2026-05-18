"""Hugging Face Trainer set-up shared by every method.

AdamW with Trainer defaults (beta1 0.9, beta2 0.999, eps 1e-8), zero weight decay,
linear schedule with warm-up ratio, bf16 autocast on GPU.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments

from fist_lora.adapters.frozen_outer import FrozenOuterLinear
from fist_lora.config.schema import RunSpec


def prepare_for_training(model: nn.Module, spec: RunSpec) -> None:
    """Gradient checkpointing and the quantised-backbone flag, applied identically to all methods.

    * Non-reentrant checkpointing plus input-embedding grads: with the default reentrant
      implementation and a fully frozen backbone, no checkpointed segment has an input
      requiring grad, and the adapters inside it silently receive *no gradient*.
    * Transformers refuses to train an ``is_quantized`` model unless trainable adapters
      are attached, which it detects through PEFT or the ``_hf_peft_config_loaded`` flag.
      Frozen-outer adapters are attached without PEFT, so the flag is set explicitly
      (it is only consulted by the Trainer guard, gradient checkpointing and
      ``save_pretrained``; adapters are saved with :func:`adapter_state_dict` instead).
    """
    cfg = spec.experiment.model
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
    if getattr(model, "is_quantized", False) and any(isinstance(m, FrozenOuterLinear) for m in model.modules()):
        model._hf_peft_config_loaded = True


def training_arguments(spec: RunSpec, output_dir: Path, has_eval: bool) -> TrainingArguments:
    t = spec.experiment.training
    use_bf16 = torch.cuda.is_available()  # CPU smoke runs use fp32
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=spec.task.epochs,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=spec.method.lr,
        weight_decay=t.weight_decay,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        bf16=use_bf16,
        tf32=None,
        eval_strategy="epoch" if (has_eval and t.eval_every_epoch) else "no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=t.logging_steps,
        logging_first_step=True,
        seed=spec.seed,
        data_seed=spec.seed,
        group_by_length=t.group_by_length,
        length_column_name="length",
        dataloader_num_workers=t.dataloader_num_workers,
        report_to="none",
        disable_tqdm=False,
        gradient_checkpointing=False,  # handled by prepare_for_training (non-reentrant)
        remove_unused_columns=True,
    )


def build_trainer(model, args, train_dataset, eval_dataset, tokenizer, collator, compute_metrics) -> Trainer:
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Everything needed to rebuild the trained adapter on the same base checkpoint.

    Frozen-outer adapters: B, A, R and the scale; plus any trainable non-adapter tensors
    (the task head, PEFT LoRA factors).  Full fine-tuning is not saved here.
    """
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, FrozenOuterLinear):
            state[f"{name}.B"] = module.B.detach().cpu()
            state[f"{name}.A"] = module.A.detach().cpu()
            state[f"{name}.R"] = module.R.detach().cpu()
            state[f"{name}.scaling"] = torch.tensor(module.scaling)
    for name, p in model.named_parameters():
        if p.requires_grad and not name.endswith(".R"):
            state[name] = p.detach().cpu()
    return state
