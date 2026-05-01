"""
Utility functions for FiST-LoRA.

Provides:
- warmup_classifier_head: Brief head-only training for meaningful calibration
- make_calibration_loader: Create a DataLoader for Fisher/gradient computation
- compute_warmup_steps: Correct warmup step calculation accounting for multi-GPU
- set_seed: Reproducibility
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, List


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_classifier_head(
    model: nn.Module,
    train_dataset,
    tokenizer,
    num_steps: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    head_keywords: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Train ONLY the head (encoder frozen) for num_steps to produce
    meaningful loss gradients for Fisher/gradient calibration.

    [PoC-VALIDATED] This warmup is used ONLY for calibration.
    Training models start with a FRESH random head, NOT the warmed head.
    Loading warmed head state into training models causes gradient conflicts
    and training collapse.

    Args:
        model: The model with a randomly initialized head.
        train_dataset: HuggingFace dataset (tokenized, torch format).
        tokenizer: Tokenizer for DataCollatorWithPadding.
        num_steps: Number of warmup steps.
        batch_size: Batch size for warmup.
        lr: Learning rate for head warmup.
        head_keywords: Keywords identifying head parameters.
        device: Device override.

    Returns:
        The model with warmed-up head (in-place).
    """
    from transformers import DataCollatorWithPadding

    if device is None:
        device = next(model.parameters()).device

    if head_keywords is None:
        head_keywords = ["classifier", "score", "qa_outputs"]

    # Freeze encoder, unfreeze classifier head
    for param in model.parameters():
        param.requires_grad_(False)

    head_params = []
    for name, param in model.named_parameters():
        if any(kw in name for kw in head_keywords):
            param.requires_grad_(True)
            head_params.append(param)

    if not head_params:
        print("[Warmup] No classifier head found; skipping warmup.")
        return model

    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    n_examples = min(num_steps * batch_size, len(train_dataset))
    loader = DataLoader(
        train_dataset.select(range(n_examples)),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    optim = torch.optim.AdamW(head_params, lr=lr)
    model.train()

    step = 0
    losses = []
    for batch in loader:
        if step >= num_steps:
            break
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k != "length"
        }
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None:
            break
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        step += 1

    if losses:
        print(
            f"[Warmup] Trained classifier head for {step} steps. "
            f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    # Restore all grads to False
    for param in model.parameters():
        param.requires_grad_(False)

    model.eval()
    return model


def make_calibration_loader(
    train_dataset,
    tokenizer,
    num_samples: int = 256,
    batch_size: int = 16,
) -> DataLoader:
    """
    Return a DataLoader over the first `num_samples` training examples.
    Uses DataCollatorWithPadding for variable-length batching.

    Args:
        train_dataset: Tokenized training dataset.
        tokenizer: Tokenizer for collation.
        num_samples: Number of calibration samples.
        batch_size: Batch size.

    Returns:
        DataLoader for calibration.
    """
    from transformers import DataCollatorWithPadding

    n = min(num_samples, len(train_dataset))
    subset = train_dataset.select(range(n))
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    return DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collator)


def compute_warmup_steps(
    dataset_size: int,
    per_device_batch_size: int,
    num_epochs: int,
    warmup_ratio: float = 0.06,
    gradient_accumulation_steps: int = 1,
) -> int:
    """
    Compute warmup steps accounting for multi-GPU and gradient accumulation.

    [PoC-VALIDATED] The old formula ignored multi-GPU, causing warmup
    to be N x too long with N GPUs.

    Args:
        dataset_size: Number of training examples.
        per_device_batch_size: Per-device batch size.
        num_epochs: Number of training epochs.
        warmup_ratio: Fraction of total steps for warmup.
        gradient_accumulation_steps: Gradient accumulation steps.

    Returns:
        Number of warmup steps.
    """
    num_gpus = max(1, torch.cuda.device_count())
    effective_bs = per_device_batch_size * num_gpus * gradient_accumulation_steps
    steps_per_epoch = math.ceil(dataset_size / effective_bs)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    return warmup_steps


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    print("[Warning] No GPU found; running on CPU (will be slow).")
    return torch.device("cpu")


def print_trainable_summary(model: nn.Module, model_name: str = ""):
    """Print a summary of trainable vs total parameters."""
    from fist_lora.model import count_trainable_params, count_total_params

    trainable = count_trainable_params(model)
    total = count_total_params(model)
    prefix = f"[{model_name}] " if model_name else ""
    print(
        f"{prefix}Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )
    return trainable, total
