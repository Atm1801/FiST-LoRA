"""Head warm-up before calibration.

The task head is trained alone for ``steps`` optimiser steps with the backbone frozen,
so that calibration gradients reflect task signal rather than a random head.  The
warmed model is used only for calibration and discarded; training runs load a fresh
model.

``torch.optim.AdamW`` with its defaults besides the learning rate; the model is in
train mode (dropout active in the frozen encoder).  Batches are drawn from a seeded
shuffle of the full training set, re-shuffled each epoch, until exactly ``steps`` steps
have been taken, also on datasets smaller than steps x batch_size (e.g. RTE).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fist_lora.calibration.grad_stats import to_device
from fist_lora.modeling.targets import head_parameters
from fist_lora.reproducibility import make_generator

log = logging.getLogger(__name__)


def warmup_head(
    model: nn.Module,
    head_modules: list[str],
    dataset,
    collate_fn,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> list[float]:
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(head_parameters(model, head_modules).values())
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=lr)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=make_generator(seed), collate_fn=collate_fn
    )
    model.train()
    losses: list[float] = []
    while len(losses) < steps:
        for batch in loader:
            loss = model(**to_device(batch, device)).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if len(losses) == steps:
                break
    for p in params:
        p.requires_grad_(False)
    model.eval()
    log.info("head warm-up: %d steps, loss %.4f -> %.4f", steps, losses[0], losses[-1])
    return losses
