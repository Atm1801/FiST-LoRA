"""Original LoRA-SB initialisation (Ponkshe et al. 2024, arXiv:2411.19557, Alg. 1).

Faithful to the official implementation (github.com/CERT-Lab/lora-sb, commit
4feb81c243e6e762c64b736c968b232caf7b44c4: ``utils/gradient_utils.py``,
``utils/initialization_utils.py``, ``train_glue.py``, ``train_cr.py``,
``train_arithmetic.py``):

* The update of the first optimiser step is approximated as
      Delta W_avg = -eta_eff * sign( sum_batches dL_batch/dW ),
  with each batch loss divided by the number of examples taken from the batch, the first
  ``num_samples`` examples of a shuffled training loader of batch size
  ``estimation_batch_size`` (GLUE 2 / 128, commonsense 170 / 10, math 50 / 3), and
  eta_eff = lr / (warmup_ratio * ceil(N_train / estimation_batch_size) * epochs).
* The model is in train mode (dropout active) and carries the head it will be trained
  with (no warm-up, no separate calibration model), so the estimate is per seed.
* SVD(Delta W_avg) = U S V^T; B = U_r, A = V_r^T, R_init = S_r / s with s = 1 (the code
  sets lora_alpha = lora_r), and B, A are frozen.

Deviation: the official code uses a randomized ``torch.svd_lowrank`` (niter=10) and casts
the factors to bf16; an exact SVD in fp32 is used here for all methods.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from fist_lora.adapters.inject import AdapterFactors
from fist_lora.calibration.grad_stats import (
    GradientRecorder,
    activation_gradients,
    module_groups,
    replayable_rng,
    to_device,
)
from fist_lora.calibration.subspace import truncated_svd
from fist_lora.config.schema import LoraSBConfig
from fist_lora.reproducibility import make_generator


def effective_lr(lr: float, warmup_ratio: float, num_train: int, estimation_batch_size: int, epochs: int) -> float:
    total_steps = math.ceil(num_train / estimation_batch_size) * epochs
    return lr / (warmup_ratio * total_steps)


def estimate_update(
    model: torch.nn.Module,
    targets: dict[str, torch.nn.Module],
    train_dataset,
    collate_fn,
    cfg: LoraSBConfig,
    eta: float,
    seed: int,
    device: torch.device,
    modules_per_pass: int | None = None,
) -> dict[str, torch.Tensor]:
    """-eta * sign(summed gradient) for every target module (fp32, on CPU)."""
    was_training = model.training
    model.train()
    out: dict[str, torch.Tensor] = {}
    with replayable_rng() as replay:
        for group in module_groups(sorted(targets), modules_per_pass):
            replay()  # identical dropout masks in every pass
            loader = DataLoader(
                train_dataset, batch_size=cfg.estimation_batch_size, shuffle=True,
                generator=make_generator(seed), collate_fn=collate_fn,
            )
            seen = 0
            with activation_gradients(model), GradientRecorder(
                {k: targets[k] for k in group}, per_example=False
            ) as rec:
                for batch in loader:
                    take = min(len(batch["input_ids"]), cfg.num_samples - seen)
                    if take <= 0:
                        break
                    batch = to_device({k: v[:take] for k, v in batch.items()}, device)
                    (model(**batch).loss / take).backward()
                    model.zero_grad(set_to_none=True)
                    seen += take
            if seen != cfg.num_samples:
                raise RuntimeError(f"LoRA-SB estimation saw {seen} examples, expected {cfg.num_samples}")
            for name in group:
                out[name] = (-eta * torch.sign(rec.grad_sum[name])).cpu()
    model.train(was_training)
    return out


def lora_sb_factors(update: dict[str, torch.Tensor], rank: int, svd_dtype: str = "float32") -> dict[str, AdapterFactors]:
    factors = {}
    for name, dw in update.items():
        svd = truncated_svd(dw, rank, svd_dtype)
        factors[name] = AdapterFactors(B=svd.U, A=svd.Vh, R=torch.diag(svd.S))  # R = S_r / s, s = 1
    return factors
