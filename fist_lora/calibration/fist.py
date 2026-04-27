"""FiST-LoRA calibration, computed once per task.

Pipeline on a dedicated calibration model (discarded afterwards):

1. warm the task head (``warmup.py``);
2. one eval-mode pass over the N calibration examples collecting, per target module,
   the diagonal empirical Fisher F = E[g^2] and the mean gradient G = E[g];
3. per module: plain SVD of W0 and SVD of the Fisher-weighted W~ (``fisher.py``), truncated to
   the largest rank needed;
4. per module and subspace: M = U^T G Vh^T (r_max x r_max).  For any r <= r_max the
   projection B_r^T G A_r^T is the leading block M[:r, :r], so every rank
   of the experiment is served by one calibration.

Only these small factors are kept; F and G (d x k each) are freed per module group.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fist_lora import __version__
from fist_lora.calibration.fisher import clip_and_normalize, fisher_weighted_matrix, quantile_linear
from fist_lora.calibration.grad_stats import fisher_and_mean_gradient, module_groups
from fist_lora.calibration.subspace import TruncatedSVD, truncated_svd
from fist_lora.calibration.warmup import warmup_head
from fist_lora.config.load import stable_hash
from fist_lora.config.schema import CalibrationConfig, ModelConfig, TaskConfig
from fist_lora.data.calibration import sample_subset
from fist_lora.modeling.load import load_model
from fist_lora.modeling.quant import effective_weight
from fist_lora.modeling.targets import resolve_targets
from fist_lora.reproducibility import seed_everything

log = logging.getLogger(__name__)


@dataclass
class ModuleCalibration:
    plain: TruncatedSVD
    fisher: TruncatedSVD
    proj_plain: torch.Tensor  # U_plain^T G Vh_plain^T, (r_max, r_max)
    proj_fisher: torch.Tensor  # U_fisher^T G Vh_fisher^T, (r_max, r_max)
    diagnostics: dict[str, float]


def calibration_key(model_cfg: ModelConfig, task: TaskConfig, calib: CalibrationConfig, max_rank: int) -> str:
    data_fields = {
        k: v for k, v in asdict(task).items()
        if k not in ("epochs", "max_eval_samples", "eval_split", "metric")
    }
    return stable_hash({
        "model": {k: v for k, v in asdict(model_cfg).items() if k not in ("gradient_checkpointing",)},
        "task": data_fields,
        "calibration": {k: v for k, v in asdict(calib).items() if k not in ("modules_per_pass",)},
        "max_rank": max_rank,
        "version": __version__,
    })


def _save(path: Path, modules: dict[str, ModuleCalibration], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "modules": {
            name: {
                "plain": (m.plain.U, m.plain.S, m.plain.Vh),
                "fisher": (m.fisher.U, m.fisher.S, m.fisher.Vh),
                "proj_plain": m.proj_plain,
                "proj_fisher": m.proj_fisher,
                "diagnostics": m.diagnostics,
            }
            for name, m in modules.items()
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load(path: Path) -> tuple[dict[str, ModuleCalibration], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    modules = {
        name: ModuleCalibration(
            plain=TruncatedSVD(*d["plain"]),
            fisher=TruncatedSVD(*d["fisher"]),
            proj_plain=d["proj_plain"],
            proj_fisher=d["proj_fisher"],
            diagnostics=d["diagnostics"],
        )
        for name, d in payload["modules"].items()
    }
    return modules, payload["meta"]


def _fisher_diagnostics(fisher: torch.Tensor, grad: torch.Tensor, q: float, eps: float) -> dict[str, float]:
    """Summary statistics of one module's Fisher estimate.

    ``clipped_mean`` is the normalisation denominator without eps.  When it is not much larger
    than eps (it can be for small models or tiny gradients), F_bar is no longer centred at
    one; the normalisation is still applied as defined, and the condition is logged.
    """
    f_bar = clip_and_normalize(fisher, q, eps)
    clipped_mean = torch.minimum(fisher, quantile_linear(fisher, q)).mean().item()
    return {
        "clipped_mean": clipped_mean,
        "eps_relative_to_clipped_mean": eps / clipped_mean if clipped_mean > 0 else float("inf"),
        "fisher_mean": fisher.mean().item(),
        "fisher_max": fisher.max().item(),
        "fisher_bar_max": f_bar.max().item(),
        "fraction_zero_fisher": (fisher == 0).float().mean().item(),
        "grad_fro_norm": torch.linalg.norm(grad).item(),
    }


def compute_fist_calibration(
    model_cfg: ModelConfig,
    task: TaskConfig,
    calib: CalibrationConfig,
    train_dataset,
    collate_fn,
    max_rank: int,
    device: torch.device,
) -> tuple[dict[str, ModuleCalibration], dict]:
    t0 = time.time()
    seed_everything(calib.seed)
    model = load_model(model_cfg, task, device)
    warm_losses = warmup_head(
        model, model_cfg.head_modules, train_dataset, collate_fn,
        steps=calib.warmup_steps, batch_size=calib.warmup_batch_size, lr=calib.warmup_lr,
        seed=calib.seed, device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    subset = sample_subset(train_dataset, calib.num_samples, calib.seed)
    targets = resolve_targets(model, model_cfg.target_modules, model_cfg.expected_num_targets)

    def batches():
        return DataLoader(subset, batch_size=calib.microbatch_size, shuffle=False, collate_fn=collate_fn)

    results: dict[str, ModuleCalibration] = {}
    for group in module_groups(sorted(targets), calib.modules_per_pass):
        fisher, grad, n = fisher_and_mean_gradient(
            model, batches(), {k: targets[k] for k in group}, model_cfg.task_type, device
        )
        if n != calib.num_samples:
            raise RuntimeError(f"calibration used {n} examples, expected {calib.num_samples}")
        for name in group:
            F, G = fisher.pop(name), grad.pop(name)
            W0 = effective_weight(targets[name])
            plain = truncated_svd(W0, max_rank, calib.svd_dtype)
            weighted = truncated_svd(
                fisher_weighted_matrix(W0, F, calib.clip_quantile, calib.eps), max_rank, calib.svd_dtype
            )
            results[name] = ModuleCalibration(
                plain=plain.to("cpu"),
                fisher=weighted.to("cpu"),
                proj_plain=(plain.U.T @ G @ plain.Vh.T).cpu(),
                proj_fisher=(weighted.U.T @ G @ weighted.Vh.T).cpu(),
                diagnostics=_fisher_diagnostics(F, G, calib.clip_quantile, calib.eps),
            )
            del F, G, W0
        if device.type == "cuda":
            torch.cuda.empty_cache()
    swamped = [n for n, m in results.items() if m.diagnostics["eps_relative_to_clipped_mean"] > 1e-2]
    if swamped:
        log.warning(
            "%d/%d modules have mean clipped Fisher < 100 * eps; F_bar is then not centred at 1 "
            "(e.g. %s)", len(swamped), len(results), swamped[0],
        )
    meta = {
        "num_examples": calib.num_samples,
        "modules_eps_not_negligible": len(swamped),
        "num_modules": len(results),
        "max_rank": max_rank,
        "warmup_loss_first": warm_losses[0],
        "warmup_loss_last": warm_losses[-1],
        "seconds": time.time() - t0,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results, meta


def get_fist_calibration(
    model_cfg: ModelConfig,
    task: TaskConfig,
    calib: CalibrationConfig,
    train_dataset,
    collate_fn,
    max_rank: int,
    device: torch.device,
    cache_root: Path,
) -> tuple[dict[str, ModuleCalibration], dict]:
    """Load the cached calibration for this (model, task, calibration config) or compute it.

    Calibration is shared by all ranks and seeds of a task, and uses its
    own seed, so it does not depend on which run happens to trigger it.
    """
    key = calibration_key(model_cfg, task, calib, max_rank)
    path = cache_root / "calibration" / task.name / f"{key}.pt"
    if path.exists():
        modules, meta = _load(path)
        log.info("loaded calibration %s", path)
    else:
        rng = torch.get_rng_state()
        modules, meta = compute_fist_calibration(
            model_cfg, task, calib, train_dataset, collate_fn, max_rank, device
        )
        torch.set_rng_state(rng)
        meta["key"] = key
        _save(path, modules, meta)
        log.info("saved calibration %s (%.1fs)", path, meta["seconds"])
    meta = dict(meta, key=key, path=str(path))
    return modules, meta


def plain_svd_factors(
    targets: dict[str, torch.nn.Module], max_rank: int, dtype: str = "float32"
) -> dict[str, TruncatedSVD]:
    """Data-free truncated SVD of every target W0 (LoRA-XS; no calibration data needed)."""
    return {name: truncated_svd(effective_weight(m), max_rank, dtype).to("cpu") for name, m in targets.items()}
