"""One training run (task x method x rank x seed), end to end.

Order of operations (matters for reproducibility and pairing across methods):

1. data are loaded and tokenised (deterministic);
2. FiST calibration, if the method needs it, is loaded from cache or computed with its
   own seed on a separate model, and the RNG state is restored afterwards;
3. the run seed is set and a *fresh* pretrained model is loaded - for sequence
   classification its head is randomly initialised here, identically for every method
   at a given seed (so per-seed comparisons are paired);
4. the method's adapter is attached and the trainable set fixed;
5. training with the Hugging Face Trainer; GLUE is evaluated after every epoch;
6. the 7B models are evaluated on their external benchmarks;
7. metrics, log history, manifest, resolved config and adapter weights are written.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import torch
import yaml

from fist_lora.adapters.inject import count_parameters, inject_frozen_outer, set_trainable
from fist_lora.adapters.peft_methods import build_peft_model
from fist_lora.calibration.fist import get_fist_calibration, plain_svd_factors
from fist_lora.calibration.lora_sb import effective_lr, estimate_update, lora_sb_factors
from fist_lora.config.load import repo_path, stable_hash
from fist_lora.config.schema import RunSpec
from fist_lora.data.collate import Collator
from fist_lora.data.loading import load_task_data
from fist_lora.evaluation.glue import best_and_final, epoch_metrics, make_compute_metrics
from fist_lora.methods.registry import adapter_scaling, frozen_outer_factors, get_method
from fist_lora.modeling.load import default_device, load_model, load_tokenizer
from fist_lora.modeling.targets import resolve_targets
from fist_lora.params import verify_architecture
from fist_lora.reproducibility import (
    configure_determinism,
    environment_manifest,
    make_generator,
    seed_everything,
    write_json,
)
from fist_lora.training.trainer import (
    adapter_state_dict,
    build_trainer,
    prepare_for_training,
    training_arguments,
)

log = logging.getLogger(__name__)


def run_dir(spec: RunSpec) -> Path:
    return repo_path(spec.experiment.output_dir) / "runs" / spec.run_id


def run_is_complete(spec: RunSpec) -> bool:
    path = run_dir(spec) / "metrics.json"
    if not path.exists():
        return False
    with open(path) as f:
        return json.load(f).get("config_hash") == stable_hash(spec.to_dict())


def attach_method(model, spec: RunSpec, train_dataset, collator, calibration, device) -> dict:
    """Attach the adapter of ``spec.method`` to ``model`` in place; return method metadata."""
    exp, cfg = spec.experiment, spec.experiment.model
    method = get_method(spec.method.registry_key)
    info: dict = {"kind": method.kind}

    if method.kind == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return info
    if method.kind == "peft":
        return {**info, "peft_model": build_peft_model(model, cfg, spec.rank, method.peft_init)}

    targets = resolve_targets(model, cfg.target_modules, cfg.expected_num_targets)
    scaling = adapter_scaling(method, cfg.alpha, spec.rank)
    init_scale = spec.method.params.get("init_scale", exp.calibration.init_scale)
    if method.outer == "lora_sb":
        t = exp.training
        eta = effective_lr(
            spec.method.lr, t.warmup_ratio, len(train_dataset), exp.lora_sb.estimation_batch_size, spec.task.epochs
        )
        update = estimate_update(
            model, targets, train_dataset, collator, exp.lora_sb, eta, spec.seed, device,
            exp.calibration.modules_per_pass,
        )
        factors = lora_sb_factors(update, spec.rank, exp.calibration.svd_dtype)
        info["lora_sb_eta_eff"] = eta
    else:
        plain = None
        if method.needs_plain_svd:
            plain = plain_svd_factors(targets, spec.rank, exp.calibration.svd_dtype)
        factors = frozen_outer_factors(
            method, spec.rank, init_scale, calibration=calibration, plain_svd=plain,
            generator=make_generator(spec.seed),
        )
    if set(factors) != set(targets):
        raise RuntimeError("adapter factors do not cover exactly the target modules")
    inject_frozen_outer(model, factors, scaling)
    set_trainable(model, cfg.head_modules, cfg.train_head)
    info["scaling"] = scaling
    if method.inner in ("gradient_projected", "sigma_diag"):
        info["init_scale"] = init_scale
    return info


def task_calibration(exp, task, train_ds, collator, device):
    """The FiST calibration of ``task`` at the largest rank any method of ``exp`` uses."""
    max_rank = max(r for m in exp.methods for r in m.ranks if r is not None)
    return get_fist_calibration(
        exp.model, task, exp.calibration, train_ds, collator, max_rank, device,
        repo_path(exp.calibration.cache_dir),
    )


def evaluate_causal(model, tokenizer, spec: RunSpec) -> dict:
    ev = spec.experiment.evaluation
    if not ev.enabled:
        return {}
    if spec.experiment.model.gradient_checkpointing:
        model.gradient_checkpointing_disable()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.config.use_cache = True
    if spec.task.kind == "commonsense":
        from fist_lora.evaluation.commonsense import evaluate_commonsense

        return evaluate_commonsense(model, tokenizer, ev)
    if spec.task.kind == "metamath":
        from fist_lora.evaluation.math import evaluate_math_benchmarks

        return evaluate_math_benchmarks(model, tokenizer, ev)
    raise ValueError(f"no external evaluation for task kind {spec.task.kind!r}")


def run_single(spec: RunSpec, device: torch.device | None = None) -> dict:
    exp, cfg = spec.experiment, spec.experiment.model
    device = device or default_device()
    method = get_method(spec.method.registry_key)
    if method.uses_rank != (spec.rank is not None):
        raise ValueError(f"method {spec.method.name} {'needs' if method.uses_rank else 'takes no'} rank")
    out = run_dir(spec)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(spec.to_dict(), f, sort_keys=False)
    configure_determinism(exp.training.deterministic, exp.training.tf32)
    t_start = time.time()

    tokenizer = load_tokenizer(cfg)
    train_ds, eval_ds = load_task_data(spec.task, tokenizer)
    collator = Collator(tokenizer, cfg.task_type)

    calibration, calib_meta = None, None
    if method.calibration == "fist":
        calibration, calib_meta = task_calibration(exp, spec.task, train_ds, collator, device)

    seed_everything(spec.seed)
    model = load_model(cfg, spec.task, device)
    if cfg.architecture is not None:
        verify_architecture(resolve_targets(model, cfg.target_modules, cfg.expected_num_targets), cfg.architecture)
    info = attach_method(model, spec, train_ds, collator, calibration, device)
    model = info.pop("peft_model", model)
    params = count_parameters(model, cfg.head_modules)
    log.info("%s: %s", spec.run_id, params)

    prepare_for_training(model, spec)
    compute_metrics = make_compute_metrics(spec.task.metric) if spec.task.kind == "glue" else None
    trainer = build_trainer(
        model, training_arguments(spec, out / "trainer", eval_ds is not None),
        train_ds, eval_ds, tokenizer, collator, compute_metrics,
    )
    t_train = time.time()
    train_output = trainer.train()
    train_seconds = time.time() - t_train
    shutil.rmtree(out / "trainer", ignore_errors=True)  # save_strategy="no": nothing to keep
    history = trainer.state.log_history

    metrics: dict = {
        "run_id": spec.run_id,
        "task": spec.task.name,
        "method": spec.method.name,
        "registry_method": spec.method.registry_key,
        "rank": spec.rank,
        "seed": spec.seed,
        "config_hash": stable_hash(spec.to_dict()),
        **params,
        "method_info": info,
        "train_loss": train_output.training_loss,
        "train_seconds": train_seconds,
        "calibration": calib_meta,
    }
    with open(out / "log_history.jsonl", "w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")
    if exp.training.save_adapter and method.kind != "full":
        torch.save(adapter_state_dict(model), out / "adapter.pt")
    write_json(out / "manifest.json", {
        "environment": environment_manifest(),
        "model": {"name": cfg.name, "revision": cfg.revision, "precision": cfg.precision},
        "task_dataset": {"name": spec.task.dataset, "revision": spec.task.dataset_revision},
        "config_hash": metrics["config_hash"],
    })
    if spec.task.kind == "glue":
        per_epoch = epoch_metrics(history, spec.task.metric)
        metrics.update(metric=spec.task.metric, per_epoch=per_epoch, **best_and_final(per_epoch))
    else:
        # Training results are persisted before the (long) external evaluation, so a failed
        # evaluation can be redone from the saved adapter with scripts/evaluate.py.
        write_json(out / "train_metrics.json", metrics)
        metrics["benchmarks"] = evaluate_and_record(model, tokenizer, spec, out)
    metrics["total_seconds"] = time.time() - t_start
    write_json(out / "metrics.json", metrics)  # written last: its presence marks completion
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def evaluate_and_record(model, tokenizer, spec: RunSpec, out: Path) -> dict:
    """External benchmarks of a causal-LM run; per-example predictions go to jsonl files."""
    results = evaluate_causal(model, tokenizer, spec)
    for name, res in results.items():
        per_example = res.pop("per_example", None)
        if per_example is not None:
            with open(out / f"predictions_{name}.jsonl", "w") as f:
                for row in per_example:
                    f.write(json.dumps(row) + "\n")
    return results
