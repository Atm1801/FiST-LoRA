"""
FiST-LoRA: Ablation Experiments
=================================

Run all ablation variants on GLUE (RoBERTa-large) to isolate the contribution
of Fisher-weighted SVD and gradient-projected R initialization.

Ablation variants:
  fist_full      — Fisher-SVD outer + gradient R (full method)
  no_fisher      — Plain SVD outer + gradient R (Fisher contribution)
  no_grad        — Fisher-SVD outer + zero R (gradient init contribution)
  lora_xs        — Plain SVD outer + zero R (baseline)
  sigma_init     — Fisher-SVD outer + diag(sigma) R (alternative init)
  scale_sweep_*  — Fisher-SVD outer + gradient R at different init_scale

Usage:
    python experiments/run_ablations.py
    python experiments/run_ablations.py --tasks sst2 mrpc --ranks 8
"""

import os
import sys
import json
import time
import math
import argparse
import traceback
from pathlib import Path

import torch
import numpy as np

from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import evaluate

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fist_lora.fisher import compute_diagonal_fisher
from fist_lora.init import (
    fisher_weighted_svd,
    plain_svd,
    gradient_projected_R,
    zero_R,
    sigma_R,
)
from fist_lora.model import (
    inject_fist_lora,
    count_trainable_params,
    collect_plain_svd,
    collect_fisher_svd,
)
from fist_lora.utils import (
    set_seed,
    warmup_classifier_head,
    make_calibration_loader,
    compute_warmup_steps,
    get_device,
    print_trainable_summary,
)

MODEL_NAME = "roberta-large"
TARGET_MODULES = ["query", "key", "value"]
ALPHA = 32.0
CALIBRATION_SAMPLES = 256

TASKS = {
    "sst2": {"num_labels": 2, "metric": "accuracy", "epochs": 3,
             "text_keys": ("sentence", None)},
    "mrpc": {"num_labels": 2, "metric": "f1", "epochs": 5,
             "text_keys": ("sentence1", "sentence2")},
    "qnli": {"num_labels": 2, "metric": "accuracy", "epochs": 3,
             "text_keys": ("question", "sentence")},
    "rte":  {"num_labels": 2, "metric": "accuracy", "epochs": 5,
             "text_keys": ("sentence1", "sentence2")},
}

ABLATION_VARIANTS = {
    "fist_full": {
        "outer_init": "fisher_svd",
        "inner_init": "gradient_projected",
        "init_scale": 0.01,
    },
    "no_fisher": {
        "outer_init": "plain_svd",
        "inner_init": "gradient_projected",
        "init_scale": 0.01,
    },
    "no_grad": {
        "outer_init": "fisher_svd",
        "inner_init": "zero",
        "init_scale": 0.0,
    },
    "lora_xs": {
        "outer_init": "plain_svd",
        "inner_init": "zero",
        "init_scale": 0.0,
    },
    "sigma_init": {
        "outer_init": "fisher_svd",
        "inner_init": "sigma",
        "init_scale": 0.01,
    },
    "scale_sweep_001": {
        "outer_init": "fisher_svd",
        "inner_init": "gradient_projected",
        "init_scale": 0.001,
    },
    "scale_sweep_01": {
        "outer_init": "fisher_svd",
        "inner_init": "gradient_projected",
        "init_scale": 0.01,
    },
    "scale_sweep_1": {
        "outer_init": "fisher_svd",
        "inner_init": "gradient_projected",
        "init_scale": 0.1,
    },
}


def tokenize_glue(raw_dataset, tokenizer, text_keys, max_length=128):
    key1, key2 = text_keys

    def tokenize_fn(examples):
        if key2 is None:
            enc = tokenizer(examples[key1], truncation=True, max_length=max_length)
        else:
            enc = tokenizer(examples[key1], examples[key2], truncation=True, max_length=max_length)
        enc["labels"] = examples["label"]
        return enc

    cols_to_remove = list(raw_dataset["train"].column_names)
    train_tok = raw_dataset["train"].map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    val_split = "validation" if "validation" in raw_dataset else "validation_matched"
    val_tok = raw_dataset[val_split].map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    train_tok.set_format("torch")
    val_tok.set_format("torch")
    return train_tok, val_tok


def build_compute_metrics(metric_name, task_name):
    metric = evaluate.load("glue", task_name)
    if metric_name == "f1":
        def fn(eval_pred):
            preds = eval_pred[0].argmax(axis=-1)
            return {"f1": metric.compute(predictions=preds, references=eval_pred[1])["f1"]}
        return fn
    def fn(eval_pred):
        preds = eval_pred[0].argmax(axis=-1)
        return {"accuracy": metric.compute(predictions=preds, references=eval_pred[1])["accuracy"]}
    return fn


def build_ablation_model(
    model, variant_cfg, rank,
    svd_plain, svd_fisher,
    calibration_loader, base_model_for_grad,
):
    """Build model for a specific ablation variant."""
    outer = variant_cfg["outer_init"]
    inner = variant_cfg["inner_init"]
    init_scale = variant_cfg["init_scale"]

    # Select outer matrices
    if outer == "fisher_svd":
        svd = svd_fisher
    else:
        svd = svd_plain

    B_dict = {n: BA[0] for n, BA in svd.items()}
    A_dict = {n: BA[2] for n, BA in svd.items()}

    # Select inner init
    if inner == "zero":
        R_dict = {n: zero_R(rank) for n in svd}
    elif inner == "sigma":
        R_dict = {n: sigma_R(svd[n][1], rank, init_scale) for n in svd}
    elif inner == "gradient_projected":
        frozen_ba = {n: (B_dict[n], A_dict[n]) for n in svd}
        R_dict = gradient_projected_R(
            base_model_for_grad, calibration_loader, frozen_ba,
            TARGET_MODULES, CALIBRATION_SAMPLES, ALPHA, rank,
            init_scale=init_scale,
        )
        R_dict = {n: R_dict.get(n, zero_R(rank)) for n in svd}
    else:
        raise ValueError(f"Unknown inner_init: {inner}")

    model = inject_fist_lora(model, TARGET_MODULES, B_dict, A_dict, R_dict, ALPHA, rank)
    return model


def main():
    parser = argparse.ArgumentParser(description="FiST-LoRA ablation experiments")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output_dir", type=str, default="results/ablations")
    parser.add_argument("--report_to", type=str, default="none")
    args = parser.parse_args()

    task_names = args.tasks or list(TASKS.keys())
    variant_names = args.variants or list(ABLATION_VARIANTS.keys())
    device = get_device()
    print(f"Device: {device}")
    print(f"Tasks: {task_names}")
    print(f"Variants: {variant_names}")
    print(f"Ranks: {args.ranks}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    for task_name in task_names:
        task_cfg = TASKS[task_name]
        print(f"\n{'#' * 70}")
        print(f"# TASK: {task_name.upper()}")
        print(f"{'#' * 70}")

        if task_name not in results:
            results[task_name] = {}

        raw_dataset = load_dataset("glue", task_name)
        train_tok, val_tok = tokenize_glue(raw_dataset, tokenizer, task_cfg["text_keys"])
        calibration_loader = make_calibration_loader(train_tok, tokenizer, CALIBRATION_SAMPLES)

        # Calibration model
        print("Loading base model for calibration...")
        base_model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=task_cfg["num_labels"]
        ).to(device)
        base_model = warmup_classifier_head(base_model, train_tok, tokenizer)

        print("Computing Fisher...")
        fisher_diags = compute_diagonal_fisher(
            base_model, calibration_loader, TARGET_MODULES, CALIBRATION_SAMPLES
        )

        svd_plain_by_rank = {}
        svd_fisher_by_rank = {}
        for rank in args.ranks:
            svd_plain_by_rank[rank] = collect_plain_svd(base_model, TARGET_MODULES, rank)
            svd_fisher_by_rank[rank] = collect_fisher_svd(
                base_model, fisher_diags, TARGET_MODULES, rank
            )

        for rank in args.ranks:
            for seed in args.seeds:
                for variant_name in variant_names:
                    run_key = f"rank_{rank}_seed_{seed}"
                    if run_key not in results[task_name]:
                        results[task_name][run_key] = {}
                    if variant_name in results[task_name][run_key]:
                        existing = results[task_name][run_key][variant_name]
                        if "metric" in existing and "error" not in existing:
                            print(f"  [Resume] Skipping {task_name}/{run_key}/{variant_name}")
                            continue

                    print(f"\n  {task_name} | rank={rank} | seed={seed} | {variant_name}")

                    set_seed(seed)
                    start_time = time.time()

                    model = RobertaForSequenceClassification.from_pretrained(
                        MODEL_NAME, num_labels=task_cfg["num_labels"]
                    )

                    try:
                        model = build_ablation_model(
                            model, ABLATION_VARIANTS[variant_name], rank,
                            svd_plain_by_rank[rank], svd_fisher_by_rank[rank],
                            calibration_loader, base_model,
                        )
                    except Exception:
                        print(f"    [ERROR] Ablation model build failed:")
                        traceback.print_exc()
                        results[task_name][run_key][variant_name] = {"error": "build_failed"}
                        continue

                    trainable, _ = print_trainable_summary(model, variant_name)

                    warmup_steps = compute_warmup_steps(len(train_tok), 32, task_cfg["epochs"])

                    training_args = TrainingArguments(
                        output_dir=os.path.join(
                            args.output_dir, task_name, f"{variant_name}_r{rank}_s{seed}"
                        ),
                        num_train_epochs=task_cfg["epochs"],
                        per_device_train_batch_size=32,
                        per_device_eval_batch_size=64,
                        learning_rate=1e-3,
                        weight_decay=0.0,
                        warmup_steps=warmup_steps,
                        eval_strategy="epoch",
                        save_strategy="no",
                        logging_steps=50,
                        seed=seed,
                        fp16=False,
                        report_to=args.report_to,
                        dataloader_num_workers=0,
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_tok,
                        eval_dataset=val_tok,
                        processing_class=tokenizer,
                        data_collator=DataCollatorWithPadding(tokenizer),
                        compute_metrics=build_compute_metrics(task_cfg["metric"], task_name),
                    )

                    try:
                        trainer.train()
                        eval_results = trainer.evaluate()
                    except Exception:
                        traceback.print_exc()
                        results[task_name][run_key][variant_name] = {
                            "error": "training_failed",
                            "trainable_params": trainable,
                        }
                        del model, trainer
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue

                    elapsed = time.time() - start_time
                    metric_key = f"eval_{task_cfg['metric']}"
                    score = eval_results.get(metric_key, "?")
                    print(f"    {metric_key} = {score}  ({elapsed:.0f}s)")

                    results[task_name][run_key][variant_name] = {
                        "metric": eval_results,
                        "trainable_params": trainable,
                        "time_seconds": elapsed,
                        "variant_config": ABLATION_VARIANTS[variant_name],
                    }

                    del model, trainer
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 100)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 100)

    metric_keys = {"sst2": "eval_accuracy", "mrpc": "eval_f1",
                   "qnli": "eval_accuracy", "rte": "eval_accuracy"}

    for rank in args.ranks:
        print(f"\n--- Rank = {rank} ---")
        header = f"{'Variant':22s}"
        for task in task_names:
            header += f" | {task:>8s}"
        header += f" | {'Avg':>8s}"
        print(header)
        print("-" * len(header))

        for variant in variant_names:
            row = f"{variant:22s}"
            means = []
            for task_name in task_names:
                scores = []
                for seed in args.seeds:
                    key = f"rank_{rank}_seed_{seed}"
                    r = results.get(task_name, {}).get(key, {}).get(variant, {})
                    if "metric" in r:
                        mk = metric_keys.get(task_name, "eval_accuracy")
                        val = r["metric"].get(mk)
                        if val is not None:
                            scores.append(val)
                if scores:
                    m = np.mean(scores)
                    means.append(m)
                    row += f" | {m:7.4f}"
                else:
                    row += f" | {'N/A':>8s}"

            if means:
                row += f" | {np.mean(means):7.4f}"
            else:
                row += f" | {'N/A':>8s}"
            print(row)


if __name__ == "__main__":
    main()
