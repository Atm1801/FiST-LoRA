"""
FiST-LoRA: GLUE Experiments
============================

RoBERTa-large on 8 GLUE tasks comparing:
  1. lora           — standard LoRA via HuggingFace PEFT
  2. lora_xs        — plain SVD outer matrices, zero R init
  3. fist_no_fisher — plain SVD outer matrices, gradient-projected R init
  4. fist_full      — Fisher-weighted SVD outer, gradient-projected R init
  5. pissa          — PiSSA (SVD-init A,B both trainable) via PEFT

Results are saved to {output_dir}/results.json and printed as tables.

Usage:
    python experiments/run_glue.py
    python experiments/run_glue.py --tasks sst2 mrpc --ranks 8 --methods fist_full lora
    python experiments/run_glue.py --config experiments/configs/glue.yaml
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
import yaml

from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import evaluate

# Add project root to path
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
from fist_lora.layers import FiSTLoRALinear
from fist_lora.model import (
    inject_fist_lora,
    count_trainable_params,
    count_total_params,
    collect_plain_svd,
    collect_fisher_svd,
)
from fist_lora.utils import (
    set_seed,
    warmup_classifier_head,
    make_calibration_loader,
    get_device,
    print_trainable_summary,
)

# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

# Per-task config (epochs/max_seq_len follow LoRA-SB and LoRA paper recipes).
# Larger tasks: fewer epochs. Small tasks: many epochs to reach convergence.
TASKS = {
    "sst2": {
        "num_labels": 2,
        "metric": "accuracy",
        "epochs": 20,
        "max_seq_len": 128,
        "text_keys": ("sentence", None),
        "val_split": "validation",
    },
    "mrpc": {
        "num_labels": 2,
        "metric": "f1",
        "epochs": 30,
        "max_seq_len": 256,
        "text_keys": ("sentence1", "sentence2"),
        "val_split": "validation",
    },
    "qnli": {
        "num_labels": 2,
        "metric": "accuracy",
        "epochs": 10,
        "max_seq_len": 256,
        "text_keys": ("question", "sentence"),
        "val_split": "validation",
    },
    "rte": {
        "num_labels": 2,
        "metric": "accuracy",
        "epochs": 50,
        "max_seq_len": 256,
        "text_keys": ("sentence1", "sentence2"),
        "val_split": "validation",
    },
    "mnli": {
        "num_labels": 3,
        "metric": "accuracy",
        "epochs": 10,
        "max_seq_len": 128,
        "text_keys": ("premise", "hypothesis"),
        "val_split": "validation_matched",
    },
    "qqp": {
        "num_labels": 2,
        "metric": "accuracy",
        "epochs": 10,
        "max_seq_len": 256,
        "text_keys": ("question1", "question2"),
        "val_split": "validation",
    },
    "cola": {
        "num_labels": 2,
        "metric": "matthews_correlation",
        "epochs": 30,
        "max_seq_len": 128,
        "text_keys": ("sentence", None),
        "val_split": "validation",
    },
    "stsb": {
        "num_labels": 1,
        "metric": "spearmanr",
        "epochs": 30,
        "max_seq_len": 256,
        "text_keys": ("sentence1", "sentence2"),
        "val_split": "validation",
    },
}

ALL_METHODS = ["lora", "lora_xs", "fist_no_fisher", "fist_full", "pissa", "lora_sb"]
MODEL_NAME = "roberta-large"
TARGET_MODULES = ["query", "key", "value"]
ALPHA = 32.0
CALIBRATION_SAMPLES = 256
INIT_SCALE = 0.01
WARMUP_RATIO = 0.06  # LoRA-SB / LoRA paper standard

# LR per method (matches each method's published recipe)
METHOD_LRS = {
    "lora": 4e-4,
    "pissa": 4e-4,
    "lora_xs": 1e-3,
    "fist_no_fisher": 1e-3,
    "fist_full": 1e-3,
    "lora_sb": 1e-3,
}


# ============================================================
# HELPERS
# ============================================================

def tokenize_glue(raw_dataset, tokenizer, text_keys, max_length=128):
    """Tokenize GLUE train + validation splits."""
    key1, key2 = text_keys

    def tokenize_fn(examples):
        if key2 is None:
            enc = tokenizer(examples[key1], truncation=True, max_length=max_length)
        else:
            enc = tokenizer(
                examples[key1], examples[key2], truncation=True, max_length=max_length
            )
        enc["labels"] = examples["label"]
        return enc

    cols_to_remove = list(raw_dataset["train"].column_names)

    train_tok = raw_dataset["train"].map(
        tokenize_fn, batched=True, remove_columns=cols_to_remove
    )
    val_split = "validation" if "validation" in raw_dataset else "validation_matched"
    val_tok = raw_dataset[val_split].map(
        tokenize_fn, batched=True, remove_columns=cols_to_remove
    )

    train_tok.set_format("torch")
    val_tok.set_format("torch")
    return train_tok, val_tok


def build_compute_metrics(metric_name: str, task_name: str):
    """Return a compute_metrics function for HuggingFace Trainer."""
    metric = evaluate.load("glue", task_name)

    if task_name == "stsb":
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.squeeze()
            result = metric.compute(predictions=preds, references=labels)
            return {k: v for k, v in result.items()}
        return compute_metrics

    if metric_name == "matthews_correlation":
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            result = metric.compute(predictions=preds, references=labels)
            return {"matthews_correlation": result["matthews_correlation"]}
        return compute_metrics

    if metric_name == "f1":
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            result = metric.compute(predictions=preds, references=labels)
            return {"f1": result["f1"]}
        return compute_metrics

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        result = metric.compute(predictions=preds, references=labels)
        return {"accuracy": result["accuracy"]}
    return compute_metrics


def safe_run(fn, fallback, description=""):
    """Run fn(); on any exception log the error and return fallback."""
    try:
        return fn()
    except Exception:
        print(f"[ERROR] {description} failed:")
        traceback.print_exc()
        return fallback


def load_existing_results(path: str) -> dict:
    """Load results JSON if it exists."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[Resume] Could not load {path}: {e} - starting fresh.")
        return {}


def is_valid_result(entry: dict) -> bool:
    """A result is valid if it has a 'metric' dict and no 'error' key."""
    return "metric" in entry and "error" not in entry


def compute_gradient_svd(model, dataloader, target_module_names, rank, num_samples=256):
    """
    Compute SVD of the average gradient for each target module.
    Used by LoRA-SB: B and A come from gradient SVD, not weight SVD.
    """
    device = next(model.parameters()).device
    model.eval()

    target_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(
            name.endswith(t) for t in target_module_names
        ):
            target_modules[name] = module

    for module in target_modules.values():
        module.weight.requires_grad_(True)

    grad_accum = {
        name: torch.zeros_like(module.weight, device=device)
        for name, module in target_modules.items()
    }

    samples_seen = 0
    for batch in dataloader:
        if samples_seen >= num_samples:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss if outputs.loss is not None else outputs.logits.sum()
        loss.backward()

        for name, module in target_modules.items():
            if module.weight.grad is not None:
                grad_accum[name] += module.weight.grad.detach()

        model.zero_grad()
        samples_seen += batch[next(iter(batch))].size(0)

    for module in target_modules.values():
        module.weight.requires_grad_(False)

    svd_results = {}
    for name in target_modules:
        G = grad_accum[name] / max(samples_seen, 1)
        U, S, Vt = torch.linalg.svd(G.float(), full_matrices=False)
        svd_results[name] = (U[:, :rank].clone(), S[:rank].clone(), Vt[:rank, :].clone())

    return svd_results


def apply_adapter(model, method, rank, svd_plain, svd_fisher, grad_R_plain, grad_R_fisher,
                  svd_grad=None, grad_R_grad=None):
    """Apply the specified adapter method to the model."""
    from peft import get_peft_model, LoraConfig, TaskType

    if method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=rank,
            lora_alpha=ALPHA,
            lora_dropout=0.0,
            target_modules=TARGET_MODULES,
            bias="none",
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, peft_config)

    elif method == "pissa":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=rank,
            lora_alpha=ALPHA,
            lora_dropout=0.0,
            target_modules=TARGET_MODULES,
            bias="none",
            modules_to_save=["classifier"],
            init_lora_weights="pissa",
        )
        model = get_peft_model(model, peft_config)

    elif method == "lora_xs":
        B_dict = {n: BA[0] for n, BA in svd_plain.items()}
        A_dict = {n: BA[2] for n, BA in svd_plain.items()}
        R_dict = {n: zero_R(rank) for n in svd_plain}
        model = inject_fist_lora(
            model, TARGET_MODULES, B_dict, A_dict, R_dict, ALPHA, rank
        )

    elif method == "fist_no_fisher":
        B_dict = {n: BA[0] for n, BA in svd_plain.items()}
        A_dict = {n: BA[2] for n, BA in svd_plain.items()}
        R_dict = {n: grad_R_plain.get(n, zero_R(rank)) for n in svd_plain}
        model = inject_fist_lora(
            model, TARGET_MODULES, B_dict, A_dict, R_dict, ALPHA, rank
        )

    elif method == "fist_full":
        B_dict = {n: BA[0] for n, BA in svd_fisher.items()}
        A_dict = {n: BA[2] for n, BA in svd_fisher.items()}
        R_dict = {n: grad_R_fisher.get(n, zero_R(rank)) for n in svd_fisher}
        model = inject_fist_lora(
            model, TARGET_MODULES, B_dict, A_dict, R_dict, ALPHA, rank
        )

    elif method == "lora_sb":
        B_dict = {n: BA[0] for n, BA in svd_grad.items()}
        A_dict = {n: BA[2] for n, BA in svd_grad.items()}
        R_dict = {n: grad_R_grad.get(n, zero_R(rank)) for n in svd_grad}
        model = inject_fist_lora(
            model, TARGET_MODULES, B_dict, A_dict, R_dict, ALPHA, rank
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return model


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FiST-LoRA GLUE experiments")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="GLUE tasks to run (default: all)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to run (default: all)")
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 32],
                        help="Ranks to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="Random seeds")
    parser.add_argument("--output_dir", type=str, default="results/glue",
                        help="Output directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (overrides CLI args)")
    parser.add_argument("--report_to", type=str, default="none",
                        help="Reporting backend (none, wandb)")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Per-device train batch size (default: 32)")
    parser.add_argument("--eval_batch_size", type=int, default=128,
                        help="Per-device eval batch size (default: 128)")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use fp16 mixed precision (unstable on RoBERTa-large)")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bf16 mixed precision (default: True)")
    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable bf16, use fp32")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Dataloader num_workers (default: 4)")
    args = parser.parse_args()

    if args.no_bf16:
        args.bf16 = False
    if args.fp16:
        args.bf16 = False

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if args.tasks is None:
            args.tasks = list(cfg.get("tasks", TASKS).keys())
        if args.methods is None:
            args.methods = list(cfg.get("methods", {}).keys())
        args.seeds = cfg.get("seeds", args.seeds)
        args.output_dir = cfg.get("output_dir", args.output_dir)

    task_names = args.tasks or list(TASKS.keys())
    methods = args.methods or ALL_METHODS
    ranks = args.ranks
    seeds = args.seeds

    device = get_device()
    print(f"Device: {device}")
    print(f"Tasks: {task_names}")
    print(f"Methods: {methods}")
    print(f"Ranks: {ranks}")
    print(f"Seeds: {seeds}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    results = load_existing_results(results_path)

    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    for task_name in task_names:
        if task_name not in TASKS:
            print(f"[WARNING] Unknown task: {task_name}, skipping.")
            continue

        task_cfg = TASKS[task_name]

        print(f"\n{'#' * 70}")
        print(f"# TASK: {task_name.upper()}")
        print(f"{'#' * 70}")

        if task_name not in results:
            results[task_name] = {}

        # Load & tokenize dataset
        print(f"Loading dataset: glue/{task_name}")
        raw_dataset = load_dataset("glue", task_name)
        train_tok, val_tok = tokenize_glue(
            raw_dataset, tokenizer, task_cfg["text_keys"],
            max_length=task_cfg["max_seq_len"],
        )
        print(f"Train size: {len(train_tok):,}  Val size: {len(val_tok):,}")

        calibration_loader = make_calibration_loader(
            train_tok, tokenizer, CALIBRATION_SAMPLES
        )

        # Pre-compute Fisher + SVD + gradient R (shared across ranks and seeds)
        print("\nLoading base model for calibration...")
        base_model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=task_cfg["num_labels"]
        ).to(device)
        base_model.eval()

        # Warmup head for meaningful calibration
        print("Warming up classifier head for calibration...")
        base_model = warmup_classifier_head(
            base_model, train_tok, tokenizer,
            num_steps=100, batch_size=32, lr=1e-3,
        )

        # Fisher diagonal
        print("Computing diagonal Fisher...")
        fisher_diags = safe_run(
            lambda: compute_diagonal_fisher(
                base_model, calibration_loader, TARGET_MODULES, CALIBRATION_SAMPLES
            ),
            fallback={},
            description="Fisher computation",
        )

        # Collect SVD and gradient R for all ranks
        svd_plain_by_rank = {}
        svd_fisher_by_rank = {}
        svd_grad_by_rank = {}
        grad_R_plain_by_rank = {}
        grad_R_fisher_by_rank = {}
        grad_R_grad_by_rank = {}

        for rank in ranks:
            svd_plain_by_rank[rank] = collect_plain_svd(
                base_model, TARGET_MODULES, rank
            )

            if fisher_diags:
                svd_fisher_by_rank[rank] = collect_fisher_svd(
                    base_model, fisher_diags, TARGET_MODULES, rank
                )
            else:
                svd_fisher_by_rank[rank] = svd_plain_by_rank[rank]

            # Gradient-projected R with plain SVD basis
            plain_ba = {
                name: (B, A)
                for name, (B, S, A) in svd_plain_by_rank[rank].items()
            }
            grad_R_plain_by_rank[rank] = safe_run(
                lambda ba=plain_ba, r=rank: gradient_projected_R(
                    base_model, calibration_loader, ba,
                    TARGET_MODULES, CALIBRATION_SAMPLES, ALPHA, r,
                    init_scale=INIT_SCALE,
                ),
                fallback={name: zero_R(rank) for name in plain_ba},
                description=f"gradient_projected_R plain rank={rank}",
            )

            # Gradient-projected R with Fisher SVD basis
            fisher_ba = {
                name: (B, A)
                for name, (B, S, A) in svd_fisher_by_rank[rank].items()
            }
            grad_R_fisher_by_rank[rank] = safe_run(
                lambda ba=fisher_ba, r=rank: gradient_projected_R(
                    base_model, calibration_loader, ba,
                    TARGET_MODULES, CALIBRATION_SAMPLES, ALPHA, r,
                    init_scale=INIT_SCALE,
                ),
                fallback={name: zero_R(rank) for name in fisher_ba},
                description=f"gradient_projected_R fisher rank={rank}",
            )

            # LoRA-SB: gradient SVD for outer matrices + gradient R
            print(f"Computing gradient SVD for LoRA-SB (rank={rank})...")
            svd_grad_by_rank[rank] = safe_run(
                lambda r=rank: compute_gradient_svd(
                    base_model, calibration_loader, TARGET_MODULES, r,
                    CALIBRATION_SAMPLES,
                ),
                fallback=svd_plain_by_rank[rank],
                description=f"gradient_svd rank={rank}",
            )
            grad_ba = {
                name: (B, A)
                for name, (B, S, A) in svd_grad_by_rank[rank].items()
            }
            grad_R_grad_by_rank[rank] = safe_run(
                lambda ba=grad_ba, r=rank: gradient_projected_R(
                    base_model, calibration_loader, ba,
                    TARGET_MODULES, CALIBRATION_SAMPLES, ALPHA, r,
                    init_scale=INIT_SCALE,
                ),
                fallback={name: zero_R(rank) for name in grad_ba},
                description=f"gradient_projected_R grad-svd rank={rank}",
            )

        # Free calibration model
        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Run experiments: rank x seed x method
        for rank in ranks:
            for seed in seeds:
                seed_key = f"rank_{rank}_seed_{seed}"
                if seed_key not in results[task_name]:
                    results[task_name][seed_key] = {}

                for method in methods:
                    existing = results[task_name].get(seed_key, {}).get(method, {})
                    if is_valid_result(existing):
                        print(f"\n  [Resume] Skipping {task_name}/{seed_key}/{method}")
                        continue

                    print(f"\n{'=' * 60}")
                    print(f"  Task: {task_name} | Rank: {rank} | Seed: {seed} | Method: {method}")
                    print(f"{'=' * 60}")

                    set_seed(seed)
                    start_time = time.time()

                    # Load FRESH model (no warmed head state)
                    model = RobertaForSequenceClassification.from_pretrained(
                        MODEL_NAME, num_labels=task_cfg["num_labels"]
                    )

                    try:
                        model = apply_adapter(
                            model, method, rank,
                            svd_plain_by_rank[rank],
                            svd_fisher_by_rank[rank],
                            grad_R_plain_by_rank[rank],
                            grad_R_fisher_by_rank[rank],
                            svd_grad=svd_grad_by_rank.get(rank),
                            grad_R_grad=grad_R_grad_by_rank.get(rank),
                        )
                    except Exception:
                        print(f"[ERROR] Adapter construction failed for {method}:")
                        traceback.print_exc()
                        results[task_name][seed_key][method] = {
                            "error": "adapter_construction_failed"
                        }
                        with open(results_path, "w") as f:
                            json.dump(results, f, indent=2, default=str)
                        continue

                    trainable, total = print_trainable_summary(model, method)

                    lr = METHOD_LRS.get(method, 1e-3)
                    train_bs = args.train_batch_size

                    training_args = TrainingArguments(
                        output_dir=os.path.join(
                            args.output_dir, task_name, f"{method}_r{rank}_s{seed}"
                        ),
                        num_train_epochs=task_cfg["epochs"],
                        per_device_train_batch_size=train_bs,
                        per_device_eval_batch_size=args.eval_batch_size,
                        learning_rate=lr,
                        weight_decay=0.0,
                        warmup_ratio=WARMUP_RATIO,
                        lr_scheduler_type="linear",
                        eval_strategy="epoch",
                        save_strategy="no",
                        logging_steps=50,
                        seed=seed,
                        fp16=args.fp16,
                        bf16=args.bf16,
                        report_to=args.report_to,
                        dataloader_num_workers=args.num_workers,
                        tf32=True,
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_tok,
                        eval_dataset=val_tok,
                        processing_class=tokenizer,
                        data_collator=DataCollatorWithPadding(tokenizer),
                        compute_metrics=build_compute_metrics(
                            task_cfg["metric"], task_name
                        ),
                    )

                    try:
                        trainer.train()
                        eval_results = trainer.evaluate()
                    except Exception:
                        print(f"[ERROR] Training/evaluation failed for {method}:")
                        traceback.print_exc()
                        results[task_name][seed_key][method] = {
                            "error": "training_failed",
                            "trainable_params": trainable,
                            "time_seconds": time.time() - start_time,
                        }
                        with open(results_path, "w") as f:
                            json.dump(results, f, indent=2, default=str)
                        del model, trainer
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue

                    elapsed = time.time() - start_time

                    results[task_name][seed_key][method] = {
                        "metric": eval_results,
                        "trainable_params": trainable,
                        "time_seconds": elapsed,
                        "seed": seed,
                        "rank": rank,
                    }

                    # Print key metric
                    metric_key = f"eval_{task_cfg['metric']}"
                    score = eval_results.get(metric_key, "?")
                    print(f"  Result: {metric_key} = {score}")
                    print(f"  Time:   {elapsed:.1f}s")

                    # Save immediately after each result
                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=2, default=str)
                    print(f"  [Saved] {task_name}/{seed_key}/{method} -> {results_path}")

                    del model, trainer
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print_summary(results, methods, ranks, seeds)


def print_summary(results, methods, ranks, seeds):
    """Print aggregated results table."""
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY (mean +/- std over seeds)")
    print("=" * 100)

    metric_keys = {
        "sst2": "eval_accuracy",
        "mrpc": "eval_f1",
        "qnli": "eval_accuracy",
        "rte": "eval_accuracy",
        "mnli": "eval_accuracy",
        "qqp": "eval_accuracy",
        "cola": "eval_matthews_correlation",
        "stsb": "eval_spearmanr",
    }

    for rank in ranks:
        print(f"\n--- Rank = {rank} ---")
        header = f"{'Method':22s}"
        for task in results:
            header += f" | {task:>8s}"
        header += f" | {'Avg':>8s} | {'Params':>10s}"
        print(header)
        print("-" * len(header))

        for method in methods:
            row = f"{method:22s}"
            all_means = []
            params_str = "?"

            for task_name in results:
                scores = []
                for seed in seeds:
                    seed_key = f"rank_{rank}_seed_{seed}"
                    r = results.get(task_name, {}).get(seed_key, {}).get(method, {})
                    if "metric" in r and "error" not in r:
                        mk = metric_keys.get(task_name, "eval_accuracy")
                        val = r["metric"].get(mk)
                        if val is not None:
                            scores.append(val)
                        if params_str == "?":
                            params_str = f"{r.get('trainable_params', '?'):>10}"

                if scores:
                    mean = np.mean(scores)
                    std = np.std(scores)
                    all_means.append(mean)
                    row += f" | {mean:7.4f}±{std:.2f}"
                else:
                    row += f" | {'N/A':>8s}"

            if all_means:
                row += f" | {np.mean(all_means):7.4f}"
            else:
                row += f" | {'N/A':>8s}"
            row += f" | {params_str}"
            print(row)


if __name__ == "__main__":
    main()
