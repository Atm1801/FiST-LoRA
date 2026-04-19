"""
Baseline: LoRA-SB (gradient-based outer matrix selection).

LoRA-SB uses the same B*R*A architecture as LoRA-XS, but initializes B and A
from the gradient SVD (not weight SVD) and initializes R to approximate the
first full fine-tuning gradient step.

Reference: https://arxiv.org/abs/2411.19557
Code: https://github.com/CERT-Lab/lora-sb

This script provides a wrapper that:
1. Uses our FiST-LoRA infrastructure for the B*R*A layer
2. Initializes B/A from gradient SVD
3. Initializes R via their method

Usage:
    python baselines/run_lora_sb.py --task sst2 --rank 8
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import evaluate

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fist_lora.init import zero_R
from fist_lora.model import inject_fist_lora
from fist_lora.utils import (
    set_seed,
    warmup_classifier_head,
    make_calibration_loader,
    compute_warmup_steps,
    get_device,
    print_trainable_summary,
)


GLUE_TASKS = {
    "sst2": {"num_labels": 2, "metric": "accuracy", "epochs": 3, "text_keys": ("sentence", None)},
    "mrpc": {"num_labels": 2, "metric": "f1", "epochs": 5, "text_keys": ("sentence1", "sentence2")},
    "qnli": {"num_labels": 2, "metric": "accuracy", "epochs": 3, "text_keys": ("question", "sentence")},
    "rte":  {"num_labels": 2, "metric": "accuracy", "epochs": 5, "text_keys": ("sentence1", "sentence2")},
}


def compute_gradient_svd(model, dataloader, target_module_names, rank, num_samples=256):
    """
    Compute SVD of the average gradient for each target module.

    This is the key difference from LoRA-XS: B and A come from the gradient
    SVD rather than the weight SVD, selecting directions that the loss is
    most sensitive to.

    Returns:
        Dict mapping module name to (B, S, A) tuples.
    """
    device = next(model.parameters()).device
    model.eval()

    target_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in target_module_names):
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


def compute_lora_sb_R(model, dataloader, svd_results, target_module_names, rank, num_samples=256, init_scale=0.01):
    """
    LoRA-SB R initialization: project gradient into gradient-SVD subspace.

    R_init = B_grad^T @ G @ A_grad^T, normalized to init_scale.
    """
    device = next(model.parameters()).device
    model.eval()

    target_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in target_module_names):
            if name in svd_results:
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

    R_dict = {}
    for name, (B, S, A) in svd_results.items():
        G = grad_accum[name] / max(samples_seen, 1)
        R_init = B.to(device).T @ G.to(device) @ A.to(device).T
        norm = R_init.norm()
        if norm > 1e-12:
            R_init = R_init * (init_scale / norm)
        else:
            R_init = zero_R(rank)
        R_dict[name] = R_init.cpu()

    return R_dict


def tokenize_glue(raw_dataset, tokenizer, text_keys, max_length=128):
    key1, key2 = text_keys
    def tokenize_fn(examples):
        if key2 is None:
            enc = tokenizer(examples[key1], truncation=True, max_length=max_length)
        else:
            enc = tokenizer(examples[key1], examples[key2], truncation=True, max_length=max_length)
        enc["labels"] = examples["label"]
        return enc
    cols = list(raw_dataset["train"].column_names)
    train = raw_dataset["train"].map(tokenize_fn, batched=True, remove_columns=cols)
    val_split = "validation" if "validation" in raw_dataset else "validation_matched"
    val = raw_dataset[val_split].map(tokenize_fn, batched=True, remove_columns=cols)
    train.set_format("torch")
    val.set_format("torch")
    return train, val


def build_compute_metrics(metric_name, task_name):
    metric = evaluate.load("glue", task_name)
    if metric_name == "f1":
        def fn(ep):
            return {"f1": metric.compute(predictions=ep[0].argmax(-1), references=ep[1])["f1"]}
        return fn
    def fn(ep):
        return {"accuracy": metric.compute(predictions=ep[0].argmax(-1), references=ep[1])["accuracy"]}
    return fn


def main():
    parser = argparse.ArgumentParser(description="LoRA-SB baseline")
    parser.add_argument("--task", type=str, required=True, choices=list(GLUE_TASKS.keys()))
    parser.add_argument("--model", type=str, default="roberta-large")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_scale", type=float, default=0.01)
    parser.add_argument("--target_modules", nargs="+", default=["query", "key", "value"])
    parser.add_argument("--output_dir", type=str, default="results/baselines/lora_sb")
    args = parser.parse_args()

    task_cfg = GLUE_TASKS[args.task]
    set_seed(args.seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    raw_dataset = load_dataset("glue", args.task)
    train_tok, val_tok = tokenize_glue(raw_dataset, tokenizer, task_cfg["text_keys"])

    # Calibration model for gradient SVD
    cal_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=task_cfg["num_labels"]
    ).to(device)
    cal_model = warmup_classifier_head(cal_model, train_tok, tokenizer)
    cal_loader = make_calibration_loader(train_tok, tokenizer, 256)

    # Gradient SVD for outer matrices
    grad_svd = compute_gradient_svd(
        cal_model, cal_loader, args.target_modules, args.rank
    )

    # LoRA-SB R init
    R_dict = compute_lora_sb_R(
        cal_model, cal_loader, grad_svd, args.target_modules,
        args.rank, init_scale=args.init_scale,
    )

    del cal_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Fresh model for training
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=task_cfg["num_labels"]
    )

    B_dict = {n: BA[0] for n, BA in grad_svd.items()}
    A_dict = {n: BA[2] for n, BA in grad_svd.items()}

    model = inject_fist_lora(
        model, args.target_modules, B_dict, A_dict, R_dict, args.alpha, args.rank
    )
    print_trainable_summary(model, "LoRA-SB")

    warmup_steps = compute_warmup_steps(len(train_tok), 32, task_cfg["epochs"])

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.task}_r{args.rank}_s{args.seed}"),
        num_train_epochs=task_cfg["epochs"],
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=args.seed,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=build_compute_metrics(task_cfg["metric"], args.task),
    )

    trainer.train()
    eval_results = trainer.evaluate()

    metric_key = f"eval_{task_cfg['metric']}"
    print(f"\nResult: {metric_key} = {eval_results.get(metric_key, '?')}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.task}_r{args.rank}_s{args.seed}.json"), "w") as f:
        json.dump({"eval": eval_results, "config": vars(args)}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
