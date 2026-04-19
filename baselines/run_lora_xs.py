"""
Baseline: LoRA-XS (plain SVD outer, zero R inner).

This uses our FiST-LoRA infrastructure with plain SVD and zero R init,
which is equivalent to the LoRA-XS method.

Usage:
    python baselines/run_lora_xs.py --task sst2 --rank 8
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
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
from fist_lora.model import inject_fist_lora, collect_plain_svd
from fist_lora.utils import set_seed, compute_warmup_steps, get_device, print_trainable_summary


GLUE_TASKS = {
    "sst2": {"num_labels": 2, "metric": "accuracy", "epochs": 3, "text_keys": ("sentence", None)},
    "mrpc": {"num_labels": 2, "metric": "f1", "epochs": 5, "text_keys": ("sentence1", "sentence2")},
    "qnli": {"num_labels": 2, "metric": "accuracy", "epochs": 3, "text_keys": ("question", "sentence")},
    "rte":  {"num_labels": 2, "metric": "accuracy", "epochs": 5, "text_keys": ("sentence1", "sentence2")},
    "mnli": {"num_labels": 3, "metric": "accuracy", "epochs": 3, "text_keys": ("premise", "hypothesis")},
    "qqp":  {"num_labels": 2, "metric": "accuracy", "epochs": 3, "text_keys": ("question1", "question2")},
    "cola": {"num_labels": 2, "metric": "matthews_correlation", "epochs": 5, "text_keys": ("sentence", None)},
    "stsb": {"num_labels": 1, "metric": "spearmanr", "epochs": 5, "text_keys": ("sentence1", "sentence2")},
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
        def fn(ep):
            return {"f1": metric.compute(predictions=ep[0].argmax(-1), references=ep[1])["f1"]}
        return fn
    def fn(ep):
        return {"accuracy": metric.compute(predictions=ep[0].argmax(-1), references=ep[1])["accuracy"]}
    return fn


def main():
    parser = argparse.ArgumentParser(description="LoRA-XS baseline")
    parser.add_argument("--task", type=str, required=True, choices=list(GLUE_TASKS.keys()))
    parser.add_argument("--model", type=str, default="roberta-large")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_modules", nargs="+", default=["query", "key", "value"])
    parser.add_argument("--output_dir", type=str, default="results/baselines/lora_xs")
    args = parser.parse_args()

    task_cfg = GLUE_TASKS[args.task]
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    raw_dataset = load_dataset("glue", args.task)
    train_tok, val_tok = tokenize_glue(raw_dataset, tokenizer, task_cfg["text_keys"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=task_cfg["num_labels"]
    )

    # Plain SVD + zero R = LoRA-XS
    svd_results = collect_plain_svd(model, args.target_modules, args.rank)
    B_dict = {n: BA[0] for n, BA in svd_results.items()}
    A_dict = {n: BA[2] for n, BA in svd_results.items()}
    R_dict = {n: zero_R(args.rank) for n in svd_results}

    model = inject_fist_lora(
        model, args.target_modules, B_dict, A_dict, R_dict, args.alpha, args.rank
    )
    print_trainable_summary(model, "LoRA-XS")

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
