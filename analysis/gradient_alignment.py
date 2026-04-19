"""
Gradient alignment tracking during training.

During training, every N steps:
1. Compute full gradient G for each target layer
2. Project: G_proj = B^T @ G @ A^T, reconstruct: G_approx = B @ G_proj @ A
3. Compute cosine_sim(G.flatten(), G_approx.flatten())
4. Log per-layer and averaged across layers

Compare Fisher-SVD subspace vs plain SVD subspace alignment.

Usage:
    python analysis/gradient_alignment.py --model roberta-large --task sst2 --rank 8
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

from fist_lora.fisher import compute_diagonal_fisher
from fist_lora.init import zero_R
from fist_lora.model import (
    inject_fist_lora,
    collect_plain_svd,
    collect_fisher_svd,
)
from fist_lora.layers import FiSTLoRALinear
from fist_lora.utils import (
    set_seed,
    warmup_classifier_head,
    make_calibration_loader,
    compute_warmup_steps,
    get_device,
)


def compute_gradient_alignment(model, dataloader, device, max_batches=4):
    """
    Compute cosine similarity between full gradient and its projection
    into each FiSTLoRALinear layer's (B, A) subspace.

    Returns dict: {layer_name: cosine_similarity}.
    """
    model.eval()

    # Collect FiSTLoRA layers
    fist_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, FiSTLoRALinear):
            fist_layers[name] = module
            module.weight.requires_grad_(True)

    if not fist_layers:
        return {}

    grad_accum = {name: torch.zeros_like(m.weight) for name, m in fist_layers.items()}
    n_batches = 0

    for batch in dataloader:
        if n_batches >= max_batches:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss if outputs.loss is not None else outputs.logits.sum()
        loss.backward()

        for name, module in fist_layers.items():
            if module.weight.grad is not None:
                grad_accum[name] += module.weight.grad.detach()

        model.zero_grad()
        n_batches += 1

    alignments = {}
    for name, module in fist_layers.items():
        G = grad_accum[name] / max(n_batches, 1)  # (d, k)
        B = module.B  # (d, r)
        A = module.A  # (r, k)

        # Project into subspace and reconstruct
        G_proj = B.T @ G @ A.T       # (r, r)
        G_approx = B @ G_proj @ A    # (d, k)

        # Cosine similarity
        g_flat = G.flatten().float()
        ga_flat = G_approx.flatten().float()

        cos_sim = torch.nn.functional.cosine_similarity(
            g_flat.unsqueeze(0), ga_flat.unsqueeze(0)
        ).item()

        alignments[name] = cos_sim
        module.weight.requires_grad_(False)

    model.train()
    return alignments


class AlignmentCallback:
    """Callback that logs gradient alignment during training."""

    def __init__(self, model, dataloader, device, log_every=100):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.log_every = log_every
        self.history = []  # list of (step, {layer: cosine_sim})

    def on_step(self, step):
        if step % self.log_every != 0:
            return
        alignments = compute_gradient_alignment(
            self.model, self.dataloader, self.device
        )
        if alignments:
            avg = np.mean(list(alignments.values()))
            self.history.append({"step": step, "alignments": alignments, "avg": avg})
            print(f"  [Alignment] step={step}: avg cosine_sim={avg:.4f}")


def plot_alignment(history_plain, history_fisher, output_path, rank):
    """Plot gradient alignment over training for both subspace types."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if history_plain:
        steps_p = [h["step"] for h in history_plain]
        avgs_p = [h["avg"] for h in history_plain]
        ax.plot(steps_p, avgs_p, "o-", label="Plain SVD subspace", alpha=0.8)

    if history_fisher:
        steps_f = [h["step"] for h in history_fisher]
        avgs_f = [h["avg"] for h in history_fisher]
        ax.plot(steps_f, avgs_f, "s-", label="Fisher-weighted SVD subspace", alpha=0.8)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Alignment (cosine similarity)")
    ax.set_title(f"Gradient Alignment During Training (rank={rank})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="roberta-large")
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--target_modules", nargs="+", default=["query", "key", "value"])
    parser.add_argument("--output_dir", type=str, default="results/analysis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    task_configs = {
        "sst2": {"num_labels": 2, "epochs": 3, "text_keys": ("sentence", None), "metric": "accuracy"},
    }
    tcfg = task_configs.get(args.task, {"num_labels": 2, "epochs": 3, "text_keys": ("sentence", None), "metric": "accuracy"})

    raw_dataset = load_dataset("glue", args.task)
    key1, key2 = tcfg["text_keys"]
    def tokenize_fn(examples):
        if key2 is None:
            enc = tokenizer(examples[key1], truncation=True, max_length=128)
        else:
            enc = tokenizer(examples[key1], examples[key2], truncation=True, max_length=128)
        enc["labels"] = examples["label"]
        return enc
    cols = list(raw_dataset["train"].column_names)
    train_tok = raw_dataset["train"].map(tokenize_fn, batched=True, remove_columns=cols)
    val_tok = raw_dataset["validation"].map(tokenize_fn, batched=True, remove_columns=cols)
    train_tok.set_format("torch")
    val_tok.set_format("torch")

    # Calibration
    cal_model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=tcfg["num_labels"]
    ).to(device)
    cal_model = warmup_classifier_head(cal_model, train_tok, tokenizer)
    cal_loader = make_calibration_loader(train_tok, tokenizer, 256)

    fisher_diags = compute_diagonal_fisher(cal_model, cal_loader, args.target_modules, 256)
    svd_plain = collect_plain_svd(cal_model, args.target_modules, args.rank)
    svd_fisher = collect_fisher_svd(cal_model, fisher_diags, args.target_modules, args.rank)
    del cal_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Train with plain SVD subspace, tracking alignment
    print("\n--- Training with PLAIN SVD subspace ---")
    model_plain = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=tcfg["num_labels"]
    )
    B_dict = {n: BA[0] for n, BA in svd_plain.items()}
    A_dict = {n: BA[2] for n, BA in svd_plain.items()}
    R_dict = {n: zero_R(args.rank) for n in svd_plain}
    model_plain = inject_fist_lora(model_plain, args.target_modules, B_dict, A_dict, R_dict, 32.0, args.rank)
    model_plain = model_plain.to(device)

    # Quick training loop with alignment tracking
    history_plain = []
    optimizer = torch.optim.AdamW(
        [p for p in model_plain.parameters() if p.requires_grad], lr=1e-3
    )
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_tok, batch_size=32, shuffle=True, collate_fn=collator
    )

    step = 0
    model_plain.train()
    for epoch in range(min(tcfg["epochs"], 2)):
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model_plain(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % args.log_every == 0:
                alignment = compute_gradient_alignment(model_plain, cal_loader, device)
                if alignment:
                    avg = np.mean(list(alignment.values()))
                    history_plain.append({"step": step, "alignments": alignment, "avg": avg})
                    print(f"  [Plain] step={step}: avg alignment={avg:.4f}")

            if step >= 1000:
                break
        if step >= 1000:
            break

    del model_plain
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Train with Fisher SVD subspace
    print("\n--- Training with FISHER SVD subspace ---")
    model_fisher = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=tcfg["num_labels"]
    )
    B_dict = {n: BA[0] for n, BA in svd_fisher.items()}
    A_dict = {n: BA[2] for n, BA in svd_fisher.items()}
    R_dict = {n: zero_R(args.rank) for n in svd_fisher}
    model_fisher = inject_fist_lora(model_fisher, args.target_modules, B_dict, A_dict, R_dict, 32.0, args.rank)
    model_fisher = model_fisher.to(device)

    history_fisher = []
    optimizer = torch.optim.AdamW(
        [p for p in model_fisher.parameters() if p.requires_grad], lr=1e-3
    )
    train_loader = torch.utils.data.DataLoader(
        train_tok, batch_size=32, shuffle=True, collate_fn=collator
    )

    step = 0
    model_fisher.train()
    for epoch in range(min(tcfg["epochs"], 2)):
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model_fisher(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % args.log_every == 0:
                alignment = compute_gradient_alignment(model_fisher, cal_loader, device)
                if alignment:
                    avg = np.mean(list(alignment.values()))
                    history_fisher.append({"step": step, "alignments": alignment, "avg": avg})
                    print(f"  [Fisher] step={step}: avg alignment={avg:.4f}")

            if step >= 1000:
                break
        if step >= 1000:
            break

    # Plot and save
    plot_alignment(
        history_plain, history_fisher,
        os.path.join(args.output_dir, f"gradient_alignment_{args.task}_r{args.rank}.png"),
        args.rank,
    )

    with open(os.path.join(args.output_dir, f"gradient_alignment_{args.task}_r{args.rank}.json"), "w") as f:
        json.dump({
            "plain": [{"step": h["step"], "avg": h["avg"]} for h in history_plain],
            "fisher": [{"step": h["step"], "avg": h["avg"]} for h in history_fisher],
        }, f, indent=2)


if __name__ == "__main__":
    main()
