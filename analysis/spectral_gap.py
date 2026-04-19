"""
Spectral gap analysis: compare Fisher-weighted SVD vs plain SVD.

For each target layer:
1. Compute Fisher diagonal
2. Compute Fisher-weighted SVD and plain SVD
3. Record spectral gap ratio: sigma_r / sigma_{r+1} for both
4. Plot: x=layer index, y=spectral gap ratio, two lines (Fisher vs plain)

Hypothesis: Fisher-SVD should have LARGER spectral gaps (cleaner subspace
separation), and layers with larger gaps should benefit more from Fisher weighting.

Usage:
    python analysis/spectral_gap.py --model roberta-large --task sst2 --rank 8
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from datasets import load_dataset

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fist_lora.fisher import compute_diagonal_fisher
from fist_lora.init import fisher_weighted_svd, plain_svd
from fist_lora.utils import (
    set_seed,
    warmup_classifier_head,
    make_calibration_loader,
    get_device,
)


def compute_spectral_gaps(model, fisher_diags, target_module_names, rank):
    """
    Compute spectral gap ratios for both Fisher-weighted and plain SVD.

    Returns list of dicts with layer info and gap ratios.
    """
    results = []
    layer_idx = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(
            name.endswith(t) for t in target_module_names
        ):
            W = module.weight.data.float()

            # Plain SVD
            _, S_plain, _ = plain_svd(W, rank + 1)
            gap_plain = (S_plain[rank - 1] / (S_plain[rank] + 1e-12)).item()

            # Fisher-weighted SVD
            if name in fisher_diags:
                _, S_fisher, _ = fisher_weighted_svd(W, fisher_diags[name].float(), rank + 1)
                gap_fisher = (S_fisher[rank - 1] / (S_fisher[rank] + 1e-12)).item()
            else:
                gap_fisher = gap_plain

            results.append({
                "layer_idx": layer_idx,
                "name": name,
                "gap_plain": gap_plain,
                "gap_fisher": gap_fisher,
                "sigma_r_plain": S_plain[rank - 1].item(),
                "sigma_r1_plain": S_plain[rank].item(),
                "sigma_r_fisher": S_fisher[rank - 1].item() if name in fisher_diags else S_plain[rank - 1].item(),
                "sigma_r1_fisher": S_fisher[rank].item() if name in fisher_diags else S_plain[rank].item(),
            })
            layer_idx += 1

    return results


def plot_spectral_gaps(results, rank, output_path):
    """Plot spectral gap comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    indices = [r["layer_idx"] for r in results]
    gaps_plain = [r["gap_plain"] for r in results]
    gaps_fisher = [r["gap_fisher"] for r in results]

    # Spectral gap ratio
    ax = axes[0]
    ax.plot(indices, gaps_plain, "o-", label="Plain SVD", alpha=0.8)
    ax.plot(indices, gaps_fisher, "s-", label="Fisher-weighted SVD", alpha=0.8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel(f"Spectral Gap Ratio (sigma_{rank}/sigma_{rank+1})")
    ax.set_title(f"Spectral Gap at Rank {rank}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Absolute singular values
    ax = axes[1]
    sigma_r_plain = [r["sigma_r_plain"] for r in results]
    sigma_r_fisher = [r["sigma_r_fisher"] for r in results]
    ax.plot(indices, sigma_r_plain, "o-", label=f"Plain sigma_{rank}", alpha=0.8)
    ax.plot(indices, sigma_r_fisher, "s-", label=f"Fisher sigma_{rank}", alpha=0.8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Singular Value")
    ax.set_title(f"Rank-{rank} Singular Value per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="roberta-large")
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--target_modules", nargs="+", default=["query", "key", "value"])
    parser.add_argument("--output_dir", type=str, default="results/analysis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Task config
    task_configs = {
        "sst2": {"num_labels": 2, "text_keys": ("sentence", None)},
        "mrpc": {"num_labels": 2, "text_keys": ("sentence1", "sentence2")},
    }
    tcfg = task_configs.get(args.task, {"num_labels": 2, "text_keys": ("sentence", None)})

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
    train_tok.set_format("torch")

    # Load model and warmup
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=tcfg["num_labels"]
    ).to(device)
    model = warmup_classifier_head(model, train_tok, tokenizer)

    calibration_loader = make_calibration_loader(train_tok, tokenizer, 256)

    print("Computing Fisher...")
    fisher_diags = compute_diagonal_fisher(
        model, calibration_loader, args.target_modules, 256
    )

    print("Computing spectral gaps...")
    results = compute_spectral_gaps(model, fisher_diags, args.target_modules, args.rank)

    # Print table
    print(f"\n{'Layer':<55s} | {'Plain Gap':>10s} | {'Fisher Gap':>10s} | {'Improvement':>12s}")
    print("-" * 95)
    for r in results:
        improvement = r["gap_fisher"] / (r["gap_plain"] + 1e-12) - 1
        print(f"{r['name']:<55s} | {r['gap_plain']:10.4f} | {r['gap_fisher']:10.4f} | {improvement:+11.2%}")

    avg_plain = np.mean([r["gap_plain"] for r in results])
    avg_fisher = np.mean([r["gap_fisher"] for r in results])
    print(f"\n{'Average':<55s} | {avg_plain:10.4f} | {avg_fisher:10.4f} | {avg_fisher/avg_plain - 1:+11.2%}")

    # Save
    plot_spectral_gaps(results, args.rank, os.path.join(args.output_dir, f"spectral_gap_{args.task}_r{args.rank}.png"))

    with open(os.path.join(args.output_dir, f"spectral_gap_{args.task}_r{args.rank}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
