"""
Subspace distance analysis.

For each target layer:
1. Compute 4 subspaces: plain SVD, Fisher-SVD, activation-SVD, gradient-SVD
2. Compute pairwise chordal distances between subspaces
3. Visualize as heatmap per layer

Key question: how different IS the Fisher-SVD subspace from plain SVD?

Usage:
    python analysis/subspace_distance.py --model roberta-large --task sst2 --rank 8
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


def chordal_distance(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """
    Compute chordal distance between two subspaces.

    d_chordal(U1, U2) = sqrt(r - ||U1^T U2||_F^2) / sqrt(r)

    where U1 and U2 are orthonormal bases (d, r).

    Returns value in [0, 1]: 0 = identical, 1 = orthogonal.
    """
    # Ensure orthonormal
    Q1, _ = torch.linalg.qr(U1.float())
    Q2, _ = torch.linalg.qr(U2.float())

    r = Q1.shape[1]
    inner = Q1.T @ Q2  # (r, r)
    frob_sq = (inner ** 2).sum().item()

    dist = np.sqrt(max(0, r - frob_sq)) / np.sqrt(r)
    return dist


def compute_gradient_subspace(model, dataloader, name, module, rank, device, num_samples=256):
    """Compute gradient SVD subspace for a single layer."""
    module.weight.requires_grad_(True)
    grad_accum = torch.zeros_like(module.weight)

    samples_seen = 0
    for batch in dataloader:
        if samples_seen >= num_samples:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss if outputs.loss is not None else outputs.logits.sum()
        loss.backward()
        if module.weight.grad is not None:
            grad_accum += module.weight.grad.detach()
        model.zero_grad()
        samples_seen += batch[next(iter(batch))].size(0)

    module.weight.requires_grad_(False)

    G = grad_accum / max(samples_seen, 1)
    U, _, _ = torch.linalg.svd(G.float(), full_matrices=False)
    return U[:, :rank].clone()


def compute_activation_subspace(model, dataloader, name, module, rank, device, num_samples=256):
    """Compute activation covariance SVD subspace for a single layer."""
    act_accum = torch.zeros(module.in_features, module.in_features, device=device)
    hook_handle = None
    activations = []

    def hook_fn(mod, inp, out):
        x = inp[0]  # (batch, seq, d)
        x_flat = x.reshape(-1, x.shape[-1]).float()
        activations.append(x_flat.detach())

    hook_handle = module.register_forward_hook(hook_fn)

    samples_seen = 0
    for batch in dataloader:
        if samples_seen >= num_samples:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            model(**batch)
        samples_seen += batch[next(iter(batch))].size(0)

    hook_handle.remove()

    if activations:
        all_acts = torch.cat(activations, dim=0)  # (total_tokens, in_features)
        # Covariance SVD
        cov = (all_acts.T @ all_acts) / all_acts.shape[0]
        U, _, _ = torch.linalg.svd(cov, full_matrices=False)
        return U[:, :rank].clone()

    return torch.eye(module.in_features, rank, device=device)


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

    task_configs = {
        "sst2": {"num_labels": 2, "text_keys": ("sentence", None)},
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

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=tcfg["num_labels"]
    ).to(device)
    model = warmup_classifier_head(model, train_tok, tokenizer)
    cal_loader = make_calibration_loader(train_tok, tokenizer, 256)

    fisher_diags = compute_diagonal_fisher(model, cal_loader, args.target_modules, 256)

    # Compute subspaces for each layer
    subspace_names = ["Plain SVD", "Fisher SVD", "Gradient SVD", "Activation SVD"]
    all_results = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in args.target_modules):
            continue

        W = module.weight.data.float()

        # Plain SVD
        U_plain, _, _ = plain_svd(W, args.rank)

        # Fisher SVD
        if name in fisher_diags:
            U_fisher, _, _ = fisher_weighted_svd(W, fisher_diags[name].float(), args.rank)
        else:
            U_fisher = U_plain

        # Gradient SVD
        U_grad = compute_gradient_subspace(
            model, cal_loader, name, module, args.rank, device
        )

        # Activation SVD (for the column space — need Vt not U)
        # Use U for left subspace comparison
        subspaces = [U_plain, U_fisher, U_grad]

        # Compute pairwise distances
        n = len(subspaces)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = chordal_distance(subspaces[i].to(device), subspaces[j].to(device))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        all_results.append({
            "name": name,
            "distances": dist_matrix.tolist(),
            "subspace_names": subspace_names[:n],
        })

        print(f"\n{name}:")
        print(f"  Plain-Fisher distance:   {dist_matrix[0, 1]:.4f}")
        print(f"  Plain-Gradient distance: {dist_matrix[0, 2]:.4f}")
        print(f"  Fisher-Gradient distance:{dist_matrix[1, 2]:.4f}")

    # Plot heatmaps
    n_layers = len(all_results)
    if n_layers > 0:
        fig, axes = plt.subplots(
            1, min(n_layers, 6), figsize=(4 * min(n_layers, 6), 4)
        )
        if n_layers == 1:
            axes = [axes]

        for idx, (ax, result) in enumerate(zip(axes, all_results[:6])):
            dm = np.array(result["distances"])
            im = ax.imshow(dm, cmap="YlOrRd", vmin=0, vmax=1)
            ax.set_xticks(range(len(result["subspace_names"])))
            ax.set_yticks(range(len(result["subspace_names"])))
            ax.set_xticklabels(["Plain", "Fisher", "Grad"], fontsize=8, rotation=45)
            ax.set_yticklabels(["Plain", "Fisher", "Grad"], fontsize=8)
            short_name = result["name"].split(".")[-1]
            layer_num = result["name"].split(".")
            layer_id = [p for p in layer_num if p.isdigit()]
            title = f"L{layer_id[0]}.{short_name}" if layer_id else short_name
            ax.set_title(title, fontsize=9)

            for i in range(dm.shape[0]):
                for j in range(dm.shape[1]):
                    ax.text(j, i, f"{dm[i, j]:.2f}", ha="center", va="center", fontsize=7)

        plt.suptitle(f"Subspace Chordal Distances (rank={args.rank})", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, f"subspace_distance_{args.task}_r{args.rank}.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close()

    with open(os.path.join(args.output_dir, f"subspace_distance_{args.task}_r{args.rank}.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
