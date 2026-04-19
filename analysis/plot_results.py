"""
Generate paper figures from experiment results.

Produces:
1. Training loss curves comparison
2. Performance vs parameter count Pareto plot
3. Ablation comparison bar charts
4. Per-task breakdown tables

Usage:
    python analysis/plot_results.py --results_dir results/
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_results(results_dir):
    """Load all results JSON files from directory tree."""
    results = {}
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".json"):
                path = os.path.join(root, f)
                with open(path) as fh:
                    try:
                        results[path] = json.load(fh)
                    except json.JSONDecodeError:
                        pass
    return results


def plot_pareto(results_data, output_path):
    """
    Performance vs parameter count Pareto plot.

    Shows each method as a point: x = trainable params, y = average score.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    method_data = {}  # {method: [(params, score), ...]}

    for path, data in results_data.items():
        if "glue" not in path:
            continue
        for task_name, task_results in data.items():
            for run_key, run_data in task_results.items():
                if not isinstance(run_data, dict):
                    continue
                for method, result in run_data.items():
                    if not isinstance(result, dict) or "metric" not in result:
                        continue
                    params = result.get("trainable_params", 0)
                    # Get the primary metric
                    metric_val = None
                    for mk in ["eval_accuracy", "eval_f1", "eval_matthews_correlation"]:
                        if mk in result["metric"]:
                            metric_val = result["metric"][mk]
                            break
                    if metric_val is not None and params > 0:
                        if method not in method_data:
                            method_data[method] = []
                        method_data[method].append((params, metric_val))

    colors = {
        "lora": "#1f77b4",
        "pissa": "#ff7f0e",
        "lora_xs": "#2ca02c",
        "fist_no_fisher": "#d62728",
        "fist_full": "#9467bd",
        "lora_sb": "#8c564b",
    }
    markers = {"lora": "o", "pissa": "D", "lora_xs": "s",
               "fist_no_fisher": "^", "fist_full": "*", "lora_sb": "v"}

    for method, points in method_data.items():
        if not points:
            continue
        params_arr = np.array([p[0] for p in points])
        scores_arr = np.array([p[1] for p in points])

        avg_params = np.mean(params_arr)
        avg_score = np.mean(scores_arr)
        std_score = np.std(scores_arr)

        ax.errorbar(
            avg_params, avg_score, yerr=std_score,
            fmt=markers.get(method, "o"),
            color=colors.get(method, "#333333"),
            markersize=10,
            label=method,
            capsize=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters", fontsize=12)
    ax.set_ylabel("Average Score", fontsize=12)
    ax.set_title("Performance vs Parameter Efficiency", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Pareto plot to {output_path}")


def plot_ablation_bars(results_data, output_path):
    """Bar chart comparing ablation variants across tasks."""
    fig, ax = plt.subplots(figsize=(12, 6))

    variants = ["lora_xs", "no_fisher", "no_grad", "sigma_init",
                 "scale_sweep_001", "scale_sweep_01", "scale_sweep_1", "fist_full"]
    variant_labels = ["LoRA-XS", "No Fisher", "No Grad", "Sigma Init",
                      "Scale 0.001", "Scale 0.01", "Scale 0.1", "FiST-LoRA"]

    task_scores = {}
    for path, data in results_data.items():
        if "ablation" not in path:
            continue
        for task_name, task_results in data.items():
            if task_name not in task_scores:
                task_scores[task_name] = {}
            for run_key, run_data in task_results.items():
                if not isinstance(run_data, dict):
                    continue
                for variant, result in run_data.items():
                    if not isinstance(result, dict) or "metric" not in result:
                        continue
                    for mk in ["eval_accuracy", "eval_f1"]:
                        if mk in result["metric"]:
                            if variant not in task_scores[task_name]:
                                task_scores[task_name][variant] = []
                            task_scores[task_name][variant].append(result["metric"][mk])

    if not task_scores:
        print("No ablation results found.")
        return

    tasks = sorted(task_scores.keys())
    x = np.arange(len(tasks))
    width = 0.8 / len(variants)

    colors_list = plt.cm.Set2(np.linspace(0, 1, len(variants)))

    for i, (variant, label) in enumerate(zip(variants, variant_labels)):
        means = []
        for task in tasks:
            scores = task_scores.get(task, {}).get(variant, [])
            means.append(np.mean(scores) if scores else 0)
        ax.bar(x + i * width, means, width, label=label, color=colors_list[i])

    ax.set_xlabel("Task")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study Results")
    ax.set_xticks(x + width * len(variants) / 2)
    ax.set_xticklabels([t.upper() for t in tasks])
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ablation plot to {output_path}")


def generate_latex_table(results_data, output_path):
    """Generate LaTeX table for main GLUE results."""
    methods = ["lora", "pissa", "lora_xs", "lora_sb", "fist_full"]
    method_labels = {
        "lora": "LoRA",
        "pissa": "PiSSA",
        "lora_xs": "LoRA-XS",
        "lora_sb": "LoRA-SB",
        "fist_full": "FiST-LoRA",
    }
    tasks = ["sst2", "mrpc", "qnli", "rte", "mnli", "qqp", "cola", "stsb"]
    metric_keys = {
        "sst2": "eval_accuracy", "mrpc": "eval_f1", "qnli": "eval_accuracy",
        "rte": "eval_accuracy", "mnli": "eval_accuracy", "qqp": "eval_accuracy",
        "cola": "eval_matthews_correlation", "stsb": "eval_spearmanr",
    }

    # Collect scores
    scores = {}
    for path, data in results_data.items():
        if "glue" not in path:
            continue
        for task_name, task_data in data.items():
            if task_name not in tasks:
                continue
            for run_key, run_data in task_data.items():
                if not isinstance(run_data, dict):
                    continue
                for method, result in run_data.items():
                    if not isinstance(result, dict) or "metric" not in result:
                        continue
                    mk = metric_keys.get(task_name)
                    if mk and mk in result["metric"]:
                        key = (method, task_name)
                        if key not in scores:
                            scores[key] = []
                        scores[key].append(result["metric"][mk])

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{GLUE Results (RoBERTa-large)}")
    cols = "l" + "c" * len(tasks) + "c"
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")

    header = "Method"
    for t in tasks:
        header += f" & {t.upper()}"
    header += r" & Avg \\"
    lines.append(header)
    lines.append(r"\midrule")

    for method in methods:
        label = method_labels.get(method, method)
        row = label
        task_means = []
        for task in tasks:
            key = (method, task)
            if key in scores and scores[key]:
                mean = np.mean(scores[key])
                std = np.std(scores[key])
                task_means.append(mean)
                row += f" & {mean:.1f}$\\pm${std:.1f}"
            else:
                row += " & -"
        if task_means:
            row += f" & {np.mean(task_means):.1f}"
        else:
            row += " & -"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved LaTeX table to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    results_data = load_results(args.results_dir)
    print(f"Found {len(results_data)} result files.")

    plot_pareto(results_data, os.path.join(args.output_dir, "pareto.png"))
    plot_ablation_bars(results_data, os.path.join(args.output_dir, "ablations.png"))
    generate_latex_table(results_data, os.path.join(args.output_dir, "glue_table.tex"))


if __name__ == "__main__":
    main()
