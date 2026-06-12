"""Training dynamics of the r^2-budget methods on GLUE (default r = 8).

(a) validation metric and (b) training loss against training progress (0-100 %),
averaged over seeds and over the six tasks.  Each run's curve is linearly interpolated
onto a common progress grid before averaging, because tasks differ in epochs and steps.
Progress of an evaluation = epoch / num_epochs; of a logged loss = step / max_step.
Curves are not extrapolated: a task contributes only inside its observed progress range.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

GRID = np.linspace(0, 100, 101)


def _read_history(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_curves(run_dir: Path, metric: str, num_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    hist = _read_history(run_dir / "log_history.jsonl")
    losses = [(h["step"], h["loss"]) for h in hist if "loss" in h and "step" in h]
    max_step = max(h["step"] for h in hist if "step" in h)
    evals = [(h["epoch"], 100 * h[f"eval_{metric}"]) for h in hist if f"eval_{metric}" in h]
    if not losses or not evals:
        raise ValueError(f"{run_dir}: missing loss or eval entries")
    lx = np.array([100 * s / max_step for s, _ in losses])
    ly = np.array([v for _, v in losses])
    ex = np.array([100 * e / num_epochs for e, _ in evals])
    ey = np.array([v for _, v in evals])
    # No extrapolation: outside a run's observed range the curve is undefined (NaN), so the
    # task average only starts once every task has been evaluated at least once.
    return (np.interp(GRID, ex, ey, left=np.nan, right=np.nan),
            np.interp(GRID, lx, ly, left=np.nan, right=np.nan))


def average_curves(
    results_dir: Path, tasks: dict[str, tuple[str, int]], methods: list[str], rank: int
) -> dict[str, dict[str, np.ndarray]]:
    """{method: {"eval": mean curve, "loss": mean curve}} over tasks (and seeds within task)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        per_task_eval, per_task_loss = [], []
        for task, (metric, epochs) in tasks.items():
            seed_curves = defaultdict(list)
            for run in sorted((results_dir / "runs" / task / method / f"r{rank}").glob("seed*")):
                if not (run / "metrics.json").exists():
                    continue
                e, lo = run_curves(run, metric, epochs)
                seed_curves["eval"].append(e)
                seed_curves["loss"].append(lo)
            if seed_curves:
                per_task_eval.append(np.mean(seed_curves["eval"], axis=0))
                per_task_loss.append(np.mean(seed_curves["loss"], axis=0))
        if len(per_task_eval) == len(tasks):
            out[method] = {"eval": np.mean(per_task_eval, axis=0), "loss": np.mean(per_task_loss, axis=0)}
    return out


def plot_training_dynamics(curves: dict[str, dict[str, np.ndarray]], labels: dict[str, str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, c in curves.items():
        axes[0].plot(GRID, c["eval"], label=labels.get(method, method))
        axes[1].plot(GRID, c["loss"], label=labels.get(method, method))
    axes[0].set(xlabel="Training Progress (%)", ylabel="Eval Metric (avg. across tasks)",
                title="(a) Average evaluation metric")
    axes[1].set(xlabel="Training Progress (%)", ylabel="Training Loss", title="(b) Average training loss")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
