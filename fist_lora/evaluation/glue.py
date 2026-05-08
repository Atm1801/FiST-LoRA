"""GLUE metrics, identical to the definitions in the official GLUE metric.

CoLA: Matthews correlation; MRPC: F1 of the positive class; STS-B: Spearman correlation;
RTE/QNLI/SST-2: accuracy.  Implemented directly with scikit-learn/scipy so evaluation
does not depend on downloading a metric script at run time.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

METRICS = ("accuracy", "f1", "matthews_correlation", "spearmanr")


def glue_metrics(logits: np.ndarray, labels: np.ndarray, metric: str) -> dict[str, float]:
    if metric not in METRICS:
        raise ValueError(f"unknown GLUE metric {metric!r}")
    if metric == "spearmanr":
        preds = np.asarray(logits).reshape(-1)
        return {
            "spearmanr": float(spearmanr(preds, labels)[0]),
            "pearson": float(pearsonr(preds, labels)[0]),
        }
    preds = np.asarray(logits).argmax(-1)
    out = {"accuracy": float(accuracy_score(labels, preds))}
    if metric == "f1":
        out["f1"] = float(f1_score(labels, preds))
    if metric == "matthews_correlation":
        out["matthews_correlation"] = float(matthews_corrcoef(labels, preds))
    return out


def make_compute_metrics(metric: str):
    def compute(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        return glue_metrics(logits, labels, metric)

    return compute


def epoch_metrics(log_history: list[dict], metric: str) -> list[dict[str, float]]:
    """Per-epoch validation results from the Trainer log history."""
    key = f"eval_{metric}"
    return [{"epoch": e["epoch"], "value": e[key]} for e in log_history if key in e]


def best_and_final(per_epoch: list[dict[str, float]]) -> dict[str, float]:
    """The reported GLUE value of a run is its best validation metric across epochs.

    All four GLUE metrics are higher-is-better.  Ties resolve to the earliest epoch.
    """
    if not per_epoch:
        raise ValueError("no per-epoch evaluations were recorded")
    values = [p["value"] for p in per_epoch]
    best_idx = int(np.argmax(values))
    return {
        "best": values[best_idx],
        "best_epoch": per_epoch[best_idx]["epoch"],
        "final": values[-1],
    }
