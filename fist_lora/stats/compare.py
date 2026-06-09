"""Paired comparisons between methods (see docs/STATISTICS.md).

Unit of analysis: one trained model.  Within a task, methods run with the same seed share
the data order and the task-head initialisation, so differences are paired by seed.

* Per task: the seed-paired differences d_s = x_s - y_s.  Reported: mean difference, its
  95% t interval, the two-sided paired t-test and the exact sign-flip permutation test.
  With n = 3 seeds the smallest attainable permutation p-value is 2/2^3 = 0.25, so a
  per-task comparison can never be declared significant at 0.05; this is reported, not
  hidden.
* Across a suite: one difference per task (the seed-averaged d), tasks as units
  (Demšar 2006): exact Wilcoxon signed-rank test and one-sample t-test on the task
  differences.  Tasks are not a random sample of a population, so these p-values
  describe consistency across the evaluated tasks only.
* Multiplicity: Holm-Bonferroni within each family (all planned comparisons of a suite at
  one level of analysis).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from fist_lora.stats.aggregate import Record, mean_ci


@dataclass
class Comparison:
    level: str  # "task" | "suite"
    task: str
    method_a: str
    rank_a: int | None
    method_b: str
    rank_b: int | None
    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    t_pvalue: float
    exact_pvalue: float  # sign-flip permutation (task level) or exact Wilcoxon (suite level)
    exact_test: str
    holm_pvalue: float = math.nan  # of exact_pvalue, within the family


def sign_flip_pvalue(diffs: np.ndarray) -> float:
    """Exact two-sided permutation p-value of mean(diffs) under H0: symmetric around 0."""
    d = np.asarray(diffs, dtype=float)
    observed = abs(d.mean())
    flips = np.array(list(itertools.product((1.0, -1.0), repeat=d.size)))
    null = np.abs((flips * d).mean(axis=1))
    return float(np.mean(null >= observed - 1e-12))


def _t_pvalue(diffs: np.ndarray) -> float:
    if diffs.size < 2 or np.allclose(diffs, diffs[0]):
        return math.nan
    return float(stats.ttest_1samp(diffs, 0.0).pvalue)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs are ignored and kept)."""
    idx = [i for i, p in enumerate(pvalues) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvalues[i])
    m = len(order)
    adjusted = [math.nan] * len(pvalues)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _values(records: list[Record], task: str, method: str, rank: int | None) -> dict[int, float]:
    return {r.seed: r.value for r in records if r.task == task and r.method == method and r.rank == rank}


def compare_task(records, task, a: tuple[str, int | None], b: tuple[str, int | None]) -> Comparison | None:
    va, vb = _values(records, task, *a), _values(records, task, *b)
    seeds = sorted(set(va) & set(vb))
    if len(seeds) < 2:
        return None
    d = np.array([va[s] - vb[s] for s in seeds])
    mean, _, _, lo, hi = mean_ci(d.tolist())
    return Comparison("task", task, a[0], a[1], b[0], b[1], len(seeds), mean, lo, hi,
                      _t_pvalue(d), sign_flip_pvalue(d), "sign-flip permutation")


def compare_suite(records, tasks: list[str], a, b) -> Comparison | None:
    diffs = []
    for task in tasks:
        va, vb = _values(records, task, *a), _values(records, task, *b)
        seeds = sorted(set(va) & set(vb))
        if not seeds:
            return None
        diffs.append(np.mean([va[s] - vb[s] for s in seeds]))
    d = np.array(diffs)
    mean, _, _, lo, hi = mean_ci(d.tolist())
    if d.size >= 2 and np.any(d != 0):
        exact = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="exact").pvalue)
    else:
        exact = math.nan
    return Comparison("suite", "avg", a[0], a[1], b[0], b[1], len(tasks), mean, lo, hi,
                      _t_pvalue(d), exact, "Wilcoxon signed-rank (exact), tasks as units")


def planned_pairs(methods_ranks: set[tuple[str, int | None]], focus: str = "fist") -> list[tuple]:
    """FiST vs every r^2 baseline at equal rank, and FiST at its largest rank vs full-rank methods."""
    pairs = []
    fist_ranks = sorted(r for m, r in methods_ranks if m == focus)
    for m, r in sorted(methods_ranks, key=lambda x: (x[0], x[1] or 0)):
        if m == focus:
            continue
        if r in fist_ranks and m in ("lora_xs", "lora_sb", "fist_no_fisher"):
            pairs.append(((focus, r), (m, r)))
        elif m in ("lora", "pissa", "full_ft") and fist_ranks:
            pairs.append(((focus, fist_ranks[-1]), (m, r)))
    return pairs


def run_comparisons(records: list[Record], tasks: list[str], focus: str = "fist") -> list[Comparison]:
    pairs = planned_pairs({(r.method, r.rank) for r in records}, focus)
    task_level = [c for a, b in pairs for t in tasks if (c := compare_task(records, t, a, b))]
    suite_level = [c for a, b in pairs if (c := compare_suite(records, tasks, a, b))]
    for family in (task_level, suite_level):
        for c, p in zip(family, holm([c.exact_pvalue for c in family])):
            c.holm_pvalue = p
    return task_level + suite_level
