"""Per-seed records and seed aggregation.

The experimental unit is one trained model (task x method x rank x seed).  Every record
here is one such model on one benchmark; aggregation is over seeds only, never over
evaluation examples.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass
class Record:
    task: str  # GLUE task, or benchmark name for the 7B suites
    method: str
    rank: int | None
    seed: int
    value: float  # percentage points
    metric: str
    adapter_params: int
    trainable_params: int


def load_records(results_dir: str | Path, glue_value: str = "best") -> list[Record]:
    """Read every ``runs/**/metrics.json`` below ``results_dir``.

    ``glue_value``: "best" (best validation metric across epochs) or "final".
    """
    records = []
    for path in sorted(Path(results_dir).glob("runs/**/metrics.json")):
        with open(path) as f:
            m = json.load(f)
        trainable = m["adapter_params"] + m["head_params"]
        common = dict(method=m["method"], rank=m["rank"], seed=m["seed"],
                      adapter_params=m["adapter_params"], trainable_params=trainable)
        if "benchmarks" in m:
            for bench, res in m["benchmarks"].items():
                key = "accuracy" if "accuracy" in res else "exact_match"
                records.append(Record(task=bench, value=100 * res[key], metric=key, **common))
        else:
            records.append(Record(task=m["task"], value=100 * m[glue_value], metric=m["metric"], **common))
    return records


def write_records(records: list[Record], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(Record.__dataclass_fields__))
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


@dataclass
class CellSummary:
    task: str
    method: str
    rank: int | None
    n: int
    mean: float
    sd: float  # sample standard deviation (ddof = 1)
    sem: float
    ci_low: float  # 95% t interval over seeds
    ci_high: float
    seeds: list[int]
    values: list[float]
    adapter_params: int


def mean_ci(values: list[float], level: float = 0.95) -> tuple[float, float, float, float, float]:
    """(mean, sd, sem, ci_low, ci_high) with a t interval; sd/sem/ci are NaN for n < 2."""
    x = np.asarray(values, dtype=float)
    n = x.size
    mean = float(x.mean())
    if n < 2:
        return mean, math.nan, math.nan, math.nan, math.nan
    sd = float(x.std(ddof=1))
    sem = sd / math.sqrt(n)
    half = float(stats.t.ppf(0.5 + level / 2, df=n - 1)) * sem
    return mean, sd, sem, mean - half, mean + half


def summarize(records: list[Record]) -> list[CellSummary]:
    cells: dict[tuple, list[Record]] = defaultdict(list)
    for r in records:
        cells[(r.task, r.method, r.rank)].append(r)
    out = []
    for (task, method, rank), rs in cells.items():
        seeds = [r.seed for r in rs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seeds for {(task, method, rank)}: {seeds}")
        rs = sorted(rs, key=lambda r: r.seed)
        mean, sd, sem, lo, hi = mean_ci([r.value for r in rs])
        out.append(CellSummary(task, method, rank, len(rs), mean, sd, sem, lo, hi,
                               [r.seed for r in rs], [r.value for r in rs], rs[0].adapter_params))
    return out


def suite_average(records: list[Record], tasks: list[str]) -> list[CellSummary]:
    """The tables' "Avg." column: mean over tasks, with seed-level spread.

    For each (method, rank) and seed, the per-seed average over ``tasks`` is formed (the
    seed is the replicate); the reported mean equals the average of the per-task seed
    means when all cells are complete.  Incomplete (method, rank) groups are skipped.
    """
    by_group: dict[tuple, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    params = {}
    for r in records:
        if r.task in tasks:
            by_group[(r.method, r.rank)][r.seed][r.task] = r.value
            params[(r.method, r.rank)] = r.adapter_params
    out = []
    for (method, rank), per_seed in by_group.items():
        complete = {s: v for s, v in per_seed.items() if set(v) == set(tasks)}
        if not complete:
            continue
        seeds = sorted(complete)
        values = [float(np.mean([complete[s][t] for t in tasks])) for s in seeds]
        mean, sd, sem, lo, hi = mean_ci(values)
        out.append(CellSummary("avg", method, rank, len(seeds), mean, sd, sem, lo, hi, seeds, values,
                               params[(method, rank)]))
    return out


def write_summary(cells: list[CellSummary], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        fields = [k for k in CellSummary.__dataclass_fields__]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cells:
            w.writerow(asdict(c))
