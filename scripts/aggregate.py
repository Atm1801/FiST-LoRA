#!/usr/bin/env python
"""Collect per-seed results of an experiment and compute seed statistics and comparisons.

Writes to <output_dir>/summary/:
  per_seed.csv     one row per trained model and benchmark (the unit of analysis)
  aggregate.csv    mean, SD (ddof=1), SEM and 95% t-CI over seeds per (task, method, rank),
                   plus the suite average ("avg") computed per seed
  comparisons.csv  paired FiST-vs-baseline comparisons (see docs/STATISTICS.md)

    python scripts/aggregate.py --config configs/experiments/glue.yaml
"""

import csv
from dataclasses import asdict

from fist_lora.cli import base_parser, setup


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--glue-value", choices=["best", "final"], default="best",
                   help="GLUE: best epoch (default) or final epoch")
    args = p.parse_args()
    setup()

    from fist_lora.config import load_experiment
    from fist_lora.config.load import repo_path
    from fist_lora.reporting.experiment import benchmarks
    from fist_lora.stats.aggregate import (
        load_records,
        suite_average,
        summarize,
        write_records,
        write_summary,
    )
    from fist_lora.stats.compare import Comparison, run_comparisons

    exp = load_experiment(args.config, args.overrides)
    root = repo_path(exp.output_dir)
    records = load_records(root, args.glue_value)
    if not records:
        raise SystemExit(f"no completed runs under {root}/runs")
    tasks = benchmarks(exp)
    out = root / "summary"
    write_records(records, out / "per_seed.csv")
    write_summary(summarize(records) + suite_average(records, tasks), out / "aggregate.csv")
    comps = run_comparisons(records, tasks)
    with open(out / "comparisons.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(Comparison.__dataclass_fields__))
        w.writeheader()
        for c in comps:
            w.writerow(asdict(c))
    print(f"{len(records)} records -> {out}")


if __name__ == "__main__":
    main()
