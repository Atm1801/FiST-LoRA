#!/usr/bin/env python
"""Run every (task, method, rank, seed) of an experiment sequentially; resumable.

Completed runs (metrics.json present with a matching config hash) are skipped.  A failed
run is reported and the sweep stops (use --keep-going to continue); failures never write
metrics.json, so they are retried on the next invocation.

    python scripts/sweep.py --config configs/experiments/glue.yaml
    python scripts/sweep.py --config configs/experiments/glue.yaml --tasks rte --methods fist lora_xs
    python scripts/sweep.py --config configs/experiments/glue.yaml --dry-run
"""

import logging
import traceback

from fist_lora.cli import add_selection, base_parser, setup


def main() -> None:
    p = base_parser(__doc__)
    add_selection(p)
    p.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    p.add_argument("--keep-going", action="store_true", help="continue after a failed run")
    args = p.parse_args()
    setup()
    log = logging.getLogger("sweep")

    from fist_lora.config import expand_runs, load_experiment
    from fist_lora.training.run import run_is_complete, run_single

    exp = load_experiment(args.config, args.overrides)
    runs = expand_runs(exp, args.tasks, args.methods, args.ranks, args.seeds)
    todo = [r for r in runs if not run_is_complete(r)]
    log.info("%s: %d runs, %d already complete", exp.name, len(runs), len(runs) - len(todo))
    if args.dry_run:
        for r in runs:
            print(("done " if r not in todo else "todo ") + r.run_id)
        return
    failed = []
    for i, spec in enumerate(todo, 1):
        log.info("[%d/%d] %s", i, len(todo), spec.run_id)
        try:
            run_single(spec)
        except Exception:
            failed.append(spec.run_id)
            log.error("run %s failed:\n%s", spec.run_id, traceback.format_exc())
            if not args.keep_going:
                raise
    if failed:
        raise SystemExit(f"{len(failed)} runs failed: {failed}")


if __name__ == "__main__":
    main()
