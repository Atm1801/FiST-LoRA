#!/usr/bin/env python
"""Pre-compute (and cache) the FiST calibration of every task of an experiment.

Training runs compute missing calibrations on demand; this script only front-loads the
cost (e.g. to inspect diagnostics before a sweep).  Prints per-module Fisher diagnostics.

    python scripts/calibrate.py --config configs/experiments/glue.yaml --tasks rte
"""

from fist_lora.cli import base_parser, setup


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--tasks", nargs="+")
    args = p.parse_args()
    setup()

    from fist_lora.config import load_experiment
    from fist_lora.data.collate import Collator
    from fist_lora.data.loading import load_task_data
    from fist_lora.modeling.load import default_device, load_tokenizer
    from fist_lora.training.run import task_calibration

    exp = load_experiment(args.config, args.overrides)
    tok = load_tokenizer(exp.model)
    for task in exp.tasks:
        if args.tasks and task.name not in args.tasks:
            continue
        train_ds, _ = load_task_data(task, tok)
        modules, meta = task_calibration(exp, task, train_ds, Collator(tok, exp.model.task_type), default_device())
        print(f"{task.name}: {meta['num_modules']} modules, {meta['num_examples']} examples, cache {meta['path']}")
        for name in sorted(modules)[:4]:
            print(f"  {name}: {modules[name].diagnostics}")


if __name__ == "__main__":
    main()
