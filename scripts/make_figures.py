#!/usr/bin/env python
"""GLUE training dynamics of the r^2 methods (default r = 8), from the runs of glue.yaml.

    python scripts/make_figures.py --config configs/experiments/glue.yaml
"""

from fist_lora.cli import base_parser, setup


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--methods", nargs="+", default=["lora_xs", "lora_sb", "fist_no_fisher", "fist"])
    args = p.parse_args()
    setup()

    from fist_lora.config import load_experiment
    from fist_lora.config.load import repo_path
    from fist_lora.reporting.figures import average_curves, plot_training_dynamics
    from fist_lora.reporting.tables import METHOD_LABELS

    exp = load_experiment(args.config, args.overrides)
    root = repo_path(exp.output_dir)
    tasks = {t.name: (t.metric, t.epochs) for t in exp.tasks}
    curves = average_curves(root, tasks, args.methods, args.rank)
    missing = set(args.methods) - set(curves)
    if missing:
        print(f"warning: incomplete runs for {sorted(missing)}; they are left out of the figure")
    if not curves:
        raise SystemExit("no method has completed runs for every task")
    plot_training_dynamics(curves, METHOD_LABELS, root / "figures" / "training_dynamics")
    print(f"wrote {root / 'figures' / 'training_dynamics'}.{{pdf,png}}")


if __name__ == "__main__":
    main()
