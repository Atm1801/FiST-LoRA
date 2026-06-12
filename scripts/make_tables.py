#!/usr/bin/env python
"""Render the result table of an experiment (methods x ranks by tasks) as Markdown and LaTeX.

    python scripts/make_tables.py --config configs/experiments/glue.yaml
"""

from fist_lora.cli import base_parser, setup


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--glue-value", choices=["best", "final"], default="best")
    args = p.parse_args()
    setup()

    from fist_lora.config import load_experiment
    from fist_lora.config.load import repo_path
    from fist_lora.reporting.experiment import benchmarks, row_order
    from fist_lora.reporting.tables import build_rows, header, to_latex, to_markdown
    from fist_lora.stats.aggregate import load_records, suite_average, summarize

    exp = load_experiment(args.config, args.overrides)
    root = repo_path(exp.output_dir)
    records = load_records(root, args.glue_value)
    if not records:
        raise SystemExit(f"no completed runs under {root}/runs")
    tasks = benchmarks(exp)
    rows = build_rows(summarize(records), suite_average(records, tasks), records, tasks,
                      row_order(exp), expected_n=len(exp.seeds))
    head = header(tasks)
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(head, rows)
    (out / f"{exp.name}.md").write_text(md)
    caption = f"{exp.name}: mean $\\pm$ SD over seeds {exp.seeds}"
    (out / f"{exp.name}.tex").write_text(to_latex(head, rows, caption))
    print(md)


if __name__ == "__main__":
    main()
