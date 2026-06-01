#!/usr/bin/env python
"""Re-run the external evaluation of a trained 7B run from its saved adapter.

Use when training finished (train_metrics.json + adapter.pt exist) but the evaluation
did not, e.g. after a crash or to evaluate with a different lm-eval batch size.

    python scripts/evaluate.py --config configs/experiments/math.yaml \
        --task metamathqa50k --method fist --rank 8 --seed 42
"""

import json

from fist_lora.cli import base_parser, setup


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    setup()

    from fist_lora.config import load_experiment
    from fist_lora.config.schema import RunSpec
    from fist_lora.modeling.load import default_device, load_tokenizer
    from fist_lora.reproducibility import write_json
    from fist_lora.training.restore import restore_trained_model
    from fist_lora.training.run import evaluate_and_record, run_dir

    exp = load_experiment(args.config, args.overrides)
    spec = RunSpec(exp, exp.task(args.task), exp.method(args.method), args.rank, args.seed)
    out = run_dir(spec)
    with open(out / "train_metrics.json") as f:
        metrics = json.load(f)
    model = restore_trained_model(spec, default_device())
    metrics["benchmarks"] = evaluate_and_record(model, load_tokenizer(exp.model), spec, out)
    metrics["evaluated_by"] = "scripts/evaluate.py"
    write_json(out / "metrics.json", metrics)
    print(json.dumps({k: {m: v for m, v in r.items() if m in ("accuracy", "exact_match")}
                      for k, r in metrics["benchmarks"].items()}, indent=2))


if __name__ == "__main__":
    main()
