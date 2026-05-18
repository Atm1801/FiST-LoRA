#!/usr/bin/env python
"""Train (and evaluate) a single run: one task x method x rank x seed.

    python scripts/train.py --config configs/experiments/glue.yaml \
        --task rte --method fist --rank 8 --seed 42
"""

from fist_lora.cli import base_parser, setup


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--rank", type=int, default=None, help="omit for full fine-tuning")
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    setup()

    from fist_lora.config import load_experiment
    from fist_lora.config.schema import RunSpec
    from fist_lora.training.run import run_single

    exp = load_experiment(args.config, args.overrides)
    method = exp.method(args.method)
    if args.rank not in method.ranks:
        raise SystemExit(f"rank {args.rank} not configured for {args.method}: {method.ranks}")
    metrics = run_single(RunSpec(exp, exp.task(args.task), method, args.rank, args.seed))
    print({k: metrics[k] for k in ("run_id", "adapter_params") if k in metrics},
          metrics.get("best", metrics.get("benchmarks")))


if __name__ == "__main__":
    main()
