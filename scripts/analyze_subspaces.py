#!/usr/bin/env python
"""Overlap between the frozen subspaces selected by different methods.

For every target module of a task, compares the rank-r column (left) and row (right)
subspaces of
  * plain SVD of W0 (LoRA-XS / FiST no Fisher),
  * Fisher-weighted SVD (FiST-LoRA),
  * SVD of the mean calibration gradient G (a gradient-derived subspace),
  * SVD of sign(G) (the original LoRA-SB update direction; its -eta scale does not change
    the subspace),
using the normalised chordal distance sqrt(r - ||U1^T U2||_F^2) / sqrt(r) in [0, 1].
All four use the same warmed head and calibration examples as FiST.

    python scripts/analyze_subspaces.py --config configs/experiments/glue.yaml --task rte --rank 8
"""

import csv
import itertools
import json
import math
from collections import defaultdict

from fist_lora.cli import base_parser, setup


def chordal(U1, U2) -> float:
    r = U1.shape[1]
    overlap = float((U1.T @ U2).pow(2).sum())
    return math.sqrt(max(0.0, r - overlap) / r)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--rank", type=int, default=8)
    args = p.parse_args()
    setup()

    import torch
    from torch.utils.data import DataLoader

    from fist_lora.calibration.fisher import fisher_weighted_matrix
    from fist_lora.calibration.grad_stats import fisher_and_mean_gradient, module_groups
    from fist_lora.calibration.subspace import truncated_svd
    from fist_lora.calibration.warmup import warmup_head
    from fist_lora.config import load_experiment
    from fist_lora.config.load import repo_path
    from fist_lora.data.calibration import sample_subset
    from fist_lora.data.collate import Collator
    from fist_lora.data.loading import load_task_data
    from fist_lora.modeling.load import default_device, load_model, load_tokenizer
    from fist_lora.modeling.quant import effective_weight
    from fist_lora.modeling.targets import resolve_targets
    from fist_lora.reproducibility import seed_everything

    exp = load_experiment(args.config, args.overrides)
    cfg, calib, task = exp.model, exp.calibration, exp.task(args.task)
    device = default_device()
    tok = load_tokenizer(cfg)
    train_ds, _ = load_task_data(task, tok)
    collate = Collator(tok, cfg.task_type)

    seed_everything(calib.seed)  # identical lifecycle to FiST calibration
    model = load_model(cfg, task, device)
    warmup_head(model, cfg.head_modules, train_ds, collate, calib.warmup_steps, calib.warmup_batch_size,
                calib.warmup_lr, calib.seed, device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    subset = sample_subset(train_ds, calib.num_samples, calib.seed)
    targets = resolve_targets(model, cfg.target_modules, cfg.expected_num_targets)

    kinds = ["plain", "fisher", "grad", "sign_grad"]
    rows = []
    for group in module_groups(sorted(targets), calib.modules_per_pass):
        loader = DataLoader(subset, batch_size=calib.microbatch_size, collate_fn=collate)
        F, G, _ = fisher_and_mean_gradient(model, loader, {k: targets[k] for k in group}, cfg.task_type, device)
        for name in group:
            W0 = effective_weight(targets[name])
            svds = {
                "plain": truncated_svd(W0, args.rank),
                "fisher": truncated_svd(fisher_weighted_matrix(W0, F.pop(name), calib.clip_quantile, calib.eps), args.rank),
                "grad": truncated_svd(G[name], args.rank),
                "sign_grad": truncated_svd(torch.sign(G.pop(name)), args.rank),
            }
            row = {"module": name}
            for a, b in itertools.combinations(kinds, 2):
                row[f"{a}-{b}/left"] = chordal(svds[a].U, svds[b].U)
                row[f"{a}-{b}/right"] = chordal(svds[a].Vh.T, svds[b].Vh.T)
            rows.append(row)

    out = repo_path(exp.output_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"subspaces_{task.name}_r{args.rank}"
    with open(out / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = defaultdict(list)
    for row in rows:
        for k, v in row.items():
            if k != "module":
                summary[k].append(v)
    summary = {k: sum(v) / len(v) for k, v in summary.items()}
    (out / f"{stem}_mean.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
