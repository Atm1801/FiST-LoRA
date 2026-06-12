#!/usr/bin/env python
"""Analytic parameter counts, adapter storage and initial perturbation size.

No training involved.  Writes results/param_tables.md.

    python scripts/param_tables.py
"""

from pathlib import Path

from fist_lora.config.load import REPO_ROOT
from fist_lora.params import (
    ARCHS,
    GB_PER_MB,
    initial_perturbation,
    lora_params,
    r2_params,
    storage_mb,
)

MODELS = (("RoBERTa-large", "roberta", 4), ("LLaMA-2-7B", "llama", 7), ("Mistral-7B", "mistral", 7))


def main() -> None:
    out = ["# Parameter counts and adapter storage\n"]
    out.append("## Trainable adapter parameters of frozen-outer methods (L * M * r^2)\n")
    out.append("| Model | L | M | r=8 | r=16 | r=24 |\n|---|---|---|---|---|---|")
    for label, fam, m in MODELS:
        a = ARCHS[fam]
        out.append(f"| {label} | {a.num_layers} | {m} | " + " | ".join(f"{r2_params(a, fam, r):,}" for r in (8, 16, 24)) + " |")
    out.append("\n## LoRA r=8 (r(d+k) per module)\n")
    out.append("| Model | LoRA r=8 |\n|---|---|")
    for label, fam, _ in MODELS:
        out.append(f"| {label} | {lora_params(ARCHS[fam], fam, 8):,} |")
    out.append("\n## fp32 adapter storage and Adam state (decimal MB)\n")
    out.append("| Model | Method | Rank | # Params | Params (MB) | Optimizer (MB) |\n|---|---|---|---|---|---|")
    for label, fam, _ in MODELS:
        a = ARCHS[fam]
        rows = [("LoRA", 8, lora_params(a, fam, 8))] + [("FiST-LoRA", r, r2_params(a, fam, r)) for r in (8, 16, 24)]
        for meth, r, n in rows:
            pm, om = storage_mb(n)
            out.append(f"| {label} | {meth} | {r} | {n:,} | {pm:.3f} | {om:.3f} |")
    out.append("\n## Total adapter storage for n LLaMA-2-7B task adapters\n")
    out.append("| Method | n=10 | n=100 | n=1000 |\n|---|---|---|---|")
    a = ARCHS["llama"]
    for meth, n_params in (("LoRA (r=8)", lora_params(a, "llama", 8)), ("FiST-LoRA (r=24)", r2_params(a, "llama", 24))):
        mb = storage_mb(n_params)[0]
        out.append(f"| {meth} | " + " | ".join(f"{n * mb:.1f} MB = {n * mb / GB_PER_MB:.3f} GB" for n in (10, 100, 1000)) + " |")
    out.append("\n## Initial perturbation ||Delta W_init||_F = (alpha/r) gamma, gamma = 0.01\n")
    out.append("| Model family | alpha | r=8 | r=16 | r=24 |\n|---|---|---|---|---|")
    for label, alpha in (("RoBERTa-large", 16), ("7B models", 32)):
        out.append(f"| {label} | {alpha} | " + " | ".join(f"{initial_perturbation(alpha, r, 0.01):.4f}" for r in (8, 16, 24)) + " |")
    text = "\n".join(out) + "\n"
    path = Path(REPO_ROOT / "results" / "param_tables.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
