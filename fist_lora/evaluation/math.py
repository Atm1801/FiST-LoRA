"""GSM8K and MATH exact-match evaluation.

Greedy decoding (``do_sample=False``), 256 / 512 new tokens, prompt
"### Question:\\n{question}\\n\\n### Answer:\\n", full test sets (1,319 / 5,000 problems).

``hendrycks/competition_math`` has been removed from the
Hugging Face Hub; MATH is read from the pinned ``EleutherAI/hendrycks_math`` mirror (the
same 5,000 test problems split into seven subject configs, concatenated in a fixed order).

Generation is batched with left padding; with an attention mask and mask-derived position
ids this is equivalent to one-at-a-time greedy decoding (checked in the tests).
"""

from __future__ import annotations

import torch
from datasets import concatenate_datasets, load_dataset

from fist_lora.config.load import resolve_data_files
from fist_lora.data.causal import MATH_PROMPT
from fist_lora.evaluation.answer_extraction import extract_answer


def load_benchmark(spec: dict, name: str):
    """Rows with ``question`` and ``reference`` fields."""
    if not spec.get("data_files") and not spec.get("dataset"):
        raise ValueError(f"evaluation.{name} needs either `dataset` or `data_files`: {spec}")
    if spec.get("data_files"):
        ds = load_dataset("json", data_files=resolve_data_files(spec["data_files"]), split="train")
    elif name == "gsm8k":
        ds = load_dataset(spec["dataset"], spec.get("config", "main"), revision=spec.get("revision"), split=spec.get("split", "test"))
    elif name == "math":
        parts = [
            load_dataset(spec["dataset"], cfg, revision=spec.get("revision"), split=spec.get("split", "test"))
            for cfg in spec["configs"]
        ]
        ds = concatenate_datasets(parts)
    else:
        raise ValueError(f"unknown benchmark {name!r}")
    qk, rk = ("question", "answer") if name == "gsm8k" else ("problem", "solution")
    ds = ds.map(lambda ex: {"question": ex[qk], "reference": ex[rk]}, remove_columns=ds.column_names)
    if spec.get("limit") is not None:
        ds = ds.select(range(min(spec["limit"], len(ds))))
    return ds


@torch.no_grad()
def greedy_generate(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    device = next(model.parameters()).device
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            new_tokens = gen[:, enc["input_ids"].shape[1] :]
            outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old_side
    return outputs


def evaluate_math_benchmarks(model, tokenizer, eval_cfg) -> dict:
    model.eval()
    results = {}
    for name in ("gsm8k", "math"):
        spec = getattr(eval_cfg, name)
        if not spec:
            continue
        ds = load_benchmark(spec, name)
        prompts = [MATH_PROMPT.format(question=q) for q in ds["question"]]
        gens = greedy_generate(model, tokenizer, prompts, spec["max_new_tokens"], eval_cfg.batch_size)
        preds = [extract_answer(g).strip() for g in gens]
        golds = [extract_answer(r).strip() for r in ds["reference"]]
        correct = [p == g for p, g in zip(preds, golds)]
        results[name] = {
            "exact_match": sum(correct) / len(correct),
            "num_correct": sum(correct),
            "num_examples": len(correct),
            "per_example": [
                {"prediction": p, "reference": g, "correct": c, "generation": gen}
                for p, g, c, gen in zip(preds, golds, correct, gens)
            ],
        }
    return results
