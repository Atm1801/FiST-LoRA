"""Commonsense evaluation with lm-evaluation-harness.

Likelihood-based, zero-shot, lm-eval==0.4.5 task names (SIQA is ``social_iqa`` in this
version).  The reported value is ``acc``; ``acc_norm`` is stored as well where the task
defines it.
"""

from __future__ import annotations

TASK_NAMES = {
    "boolq": "boolq",
    "piqa": "piqa",
    "siqa": "social_iqa",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
    "openbookqa": "openbookqa",
}


def evaluate_commonsense(model, tokenizer, eval_cfg) -> dict:
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("commonsense evaluation needs `pip install -e .[eval]` (lm-eval==0.4.5)") from e

    model.eval()
    tasks = [TASK_NAMES.get(t, t) for t in eval_cfg.lm_eval_tasks]
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_cfg.batch_size)
    out = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=eval_cfg.num_fewshot,
        limit=eval_cfg.limit,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
        log_samples=False,
    )
    results = {}
    for short, name in zip(eval_cfg.lm_eval_tasks, tasks):
        r = out["results"][name]
        results[short] = {
            "accuracy": r["acc,none"],
            "acc_stderr": r.get("acc_stderr,none"),
            "acc_norm": r.get("acc_norm,none"),
            "lm_eval_task": name,
            "task_version": out.get("versions", {}).get(name),
            "num_examples": out.get("n-samples", {}).get(name, {}).get("effective"),
        }
    return results
