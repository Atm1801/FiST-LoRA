"""Helpers that map an experiment config to the rows/columns of its result table."""

from __future__ import annotations

from fist_lora.config.schema import ExperimentConfig


def benchmarks(exp: ExperimentConfig) -> list[str]:
    """Table columns: GLUE tasks, or the external benchmarks of the 7B suites."""
    kinds = {t.kind for t in exp.tasks}
    if kinds == {"glue"}:
        return [t.name for t in exp.tasks]
    if kinds == {"commonsense"}:
        return list(exp.evaluation.lm_eval_tasks)
    if kinds == {"metamath"}:
        return [b for b in ("gsm8k", "math") if getattr(exp.evaluation, b)]
    raise ValueError(f"cannot tabulate an experiment mixing task kinds {kinds}")


def row_order(exp: ExperimentConfig) -> list[tuple[str, int | None]]:
    return [(m.name, r) for m in exp.methods for r in m.ranks]
