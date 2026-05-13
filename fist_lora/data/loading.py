"""Dispatch from a task config to its (train, eval) datasets."""

from __future__ import annotations

from fist_lora.config.schema import TaskConfig
from fist_lora.data.causal import load_commonsense170k, load_metamathqa
from fist_lora.data.glue import load_glue


def load_task_data(task: TaskConfig, tokenizer):
    """Return (train_dataset, eval_dataset_or_None) with columns input_ids/attention_mask/labels/length.

    The 7B tasks have no in-training validation set: they are evaluated on external
    benchmarks after training (lm-eval-harness, GSM8K/MATH).
    """
    if task.kind == "glue":
        return load_glue(task, tokenizer)
    if task.kind == "commonsense":
        return load_commonsense170k(task, tokenizer), None
    if task.kind == "metamath":
        return load_metamathqa(task, tokenizer), None
    raise ValueError(f"unknown task kind {task.kind!r}")
