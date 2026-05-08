"""GLUE tasks: tokenisation of train and validation splits."""

from __future__ import annotations

from datasets import load_dataset

from fist_lora.config.load import resolve_data_files
from fist_lora.config.schema import TaskConfig


def load_glue(task: TaskConfig, tokenizer):
    if task.data_files:
        raw = load_dataset("json", data_files=resolve_data_files(task.data_files))  # offline fixtures: {"train", "validation"}
    else:
        raw = load_dataset(task.dataset, task.dataset_config, revision=task.dataset_revision)
    key1, key2 = task.text_keys
    is_regression = task.num_labels == 1

    def tokenize(batch):
        texts = (batch[key1],) if key2 is None else (batch[key1], batch[key2])
        enc = tokenizer(*texts, truncation=True, max_length=task.max_length)
        enc["labels"] = [float(x) for x in batch["label"]] if is_regression else batch["label"]
        enc["length"] = [len(ids) for ids in enc["input_ids"]]
        return enc

    splits = {}
    for split, cap in (("train", task.max_train_samples), (task.eval_split, task.max_eval_samples)):
        ds = raw[split]
        if cap is not None:
            ds = ds.select(range(min(cap, len(ds))))
        splits[split] = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    return splits["train"], splits[task.eval_split]
