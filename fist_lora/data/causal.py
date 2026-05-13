"""Instruction-tuning data for the 7B experiments.

CommonSense170K (Hu et al. 2023, LLM-Adapters): prompt template, EOS handling and
``train_on_inputs=True`` (loss on the full sequence) reproduce LLM-Adapters' ``finetune.py``
at commit fe675038fb60d61b3fc03d98673d6d11d2bef4f9, the source of the dataset.

MetaMathQA 50K: a seeded 50K subset (shuffle seed 42), formatted with the evaluation
template ("### Question:\\n{q}\\n\\n### Answer:\\n") followed by the response and EOS, loss
on the full sequence.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from datasets import load_dataset

from fist_lora.config.load import repo_path, resolve_data_files
from fist_lora.config.schema import TaskConfig

MATH_PROMPT = "### Question:\n{question}\n\n### Answer:\n"


def llm_adapters_prompt(instruction: str, input_text: str, output: str) -> str:
    """Verbatim ``generate_prompt`` of LLM-Adapters finetune.py (including its indentation)."""
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input_text}

                ### Response:
                {output}"""  # noqa: W291, W293
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Response:
                {output}"""  # noqa: W291, W293


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def ensure_local_file(task: TaskConfig) -> Path:
    """Download a pinned raw file once and verify it against its git blob hash."""
    path = repo_path(task.data_files)
    if not path.exists():
        if not task.data_url:
            raise FileNotFoundError(f"{path} not found and no data_url configured")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(task.data_url, path)
    if task.data_git_blob_sha:
        actual = git_blob_sha1(path)
        if actual != task.data_git_blob_sha:
            raise ValueError(f"{path}: git blob sha {actual} != pinned {task.data_git_blob_sha}")
    return path


def _tokenize_with_eos(tokenizer, text: str, max_length: int) -> dict:
    enc = tokenizer(text, truncation=True, max_length=max_length, padding=False)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    if ids[-1] != tokenizer.eos_token_id and len(ids) < max_length:
        ids = ids + [tokenizer.eos_token_id]
        mask = mask + [1]
    return {"input_ids": ids, "attention_mask": mask, "labels": list(ids), "length": len(ids)}


def load_commonsense170k(task: TaskConfig, tokenizer):
    path = ensure_local_file(task)
    ds = load_dataset("json", data_files=str(path), split="train")
    if task.max_train_samples is not None:
        ds = ds.select(range(min(task.max_train_samples, len(ds))))

    def fmt(ex):
        text = llm_adapters_prompt(ex["instruction"], ex.get("input") or "", ex["output"])
        return _tokenize_with_eos(tokenizer, text, task.max_length)

    return ds.map(fmt, remove_columns=ds.column_names)


def load_metamathqa(task: TaskConfig, tokenizer):
    if task.data_files:
        ds = load_dataset("json", data_files=resolve_data_files(task.data_files), split="train")
    else:
        ds = load_dataset(task.dataset, revision=task.dataset_revision, split="train")
    if task.subset_size is not None and len(ds) > task.subset_size:
        ds = ds.shuffle(seed=task.subset_seed).select(range(task.subset_size))
    if task.max_train_samples is not None:
        ds = ds.select(range(min(task.max_train_samples, len(ds))))

    def fmt(ex):
        text = MATH_PROMPT.format(question=ex["query"]) + ex["response"]
        return _tokenize_with_eos(tokenizer, text, task.max_length)

    return ds.map(fmt, remove_columns=ds.column_names)
