"""Offline test fixtures: tiny models built from configs (no hub access) and synthetic data."""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    RobertaConfig,
    RobertaForSequenceClassification,
)

from fist_lora.config.schema import ModelConfig

ROBERTA_TARGETS = [r".*\.attention\.self\.(query|key|value)", r".*\.attention\.output\.dense"]
LLAMA_TARGETS = [r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)", r".*\.mlp\.(gate_proj|up_proj|down_proj)"]


def tiny_roberta(num_labels: int = 2, seed: int = 0) -> RobertaForSequenceClassification:
    torch.manual_seed(seed)
    cfg = RobertaConfig(
        vocab_size=100, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=24, max_position_embeddings=64, num_labels=num_labels,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    )
    return RobertaForSequenceClassification(cfg).eval()


def tiny_llama(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=100, hidden_size=16, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=24, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    return LlamaForCausalLM(cfg).eval()


def roberta_model_cfg(**kw) -> ModelConfig:
    base = dict(
        name="tiny", revision=None, task_type="seq_cls", target_modules=ROBERTA_TARGETS,
        expected_num_targets=8, alpha=16, precision="bf16-mixed", gradient_checkpointing=False,
        head_modules=["classifier"], train_head=True,
    )
    base.update(kw)
    return ModelConfig(**base)


def llama_model_cfg(**kw) -> ModelConfig:
    base = dict(
        name="tiny", revision=None, task_type="causal_lm", target_modules=LLAMA_TARGETS,
        expected_num_targets=14, alpha=32, precision="bf16-mixed", gradient_checkpointing=True,
        head_modules=["lm_head"], train_head=False,
    )
    base.update(kw)
    return ModelConfig(**base)


def cls_rows(n: int = 12, seed: int = 0, regression: bool = False) -> list[dict]:
    """Variable-length sequences, so batching involves padding."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(n):
        length = int(torch.randint(4, 12, (1,), generator=g))
        ids = torch.randint(3, 100, (length,), generator=g).tolist()
        label = float(torch.rand(1, generator=g)) * 5 if regression else int(torch.randint(0, 2, (1,), generator=g))
        rows.append({"input_ids": ids, "attention_mask": [1] * length, "labels": label, "length": length})
    return rows


def lm_rows(n: int = 8, seed: int = 0) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(n):
        length = int(torch.randint(5, 14, (1,), generator=g))
        ids = torch.randint(3, 100, (length,), generator=g).tolist()
        labels = list(ids)
        labels[0] = -100  # a masked prompt token
        rows.append({"input_ids": ids, "attention_mask": [1] * length, "labels": labels, "length": length})
    return rows


def pad_batch(rows: list[dict], task_type: str) -> dict[str, torch.Tensor]:
    """Right padding with pad id 0; labels padded with -100 for causal LM."""
    width = max(len(r["input_ids"]) for r in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r["input_ids"])] = torch.tensor(r["input_ids"])
        mask[i, : len(r["input_ids"])] = 1
    batch = {"input_ids": ids, "attention_mask": mask}
    if task_type == "seq_cls":
        labels = [r["labels"] for r in rows]
        batch["labels"] = torch.tensor(labels, dtype=torch.float if isinstance(labels[0], float) else torch.long)
    else:
        lab = torch.full((len(rows), width), -100, dtype=torch.long)
        for i, r in enumerate(rows):
            lab[i, : len(r["labels"])] = torch.tensor(r["labels"])
        batch["labels"] = lab
    return batch


class PadCollator:
    def __init__(self, task_type: str) -> None:
        self.task_type = task_type

    def __call__(self, features):
        return pad_batch([dict(f) for f in features], self.task_type)


@pytest.fixture
def cls_dataset():
    return Dataset.from_list(cls_rows(24))


@pytest.fixture
def lm_dataset():
    return Dataset.from_list(lm_rows(12))
