"""Padding collators shared by training, calibration and LoRA-SB estimation."""

from __future__ import annotations

from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding

MODEL_KEYS = ("input_ids", "attention_mask", "labels")


class Collator:
    """Drops bookkeeping columns (e.g. ``length``) and pads to a multiple of 8."""

    def __init__(self, tokenizer, task_type: str) -> None:
        if task_type == "seq_cls":
            self._inner = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            self._inner = DataCollatorForSeq2Seq(
                tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100
            )

    def __call__(self, features):
        return self._inner([{k: f[k] for k in MODEL_KEYS if k in f} for f in features])
