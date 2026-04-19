"""
FiST-LoRA configuration dataclass.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FiSTLoRAConfig:
    """Configuration for FiST-LoRA adaptation."""

    # Model
    model_name: str = "roberta-large"
    task_type: str = "seq_cls"  # "seq_cls" or "causal_lm"
    num_labels: int = 2

    # LoRA architecture
    rank: int = 32
    alpha: float = 32.0
    target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value"]
    )

    # Initialization method
    method: str = "fist_full"  # "fist_full", "fist_no_fisher", "lora_xs", "no_grad"
    init_scale: float = 0.01

    # Calibration
    calibration_samples: int = 256
    head_warmup_steps: int = 100
    head_warmup_lr: float = 1e-3
    head_warmup_batch_size: int = 32

    # Training
    learning_rate: float = 1e-3
    num_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    weight_decay: float = 0.0
    warmup_ratio: float = 0.06
    fp16: bool = False
    bf16: bool = False
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    max_seq_length: int = 128

    # Quantization (for 7B+ models)
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Head unfreezing
    head_keywords_cls: List[str] = field(
        default_factory=lambda: ["classifier", "score", "qa_outputs"]
    )
    head_keywords_lm: List[str] = field(
        default_factory=lambda: ["lm_head"]
    )

    # Reproducibility
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # Output
    output_dir: str = "results"
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    save_strategy: str = "no"
    report_to: str = "none"

    @property
    def head_keywords(self) -> List[str]:
        if self.task_type == "causal_lm":
            return list(self.head_keywords_lm)
        return list(self.head_keywords_cls)
