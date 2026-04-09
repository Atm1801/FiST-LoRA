"""Typed configuration objects.

Every experiment is fully described by an :class:`ExperimentConfig` loaded from YAML
(``configs/experiments/*.yaml``).  A single training run is an :class:`RunSpec`, the
cartesian product element (task, method, rank, seed) of an experiment.  Nothing that
affects results is allowed to live outside these objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ArchitectureSpec:
    """Dimensions used for analytic parameter counting (``fist_lora.params``).

    Checked against the loaded model at injection time, so a mismatch between the
    YAML and the real checkpoint fails loudly.
    """

    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int


@dataclass
class ModelConfig:
    name: str
    revision: str | None
    task_type: str  # "seq_cls" | "causal_lm"
    # Regexes matched with re.fullmatch against dotted module names.
    target_modules: list[str]
    expected_num_targets: int | None
    alpha: float
    # "bf16-mixed": fp32 weights with bf16 autocast; "nf4": 4-bit NF4 + double quant, bf16 compute.
    precision: str
    gradient_checkpointing: bool
    head_modules: list[str]
    # Whether the task head is trained (seq_cls: yes; causal LM: frozen pretrained lm_head).
    train_head: bool
    tokenizer_name: str | None = None
    pad_token: str | None = None  # e.g. "unk" or "eos" for LLaMA/Mistral tokenizers
    architecture: ArchitectureSpec | None = None


@dataclass
class TaskConfig:
    name: str
    kind: str  # "glue" | "commonsense" | "metamath"
    epochs: int
    max_length: int
    metric: str | None = None
    num_labels: int | None = None
    dataset: str | None = None
    dataset_config: str | None = None
    dataset_revision: str | None = None
    text_keys: list[str | None] | None = None
    eval_split: str = "validation"
    # Optional caps (smoke tests only; None = full split).
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    # Local file sources (CommonSense170K json, fixtures).
    data_files: str | dict[str, str] | None = None
    data_url: str | None = None
    data_git_blob_sha: str | None = None
    subset_size: int | None = None
    subset_seed: int | None = None


@dataclass
class CalibrationConfig:
    num_samples: int = 256
    seed: int = 42
    warmup_steps: int = 100
    warmup_batch_size: int = 32
    warmup_lr: float = 1e-3
    clip_quantile: float = 0.95
    eps: float = 1e-8
    init_scale: float = 0.01
    microbatch_size: int = 1
    # Maximum number of target modules whose statistics are accumulated in one pass.
    # 7B models need several passes (F and G for all 224 modules would need ~52 GB fp32).
    modules_per_pass: int | None = None
    svd_dtype: str = "float32"
    cache_dir: str = "cache"


@dataclass
class LoraSBConfig:
    """Original LoRA-SB gradient-estimation settings (Ponkshe et al. 2024, official code)."""

    num_samples: int = 2
    estimation_batch_size: int = 128


@dataclass
class TrainingConfig:
    per_device_train_batch_size: int = 128
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 256
    weight_decay: float = 0.0
    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "linear"
    logging_steps: int = 10
    group_by_length: bool = False
    eval_every_epoch: bool = True
    dataloader_num_workers: int = 0
    deterministic: bool = True
    tf32: bool = False
    max_steps: int = -1  # smoke tests only
    save_adapter: bool = True


@dataclass
class EvaluationConfig:
    enabled: bool = True
    batch_size: int = 16
    # Commonsense (lm-eval-harness task names, lm-eval==0.4.5).
    lm_eval_tasks: list[str] = field(default_factory=list)
    num_fewshot: int = 0
    limit: int | None = None
    # Math: GSM8K / MATH generation settings.
    gsm8k: dict[str, Any] = field(default_factory=dict)
    math: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodEntry:
    """A method as it appears in an experiment.

    ``name`` is the unique label used in result paths and tables; ``method`` is the key in
    :data:`fist_lora.methods.registry.METHODS` (defaults to ``name``), so one registry
    method can appear several times with different ``params`` (e.g. the gamma sweep).
    """

    name: str
    lr: float
    ranks: list[int | None] = field(default_factory=lambda: [None])
    params: dict[str, Any] = field(default_factory=dict)
    method: str | None = None

    @property
    def registry_key(self) -> str:
        return self.method or self.name


@dataclass
class ExperimentConfig:
    name: str
    model: ModelConfig
    tasks: list[TaskConfig]
    methods: list[MethodEntry]
    seeds: list[int]
    calibration: CalibrationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    lora_sb: LoraSBConfig
    output_dir: str
    source_path: str | None = None

    def task(self, name: str) -> TaskConfig:
        for t in self.tasks:
            if t.name == name:
                return t
        raise KeyError(f"task {name!r} not in experiment {self.name!r}: {[t.name for t in self.tasks]}")

    def method(self, name: str) -> MethodEntry:
        for m in self.methods:
            if m.name == name:
                return m
        raise KeyError(f"method {name!r} not in experiment {self.name!r}: {[m.name for m in self.methods]}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunSpec:
    """One training run: the unit of analysis for all statistics (one trained model)."""

    experiment: ExperimentConfig
    task: TaskConfig
    method: MethodEntry
    rank: int | None
    seed: int

    @property
    def run_id(self) -> str:
        rank = f"r{self.rank}" if self.rank is not None else "full"
        return f"{self.task.name}/{self.method.name}/{rank}/seed{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        exp = self.experiment.to_dict()
        exp.pop("tasks")
        exp.pop("methods")
        exp.pop("seeds")
        return {
            "experiment": exp,
            "task": asdict(self.task),
            "method": asdict(self.method),
            "rank": self.rank,
            "seed": self.seed,
        }
