"""YAML loading with ``_base_`` composition and dotted command-line overrides.

Composition rules (deliberately minimal, no external config framework):

* ``_base_: [a.yaml, b.yaml]`` - files merged in order (paths relative to the including
  file), then the including file's own keys are deep-merged on top.
* ``tasks:`` entries may be strings, resolved to ``configs/tasks/<entry>.yaml``.
* ``--set a.b.c=value`` overrides are applied last; values are parsed as YAML scalars.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from fist_lora.config.schema import (
    ArchitectureSpec,
    CalibrationConfig,
    EvaluationConfig,
    ExperimentConfig,
    LoraSBConfig,
    MethodEntry,
    ModelConfig,
    RunSpec,
    TaskConfig,
    TrainingConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml_tree(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    merged: dict[str, Any] = {}
    for base in raw.pop("_base_", []):
        merged = deep_merge(merged, load_yaml_tree(path.parent / base))
    return deep_merge(merged, raw)


def apply_overrides(tree: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    tree = copy.deepcopy(tree)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, value = item.split("=", 1)
        node = tree
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return tree


def _resolve_task(entry: str | dict[str, Any]) -> TaskConfig:
    if isinstance(entry, str):
        entry = load_yaml_tree(CONFIG_ROOT / "tasks" / f"{entry}.yaml")
    return TaskConfig(**entry)


def _build_model(tree: dict[str, Any]) -> ModelConfig:
    tree = dict(tree)
    arch = tree.pop("architecture", None)
    return ModelConfig(**tree, architecture=ArchitectureSpec(**arch) if arch else None)


def build_experiment(tree: dict[str, Any], source_path: str | None = None) -> ExperimentConfig:
    required = ["name", "model", "tasks", "methods", "seeds"]
    missing = [k for k in required if k not in tree]
    if missing:
        raise ValueError(f"experiment config is missing keys: {missing}")
    methods = []
    for m in tree["methods"]:
        m = dict(m)
        if "ranks" in m and m["ranks"] is not None:
            m["ranks"] = list(m["ranks"])
        methods.append(MethodEntry(**m))
    return ExperimentConfig(
        name=tree["name"],
        model=_build_model(tree["model"]),
        tasks=[_resolve_task(t) for t in tree["tasks"]],
        methods=methods,
        seeds=list(tree["seeds"]),
        calibration=CalibrationConfig(**tree.get("calibration", {})),
        training=TrainingConfig(**tree.get("training", {})),
        evaluation=EvaluationConfig(**tree.get("evaluation", {})),
        lora_sb=LoraSBConfig(**tree.get("lora_sb", {})),
        output_dir=tree.get("output_dir", f"results/{tree['name']}"),
        source_path=source_path,
    )


def load_experiment(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    tree = apply_overrides(load_yaml_tree(path), overrides)
    return build_experiment(tree, source_path=str(Path(path).resolve()))


def expand_runs(
    exp: ExperimentConfig,
    tasks: list[str] | None = None,
    methods: list[str] | None = None,
    ranks: list[int] | None = None,
    seeds: list[int] | None = None,
) -> list[RunSpec]:
    """Cartesian product task x method x rank x seed, optionally filtered."""
    runs = []
    for task, method in itertools.product(exp.tasks, exp.methods):
        if tasks and task.name not in tasks:
            continue
        if methods and method.name not in methods:
            continue
        for rank in method.ranks:
            if ranks and rank is not None and rank not in ranks:
                continue
            for seed in exp.seeds:
                if seeds and seed not in seeds:
                    continue
                runs.append(RunSpec(exp, task, method, rank, seed))
    return runs


_HASH_EXCLUDE = {"output_dir", "source_path", "cache_dir", "enabled", "save_adapter"}


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in sorted(obj.items()) if k not in _HASH_EXCLUDE}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def stable_hash(obj: Any, length: int = 16) -> str:
    """Hash of a JSON-serialisable object, ignoring pure bookkeeping keys (paths, toggles)."""
    payload = json.dumps(_strip(obj), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def repo_path(p: str | Path) -> Path:
    """Resolve a config path relative to the repository root (never the working directory)."""
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_data_files(data_files: str | list | dict) -> str | list | dict:
    """Apply :func:`repo_path` to a ``datasets`` ``data_files`` specification."""
    if isinstance(data_files, dict):
        return {k: resolve_data_files(v) for k, v in data_files.items()}
    if isinstance(data_files, list):
        return [str(repo_path(p)) for p in data_files]
    return str(repo_path(data_files))
