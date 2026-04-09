"""Seeding, determinism and run manifests.

``configure_determinism`` must run before the first CUDA call of a process (the CLI
entry points call it first thing), because cuBLAS reads ``CUBLAS_WORKSPACE_CONFIG`` when
its handle is created.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fist_lora import __version__


def configure_determinism(deterministic: bool = True, tf32: bool = False) -> None:
    """Process-wide numeric settings.

    Deterministic kernels are requested with ``warn_only=True``: a handful of CUDA ops
    (e.g. some scatter/index kernels) have no deterministic implementation, and aborting
    a multi-hour run over them is worse than a logged warning.  Bitwise reproducibility
    is therefore only guaranteed on identical hardware/software stacks.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_generator(seed: int) -> torch.Generator:
    """Dedicated RNG stream, so data sampling never depends on global RNG consumption."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive numpy/python seeds from torch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str]:
    from importlib import metadata

    names = [
        "torch", "transformers", "peft", "accelerate", "datasets", "numpy", "scipy",
        "scikit-learn", "bitsandbytes", "lm-eval",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def environment_manifest() -> dict[str, Any]:
    """Everything needed to identify the software/hardware stack of a run."""
    status = _git("status", "--porcelain")
    cuda = torch.cuda.is_available()
    return {
        "fist_lora_version": __version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str, sort_keys=True)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written result file
