"""Shared command-line plumbing for the scripts in ``scripts/``."""

from __future__ import annotations

import argparse
import logging
import os

from fist_lora.reproducibility import configure_determinism


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="experiment YAML (configs/experiments/*.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set training.logging_steps=50 (repeatable)")
    return p


def add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tasks", nargs="+", help="subset of task names")
    p.add_argument("--methods", nargs="+", help="subset of method names")
    p.add_argument("--ranks", nargs="+", type=int, help="subset of ranks")
    p.add_argument("--seeds", nargs="+", type=int, help="subset of seeds")


def setup(level: str = "INFO") -> None:
    # Must precede any CUDA work (cuBLAS reads CUBLAS_WORKSPACE_CONFIG at handle creation).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # git subprocess in the manifest forks
    configure_determinism()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
