"""Deterministic calibration subsets."""

from __future__ import annotations


def sample_subset(dataset, num_samples: int, seed: int):
    """Uniform sample without replacement, fixed by ``seed``.

    The N calibration examples (256 by default) come from the task training set.  A seeded
    sample is used instead of the first N rows because some training files are ordered by
    source (CommonSense170K concatenates its eight source datasets).
    """
    if num_samples > len(dataset):
        raise ValueError(f"requested {num_samples} calibration examples from {len(dataset)}")
    return dataset.shuffle(seed=seed).select(range(num_samples))
