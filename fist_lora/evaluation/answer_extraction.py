"""Final-answer extraction for GSM8K/MATH.

Applied identically to model generations and to references:

1. the ``#### <number>`` pattern (GSM8K references);
2. otherwise the first ``\\boxed{...}`` span, up to the first closing brace (nested LaTeX is
   therefore truncated - a known property of this scorer);
3. otherwise the last numeric literal in the string.

Scoring is exact string match after stripping whitespace.  Both quirks apply identically
to references and generations of every method.
"""

from __future__ import annotations

import re

_HASH = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
_NUMBER = re.compile(r"-?[\d,]+\.?\d*")


def extract_answer(text: str) -> str:
    match = _HASH.search(text)
    if match:
        return match.group(1).replace(",", "")
    match = _BOXED.search(text)
    if match:
        return match.group(1).strip()
    numbers = _NUMBER.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def is_correct(generation: str, reference: str) -> bool:
    return extract_answer(generation).strip() == extract_answer(reference).strip()
