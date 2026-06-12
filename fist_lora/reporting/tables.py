"""Result tables per experiment (Markdown and LaTeX): methods x ranks by tasks.

Each cell is the mean over seeds, with the sample SD over seeds next to it.  Cells with fewer seeds than the experiment defines are
flagged with the seed count.
"""

from __future__ import annotations

import math

from fist_lora.stats.aggregate import CellSummary, Record

METHOD_LABELS = {
    "full_ft": "Full FT",
    "lora": "LoRA",
    "pissa": "PiSSA",
    "lora_xs": "LoRA-XS",
    "lora_sb": "LoRA-SB",
    "fist_no_fisher": "FiST (no Fisher)",
    "fist": "FiST-LoRA",
    "svd_zero": "SVD + zero R",
    "fisher_zero": "Fisher-SVD + zero R",
    "svd_sigma": "SVD + diag(S) R",
    "fisher_sigma": "Fisher-SVD + diag(S) R",
}

TASK_LABELS = {
    "cola": "CoLA (Mcc)", "rte": "RTE (Acc)", "mrpc": "MRPC (F1)", "stsb": "STS-B (Spr.)",
    "qnli": "QNLI (Acc)", "sst2": "SST-2 (Acc)", "boolq": "BoolQ", "piqa": "PIQA", "siqa": "SIQA",
    "hellaswag": "HellaS.", "winogrande": "WinoG.", "arc_easy": "ARC-e", "arc_challenge": "ARC-c",
    "openbookqa": "OBQA", "gsm8k": "GSM8K", "math": "MATH", "avg": "Avg.",
}


def format_params(n: int) -> str:
    """Thousands below 10M (e.g. "1572.86 K"), millions above (e.g. "19.99 M")."""
    return f"{n / 1e6:.2f} M" if n >= 1e7 else f"{n / 1e3:.2f} K"


def _cell(c: CellSummary | None, expected_n: int) -> str:
    if c is None:
        return "-"
    s = f"{c.mean:.2f}"
    if not math.isnan(c.sd):
        s += f" ± {c.sd:.2f}"
    if c.n != expected_n:
        s += f" (n={c.n})"
    return s


def build_rows(
    cells: list[CellSummary],
    avg_cells: list[CellSummary],
    records: list[Record],
    tasks: list[str],
    row_order: list[tuple[str, int | None]],
    expected_n: int,
) -> list[list[str]]:
    lookup = {(c.task, c.method, c.rank): c for c in cells + avg_cells}
    params = {}
    for r in records:
        params[(r.method, r.rank)] = r.trainable_params if r.method == "full_ft" else r.adapter_params
    rows = []
    for method, rank in row_order:
        if (method, rank) not in params:
            continue
        row = [METHOD_LABELS.get(method, method), "-" if rank is None else str(rank),
               format_params(params[(method, rank)])]
        row += [_cell(lookup.get((t, method, rank)), expected_n) for t in tasks]
        if len(tasks) > 1:
            row.append(_cell(lookup.get(("avg", method, rank)), expected_n))
        rows.append(row)
    return rows


def header(tasks: list[str]) -> list[str]:
    cols = ["Method", "Rank", "# Params"] + [TASK_LABELS.get(t, t) for t in tasks]
    return cols + (["Avg."] if len(tasks) > 1 else [])


def to_markdown(head: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def to_latex(head: list[str], rows: list[list[str]], caption: str) -> str:
    def esc(s: str) -> str:
        return s.replace("±", r"$\pm$").replace("_", r"\_").replace("%", r"\%")

    lines = [
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
        r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{ll" + "r" * (len(head) - 2) + "}", r"\toprule",
        " & ".join(esc(h) for h in head) + r" \\", r"\midrule",
    ]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"
