"""Seed aggregation and paired comparisons."""

import json
import math

import numpy as np
import pytest
from scipy import stats

from fist_lora.stats.aggregate import Record, load_records, mean_ci, suite_average, summarize
from fist_lora.stats.compare import (
    compare_suite,
    compare_task,
    holm,
    planned_pairs,
    run_comparisons,
    sign_flip_pvalue,
)


def rec(task, method, rank, seed, value, params=100):
    return Record(task, method, rank, seed, value, "accuracy", params, params + 10)


def test_mean_ci_matches_scipy():
    x = [80.0, 82.5, 81.0]
    mean, sd, sem, lo, hi = mean_ci(x)
    assert mean == pytest.approx(np.mean(x)) and sd == pytest.approx(np.std(x, ddof=1))
    assert sem == pytest.approx(stats.sem(x))
    ref = stats.t.interval(0.95, df=2, loc=np.mean(x), scale=stats.sem(x))
    assert (lo, hi) == pytest.approx(ref)
    assert all(math.isnan(v) for v in mean_ci([1.0])[1:])


def test_sign_flip_pvalue_bounds_with_three_seeds():
    assert sign_flip_pvalue(np.array([1.0, 2.0, 3.0])) == pytest.approx(0.25)  # minimum attainable at n=3
    assert sign_flip_pvalue(np.array([1.0, -1.0, 0.5])) > 0.25


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = holm([0.2, math.nan])
    assert out[0] == pytest.approx(0.2) and math.isnan(out[1])


def test_summarize_rejects_duplicate_seeds_and_keeps_values():
    rs = [rec("rte", "fist", 8, s, v) for s, v in ((42, 80), (123, 82), (456, 81))]
    (cell,) = summarize(rs)
    assert cell.n == 3 and cell.values == [80, 82, 81] and cell.seeds == [42, 123, 456]
    with pytest.raises(ValueError):
        summarize(rs + [rec("rte", "fist", 8, 42, 79)])


def test_suite_average_is_per_seed_average_over_tasks():
    rs = []
    for s, (a, b) in zip((1, 2, 3), ((80, 60), (82, 62), (81, 64))):
        rs += [rec("t1", "m", 8, s, a), rec("t2", "m", 8, s, b)]
    rs.append(rec("t1", "incomplete", 8, 1, 50))
    (avg,) = suite_average(rs, ["t1", "t2"])
    assert avg.values == [70, 72, 72.5] and avg.mean == pytest.approx(np.mean([70, 72, 72.5]))


def test_paired_comparisons():
    rs = []
    for s, (x, y) in zip((42, 123, 456), ((81, 80), (83, 80.5), (82, 81))):
        for t, off in (("rte", 0), ("cola", -10)):
            rs += [rec(t, "fist", 8, s, x + off), rec(t, "lora_xs", 8, s, y + off)]
    c = compare_task(rs, "rte", ("fist", 8), ("lora_xs", 8))
    assert c.mean_diff == pytest.approx(np.mean([1, 2.5, 1])) and c.n == 3
    assert c.t_pvalue == pytest.approx(stats.ttest_rel([81, 83, 82], [80, 80.5, 81]).pvalue)
    suite = compare_suite(rs, ["rte", "cola"], ("fist", 8), ("lora_xs", 8))
    assert suite.n == 2 and suite.mean_diff == pytest.approx(1.5)
    assert planned_pairs({("fist", 8), ("lora_xs", 8), ("lora", 8), ("fist", 24), ("full_ft", None)}) == [
        (("fist", 24), ("full_ft", None)), (("fist", 24), ("lora", 8)), (("fist", 8), ("lora_xs", 8))]
    comps = run_comparisons(rs, ["rte", "cola"])
    assert {c.level for c in comps} == {"task", "suite"} and all(not math.isnan(c.holm_pvalue) for c in comps if c.level == "task")


def test_load_records_reads_best_epoch_and_benchmarks(tmp_path):
    glue = {"method": "fist", "rank": 8, "seed": 42, "task": "rte", "metric": "accuracy",
            "best": 0.8, "final": 0.75, "adapter_params": 6144, "head_params": 10}
    bench = {"method": "fist", "rank": 8, "seed": 42, "task": "cs", "adapter_params": 1, "head_params": 0,
             "benchmarks": {"boolq": {"accuracy": 0.6}, "piqa": {"accuracy": 0.7}}}
    for i, m in enumerate((glue, bench)):
        p = tmp_path / "runs" / f"x{i}" / "metrics.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(m))
    recs = {r.task: r for r in load_records(tmp_path)}
    assert recs["rte"].value == pytest.approx(80) and recs["boolq"].value == pytest.approx(60)
    assert {r.task: r.value for r in load_records(tmp_path, "final")}["rte"] == pytest.approx(75)
