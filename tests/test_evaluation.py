"""GLUE metrics, best-epoch selection, and the GSM8K/MATH answer extractor."""

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

from fist_lora.evaluation.answer_extraction import extract_answer, is_correct
from fist_lora.evaluation.glue import best_and_final, epoch_metrics, glue_metrics


def test_classification_metrics():
    labels = np.array([0, 1, 1, 0, 1, 0])
    logits = np.array([[2, 1], [0, 3], [1, 0], [3, 0], [0, 1], [0, 2]], dtype=float)
    preds = logits.argmax(-1)
    assert glue_metrics(logits, labels, "accuracy") == {"accuracy": pytest.approx(4 / 6)}
    assert glue_metrics(logits, labels, "f1")["f1"] == pytest.approx(f1_score(labels, preds))
    assert glue_metrics(logits, labels, "matthews_correlation")["matthews_correlation"] == pytest.approx(
        matthews_corrcoef(labels, preds))


def test_regression_metric_uses_raw_predictions():
    labels = np.array([0.5, 2.0, 3.5, 4.0])
    logits = np.array([[0.1], [2.2], [3.0], [4.9]])
    assert glue_metrics(logits, labels, "spearmanr")["spearmanr"] == pytest.approx(spearmanr(logits[:, 0], labels)[0])


def test_best_epoch_not_final_epoch():
    history = [
        {"loss": 0.7, "step": 10, "epoch": 0.5},
        {"eval_f1": 0.80, "epoch": 1.0},
        {"eval_f1": 0.86, "epoch": 2.0},
        {"eval_f1": 0.83, "epoch": 3.0},
        {"train_loss": 0.4, "epoch": 3.0},
    ]
    per_epoch = epoch_metrics(history, "f1")
    assert [p["epoch"] for p in per_epoch] == [1.0, 2.0, 3.0]
    assert best_and_final(per_epoch) == {"best": 0.86, "best_epoch": 2.0, "final": 0.83}
    with pytest.raises(ValueError):
        best_and_final([])


@pytest.mark.parametrize("text,expected", [
    ("Natalia sold 48/2 = 24 clips.\n#### 72", "72"),
    ("#### 1,234", "1234"),
    ("#### -5", "-5"),
    ("so the answer is $\\boxed{17}$ and #### 18", "18"),  # #### takes priority
    ("The answer is $\\boxed{\\frac{1}{2}}$", "\\frac{1"),  # truncated at the first '}'
    ("We get \\boxed{ 3 } overall", "3"),
    ("First 12 apples, then 7 more. The answer is: 19", "19"),  # trailing numeral fallback
    ("no numbers here", ""),
])
def test_answer_extraction_rules(text, expected):
    assert extract_answer(text) == expected


def test_same_extractor_applied_to_reference_and_generation():
    assert is_correct("... The answer is: 72", "work\n#### 72")
    assert is_correct("I think $\\boxed{5}$.", "$\\boxed{5}$")
    assert not is_correct("The answer is: 71", "#### 72")
