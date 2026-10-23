"""Tests for scoring. They use the real saved models, so they are skipped if the models folder is empty."""
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from banksight import config
from banksight.ml.calibration import apply_calibration
from banksight.ml.loader import load_models
from banksight.ml.scoring import score_complaint

NARRATIVE = "I was charged a fee I did not expect and the company has not answered my letters for two months."
@pytest.fixture(scope="module")
def b4_models():
    """Load the models once for all tests in this file (loading is the slow part)."""
    return load_models(use_roberta=False)


needs_models = pytest.mark.skipif(not (config.B4_FILE.exists() and config.TOPIC_FILE.exists() and config.PII_FILE.exists()),
                                  reason="notebook models not found in models/")


def test_isotonic_calibration_stays_between_0_and_1():
    scores = np.linspace(0, 1, 50)
    outcomes = (scores > 0.5).astype(int)
    calibration = {"method": "isotonic", "isotonic": IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(scores, outcomes)}
    result = apply_calibration(calibration, [0.1, 0.5, 0.9])
    assert ((result >= 0) & (result <= 1)).all()


@needs_models
def test_b4_fallback_returns_a_valid_answer(b4_models):
    models = b4_models
    product = models["b4"]["product_cats"][0]
    answer = score_complaint(models, product, "Fees", NARRATIVE)
    assert 0 <= answer["relief_probability"] <= 1
    assert answer["model_used"] == "b4"
    assert answer["review_priority"] in ("review_first", "standard_review")


@needs_models
def test_short_text_is_rejected(b4_models):
    models = b4_models
    with pytest.raises(ValueError):
        score_complaint(models, models["b4"]["product_cats"][0], "Fees", "too short")


@needs_models
def test_answer_never_contains_a_denial_and_carries_the_notice(b4_models):
    models = b4_models
    answer = score_complaint(models, models["b4"]["product_cats"][0], None, NARRATIVE)
    assert "deny" not in answer["review_priority"].lower()
    assert "triage" in answer["notice"].lower()


@pytest.mark.skipif(not (config.ROBERTA_FOLDER.exists() and config.CHAMPION_FILE.exists()), reason="RoBERTa files not found")
def test_roberta_path_works_when_files_exist():
    models = load_models(use_roberta=True)
    if models["roberta"] is None:
        pytest.skip(models["roberta_status"])
    answer = score_complaint(models, models["b4"]["product_cats"][0], "Fees", NARRATIVE)
    assert answer["model_used"] == "roberta"
    assert 0 <= answer["relief_probability"] <= 1
