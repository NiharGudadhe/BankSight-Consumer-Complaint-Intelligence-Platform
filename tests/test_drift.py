"""Tests for the drift functions. They need no model files and no database."""
import pandas as pd

from banksight.monitoring.drift import (add_distribution_parts, biggest_shifts, compare_to_reference, opening_keys, overall_status, psi,
                                        repeated_keys, shares_of, status_of, summarise_window)


def make_window(topics, products, probabilities, relief):
    n = len(topics)
    full = pd.DataFrame({"product_group": products, "topic_id": topics, "words": [100] * n,
                         "text": [f"complaint number {i} " * 5 for i in range(n)], "got_relief": relief})
    scored = pd.DataFrame({"probability": probabilities, "flagged": [p >= 0.25 for p in probabilities]})
    return full, scored


def test_psi_is_zero_for_identical_shares_and_grows_with_change():
    assert psi([0.5, 0.5], [0.5, 0.5]) < 1e-9
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0.25


def test_shares_put_unseen_categories_in_the_new_bucket():
    result = shares_of(["a", "a", "b", "z"], ["a", "b"])
    assert result == [0.5, 0.25, 0.25]


def test_repeated_keys_finds_templates():
    texts = ["same opening text"] * 6 + ["something unique %d" % i for i in range(4)]
    keys = opening_keys(texts)
    assert keys.isin(repeated_keys(keys)).mean() == 0.6


def test_biggest_shifts_names_the_category_that_moved():
    text = biggest_shifts(["A", "B"], [0.5, 0.5, 0.0], [0.2, 0.8, 0.0], top=1)
    assert "B" in text or "A" in text
    assert "points" in text


def test_known_templates_and_new_campaigns_are_told_apart():
    n = 200
    reference_full, scored = make_window([0] * n, ["A"] * n, [0.2] * n, [0.0] * n)
    reference_full["text"] = ["known template wording"] * 100 + [f"unique story {i}" for i in range(100)]
    reference = add_distribution_parts(summarise_window(reference_full, scored), reference_full, scored)
    assert abs(reference["repeated_share"] - 0.5) < 1e-9

    window = reference_full.copy()
    window["text"] = ["known template wording"] * 20 + ["a brand new campaign text"] * 40 + [f"other story {i}" for i in range(140)]
    snapshot = summarise_window(window, scored, reference)
    assert abs(snapshot["repeated_share"] - 0.1) < 1e-9
    assert abs(snapshot["new_campaign_share"] - 0.2) < 1e-9


def test_status_levels():
    assert status_of(0.05, 0.10, 0.25) == "ok"
    assert status_of(0.15, 0.10, 0.25) == "warn"
    assert status_of(0.30, 0.10, 0.25) == "alert"


def test_same_window_is_ok_and_a_shifted_window_alerts():
    n = 400
    topics = [0, 1] * (n // 2)
    products = ["A", "B"] * (n // 2)
    probabilities = [0.1 + 0.002 * i for i in range(n)]
    relief = [0.0, 1.0] * (n // 2)

    full, scored = make_window(topics, products, probabilities, relief)
    reference = add_distribution_parts(summarise_window(full, scored), full, scored)

    same = add_distribution_parts(summarise_window(full, scored, reference), full, scored, reference)
    assert overall_status(compare_to_reference(reference, same, "same")) == "ok"

    shifted_full, shifted_scored = make_window([0] * n, ["A"] * n, [0.9] * n, [1.0] * n)
    shifted = add_distribution_parts(summarise_window(shifted_full, shifted_scored, reference), shifted_full, shifted_scored, reference)
    assert overall_status(compare_to_reference(reference, shifted, "shifted")) == "alert"


def test_small_window_is_not_judged():
    full, scored = make_window([0, 1] * 5, ["A", "B"] * 5, [0.2] * 10, [0.0, 1.0] * 5)
    reference = add_distribution_parts(summarise_window(full, scored), full, scored)
    report = compare_to_reference(reference, reference, "tiny")
    assert overall_status(report) == "ok"
    assert report["status"].str.startswith("skipped").all()
