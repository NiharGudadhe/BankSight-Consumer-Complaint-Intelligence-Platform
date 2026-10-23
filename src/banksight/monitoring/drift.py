"""Drift monitoring: is today's data still like the data the model was checked on?

We compare a recent window of complaints with a REFERENCE window (the validation period) on seven simple measures.
Every measure gets a status: ok, warn or alert. All thresholds are in config.py so they are easy to explain and to change.
"""
import hashlib

import numpy as np
import pandas as pd

from banksight import config


def psi(reference_shares, current_shares):
    """Population Stability Index: how different two sets of shares are. 0 = identical.

    Rule of thumb used in credit-risk teams: under 0.10 stable, 0.10 to 0.25 some change, over 0.25 large change.
    """
    reference = np.clip(np.asarray(reference_shares, dtype=float), 1e-4, None)
    current = np.clip(np.asarray(current_shares, dtype=float), 1e-4, None)
    reference = reference / reference.sum()
    current = current / current.sum()
    return float(np.sum((current - reference) * np.log(current / reference)))


def shares_of(values, categories):
    """Share of `values` in each category. Anything not in `categories` is counted in one extra last bucket ('new')."""
    values = pd.Series(list(values))
    known = values.where(values.isin(categories), "__new__")
    counts = known.value_counts()
    return [float(counts.get(category, 0) / len(values)) for category in list(categories) + ["__new__"]]


def opening_keys(masked_texts):
    """A short fingerprint of the first 150 characters of each complaint (the 'opening'), as in notebook 07b."""
    return pd.Series([hashlib.md5(text[:config.OPENING_CHARS].encode()).hexdigest()[:10] for text in masked_texts])


def repeated_keys(keys):
    """The set of openings that appear 5 or more times in this window: template or campaign wording."""
    counts = keys.value_counts()
    return set(counts[counts >= config.MIN_REPEATS].index)


def biggest_shifts(categories, reference_shares, current_shares, names=None, top=3):
    """Describe the categories whose share changed most, e.g. 'Credit reporting 25% -> 48% (+23 points)'.

    The last share belongs to the extra 'new' bucket (categories not seen in the reference).
    """
    labels = [str(category) for category in categories] + ["new (unseen in the reference)"]
    if names is not None:
        labels = [names.get(str(category), str(category)) for category in categories] + ["new (unseen in the reference)"]

    changes = np.asarray(current_shares) - np.asarray(reference_shares)
    parts = []
    for index in np.argsort(-np.abs(changes))[:top]:
        parts.append(f"{labels[index]} {reference_shares[index]:.0%} -> {current_shares[index]:.0%} ({changes[index] * 100:+.0f} points)")
    return "; ".join(parts)


def summarise_window(full_window, scored_sample, reference=None):
    """Turn a window of complaints into one dictionary of numbers (the 'snapshot').

    full_window: every complaint of the window, with columns product_group, topic_id, words, text, got_relief.
    scored_sample: the complaints we actually scored with the model, with columns probability, flagged.
    reference: None when building the reference; otherwise the saved reference snapshot.

    Template wording is measured against the reference's KNOWN templates, so a one-month window and a four-month
    reference are compared fairly. Repeats that are new in this window are reported separately as 'new campaigns'.
    """
    keys = opening_keys(full_window["text"])
    repeated_in_window = repeated_keys(keys)

    snapshot = {
        "n_complaints": int(len(full_window)),
        "n_scored": int(len(scored_sample)),
        "median_words": float(full_window["words"].median()),
        "mean_probability": float(scored_sample["probability"].mean()),
        "flag_share": float(scored_sample["flagged"].mean()),
    }

    if reference is None:
        snapshot["known_template_keys"] = sorted(repeated_in_window)
        snapshot["repeated_share"] = float(keys.isin(repeated_in_window).mean())
        snapshot["new_campaign_share"] = 0.0
    else:
        known = set(reference["known_template_keys"])
        snapshot["repeated_share"] = float(keys.isin(known).mean())
        snapshot["new_campaign_share"] = float(keys.isin(repeated_in_window - known).mean())

    labelled = full_window["got_relief"].dropna()
    snapshot["n_labelled"] = int(len(labelled))
    snapshot["relief_rate"] = float(labelled.mean()) if len(labelled) > 0 else None
    return snapshot


def add_distribution_parts(snapshot, full_window, scored_sample, reference=None):
    """Add the three distributions (topics, products, scores) to a snapshot.

    For the reference window (reference=None) the categories and score edges are decided here and saved.
    For a later window they are copied from the reference, so both windows are measured on the same ruler.
    """
    if reference is None:
        topic_categories = sorted(full_window["topic_id"].unique().tolist())
        product_categories = sorted(full_window["product_group"].unique().tolist())
        score_edges = np.quantile(scored_sample["probability"], np.linspace(0.1, 0.9, 9)).tolist()
    else:
        topic_categories = reference["topic_categories"]
        product_categories = reference["product_categories"]
        score_edges = reference["score_edges"]

    score_bins = np.digitize(scored_sample["probability"], score_edges)
    score_counts = np.bincount(score_bins, minlength=len(score_edges) + 1)

    snapshot["topic_categories"] = topic_categories
    snapshot["topic_shares"] = shares_of(full_window["topic_id"], topic_categories)
    snapshot["product_categories"] = product_categories
    snapshot["product_shares"] = shares_of(full_window["product_group"], product_categories)
    snapshot["score_edges"] = score_edges
    snapshot["score_shares"] = (score_counts / score_counts.sum()).tolist()
    return snapshot


def status_of(value, warn, alert):
    """ok / warn / alert from one number and two thresholds."""
    if value >= alert:
        return "alert"
    if value >= warn:
        return "warn"
    return "ok"


def compare_to_reference(reference, current, window_label):
    """Compare a window's snapshot with the reference snapshot. Returns a table, one row per measure."""
    rows = []
    too_small = current["n_complaints"] < config.MIN_WINDOW_COMPLAINTS

    def add_row(measure, reference_value, current_value, change, warn, alert, meaning, details=""):
        if too_small:
            status = "skipped (window too small)"
        elif change is None:
            status = "skipped (no labels)"
        else:
            status = status_of(abs(change), warn, alert)
        rows.append({"window": window_label, "measure": measure, "reference": reference_value, "current": current_value,
                     "change": change, "warn_at": warn, "alert_at": alert, "status": status, "meaning": meaning, "details": details})

    add_row("topic_psi", None, None, psi(reference["topic_shares"], current["topic_shares"]),
            config.PSI_WARN, config.PSI_ALERT, "complaint themes are shifting",
            biggest_shifts(reference["topic_categories"], reference["topic_shares"], current["topic_shares"], reference.get("topic_names")))
    add_row("product_psi", None, None, psi(reference["product_shares"], current["product_shares"]),
            config.PSI_WARN, config.PSI_ALERT, "the product mix is shifting",
            biggest_shifts(reference["product_categories"], reference["product_shares"], current["product_shares"]))
    add_row("score_psi", None, None, psi(reference["score_shares"], current["score_shares"]),
            config.PSI_WARN, config.PSI_ALERT, "the model's scores are shifting")
    add_row("flag_share", reference["flag_share"], current["flag_share"], current["flag_share"] - reference["flag_share"],
            config.FLAG_SHARE_WARN, config.FLAG_SHARE_ALERT, "share of complaints flagged review_first has moved")
    add_row("repeated_share", reference["repeated_share"], current["repeated_share"], current["repeated_share"] - reference["repeated_share"],
            config.REPEATED_WARN, config.REPEATED_ALERT, "the share of complaints using KNOWN template wording has moved")
    add_row("new_campaign_share", reference["new_campaign_share"], current["new_campaign_share"], current["new_campaign_share"] - reference["new_campaign_share"],
            config.REPEATED_WARN, config.REPEATED_ALERT, "NEW template wording (a new campaign) is appearing")
    add_row("median_words", reference["median_words"], current["median_words"],
            current["median_words"] / reference["median_words"] - 1, config.LENGTH_WARN, config.LENGTH_ALERT,
            "complaints are getting longer or shorter")

    relief_change = None
    if reference["relief_rate"] is not None and current["relief_rate"] is not None and current["n_labelled"] >= config.MIN_WINDOW_COMPLAINTS:
        relief_change = current["relief_rate"] - reference["relief_rate"]
    add_row("relief_rate", reference["relief_rate"], current["relief_rate"], relief_change,
            config.RELIEF_WARN, config.RELIEF_ALERT, "the real relief rate has moved (outcomes arrive late; for reporting only)")

    return pd.DataFrame(rows)


def overall_status(report):
    """The worst status in the table: alert, then warn, then ok."""
    statuses = report["status"].tolist()
    if "alert" in statuses:
        return "alert"
    if "warn" in statuses:
        return "warn"
    return "ok"


def advice_for(report):
    """Plain-English next steps for every measure that is not ok, with what changed most."""
    advice = []
    for _, row in report.iterrows():
        if row["status"] not in ("warn", "alert"):
            continue
        if row["details"]:
            advice.append(f"{row['measure']} biggest moves: {row['details']}")
        if row["measure"] in ("score_psi", "flag_share", "relief_rate"):
            advice.append(f"{row['measure']} is {row['status']}: check calibration on recent labelled complaints; consider re-calibrating or retraining.")
        elif row["measure"] == "topic_psi":
            advice.append(f"topic_psi is {row['status']}: new themes may be appearing; re-run notebook 07 on recent data.")
        elif row["measure"] in ("repeated_share", "new_campaign_share"):
            advice.append(f"{row['measure']} is {row['status']}: template wording has changed; re-run notebook 07b and check performance on non-template complaints.")
        else:
            advice.append(f"{row['measure']} is {row['status']}: {row['meaning']}.")
    return advice
