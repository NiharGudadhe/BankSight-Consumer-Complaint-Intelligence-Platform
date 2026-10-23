"""Score a complaint: RoBERTa first, B4 as the fallback. Never produces a denial."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from banksight import config
from banksight.ml.calibration import apply_calibration
from banksight.nlp.text_cleaning import build_roberta_text, mask_for_b4
from banksight.nlp.topics import find_topic


def check_input(narrative):
    """Reject text that is empty, too short or too long. The API turns this error into a 422 reply."""
    if narrative is None or len(narrative.strip()) < config.MIN_NARRATIVE_CHARS:
        raise ValueError(f"The complaint text must have at least {config.MIN_NARRATIVE_CHARS} characters.")
    if len(narrative) > config.MAX_NARRATIVE_CHARS:
        raise ValueError(f"The complaint text must have at most {config.MAX_NARRATIVE_CHARS} characters.")


def raw_score_b4(models, product_group, issue, masked_text):
    """B4: TF-IDF text + product + issue, then logistic regression (notebook 04). Returns a raw probability."""
    b4 = models["b4"]
    text_part = b4["tfidf"].transform([masked_text])
    category_part = b4["ohe"].transform(pd.DataFrame({"product_group": [product_group], "issue": [issue]}))
    features = sp.hstack([text_part, category_part]).tocsr()
    return float(b4["b4_model"].predict_proba(features)[0, 1])


def raw_score_roberta(models, product_group, issue, narrative):
    """RoBERTa: reads `product | issue | narrative` and returns a raw probability (notebooks 08b and 08e)."""
    import torch   # imported here so B4-only use needs no torch

    text = build_roberta_text(product_group, issue, narrative, models["pii_patterns"], config.MAX_CHARS)
    batch = models["tokenizer"]([text], truncation=True, max_length=config.MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        score = models["roberta"](**batch).logits.squeeze(-1)
    return float(torch.sigmoid(score.float()).item())


def score_complaint(models, product_group, issue, narrative):
    """Score one complaint and return a plain dictionary (this is what the API sends back)."""
    check_input(narrative)
    issue = issue if issue else "Unknown"
    masked_text = mask_for_b4(narrative)
    topic_number, topic_label = find_topic(masked_text, models["topics"])

    model_used = "b4"
    raw_probability = None

    if models["roberta"] is not None:
        try:
            raw_probability = raw_score_roberta(models, product_group, issue, narrative)
            calibration = models["champion"]["calibration"]
            threshold = models["champion"]["threshold"]
            model_used = "roberta"
        except Exception:
            raw_probability = None    # anything goes wrong with RoBERTa -> fall back to B4

    if raw_probability is None:
        raw_probability = raw_score_b4(models, product_group, issue, masked_text)
        calibration = models["b4"]["calibration"]
        threshold = models["b4"]["threshold"]
        model_used = "b4" if models["roberta"] is None else "b4_fallback"

    probability = float(apply_calibration(calibration, [raw_probability])[0])
    review_first = probability >= threshold

    return {
        "relief_probability": round(probability, 4),
        "review_priority": "review_first" if review_first else "standard_review",
        "threshold": round(float(threshold), 4),
        "model_used": model_used,
        "topic_id": topic_number,
        "topic_label": topic_label,
        "notice": config.NOTICE,
    }


def score_complaints(models, complaints):
    """Score a list of complaints (each a dictionary with product_group, issue, narrative), one by one."""
    results = []
    for complaint in complaints:
        results.append(score_complaint(models, complaint["product_group"], complaint.get("issue"), complaint["narrative"]))
    return results
