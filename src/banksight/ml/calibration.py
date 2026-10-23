"""Turn raw model scores into honest probabilities (notebooks 04 and 08f)."""
import numpy as np


def apply_calibration(calibration, raw_scores):
    """Apply the saved calibrator. `calibration` has the keys method, platt and isotonic.

    Platt scaling is a small logistic curve on the log-odds; isotonic is a step function that only goes up.
    """
    raw_scores = np.asarray(raw_scores, dtype=float)
    if calibration["method"] == "platt":
        clipped = np.clip(raw_scores, 1e-4, 1 - 1e-4)
        log_odds = np.log(clipped / (1 - clipped))
        return calibration["platt"].predict_proba(log_odds.reshape(-1, 1))[:, 1]
    if calibration["method"] == "isotonic":
        return calibration["isotonic"].predict(raw_scores)
    return raw_scores
