"""Settings used by the API, the pipeline and the tests: file locations and fixed numbers.

Everything here comes from the notebooks, so the service and the notebooks always agree.
"""
from pathlib import Path

# the project folder (the one that contains src/, models/, data/ ...)
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

# the files the notebooks saved
PII_FILE = MODEL_DIR / "07_pii_patterns.joblib"        # notebook 07
TOPIC_FILE = MODEL_DIR / "07_topic_model.joblib"       # notebook 07
B4_FILE = MODEL_DIR / "04_champion_bundle.joblib"      # notebook 04 (the fast fallback model)
CHAMPION_FILE = MODEL_DIR / "08_champion_roberta.joblib"   # notebook 08f (calibration + threshold for RoBERTa)
ROBERTA_FOLDER = MODEL_DIR / "roberta_relief"          # notebook 08b (the fine-tuned model)

# the same numbers the notebooks used
MAX_CHARS = 1800        # text is cut to this many characters before RoBERTa (notebooks 08a and 08d)
MAX_LENGTH = 256        # RoBERTa reads at most this many word-pieces
MIN_NARRATIVE_CHARS = 20      # shorter text is rejected as "not a real complaint"
MAX_NARRATIVE_CHARS = 20000   # longer text is rejected (protects the service)

# this wording is attached to every answer: the model is a triage tool
NOTICE = ("Triage aid only. A low score never means a customer gets nothing: "
          "every complaint still gets a human review. Do not use it to deny or close complaints.")

# ---- drift monitoring (step 6) -------------------------------------------------------------
# Thresholds are assumptions, not data. PSI 0.10 / 0.25 is the usual rule of thumb; the others are judgement calls you can change.
OPENING_CHARS = 150          # same as notebook 07b
MIN_REPEATS = 5              # same as notebook 07b
MIN_WINDOW_COMPLAINTS = 100  # windows smaller than this are not judged

PSI_WARN = 0.10
PSI_ALERT = 0.25
FLAG_SHARE_WARN = 0.05       # absolute change in the share flagged review_first
FLAG_SHARE_ALERT = 0.10
REPEATED_WARN = 0.05         # absolute change in the share of templated complaints
REPEATED_ALERT = 0.10
LENGTH_WARN = 0.15           # relative change in median word count
LENGTH_ALERT = 0.30
RELIEF_WARN = 0.03           # absolute change in the real relief rate
RELIEF_ALERT = 0.06

DRIFT_DIR = ROOT / "reports" / "drift"
REFERENCE_FILE = DRIFT_DIR / "reference_snapshot.json"
HISTORY_FILE = DRIFT_DIR / "drift_history.csv"
SCORED_ROWS_PER_WINDOW = 300  # how many complaints per window are scored by the model (the rest are only counted)
