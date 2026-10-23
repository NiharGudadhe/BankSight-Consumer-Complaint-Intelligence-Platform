"""Load everything the notebooks saved, once, when the service starts."""
import joblib

from banksight import config


def load_models(use_roberta=True):
    """Return one dictionary with all models. If RoBERTa is missing or not wanted, the dictionary has `roberta` = None
    and the service uses the B4 fallback."""
    for path in (config.PII_FILE, config.TOPIC_FILE, config.B4_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name} in models/. Run the notebooks first.")

    models = {
        "pii_patterns": joblib.load(config.PII_FILE),
        "topics": joblib.load(config.TOPIC_FILE),
        "b4": joblib.load(config.B4_FILE),
        "champion": None,
        "roberta": None,
        "tokenizer": None,
        "roberta_status": "not requested",
    }

    if not use_roberta:
        return models

    if not config.CHAMPION_FILE.exists() or not config.ROBERTA_FOLDER.exists():
        models["roberta_status"] = "files missing, using the B4 fallback"
        return models

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer   # imported here so B4-only use needs no torch
        models["champion"] = joblib.load(config.CHAMPION_FILE)
        models["tokenizer"] = AutoTokenizer.from_pretrained(config.ROBERTA_FOLDER)
        models["roberta"] = AutoModelForSequenceClassification.from_pretrained(config.ROBERTA_FOLDER)
        models["roberta"].eval()
        models["roberta_status"] = "loaded"
    except Exception as error:
        models["roberta_status"] = f"could not load ({type(error).__name__}), using the B4 fallback"

    return models
