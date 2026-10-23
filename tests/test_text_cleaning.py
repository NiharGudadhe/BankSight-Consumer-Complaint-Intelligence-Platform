"""Tests for the text cleaning functions. They need no model files."""
from banksight.nlp.text_cleaning import build_roberta_text, mask_for_b4, redact_pii

PATTERNS = {"email": r"[\w.]+@[\w.]+", "phone": r"\d{3}-\d{3}-\d{4}"}


def test_redact_pii_hides_email_and_phone():
    text = redact_pii("write to jo@mail.com or call 555-123-4567", PATTERNS)
    assert "jo@mail.com" not in text
    assert "555-123-4567" not in text
    assert "[EMAIL]" in text and "[PHONE]" in text


def test_mask_for_b4_replaces_amounts_dates_and_x_masks():
    text = mask_for_b4("They took $1,200.50 on XX/XX/2024 from account XXXX")
    assert "amtmask" in text and "datemask" in text and "xmask" in text
    assert "$" not in text


def test_build_roberta_text_has_three_parts_and_is_cut():
    text = build_roberta_text("Credit card", "Fees", "a" * 5000, PATTERNS, 100)
    assert text.startswith("Credit card | Fees | ")
    assert len(text) == len("Credit card | Fees | ") + 100
