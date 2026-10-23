"""Text cleaning: hide personal data, and the masking used for the B4 model.

Each function is a small version of a notebook cell, so the service treats text exactly as the notebooks did.
"""
import re


def redact_pii(text, patterns):
    """Replace emails, phone numbers and similar patterns with tags like [EMAIL].

    Same as notebooks 08a and 08d. `patterns` is the dictionary saved by notebook 07 (name -> pattern).
    """
    for name, pattern in patterns.items():
        text = re.sub(pattern, f"[{name.upper()}]", text)
    return text


def mask_for_b4(text):
    """Replace dollar amounts, dates and XXXX masks with fixed tags (notebooks 04, 07, 08c and 09)."""
    text = re.sub(r"\{\$[\d,\.]+\}|\$\s?[\d,]+(?:\.\d+)?", " amtmask ", text)
    text = re.sub(r"\bXX/XX/(?:XXXX|\d{2,4})\b|\bXX/\d{2}/\d{2,4}\b", " datemask ", text)
    text = re.sub(r"\bX{2,}\b", " xmask ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_roberta_text(product_group, issue, narrative, patterns, max_chars):
    """Build the one line RoBERTa reads: `product | issue | narrative` (notebooks 08a and 08d)."""
    redacted = redact_pii(narrative, patterns)[:max_chars]
    return f"{product_group} | {issue} | {redacted}"
