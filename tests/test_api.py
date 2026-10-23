"""Tests for the API. They start the app in-process (no server needed) and use the light B4 model so they are fast."""
import os

os.environ["BANKSIGHT_USE_ROBERTA"] = "0"    # must be set before the app starts

import pytest
from fastapi.testclient import TestClient

from banksight import config
from banksight.api.main import app

NARRATIVE = "I was charged a fee I did not expect and the company has not answered my letters for two months."
pytestmark = pytest.mark.skipif(not (config.B4_FILE.exists() and config.TOPIC_FILE.exists() and config.PII_FILE.exists()),
                                reason="notebook models not found in models/")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:    # the `with` makes the models load once
        yield test_client


@pytest.fixture(scope="module")
def product(client):
    return client.get("/products").json()["product_groups"][0]


def test_health_says_ok(client):
    answer = client.get("/health").json()
    assert answer["status"] == "ok"
    assert answer["active_model"] == "b4"


def test_predict_returns_a_valid_answer(client, product):
    response = client.post("/predict", json={"product_group": product, "issue": "Fees", "narrative": NARRATIVE})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["relief_probability"] <= 1
    assert body["review_priority"] in ("review_first", "standard_review")
    assert "triage" in body["notice"].lower()


def test_predict_works_without_an_issue(client, product):
    response = client.post("/predict", json={"product_group": product, "narrative": NARRATIVE})
    assert response.status_code == 200


def test_unknown_product_is_rejected(client):
    response = client.post("/predict", json={"product_group": "Space tourism", "narrative": NARRATIVE})
    assert response.status_code == 422


def test_short_narrative_is_rejected(client, product):
    response = client.post("/predict", json={"product_group": product, "narrative": "too short"})
    assert response.status_code == 422


def test_whitespace_only_narrative_is_rejected(client, product):
    response = client.post("/predict", json={"product_group": product, "narrative": " " * 50})
    assert response.status_code == 422


def test_batch_scores_every_complaint(client, product):
    complaint = {"product_group": product, "issue": "Fees", "narrative": NARRATIVE}
    response = client.post("/predict/batch", json={"complaints": [complaint, complaint]})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_batch_over_50_is_rejected(client, product):
    complaint = {"product_group": product, "narrative": NARRATIVE}
    response = client.post("/predict/batch", json={"complaints": [complaint] * 51})
    assert response.status_code == 422
