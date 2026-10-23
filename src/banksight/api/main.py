import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from banksight import config
from banksight.api.schemas import BatchIn, BatchOut, ComplaintIn, PredictionOut
from banksight.ml.loader import load_models
from banksight.ml.scoring import score_complaint


@asynccontextmanager
async def lifespan(app):
    """Runs once when the service starts: load the models into memory so each request is fast."""
    use_roberta = os.getenv("BANKSIGHT_USE_ROBERTA", "1") == "1"    # set it to 0 to run on the light B4 model only
    app.state.models = load_models(use_roberta=use_roberta)
    yield


app = FastAPI(
    title="BankSight Complaint Triage API",
    version="1.0.0",
    description="Ranks consumer complaints by how likely they are to end with relief. " + config.NOTICE,
    lifespan=lifespan,
)


def check_product(product_group, models):
    """Reject product groups the models have never seen, with a clear message (HTTP 422)."""
    known_products = models["b4"]["product_cats"]
    if product_group not in known_products:
        raise HTTPException(status_code=422, detail=f"Unknown product_group '{product_group}'. Allowed: {known_products}")


@app.get("/health")
def health():
    """Is the service up, and which model is answering?"""
    models = app.state.models
    return {"status": "ok", "roberta_status": models["roberta_status"],
            "active_model": "roberta" if models["roberta"] is not None else "b4"}


@app.get("/products")
def products():
    """The product groups the service accepts."""
    return {"product_groups": app.state.models["b4"]["product_cats"]}


@app.post("/predict", response_model=PredictionOut)
def predict(complaint: ComplaintIn):
    """Score one complaint."""
    models = app.state.models
    check_product(complaint.product_group, models)
    try:
        return score_complaint(models, complaint.product_group, complaint.issue, complaint.narrative)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(batch: BatchIn):
    """Score up to 50 complaints in one request. If one is invalid, nothing is scored and you get a 422."""
    models = app.state.models
    for complaint in batch.complaints:
        check_product(complaint.product_group, models)

    results = []
    try:
        for complaint in batch.complaints:
            results.append(score_complaint(models, complaint.product_group, complaint.issue, complaint.narrative))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    return {"results": results}
