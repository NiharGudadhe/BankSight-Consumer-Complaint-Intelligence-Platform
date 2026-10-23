"""The shapes of what goes into and comes out of the API. FastAPI checks every request against these."""
from typing import List, Optional

from pydantic import BaseModel, Field

from banksight import config


class ComplaintIn(BaseModel):
    """One complaint to score."""
    product_group: str = Field(..., description="Product group, e.g. 'Credit card or prepaid card'. See GET /products.")
    issue: Optional[str] = Field(None, description="The issue chosen on the complaint form, if known.")
    narrative: str = Field(..., min_length=config.MIN_NARRATIVE_CHARS, max_length=config.MAX_NARRATIVE_CHARS,
                           description="The complaint text written by the customer.")


class PredictionOut(BaseModel):
    """The answer for one complaint. There is deliberately no 'deny' field."""
    relief_probability: float
    review_priority: str      # 'review_first' or 'standard_review'
    threshold: float
    model_used: str           # 'roberta', 'b4' or 'b4_fallback'
    topic_id: int
    topic_label: str
    notice: str


class BatchIn(BaseModel):
    """Up to 50 complaints in one request."""
    complaints: List[ComplaintIn] = Field(..., min_length=1, max_length=50)


class BatchOut(BaseModel):
    results: List[PredictionOut]
