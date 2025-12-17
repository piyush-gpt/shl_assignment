from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.retriever import recommend as recommend_fn


app = FastAPI(title="SHL Assessment Recommendation API")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the API")


class RecommendRequest(BaseModel):
    query: str = Field(..., description="Job description or natural language hiring query")


class RecommendedAssessment(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: Optional[int]
    remote_support: str
    test_type: List[str]


class RecommendResponse(BaseModel):
    recommended_assessments: List[RecommendedAssessment]


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health Check Endpoint
    Matches the spec:
    {
      "status": "healthy"
    }
    """
    return HealthResponse(status="healthy")


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest) -> RecommendResponse:
    """
    Assessment Recommendation Endpoint

    Request:
    {
      "query": "JD/query in string"
    }

    Response:
    {
      "recommended_assessments": [
        {
          "url": "valid URL in string",
          "name": "name in string",
          "adaptive_support": "Yes/No",
          "description": "description in string",
          "duration": 60,
          "remote_support": "Yes/No",
          "test_type": ["list of string"]
        }
      ]
    }
    """
    raw = recommend_fn(payload.query)
    return RecommendResponse(**raw)


# For local testing:
#   uvicorn api:app --reload


