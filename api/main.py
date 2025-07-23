# api/main.py
"""
Main FastAPI Application for the Agency Monitor Project - Production Version

This API serves all data, predictions, and explanations, integrating with the
underlying AI services and database. It includes security, rate limiting, and
robust error handling.
"""
import os
import logging
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Security, status, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# --- Local Imports ---
from api.ai_integration import ai_service
from api.database import get_db, IQANote, save_iqa_note, get_iqa_notes

# --- Configuration & Initialization ---
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SECRET_API_KEY = os.getenv("SECRET_API_KEY")
if not SECRET_API_KEY:
    logger.warning("SECRET_API_KEY environment variable not set. Using insecure default.")
    SECRET_API_KEY = "a_very_secret_key_for_dev"

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

app = FastAPI(
    title="Agency Monitor API",
    description="API for serving Agency Calculus data, predictions, and explanations.",
    version="2.0.0"
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# --- Security Dependency ---
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == SECRET_API_KEY:
        return api_key
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

# --- Pydantic Models ---
class ExplainRequest(BaseModel):
    year: int = Field(..., example=2024, description="The year for which to explain the forecast (e.g., explain 2024 using 2023 data).")

class PolicyRequest(BaseModel):
    policy_text: str = Field(..., min_length=50, example="A bill to promote economic growth...")
    policy_name: Optional[str] = "Untitled Policy"
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    normalization: str = Field("density", pattern="^(density|magnitude)$")

class IQASubmitRequest(BaseModel):
    analyst: str = Field(..., example="Dr. Smith")
    note: str = Field(..., min_length=10, example="Observed increasing social unrest.")
    year: int = Field(..., example=2023)
    category: str = Field("Social", example="Social")

# --- API Endpoints ---
@app.get("/health", summary="Check API health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": app.version}

@app.get("/api/v1/country/{country_code}/forecast", tags=["Predictions"], dependencies=)
@limiter.limit("50/minute")
async def get_forecast_endpoint(request: Request, country_code: str, steps: int = 10, weighting: str = "Framework Average"):
    try:
        forecast = ai_service.generate_forecast(country_code.upper(), steps, weighting)
        return forecast
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating forecast for {country_code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate forecast")

@app.post("/api/v1/country/{country_code}/explain", tags=["Explanations"], dependencies=)
@limiter.limit("50/minute")
async def get_explanation_endpoint(request: Request, country_code: str, req_body: ExplainRequest):
    try:
        explanation = ai_service.explain_forecast(country_code.upper(), req_body.year)
        return explanation
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating explanation for {country_code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate explanation")

@app.post("/api/v1/analysis/score-policy", tags=["NLP Analysis"], dependencies=)
@limiter.limit("30/minute")
async def score_policy_endpoint(request: Request, req_body: PolicyRequest):
    try:
        result = ai_service.score_policy(text=req_body.policy_text, policy_name=req_body.policy_name)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Error scoring policy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to score policy text")

@app.get("/api/v1/country/{country_code}/iqa", tags=["IQA"], dependencies=)
@limiter.limit("100/minute")
async def get_iqa_endpoint(request: Request, country_code: str, year: Optional[int] = None):
    try:
        notes = get_iqa_notes(country_code.upper(), year)
        return {"country_code": country_code, "notes": notes}
    except Exception as e:
        logger.error(f"Error fetching IQA notes for {country_code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/country/{country_code}/iqa", tags=["IQA"], dependencies=)
@limiter.limit("50/minute")
async def submit_iqa_endpoint(request: Request, country_code: str, req_body: IQASubmitRequest):
    try:
        note_data = req_body.dict()
        note_data['country_code'] = country_code.upper()
        save_iqa_note(note_data)
        logger.info(f"IQA Note Submitted for {country_code}: {note_data}")
        return {"status": "success", "message": "IQA note submitted successfully"}
    except Exception as e:
        logger.error(f"Error submitting IQA note for {country_code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit IQA note")

if __name__ == "__main__":
    logger.info("Starting Agency Monitor API...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)