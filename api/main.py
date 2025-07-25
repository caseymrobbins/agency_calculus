import os
import logging
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Security, status, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Local Imports ---
from ai_integration import ai_service
from database import get_iqa_notes, save_iqa_note

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SECRET_API_KEY = os.getenv("SECRET_API_KEY")
if not SECRET_API_KEY:
    logger.warning("SECRET_API_KEY environment variable not set. Using insecure default.")
    SECRET_API_KEY = "a_very_secret_key_for_dev"
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

app = FastAPI(
    title="Agency Monitor API",
    description="API for serving Agency Calculus data, predictions, and explanations",
    version="2.0"
)

# --- Rate Limiting & Middleware ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"Response: {response.status_code}")
        return response

app.add_middleware(RequestLoggingMiddleware)

# --- Security Dependency ---
async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == SECRET_API_KEY:
        return api_key
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")

# --- Pydantic Models ---
class ExplainRequest(BaseModel):
    year: int = Field(..., example=2024, description="The year for which to explain the forecast (e.g., explain 2024 using 2023 data)")

class PolicyRequest(BaseModel):
    policy_text: str = Field(..., example="New bill to increase education funding.", description="The policy text to score")
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    normalization: str = Field("density", enum=["density", "magnitude"])

class IQASubmitRequest(BaseModel):
    analyst: str = Field(..., example="Dr. Smith")
    note: str = Field(..., example="Note on policy impact")
    year: int = Field(..., example=2024)
    category: str = Field("other", enum=["economic", "political", "social", "health", "educational", "other"])

# --- API Endpoints ---
@app.get("/health", summary="Check API health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": app.version}

@app.get("/api/v1/country/{country_code}/forecast", summary="Generate brittleness forecast", tags=["Predictions"], dependencies=[Depends(get_api_key)])
@limiter.limit("50/minute")
async def get_forecast(request: Request, country_code: str, steps: int = 5, weighting_scheme: str = "framework_average"):
    try:
        forecast = ai_service.generate_forecast(country_code, steps, weighting_scheme)
        return forecast
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating forecast for {country_code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate forecast")

@app.post("/api/v1/country/{country_code}/explain", summary="Explain forecast for a year", tags=["Explanations"], dependencies=[Depends(get_api_key)])
@limiter.limit("50/minute")
async def get_explanation_endpoint(request: Request, country_code: str, req_body: ExplainRequest):
    try:
        explanation = ai_service.explain_forecast(country_code, req_body.year)
        return {"country_code": country_code, "year": req_body.year, "explanation": explanation}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error explaining forecast for {country_code} year {req_body.year}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate explanation")

@app.post("/api/v1/policy/score", summary="Score policy impact", tags=["Policy Analysis"], dependencies=[Depends(get_api_key)])
@limiter.limit("50/minute")
async def score_policy_endpoint(request: Request, req_body: PolicyRequest):
    try:
        result = ai_service.score_policy(req_body.policy_text, confidence_threshold=req_body.confidence_threshold, normalization=req_body.normalization)
        return result
    except Exception as e:
        logger.error(f"Error scoring policy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to score policy")

@app.get("/api/v1/country/{country_code}/iqa", summary="Get IQA notes", tags=["IQA"], dependencies=[Depends(get_api_key)])
@limiter.limit("100/minute")
async def get_iqa_endpoint(request: Request, country_code: str, year: Optional[int] = None):
    try:
        notes = get_iqa_notes(country_code.upper(), year)
        return {'country_code': country_code, 'notes': notes}
    except Exception as e:
        logger.error(f"Error fetching IQA notes for {country_code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/country/{country_code}/iqa", summary="Submit IQA note", tags=["IQA"], dependencies=[Depends(get_api_key)])
@limiter.limit("50/minute")
async def submit_iqa_endpoint(request: Request, country_code: str, req_body: IQASubmitRequest):
    try:
        note_data = {
            'country_code': country_code.upper(),
            'year': req_body.year,
            'analyst': req_body.analyst,
            'note': req_body.note,
            'category': req_body.category
        }
        save_iqa_note(note_data)
        return {"status": "success", "message": "IQA note saved"}
    except Exception as e:
        logger.error(f"Error saving IQA note for {country_code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save IQA note")

if __name__ == "__main__":
    logger.info("Starting Agency Monitor API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)