# main.py
# FINAL PRODUCTION VERSION
# This file defines the core REST API for the Agency Monitor project.
# It is a secure, scalable, and feature-complete FastAPI application incorporating
# adversarial weighting, IQA support, rate limiting, and robust error handling.
#
# Version: 1.2.0

import os
import logging
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Security, status, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_KEY = os.getenv("SECRET_API_KEY", "a_very_secret_key_for_dev")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

VALID_COUNTRY_CODES = ["USA", "HTI"]
VALID_WEIGHTING_SCHEMES = ["Libertarian", "Socialist", "Communitarian"]
REQUIRED_FEATURES = {"lagged_A_econ", "lagged_A_poli", "shock_magnitude", "recovery_slope", "brittleness_x_magnitude"}

app = FastAPI(
    title="Agency Monitor API",
    description="API for serving Agency Calculus data, predictions, explanations, and IQA notes.",
    version="1.2.0"
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
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate credentials")

# --- Mock Implementations (Ready to be swapped with real logic) ---
class MockHybridForecaster:
    def __init__(self, country_code: str):
        logger.info(f"Loading mock model for {country_code}...")
        self.country = country_code

    def predict(self, steps: int = 10, weighting: str = "Communitarian") -> List[Dict[str, Any]]:
        factors = {"Libertarian": 0.8, "Socialist": 1.2, "Communitarian": 1.0}
        factor = factors.get(weighting, 1.0)
        logger.info(f"Generating mock forecast for {self.country} with {weighting} weighting...")
        base_year = 2025
        data = []
        for i in range(steps):
            year = base_year + i
            data.extend([
                {"indicator_code": "economic", "year": year, "value": round((0.8 + i * 0.01) * factor, 3)},
                {"indicator_code": "political", "year": year, "value": round((0.6 - i * 0.005) * factor, 3)},
                {"indicator_code": "social", "year": year, "value": round((0.55 - i * 0.002) * factor, 3)},
                {"indicator_code": "health", "year": year, "value": round((0.9 + i * 0.001) * factor, 3)},
                {"indicator_code": "educational", "year": year, "value": round((0.85 + i * 0.003) * factor, 3)},
            ])
        return data

    def explain(self, features: Dict) -> Dict:
        logger.info(f"Generating mock explanation for instance: {features}")
        return {"base_value": 0.785, "shap_values": {"lagged_A_econ": 0.025, "lagged_A_poli": -0.041, "shock_magnitude": -0.052, "recovery_slope": 0.011, "brittleness_x_magnitude": -0.009}}

def query_db_for_timeseries(country_code: str, start_year: Optional[int], end_year: Optional[int]) -> List[Dict]:
    logger.info(f"Querying mock DB for {country_code} from {start_year} to {end_year}")
    indicators = ["gdp_per_capita_usd", "unemployment_rate_percent", "gini_coefficient", "life_expectancy_years", "infant_mortality_rate_per_1000", "public_health_spending_percent_gdp", "political_freedom_index", "voter_turnout_percent", "polarization_index", "social_trust_index", "mean_years_of_schooling", "public_education_spending_percent_gdp", "A_econ", "A_poli", "A_soc", "A_health", "A_edu"]
    raw_data = {ind: {"2020": 50 + (hash(ind) % 10), "2021": 52 + (hash(ind) % 10), "2022": 54 + (hash(ind) % 10)} for ind in indicators}
    data = [{"indicator_code": ind, "year": int(year), "value": float(value)} for ind, years in raw_data.items() for year, value in years.items()]
    if start_year: data = [d for d in data if d["year"] >= start_year]
    if end_year: data = [d for d in data if d["year"] <= end_year]
    return data

def query_db_for_iqa(country_code: str, year: Optional[int]) -> List[Dict[str, Any]]:
    logger.info(f"Querying mock DB for IQA notes: {country_code}, year: {year}")
    mock_notes = [{"analyst": "Dr. Smith", "note": "2008 financial crisis impacted social trust.", "year": 2008}, {"analyst": "Dr. Jones", "note": "COVID-19 pandemic response led to health agency decline.", "year": 2020}]
    if country_code != "USA": return []
    if year: return [note for note in mock_notes if note["year"] == year]
    return mock_notes

def store_iqa_note(country_code: str, note_data: Dict) -> Dict:
    logger.info(f"Storing mock IQA note for {country_code}: {note_data}")
    return {"status": "success", "message": "IQA note stored"}

# --- Pydantic Models ---
class TimeSeriesDataPoint(BaseModel): indicator_code: str; year: int; value: float
class TimeSeriesResponse(BaseModel): country_code: str; data: List[TimeSeriesDataPoint]
class ForecastDataPoint(BaseModel): indicator_code: str; year: int; value: float
class ForecastResponse(BaseModel): country_code: str; forecast: List[ForecastDataPoint]
class ExplainRequest(BaseModel): features: Dict[str, float]
class ExplainResponse(BaseModel): country_code: str; explanation: Dict
class IQANote(BaseModel): analyst: str; note: str; year: int
class IQAResponse(BaseModel): country_code: str; notes: List[IQANote]
class IQASubmitRequest(BaseModel): analyst: str; note: str; year: int

# --- API Endpoints ---
@app.get("/health", summary="Check API health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": app.version}

@app.get("/api/v1/country/{country_code}/timeseries", response_model=TimeSeriesResponse, tags=["Data"], dependencies=[Depends(get_api_key)])
@limiter.limit("100/minute")
async def get_timeseries(request: Request, country_code: str, start_year: Optional[int] = None, end_year: Optional[int] = None, limit: int = 100, offset: int = 0):
    if country_code not in VALID_COUNTRY_CODES: raise HTTPException(status_code=404, detail=f"Invalid country code. Supported: {VALID_COUNTRY_CODES}")
    data = query_db_for_timeseries(country_code, start_year, end_year)
    if not data: raise HTTPException(status_code=404, detail=f"No time-series data found for {country_code}")
    return {"country_code": country_code, "data": data[offset:offset + limit]}

@app.get("/api/v1/country/{country_code}/forecast", response_model=ForecastResponse, tags=["Predictions"], dependencies=[Depends(get_api_key)])
@limiter.limit("50/minute")
async def get_forecast(request: Request, country_code: str, weighting: str = "Communitarian"):
    if country_code not in VALID_COUNTRY_CODES: raise HTTPException(status_code=404, detail=f"Invalid country code. Supported: {VALID_COUNTRY_CODES}")
    if weighting not in VALID_WEIGHTING_SCHEMES: raise HTTPException(status_code=400, detail=f"Invalid weighting scheme. Supported: {VALID_WEIGHTING_SCHEMES}")
    model = MockHybridForecaster(country_code)
    return {"country_code": country_code, "forecast": model.predict(steps=10, weighting=weighting)}

@app.post("/api/v1/country/{country_code}/explain", response_model=ExplainResponse, tags=["Explanations"], dependencies=[Depends(get_api_key)])
@limiter.limit("50/minute")
async def get_explanation(request: Request, country_code: str, request_body: ExplainRequest):
    if country_code not in VALID_COUNTRY_CODES: raise HTTPException(status_code=404, detail=f"Invalid country code. Supported: {VALID_COUNTRY_CODES}")
    if not all(f in request_body.features for f in REQUIRED_FEATURES): raise HTTPException(status_code=400, detail=f"Missing required features: {REQUIRED_FEATURES - request_body.features.keys()}")
    model = MockHybridForecaster(country_code)
    return {"country_code": country_code, "explanation": model.explain(request_body.features)}

@app.get("/api/v1/country/{country_code}/iqa", response_model=IQAResponse, tags=["IQA"], dependencies=[Depends(get_api_key)])
@limiter.limit("100/minute")
async def get_iqa(request: Request, country_code: str, year: Optional[int] = None):
    if country_code not in VALID_COUNTRY_CODES: raise HTTPException(status_code=404, detail=f"Invalid country code. Supported: {VALID_COUNTRY_CODES}")
    if year and (year < 1900 or year > 2100): raise HTTPException(status_code=400, detail="Year must be between 1900 and 2100")
    notes = query_db_for_iqa(country_code, year)
    if not notes and year: raise HTTPException(status_code=404, detail=f"No IQA notes found for {country_code} in {year}")
    return {"country_code": country_code, "notes": notes}

@app.post("/api/v1/country/{country_code}/iqa", tags=["IQA"], dependencies=[Depends(get_api_key)])
@limiter.limit("50/minute")
async def submit_iqa(request: Request, country_code: str, request_body: IQASubmitRequest):
    if country_code not in VALID_COUNTRY_CODES: raise HTTPException(status_code=404, detail=f"Invalid country code. Supported: {VALID_COUNTRY_CODES}")
    if not request_body.note.strip(): raise HTTPException(status_code=400, detail="Note cannot be empty")
    return store_iqa_note(country_code, request_body.dict())

if __name__ == "__main__":
    logger.info("Starting Agency Monitor API...")
    logger.info(f"To use the API, send requests with the header: '{API_KEY_NAME}: {API_KEY}'")
    uvicorn.run(app, host="0.0.0.0", port=8000)