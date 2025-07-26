from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
import logging

# Assuming imports from your project structure; adjust paths as needed
# from agency import BrittlenessEngine  # Uncomment and use for real calculation
# from api.database import get_db_connection  # Uncomment if database is needed

app = FastAPI(
    title="Agency Calculus API",
    description="API for Agency Monitor system to measure societal brittleness.",
    version="1.0.0"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Router with prefix
router = APIRouter(prefix="/api/v1")

# Optional: Define a request body model if the POST needs input data (e.g., for custom explanations)
class ExplainRequest(BaseModel):
    details_level: str = "basic"  # Example: 'basic', 'detailed'

@router.post("/country/{country_code}/explain")
async def explain_country(country_code: str, request: ExplainRequest | None = None):
    """
    Endpoint to explain the brittleness score for a given country.
    Currently returns a static example; integrate with BrittlenessEngine for dynamic response.
    """
    logger.info(f"Request received for country: {country_code}")
    
    if country_code.upper() != "USA":
        raise HTTPException(status_code=400, detail="Only USA is supported in this example.")
    
    # Stub logic: In production, fetch data from database, run BrittlenessEngine
    # Example:
    # conn = get_db_connection()
    # # Query data for country
    # engine = BrittlenessEngine()
    # score = engine.calculate(country_code, data)
    # domain_explanations = engine.explain_by_domain(score)
    
    # Static example based on repo description (USA brittleness at 9+/10)
    # 'top_feature_impacts' is now a dict {feature: impact} to match dashboard's from_dict(orient='index')
    brittleness_score = 9.2
    response = {
        "country": country_code,
        "brittleness_score": brittleness_score,
        "explanation": {
            "Economic": {
                "text": (
                    "Economic brittleness is high due to wealth concentration and restricted agency flow. "
                    "Calculated as Nominal GDP / Economic Agency (e.g., job mobility, entrepreneurship rates)."
                ),
                "predicted_residual": 12.5,  # Example value in % or units expected by dashboard
                "base_value": 8.0,  # For the help text in metric delta
                "top_feature_impacts": {
                    "Wealth Inequality (Gini)": 4.2,
                    "Job Mobility Rate": -2.1,
                    "Entrepreneurship Index": 3.5
                },
                "score_contribution": 9.5
            },
            "Political": {
                "text": (
                    "Political domain shows fragility from polarized governance and low voter agency. "
                    "Based on Agency Calculus: Political Agency = Voter turnout * Policy choice diversity."
                ),
                "predicted_residual": 10.8,
                "base_value": 7.5,
                "top_feature_impacts": {
                    "Voter Turnout": -3.0,
                    "Polarization Index": 5.1,
                    "Policy Diversity": 2.4
                },
                "score_contribution": 9.0
            },
            "Social": {
                "text": (
                    "Social brittleness arises from inequality and reduced community agency. "
                    "Metric: Social Agency = Cohesion indices / Inequality factors."
                ),
                "predicted_residual": 9.3,
                "base_value": 6.8,
                "top_feature_impacts": {
                    "Social Cohesion Index": -1.5,
                    "Inequality Factors": 4.0,
                    "Community Engagement": 2.2
                },
                "score_contribution": 8.8
            },
            "Health": {
                "text": (
                    "Health systems exhibit brittleness due to access disparities. "
                    "Formula: Health Agency = Life expectancy adjustment / Healthcare choice space."
                ),
                "predicted_residual": 11.7,
                "base_value": 9.0,
                "top_feature_impacts": {
                    "Life Expectancy": 3.8,
                    "Access Disparities": -2.5,
                    "Healthcare Choices": 4.1
                },
                "score_contribution": 9.3
            },
            "Educational": {
                "text": (
                    "Educational domain is brittle from unequal opportunities. "
                    "Computed as: Educational Agency = Literacy rates * Educational mobility."
                ),
                "predicted_residual": 10.2,
                "base_value": 7.2,
                "top_feature_impacts": {
                    "Literacy Rates": 2.9,
                    "Educational Mobility": -1.8,
                    "Opportunity Equality": 3.6
                },
                "score_contribution": 9.2
            }
        },
        "details_level": request.details_level if request else "basic"
    }
    
    return response

# Include the router in the app
app.include_router(router)

# Optional root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Agency Calculus API is running."}