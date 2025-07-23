# agency/integration.py
"""
Integration module for the complete Agency Calculus system.
Provides a high-level service to process API requests and perform
adversarial comparisons.
"""
from typing import Dict, Any, Optional, List
import json
import logging
import numpy as np

from agency.calculator import AgencyCalculator
from agency.brittleness_engine import BrittlenessEngine

logger = logging.getLogger(__name__)

def create_brittleness_engine(ideology: str = "framework_average") -> BrittlenessEngine:
    """Factory function to create a configured BrittlenessEngine."""
    calculator = AgencyCalculator(ideology=ideology)
    engine = BrittlenessEngine(calculator)
    return engine

class AgencyMonitorService:
    """
    Main service class that orchestrates all agency calculations.
    This is the primary entry point for API calls.
    """
    def __init__(self, ideology: str = "framework_average"):
        self.ideology = ideology
        self.engine = create_brittleness_engine(ideology)
        logger.info(f"AgencyMonitorService initialized with '{ideology}' perspective.")

    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a complete agency monitor request JSON."""
        try:
            agency_scores = request_data["agency_scores"]
            nominal_gdp = request_data["economic_indicators"]["nominal_gdp"]
            
            # --- Brittleness Calculation ---
            brittleness_result = self.engine.calculate_systemic_brittleness(
                agency_scores, nominal_gdp, return_details=True
            )
            
            response = {
                "country_code": request_data.get("country_code", "UNKNOWN"),
                "timestamp": request_data.get("timestamp"),
                "ideology": self.ideology,
                "brittleness": brittleness_result.to_dict()
            }
            
            # --- Optional Violation Calculation ---
            if "power_metrics" in request_data and "context" in request_data:
                violation_result = self.engine.agency_calculator.calculate_violation_magnitude(
                    delta_agency=request_data["context"].get("delta_agency", 0),
                    perpetrator_power_metrics=request_data["power_metrics"].get("perpetrator", {}),
                    victim_power_metrics=request_data["power_metrics"].get("victim", {}),
                    attack_factor_c=request_data["context"].get("attack_factor_c", 1.0),
                    k=request_data["context"].get("disparity_sensitivity_k", 1.0),
                    return_details=True
                )
                response["violation"] = violation_result.to_dict()
                
            return response
            
        except KeyError as e:
            logger.error(f"Missing key in request data: {e}")
            return {"error": f"Missing required key: {e}"}
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            return {"error": str(e)}

    def get_adversarial_comparison(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Processes the same data under all ideological perspectives."""
        results = {}
        ideologies = ["framework_average", "libertarian", "socialist", "communitarian"]
        
        for ideology in ideologies:
            service = AgencyMonitorService(ideology)
            results[ideology] = service.process_request(request_data)
            
        scores = [r["brittleness"]["brittleness_score"] for r in results.values() if "brittleness" in r]
        
        return {
            "perspectives": results,
            "comparison_summary": {
                "min_brittleness": min(scores) if scores else None,
                "max_brittleness": max(scores) if scores else None,
                "mean_brittleness": np.mean(scores) if scores else None,
                "variance": np.var(scores) if scores else None,
                "consensus": (np.var(scores) < 0.5) if scores else False
            }
        }

# --- Convenience Functions for API Endpoints ---
def calculate_brittleness(agency_scores: Dict[str, float], gdp: float, ideology: str = "framework_average") -> float:
    """Simple function to get just the brittleness score."""
    engine = create_brittleness_engine(ideology)
    return engine.calculate_systemic_brittleness(agency_scores, gdp, return_details=False)

def analyze_country_from_json(request_json: str) -> str:
    """Analyze a country from a JSON string, returning a JSON string."""
    try:
        request_data = json.loads(request_json)
        ideology = request_data.get("ideology", "framework_average")
        service = AgencyMonitorService(ideology)
        response = service.process_request(request_data)
        return json.dumps(response, indent=2)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON provided: {e}"})