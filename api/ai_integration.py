# api/ai_integration.py
"""
AI Integration Service - Production Version
This module serves as the bridge between the FastAPI endpoints and the underlying
AI models. It manages model loading, orchestration of predictions, and formatting
of outputs like forecasts and SHAP explanations.
"""
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

# Import core AI components
from ai.hybrid_forecaster import HybridForecaster
from ai.policy_scorer import score_policy_text as nlp_score_policy_text

# Import AC4 calculus components, assuming they exist in this structure
# from agency.calculator import AgencyCalculator
# from agency.brittleness_engine import BrittlenessEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIService:
    """
    Singleton service class that manages AI model loading, prediction, and explanation.
    """
    def __init__(self, model_dir: str = 'ai/models/'):
        self.forecasters: Dict[str, HybridForecaster] = {}
        # Placeholder for other services if they were stateful
        # self.calculator = AgencyCalculator()
        # self.brittleness_engine = BrittlenessEngine(self.calculator)
        self._load_forecasters(model_dir)

    def _load_forecasters(self, model_dir: str):
        """Load all available HybridForecaster models from the specified directory."""
        logger.info(f"Loading forecasting models from directory: {model_dir}")
        model_path = Path(model_dir)
        if not model_path.exists():
            logger.warning(f"Model directory not found: {model_dir}. No models loaded.")
            return

        for f in model_path.glob('*_hybrid_forecaster.pkl'):
            # FIX: Use a more robust way to extract the country code from the filename
            # e.g., 'USA_hybrid_forecaster.pkl' -> 'USA'
            country_code = f.stem.split('_')[0].upper()
            try:
                self.forecasters[country_code] = HybridForecaster.load_model(str(f))
                logger.info(f"Successfully loaded forecaster for {country_code}.")
            except Exception as e:
                logger.error(f"Failed to load model for {country_code} from {f}: {e}")

    def _project_future_features(self, historical_exog: pd.DataFrame, steps: int) -> pd.DataFrame:
        """Projects future exogenous features using a simple trend extrapolation."""
        # This is a simplified projection. A more advanced system would forecast these features.
        future_dates = pd.date_range(
            start=historical_exog.index[-1] + pd.DateOffset(years=1),
            periods=steps, freq='A' # Use annual frequency
        )
        future_exog = pd.DataFrame(index=future_dates)
        for col in historical_exog.columns:
            # Simple forward fill for this example; a real system might use trends or other models
            future_exog[col] = historical_exog[col].iloc[-1]
        return future_exog

    def generate_forecast(self, country_code: str, steps: int, weighting_scheme: str) -> Dict:
        """
        Orchestrates the full forecasting and brittleness calculation pipeline.
        """
        if country_code not in self.forecasters:
            raise ValueError(f"No forecasting model available for country code: {country_code}")

        forecaster = self.forecasters[country_code]
        
        # Project future exogenous features
        future_exog_df = self._project_future_features(forecaster.training_features_[forecaster.exog_columns_], steps)
        
        # Get 5-domain forecast from the hybrid model
        agency_forecast_df = forecaster.predict(steps=steps, future_exog_df=future_exog_df)

        # --- Post-processing: Calculate Brittleness Score ---
        # This part is a placeholder for the real AgencyCalculator integration
        # For now, we'll create a mock brittleness score for demonstration
        last_gdp = 25e12 if country_code == 'USA' else 15e9 # $25T for USA, $15B for Haiti
        gdp_forecast = [last_gdp * (1.02**i) for i in range(1, steps + 1)]

        # FIX: Initialize results as an empty list
        results = []
        for i, timestamp in enumerate(agency_forecast_df.index):
            agency_scores = agency_forecast_df.iloc[i].to_dict()
            total_agency = np.mean(list(agency_scores.values())) # Simplified total agency
            gdp = gdp_forecast[i]
            
            # Mock brittleness calculation V = GDP / A_total (simplified)
            # Scaled to be roughly in a 0-10 range for this example
            brittleness_score = (gdp / (total_agency * 3e12)) 
            
            risk_level = "Critical" if brittleness_score > 8.5 else "High" if brittleness_score > 7 else "Medium"

            results.append({
                'year': timestamp.year,
                'agency_scores': {k: round(v, 4) for k, v in agency_scores.items()},
                'brittleness_score': round(brittleness_score, 2),
                'risk_level': risk_level
            })
        return {'country_code': country_code, 'weighting_scheme': weighting_scheme, 'forecast': results}

    def explain_forecast(self, country_code: str, year: int) -> Dict:
        """Generates SHAP explanation for a specific forecast."""
        if country_code not in self.forecasters:
            raise ValueError(f"No model available for {country_code}")

        forecaster = self.forecasters[country_code]
        feature_date = pd.to_datetime(f"{year-1}-12-31")
        
        if feature_date not in forecaster.training_features_.index:
            raise ValueError(f"Cannot explain year {year}. Features for {year-1} not in training data.")

        instance_features = forecaster.training_features_.loc[[feature_date]]
        base_values, shap_values_list = forecaster.explain(instance_features)

        explanations = {}
        for i, domain in enumerate(forecaster.endog_columns_):
            shap_vals = shap_values_list[i].flatten() # Ensure it's a 1D array
            feature_impacts = {
                name: round(float(val), 5) for name, val in zip(instance_features.columns, shap_vals)
            }
            # Sort by absolute impact for clarity
            sorted_impacts = dict(sorted(feature_impacts.items(), key=lambda item: abs(item[1]), reverse=True)[:10])
            
            explanations[domain] = {
                'base_value_residual': round(float(base_values[i]), 5),
                'predicted_residual': round(float(base_values[i] + sum(shap_vals)), 5),
                'top_feature_impacts': sorted_impacts
            }
        return explanations

    def score_policy(self, text: str, **kwargs) -> Dict:
        """Scores a policy text using the NLP model."""
        return nlp_score_policy_text(text, **kwargs)

# --- Singleton Instance ---
# This creates a single instance of the service when the module is imported,
# ensuring models are loaded only once, which is crucial for performance.
ai_service = AIService()