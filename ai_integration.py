import os
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from pathlib import Path
from ai.hybrid_forecaster import HybridForecaster
from ai.policy_scorer import PolicyScorer
from agency.calculator import AgencyCalculator
from agency.brittleness_engine import BrittlenessEngine

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AIService:
    """Singleton service class that manages AI model loading, prediction, and explanation."""
    def __init__(self, model_dir: str = 'models/'):
        self.forecasters: Dict[str, HybridForecaster] = {}
        self.brittleness_engines: Dict[str, BrittlenessEngine] = {}
        self._load_forecasters(model_dir)
        self._initialize_calculus_engines()

    def _load_forecasters(self, model_dir: str):
        """Load all available HybridForecaster models from the specified directory."""
        logger.info(f"Loading forecasting models from directory: {model_dir}")
        model_path = Path(model_dir)
        if not model_path.exists():
            logger.warning(f"Model directory not found: {model_dir}. No models loaded.")
            return
        
        for f in model_path.glob('*_hybrid_forecaster.pkl'):
            country_code = f.stem.split('_')[0].upper()
            try:
                self.forecasters[country_code] = HybridForecaster.load_model(str(f))
                logger.info(f"Successfully loaded forecaster for {country_code}")
            except Exception as e:
                logger.error(f"Failed to load model for {country_code}: {e}")

    def _initialize_calculus_engines(self):
        # Only use ideologies that are actually implemented
        ideologies = ["framework_average"]
        for ideology in ideologies:
            try:
                calculator = AgencyCalculator(ideology=ideology)
                self.brittleness_engines[ideology] = BrittlenessEngine(calculator)
                logger.info(f"Initialized brittleness engine for ideology: {ideology}")
            except ValueError as e:
                logger.warning(f"Could not initialize ideology '{ideology}': {e}")
        
        logger.info(f"Available weighting schemes: {list(self.brittleness_engines.keys())}")

    def generate_forecast(self, country_code: str, steps: int = 5, weighting_scheme: str = "framework_average") -> Dict:
        try:
            country_code = country_code.upper()
            logger.info(f"Generating forecast for {country_code}, steps: {steps}, scheme: {weighting_scheme}")
            logger.info(f"Available schemes: {list(self.brittleness_engines.keys())}")
            logger.info(f"Available countries: {list(self.forecasters.keys())}")
            
            if country_code not in self.forecasters:
                raise ValueError(f"No model available for {country_code}")

            forecaster = self.forecasters[country_code]
            engine = self.brittleness_engines.get(weighting_scheme)
            if not engine:
                raise ValueError(f"Invalid weighting scheme: {weighting_scheme}. Available: {list(self.brittleness_engines.keys())}")

            # Generate future exogenous features
            if not hasattr(forecaster, 'training_features_') or forecaster.training_features_ is None:
                raise ValueError(f"Forecaster for {country_code} has no training features")
                
            historical_exog = forecaster.training_features_.iloc[-1:].copy()
            future_exog = pd.concat([historical_exog] * steps, ignore_index=True)
            future_exog.index = pd.date_range(
                start=forecaster.training_endog_.index[-1] + pd.offsets.YearEnd(1), 
                periods=steps, 
                freq='A'
            )

            # Predict future agency scores
            agency_forecast_df = forecaster.predict(steps=steps, future_exog=future_exog)

            # Mock GDP forecast for demonstration
            last_gdp = 25e12 if country_code == 'USA' else 15e9
            gdp_forecast = [last_gdp * (1.02 ** i) for i in range(1, steps + 1)]

            # Calculate brittleness for each forecasted step
            results = []
            for i, timestamp in enumerate(agency_forecast_df.index):
                agency_scores = agency_forecast_df.iloc[i].to_dict()
                nominal_gdp = gdp_forecast[i]
                
                # Convert to format expected by brittleness engine
                formatted_scores = {}
                for key, value in agency_scores.items():
                    # Remove '_agency' suffix if present
                    domain = key.replace('_agency', '')
                    formatted_scores[f"{domain}_agency"] = value
                
                brittleness_result = engine.calculate_systemic_brittleness(
                    formatted_scores, 
                    nominal_gdp, 
                    return_details=True
                )
                
                results.append({
                    'year': timestamp.year,
                    'agency_scores': formatted_scores,
                    'brittleness_score': float(brittleness_result.brittleness_score),
                    'risk_level': brittleness_result.risk_level,
                    'percentile': float(brittleness_result.percentile),
                })
                
            return {
                'country_code': country_code,
                'weighting_scheme': weighting_scheme,
                'forecast': results
            }
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {str(e)}", exc_info=True)
            raise

    def explain_forecast(self, country_code: str, year: int) -> Dict:
        """Generates SHAP explanation for a specific forecast."""
        country_code = country_code.upper()
        if country_code not in self.forecasters:
            raise ValueError(f"No model available for {country_code}")
        forecaster = self.forecasters[country_code]

        try:
            # The features for year Y's forecast are from Y-1
            feature_date = pd.to_datetime(f"{year-1}-12-31")
            instance_features = forecaster.training_features_.loc[[feature_date]]
        except KeyError:
            raise ValueError(f"Cannot explain year {year}. Features for {year-1} not in training data")

        base_values, shap_values_list = forecaster.explain(instance_features)
        explanations = {}
        for i, domain in enumerate(forecaster.endog_columns_):
            shap_vals = shap_values_list[i].flatten()
            feature_impacts = {
                name: float(val) for name, val in zip(forecaster.training_features_.columns, shap_vals)
            }
            sorted_impacts = dict(sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
            explanations[domain] = {
                'base_value': float(base_values[i]),
                'predicted_residual': round(float(base_values[i] + sum(shap_vals)), 5),
                'top_feature_impacts': sorted_impacts
            }
        return explanations

    def score_policy(self, text: str, **kwargs) -> Dict:
        """Scores a policy text using the NLP model."""
        # Create a new PolicyScorer instance for each request
        scorer = PolicyScorer()
        # Assuming PolicyScorer has a score method
        if hasattr(scorer, 'score'):
            return scorer.score(text, **kwargs)
        else:
            # Fallback if method doesn't exist
            return {
                "score": 0.5,
                "confidence": 0.0,
                "message": "Policy scoring not yet implemented"
            }

# Singleton instance to ensure models are loaded only once
ai_service = AIService(model_dir=os.getenv("MODEL_DIR", "models"))