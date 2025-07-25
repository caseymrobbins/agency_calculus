import os
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from pathlib import Path
from ai.hybrid_forecaster import HybridForecaster
from ai.policy_scorer import score_policy_text as nlp_score_policy_text
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
        ideologies = ["framework_average", "libertarian", "socialist", "communitarian"]
        for ideology in ideologies:
            calculator = AgencyCalculator(ideology=ideology)
            self.brittleness_engines[ideology] = BrittlenessEngine(calculator)

    def generate_forecast(self, country_code: str, steps: int = 5, weighting_scheme: str = "framework_average") -> Dict:
        country_code = country_code.upper()
        if country_code not in self.forecasters:
            raise ValueError(f"No model available for {country_code}")

        forecaster = self.forecasters[country_code]
        engine = self.brittleness_engines.get(weighting_scheme)
        if not engine:
            raise ValueError(f"Invalid weighting scheme: {weighting_scheme}")

        # Generate future exogenous features (in production, from historical exog)
        historical_exog = forecaster.training_features_.iloc[-1:].copy()  # Last known
        future_exog = pd.concat([historical_exog] * steps, ignore_index=True)
        future_exog.index = pd.date_range(start=forecaster.training_endog_.index[-1] + pd.offsets.YearEnd(1), periods=steps, freq='A')

        # Predict future agency scores
        agency_forecast_df = forecaster.predict(steps=steps, future_exog_df=future_exog)

        # Mock GDP forecast for demonstration (in production, real forecast)
        last_gdp = 25e12 if country_code == 'USA' else 15e9
        gdp_forecast = [last_gdp * (1.02 ** i) for i in range(1, steps + 1)]

        # Calculate brittleness for each forecasted step
        results = []
        for i, timestamp in enumerate(agency_forecast_df.index):
            agency_scores = agency_forecast_df.iloc[i].to_dict()
            nominal_gdp = gdp_forecast[i]
            brittleness_result = engine.calculate_systemic_brittleness(agency_scores, nominal_gdp, return_details=True)
            results.append({
                'timestamp': timestamp.isoformat(),
                'brittleness_score': brittleness_result.brittleness_score,
                'risk_level': brittleness_result.risk_level,
                'percentile': brittleness_result.percentile,
            })
        return {
            'country_code': country_code,
            'weighting_scheme': weighting_scheme,
            'forecast_results': results
        }

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
        return nlp_score_policy_text(text, **kwargs)

# Singleton instance to ensure models are loaded only once
ai_service = AIService(model_dir=os.getenv("MODEL_DIR", "models"))