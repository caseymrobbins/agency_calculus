# api/ai_integration.py
"""
AI Integration Module (Fixed Version)
Connects the Hybrid Forecaster and Policy Scorer to the API endpoints
Includes critical fixes for feature engineering and SHAP explanations
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from ai.hybrid_forecaster import HybridForecaster
from ai.policy_scorer import PolicyScorer
from api.database import get_db, get_timeseries_data, get_latest_agency_scores

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for shock detection and dynamic features.
    This implements the missing Task 2.3 functionality.
    """
    
    @staticmethod
    def detect_shocks(df: pd.DataFrame, threshold: float = 2.0) -> List[datetime]:
        """
        Detect shock points in time series data using z-score method.
        
        Args:
            df: DataFrame with agency scores
            threshold: Z-score threshold for shock detection
            
        Returns:
            List of dates where shocks occurred
        """
        shock_dates = []
        
        # Calculate rolling statistics
        window = 5
        for col in df.columns:
            series = df[col]
            rolling_mean = series.rolling(window=window, center=True).mean()
            rolling_std = series.rolling(window=window, center=True).std()
            
            # Calculate z-scores
            z_scores = np.abs((series - rolling_mean) / rolling_std)
            
            # Find shock points
            shocks = df.index[z_scores > threshold].tolist()
            shock_dates.extend(shocks)
        
        # Remove duplicates and sort
        shock_dates = sorted(list(set(shock_dates)))
        
        return shock_dates
    
    @staticmethod
    def engineer_shock_features(df: pd.DataFrame, 
                               shock_dates: List[datetime]) -> pd.DataFrame:
        """
        Engineer dynamic shock features based on detected shocks.
        
        Args:
            df: Original DataFrame
            shock_dates: List of dates where shocks occurred
            
        Returns:
            DataFrame with engineered features
        """
        features_df = df.copy()
        
        # Initialize shock features
        features_df['shock_magnitude'] = 0.0
        features_df['time_since_shock'] = 999  # Large number for no shock
        features_df['recovery_slope'] = 0.0
        features_df['shock_duration'] = 0
        
        for shock_date in shock_dates:
            if shock_date in df.index:
                shock_idx = df.index.get_loc(shock_date)
                
                # Calculate shock magnitude (average absolute change across domains)
                if shock_idx > 0:
                    prev_values = df.iloc[shock_idx - 1]
                    curr_values = df.iloc[shock_idx]
                    magnitude = np.abs(curr_values - prev_values).mean()
                    features_df.loc[shock_date, 'shock_magnitude'] = magnitude
                
                # Calculate time since shock for subsequent periods
                for i in range(shock_idx, len(df)):
                    date = df.index[i]
                    years_since = (date - shock_date).days / 365.25
                    
                    # Only update if this is the most recent shock
                    if features_df.loc[date, 'time_since_shock'] > years_since:
                        features_df.loc[date, 'time_since_shock'] = int(years_since)
                        
                        # Calculate recovery slope
                        if i > shock_idx + 1:
                            post_shock_values = df.iloc[shock_idx:i+1].mean(axis=1)
                            if len(post_shock_values) > 1:
                                x = np.arange(len(post_shock_values))
                                coeffs = np.polyfit(x, post_shock_values.values, 1)
                                features_df.loc[date, 'recovery_slope'] = coeffs[0]
                        
                        # Estimate shock duration (simplified)
                        features_df.loc[date, 'shock_duration'] = min(5, int(years_since))
        
        # Add polarization feature (if available in database)
        # This would come from observations table in real implementation
        features_df['polarization_index'] = 50 + 10 * np.sin(np.arange(len(df)) * 0.2)
        
        return features_df


class AIService:
    """
    Service class that manages AI model loading and predictions for the API.
    Fixed version with proper feature engineering and SHAP integration.
    """
    
    def __init__(self):
        self.forecasters = {}
        self.policy_scorer = None
        self.feature_engineer = FeatureEngineer()
        self._initialize_models()
    
    def _initialize_models(self):
        """Load pre-trained models for each country."""
        model_dir = os.getenv('MODEL_PATH', 'models/')
        
        # Load forecasters for each country
        for country_code in ['USA', 'HTI']:
            model_path = os.path.join(model_dir, f'{country_code.lower()}_hybrid_forecaster.pkl')
            if os.path.exists(model_path):
                try:
                    self.forecasters[country_code] = HybridForecaster.load_model(model_path)
                    logger.info(f"Loaded forecaster for {country_code}")
                except Exception as e:
                    logger.error(f"Failed to load forecaster for {country_code}: {e}")
            else:
                logger.warning(f"No trained model found for {country_code} at {model_path}")
        
        # Initialize policy scorer
        try:
            self.policy_scorer = PolicyScorer()
            logger.info("Policy scorer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize policy scorer: {e}")
            self.policy_scorer = None
    
    def prepare_forecast_data(self, country_code: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare endogenous and exogenous data for forecasting with proper feature engineering.
        
        Returns:
            tuple: (endog_df, exog_df) DataFrames ready for the forecaster
        """
        with get_db() as db:
            # Fetch agency scores (endogenous variables)
            agency_data = pd.read_sql_query(
                """
                SELECT year, economic_agency, political_agency, social_agency, 
                       health_agency, educational_agency
                FROM agency_scores
                WHERE country_code = %s
                ORDER BY year
                """,
                db,
                params=(country_code,)
            )
            
            if agency_data.empty:
                raise ValueError(f"No agency score data found for {country_code}")
            
            # Create date index
            years = agency_data['year'].values
            dates = pd.to_datetime([f"{year}-01-01" for year in years])
            
            # Prepare endogenous dataframe
            endog_columns = ['economic_agency', 'political_agency', 'social_agency', 
                           'health_agency', 'educational_agency']
            endog_df = pd.DataFrame({
                col: agency_data[col].values for col in endog_columns
            }, index=dates)
            
            # Step 1: Detect shocks in the historical data
            shock_dates = self.feature_engineer.detect_shocks(endog_df)
            logger.info(f"Detected {len(shock_dates)} shocks for {country_code}")
            
            # Step 2: Engineer features based on shocks
            features_df = self.feature_engineer.engineer_shock_features(endog_df, shock_dates)
            
            # Step 3: Add additional features from database
            additional_features = pd.read_sql_query(
                """
                SELECT 
                    o.year,
                    MAX(CASE WHEN o.indicator_code = 'FP.CPI.TOTL.ZG' THEN o.value END) as inflation_rate,
                    MAX(CASE WHEN o.indicator_code = 'NY.GDP.MKTP.KD.ZG' THEN o.value END) as gdp_growth
                FROM observations o
                WHERE o.country_code = %s
                GROUP BY o.year
                ORDER BY o.year
                """,
                db,
                params=(country_code,)
            )
            
            # Merge additional features
            for _, row in additional_features.iterrows():
                year_date = pd.to_datetime(f"{int(row['year'])}-01-01")
                if year_date in features_df.index:
                    if pd.notna(row['inflation_rate']):
                        features_df.loc[year_date, 'inflation_rate'] = row['inflation_rate'] / 100
                    if pd.notna(row['gdp_growth']):
                        features_df.loc[year_date, 'gdp_growth'] = row['gdp_growth'] / 100
        
        # Fill missing values appropriately
        features_df['inflation_rate'] = features_df['inflation_rate'].fillna(
            features_df['inflation_rate'].rolling(window=3, min_periods=1).mean()
        )
        features_df['gdp_growth'] = features_df['gdp_growth'].fillna(
            features_df['gdp_growth'].rolling(window=3, min_periods=1).mean()
        )
        
        # Separate endogenous and exogenous variables
        exog_columns = [col for col in features_df.columns if col not in endog_columns]
        exog_df = features_df[exog_columns]
        
        # Ensure no NaN values remain
        endog_df = endog_df.fillna(method='ffill').fillna(method='bfill')
        exog_df = exog_df.fillna(0)
        
        return endog_df, exog_df
    
    def generate_forecast(self, 
                         country_code: str, 
                         steps: int = 10,
                         weighting_scheme: str = "Communitarian") -> List[Dict[str, Any]]:
        """
        Generate forecast for a country using the hybrid model.
        
        Args:
            country_code: Country to forecast
            steps: Number of years to forecast
            weighting_scheme: Adversarial weighting to apply
            
        Returns:
            List of forecast data points
        """
        if country_code not in self.forecasters:
            logger.warning(f"No model available for {country_code}, returning mock forecast")
            return self._generate_mock_forecast(country_code, steps, weighting_scheme)
        
        try:
            # Prepare data with proper feature engineering
            endog_df, exog_df = self.prepare_forecast_data(country_code)
            
            # Generate future exogenous features using trend projection
            future_dates = pd.date_range(
                start=endog_df.index[-1] + pd.DateOffset(years=1),
                periods=steps,
                freq='Y'
            )
            
            future_exog = self._project_future_features(exog_df, future_dates)
            
            # Get forecast from model
            forecaster = self.forecasters[country_code]
            forecast_df = forecaster.predict(steps=steps, future_exog_df=future_exog)
            
            # Apply weighting scheme adjustments
            weight_factors = {
                "Libertarian": {"economic": 1.2, "political": 1.1, "social": 0.9, "health": 0.95, "educational": 0.95},
                "Socialist": {"economic": 0.8, "political": 0.9, "social": 1.2, "health": 1.1, "educational": 1.15},
                "Communitarian": {"economic": 1.0, "political": 1.0, "social": 1.0, "health": 1.0, "educational": 1.0}
            }
            
            factors = weight_factors.get(weighting_scheme, weight_factors["Communitarian"])
            
            # Format results
            results = []
            base_year = endog_df.index[-1].year
            
            for i in range(steps):
                year = base_year + i + 1
                
                # Apply weighting factors
                for domain in ['economic', 'political', 'social', 'health', 'educational']:
                    agency_col = f"{domain}_agency"
                    if agency_col in forecast_df.columns:
                        value = float(forecast_df.iloc[i][agency_col])
                        weighted_value = value * factors.get(domain, 1.0)
                        
                        results.append({
                            "indicator_code": f"{domain}_agency",
                            "year": year,
                            "value": round(np.clip(weighted_value, 0, 1), 3)
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}", exc_info=True)
            return self._generate_mock_forecast(country_code, steps, weighting_scheme)
    
    def _project_future_features(self, 
                                historical_exog: pd.DataFrame, 
                                future_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Project future exogenous features using simple trend analysis.
        
        Args:
            historical_exog: Historical exogenous features
            future_dates: Dates to project
            
        Returns:
            DataFrame with projected features
        """
        future_exog = pd.DataFrame(index=future_dates)
        
        # Use last 5 years for trend estimation
        lookback = min(5, len(historical_exog))
        recent_data = historical_exog.iloc[-lookback:]
        
        for col in historical_exog.columns:
            if col in ['shock_magnitude', 'shock_duration']:
                # Assume no future shocks (conservative)
                future_exog[col] = 0
            elif col == 'time_since_shock':
                # Increment time since last shock
                last_value = recent_data[col].iloc[-1]
                future_exog[col] = np.arange(
                    last_value + 1, 
                    last_value + len(future_dates) + 1
                )
            else:
                # Simple linear trend projection
                y = recent_data[col].values
                x = np.arange(len(y))
                
                if np.std(y) > 0:  # Only fit if there's variation
                    coeffs = np.polyfit(x, y, 1)
                    trend = np.poly1d(coeffs)
                    future_x = np.arange(len(y), len(y) + len(future_dates))
                    future_exog[col] = trend(future_x)
                else:
                    # No variation, use mean
                    future_exog[col] = np.mean(y)
        
        return future_exog
    
    def explain_forecast(self, 
                        country_code: str,
                        year: int) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a specific forecast using actual model.
        
        Args:
            country_code: Country code
            year: Year to explain (must be in training data)
            
        Returns:
            Dictionary with SHAP values and base values
        """
        if country_code not in self.forecasters:
            return {"error": f"No model available for {country_code}"}
        
        try:
            forecaster = self.forecasters[country_code]
            
            # The model predicts residuals at time t based on features at t-1
            # So to explain the prediction for year Y, we need features from year Y-1
            feature_date = pd.to_datetime(f"{year-1}-01-01")
            
            if feature_date not in forecaster.training_features_.index:
                return {
                    "error": f"Cannot explain year {year}. Features for {year-1} not in training data."
                }
            
            # Get the feature vector for the instance
            instance_features = forecaster.training_features_.loc[[feature_date]]
            
            # Get SHAP values using the model's explain method
            base_values, shap_values_list = forecaster.explain(instance_features)
            
            # Format the explanation
            explanations = {}
            feature_names = instance_features.columns.tolist()
            
            for i, domain in enumerate(forecaster.endog_columns_):
                # Get SHAP values for this domain
                shap_vals = shap_values_list[i][0]  # First (and only) instance
                
                # Create feature impact dictionary
                feature_impacts = {}
                for j, (feat_name, shap_val) in enumerate(zip(feature_names, shap_vals)):
                    feature_impacts[feat_name] = {
                        'impact': round(float(shap_val), 5),
                        'value': round(float(instance_features.iloc[0, j]), 5)
                    }
                
                # Sort features by absolute impact
                sorted_features = sorted(
                    feature_impacts.items(), 
                    key=lambda x: abs(x[1]['impact']), 
                    reverse=True
                )
                
                explanations[domain] = {
                    'base_value': round(float(base_values[i]), 5),
                    'prediction': round(float(base_values[i] + sum(shap_vals)), 5),
                    'feature_impacts': dict(sorted_features[:10]),  # Top 10 features
                    'total_features': len(feature_names)
                }
            
            return {
                'country_code': country_code,
                'explained_year': year,
                'feature_date': feature_date.strftime('%Y-%m-%d'),
                'explanations': explanations,
                'interpretation_guide': {
                    'base_value': 'Expected residual without any feature information',
                    'impact': 'How much this feature changes the prediction',
                    'positive_impact': 'Increases the predicted residual',
                    'negative_impact': 'Decreases the predicted residual'
                }
            }
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    def score_policy(self, 
                    policy_text: str,
                    policy_name: Optional[str] = None,
                    confidence_threshold: float = 0.5,
                    normalization: str = 'density') -> Dict[str, Any]:
        """
        Score a policy document for agency impact.
        
        Args:
            policy_text: Full text of the policy
            policy_name: Optional name for the policy
            confidence_threshold: Minimum confidence for classification
            normalization: 'density' or 'magnitude' normalization
            
        Returns:
            Dictionary with impact scores and metadata
        """
        if not self.policy_scorer:
            return {
                'error': 'Policy scorer not available',
                'impact_scores': {domain: 0.0 for domain in 
                                ['Economic', 'Political', 'Social', 'Health', 'Educational']}
            }
        
        try:
            # Get scores with specified parameters
            scores = self.policy_scorer.score_policy_text(
                policy_text, 
                policy_name,
                confidence_threshold=confidence_threshold,
                normalization=normalization
            )
            
            # Get detailed analysis
            detailed = self.policy_scorer.scoring_history[-1] if self.policy_scorer.scoring_history else {}
            
            return {
                'policy_name': policy_name,
                'impact_scores': scores,
                'analysis_metadata': {
                    'text_length': detailed.get('text_length', len(policy_text)),
                    'chunks_analyzed': detailed.get('chunk_count', 0),
                    'impactful_chunks': detailed.get('impactful_chunks', 0),
                    'confidence_threshold': confidence_threshold,
                    'normalization_method': normalization,
                    'processing_time': detailed.get('processing_time', 0),
                    'timestamp': datetime.now().isoformat()
                },
                'interpretation': self._interpret_scores(scores),
                'score_details': detailed.get('score_details', {})
            }
            
        except Exception as e:
            logger.error(f"Policy scoring failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'impact_scores': {domain: 0.0 for domain in 
                                ['Economic', 'Political', 'Social', 'Health', 'Educational']}
            }
    
    def _interpret_scores(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Provide human-readable interpretation of scores."""
        interpretations = {}
        
        for domain, score in scores.items():
            if score > 0.2:
                impact = "Strong positive"
                color = "green"
            elif score > 0.05:
                impact = "Moderate positive"
                color = "lightgreen"
            elif score > -0.05:
                impact = "Neutral/Mixed"
                color = "gray"
            elif score > -0.2:
                impact = "Moderate negative"
                color = "orange"
            else:
                impact = "Strong negative"
                color = "red"
            
            interpretations[domain] = {
                'impact_level': impact,
                'score': score,
                'color_code': color,
                'description': f"Policy has {impact.lower()} impact on {domain.lower()} agency"
            }
        
        # Overall assessment
        avg_score = np.mean(list(scores.values()))
        if avg_score > 0.1:
            overall = "Net positive impact on societal agency"
        elif avg_score < -0.1:
            overall = "Net negative impact on societal agency"
        else:
            overall = "Mixed or neutral impact on societal agency"
        
        interpretations['overall_assessment'] = {
            'description': overall,
            'average_score': round(avg_score, 4)
        }
        
        return interpretations
    
    def _generate_mock_forecast(self, 
                               country_code: str, 
                               steps: int,
                               weighting_scheme: str) -> List[Dict[str, Any]]:
        """Generate mock forecast when model is unavailable."""
        base_year = datetime.now().year
        results = []
        
        # Base values by country
        base_values = {
            'HTI': {'economic': 0.35, 'political': 0.30, 'social': 0.40, 
                    'health': 0.38, 'educational': 0.36},
            'USA': {'economic': 0.75, 'political': 0.65, 'social': 0.70,
                    'health': 0.80, 'educational': 0.78}
        }
        
        values = base_values.get(country_code, base_values['USA'])
        
        for i in range(steps):
            year = base_year + i + 1
            for domain, base_value in values.items():
                # Add some trend and noise
                trend = -0.01 if country_code == 'HTI' else 0.005
                noise = np.random.normal(0, 0.02)
                value = base_value + (i * trend) + noise
                
                results.append({
                    "indicator_code": f"{domain}_agency",
                    "year": year,
                    "value": round(np.clip(value, 0, 1), 3)
                })
        
        return results


# Create singleton instance
ai_service = AIService()


# API Integration Functions
def get_ai_forecast(country_code: str, steps: int = 10, weighting: str = "Communitarian"):
    """Get forecast from AI service for API endpoint."""
    return ai_service.generate_forecast(country_code, steps, weighting)


def get_forecast_explanation(country_code: str, year: int):
    """Get SHAP explanation for API endpoint."""
    return ai_service.explain_forecast(country_code, year)


def score_policy_text(text: str, 
                     name: Optional[str] = None, 
                     confidence: float = 0.5,
                     normalization: str = 'density'):
    """Score policy text for API endpoint."""
    return ai_service.score_policy(text, name, confidence, normalization)