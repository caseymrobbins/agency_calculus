# ai/brittleness_predictor.py

"""
Component: Brittleness Predictor
Purpose: Core predictive model using XGBoost to predict societal brittleness scores (0-10 scale).
         Trained on historical collapse data and engineered features.
         
Inputs: 
- Featured data from feature_engineering.py
- Real-time snapshots from realtime_aggregator.py

Outputs:
- Brittleness score (0-10)
- Confidence intervals
- Feature importance
- Risk factors breakdown

Integration: Central prediction engine for the Agency Monitor system
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class BrittlenessPrediction:
    """Structured prediction output"""
    country_code: str
    timestamp: str
    brittleness_score: float
    confidence_interval: Tuple[float, float]
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    top_risk_factors: List[Dict[str, float]]
    trajectory: str  # IMPROVING, STABLE, DETERIORATING, CRITICAL_DECLINE
    days_to_critical: Optional[int]  # Estimated days until brittleness > 8
    
    def to_dict(self) -> Dict:
        return {
            'country_code': self.country_code,
            'timestamp': self.timestamp,
            'brittleness_score': round(self.brittleness_score, 2),
            'confidence_interval': [round(ci, 2) for ci in self.confidence_interval],
            'risk_level': self.risk_level,
            'top_risk_factors': self.top_risk_factors,
            'trajectory': self.trajectory,
            'days_to_critical': self.days_to_critical
        }


class BrittlenessPredictor:
    """
    XGBoost-based predictor for societal brittleness.
    Uses ensemble of models for robustness and uncertainty quantification.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Model ensemble for uncertainty quantification
        self.models = []
        self.n_estimators = 5  # Number of models in ensemble
        
        # Feature names for consistency
        self.feature_names = None
        self.feature_importance = None
        
        # Scaler for normalization
        self.scaler = StandardScaler()
        
        # Risk thresholds
        self.risk_thresholds = {
            'LOW': 3.0,
            'MEDIUM': 5.0,
            'HIGH': 7.0,
            'CRITICAL': 8.0
        }
        
        # Load pre-trained model if provided
        if model_path:
            self.load_model(model_path)
    
    def train(self, 
              train_data_path: str,
              target_col: str = 'brittleness_score',
              test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the brittleness prediction model.
        Uses time series cross-validation for robust evaluation.
        """
        self.logger.info("Starting model training...")
        
        # Load training data
        with open(train_data_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['data'])
        
        # Prepare features and target
        # Assuming target is pre-calculated from historical data
        if target_col not in df.columns:
            # Calculate target from agency decline rate and volatility
            df[target_col] = self._calculate_brittleness_target(df)
        
        # Feature selection
        exclude_cols = ['country', 'date', 'year', 'timestamp', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove features with too many missing values
        missing_threshold = 0.3
        feature_cols = [col for col in feature_cols 
                       if df[col].isna().sum() / len(df) < missing_threshold]
        
        self.feature_names = feature_cols
        
        # Prepare data
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        # Remove samples with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train ensemble of models
        self.models = []
        cv_scores = []
        
        for i in range(self.n_estimators):
            self.logger.info(f"Training model {i+1}/{self.n_estimators}")
            
            # XGBoost parameters optimized for brittleness prediction
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42 + i,  # Different seed for each model
                'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**params)
            
            # Cross-validation
            scores = cross_val_score(
                model, X_scaled, y, 
                cv=tscv, 
                scoring='neg_mean_squared_error'
            )
            cv_scores.extend(-scores)
            
            # Train on full data
            model.fit(X_scaled, y)
            self.models.append(model)
        
        # Calculate feature importance (average across ensemble)
        self._calculate_feature_importance()
        
        # Evaluate on hold-out test set
        test_idx = int(len(X) * (1 - test_size))
        X_test = X_scaled[test_idx:]
        y_test = y.iloc[test_idx:]
        
        predictions = self.predict_batch(X[test_idx:])
        y_pred = [p.brittleness_score for p in predictions]
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cv_mse_mean': np.mean(cv_scores),
            'cv_mse_std': np.std(cv_scores)
        }
        
        self.logger.info(f"Training complete. Test MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}")
        
        # Save training metadata
        self.training_metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'metrics': metrics,
            'feature_names': self.feature_names
        }
        
        return metrics
    
    def predict(self, features: Dict[str, float], country_code: str = 'Unknown') -> BrittlenessPrediction:
        """
        Make a single prediction with uncertainty quantification.
        """
        # Convert to DataFrame for consistency
        df = pd.DataFrame([features])
        predictions = self.predict_batch(df, [country_code])
        return predictions[0]
    
    def predict_batch(self, 
                     features_df: pd.DataFrame,
                     country_codes: Optional[List[str]] = None) -> List[BrittlenessPrediction]:
        """
        Make predictions for multiple samples.
        """
        if not self.models:
            raise ValueError("Model not trained or loaded")
        
        # Ensure feature consistency
        X = features_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        
        for idx, row in enumerate(X_scaled):
            # Get predictions from all models in ensemble
            ensemble_preds = []
            for model in self.models:
                pred = model.predict(row.reshape(1, -1))[0]
                # Clip to valid range
                pred = np.clip(pred, 0, 10)
                ensemble_preds.append(pred)
            
            # Calculate mean and confidence interval
            mean_pred = np.mean(ensemble_preds)
            std_pred = np.std(ensemble_preds)
            
            # 95% confidence interval
            ci_lower = max(0, mean_pred - 1.96 * std_pred)
            ci_upper = min(10, mean_pred + 1.96 * std_pred)
            
            # Determine risk level
            risk_level = self._get_risk_level(mean_pred)
            
            # Get top risk factors
            feature_values = dict(zip(self.feature_names, X.iloc[idx]))
            top_factors = self._get_top_risk_factors(feature_values, idx)
            
            # Determine trajectory
            trajectory = self._calculate_trajectory(features_df, idx)
            
            # Estimate days to critical
            days_to_critical = self._estimate_days_to_critical(
                mean_pred, trajectory, features_df, idx
            )
            
            # Create prediction object
            country = country_codes[idx] if country_codes else 'Unknown'
            
            prediction = BrittlenessPrediction(
                country_code=country,
                timestamp=datetime.now().isoformat(),
                brittleness_score=mean_pred,
                confidence_interval=(ci_lower, ci_upper),
                risk_level=risk_level,
                top_risk_factors=top_factors,
                trajectory=trajectory,
                days_to_critical=days_to_critical
            )
            
            predictions.append(prediction)
        
        return predictions
    
    def _calculate_brittleness_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate brittleness score from raw data for training.
        This is a simplified version - in production, use validated historical scores.
        """
        # Brittleness components
        components = []
        
        # 1. Low total agency (inverted)
        if 'total_agency' in df.columns:
            components.append((1 - df['total_agency']) * 10)
        
        # 2. High volatility
        volatility_cols = [col for col in df.columns if 'volatility' in col and 'total' in col]
        if volatility_cols:
            avg_volatility = df[volatility_cols].mean(axis=1)
            components.append(avg_volatility * 10)
        
        # 3. Negative trajectory
        if 'total_agency_delta1' in df.columns:
            components.append((-df['total_agency_delta1'].clip(-1, 0)) * 10)
        
        # 4. System stress
        if 'systemic_stress' in df.columns:
            components.append(df['systemic_stress'] * 10)
        
        # 5. Shock impact
        if 'shock_magnitude' in df.columns:
            components.append(df['shock_magnitude'] * 5)
        
        # Combine components
        if components:
            brittleness = np.mean(components, axis=0)
            # Add some noise for realism
            brittleness += np.random.normal(0, 0.5, len(brittleness))
            return brittleness.clip(0, 10)
        else:
            return pd.Series([5.0] * len(df))
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance from ensemble."""
        if not self.models or not self.feature_names:
            return
        
        # Average importance across all models
        importance_matrix = []
        
        for model in self.models:
            importance = model.feature_importances_
            importance_matrix.append(importance)
        
        avg_importance = np.mean(importance_matrix, axis=0)
        
        # Create importance dictionary
        self.feature_importance = dict(zip(self.feature_names, avg_importance))
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), 
                   key=lambda x: x[1], 
                   reverse=True)
        )
    
    def _get_risk_level(self, brittleness_score: float) -> str:
        """Determine risk level from brittleness score."""
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if brittleness_score >= self.risk_thresholds[level]:
                return level
        return 'LOW'
    
    def _get_top_risk_factors(self, 
                             feature_values: Dict[str, float],
                             sample_idx: int,
                             n_top: int = 5) -> List[Dict[str, float]]:
        """
        Identify top contributing risk factors using SHAP values or feature importance.
        """
        if not self.feature_importance:
            return []
        
        risk_factors = []
        
        # For now, use feature importance * feature value as proxy
        # In production, use SHAP for better explanations
        for feature, importance in list(self.feature_importance.items())[:n_top * 2]:
            if feature in feature_values:
                value = feature_values[feature]
                
                # Only include if contributing to risk (value > 0 for risk features)
                if self._is_risk_feature(feature) and value > 0:
                    contribution = importance * abs(value)
                    risk_factors.append({
                        'feature': feature,
                        'value': round(value, 3),
                        'contribution': round(contribution, 3)
                    })
        
        # Sort by contribution and take top N
        risk_factors.sort(key=lambda x: x['contribution'], reverse=True)
        return risk_factors[:n_top]
    
    def _is_risk_feature(self, feature_name: str) -> bool:
        """Determine if a feature contributes to risk (vs resilience)."""
        risk_indicators = [
            'volatility', 'delta1_negative', 'shock', 'crisis', 
            'risk', 'stress', 'decline', 'low_agency'
        ]
        
        resilience_indicators = [
            'recovery', 'buffer', 'resilience', 'stability'
        ]
        
        feature_lower = feature_name.lower()
        
        # Check if it's a risk feature
        for indicator in risk_indicators:
            if indicator in feature_lower:
                return True
        
        # Check if it's inverted (resilience feature)
        for indicator in resilience_indicators:
            if indicator in feature_lower:
                return False
        
        # Default to risk feature
        return True
    
    def _calculate_trajectory(self, features_df: pd.DataFrame, idx: int) -> str:
        """
        Determine trajectory based on recent trends.
        """
        # Look for delta and trend features
        trend_features = [col for col in self.feature_names 
                         if 'trend' in col or 'delta' in col]
        
        if not trend_features:
            return 'STABLE'
        
        # Get trend values
        trends = []
        for feature in trend_features:
            if feature in features_df.columns:
                value = features_df.iloc[idx][feature]
                if not pd.isna(value):
                    trends.append(value)
        
        if not trends:
            return 'STABLE'
        
        avg_trend = np.mean(trends)
        
        # Classify trajectory
        if avg_trend < -0.1:
            return 'CRITICAL_DECLINE'
        elif avg_trend < -0.05:
            return 'DETERIORATING'
        elif avg_trend > 0.05:
            return 'IMPROVING'
        else:
            return 'STABLE'
    
    def _estimate_days_to_critical(self,
                                  current_brittleness: float,
                                  trajectory: str,
                                  features_df: pd.DataFrame,
                                  idx: int) -> Optional[int]:
        """
        Estimate days until brittleness reaches critical level (8.0).
        """
        if current_brittleness >= 8.0:
            return 0
        
        if trajectory in ['IMPROVING', 'STABLE']:
            return None
        
        # Estimate rate of change
        rate_features = [col for col in self.feature_names if 'delta1' in col]
        if not rate_features:
            return None
        
        # Calculate average rate of brittleness increase
        rates = []
        for feature in rate_features:
            if feature in features_df.columns:
                value = features_df.iloc[idx][feature]
                if not pd.isna(value) and value < 0:  # Negative = increasing brittleness
                    rates.append(abs(value))
        
        if not rates:
            return None
        
        avg_rate = np.mean(rates) * 10  # Scale to brittleness units
        
        if avg_rate <= 0:
            return None
        
        # Simple linear projection
        distance_to_critical = 8.0 - current_brittleness
        days_to_critical = int(distance_to_critical / (avg_rate / 365))
        
        # Cap at 2 years
        return min(days_to_critical, 730) if days_to_critical > 0 else None
    
    def explain_prediction(self, 
                          features: Dict[str, float],
                          plot: bool = True) -> Dict[str, Any]:
        """
        Generate detailed explanation for a prediction using SHAP.
        """
        if not self.models:
            raise ValueError("Model not trained or loaded")
        
        # Prepare features
        df = pd.DataFrame([features])
        X = df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Use first model for SHAP (representative)
        model = self.models[0]
        
        # Create SHAP explainer
        explainer = shap.Explainer(model, X_scaled)
        shap_values = explainer(X_scaled)
        
        # Get prediction
        prediction = model.predict(X_scaled)[0]
        
        # Extract SHAP values
        feature_impacts = {}
        for i, feature in enumerate(self.feature_names):
            feature_impacts[feature] = {
                'value': X.iloc[0, i],
                'impact': shap_values.values[0, i],
                'abs_impact': abs(shap_values.values[0, i])
            }
        
        # Sort by absolute impact
        sorted_impacts = dict(sorted(
            feature_impacts.items(),
            key=lambda x: x[1]['abs_impact'],
            reverse=True
        ))
        
        # Plot if requested
        if plot:
            # Waterfall plot
            shap.waterfall_plot(shap_values[0], max_display=15)
            plt.title(f'Brittleness Score Explanation (Prediction: {prediction:.2f})')
            plt.tight_layout()
            plt.show()
        
        explanation = {
            'prediction': prediction,
            'base_value': explainer.expected_value,
            'feature_impacts': sorted_impacts,
            'top_increasing_factors': [
                (k, v) for k, v in sorted_impacts.items() 
                if v['impact'] > 0
            ][:5],
            'top_decreasing_factors': [
                (k, v) for k, v in sorted_impacts.items() 
                if v['impact'] < 0
            ][:5]
        }
        
        return explanation
    
    def save_model(self, path: str):
        """Save the trained model ensemble."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'risk_thresholds': self.risk_thresholds,
            'training_metadata': getattr(self, 'training_metadata', {})
        }
        
        joblib.dump(model_data, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a pre-trained model ensemble."""
        model_data = joblib.load(path)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.risk_thresholds = model_data.get('risk_thresholds', self.risk_thresholds)
        self.training_metadata = model_data.get('training_metadata', {})
        
        self.logger.info(f"Model loaded from {path}")
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot top feature importances."""
        if not self.feature_importance:
            self.logger.warning("No feature importance data available")
            return
        
        # Get top features
        top_features = list(self.feature_importance.items())[:top_n]
        features, importances = zip(*top_features)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(features))
        
        plt.barh(y_pos, importances)
        plt.yticks(y_pos, features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features for Brittleness Prediction')
        plt.tight_layout()
        plt.show()


# --- Integration Functions ---

async def predict_realtime(country_code: str, 
                          aggregator,
                          predictor: BrittlenessPredictor) -> BrittlenessPrediction:
    """
    Make real-time prediction by combining aggregator and predictor.
    """
    # Get latest snapshot
    snapshot = await aggregator.get_latest_snapshot(country_code)
    
    if not snapshot:
        raise ValueError(f"Could not get snapshot for {country_code}")
    
    # Convert snapshot to features
    # In production, this would include historical features too
    features = {
        'total_agency': snapshot.total_agency,
        'total_agency_delta1': snapshot.agency_delta,
        'total_agency_volatility_short': snapshot.volatility_7d,
        'economic': snapshot.economic_agency,
        'political': snapshot.political_agency,
        'social': snapshot.social_agency,
        'health': snapshot.health_agency,
        'educational': snapshot.educational_agency,
        # Add more features as needed
    }
    
    # Make prediction
    prediction = predictor.predict(features, country_code)
    
    return prediction


# --- Example Usage ---

def main():
    # Initialize predictor
    predictor = BrittlenessPredictor()
    
    # Example: Train on historical data
    # predictor.train('featured_training_data.json')
    
    # Example: Make a prediction
    sample_features = {
        'total_agency': 0.35,
        'total_agency_volatility_short': 0.18,
        'total_agency_delta1': -0.05,
        'economic': 0.31,
        'political': 0.28,
        'social': 0.42,
        'health': 0.36,
        'educational': 0.34,
        'systemic_stress': 0.72,
        'cascade_risk': 0.6,
        'shock_magnitude': 0.15,
        'polarization_index': 0.8
    }
    
    prediction = predictor.predict(sample_features, 'HT')
    
    print("\n=== BRITTLENESS PREDICTION ===")
    print(f"Country: {prediction.country_code}")
    print(f"Brittleness Score: {prediction.brittleness_score:.2f}/10")
    print(f"Confidence Interval: [{prediction.confidence_interval[0]:.2f}, {prediction.confidence_interval[1]:.2f}]")
    print(f"Risk Level: {prediction.risk_level}")
    print(f"Trajectory: {prediction.trajectory}")
    
    if prediction.days_to_critical:
        print(f"⚠️  Days to Critical: {prediction.days_to_critical}")
    
    print("\nTop Risk Factors:")
    for factor in prediction.top_risk_factors:
        print(f"  - {factor['feature']}: {factor['value']:.3f} (contribution: {factor['contribution']:.3f})")
    
    # Save model
    # predictor.save_model('models/brittleness_predictor_v1.pkl')


if __name__ == "__main__":
    main()