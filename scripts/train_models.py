# scripts/train_models.py
"""
Training Script for Hybrid Forecasting Models
Trains a HybridForecaster for each country using historical data
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.hybrid_forecaster import HybridForecaster
from api.ai_integration import FeatureEngineer
from api.database import get_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training of hybrid forecasting models."""
    
    def __init__(self, model_dir: str = 'models/'):
        self.model_dir = model_dir
        self.feature_engineer = FeatureEngineer()
        os.makedirs(model_dir, exist_ok=True)
    
    def load_training_data(self, country_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare training data for a country.
        
        Returns:
            tuple: (endog_df, exog_df) DataFrames
        """
        logger.info(f"Loading training data for {country_code}")
        
        with get_db() as db:
            # Load agency scores
            query = """
                SELECT year, economic_agency, political_agency, social_agency,
                       health_agency, educational_agency
                FROM agency_scores
                WHERE country_code = %s
                ORDER BY year
            """
            agency_df = pd.read_sql_query(query, db, params=(country_code,))
            
            if agency_df.empty:
                raise ValueError(f"No agency score data found for {country_code}")
            
            # Create datetime index
            dates = pd.to_datetime([f"{y}-01-01" for y in agency_df['year']])
            
            # Create endogenous dataframe
            endog_columns = ['economic_agency', 'political_agency', 'social_agency',
                           'health_agency', 'educational_agency']
            endog_df = pd.DataFrame({
                col: agency_df[col].values for col in endog_columns
            }, index=dates)
            
            # Detect shocks
            shock_dates = self.feature_engineer.detect_shocks(endog_df)
            logger.info(f"Detected {len(shock_dates)} shocks in {country_code} data")
            
            # Engineer features
            features_df = self.feature_engineer.engineer_shock_features(endog_df, shock_dates)
            
            # Load additional economic indicators
            indicator_query = """
                SELECT 
                    o.year,
                    MAX(CASE WHEN o.indicator_code = 'FP.CPI.TOTL.ZG' THEN o.value END) as inflation_rate,
                    MAX(CASE WHEN o.indicator_code = 'NY.GDP.MKTP.KD.ZG' THEN o.value END) as gdp_growth,
                    MAX(CASE WHEN o.indicator_code = 'SI.POV.GINI' THEN o.value END) as gini_index
                FROM observations o
                WHERE o.country_code = %s
                GROUP BY o.year
                ORDER BY o.year
            """
            indicators_df = pd.read_sql_query(indicator_query, db, params=(country_code,))
            
            # Merge indicators
            for _, row in indicators_df.iterrows():
                year_date = pd.to_datetime(f"{int(row['year'])}-01-01")
                if year_date in features_df.index:
                    for col in ['inflation_rate', 'gdp_growth', 'gini_index']:
                        if pd.notna(row[col]):
                            features_df.loc[year_date, col] = row[col]
        
        # Normalize features
        features_df['inflation_rate'] = features_df.get('inflation_rate', 0) / 100
        features_df['gdp_growth'] = features_df.get('gdp_growth', 0) / 100
        features_df['gini_index'] = features_df.get('gini_index', 50) / 100
        
        # Fill missing values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Separate exogenous features
        exog_columns = [col for col in features_df.columns if col not in endog_columns]
        exog_df = features_df[exog_columns]
        
        logger.info(f"Loaded {len(endog_df)} observations with {len(exog_columns)} features")
        
        return endog_df, exog_df
    
    def evaluate_model(self, 
                      model: HybridForecaster,
                      test_endog: pd.DataFrame,
                      test_exog: pd.DataFrame) -> dict:
        """
        Evaluate model performance on test data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Make predictions
        steps = len(test_endog)
        predictions = model.predict(steps=steps, future_exog_df=test_exog)
        
        # Calculate metrics for each domain
        for col in test_endog.columns:
            y_true = test_endog[col].values
            y_pred = predictions[col].values
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # Normalized metrics (since values are 0-1)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            
            metrics[col] = {
                'mse': round(mse, 6),
                'mae': round(mae, 6),
                'rmse': round(rmse, 6),
                'mape': round(mape, 2)
            }
        
        # Overall metrics
        overall_mse = np.mean([m['mse'] for m in metrics.values()])
        overall_mae = np.mean([m['mae'] for m in metrics.values()])
        
        metrics['overall'] = {
            'mse': round(overall_mse, 6),
            'mae': round(overall_mae, 6),
            'rmse': round(np.sqrt(overall_mse), 6)
        }
        
        return metrics
    
    def train_country_model(self, 
                           country_code: str,
                           test_size: int = 5,
                           optimize_hyperparams: bool = False) -> dict:
        """
        Train a hybrid forecasting model for a specific country.
        
        Args:
            country_code: Country to train model for
            test_size: Number of years to hold out for testing
            optimize_hyperparams: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model for {country_code}")
        logger.info(f"{'='*50}")
        
        # Load data
        endog_df, exog_df = self.load_training_data(country_code)
        
        # Train/test split
        split_idx = len(endog_df) - test_size
        train_endog = endog_df.iloc[:split_idx]
        train_exog = exog_df.iloc[:split_idx]
        test_endog = endog_df.iloc[split_idx:]
        test_exog = exog_df.iloc[split_idx:]
        
        logger.info(f"Training samples: {len(train_endog)}")
        logger.info(f"Test samples: {len(test_endog)}")
        
        # Initialize model
        xgb_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,  # Reduced to handle correlated features
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        if optimize_hyperparams:
            logger.info("Performing hyperparameter optimization...")
            xgb_params = self._optimize_hyperparameters(
                train_endog, train_exog, xgb_params
            )
        
        # Train model
        model = HybridForecaster(max_lags=8, xgb_params=xgb_params)
        model.fit(train_endog, train_exog)
        
        # Get residual diagnostics
        diagnostics = model.get_residual_diagnostics()
        logger.info("\nResidual Diagnostics:")
        logger.info(f"  Stationary residuals: {sum(diagnostics['residual_stationarity'].values())}/{len(endog_df.columns)}")
        logger.info(f"  Autocorrelation issues: {sum(d['has_autocorrelation'] for d in diagnostics['ljung_box_tests'].values())}/{len(endog_df.columns)}")
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        metrics = self.evaluate_model(model, test_endog, test_exog)
        
        # Display results
        logger.info("\nTest Set Performance:")
        for domain, domain_metrics in metrics.items():
            if domain != 'overall':
                logger.info(f"  {domain}:")
                logger.info(f"    RMSE: {domain_metrics['rmse']:.4f}")
                logger.info(f"    MAE:  {domain_metrics['mae']:.4f}")
                logger.info(f"    MAPE: {domain_metrics['mape']:.1f}%")
        
        logger.info(f"\n  Overall:")
        logger.info(f"    RMSE: {metrics['overall']['rmse']:.4f}")
        logger.info(f"    MAE:  {metrics['overall']['mae']:.4f}")
        
        # Save model
        model_path = os.path.join(self.model_dir, f"{country_code.lower()}_hybrid_forecaster.pkl")
        model.save_model(model_path)
        logger.info(f"\nModel saved to: {model_path}")
        
        # Return results
        return {
            'country_code': country_code,
            'training_samples': len(train_endog),
            'test_samples': len(test_endog),
            'lag_order': model.lag_order_,
            'metrics': metrics,
            'diagnostics': diagnostics,
            'model_path': model_path
        }
    
    def _optimize_hyperparameters(self, 
                                 train_endog: pd.DataFrame,
                                 train_exog: pd.DataFrame,
                                 base_params: dict) -> dict:
        """
        Simple grid search for XGBoost hyperparameters.
        
        Returns:
            Optimized parameters dictionary
        """
        # Define search space (simplified)
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9]
        }
        
        best_score = float('inf')
        best_params = base_params.copy()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search (simplified - in production use GridSearchCV or Optuna)
        for n_est in param_grid['n_estimators']:
            for max_d in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    for subsamp in param_grid['subsample']:
                        params = base_params.copy()
                        params.update({
                            'n_estimators': n_est,
                            'max_depth': max_d,
                            'learning_rate': lr,
                            'subsample': subsamp
                        })
                        
                        # Cross-validation
                        cv_scores = []
                        for train_idx, val_idx in tscv.split(train_endog):
                            # Split data
                            cv_train_endog = train_endog.iloc[train_idx]
                            cv_train_exog = train_exog.iloc[train_idx]
                            cv_val_endog = train_endog.iloc[val_idx]
                            cv_val_exog = train_exog.iloc[val_idx]
                            
                            # Train model
                            model = HybridForecaster(max_lags=5, xgb_params=params)
                            model.fit(cv_train_endog, cv_train_exog)
                            
                            # Evaluate
                            predictions = model.predict(
                                steps=len(cv_val_endog),
                                future_exog_df=cv_val_exog
                            )
                            
                            # Calculate MSE
                            mse = mean_squared_error(
                                cv_val_endog.values.flatten(),
                                predictions.values.flatten()
                            )
                            cv_scores.append(mse)
                        
                        # Average CV score
                        avg_score = np.mean(cv_scores)
                        
                        if avg_score < best_score:
                            best_score = avg_score
                            best_params = params.copy()
        
        logger.info(f"Best CV score: {best_score:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def train_all_models(self, countries: list = None):
        """
        Train models for all specified countries.
        
        Args:
            countries: List of country codes (default: USA, HTI)
        """
        if countries is None:
            countries = ['USA', 'HTI']
        
        results = {}
        
        for country in countries:
            try:
                result = self.train_country_model(
                    country,
                    test_size=5,
                    optimize_hyperparams=False  # Set to True for better models
                )
                results[country] = result
                
            except Exception as e:
                logger.error(f"Failed to train model for {country}: {e}")
                results[country] = {'error': str(e)}
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("TRAINING SUMMARY")
        logger.info("="*50)
        
        for country, result in results.items():
            if 'error' in result:
                logger.info(f"{country}: FAILED - {result['error']}")
            else:
                logger.info(f"{country}: SUCCESS")
                logger.info(f"  - Overall RMSE: {result['metrics']['overall']['rmse']:.4f}")
                logger.info(f"  - Model saved to: {result['model_path']}")
        
        return results


def main():
    """Main training script."""
    from dotenv import load_dotenv
    load_dotenv()
    
    trainer = ModelTrainer()
    
    # Train models for all countries
    results = trainer.train_all_models(['USA', 'HTI'])
    
    # Generate summary report
    report_path = 'models/training_report.json'
    with open(report_path, 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nTraining report saved to: {report_path}")


if __name__ == "__main__":
    main()