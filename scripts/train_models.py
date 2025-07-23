# scripts/train_models.py
"""
Training Script for Hybrid Forecasting Models
This script orchestrates the training, evaluation, and serialization of
HybridForecaster models for each target country defined in the configuration.
It represents a key component of the MLOps workflow.
"""
import os
import sys
import logging
import pandas as pd
import yaml
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents))

from ai.hybrid_forecaster import HybridForecaster
from api.database import get_db # Assuming a function to get DB connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config/config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_training_data(country_code: str) -> tuple:
    """Loads and prepares training data for a country from the database."""
    logger.info(f"Loading and preparing training data for {country_code}...")
    # This is a placeholder for a robust data loading function from the database.
    # In a real system, this would query the 'agency_scores' and 'observations' tables.
    # For this example, we'll create synthetic data.
    np.random.seed(hash(country_code) % (2**32 - 1))
    dates = pd.to_datetime(pd.date_range(start='1990-12-31', end='2023-12-31', freq='A-DEC'))
    n_years = len(dates)
    
    endog_data = {
        'economic_agency': 0.5 + 0.005 * np.arange(n_years) + np.random.normal(0, 0.05, n_years),
        'political_agency': 0.6 - 0.002 * np.arange(n_years) + np.random.normal(0, 0.05, n_years),
        'social_agency': 0.55 + np.random.normal(0, 0.05, n_years),
        'health_agency': 0.7 + 0.008 * np.arange(n_years) + np.random.normal(0, 0.05, n_years),
        'educational_agency': 0.65 + 0.006 * np.arange(n_years) + np.random.normal(0, 0.05, n_years)
    }
    endog_df = pd.DataFrame(endog_data, index=dates).clip(0, 1)

    exog_data = {
        'shock_magnitude': np.random.exponential(0.1, n_years),
        'recovery_slope': np.random.uniform(-0.1, 0.1, n_years),
        'time_since_shock': np.random.randint(0, 10, n_years)
    }
    exog_df = pd.DataFrame(exog_data, index=dates)
    
    logger.info(f"Loaded {len(endog_df)} data points for {country_code}.")
    return endog_df, exog_df

def main():
    """Main training script."""
    logger.info("--- Starting Model Training Pipeline ---")
    config = load_config()
    countries = config['ingestion']['target_countries']
    model_config = config['models']['hybrid_forecaster']
    model_dir = Path(config['models']['directory'])
    model_dir.mkdir(exist_ok=True)

    training_report = {}

    for country in countries:
        try:
            logger.info(f"\n{'='*50}\nTraining model for {country}\n{'='*50}")
            
            # 1. Load Data
            endog_df, exog_df = load_training_data(country)

            # 2. Initialize and Fit Model
            forecaster = HybridForecaster(
                max_lags=model_config['max_lags'],
                xgb_params=model_config['xgb_params']
            )
            forecaster.fit(endog_df, exog_df)

            # 3. Save Model
            model_path = model_dir / f"{country.lower()}_hybrid_forecaster.pkl"
            forecaster.save_model(str(model_path))
            
            training_report[country] = {
                'status': 'SUCCESS',
                'model_path': str(model_path),
                'training_samples': len(endog_df),
                'lag_order': forecaster.lag_order_,
                'differenced_columns': [k for k, v in forecaster.differenced_columns_.items() if v]
            }

        except Exception as e:
            logger.error(f"Failed to train model for {country}: {e}", exc_info=True)
            training_report[country] = {'status': 'FAILURE', 'error': str(e)}

    # 4. Save Training Report
    report_path = model_dir / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(training_report, f, indent=2)
    
    logger.info(f"\n--- Model Training Pipeline Complete. Report saved to {report_path} ---")

if __name__ == "__main__":
    main()