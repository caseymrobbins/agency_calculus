from pathlib import Path
import sys
import logging
import pandas as pd
import yaml
import json
import numpy as np
import os  # Added import os
from sqlalchemy import text  # Added for safe query execution

# Add project root to path to allow module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ai.hybrid_forecaster import HybridForecaster
from api.database import get_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'ingestion/config.yaml') -> dict:
    """Loads and validates the main configuration file."""
    try:
        full_config_path = PROJECT_ROOT / config_path
        with open(full_config_path, 'r') as f:
            config_str = f.read()
        config_str = os.path.expandvars(config_str)
        config = yaml.safe_load(config_str)
        if not config:
            raise ValueError("Configuration file is empty")
        return config
    except Exception as e:
        logger.critical(f"FATAL: Could not load config from {full_config_path}: {e}")
        raise

def load_training_data(country_code: str) -> tuple:
    """Loads and prepares training data for a country from the database."""
    logger.info(f"Loading and preparing training data for {country_code}...")
    with get_db() as db:
        query = text("""
            SELECT year, indicator_code, value
            FROM observations
            WHERE country_code = :country_code AND dataset_version LIKE 'WB-API%'
            ORDER BY year
        """)
        result = db.execute(query, {'country_code': country_code})
        rows = result.fetchall()
        columns = result.keys()
        df = pd.DataFrame(rows, columns=columns)
    
    if df.empty:
        logger.error(f"No data found for {country_code} in observations table.")
        raise ValueError(f"No data available for {country_code}")

    # Pivot data to have indicators as columns, years as index
    df_pivot = df.pivot(index='year', columns='indicator_code', values='value')
    df_pivot = df_pivot.apply(pd.to_numeric, errors='coerce')
    df_pivot.index = pd.to_datetime(df_pivot.index, format='%Y')

    # Define agency mappings based on Agency Calculus 4.3 domains
    agency_mappings = {
        'economic_agency': ['NY.GDP.MKTP.CD', 'NY.GDP.PCAP.CD', 'SI.POV.GINI'],
        'health_agency': ['SP.DYN.LE00.IN', 'SP.DYN.IMRT.IN', 'SH.XPD.GHED.GD.ZS'],
        'educational_agency': ['SE.ADT.LITR.ZS', 'SE.XPD.TOTL.GD.ZS']
    }

    # Aggregate indicators into agency scores (simple mean for now)
    endog_data = {}
    for agency, indicators in agency_mappings.items():
        normalized_cols = []
        for ind in indicators:
            if ind in df_pivot.columns:
                col = df_pivot[ind].dropna()
                if not col.empty and col.std() > 1e-10:  # Skip constant or empty
                    min_val = col.min()
                    max_val = col.max()
                    normalized = (df_pivot[ind] - min_val) / (max_val - min_val)
                    normalized_cols.append(normalized)
        if normalized_cols:
            endog_data[agency] = pd.concat(normalized_cols, axis=1).mean(axis=1).fillna(0.5)
        # Skip if no data, no default constant

    endog_df = pd.DataFrame(endog_data, index=df_pivot.index).clip(0, 1)
    endog_df = endog_df.loc[:, endog_df.std() > 1e-10]  # Remove constant columns

    if endog_df.empty:
        logger.error(f"No valid endog data for {country_code} after processing.")
        raise ValueError(f"No valid endog data for {country_code}")

    # Placeholder exogenous variables (replace with real data if available)
    exog_data = {
        'shock_magnitude': np.random.exponential(0.1, len(endog_df)) + np.random.normal(0, 0.01, len(endog_df)),  # Add noise
        'recovery_slope': np.random.uniform(-0.1, 0.1, len(endog_df)) + np.random.normal(0, 0.01, len(endog_df)),
        'time_since_shock': np.random.randint(0, 10, len(endog_df)) + np.random.normal(0, 0.01, len(endog_df))
    }
    exog_df = pd.DataFrame(exog_data, index=endog_df.index)
    
    logger.info(f"Loaded {len(endog_df)} data points for {country_code} with {len(endog_df.columns)} agencies.")
    return endog_df, exog_df

def main():
    """Main training script."""
    logger.info("--- Starting Model Training Pipeline ---")
    config = load_config()
    countries = config['world_bank']['countries']  # Updated to match ETL config
    # Hardcode model_config since not in config.yaml
    model_config = {
        'max_lags': 3,
        'xgb_params': {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100
        }
    }
    model_dir = Path('models')  # Hardcode directory since not in config
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
                'training_samples': int(len(endog_df)),
                'lag_order': int(forecaster.lag_order_),
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