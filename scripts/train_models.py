import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Added to fix import path for 'ai'

import logging
import json
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
import yaml
from ai.hybrid_forecaster import HybridForecaster
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='ingestion/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_training_data(country_code):
    config = load_config()
    engine = create_engine(config['database']['url'])
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        # Query all indicators for the country (no specific filter on dataset_version or indicators; pivot for endog/exog)
        stmt = text("""
            SELECT year, indicator_code, value 
            FROM observations 
            WHERE country_code = :country_code
            ORDER BY year, indicator_code
        """)
        df = pd.read_sql(stmt, db.bind, params={'country_code': country_code})

        if df.empty:
            raise ValueError(f"No data available for {country_code}")

        # Pivot to wide format: years as rows, indicators as columns
        df_pivoted = df.pivot(index='year', columns='indicator_code', values='value')

        # Define endog (target vars, e.g., core agency indices) and exog (features, e.g., freedoms/equality)
        endog_columns = [col for col in df_pivoted.columns if col in ['v2x_polyarchy', 'v2x_libdem', 'v2x_egaldem']]  # Example targets
        exog_columns = [col for col in df_pivoted.columns if col not in endog_columns]  # All others as features

        endog_df = df_pivoted[endog_columns].dropna()  # Drop NaN years
        exog_df = df_pivoted[exog_columns].reindex(endog_df.index).fillna(0)  # Align and fill NaNs

        logging.info(f"Loaded {len(endog_df)} rows for {country_code} (endog: {endog_columns}, exog: {len(exog_columns)} columns)")
        return endog_df, exog_df

    finally:
        db.close()

def train_model(country_code):
    endog_df, exog_df = load_training_data(country_code)
    model = HybridForecaster(var_order=1, max_iter=100)  # Adjust params as needed
    model.fit(endog_df, exog_df)

    model_path = f"models/{country_code}_hybrid_model.pkl"
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"Model for {country_code} saved to {model_path}")

    return {
        "country": country_code,
        "training_date": datetime.now().isoformat(),
        "num_observations": len(endog_df),
        "endog_vars": endog_df.columns.tolist(),
        "exog_vars": exog_df.columns.tolist(),
        "model_path": model_path
    }

def main():
    logging.info("--- Starting Model Training Pipeline ---")
    config = load_config()
    countries = config['etl']['countries']  # ['USA', 'HTI']

    report = []
    for country in countries:
        logging.info(f"\n{'='*50}\nTraining model for {country}\n{'='*50}")
        logging.info(f"Loading and preparing training data for {country}...")
        try:
            country_report = train_model(country)
            report.append(country_report)
        except Exception as e:
            logging.error(f"Failed to train model for {country}: {str(e)}")

    logging.info("\n--- Model Training Pipeline Complete. Report saved to models/training_report.json ---")
    with open('models/training_report.json', 'w') as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    main()