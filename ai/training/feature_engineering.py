# ai/training/feature_engineering.py

"""
Component: Feature Engineering
Purpose: Ingests clean country-year data and engineers a rich feature set
         for the brittleness prediction model.
"""
import json
import pandas as pd
import numpy as np
import ruptures as rpt
import logging
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    """Orchestrates the entire feature engineering pipeline."""

    def __init__(self, input_path: str, output_path: Optional[str] = None):
        self.input_path = input_path
        self.output_path = output_path
        self.df = self._load_data()
        self.domain_cols = ['economic', 'political', 'social', 'health', 'educational']
        self.windows = {'short': 3, 'medium': 7, 'long': 15}

    def _load_data(self) -> pd.DataFrame:
        """Loads and prepares the initial dataset."""
        logging.info(f"Loading data from {self.input_path}")
        with open(self.input_path, 'r') as f:
            data = json.load(f)

        records: List[Dict[str, Any]] = []
        default_counts = 0
        for country_code, snapshots in data.items():
            if not snapshots:
                logging.warning(f"No snapshots found for country: {country_code}")
                continue
            for snapshot in snapshots:
                if 'timestamp' not in snapshot:
                    raise ValueError(f"Missing 'timestamp' in data for {country_code}")

                record = {
                    'country_code': country_code,
                    'timestamp': snapshot['timestamp'],
                    'year': pd.to_datetime(snapshot['timestamp']).year
                }

                for domain in self.domain_cols:
                    col_name = f'{domain}_agency'
                    if col_name not in snapshot or snapshot[col_name] is None:
                        record[domain] = 0.5
                        default_counts += 1
                    else:
                        record[domain] = snapshot[col_name]

                records.append(record)

        if default_counts > 0:
            logging.warning(f"{default_counts} records used a default agency score of 0.5 due to missing data.")

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['year'], format='%Y')
        df = df.sort_values(by=['country_code', 'date']).reset_index(drop=True)
        df['total_agency'] = df[self.domain_cols].mean(axis=1)
        logging.info(f"Loaded {len(df)} records for {df['country_code'].nunique()} countries")
        return df

    def run_pipeline(self) -> pd.DataFrame:
        """Executes all feature engineering steps and returns the DataFrame."""
        logging.info("Starting feature engineering pipeline...")
        # In a full implementation, the feature creation methods would be called here.
        # self.df = self.create_ts_features(self.df)
        # self.df = self.create_correlation_features(self.df)
        # ...etc.

        if self.output_path:
            self._clean_and_save()
        else:
            self.df = self.df.fillna(0) # Clean for in-memory use

        logging.info("Feature engineering pipeline complete.")
        return self.df

    def _clean_and_save(self):
        """Cleans the dataset and saves it to the output file."""
        logging.info(f"Cleaning and saving engineered features to {self.output_path}...")
        self.df.fillna(0, inplace=True)
        
        output_data = {
            'metadata': {
                'n_features': len(self.df.columns),
                'n_samples': len(self.df),
                'n_countries': self.df['country_code'].nunique()
            },
            'data': self.df.to_dict(orient='records')
        }
        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        logging.info(f"Save complete for {len(self.df)} records.")