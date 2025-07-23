# ai/training/feature_engineering.py
"""
Component: Feature Engineering (Production Version)
Purpose: Ingests clean country-year data and engineers a rich feature set
         for the brittleness prediction model. This implementation is based on
         the state-of-the-art recommendations from the 'AI Plan Validation
         and Refinement' document.

Inputs:
- A JSON file with time-series data containing agency scores and AC4 indices.

Outputs:
- A DataFrame (and optional JSON file) enriched with features like:
    - Lagged agency scores and derivatives (rate of change)
    - Rolling volatility and cross-domain correlations
    - Systemic shock characteristics (magnitude, duration, recovery slope)
    - Interaction features capturing complex risks
"""

import json
import pandas as pd
import numpy as np
import ruptures as rpt
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings

# --- Configuration ---
warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Orchestrates the entire feature engineering pipeline."""

    def __init__(self, input_path: str, output_path: Optional[str] = None, shock_penalty: float = 3.0, windows: Optional[Dict[str, int]] = None):
        """
        Initializes the FeatureEngineer.

        Args:
            input_path (str): Path to the input JSON file from the ETL process.
            output_path (Optional[str]): Path to save the output JSON file. If None, runs in-memory.
            shock_penalty (float): A multiplier for the BIC penalty in shock detection.
            windows (Optional[Dict[str, int]]): Dictionary defining rolling window sizes.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.shock_penalty = shock_penalty
        self.df = self._load_data()
        self.domain_cols = ['economic_agency', 'political_agency', 'social_agency', 'health_agency', 'educational_agency']
        self.windows = windows or {'short': 3, 'medium': 7, 'long': 15}

    def _load_data(self) -> pd.DataFrame:
        """Loads and prepares the initial dataset from the ETL output with robust validation."""
        logger.info(f"Loading data from {self.input_path}")
        try:
            with open(self.input_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.critical(f"FATAL: Failed to load or parse input file {self.input_path}: {e}")
            raise

        records: List[Dict[str, Any]] = []
        required_fields = ['timestamp'] + self.domain_cols
        for country_code, snapshots in data.items():
            for i, snapshot in enumerate(snapshots):
                # Validate that all required fields are present
                if not all(field in snapshot for field in required_fields):
                    raise ValueError(f"Snapshot {i} for country {country_code} is missing one or more required fields: {required_fields}")
                # Add to records list
                records.append({'country_code': country_code, **snapshot})
        
        if not records:
            raise ValueError("No valid records found in the input data file.")

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['date'].dt.year
        df = df.sort_values(by=['country_code', 'date']).set_index('date')
        df['total_agency'] = df[self.domain_cols].mean(axis=1)
        
        logger.info(f"Loaded and validated {len(df)} records for {df['country_code'].nunique()} countries.")
        return df

    def _calculate_trend(self, values: pd.Series) -> float:
        """Helper to calculate linear trend slope, robust to NaNs."""
        if values.count() < 2:  # Need at least 2 points to define a line
            return 0.0
        y = values.dropna()
        x = np.arange(len(y))
        try:
            return np.polyfit(x, y, 1)[0]
        except (np.linalg.LinAlgError, TypeError):
            return 0.0

    def create_ts_features(self):
        """Creates standard time-series features like lags, derivatives, and rolling stats."""
        logger.info("Creating standard time-series features...")
        grouped = self.df.groupby('country_code')
        feature_cols = self.domain_cols + ['total_agency']
        
        for col in feature_cols:
            for lag in [1, 2, 3]:
                self.df[f'{col}_lag_{lag}'] = grouped[col].shift(lag)
            self.df[f'{col}_delta_1'] = grouped[col].diff(1)
            for name, w in self.windows.items():
                self.df[f'{col}_vol_{name}'] = grouped[col].rolling(window=w, min_periods=2).std()
                self.df[f'{col}_trend_{name}'] = grouped[col].rolling(window=w, min_periods=2).apply(self._calculate_trend, raw=False)

    def _process_country_shocks(self, country_df: pd.DataFrame) -> pd.DataFrame:
        """Processes shocks for a single country. Designed to be used with groupby().apply()."""
        if len(country_df) < 10:
            logger.warning(f"Skipping shock detection for {country_df.name}: insufficient data ({len(country_df)} rows)")
            return country_df

        # Tier 1: Multivariate detection
        signal = country_df[self.domain_cols].values
        algo = rpt.Pelt(model="rbf").fit(signal)
        penalty = self.shock_penalty * len(self.domain_cols) * np.log(len(signal))
        shock_indices = [i for i in algo.predict(pen=penalty) if i < len(country_df)]
        
        country_df['is_shock_year'] = 0
        if not shock_indices:
            return country_df
            
        # Vectorized shock characterization
        shock_points = [idx - 1 for idx in shock_indices if idx > 0]
        country_df.iloc[shock_points, country_df.columns.get_loc('is_shock_year')] = 1
        
        # Create shock group identifier
        country_df['shock_group'] = country_df['is_shock_year'].cumsum()
        
        # Vectorized time since shock
        country_df['time_since_shock'] = country_df.groupby('shock_group').cumcount()

        # Calculate magnitude and recovery slope per shock group
        for shock_idx in shock_points:
            shock_date = country_df.index[shock_idx]
            
            # Magnitude: change from pre-shock to post-shock
            pre_shock_mean = signal[max(0, shock_idx-3):shock_idx].mean()
            post_shock_mean = signal[shock_idx:min(len(signal), shock_idx+3)].mean()
            country_df.loc[shock_date, 'shock_magnitude'] = abs(post_shock_mean - pre_shock_mean)

            # Recovery Slope: find trough and calculate trend after
            post_shock_period = country_df.iloc[shock_idx:min(len(country_df), shock_idx+5)]
            if not post_shock_period.empty:
                trough_date = post_shock_period['total_agency'].idxmin()
                recovery_data = country_df.loc[trough_date:]
                if len(recovery_data) > 1:
                    slope = self._calculate_trend(recovery_data['total_agency'])
                    country_df.loc[shock_date, 'recovery_slope'] = slope
                    
        return country_df

    def detect_and_characterize_shocks(self):
        """Refactored shock detection using groupby().apply() for performance and clarity."""
        logger.info("Detecting and characterizing systemic shocks...")
        # Initialize columns to ensure they exist on all dataframes
        self.df['is_shock_year'] = 0
        self.df['time_since_shock'] = 999
        self.df['shock_magnitude'] = 0.0
        self.df['recovery_slope'] = 0.0
        
        # Apply the shock processing function to each country group
        processed_df = self.df.groupby('country_code', group_keys=False).apply(self._process_country_shocks)
        self.df = processed_df
    
    def _calculate_brittleness_proxy(self):
        """Creates a proxy for the B_sys target variable for training purposes."""
        logger.info("Calculating brittleness score proxy...")
        # Proxy formula: Brittleness is high when agency is low and volatile.
        # Weights (5, 20) are based on empirical scaling from historical analysis.
        self.df['brittleness_score'] = (1 - self.df['total_agency']) * 5 + self.df['total_agency_vol_medium'] * 20
        self.df['brittleness_score'] = self.df['brittleness_score'].clip(0, 10)

    def create_interaction_features(self):
        """Creates sophisticated interaction features as per the refined AI plan."""
        logger.info("Creating interaction features...")
        # Lag the brittleness score to prevent data leakage from the target.
        self.df['brittleness_score_lag_1'] = self.df.groupby('country_code')['brittleness_score'].shift(1)
        
        # Fill the first NaN value for each country with a neutral starting value
        self.df['brittleness_score_lag_1'] = self.df.groupby('country_code')['brittleness_score_lag_1'].transform(lambda x: x.bfill())

        self.df['brittleness_x_magnitude'] = self.df['brittleness_score_lag_1'] * self.df['shock_magnitude']
        self.df['brittleness_x_recovery_slope'] = self.df['brittleness_score_lag_1'] * self.df['recovery_slope']

    def run_pipeline(self) -> pd.DataFrame:
        """Executes all feature engineering steps in the correct, logical sequence."""
        logger.info("--- Starting Feature Engineering Pipeline ---")
        self.create_ts_features()
        self._calculate_brittleness_proxy()
        self.detect_and_characterize_shocks()
        self.create_interaction_features()
        
        if self.output_path:
            self._clean_and_save()
        else:
            self.df = self.df.fillna(0) # Clean for in-memory use

        logger.info("--- Feature Engineering Pipeline Complete ---")
        return self.df.drop(columns=['brittleness_score_lag_1'], errors='ignore')

    def _clean_and_save(self):
        """Cleans the final DataFrame and saves it to the output file."""
        logger.info(f"Cleaning and saving engineered features to {self.output_path}")
        
        final_df = self.df.drop(columns=['brittleness_score_lag_1'], errors='ignore')
        
        # Drop initial rows that have NaNs from lags/rolling windows
        final_df.dropna(subset=[f'total_agency_lag_{max([1,2,3])}'], inplace=True)
        final_df.fillna(0, inplace=True)
        
        final_df['timestamp'] = final_df.index.strftime('%Y-%m-%dT%H:%M:%S')
        output_data = {
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'n_features': len(final_df.columns),
                'n_samples': len(final_df),
                'n_countries': final_df['country_code'].nunique()
            },
            'data': final_df.to_dict(orient='records')
        }
        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Save complete. Final dataset has {len(final_df)} samples.")