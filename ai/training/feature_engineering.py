"""
Component: Feature Engineering (Production Version)
Purpose: Ingests clean country-year data from the database and engineers a rich 
         feature set for the HybridForecaster model. This includes lagged agency 
         scores, derivatives, rolling volatility, and dynamic shock characteristics.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from ai.changepoint_detector import ChangepointDetector

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Orchestrates the entire feature engineering pipeline for the HybridForecaster."""
    def __init__(self, shock_penalty_scale: float = 2.0, windows: Optional[Dict[str, int]] = None):
        self.shock_detector = ChangepointDetector(penalty_scale=shock_penalty_scale)
        self.domain_cols = ['economic_agency', 'political_agency', 'social_agency', 'health_agency', 'educational_agency']
        self.windows = windows or {'short': 3, 'medium': 7, 'long': 15}

    def _calculate_trend(self, values: pd.Series) -> float:
        """Helper to calculate linear trend slope, robust to NaNs."""
        if values.count() < 2: return 0.0
        y = values.dropna()
        x = np.arange(len(y))
        try:
            return np.polyfit(x, y, 1)[0]
        except (np.linalg.LinAlgError, TypeError):
            return 0.0

    def create_ts_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates standard time-series features like lags, derivatives, and rolling stats."""
        logger.info("Creating standard time-series features...")
        df_out = df.copy()
        grouped = df_out.groupby('country_code')
        
        for col in self.domain_cols:
            for lag in range(1, 4):
                df_out[f'{col}_lag_{lag}'] = grouped[col].shift(lag)
            df_out[f'{col}_delta_1'] = grouped[col].diff(1)
            for name, w in self.windows.items():
                df_out[f'{col}_vol_{name}'] = grouped[col].rolling(window=w, min_periods=2).std()
                df_out[f'{col}_trend_{name}'] = grouped[col].rolling(window=w, min_periods=2).apply(self._calculate_trend, raw=False)
        return df_out

    def _process_country_shocks(self, country_df: pd.DataFrame) -> pd.DataFrame:
        """Processes shocks for a single country. Designed to be used with groupby().apply()."""
        if len(country_df) < 10:
            return country_df
        
        changepoints = self.shock_detector.detect(country_df, self.domain_cols)
        if not changepoints:
            return country_df

        for cp in changepoints:
            shock_date = cp.timestamp
            country_df.loc[shock_date, 'is_shock_year'] = 1
            country_df.loc[shock_date, 'shock_magnitude'] = cp.magnitude
            country_df.loc[shock_date, 'shock_confidence'] = cp.confidence
            
            # Recovery slope calculation
            post_shock_period = country_df.loc[shock_date:].iloc[:5]
            if len(post_shock_period) > 1:
                slope = self._calculate_trend(post_shock_period['total_agency'])
                country_df.loc[shock_date, 'recovery_slope'] = slope
        
        country_df['shock_group'] = country_df['is_shock_year'].cumsum()
        country_df['time_since_shock'] = country_df.groupby('shock_group').cumcount()
        return country_df

    def detect_and_characterize_shocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detects and characterizes systemic shocks for all countries."""
        logger.info("Detecting and characterizing systemic shocks...")
        df_out = df.copy()
        df_out['total_agency'] = df_out[self.domain_cols].mean(axis=1) # Needed for recovery slope
        
        # Initialize columns
        df_out['is_shock_year'] = 0.0
        df_out['shock_magnitude'] = 0.0
        df_out['shock_confidence'] = 0.0
        df_out['recovery_slope'] = 0.0
        df_out['time_since_shock'] = 0

        processed_df = df_out.groupby('country_code', group_keys=False).apply(self._process_country_shocks)
        return processed_df.drop(columns=['total_agency', 'shock_group'], errors='ignore')

    def run_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes all feature engineering steps and splits into endogenous and exogenous sets.
        
        Args:
            df (pd.DataFrame): Input DataFrame with columns ['country_code', 'year'] and the five agency domains.
                               Must be sorted by country_code and year.
        
        Returns:
            Tuple: A tuple of (endog_df, exog_df).
        """
        logger.info("--- Starting Feature Engineering Pipeline ---")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.set_index('year', inplace=True)
        
        # Create exogenous features first
        exog_df = self.create_ts_features(df)
        exog_df = self.detect_and_characterize_shocks(exog_df)
        
        # Select endogenous and exogenous columns
        endog_df = df[self.domain_cols]
        exog_cols = [col for col in exog_df.columns if col not in self.domain_cols and col != 'country_code']
        exog_df = exog_df[exog_cols]
        
        # Align indices and handle NaNs
        common_index = endog_df.index.intersection(exog_df.index)
        endog_df = endog_df.loc[common_index]
        exog_df = exog_df.loc[common_index]
        
        # Forward-fill and then back-fill to handle NaNs from shifts/rolling windows
        exog_df.fillna(method='ffill', inplace=True)
        exog_df.fillna(method='bfill', inplace=True)
        exog_df.fillna(0, inplace=True) # Fill any remaining NaNs
        
        logger.info("--- Feature Engineering Pipeline Complete ---")
        logger.info(f"Endogenous shape: {endog_df.shape}, Exogenous shape: {exog_df.shape}")
        
        return endog_df, exog_df