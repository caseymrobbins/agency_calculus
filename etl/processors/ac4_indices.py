## File: etl/processors/ac4_indices.py

"""
Component: etl/processors/ac4_indices.py
Purpose:   Calculates the abstract AC4 indices (Polarization, Bipartisanship, Social Trust)
           from raw academic and survey data. This is the production-ready version.
Version:   3.0 (Production Hardened)
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any
import logging
import time

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config/ac4_indices_config.yaml') -> Dict[str, Any]:
    """Loads and validates the processor's configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Basic validation
        assert 'vdem_columns' in config and 'parameters' in config
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except (FileNotFoundError, AssertionError) as e:
        logger.critical(f"FATAL: Could not load or validate config at {config_path}. Error: {e}")
        raise

# Load config at module level
CONFIG = load_config()
VDEM_COLS = CONFIG['vdem_columns']
WVS_COLS = CONFIG['wvs_columns']
PARAMS = CONFIG['parameters']

def calculate_polarization(vdem_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Polarization Index using a vectorized approach."""
    required_col = VDEM_COLS['polarization']
    if required_col not in vdem_df.columns:
        logger.error(f"Required column '{required_col}' not found. Skipping polarization.")
        vdem_df['polarization_index'] = np.nan
        return vdem_df

    logger.info(f"Calculating Polarization Index from '{required_col}'.")
    
    # Vectorized calculation and clamping
    polarization_scores = (vdem_df[required_col].clip(0, PARAMS['polarization']['max_value']) / PARAMS['polarization']['max_value']) * 100
    vdem_df['polarization_index'] = polarization_scores
    return vdem_df

def calculate_bipartisanship_proxy(vdem_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Bipartisanship Index Proxy using stable historical normalization."""
    required_col = VDEM_COLS['bipartisanship']
    if required_col not in vdem_df.columns:
        logger.error(f"Required column '{required_col}' not found. Skipping bipartisanship.")
        vdem_df['bipartisanship_index'] = np.nan
        return vdem_df
        
    logger.info(f"Calculating Bipartisanship Index Proxy from '{required_col}'.")
    
    # Vectorized calculation using stable min/max from config
    min_val = PARAMS['bipartisanship']['historical_min']
    max_val = PARAMS['bipartisanship']['historical_max']
    
    if (max_val - min_val) == 0:
        bipartisanship_scores = PARAMS['bipartisanship']['no_variation_default']
    else:
        # Clip input values before normalization to handle outliers
        clipped_values = vdem_df[required_col].clip(min_val, max_val)
        bipartisanship_scores = ((clipped_values - min_val) / (max_val - min_val)) * 100
        
    vdem_df['bipartisanship_index'] = bipartisanship_scores
    return vdem_df

def calculate_social_trust(wvs_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Social Trust Index using a high-performance, vectorized approach."""
    required_cols = [WVS_COLS['country'], WVS_COLS['year'], WVS_COLS['trust']]
    if not all(col in wvs_df.columns for col in required_cols):
        logger.error(f"WVS DataFrame missing one of required columns: {required_cols}. Skipping social trust.")
        return pd.DataFrame(columns=[WVS_COLS['country'], WVS_COLS['year'], 'social_trust_index'])

    logger.info(f"Calculating Social Trust from {len(wvs_df)} survey responses...")
    
    # Filter for valid responses in a single pass
    trust_val = PARAMS['social_trust']['trust_value']
    no_trust_val = PARAMS['social_trust']['no_trust_value']
    valid_mask = wvs_df[WVS_COLS['trust']].isin([trust_val, no_trust_val])
    valid_responses = wvs_df[valid_mask].copy()

    if valid_responses.empty:
        logger.warning("No valid 'generalized_trust' responses found.")
        return pd.DataFrame(columns=[WVS_COLS['country'], WVS_COLS['year'], 'social_trust_index'])

    # Vectorized aggregation
    # Create a boolean series for 'trust' responses
    valid_responses['is_trust'] = (valid_responses[WVS_COLS['trust']] == trust_val)
    
    # Group by country and year, then calculate the mean of the boolean series (True=1, False=0)
    # The mean of a boolean series is the proportion of True values.
    grouped = valid_responses.groupby([WVS_COLS['country'], WVS_COLS['year']])['is_trust']
    trust_agg = (grouped.mean() * 100).reset_index()
    trust_agg.rename(columns={'is_trust': 'social_trust_index'}, inplace=True)
    
    logger.info(f"Social Trust Index calculated for {len(trust_agg)} country-year groups.")
    return trust_agg

def process_ac4_indices(vdem_df: pd.DataFrame, wvs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates calculation of all AC4 indices and merges results. Main entry point for ETL.
    """
    start_time = time.time()
    logger.info("--- Starting AC4 Indices Processing Pipeline ---")
    logger.info(f"Processing V-Dem data with shape {vdem_df.shape} and WVS data with shape {wvs_df.shape}")

    # Process V-Dem based indices
    vdem_processed = calculate_polarization(vdem_df)
    vdem_processed = calculate_bipartisanship_proxy(vdem_processed)
    
    # Process WVS based index
    social_trust_df = calculate_social_trust(wvs_df)

    # Merge results
    final_df = pd.merge(
        vdem_processed, 
        social_trust_df, 
        on=[VDEM_COLS['country'], VDEM_COLS['year']], 
        how='left'
    )
    
    processing_time = time.time() - start_time
    logger.info(f"Final merged DataFrame has shape {final_df.shape}.")
    logger.info(f"--- AC4 Indices Processing Complete in {processing_time:.2f} seconds ---")
    
    return final_df