"""
ETL script for World Bank data ingestion.
Fetches indicators via wbdata API and upserts into the database.
"""
import os
import sys
import logging
from datetime import date, datetime
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd

# Add project root to path to allow module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from api.database import get_db, bulk_upsert_observations

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Loads and validates the main configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Configuration file is empty")
        return config
    except Exception as e:
        logger.critical(f"FATAL: Could not load config from {config_path}: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_world_bank_data_batch(
    country_codes: List[str],
    indicators: Dict[str, str],
    start_year: int = 1960,
    end_year: Optional[int] = None
) -> List[Dict]:
    """Fetches time-series data for multiple countries and indicators in a single efficient API call."""
    logger.info(f"Starting batch fetch for {len(country_codes)} countries and {len(indicators)} indicators.")
    end_year = end_year or datetime.now().year
    data_date_range = (datetime(start_year, 1, 1), datetime(end_year, 12, 31))
    
    import wbdata
    import wbdata.cache
    wbdata.cache.CACHE_BACKEND = None  # Disable cache
    
    try:
        df = wbdata.get_dataframe(indicators, country=country_codes, date=data_date_range)
        df.reset_index(inplace=True)
        
        df_long = df.melt(id_vars=['country', 'date'], var_name='indicator_name', value_name='value')
        df_long.dropna(subset=['value'], inplace=True)
        df_long.rename(columns={'date': 'year', 'country': 'country_name'}, inplace=True)

        name_to_code_map = {v: k for k, v in zip(indicators.values(), indicators.keys())}
        df_long['indicator_code'] = df_long['indicator_name'].map(name_to_code_map)
        
        # Hardcoded map for target countries to avoid get_countries cache error
        country_name_to_code_map = {
            'United States': 'USA',
            'Haiti': 'HTI',
            'Chile': 'CHL'
        }
        df_long['country_code'] = df_long['country_name'].map(country_name_to_code_map)
        
        final_df = df_long[['country_code', 'year', 'indicator_code', 'value']]
        final_df['year'] = pd.to_numeric(final_df['year'])
        
        records = final_df.to_dict(orient='records')
        logger.info(f"Successfully fetched and processed {len(records)} records from World Bank API.")
        return records
    except Exception as e:
        logger.error(f"Failed to fetch batch World Bank data: {e}", exc_info=True)
        return []

def run_ingestion():
    """Main function to orchestrate the World Bank data ingestion pipeline."""
    logger.info("--- Starting World Bank Data Ingestion Pipeline ---")
    try:
        config = load_config()
        wb_config = config.get('ingestion', {}).get('world_bank', {})
        countries = config.get('ingestion', {}).get('target_countries', [])
        indicators_list = wb_config.get('indicators', [])
        
        if not countries or not indicators_list:
            raise ValueError("Config must specify 'target_countries' and 'world_bank.indicators'.")
            
        indicators_dict = {ind['code']: ind['name'] for ind in indicators_list}
        
        all_observations = fetch_world_bank_data_batch(countries, indicators_dict)
        
        dataset_version = f"WB-API-{date.today().isoformat()}"
        for obs in all_observations:
            obs['dataset_version'] = dataset_version
            obs['notes'] = f"Data sourced from World Bank API on {date.today().isoformat()}"
            
        if not all_observations:
            logger.warning("No new observations were fetched from the World Bank API.")
            return

        logger.info(f"Preparing to upsert {len(all_observations)} observations into the database.")
        with get_db() as db:
            result = bulk_upsert_observations(db, all_observations)
            logger.info(f"Database upsert complete. Rows affected: {result.get('affected_rows', 'N/A')}")
            
        logger.info("--- World Bank data ingestion finished successfully. ---")
    except Exception as e:
        logger.critical(f"A critical error occurred in the ETL pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_ingestion()