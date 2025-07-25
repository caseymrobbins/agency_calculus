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
import pandera as pa
from pandera import Column, Check, DataFrameSchema

# Add project root to path to allow module imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.api.database import get_db, bulk_upsert_observations

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('world_bank_ingestion.log')]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'ingestion/config.yaml') -> dict:
    """Loads and validates the main configuration file, expanding env vars."""
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

def create_pandera_schema(indicators_list: List[Dict[str, Any]]) -> DataFrameSchema:
    """Creates a pandera schema for validation based on indicators list."""
    columns = {
        'country_code': Column(str, checks=Check.str_length(3, 3), nullable=False),
        'year': Column(int, checks=Check.ge(1960), nullable=False),
        'indicator_code': Column(str, nullable=False),
        'value': Column(float, nullable=False)
    }
    # Add per-indicator checks (e.g., value_range from config)
    for ind in indicators_list:
        if 'value_range' in ind:
            min_val = ind['value_range'].get('min', float('-inf'))
            max_val = ind['value_range'].get('max', float('inf'))
            columns['value'].checks.append(Check.in_range(min_val, max_val))
    return DataFrameSchema(columns)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_world_bank_data_batch(
    country_codes: List[str],
    indicators: Dict[str, str],
    start_year: int = 1960,
    end_year: Optional[int] = None,
    batch_size: int = 50  # Batch countries to avoid API overload
) -> List[Dict]:
    """Fetches time-series data for multiple countries and indicators in batches."""
    logger.info(f"Starting batch fetch for {len(country_codes)} countries and {len(indicators)} indicators.")
    end_year = end_year or datetime.now().year
    data_date_range = (datetime(start_year, 1, 1), datetime(end_year, 12, 31))
    
    import wbdata
    wbdata.cache.CACHE_BACKEND = None  # Disable cache for fresh data
    
    all_records = []
    dropped = 0
    for i in range(0, len(country_codes), batch_size):
        batch_countries = country_codes[i:i + batch_size]
        logger.info(f"Fetching batch {i//batch_size + 1}: {batch_countries}")
        try:
            df = wbdata.get_dataframe(indicators, country=batch_countries, date=data_date_range)
            df.reset_index(inplace=True)
            
            df_long = df.melt(id_vars=['country', 'date'], var_name='indicator_name', value_name='value')
            prev_len = len(df_long)
            df_long.dropna(subset=['value'], inplace=True)
            dropped += prev_len - len(df_long)
            
            df_long.rename(columns={'date': 'year', 'country': 'country_name'}, inplace=True)

            # Dynamic country name to code from pycountry (no hardcoded map)
            df_long['country_code'] = df_long['country_name'].apply(country_name_to_code)
            df_long = df_long.dropna(subset=['country_code'])

            # Map name to code (indicators is {code: name}, so reverse)
            name_to_code_map = {v: k for k, v in indicators.items()}
            df_long['indicator_code'] = df_long['indicator_name'].map(name_to_code_map)
            
            df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')
            
            all_records.extend(df_long[['country_code', 'year', 'indicator_code', 'value']].to_dict('records'))
        except Exception as e:
            logger.error(f"Batch fetch failed for {batch_countries}: {e}", exc_info=True)
    
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows due to NaN values.")
    logger.info(f"Successfully fetched and processed {len(all_records)} records from World Bank API.")
    return all_records

def country_name_to_code(name: str) -> Optional[str]:
    """Converts country name to 3-letter ISO code using pycountry."""
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        logger.warning(f"Country name not found: {name}")
        return None

def run_ingestion(config_path: str = 'ingestion/config.yaml', dry_run: bool = False):
    """Main function to orchestrate the World Bank data ingestion pipeline."""
    logger.info("--- Starting World Bank Data Ingestion Pipeline ---")
    try:
        config = load_config(config_path)
        wb_config = config.get('world_bank', {})
        countries = wb_config.get('countries', [])
        indicators_list = wb_config.get('indicators', [])
        
        if not countries or not indicators_list:
            raise ValueError("Config must specify 'world_bank.countries' and 'world_bank.indicators'.")
            
        indicators_dict = {ind['code']: ind['name'] for ind in indicators_list}
        
        validation_schema = create_pandera_schema(indicators_list)
        
        all_observations = fetch_world_bank_data_batch(countries, indicators_dict)
        
        dataset_version = f"WB-API-{date.today().isoformat()}"
        for obs in all_observations:
            obs['dataset_version'] = dataset_version
            obs['notes'] = f"Data sourced from World Bank API on {date.today().isoformat()}"
            
        if not all_observations:
            logger.warning("No new observations were fetched from the World Bank API.")
            return

        # Validate with pandera
        df_obs = pd.DataFrame(all_observations)
        try:
            validation_schema.validate(df_obs)
        except pa.errors.SchemaError as e:
            logger.warning(f"Validation failed: {e}. Dropping invalid rows.")
            df_obs = validation_schema.validate(df_obs, lazy=True)
        all_observations = df_obs.to_dict('records')

        logger.info(f"Preparing to upsert {len(all_observations)} observations into the database.")
        with get_db() as db:
            result = bulk_upsert_observations(db, all_observations)
            if not dry_run:
                db.commit()
            else:
                db.rollback()
                logger.info("Dry-run: Rolled back changes.")
            logger.info(f"Database upsert complete. Rows affected: {result.get('affected_rows', 'N/A')}")

        logger.info("--- World Bank data ingestion finished successfully. ---")
    except Exception as e:
        logger.critical(f"A critical error occurred in the ETL pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="World Bank ETL Ingestion Script")
    parser.add_argument("--config-path", default="ingestion/config.yaml", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run without DB commits")
    args = parser.parse_args()
    
    run_ingestion(args.config_path, args.dry_run)