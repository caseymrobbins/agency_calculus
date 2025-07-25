import os
import sys
import logging
from datetime import date, datetime
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandera.pandas as pa  # Updated import
from pandera import Column, Check, DataFrameSchema
import pandas as pd
import requests
import time
import pycountry

# Add project root to path to allow module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from api.database import get_db, bulk_upsert_observations

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
    for ind in indicators_list:
        if 'value_range' in ind:
            min_val = float(ind['value_range'].get('min', float('-inf')))
            max_val = float(ind['value_range'].get('max', float('inf')))
            columns['value'].checks.append(Check.in_range(min_val, max_val))
    return DataFrameSchema(columns)

def iso3_to_iso2(iso3: str) -> Optional[str]:
    """Converts ISO3 country code to ISO2 using pycountry."""
    country = pycountry.countries.get(alpha_3=iso3)
    if country:
        return country.alpha_2
    else:
        logger.warning(f"Invalid ISO3 code: {iso3}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_world_bank_data_batch(
    country_codes: List[str],
    indicators: Dict[str, str],
    start_year: int = 1960,
    end_year: Optional[int] = None,
    batch_size: int = 20
) -> List[Dict]:
    """Fetches time-series data for multiple countries and indicators using direct API calls."""
    logger.info(f"Starting direct API fetch for {len(country_codes)} countries and {len(indicators)} indicators.")
    current_year = datetime.now().year
    end_year = end_year or (current_year - 1)
    end_year = min(end_year, current_year - 1)
    all_records = []
    dropped = 0
    base_url = "https://api.worldbank.org/v2/country/{country}/indicator/{ind}?date={start}:{end}&format=json&per_page=1000"

    for ind_code, ind_name in indicators.items():
        for i in range(0, len(country_codes), batch_size):
            batch_countries = country_codes[i:i + batch_size]
            logger.info(f"Fetching {ind_name} ({ind_code}) for batch {i//batch_size + 1}: {batch_countries}")
            for country in batch_countries:
                country_iso2 = iso3_to_iso2(country)
                if not country_iso2:
                    continue
                url = base_url.format(country=country_iso2, ind=ind_code, start=start_year, end=end_year)
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    data = response.json()
                    if len(data) < 2 or not isinstance(data[1], list):
                        logger.warning(f"No data for {country}/{ind_code}")
                        continue
                    entries = data[1]
                    for entry in entries:
                        if entry['value'] is not None:
                            all_records.append({
                                'country_code': entry['countryiso3code'],
                                'year': int(entry['date']),
                                'indicator_code': ind_code,
                                'value': float(entry['value'])
                            })
                        else:
                            dropped += 1
                    time.sleep(0.5)
                except requests.exceptions.RequestException as e:
                    logger.error(f"API fetch failed for {country}/{ind_code}: {e}")
    
    if dropped > 0:
        logger.warning(f"Dropped {dropped} entries due to null values.")
    logger.info(f"Successfully fetched and processed {len(all_records)} records from World Bank API.")
    return all_records

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

        df_obs = pd.DataFrame(all_observations)
        try:
            df_obs = validation_schema.validate(df_obs, lazy=True)
        except pa.errors.SchemaErrors as e:
            logger.warning(f"Validation failed with {len(e.failure_cases)} errors: {e}. Dropping invalid rows.")
            invalid_indices = e.failure_cases['index'].unique()
            df_obs = df_obs.drop(index=invalid_indices).reset_index(drop=True)
            if df_obs.empty:
                logger.error("All rows invalid after validation. No data to upsert.")
                return
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