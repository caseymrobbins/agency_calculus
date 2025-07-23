# etl/populate_metadata.py
"""
Component: Metadata Population
Purpose: This script is the first step in the ETL process. It populates the
         'countries' and 'indicators' dimension tables from all available
         sources to ensure data integrity before loading observations.
"""
import os
import sys
import yaml
import pandas as pd
import logging
import pycountry
from typing import List, Dict, Any, Set
from pathlib import Path

# Add project root for module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from api.database import get_db, bulk_upsert

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'ingestion/config.yaml') -> Dict[str, Any]:
    """Loads the ingestion configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.critical(f"FATAL: Could not load config from {config_path}: {e}")
        raise

def get_vdem_countries(file_path: str) -> Set[str]:
    """Extracts unique 3-letter country codes from the V-Dem dataset."""
    try:
        logger.info(f"Reading V-Dem countries from {file_path}...")
        df = pd.read_csv(file_path, usecols=['country_text_id'])
        return set(df['country_text_id'].dropna().unique())
    except Exception as e:
        logger.error(f"Could not process V-Dem file for countries: {e}")
        return set()

def get_country_name(code: str) -> str:
    """Finds the country name from a 2 or 3 letter code."""
    try:
        country = pycountry.countries.get(alpha_3=code) or pycountry.countries.get(alpha_2=code)
        return country.name if country else code
    except Exception:
        return code

def run_metadata_population():
    """Main function to populate dimension tables."""
    logger.info("--- Starting Metadata Population Pipeline ---")
    config = load_config()
    
    all_country_codes = set()
    all_indicators = []

    # 1. Gather all country codes
    # From World Bank config
    wb_countries = config.get('world_bank', {}).get('countries', [])
    all_country_codes.update(wb_countries)
    logger.info(f"Found {len(wb_countries)} country codes in World Bank config.")
    
    # From V-Dem CSV
    vdem_config = config.get('v_dem', {})
    if vdem_config and vdem_config.get('file_path'):
        vdem_codes = get_vdem_countries(vdem_config['file_path'])
        all_country_codes.update(vdem_codes)
        logger.info(f"Found {len(vdem_codes)} unique country codes in V-Dem file.")

    # 2. Prepare country data for database
    countries_to_upsert = [
        {'code': code, 'name': get_country_name(code)}
        for code in all_country_codes if code
    ]
    logger.info(f"Total unique countries to synchronize: {len(countries_to_upsert)}")

    # 3. Gather all indicators
    for source, details in config.items():
        if 'indicators' in details and isinstance(details['indicators'], dict):
            for code, name in details['indicators'].items():
                all_indicators.append({
                    'code': code,
                    'name': name,
                    'source': source
                })
    logger.info(f"Total unique indicators to synchronize: {len(all_indicators)}")
    
    # 4. Upsert to database
    with get_db() as db:
        # Upsert Countries
        if countries_to_upsert:
            logger.info("Synchronizing countries table...")
            result_countries = bulk_upsert(db, 'countries', countries_to_upsert, ['code'])
            logger.info(f"Countries table synchronized. Rows affected: {result_countries.get('affected_rows', 0)}")
        
        # Upsert Indicators
        if all_indicators:
            logger.info("Synchronizing indicators table...")
            result_indicators = bulk_upsert(db, 'indicators', all_indicators, ['code'])
            logger.info(f"Indicators table synchronized. Rows affected: {result_indicators.get('affected_rows', 0)}")

    logger.info("--- Metadata Population Finished Successfully ---")


if __name__ == "__main__":
    run_metadata_population()