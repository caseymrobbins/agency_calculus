# etl/populate_metadata.py
import os
import sys
import yaml
import pandas as pd
import logging
import pycountry
from typing import List, Dict, Any, Set
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from api.database import get_db, bulk_upsert, Country, Indicator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'ingestion/config.yaml') -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.critical(f"FATAL: Could not load config from {config_path}: {e}")
        raise

def get_vdem_countries(file_path: str) -> Set[str]:
    try:
        logger.info(f"Reading V-Dem countries from {file_path}...")
        df = pd.read_csv(file_path, usecols=['country_text_id'])
        return set(df['country_text_id'].dropna().unique())
    except Exception as e:
        logger.error(f"Could not process V-Dem file for countries: {e}")
        return set()

def get_country_name(code: str) -> str:
    try:
        country = pycountry.countries.get(alpha_3=code)
        return country.name if country else code
    except Exception:
        return code

def run_metadata_population():
    logger.info("--- Starting Metadata Population Pipeline ---")
    config = load_config()
    all_country_codes = set()
    all_indicators = []

    # Gather country codes
    wb_countries = config.get('world_bank', {}).get('countries', [])
    all_country_codes.update(wb_countries)
    vdem_config = config.get('v_dem', {})
    if vdem_config and vdem_config.get('file_path'):
        all_country_codes.update(get_vdem_countries(vdem_config['file_path']))

    # Prepare country data
    countries_to_upsert = [{'code': code, 'name': get_country_name(code)} for code in all_country_codes if code]
    
    # Prepare indicator data
    for source, details in config.items():
        if 'indicators' in details and isinstance(details['indicators'], (dict, list)):
            if isinstance(details['indicators'], dict): # Handle v_dem, freedom_house format
                 for k, v in details['indicators'].items():
                     all_indicators.append({'code': v, 'name': f"{source} - {k}", 'source': source, 'access_method': 'Bulk'})
            else: # Handle world_bank format
                for ind in details['indicators']:
                    all_indicators.append({'code': ind['code'], 'name': ind['name'], 'source': 'world_bank', 'access_method': 'API'})

    logger.info(f"Total unique countries to sync: {len(countries_to_upsert)}")
    logger.info(f"Total unique indicators to sync: {len(all_indicators)}")

    with get_db() as db:
        if countries_to_upsert:
            bulk_upsert(db, Country.__table__, countries_to_upsert, ['code'])
        if all_indicators:
            bulk_upsert(db, Indicator.__table__, all_indicators, ['code'])

    logger.info("--- Metadata Population Finished Successfully ---")

if __name__ == "__main__":
    run_metadata_population()