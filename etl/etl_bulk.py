# etl/etl_bulk.py
import os
import sys
import logging
import yaml
import pandas as pd
import pycountry
from datetime import date
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

# Add project root for module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from api.database import get_db, bulk_upsert, bulk_upsert_observations, Indicator

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'ingestion/config.yaml') -> dict:
    """Loads the ingestion configuration from the YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.critical(f"FATAL: Could not load config from {config_path}: {e}")
        raise

def country_name_to_code(name: str) -> str:
    """Converts a country name to a 3-letter ISO code."""
    try:
        # Handle common inconsistencies
        name_map = {
            "United States": "USA",
            "Cote d'Ivoire": "CIV",
            "Russia": "RUS",
            "Vietnam": "VNM",
            "Iran": "IRN",
            "Syria": "SYR",
            "Bolivia": "BOL",
            "Venezuela": "VEN",
            "Taiwan": "TWN",
            "South Korea": "KOR",
            "North Korea": "PRK"
        }
        if name in name_map:
            return name_map[name]
        
        country = pycountry.countries.search_fuzzy(name)[0]
        return country.alpha_3
    except (LookupError, IndexError):
        # logger.warning(f"Could not find a 3-letter code for country: '{name}'. Skipping.")
        return None

def ensure_indicators_exist(db, source_key: str, indicators: Dict[str, str]):
    """Ensures that the indicators for a given source exist in the database."""
    logger.info(f"Synchronizing indicators for source: {source_key}")
    indicator_data = [
        {
            'code': v,
            'name': f"{source_key.replace('_', ' ').title()} - {k}",
            'source': source_key,
            'access_method': 'Bulk'
        } for k, v in indicators.items()
    ]
    if indicator_data:
        bulk_upsert(db, 'indicators', indicator_data, ['code'])
        db.commit()

def process_vdem_data(file_path: str, indicators_map: Dict[str, str]) -> List[Dict[str, Any]]:
    # ... (This function remains the same as the last version) ...
    logger.info(f"Processing V-Dem data from {file_path} (in chunks)...")
    all_records = []
    try:
        total_lines = sum(1 for row in open(file_path, 'r', encoding='utf-8'))
        chunk_iter = pd.read_csv(file_path, low_memory=False, chunksize=100000)
        for chunk in tqdm(chunk_iter, total=(total_lines // 100000) + 1, desc="Processing V-Dem Chunks"):
            required_cols = ['country_text_id', 'year'] + list(indicators_map.keys())
            chunk_cols = [col for col in required_cols if col in chunk.columns]
            chunk = chunk[chunk_cols]
            chunk.rename(columns={'country_text_id': 'country_code'}, inplace=True)
            df_long = chunk.melt(id_vars=['country_code', 'year'], var_name='indicator_code', value_name='value')
            df_long.dropna(subset=['value'], inplace=True)
            all_records.extend(df_long.to_dict('records'))
        logger.info(f"Finished processing. Total records extracted from V-Dem: {len(all_records)}")
        return all_records
    except Exception as e:
        logger.error(f"Failed to process V-Dem file: {e}", exc_info=True)
        return []

def process_freedom_house_data(file_path: str, indicators_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Processes the Freedom House Excel file to extract Political Rights and Civil Liberties."""
    logger.info(f"Processing Freedom House data from {file_path}...")
    try:
        # Freedom House data is often in a specific sheet and may have header rows to skip
        df = pd.read_excel(file_path, sheet_name='FIW2013-2024', header=0)
        df = df[['Country/Territory', 'Edition', 'PR', 'CL']]
        df.rename(columns={'Country/Territory': 'country_name', 'Edition': 'year'}, inplace=True)
        
        # Convert country names to codes
        df['country_code'] = df['country_name'].apply(country_name_to_code)
        df.dropna(subset=['country_code'], inplace=True)

        # Melt the data from wide to long format
        df_long = df.melt(id_vars=['country_code', 'year'], value_vars=['PR', 'CL'], var_name='indicator_short', value_name='value')
        
        # Map the short codes (PR, CL) to our full indicator codes from the config
        df_long['indicator_code'] = df_long['indicator_short'].map(indicators_map)
        df_long.dropna(subset=['value', 'indicator_code'], inplace=True)

        logger.info(f"Finished processing. Total records extracted from Freedom House: {len(df_long)}")
        return df_long[['country_code', 'year', 'indicator_code', 'value']].to_dict('records')

    except Exception as e:
        logger.error(f"Failed to process Freedom House file: {e}", exc_info=True)
        return []

def process_undp_data(file_path: str) -> List[Dict[str, Any]]:
    logger.warning(f"Processing for UNDP at {file_path} is not yet implemented.")
    return []

def run_bulk_ingestion(source_key: str):
    """Main orchestrator for a single bulk data source."""
    logger.info(f"--- Starting Bulk Ingestion for source: {source_key} ---")
    
    try:
        config = load_config()
        source_config = config.get(source_key)
        if not source_config:
            raise ValueError(f"Source '{source_key}' not found in ingestion/config.yaml")

        file_path = source_config.get('file_path')
        dataset_version = source_config.get('dataset_version', '1.0')
        indicators = source_config.get('indicators', {})
        
        # Ensure indicators exist in the database before processing
        with get_db() as db:
            ensure_indicators_exist(db, source_key, indicators)

        observations = []
        if source_key == 'v_dem':
            observations = process_vdem_data(file_path, indicators)
        elif source_key == 'freedom_house':
            observations = process_freedom_house_data(file_path, indicators)
        elif source_key == 'undp':
            observations = process_undp_data(file_path)
        else:
            logger.error(f"No processor found for source: {source_key}")
            return

        if not observations:
            logger.warning(f"No observations processed for source: {source_key}. Exiting.")
            return
            
        for obs in observations:
            obs['dataset_version'] = dataset_version
            obs['notes'] = f"Data sourced from {source_key} file: {os.path.basename(file_path)}"

        logger.info(f"Preparing to upsert {len(observations)} observations for {source_key}.")
        with get_db() as db:
            result = bulk_upsert_observations(db, observations)
            logger.info(f"Database upsert complete. Rows affected: {result.get('affected_rows', 'N/A')}")

        logger.info(f"--- Bulk ingestion for {source_key} finished successfully. ---")

    except Exception as e:
        logger.critical(f"A critical error occurred during bulk ingestion for {source_key}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python -m etl.etl_bulk <source_key>")
        logger.error("Examples: python -m etl.etl_bulk v_dem, python -m etl.etl_bulk freedom_house")
        sys.exit(1)
    
    source_to_process = sys.argv[1]
    run_bulk_ingestion(source_to_process)