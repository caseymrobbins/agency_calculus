# etl/etl_bulk.py
import os
import sys
import logging
import yaml
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

# Add project root for module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from api.database import get_db, bulk_upsert_observations

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

def process_vdem_data(file_path: str, indicators_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Processes the V-Dem dataset to extract relevant political agency indicators."""
    logger.info(f"Processing V-Dem data from {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
        
        # Select and rename columns
        df = df[['country_text_id', 'year'] + list(indicators_map.keys())]
        df.rename(columns={'country_text_id': 'country_code'}, inplace=True)
        
        # Melt dataframe to long format
        df_long = df.melt(
            id_vars=['country_code', 'year'],
            var_name='indicator_code',
            value_name='value'
        )
        df_long.dropna(subset=['value'], inplace=True)
        return df_long.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to process V-Dem file: {e}", exc_info=True)
        return []

# Placeholder functions for other data sources
def process_freedom_house_data(file_path: str) -> List[Dict[str, Any]]:
    logger.warning(f"Processing for Freedom House at {file_path} is not yet implemented.")
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
        
        observations = []
        if source_key == 'v_dem':
            observations = process_vdem_data(file_path, indicators)
        elif source_key == 'freedom_house':
            # observations = process_freedom_house_data(file_path)
            pass
        elif source_key == 'undp':
            # observations = process_undp_data(file_path)
            pass
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
        logger.error("Example: python -m etl.etl_bulk v_dem")
        sys.exit(1)
    
    source_to_process = sys.argv[1]
    run_bulk_ingestion(source_to_process)