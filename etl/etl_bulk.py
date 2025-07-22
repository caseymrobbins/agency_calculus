# etl/etl_bulk.py
"""
Component: ETL Pipeline for Bulk Data (Refactored)
Purpose: Ingests large, versioned datasets from local files (e.g., V-Dem),
         parses them, and loads the data into the database using the
         high-performance SQLAlchemy ORM layer.
Inputs: A configuration file (etl_config.yaml) specifying file paths and versions.
Outputs: Populates the `observations` table in the PostgreSQL database.
Integration: Uses the session management and bulk upsert functions from `api.database`.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add project root to path to allow imports from other directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- New Imports ---
# Import the SQLAlchemy-based functions and YAML loader
from api.database import get_db, bulk_upsert_observations
from etl.etl_world_bank import load_config as load_main_config, log_ingestion_status # Re-using these helpers

# --- Parser Imports ---
# Modular parsers for different bulk data sources
from etl.parsers.vdem_parser import parse_vdem_data

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_parser(source_name: str):
    """
    Returns the appropriate parser function based on the source name.
    This makes the script modular and easy to extend for new bulk sources.
    """
    if source_name.lower() == 'v-dem':
        return parse_vdem_data
    # To add a new source (e.g., Freedom House):
    # 1. Create a new parser in `etl/parsers/freedom_house_parser.py`
    # 2. Add the import and an `elif` condition here.
    # elif source_name.lower() == 'freedom-house':
    #     return parse_freedom_house_data
    else:
        raise NotImplementedError(f"No parser implemented for source: {source_name}")


def main(source_to_ingest: str):
    """
    Main function to orchestrate the bulk data ingestion process for a given source.
    """
    logger.info(f"--- Starting Bulk Data Ingestion for: {source_to_ingest} ---")
    log_id = None

    try:
        # The get_db context manager handles our session and transactions
        with get_db() as db:
            from sqlalchemy import text # Import text for raw SQL
            
            # Load the master configuration file
            config = load_main_config('config/etl_config.yaml')
            
            # Get the specific configuration for the source we are ingesting
            source_config_key = source_to_ingest.lower().replace('-', '_')
            source_config = config.get(source_config_key)

            if not source_config:
                raise ValueError(f"No configuration found for source '{source_to_ingest}' in etl_config.yaml")

            file_path = source_config['file_path']
            dataset_version = source_config['dataset_version']
            source_name = source_config['source_name']
            
            # Start ingestion log
            log_result = db.execute(
                text("INSERT INTO ingestion_logs (source, access_method, dataset_version, status) VALUES (:source, :access, :version, :status) RETURNING id;"),
                {'source': source_name, 'access': 'Bulk', 'version': dataset_version, 'status': 'RUNNING'}
            ).fetchone()
            log_id = log_result[0]
            db.commit()

            # Get the correct parser function for this source
            parser_func = get_parser(source_name)

            # --- Parsing ---
            # Parse the entire data file into a list of observation dictionaries
            observations = parser_func(file_path)

            # --- Database Loading ---
            # Add the dataset_version and notes to each observation before loading
            notes = f"Data sourced from {source_name} bulk file, version {dataset_version}"
            for obs in observations:
                obs['dataset_version'] = dataset_version
                obs['notes'] = notes
            
            # Perform a single bulk upsert for maximum efficiency
            if observations:
                logger.info(f"Upserting {len(observations)} total records from {source_name} into the database...")
                result = bulk_upsert_observations(db, observations)
                total_processed = result.get("affected_rows", 0)
                logger.info(f"Bulk upsert complete. {total_processed} rows affected.")
            else:
                total_processed = 0

            # Log the successful completion
            log_ingestion_status(db, log_id, 'SUCCESS', records_processed=total_processed)
            db.commit()
            logger.info(f"Pipeline finished successfully. Total records affected: {total_processed}.")

    except Exception as e:
        logger.error(f"A critical error occurred: {e}", exc_info=True)
        if log_id:
            # Log the failure status to the database
            with get_db() as db_fail:
                log_ingestion_status(db_fail, log_id, 'FAILURE', error_message=str(e))
                db_fail.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a bulk data ingestion pipeline for a specific source.")
    parser.add_argument("source", type=str, help="The name of the data source to ingest (e.g., 'v-dem'). Must match a key in etl_config.yaml.")
    args = parser.parse_args()
    
    main(args.source)