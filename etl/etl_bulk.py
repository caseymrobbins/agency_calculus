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
        # Adjusted path to ingestion config
        full_config_path = Path(__file__).resolve().parents[2] / config_path
        with open(full_config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.critical(f"FATAL: Could not load config from {config_config_path}: {e}")
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
            "North Korea": "PRK",
            # Add more as encountered
        }
        if name in name_map:
            return name_map[name]
        
        country = pycountry.countries.search_fuzzy(name)[0]
        return country.alpha_3
    except (LookupError, IndexError):
        # logger.warning(f"Could not find a 3-letter code for country: '{name}'. Skipping.")
        return None

def ensure_indicators_exist(db, source_key: str, indicators_map: Dict[str, str]):
    """Ensures that the indicators for a given source exist in the database.
    indicators_map: A dictionary where keys are source-specific indicator names/codes
                    and values are our standardized indicator codes.
    """
    logger.info(f"Synchronizing indicators for source: {source_key}")
    indicator_data = []
    for source_indicator_key, standardized_code in indicators_map.items():
        # Determine domain based on source (this might need refinement for more granularity)
        domain = 'political'
        if source_key == 'undp':
            domain = 'educational'
        elif source_key == 'world_bank': # This script is for bulk, but good to keep consistent
            domain = 'economic' # Placeholder, would need to map properly if WB used here.

        indicator_data.append({
            'code': standardized_code, # Use the standardized code as the primary key
            'name': f"{source_key.replace('_', ' ').title()} - {source_indicator_key}",
            'source': source_key,
            'access_method': 'Bulk',
            'domain': domain # Assign a default domain based on source key
        })
    
    if indicator_data:
        # Use the Indicator ORM model directly
        bulk_upsert(db, Indicator.__table__, indicator_data, ['code'])
        db.commit()

def process_vdem_data(file_path: str, indicators_map: Dict[str, str]) -> List[Dict[str, Any]]:
    logger.info(f"Processing V-Dem data from {file_path} (in chunks)...")
    all_records = []
    try:
        # Adjusted path to raw data
        full_file_path = Path(__file__).resolve().parents[2] / file_path
        
        total_lines = sum(1 for row in open(full_file_path, 'r', encoding='utf-8'))
        chunk_iter = pd.read_csv(full_file_path, low_memory=False, chunksize=100000)
        for chunk in tqdm(chunk_iter, total=(total_lines // 100000) + 1, desc="Processing V-Dem Chunks"):
            required_cols = ['country_text_id', 'year'] + list(indicators_map.keys())
            chunk_cols = [col for col in required_cols if col in chunk.columns]
            chunk = chunk[chunk_cols]
        
            chunk.rename(columns={'country_text_id': 'country_code'}, inplace=True)
            
            # Melt the chunk, then map to standardized indicator codes
            df_long = chunk.melt(id_vars=['country_code', 'year'], var_name='source_indicator_code', value_name='value')
            df_long['indicator_code'] = df_long['source_indicator_code'].map(indicators_map)
            df_long.dropna(subset=['value', 'indicator_code'], inplace=True) # Drop if value or mapped indicator is missing
            
            all_records.extend(df_long[['country_code', 'year', 'indicator_code', 'value']].to_dict('records'))
        logger.info(f"Finished processing. Total records extracted from V-Dem: {len(all_records)}")
        return all_records
    except Exception as e:
        logger.error(f"Failed to process V-Dem file: {e}", exc_info=True)
        return []

def process_freedom_house_data(file_path: str, indicators_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Processes the Freedom House Excel file to extract Political Rights and Civil Liberties."""
    logger.info(f"Processing Freedom House data from {file_path}...")
    try:
        full_file_path = Path(__file__).resolve().parents[2] / file_path
        df = pd.read_excel(full_file_path, sheet_name='FIW2013-2024', header=0)
        df = df[['Country/Territory', 'Edition', 'PR', 'CL']]
        df.rename(columns={'Country/Territory': 'country_name', 'Edition': 'year'}, inplace=True)
        
        df['country_code'] = df['country_name'].apply(country_name_to_code)
        df.dropna(subset=['country_code'], inplace=True)

        df_long = df.melt(id_vars=['country_code', 'year'], value_vars=['PR', 'CL'], var_name='source_indicator_code', value_name='value')
        
        # Map the source_indicator_code (PR, CL) to our full indicator codes from the config
        df_long['indicator_code'] = df_long['source_indicator_code'].map(indicators_map)
        df_long.dropna(subset=['value', 'indicator_code'], inplace=True)

        # Normalize PR (Political Rights) and CL (Civil Liberties) to 0-100 scale as per documentation 
        # PR is 0-40, CL is 0-60.
        # political_freedom_index = (PR_score / 40) * 100
        # civil_liberties_index = (CL_score / 60) * 100
        # The prompt only explicitly mentioned political_freedom_index normalization for PR,
        # so we'll apply it consistently to both if they're handled this way.
        
        # Apply normalization based on the original scale
        def normalize_fh_score(row):
            if row['source_indicator_code'] == 'PR':
                return (row['value'] / 40.0) * 100
            elif row['source_indicator_code'] == 'CL':
                return (row['value'] / 60.0) * 100
            return row['value'] # Return as is if not PR or CL

        df_long['value'] = df_long.apply(normalize_fh_score, axis=1)

        logger.info(f"Finished processing. Total records extracted from Freedom House: {len(df_long)}")
        return df_long[['country_code', 'year', 'indicator_code', 'value']].to_dict('records')

    except Exception as e:
        logger.error(f"Failed to process Freedom House file: {e}", exc_info=True)
        return []

def process_undp_data(file_path: str, indicators_map: Dict[str, str]) -> List[Dict[str, Any]]:
    logger.info(f"Starting parsing of UNDP file at: {file_path}")

    try:
        full_file_path = Path(__file__).resolve().parents[2] / file_path
        df = pd.read_csv(full_file_path, encoding='utf-8')
        logger.info(f"Successfully loaded UNDP CSV file with {len(df)} rows.")

        # The specific indicator name we need from the UNDP data [cite: 1777, 1778]
        # This mapping is now dynamic from config
        source_indicator_name = list(indicators_map.keys())[0] # Assuming only one for UNDP for now
        standardized_code = indicators_map[source_indicator_name]

        df_indicator = df[df['Indicator'] == source_indicator_name]

        if df_indicator.empty:
            raise ValueError(f"Indicator '{source_indicator_name}' not found in the file.")

        id_vars = ['ISO3', 'Country'] # ISO3 is the country code for UNDP
        year_cols = [col for col in df_indicator.columns if col.isdigit()]
        
        df_long = pd.melt(
            df_indicator,
            id_vars=id_vars,
            value_vars=year_cols,
            var_name='year',
            value_name='value'
        )
        logger.info(f"Melted the DataFrame into {len(df_long)} observations.")

        df_long.rename(columns={'ISO3': 'country_code'}, inplace=True)
        df_long['year'] = pd.to_numeric(df_long['year'])
        df_long.dropna(subset=['value'], inplace=True)

        df_long['indicator_code'] = standardized_code # Use the standardized code

        final_df = df_long[['country_code', 'year', 'indicator_code', 'value']]
        
        observations = final_df.to_dict(orient='records')
        logger.info(f"Successfully parsed {len(observations)} UNDP records.")
        return observations

    except FileNotFoundError:
        logger.error(f"UNDP data file not found at path: {file_path}")
        raise
    except (ValueError, KeyError) as e:
        logger.error(f"Parsing error due to unexpected file format in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during UNDP parsing: {e}", exc_info=True)
        raise

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
        indicators_map = source_config.get('indicators', {}) # This is now a map for standardized codes
        source_name = source_config.get('source_name', source_key)

        # Ensure indicators exist in the database before processing
        with get_db() as db:
            ensure_indicators_exist(db, source_name, indicators_map)

        observations = []
        if source_key == 'v_dem':
            observations = process_vdem_data(file_path, indicators_map)
        elif source_key == 'freedom_house':
            observations = process_freedom_house_data(file_path, indicators_map)
        elif source_key == 'undp':
            observations = process_undp_data(file_path, indicators_map)
        else:
            logger.error(f"No processor found for source: {source_key}")
            return

        if not observations:
            logger.warning(f"No observations processed for source: {source_key}. Exiting.")
            return
            
        for obs in observations:
            obs['dataset_version'] = dataset_version
            obs['notes'] = f"Data sourced from {source_name} file: {os.path.basename(file_path)}"

        logger.info(f"Preparing to upsert {len(observations)} observations for {source_name}.")
        with get_db() as db:
            result = bulk_upsert_observations(db, observations)
            logger.info(f"Database upsert complete. Rows affected: {result.get('affected_rows', 'N/A')}")

        logger.info(f"--- Bulk ingestion for {source_name} finished successfully. ---")

    except Exception as e:
        logger.critical(f"A critical error occurred during bulk ingestion for {source_key}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python -m etl.etl_bulk <source_key>")
        logger.error("Examples: python -m etl.etl_bulk v_dem, python -m etl.etl_bulk freedom_house, python -m etl.etl_bulk undp")
        sys.exit(1)
    
    source_to_process = sys.argv[1]
    run_bulk_ingestion(source_to_process)