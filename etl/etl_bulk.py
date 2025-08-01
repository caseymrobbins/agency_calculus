import os
import sys
import logging
import yaml
import pandas as pd
import pycountry
from datetime import date
from typing import List, Dict, Any, Generator, Optional
from pathlib import Path
from tqdm import tqdm
import itertools  # For efficient line count alternative
import pandera as pa  # For data validation

# Add project root for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Assuming etl is under src/etl
sys.path.append(str(PROJECT_ROOT))

from agency_calculus.api.database import get_db, bulk_upsert, bulk_upsert_observations, Indicator

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('ingestion.log')]  # Log to file
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'ingestion/config.yaml') -> Dict[str, Any]:
    """Loads the ingestion configuration from the YAML file, expanding env vars in paths."""
    try:
        full_config_path = PROJECT_ROOT / config_path
        with open(full_config_path, 'r') as f:
            config_str = f.read()
        # Expand environment variables in the entire config string
        config_str = os.path.expandvars(config_str)
        config = yaml.safe_load(config_str)
        # Validate required top-level keys
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        return config
    except FileNotFoundError as e:
        logger.critical(f"FATAL: Config file not found at {full_config_path}: {e}")
        raise
    except yaml.YAMLError as e:
        logger.critical(f"FATAL: YAML parsing error in {full_config_path}: {e}")
        raise
    except Exception as e:
        logger.critical(f"FATAL: Unexpected error loading config from {full_config_path}: {e}")
        raise

def country_name_to_code(name: str) -> Optional[str]:
    """Converts a country name to a 3-letter ISO code. Returns None on failure."""
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
            "Kosovo": "XKX",  # Manual for Kosovo (not in ISO)
            # Add more as encountered
        }
        if name in name_map:
            return name_map[name]
        
        country = pycountry.countries.search_fuzzy(name)[0]
        return country.alpha_3
    except (LookupError, IndexError) as e:
        logger.warning(f"Could not find a 3-letter code for country: '{name}'. Skipping. Error: {e}")
        return None

def ensure_indicators_exist(db, source_key: str, indicators_config: List[Dict[str, Any]]):
    """Ensures that the indicators for a given source exist in the database.
    indicators_config: List of dicts with 'source_key', 'code', 'name', 'domain', etc.
    """
    logger.info(f"Synchronizing indicators for source: {source_key}")
    indicator_data = []
    for ind in indicators_config:
        if 'code' not in ind or 'source_key' not in ind:
            logger.warning(f"Skipping invalid indicator config: {ind}")
            continue
        indicator_data.append({
            'indicator_code': ind['code'],
            'indicator_name': ind.get('name', f"{source_key.title()} - {ind['source_key']}"),
            'source': source_key,
            'access_method': 'Bulk',
            'domain': ind.get('domain', 'other'),  # From config
            'description': ind.get('description'),
            'unit_of_measure': ind.get('unit')
        })
    
    if indicator_data:
        bulk_upsert(db, Indicator.__table__, indicator_data, ['indicator_code'])

def get_file_line_count(file_path: str) -> int:
    """Efficiently gets approximate line count (cross-platform, buffered)."""
    try:
        with open(file_path, 'r', encoding='utf-8', buffering=8192) as f:  # Buffered read
            return sum(1 for _ in f)
    except Exception as e:
        logger.warning(f"Failed to get line count for {file_path}: {e}. Defaulting to unknown.")
        return 0

def create_pandera_schema(indicators_config: List[Dict[str, Any]]) -> pa.DataFrameSchema:
    """Creates a pandera schema for validation based on config."""
    columns = {
        'country_code': pa.Column(str, checks=pa.Check.str_length(3, 3), nullable=False),
        'year': pa.Column(int, checks=pa.Check.ge(1900), nullable=False),
        'indicator_code': pa.Column(str, nullable=False),
        'value': pa.Column(float, nullable=False)
    }
    # Add per-indicator checks
    for ind in indicators_config:
        if 'value_range' in ind:
            min_val = ind['value_range'].get('min', None)
            max_val = ind['value_range'].get('max', None)
            # pandera doesn't support per-group checks easily, so log and implement in code if needed
    return pa.DataFrameSchema(columns)

def process_vdem_data(file_path: str, indicators_map: Dict[str, str], validation_schema: pa.DataFrameSchema) -> Generator[Dict[str, Any], None, None]:
    """Processes V-Dem data yielding records for memory efficiency."""
    logger.info(f"Processing V-Dem data from {file_path} (in chunks)...")
    try:
        full_file_path = PROJECT_ROOT / file_path
        total_lines = get_file_line_count(full_file_path)
        chunk_iter = pd.read_csv(full_file_path, low_memory=False, chunksize=100000, encoding='utf-8')
        skipped = 0
        for chunk in tqdm(chunk_iter, total=(total_lines // 100000) + 1 if total_lines else 1, desc="Processing V-Dem Chunks"):
            required_cols = ['country_text_id', 'year'] + list(indicators_map.keys())
            chunk_cols = [col for col in required_cols if col in chunk.columns]
            if len(chunk_cols) < len(required_cols):
                missing = set(required_cols) - set(chunk_cols)
                logger.warning(f"Missing columns in chunk: {missing}. Skipping chunk.")
                skipped += len(chunk)
                continue
            chunk = chunk[chunk_cols]
        
            chunk.rename(columns={'country_text_id': 'country_code'}, inplace=True)
            
            # Melt the chunk, then map to standardized indicator codes
            df_long = chunk.melt(id_vars=['country_code', 'year'], var_name='source_indicator_code', value_name='value')
            df_long['indicator_code'] = df_long['source_indicator_code'].map(indicators_map)
            df_long = df_long.dropna(subset=['value', 'indicator_code'])  # Drop if value or mapped indicator is missing
            df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')  # Ensure numeric
            df_long = df_long.dropna(subset=['value'])  # Drop invalid numerics
            
            # Validate with pandera
            try:
                validation_schema.validate(df_long)
            except pa.errors.SchemaError as e:
                logger.warning(f"Validation failed for chunk: {e}. Dropping invalid rows.")
                df_long = validation_schema.validate(df_long, lazy=True)  # Drops invalid
            
            for record in df_long[['country_code', 'year', 'indicator_code', 'value']].to_dict('records'):
                yield record
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} rows due to missing columns.")
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in V-Dem file: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to process V-Dem file: {e}", exc_info=True)
        raise

def process_freedom_house_data(file_path: str, indicators_map: Dict[str, str], normalization_map: Dict[str, Dict[str, float]], sheet_name: str, validation_schema: pa.DataFrameSchema) -> Generator[Dict[str, Any], None, None]:
    """Processes Freedom House Excel yielding records."""
    logger.info(f"Processing Freedom House data from {file_path}...")
    skipped_countries = []
    try:
        full_file_path = PROJECT_ROOT / file_path
        df = pd.read_excel(full_file_path, sheet_name=sheet_name, header=0, engine='openpyxl')
        df = df[['Country/Territory', 'Edition', 'PR', 'CL']]
        df.rename(columns={'Country/Territory': 'country_name', 'Edition': 'year'}, inplace=True)
        
        df['country_code'] = df['country_name'].apply(lambda x: country_name_to_code(x))
        skipped_countries = df[df['country_code'].isna()]['country_name'].tolist()
        df = df.dropna(subset=['country_code'])

        df_long = df.melt(id_vars=['country_code', 'year'], value_vars=['PR', 'CL'], var_name='source_indicator_code', value_name='value')
        
        # Map the source_indicator_code (PR, CL) to our full indicator codes from the config
        df_long['indicator_code'] = df_long['source_indicator_code'].map(indicators_map)
        df_long = df_long.dropna(subset=['value', 'indicator_code'])

        # Normalize based on config (e.g., {'PR': {'max': 40}, 'CL': {'max': 60}})
        def normalize_fh_score(row):
            max_val = normalization_map.get(row['source_indicator_code'], {}).get('max', 1)
            return (row['value'] / max_val) * 100 if max_val != 1 else row['value']

        df_long['value'] = df_long.apply(normalize_fh_score, axis=1)
        df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')

        # Validate with pandera
        try:
            validation_schema.validate(df_long)
        except pa.errors.SchemaError as e:
            logger.warning(f"Validation failed: {e}. Dropping invalid rows.")
            df_long = validation_schema.validate(df_long, lazy=True)  # Drops invalid

        if skipped_countries:
            logger.warning(f"Skipped countries due to mapping failure: {skipped_countries}")

        for record in df_long[['country_code', 'year', 'indicator_code', 'value']].to_dict('records'):
            yield record

    except FileNotFoundError as e:
        logger.error(f"File not found: {full_file_path}: {e}")
        raise
    except ValueError as e:  # e.g., sheet not found
        logger.error(f"Excel parsing error: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data in sheet: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in Freedom House processing: {e}", exc_info=True)
        raise

def process_undp_data(file_path: str, indicators_map: Dict[str, str], validation_schema: pa.DataFrameSchema) -> Generator[Dict[str, Any], None, None]:
    logger.info(f"Starting parsing of UNDP file at: {file_path}")

    try:
        full_file_path = PROJECT_ROOT / file_path
        df = pd.read_csv(full_file_path, encoding='utf-8')
        logger.info(f"Successfully loaded UNDP CSV file with {len(df)} rows.")

        source_indicator_name = list(indicators_map.keys())[0]  # Assuming single for UNDP
        standardized_code = indicators_map[source_indicator_name]

        df_indicator = df[df['Indicator'] == source_indicator_name]

        if df_indicator.empty:
            raise ValueError(f"Indicator '{source_indicator_name}' not found in the file.")

        id_vars = ['ISO3', 'Country']  # ISO3 is the country code for UNDP
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
        df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')
        df_long = df_long.dropna(subset=['year', 'value'])
        df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
        df_long = df_long.dropna(subset=['value'])

        df_long['indicator_code'] = standardized_code  # Use the standardized code

        # Validate with pandera
        try:
            validation_schema.validate(df_long)
        except pa.errors.SchemaError as e:
            logger.warning(f"Validation failed: {e}. Dropping invalid rows.")
            df_long = validation_schema.validate(df_long, lazy=True)  # Drops invalid

        for record in df_long[['country_code', 'year', 'indicator_code', 'value']].to_dict('records'):
            yield record

    except FileNotFoundError as e:
        logger.error(f"UNDP data file not found at path: {full_file_path}: {e}")
        raise
    except (ValueError, KeyError) as e:
        logger.error(f"Parsing error due to unexpected file format in {full_file_path}: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in UNDP file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during UNDP parsing: {e}", exc_info=True)
        raise

def batch_upsert_observations(db, observations_gen: Generator[Dict[str, Any], None, None], batch_size: int = 10000) -> Dict[str, int]:
    """Batches upserts from a generator for memory efficiency."""
    results = {'affected_rows': 0, 'skipped': 0}
    batch = []
    for record in observations_gen:
        batch.append(record)
        if len(batch) >= batch_size:
            result = bulk_upsert_observations(db, batch)
            results['affected_rows'] += result.get('affected_rows', 0)
            batch = []
    if batch:
        result = bulk_upsert_observations(db, batch)
        results['affected_rows'] += result.get('affected_rows', 0)
    return results

def run_bulk_ingestion(source_key: str, batch_size: Optional[int] = None, dry_run: bool = False):
    """Main orchestrator for a single bulk data source. Batch size override for tuning."""
    logger.info(f"--- Starting Bulk Ingestion for source: {source_key} ---")
    
    try:
        config = load_config()
        source_config = config.get(source_key)
        if not source_config:
            raise ValueError(f"Source '{source_key}' not found in ingestion/config.yaml")

        file_path = source_config.get('file_path')
        dataset_version = source_config.get('dataset_version', '1.0')
        indicators_config = source_config.get('indicators', [])  # Now list of dicts with code, domain, etc.
        normalization_map = source_config.get('normalization', {})  # For FH
        sheet_name = source_config.get('sheet_name', 'FIW2013-2025')  # Default for FH; make config
        source_name = source_config.get('source_name', source_key)
        batch_size = batch_size or source_config.get('batch_size', 10000)  # Configurable
        countries_filter = set(source_config.get('countries', []))  # For filtering

        # Create pandera schema from config
        validation_schema = create_pandera_schema(indicators_config)

        # Ensure indicators exist in the database before processing
        with get_db() as db:
            ensure_indicators_exist(db, source_name, indicators_config)
            db.commit()

        observations_gen = None
        if source_key == 'v_dem':
            observations_gen = process_vdem_data(file_path, {ind['source_key']: ind['code'] for ind in indicators_config}, validation_schema)
        elif source_key == 'freedom_house':
            observations_gen = process_freedom_house_data(file_path, {ind['source_key']: ind['code'] for ind in indicators_config}, normalization_map, sheet_name, validation_schema)
        elif source_key == 'undp':
            observations_gen = process_undp_data(file_path, {ind['source_key']: ind['code'] for ind in indicators_config}, validation_schema)
        else:
            logger.error(f"No processor found for source: {source_key}")
            return

        # Add version/notes and filter countries (generator wrapper)
        def add_meta_and_filter_gen(gen):
            for obs in gen:
                if countries_filter and obs['country_code'] not in countries_filter:
                    results['skipped'] += 1
                    continue
                obs['dataset_version'] = dataset_version
                obs['notes'] = f"Data sourced from {source_name} file: {os.path.basename(file_path)}"
                yield obs

        observations_gen = add_meta_and_filter_gen(observations_gen) if observations_gen else []

        logger.info(f"Preparing to upsert observations for {source_name}.")
        with get_db() as db:
            result = batch_upsert_observations(db, observations_gen, batch_size)
            if not dry_run:
                db.commit()
            else:
                db.rollback()
                logger.info("Dry-run: Rolled back changes.")
            logger.info(f"Database upsert complete. Rows affected: {result.get('affected_rows', 'N/A')}. Skipped: {result.get('skipped', 0)}")

        logger.info(f"--- Bulk ingestion for {source_name} finished successfully. ---")

    except ValueError as e:
        logger.error(f"Configuration error during bulk ingestion for {source_key}: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"File parsing error for {source_key}: {e}")
        raise
    except Exception as e:
        logger.critical(f"A critical error occurred during bulk ingestion for {source_key}: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bulk ETL Ingestion Script")
    parser.add_argument("source_key", help="Source to process (e.g., v_dem)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for upserts")
    parser.add_argument("--dry-run", action="store_true", help="Run without DB commits")
    args = parser.parse_args()
    
    run_bulk_ingestion(args.source_key, args.batch_size, args.dry_run)