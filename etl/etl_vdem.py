import logging
import argparse
import yaml
import pandas as pd
import pandera.pandas as pa
from datetime import date
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_vdem_ingestion(config_path, dry_run=False):
    try:
        config = load_config(config_path)
        file_path = config['vdem']['file_path']
        dataset_version = config['vdem']['dataset_version']
        chunksize = config['etl']['chunksize']
        countries = config['etl']['countries']
        indicators = config['vdem']['indicators']

        # Define Pandera schema with broader ranges and relaxed year check for all historical data
        schema = pa.DataFrameSchema({
            "country_code": pa.Column(str, checks=pa.Check.isin(countries)),
            "year": pa.Column(int, checks=pa.Check.ge(1789)),  # Includes all V-Dem years (1789+)
            "indicator_code": pa.Column(str, checks=pa.Check.isin(indicators)),
            "value": pa.Column(float, checks=pa.Check.in_range(-100.0, 100.0), nullable=True),
        })

        # Explicit dtypes (updated for new indicators)
        dtypes = {
            'country_text_id': 'str',
            'year': 'Int64',
            'v2x_polyarchy': 'float',
            'v2x_libdem': 'float',
            'v2x_partipdem': 'float',
            'v2x_delibdem': 'float',
            'v2x_egaldem': 'float',
            'v2x_freexp_altinf': 'float',
            'v2x_frassoc_thick': 'float',
            'v2x_suffr': 'float',
            'v2x_elecoff': 'float',
            'v2xcl_rol': 'float',
            'v2x_cspart': 'float',
            'v2pepwrsoc': 'float',
            'v2clprptyw': 'float',  # Added for women's property/economic rights
            'v2xpe_exlsocgr': 'float',  # Added for socio-economic exclusion
            'v2xpe_exlpol': 'float',
            'v2x_egal': 'float',
            'v2xpe_exlgeo': 'float',
        }

        engine = create_engine(config['database']['url'])
        Session = sessionmaker(bind=engine)
        db = Session()

        total_processed = 0
        total_valid = 0

        logging.info(f"Processing V-Dem file: {file_path}")

        for chunk in pd.read_csv(file_path, chunksize=chunksize, dtype=dtypes, low_memory=False):
            # Filter for selected countries
            chunk = chunk[chunk['country_text_id'].isin(countries)]
            chunk = chunk.melt(id_vars=['country_text_id', 'year'], value_vars=indicators,
                               var_name='indicator_code', value_name='value')
            chunk = chunk.rename(columns={'country_text_id': 'country_code'})  # To match schema

            # Validate chunk
            try:
                validated = schema.validate(chunk, lazy=True)
            except pa.errors.SchemaErrors as err:
                logging.warning(f"Validation failed with {len(err.failure_cases)} errors: {err.failure_cases.to_dict(orient='records')}")
                validated = chunk[~chunk.index.isin(err.failure_cases.index)]

            total_processed += len(chunk)
            total_valid += len(validated)

            if dry_run:
                logging.info(f"Dry run: Would insert {len(validated)} valid rows")
                continue

            # Prepare for upsert
            observations = []
            for _, row in validated.iterrows():
                obs = {
                    'country_code': row['country_code'],
                    'indicator_code': row['indicator_code'],
                    'year': row['year'],
                    'value': row['value'],
                    'dataset_version': f'V-Dem-{dataset_version}',
                    'notes': f"Data sourced from V-Dem {dataset_version} on {date.today().isoformat()}",
                }
                observations.append(obs)

            if observations:
                stmt = text("""
                    INSERT INTO observations (country_code, indicator_code, year, value, dataset_version, notes)
                    VALUES (:country_code, :indicator_code, :year, :value, :dataset_version, :notes)
                    ON CONFLICT (country_code, indicator_code, year, dataset_version)
                    DO UPDATE SET value = EXCLUDED.value, notes = EXCLUDED.notes
                """)
                db.execute(stmt, observations)
                db.commit()

        logging.info(f"Processed {total_processed} rows, inserted {total_valid} valid rows.")
        logging.info("--- V-Dem data ingestion finished successfully. ---")

    except Exception as e:
        logging.critical(f"A critical error occurred in the V-Dem ETL pipeline: {str(e)}")
        raise
    finally:
        if 'db' in locals():
            db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V-Dem ETL Pipeline")
    parser.add_argument('--config-path', required=True, help="Path to config.yaml")
    parser.add_argument('--dry-run', action='store_true', help="Run without inserting to DB")
    args = parser.parse_args()

    run_vdem_ingestion(args.config_path, args.dry_run)