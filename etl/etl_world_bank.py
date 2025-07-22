import os
import requests
import yaml
import logging
from datetime import date, datetime
from uuid import UUID
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import execute_batch
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pool = None

# --- Database Interaction ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database using a connection pool."""
    global pool
    if pool is None:
        pool = SimpleConnectionPool(
            minconn=1, maxconn=10,
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "agency_monitor"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres")
        )
    return pool.getconn()

def release_db_connection(conn):
    """Releases a database connection back to the pool."""
    global pool
    if pool and conn:
        pool.putconn(conn)

def log_ingestion_status(cursor, log_id: UUID, status: str, records_processed: int = None, error_message: str = None):
    """Updates the ingestion_logs table with the final status of the job."""
    end_time = datetime.now()
    sql = """
        UPDATE ingestion_logs
        SET status = %s, records_processed = %s, error_message = %s, end_time = %s
        WHERE id = %s;
    """
    cursor.execute(sql, (status, records_processed, error_message, end_time, log_id))

# --- Data Ingestion Logic ---
def load_config(config_path: str = 'ingestion/config.yaml') -> dict:
    """Loads and validates the country and indicator configuration from YAML."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Configuration file is empty")
        if 'countries' not in config or not isinstance(config['countries'], list) or not config['countries']:
            raise ValueError("Configuration must include a non-empty 'countries' list")
        if 'indicators' not in config or not isinstance(config['indicators'], list) or not config['indicators']:
            raise ValueError("Configuration must include a non-empty 'indicators' list")
        for ind in config['indicators']:
            if 'code' not in ind or 'name' not in ind:
                raise ValueError(f"Indicator missing 'code' or 'name': {ind}")
        return config
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        logging.error(f"Error loading configuration: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_world_bank_data(country_code: str, indicator_code: str) -> list:
    """Fetches time-series data from the World Bank API with pagination and retries."""
    base_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {'format': 'json', 'per_page': 1000, 'page': 1}
    all_data = []
    
    logging.info(f"Fetching page {params['page']} for {indicator_code}...")
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    if len(data) < 2 or data[1] is None:
        logging.warning(f"No data found for {indicator_code} in {country_code} on first page.")
        return []

    all_data.extend(data[1])
    total_pages = data[0]['pages']
    
    for page in range(2, total_pages + 1):
        params['page'] = page
        logging.info(f"Fetching page {page}/{total_pages} for {indicator_code}...")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        all_data.extend(data[1])
        
    return all_data

def insert_observations(cursor, observations: list) -> int:
    """Inserts or updates a list of observations using batch execution."""
    if not observations:
        return 0
    sql = """
        INSERT INTO observations (country_code, indicator_code, year, value, dataset_version, notes)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (country_code, indicator_code, year, dataset_version)
        DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP;
    """
    data_to_insert = [
        (obs['country_code'], obs['indicator_code'], obs['year'], obs['value'], obs['dataset_version'], obs['notes'])
        for obs in observations
    ]
    execute_batch(cursor, sql, data_to_insert)
    return len(observations)

def main():
    """Main function to orchestrate the World Bank data ingestion pipeline."""
    logging.info("Starting World Bank data ingestion pipeline...")
    log_id = None
    conn = None
    try:
        config_path = os.getenv("CONFIG_PATH", "ingestion/config.yaml")
        config = load_config(config_path)
        conn = get_db_connection()
        cursor = conn.cursor()
        dataset_version = date.today().isoformat()
        
        cursor.execute(
            "INSERT INTO ingestion_logs (source, access_method, dataset_version, status) VALUES (%s, %s, %s, %s) RETURNING id;",
            ('World Bank', 'API', dataset_version, 'RUNNING')
        )
        log_id = cursor.fetchone()[0]
        conn.commit()
        
        total_inserted = 0
        for country_code in config['countries']:
            logging.info(f"--- Processing Country: {country_code} ---")
            for indicator in config['indicators']:
                indicator_code = indicator['code']
                logging.info(f"Fetching data for indicator: {indicator['name']} ({indicator_code})")
                
                raw_data = fetch_world_bank_data(country_code, indicator_code)
                
                observations_to_insert = [
                    {
                        'country_code': country_code,
                        'indicator_code': indicator_code,
                        'year': int(record['date']),
                        'value': float(record['value']),
                        'dataset_version': dataset_version,
                        'notes': f"Data sourced from World Bank API on {dataset_version}"
                    }
                    for record in raw_data if record.get('value') is not None
                ]
                
                if not observations_to_insert:
                    logging.info(f"No valid records found for {indicator_code} in {country_code} (all values were null).")
                else:
                    inserted_count = insert_observations(cursor, observations_to_insert)
                    logging.info(f"Processed {inserted_count} records for {indicator_code} in {country_code}.")
                    total_inserted += inserted_count
                    conn.commit()
        
        log_ingestion_status(cursor, log_id, 'SUCCESS', records_processed=total_inserted)
        conn.commit()
        logging.info(f"Pipeline finished successfully. Total records inserted/updated: {total_inserted}.")

    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)
        if conn and log_id:
            conn.rollback()
            try:
                log_cursor = conn.cursor()
                log_ingestion_status(log_cursor, log_id, 'FAILURE', error_message=str(e))
                conn.commit()
                log_cursor.close()
            except Exception as log_e:
                logging.error(f"Failed to log the failure status to the database: {log_e}")
    finally:
        if conn:
            release_db_connection(conn)
            logging.info("Database connection released.")

if __name__ == "__main__":
    # Requirements: pip install requests pyyaml psycopg2-binary tenacity
    # Set environment variables: DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, CONFIG_PATH
    main()