"""
ETL script to iterate over a CSV file and insert rows manually into the observations table.
"""
import pandas as pd
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv
import os
from datetime import date
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in .env")

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def insert_row(session, table, row_data: Dict):
    """Insert a single row into the table using insert."""
    stmt = insert(table).values(row_data)
    stmt = stmt.on_conflict_do_nothing()  # Skip if duplicate
    session.execute(stmt)

def process_csv_manual(csv_path: str, dataset_version: str = "", notes: str = ""):
    """Iterate over CSV rows and insert into database."""
    logger.info(f"Processing CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Column mapping - adjust based on your CSV structure
    df = df.rename(columns={
        'Country Code': 'country_code',
        'Year': 'year',
        'Indicator Code': 'indicator_code',
        'Value': 'value'
    })  # Adjust column names as per your CSV
    
    df = df.dropna(subset=['country_code', 'year', 'indicator_code', 'value'])
    df['dataset_version'] = dataset_version
    df['notes'] = notes

    with Session() as session:
        table = {'__tablename__': 'observations'}  # Use the observations table
        for _, row in df.iterrows():
            row_data = {
                'country_code': str(row['country_code']).upper(),
                'year': int(row['year']),
                'indicator_code': str(row['indicator_code']),
                'value': float(row['value']),
                'dataset_version': row['dataset_version'],
                'notes': row['notes']
            }
            insert_row(session, table, row_data)
            logger.info(f"Inserted row for {row['country_code']} - {row['year']}")
        session.commit()
    logger.info("CSV processing complete.")

if __name__ == "__main__":
    csv_path = "data/raw/world_bank_data.csv"  # Replace with your CSV path
    dataset_version = f"Manual-CSV-{date.today().isoformat()}"
    process_csv_manual(csv_path, dataset_version, notes="Manual CSV insert")