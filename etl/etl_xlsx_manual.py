"""
ETL script to iterate over an XLSX file and insert rows manually into the observations table.
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

def process_xlsx_manual(xlsx_path: str, sheet_name: str = 'Sheet1', dataset_version: str = "", notes: str = ""):
    """Iterate over XLSX rows and insert into database."""
    logger.info(f"Processing XLSX file: {xlsx_path}, sheet: {sheet_name}")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine='openpyxl')
    
    # Column mapping - adjust based on your XLSX structure
    df = df.rename(columns={
        'Country Code': 'country_code',
        'Year': 'year',
        'Indicator Code': 'indicator_code',
        'Value': 'value'
    })  # Adjust column names as per your XLSX
    
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
    logger.info("XLSX processing complete.")

if __name__ == "__main__":
    xlsx_path = "data/raw/world_bank_data.xlsx"  # Replace with your XLSX path
    sheet_name = "Sheet1"  # Replace with your sheet name
    dataset_version = f"Manual-XLSX-{date.today().isoformat()}"
    process_xlsx_manual(xlsx_path, sheet_name, dataset_version, notes="Manual XLSX insert")