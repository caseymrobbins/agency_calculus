#!/usr/bin/env python3
"""
V-Dem ETL Script for Agency Calculus (Corrected & Optimized)

This script processes V-Dem (Varieties of Democracy) data and loads it into PostgreSQL.
It uses an efficient, vectorized approach to handle large datasets.
"""

import os
import sys
import logging
import argparse
import yaml
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Any
from pathlib import Path  # FIX: Import the Path object to resolve the NameError

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agency_calculus.api.database import bulk_upsert_indicators, bulk_upsert_observations

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VDemETL:
    """V-Dem ETL processor for Agency Calculus."""
    
    def __init__(self, config_path: str):
        """Initialize the V-Dem ETL processor."""
        self.config = self._load_config(config_path)
        self.dataset_version = f"V-Dem-v15-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        self.vdem_indicators = self._get_indicator_definitions()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f).get('v_dem', {})
        except Exception as e:
            logger.error(f"Error loading or parsing config file: {e}")
            raise

    def _get_indicator_definitions(self) -> Dict[str, Dict[str, str]]:
        """Placeholder for a more robust way to get indicator metadata."""
        # In a real system, this might come from a separate metadata file or API
        # For now, we'll keep a subset of definitions.
        return {
            'v2x_polyarchy': {'name': 'Electoral Democracy Index', 'description': '...', 'unit': 'Index (0-1)', 'topic': 'Democracy'},
            'v2x_libdem': {'name': 'Liberal Democracy Index', 'description': '...', 'unit': 'Index (0-1)', 'topic': 'Democracy'},
            'v2x_partipdem': {'name': 'Participatory Democracy Index', 'description': '...', 'unit': 'Index (0-1)', 'topic': 'Democracy'},
            'v2x_delibdem': {'name': 'Deliberative Democracy Index', 'description': '...', 'unit': 'Index (0-1)', 'topic': 'Democracy'},
            'v2x_egaldem': {'name': 'Egalitarian Democracy Index', 'description': '...', 'unit': 'Index (0-1)', 'topic': 'Democracy'},
            'v2x_freexp_altinf': {'name': 'Freedom of Expression and Alternative Sources of Information Index', 'description': '...', 'unit': 'Index (0-1)', 'topic': 'Political Rights'},
            'v2x_frassoc_thick': {'name': 'Freedom of Association Index (Thick)', 'description': '...', 'unit': 'Index (0-1)', 'topic': 'Political Rights'},
            'v2x_suffr': {'name': 'Suffrage', 'description': '...', 'unit': 'Percentage', 'topic': 'Political Rights'},
            'v2xcl_rol': {'name': 'Equality Before the Law and Individual Liberty Index', 'description': '...', 'unit': 'Index (0-1)', 'topic': 'Civil Liberties'},
            'v2x_cspart': {'name': 'Civil Society Participation Index', 'description': '...', 'unit': 'Index (0-1)', 'topic': 'Participation'},
            'v2pepwrsoc': {'name': 'Power Distributed by Social Group', 'description': '...', 'unit': 'Ordinal', 'topic': 'Equality'}
        }

    def ensure_indicators_exist(self):
        """Ensure all configured V-Dem indicators exist in the indicators table."""
        logger.info("Ensuring V-Dem indicators exist in database...")
        indicators_to_process = self.config.get('indicators', [])
        
        indicators_to_insert = []
        for code in indicators_to_process:
            info = self.vdem_indicators.get(code, {})
            indicators_to_insert.append({
                'indicator_code': code,
                'name': info.get('name', code),
                'description': info.get('description', f'V-Dem indicator {code}'),
                'unit': info.get('unit', 'Index/Scale'),
                'source': 'V-Dem (Varieties of Democracy)',
                'topic': info.get('topic', 'Political/Democracy'),
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            })
        
        if indicators_to_insert:
            rows_affected = bulk_upsert_indicators(indicators_to_insert)
            logger.info(f"✅ Ensured {len(indicators_to_insert)} V-Dem indicators exist ({rows_affected} rows affected)")

    def process_vdem_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Processes the V-Dem data using an efficient, vectorized approach."""
        logger.info(f"Loading V-Dem data from {file_path}...")
        
        countries = self.config.get('countries', [])
        indicators = self.config.get('indicators', [])
        start_year = self.config.get('start_year', 1960)
        end_year = self.config.get('end_year', 2024)

        if not indicators:
            logger.warning("No indicators specified in the config. Nothing to process.")
            return []

        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"Loaded V-Dem data: {len(df)} rows, {len(df.columns)} columns")

        # --- Efficient Processing using `melt` ---
        id_vars = ['country_text_id', 'year']
        
        value_vars = [ind for ind in indicators if ind in df.columns]
        if len(value_vars) != len(indicators):
            missing = set(indicators) - set(value_vars)
            logger.warning(f"The following indicators were not found in the CSV and will be skipped: {missing}")

        if not value_vars:
            logger.error("None of the configured indicators were found in the CSV file. Aborting.")
            return []

        if countries:
            df = df[df['country_text_id'].isin(countries)]
        df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        
        df_filtered = df[id_vars + value_vars]
        
        df_long = pd.melt(
            df_filtered,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='indicator_code',
            value_name='value'
        )

        df_long.rename(columns={'country_text_id': 'country_code'}, inplace=True)
        df_long.dropna(subset=['value'], inplace=True)
        
        now = datetime.now(timezone.utc)
        df_long['dataset_version'] = self.dataset_version
        df_long['notes'] = f'Data sourced from V-Dem v15 on {now.strftime("%Y-%m-%d")}'
        df_long['created_at'] = now
        df_long['updated_at'] = now

        observations = df_long.to_dict('records')
        logger.info(f"Prepared {len(observations)} observations for {len(value_vars)} indicators")
        return observations

    def run(self, file_path: str, dry_run: bool = False):
        """Run the complete V-Dem ETL process."""
        logger.info("Starting V-Dem ETL process...")
        
        try:
            if not dry_run:
                self.ensure_indicators_exist()
            
            observations = self.process_vdem_data(file_path)
            
            if not dry_run:
                if observations:
                    bulk_upsert_observations(observations)
                else:
                    logger.warning("No observations were generated, skipping database load.")
            else:
                logger.info(f"DRY RUN: Would have loaded {len(observations)} observations.")
            
            logger.info("✅ V-Dem ETL process completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ V-Dem ETL process failed: {e}", exc_info=True)
            raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='V-Dem Data ETL')
    parser.add_argument('--config-path', required=True, help='Configuration file path')
    parser.add_argument('--dry-run', action='store_true', help='Run without writing to the database')
    parser.add_argument('--file-path', default='data/raw/V-Dem-CY-Full+Others-v15.csv', help='Path to V-Dem CSV file')
    
    args = parser.parse_args()
    
    etl = VDemETL(args.config_path)
    etl.run(args.file_path, dry_run=args.dry_run)

if __name__ == "__main__":
    main()