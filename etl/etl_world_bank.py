#!/usr/bin/env python3
"""
World Bank Data ETL Script for Agency Calculus
File: etl/etl_world_bank.py

This script fetches data from the World Bank API and loads it into PostgreSQL.
"""

import os
import sys
import logging
import argparse
import yaml
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import wbgapi as wb
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agency_calculus.api.database import bulk_upsert_observations, bulk_upsert_countries, bulk_upsert_indicators

# Load environment variables
load_dotenv()

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/etl_world_bank.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WorldBankETL:
    """World Bank ETL processor."""
    
    def __init__(self, config_path: str = 'ingestion/config.yaml'):
        """Initialize the ETL processor."""
        self.config = self._load_config(config_path)
        self.batch_size = self.config.get('world_bank', {}).get('batch_size', 100)
        self.dataset_version = f"WB-API-{datetime.now().strftime('%Y-%m-%d')}"
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            # Expand environment variables in path
            config_path = os.path.expandvars(config_path)
            with open(config_path, 'r') as f:
                config_str = f.read()
                # Expand environment variables in content
                config_str = os.path.expandvars(config_str)
                return yaml.safe_load(config_str)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
    
    def fetch_countries(self) -> List[Dict[str, Any]]:
        """Fetch country metadata from World Bank."""
        logger.info("Fetching country metadata...")
        
        countries = []
        for country_code in self.config['world_bank']['countries']:
            try:
                # Get country information from World Bank API
                country_info = wb.economy.info(country_code)
                if hasattr(country_info, 'get') and country_info:
                    countries.append({
                        'country_code': country_code,
                        'name': country_info.get('value', country_code),
                        'region': country_info.get('region', {}).get('value', ''),
                        'income_level': country_info.get('incomeLevel', {}).get('value', ''),
                        'created_at': datetime.now(timezone.utc),
                        'updated_at': datetime.now(timezone.utc)
                    })
                    logger.info(f"✅ Fetched metadata for {country_code}")
                else:
                    logger.warning(f"⚠️ No metadata found for {country_code}")
                    
            except Exception as e:
                logger.error(f"❌ Error fetching metadata for {country_code}: {e}")
                
            # Always add basic entry (fallback or primary)
            if not any(c['country_code'] == country_code for c in countries):
                countries.append({
                    'country_code': country_code,
                    'name': country_code,
                    'region': '',
                    'income_level': '',
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc)
                })
        
        logger.info(f"Fetched {len(countries)} countries")
        return countries
    
    def fetch_indicators(self) -> List[Dict[str, Any]]:
        """Fetch indicator metadata from World Bank."""
        logger.info("Fetching indicator metadata...")
        
        indicators = []
        # Handle complex indicator structure from config
        for indicator_config in self.config['world_bank']['indicators']:
            indicator_code = indicator_config['code']  # Extract code from config object
            try:
                # Get indicator information from World Bank API
                indicator_info = wb.series.info(indicator_code)
                if hasattr(indicator_info, 'get') and indicator_info:
                    indicators.append({
                        'indicator_code': indicator_code,
                        'name': indicator_config.get('name', indicator_info.get('value', indicator_code))[:500],
                        'description': str(indicator_info.get('sourceNote', ''))[:1000] if indicator_info.get('sourceNote') else '',
                        'unit': indicator_info.get('unit', ''),
                        'source': str(indicator_info.get('source', {}).get('value', 'World Bank'))[:255],
                        'topic': str(indicator_info.get('topics', [{}])[0].get('value', ''))[:255] if indicator_info.get('topics') else '',
                        'created_at': datetime.now(timezone.utc),
                        'updated_at': datetime.now(timezone.utc)
                    })
                    logger.info(f"✅ Fetched metadata for {indicator_code}")
                else:
                    logger.warning(f"⚠️ No API metadata found for {indicator_code}, using config metadata")
                    
            except Exception as e:
                logger.error(f"❌ Error fetching API metadata for {indicator_code}: {e}")
                
            # Always add entry with config metadata (fallback or primary)
            if not any(ind['indicator_code'] == indicator_code for ind in indicators):
                indicators.append({
                    'indicator_code': indicator_code,
                    'name': indicator_config.get('name', indicator_code)[:500],
                    'description': f"World Bank indicator {indicator_code}",
                    'unit': '',
                    'source': 'World Bank',
                    'topic': 'Economic/Social',
                    'created_at': datetime.now(timezone.utc),
                    'updated_at': datetime.now(timezone.utc)
                })
        
        logger.info(f"Fetched {len(indicators)} indicators")
        return indicators
    
    def fetch_observations(self) -> List[Dict[str, Any]]:
        """Fetch observation data from World Bank."""
        logger.info("Fetching observation data...")
        
        countries = self.config['world_bank']['countries']
        # Extract indicator codes from complex structure
        indicators = [ind['code'] for ind in self.config['world_bank']['indicators']]
        start_year = self.config['world_bank']['start_year']
        end_year = self.config['world_bank']['end_year']
        
        observations = []
        
        for indicator_code in indicators:
            logger.info(f"Fetching data for indicator: {indicator_code}")
            
            try:
                # Fetch data for all countries and years for this indicator
                df = wb.data.DataFrame(
                    series=indicator_code,
                    economy=countries,
                    time=range(start_year, end_year + 1),
                    skipAggs=True,  # Skip aggregates
                    numericTimeKeys=True
                )
                
                if df.empty:
                    logger.warning(f"⚠️ No data returned for {indicator_code}")
                    continue
                
                # Reset index to get country codes and years as columns
                df = df.reset_index()
                
                # Melt the dataframe to get observations in long format
                df_melted = df.melt(
                    id_vars=['economy'],
                    var_name='year',
                    value_name='value'
                )
                
                # Clean and prepare data
                df_melted = df_melted.dropna(subset=['value'])  # Remove null values
                df_melted['indicator_code'] = indicator_code
                df_melted['dataset_version'] = self.dataset_version
                df_melted['notes'] = f'Data sourced from World Bank API on {datetime.now().strftime("%Y-%m-%d")}'
                df_melted['created_at'] = datetime.now(timezone.utc)
                df_melted['updated_at'] = datetime.now(timezone.utc)
                
                # Rename columns to match database schema
                df_melted = df_melted.rename(columns={'economy': 'country_code'})
                
                # Convert to list of dictionaries
                indicator_observations = df_melted.to_dict('records')
                observations.extend(indicator_observations)
                
                logger.info(f"✅ Fetched {len(indicator_observations)} observations for {indicator_code}")
                
            except Exception as e:
                logger.error(f"❌ Error fetching data for {indicator_code}: {e}")
                continue
        
        logger.info(f"Total observations fetched: {len(observations)}")
        return observations
    
    def load_data(self, countries: List[Dict], indicators: List[Dict], observations: List[Dict], dry_run: bool = False):
        """Load data into database."""
        if dry_run:
            logger.info("DRY RUN - Would insert:")
            logger.info(f"  {len(countries)} countries")
            logger.info(f"  {len(indicators)} indicators")
            logger.info(f"  {len(observations)} observations")
            if observations:
                logger.info(f"  Sample observation: {observations[0]}")
            return
        
        logger.info("Loading data into database...")
        
        # Load countries
        if countries:
            rows_affected = bulk_upsert_countries(countries)
            logger.info(f"✅ Loaded {rows_affected} countries")
        
        # Load indicators
        if indicators:
            rows_affected = bulk_upsert_indicators(indicators)
            logger.info(f"✅ Loaded {rows_affected} indicators")
        
        # Load observations in batches
        if observations:
            total_loaded = 0
            for i in range(0, len(observations), self.batch_size):
                batch = observations[i:i + self.batch_size]
                rows_affected = bulk_upsert_observations(batch)
                total_loaded += rows_affected
                logger.info(f"✅ Loaded batch {i//self.batch_size + 1}: {rows_affected} observations")
            
            logger.info(f"✅ Total observations loaded: {total_loaded}")
    
    def run(self, dry_run: bool = False):
        """Run the complete ETL process."""
        logger.info("Starting World Bank ETL process...")
        
        try:
            # Fetch data
            countries = self.fetch_countries()
            indicators = self.fetch_indicators()
            observations = self.fetch_observations()
            
            # Load data
            self.load_data(countries, indicators, observations, dry_run)
            
            logger.info("✅ World Bank ETL process completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ ETL process failed: {e}")
            raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='World Bank Data ETL')
    parser.add_argument('--config', '-c', default='ingestion/config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run in dry-run mode (no database writes)')
    
    args = parser.parse_args()
    
    # Run ETL
    etl = WorldBankETL(args.config)
    etl.run(dry_run=args.dry_run)

if __name__ == "__main__":
    main()