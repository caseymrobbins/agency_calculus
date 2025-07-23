# etl/parsers/undp_parser.py

"""
Component: etl/parsers/undp_parser.py
Purpose: Parser for the UNDP Human Development Report dataset.
         Specifically targets "Mean years of schooling".
Inputs: Path to the UNDP data file (CSV format is preferred).
Outputs: A list of observation dictionaries in the standard format for ingestion.
Integration: Called by the etl/etl_bulk.py script.
"""

import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def parse_undp_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses the UNDP Human Development Report data file.

    Methodology:
    1.  Loads the CSV data, which is typically in a wide format where years are columns.
    2.  Identifies the required indicator: 'Mean years of schooling, adults (years) (ages 25 and older)'.
    3.  Uses pandas.melt to transform the data from its wide format (country x years)
        to a long format (country, year, value) suitable for our database.
    4.  Cleans the data, maps country codes, and sets the standard indicator_code.

    Args:
        file_path (str): The full path to the UNDP data file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a
                               single observation record for mean_years_of_schooling.
    """
    logger.info(f"Starting parsing of UNDP file at: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"Successfully loaded UNDP CSV file with {len(df)} rows.")

        # The specific indicator name we need from the UNDP data [cite: 2608]
        INDICATOR_NAME = 'Mean years of schooling, adults (years) (ages 25 and older)'

        # Filter the DataFrame to only the rows for our target indicator
        df_indicator = df[df['Indicator'] == INDICATOR_NAME]

        if df_indicator.empty:
            raise ValueError(f"Indicator '{INDICATOR_NAME}' not found in the file.")

        # --- Data Transformation (Melt) ---
        # The data is wide, with years as columns. We need to melt it to long format.
        id_vars = ['ISO_Code', 'Country']
        year_cols = [col for col in df_indicator.columns if col.isdigit()]
        
        df_long = pd.melt(
            df_indicator,
            id_vars=id_vars,
            value_vars=year_cols,
            var_name='year',
            value_name='value'
        )
        logger.info(f"Melted the DataFrame into {len(df_long)} observations.")

        # --- Data Cleaning and Standardization ---
        df_long.rename(columns={'ISO_Code': 'country_code'}, inplace=True)
        df_long['year'] = pd.to_numeric(df_long['year'])
        df_long.dropna(subset=['value'], inplace=True)

        df_long['indicator_code'] = 'mean_years_of_schooling'

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