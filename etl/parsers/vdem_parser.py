import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def parse_vdem_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses the V-Dem Country-Year Full+Others CSV file.

    This function implements the agreed-upon strategy:
    1. Reads the massive V-Dem CSV file.
    2. Excludes administrative metadata columns.
    3. Keeps 'country_text_id' (for ISO code) and 'year'.
    4. Keeps all substantive indicator columns from 'v2x_polyarchy' onwards.
    5. Transforms the data from its wide format to a long format suitable for
       our 'observations' table.

    Args:
        file_path (str): The full path to the V-Dem CSV file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                               represents a single observation record.
    """
    logger.info(f"Starting parsing of V-Dem file at: {file_path}")

    try:
        # Load the entire CSV into a pandas DataFrame
        df = pd.read_csv(file_path, low_memory=False)

        # --- Column Selection Logic ---
        # Identify the columns to keep. We start with our identifiers.
        id_vars = ['country_text_id', 'year']

        # Find the start and end of the substantive indicator columns
        try:
            start_col_index = df.columns.get_loc('v2x_polyarchy')
            end_col_index = df.columns.get_loc('e_pt_coup_attempts')
        except KeyError as e:
            logger.error(f"A required marker column was not found in the V-Dem CSV: {e}")
            raise

        # Get the list of all indicator columns to keep
        value_vars = df.columns[start_col_index : end_col_index + 1].tolist()

        # Select only the columns we need from the original DataFrame
        df_filtered = df[id_vars + value_vars]
        logger.info(f"Filtered down to {len(df_filtered.columns)} columns from the original {len(df.columns)}.")

        # --- Data Transformation (Melt) ---
        # Convert the DataFrame from wide to long format.
        # This will create a row for each indicator, for each country, for each year.
        df_long = pd.melt(
            df_filtered,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='indicator_code',
            value_name='value'
        )
        logger.info(f"Melted the DataFrame into {len(df_long)} observations.")

        # Rename 'country_text_id' to 'country_code' to match our database schema
        df_long.rename(columns={'country_text_id': 'country_code'}, inplace=True)

        # Drop rows where the value is missing (NaN)
        df_long.dropna(subset=['value'], inplace=True)
        logger.info(f"Removed missing values, {len(df_long)} valid observations remain.")

        # Convert the final DataFrame to a list of dictionaries for ingestion
        observations = df_long.to_dict(orient='records')
        
        logger.info("Successfully parsed V-Dem data.")
        return observations

    except FileNotFoundError:
        logger.error(f"V-Dem data file not found at path: {file_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during V-Dem parsing: {e}", exc_info=True)
        raise