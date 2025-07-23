## File: tests/test_ac4_indices_prod.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for module discovery
sys.path.append(str(Path(__file__).resolve().parents[1]))
from etl.processors.ac4_indices import process_ac4_indices, CONFIG

# Get column names from the loaded config for consistency
VDEM_COLS = CONFIG['vdem_columns']
WVS_COLS = CONFIG['wvs_columns']

@pytest.fixture
def sample_vdem_data() -> pd.DataFrame:
    """Provides a sample V-Dem DataFrame for testing."""
    return pd.DataFrame({
        VDEM_COLS['country']: ['USA', 'USA', 'CAN', 'SWE'],
        VDEM_COLS['year']: [2020, 2021, 2020, 2020],
        VDEM_COLS['polarization']: [2.0, 3.5, 4.0, np.nan],
        VDEM_COLS['bipartisanship']: [0.0, 1.0, -4.0, 5.0] # 5.0 is out of range
    })

@pytest.fixture
def sample_wvs_data() -> pd.DataFrame:
    """Provides a sample WVS DataFrame for testing."""
    return pd.DataFrame({
        WVS_COLS['country']: ['USA', 'USA', 'USA', 'USA', 'CAN', 'CAN'],
        WVS_COLS['year']: [2020, 2020, 2021, 2021, 2020, 2020],
        WVS_COLS['trust']: [1, 2, 2, 99, 1, 1] # 1=Trust, 2=No Trust, 99=Invalid
    })

def test_pipeline_integration_success(sample_vdem_data, sample_wvs_data):
    """Tests the end-to-end processing and merging of indices."""
    result_df = process_ac4_indices(sample_vdem_data, sample_wvs_data)
    
    # Check shape and columns
    assert len(result_df) == 4
    assert 'polarization_index' in result_df.columns
    assert 'bipartisanship_index' in result_df.columns
    assert 'social_trust_index' in result_df.columns
    
    # -- Validate specific calculations --
    # USA 2020
    usa_2020 = result_df[result_df['country_code'] == 'USA'].iloc[0]
    assert np.isclose(usa_2020['polarization_index'], 50.0) # (2.0 / 4.0) * 100
    assert np.isclose(usa_2020['bipartisanship_index'], 50.0) # ((0.0 - (-4.0)) / (4.0 - (-4.0))) * 100
    assert np.isclose(usa_2020['social_trust_index'], 50.0) # 1 trust out of 2 valid responses

    # CAN 2020
    can_2020 = result_df[result_df['country_code'] == 'CAN'].iloc[0]
    assert np.isclose(can_2020['polarization_index'], 100.0)
    assert np.isclose(can_2020['bipartisanship_index'], 0.0) # Clipped from < -4.0
    assert np.isclose(can_2020['social_trust_index'], 100.0) # 2 trust out of 2 valid

    # SWE 2020 (should have NaN for polarization and social_trust)
    swe_2020 = result_df[result_df['country_code'] == 'SWE'].iloc[0]
    assert pd.isna(swe_2020['polarization_index'])
    assert np.isclose(swe_2020['bipartisanship_index'], 100.0) # Clipped from > 4.0
    assert pd.isna(swe_2020['social_trust_index'])