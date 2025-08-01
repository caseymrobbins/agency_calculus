#!/usr/bin/env python3
"""
Investigate V-Dem data structure
"""

import pandas as pd
import numpy as np

def investigate_vdem_data():
    """Investigate the V-Dem dataset structure."""
    
    print("Loading V-Dem data...")
    df = pd.read_csv('data/raw/V-Dem-CY-Full+Others-v15.csv')
    
    print('=== V-Dem Data Overview ===')
    print(f'Total rows: {len(df):,}')
    print(f'Total columns: {len(df.columns):,}')
    print(f'Year range: {df["year"].min()} - {df["year"].max()}')
    print()
    
    print('=== Available Countries (top 20) ===')
    print(df['country_text_id'].value_counts().head(20))
    print()
    
    print('=== Check our target countries ===')
    target_countries = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'IND', 'BRA', 'RUS']
    for country in target_countries:
        subset = df[df['country_text_id'] == country]
        if len(subset) > 0:
            years = subset['year'].agg(['min', 'max'])
            recent_data = len(subset[subset['year'] >= 2000])
            print(f'{country}: {len(subset)} total rows, {recent_data} since 2000, years: {years["min"]}-{years["max"]}')
        else:
            print(f'{country}: No data found')
    print()
    
    print('=== Check our target indicators ===')
    target_indicators = [
        'v2x_polyarchy', 'v2x_libdem', 'v2x_rule', 'v2x_corr', 'v2x_civlib',
        'v2x_freexp', 'v2x_frassoc_thick', 'v2xel_frefair', 'v2x_jucon', 'v2x_legcon'
    ]
    
    found_indicators = []
    missing_indicators = []
    
    for indicator in target_indicators:
        if indicator in df.columns:
            non_null = df[indicator].notna().sum()
            recent_non_null = df[(df['year'] >= 2000) & df[indicator].notna()]
            country_coverage = len(recent_non_null['country_text_id'].unique())
            print(f'✅ {indicator}: {non_null:,} total values, {len(recent_non_null)} since 2000, {country_coverage} countries')
            found_indicators.append(indicator)
        else:
            print(f'❌ {indicator}: NOT FOUND')
            missing_indicators.append(indicator)
    print()
    
    # Look for similar indicators for missing ones
    if missing_indicators:
        print('=== Looking for similar indicators ===')
        for missing in missing_indicators:
            base = missing.split('_')[0] + '_' + missing.split('_')[1]  # e.g., v2x_legcon -> v2x_leg
            similar = [col for col in df.columns if base in col][:5]
            if similar:
                print(f'{missing} similar: {similar}')
        print()
    
    print('=== Sample V-Dem democracy indicators ===')
    democracy_cols = [col for col in df.columns if col.startswith('v2x_') and 'dem' in col]
    print(f'Democracy indicators found: {democracy_cols[:10]}')
    print()
    
    print('=== Sample data calculation ===')
    # Calculate what we should expect
    sample_countries = ['USA', 'DEU', 'FRA', 'GBR', 'JPN', 'IND']
    sample_indicators = found_indicators[:5]  # First 5 working indicators
    
    total_expected = 0
    for country in sample_countries:
        country_data = df[df['country_text_id'] == country]
        recent_data = country_data[(country_data['year'] >= 2000) & (country_data['year'] <= 2023)]
        
        if len(recent_data) > 0:
            for indicator in sample_indicators:
                valid_values = recent_data[indicator].notna().sum()
                total_expected += valid_values
            print(f'{country}: {len(recent_data)} years of data available')
    
    print(f'\nExpected observations for {len(sample_countries)} countries × {len(sample_indicators)} indicators: ~{total_expected}')
    print(f'You got: 1,350 (missing: {total_expected - 1350 if total_expected > 1350 else "less than expected"})')
    
    return found_indicators

if __name__ == "__main__":
    investigate_vdem_data()
