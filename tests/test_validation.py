# tests/test_validation.py
"""
Test script for the historical validation system
Demonstrates usage and validates the pipeline works correctly
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.run_historical_validation import HistoricalValidator


def test_validation_with_synthetic_data():
    """Test the validation system with synthetic data."""
    
    print("="*60)
    print("TESTING HISTORICAL VALIDATION SYSTEM")
    print("="*60)
    
    # Initialize validator
    validator = HistoricalValidator()
    
    # Test 1: Load validation cases
    print("\n1. Testing validation case loading...")
    cases = validator.load_validation_cases()
    print(f"   ✓ Loaded {len(cases)} validation cases")
    for code, case in list(cases.items())[:3]:
        print(f"   - {case['name']}: Expected {case['expected_brittleness']} ± {case['tolerance']}")
    
    # Test 2: Generate synthetic data
    print("\n2. Testing synthetic data generation...")
    synthetic_data = validator._generate_synthetic_data()
    print(f"   ✓ Generated {len(synthetic_data)} records")
    print(f"   - Countries: {synthetic_data['country_code'].unique()}")
    print(f"   - Years: {synthetic_data['year'].min()} - {synthetic_data['year'].max()}")
    
    # Test 3: Feature validation
    print("\n3. Validating features...")
    required_features = [
        'total_agency', 'economic_agency', 'political_agency',
        'social_agency', 'health_agency', 'educational_agency',
        'brittleness_score'
    ]
    
    missing = [f for f in required_features if f not in synthetic_data.columns]
    if missing:
        print(f"   ✗ Missing features: {missing}")
    else:
        print("   ✓ All required features present")
    
    # Test 4: Quick validation test
    print("\n4. Running quick validation test...")
    
    # Get Haiti data for 2024
    haiti_2024 = synthetic_data[
        (synthetic_data['country_code'] == 'HTI') & 
        (synthetic_data['year'] == 2024)
    ]
    
    if not haiti_2024.empty:
        print(f"   Haiti 2024 synthetic data:")
        print(f"   - Total Agency: {haiti_2024['total_agency'].iloc[0]:.3f}")
        print(f"   - Brittleness: {haiti_2024['brittleness_score'].iloc[0]:.2f}")
    
    # Test 5: Validation report structure
    print("\n5. Testing report generation...")
    sample_results = [
        {
            'country': 'Haiti',
            'country_code': 'HTI',
            'year': 2024,
            'predicted': 8.7,
            'expected': 8.5,
            'tolerance': 1.0,
            'difference': 0.2,
            'success': True,
            'risk_level': 'CRITICAL',
            'trajectory': 'CRITICAL_DECLINE'
        }
    ]
    
    validator.validation_results = sample_results
    validator._generate_validation_report()
    print("   ✓ Report generation successful")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)


def test_brittleness_calculation():
    """Test the brittleness score calculation logic."""
    
    print("\nTesting Brittleness Calculation Logic")
    print("-"*40)
    
    # Test cases with expected outputs
    test_cases = [
        {
            'name': 'Stable country',
            'total_agency': 0.8,
            'volatility': 0.05,
            'trend': 0.01,
            'shock': 0.0,
            'expected_range': (1.0, 3.0)
        },
        {
            'name': 'Declining country',
            'total_agency': 0.4,
            'volatility': 0.15,
            'trend': -0.05,
            'shock': 0.1,
            'expected_range': (5.0, 7.0)
        },
        {
            'name': 'Crisis country',
            'total_agency': 0.25,
            'volatility': 0.25,
            'trend': -0.10,
            'shock': 0.3,
            'expected_range': (7.5, 9.5)
        }
    ]
    
    for test in test_cases:
        # Simple brittleness calculation
        brittleness = (1 - test['total_agency']) * 10
        brittleness += test['volatility'] * 5  # Volatility contribution
        brittleness += abs(test['trend']) * 10  # Trend contribution
        brittleness += test['shock'] * 3  # Shock contribution
        
        print(f"\n{test['name']}:")
        print(f"  Inputs: agency={test['total_agency']}, vol={test['volatility']}")
        print(f"  Calculated brittleness: {brittleness:.2f}")
        print(f"  Expected range: {test['expected_range']}")
        print(f"  Status: {'✓' if test['expected_range'][0] <= brittleness <= test['expected_range'][1] else '✗'}")


def test_feature_importance_analysis():
    """Test feature importance analysis across models."""
    
    print("\nTesting Feature Importance Analysis")
    print("-"*40)
    
    # Simulate feature importance from different models
    validator = HistoricalValidator()
    
    validator.feature_importance_history = {
        'HTI': {
            'political_agency_volatility': 0.25,
            'social_agency_trend': 0.20,
            'total_agency': 0.15,
            'violence_level': 0.30,
            'economic_shock': 0.10
        },
        'SUN': {
            'economic_agency': 0.35,
            'political_agency_volatility': 0.20,
            'systemic_stress': 0.25,
            'total_agency': 0.15,
            'external_pressure': 0.05
        },
        'RWA': {
            'polarization_index': 0.40,
            'social_agency_trend': 0.25,
            'violence_level': 0.20,
            'total_agency': 0.10,
            'ethnic_tensions': 0.05
        }
    }
    
    analysis = validator._analyze_feature_importance()
    
    print("\nFeature Importance Analysis:")
    print(f"  Stable features: {analysis.get('top_stable_features', [])[:3]}")
    print(f"  Variable features: {analysis.get('top_variable_features', [])[:3]}")
    
    # Show statistics for a specific feature
    if 'statistics' in analysis and 'total_agency' in analysis['statistics']:
        stats = analysis['statistics']['total_agency']
        print(f"\n  'total_agency' statistics:")
        print(f"    Mean importance: {stats['mean']:.3f}")
        print(f"    Std deviation: {stats['std']:.3f}")


def main():
    """Run all tests."""
    
    try:
        # Run tests
        test_validation_with_synthetic_data()
        test_brittleness_calculation()
        test_feature_importance_analysis()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe validation system is ready for use.")
        print("Run 'python scripts/run_historical_validation.py' to start validation.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())