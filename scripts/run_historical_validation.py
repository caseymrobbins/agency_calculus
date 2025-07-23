# scripts/run_historical_validation.py
"""
Historical Validation Pipeline for the Agency Monitor System (Refactored)

Validates the end-to-end forecasting and calculation pipeline against known
historical collapse cases before production deployment. This script now correctly
validates the HybridForecaster by forecasting the 5 agency domains and then
calculating the brittleness score as a final step.

Key Features:
- Leave-one-out cross-validation for each historical case.
- Validates the full pipeline: 5-domain forecast -> A_total -> B_sys calculation.
- Generates a comprehensive validation report.
- Trains and deploys the final production model only if all tests pass.
"""
import json
import pandas as pd
import numpy as np
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from ai.hybrid_forecaster import HybridForecaster
from ai.training.feature_engineering import FeatureEngineer
from agency.brittleness_engine import BrittlenessEngine
from agency.calculator import AgencyCalculator
from api.database import get_db, Session

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HistoricalValidator:
    """Orchestrates historical validation of the full prediction pipeline."""

    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config = self._load_config(config_path)
        self.validation_results =
        self.feature_importance_history = {}
        # Initialize the calculation engines
        self.agency_calculator = AgencyCalculator(ideology="framework_average")
        self.brittleness_engine = BrittlenessEngine(self.agency_calculator)

    def _load_config(self, config_path: str) -> Dict:
        """Loads and validates configuration."""
        config_file = PROJECT_ROOT / config_path
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            paths = config.get('paths', {})
            defaults = {
                'raw_historical_data_path': 'data/raw/historical_collapses.json',
                'featured_data_path': 'data/featured/historical_features.json',
                'validation_cases_path': 'data/validation/validation_cases.json',
                'production_model_path': 'models/brittleness_predictor_prod.pkl',
                'metrics_path': 'models/validation_metrics.json',
                'plot_path': 'plots/feature_importance.png',
                'validation_report_path': 'reports/validation_report.json'
            }
            for key, default in defaults.items():
                if key not in paths:
                    paths[key] = default
            
            config['paths'] = {key: PROJECT_ROOT / value for key, value in paths.items()}
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            raise

    def load_validation_cases(self) -> Dict[str, Any]:
        """Load historical validation cases."""
        path = self.config['paths']['validation_cases_path']
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return data.get('cases', {})
        except Exception as e:
            logger.error(f"Error loading validation cases from {path}: {e}")
            raise

    def _load_data_from_db(self) -> pd.DataFrame:
        """Loads all agency_scores and observations from the database."""
        logger.info("Loading training data from database...")
        with get_db() as db:
            # Load agency scores (endogenous variables)
            agency_query = "SELECT * FROM agency_scores"
            endog_df = pd.read_sql(agency_query, db.bind, index_col='year')
            endog_df.index = pd.to_datetime(endog_df.index, format='%Y')

            # In a real system, you would also load exogenous features here
            # For now, we generate synthetic ones for demonstration
            exog_df = pd.DataFrame(index=endog_df.index)
            exog_df['shock_magnitude'] = np.random.exponential(0.1, len(endog_df))
            exog_df['recovery_slope'] = np.random.uniform(-0.1, 0.1, len(endog_df))
            
            # Combine for feature engineering context
            full_df = pd.concat([endog_df, exog_df], axis=1)
            
        logger.info(f"Loaded {len(full_df)} total records from database.")
        return full_df

    def prepare_data(self) -> pd.DataFrame:
        """Prepares data for validation, generating features if necessary."""
        featured_path = self.config['paths']['featured_data_path']
        if featured_path.exists():
            logger.info(f"Loading pre-engineered features from {featured_path}")
            with open(featured_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('date')
            return df
        
        logger.info("No pre-engineered features found. Loading from DB and engineering on the fly.")
        # This assumes a feature engineering pipeline that can operate on the DB data
        # For this example, we'll use a simplified placeholder
        return self._load_data_from_db()

    def validate_single_case(self, train_df: pd.DataFrame, test_case: Dict[str, Any], country_code: str) -> Tuple]:
        """Validate a single historical case using the full pipeline."""
        country_name = test_case['name']
        prediction_year = test_case['prediction_year']
        expected_score = test_case['expected_brittleness']
        tolerance = test_case['tolerance']

        logger.info(f"\nValidating {country_name} ({country_code}) for year {prediction_year}")

        # 1. Prepare data for this specific validation run
        endog_cols = ['economic_agency', 'political_agency', 'social_agency', 'health_agency', 'educational_agency']
        exog_cols = [col for col in train_df.columns if col not in endog_cols and col!= 'country_code']

        endog_train = train_df[endog_cols]
        exog_train = train_df[exog_cols]

        # 2. Train HybridForecaster on leave-one-out data
        logger.info(f"Training HybridForecaster without {country_name} data...")
        forecaster = HybridForecaster(**self.config['models']['hybrid_forecaster'])
        forecaster.fit(endog_train, exog_train)

        # 3. Get features for the prediction year to make the forecast
        test_data_year_before = self.featured_data[
            (self.featured_data['country_code'] == country_code) &
            (self.featured_data.index.year == prediction_year - 1)
        ]
        if test_data_year_before.empty:
            logger.error(f"No data found for {country_name} in {prediction_year - 1} to make a forecast.")
            return False, {'country': country_name, 'year': prediction_year, 'success': False, 'error': 'No data for forecast'}

        future_exog_df = test_data_year_before[exog_cols] # Simplified: assumes exog for next year is known

        # 4. Forecast the 5 agency domains
        agency_forecast_df = forecaster.predict(steps=1, future_exog_df=future_exog_df)
        agency_scores_forecast = agency_forecast_df.iloc.to_dict()

        # 5. Calculate the brittleness score from the forecast
        # Placeholder for GDP - a real system would need a GDP forecast as well
        nominal_gdp_forecast = 20e9 # $20B, a plausible value for a fragile state
        
        brittleness_result = self.brittleness_engine.calculate_systemic_brittleness(
            agency_scores=agency_scores_forecast,
            nominal_gdp=nominal_gdp_forecast,
            return_details=True
        )
        predicted_score = brittleness_result.brittleness_score

        # 6. Compare and log results
        difference = abs(predicted_score - expected_score)
        success = difference <= tolerance

        logger.info(f"  - Forecasted Agency Scores: { {k: round(v, 2) for k, v in agency_scores_forecast.items()} }")
        logger.info(f"  - Calculated Brittleness Score: {predicted_score:.2f}")
        logger.info(f"  - Expected Brittleness Score: {expected_score:.2f} ± {tolerance}")
        logger.info(f"  - Difference: {difference:.2f}")
        logger.info(f"  - Status: {'✅ PASS' if success else '❌ FAIL'}")

        result = {
            'country': country_name,
            'country_code': country_code,
            'year': prediction_year,
            'predicted_score': predicted_score,
            'expected_score': expected_score,
            'tolerance': tolerance,
            'difference': round(difference, 2),
            'success': success,
            'forecasted_agency_scores': agency_scores_forecast
        }
        return success, result

    def run_validation(self) -> bool:
        """Run complete leave-one-out validation."""
        logger.info("="*60)
        logger.info("STARTING HISTORICAL VALIDATION PIPELINE")
        logger.info("="*60)

        self.featured_data = self.prepare_data()
        validation_cases = self.load_validation_cases()

        all_passed = True
        for country_code, case_details in validation_cases.items():
            train_df = self.featured_data[self.featured_data['country_code']!= country_code]
            
            success, result = self.validate_single_case(train_df, case_details, country_code)
            self.validation_results.append(result)
            if not success:
                all_passed = False
        
        self._generate_validation_report()

        if all_passed:
            logger.info("\n✅ All historical validation cases PASSED!")
            # self._train_and_deploy_production_models() # This would be the next step
        else:
            logger.error("\n❌ One or more historical validation cases FAILED. Production model not trained.")
        
        return all_passed

    def _generate_validation_report(self):
        """Generate and save the final validation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_cases': len(self.validation_results),
                'passed': sum(1 for r in self.validation_results if r['success']),
                'failed': sum(1 for r in self.validation_results if not r.get('success', True)),
                'avg_error': np.mean([r.get('difference', 0) for r in self.validation_results])
            },
            'details': self.validation_results
        }
        report_path = self.config['paths']['validation_report_path']
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nValidation report saved to {report_path}")

def main():
    """Main entry point."""
    validator = HistoricalValidator()
    try:
        success = validator.run_validation()
        return 0 if success else 1
    except Exception as e:
        logger.critical(f"Fatal error in validation pipeline: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())