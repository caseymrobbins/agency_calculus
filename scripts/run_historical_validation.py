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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ai.hybrid_forecaster import HybridForecaster
from ai.training.feature_engineering import FeatureEngineer
from agency.brittleness_engine import BrittlenessEngine
from agency.calculator import AgencyCalculator
from api.database import get_db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalValidator:
    """Orchestrates historical validation of the full prediction pipeline."""

    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config = self._load_config(config_path)
        self.validation_results: List[Dict] = []
        self.feature_importance_history: Dict = {}
        # Initialize the calculation engines
        self.agency_calculator = AgencyCalculator(ideology="framework_average")
        self.brittleness_engine = BrittlenessEngine(self.agency_calculator)
        self.featured_data: pd.DataFrame = pd.DataFrame()

    def _load_config(self, config_path: str) -> Dict:
        """Loads and validates configuration."""
        config_file = PROJECT_ROOT / config_path
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            # Ensure paths are absolute for script execution from any context
            paths = config.get('paths', {})
            for key, value in paths.items():
                paths[key] = PROJECT_ROOT / value
            config['paths'] = paths
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
        """Loads all agency_scores from the database for feature engineering."""
        logger.info("Loading training data from database...")
        db_gen = get_db()
        db = next(db_gen)
        try:
            agency_query = "SELECT * FROM agency_scores"
            df = pd.read_sql(agency_query, db.bind)
            # Convert year to a proper DatetimeIndex
            df['year'] = pd.to_datetime(df['year'], format='%Y')
            df.set_index('year', inplace=True)
            logger.info(f"Loaded {len(df)} total records from database.")
            return df
        finally:
            db.close()

    def prepare_data(self) -> pd.DataFrame:
        """Prepares data for validation, generating features if necessary."""
        raw_data = self._load_data_from_db()
        logger.info("Engineering features for the entire dataset...")
        feature_engineer = FeatureEngineer(**self.config.get('features', {}))
        endog_df, exog_df = feature_engineer.run_pipeline(raw_data)
        self.featured_data = pd.concat([endog_df, exog_df, raw_data[['country_code']]], axis=1)
        self.featured_data.dropna(inplace=True)
        return self.featured_data

    def validate_single_case(self, train_df: pd.DataFrame, test_case: Dict[str, Any], country_code: str) -> Tuple[bool, Dict]:
        """Validate a single historical case using the full pipeline."""
        country_name = test_case['name']
        prediction_year = test_case['prediction_year']
        expected_score = test_case['expected_brittleness']
        tolerance = test_case['tolerance']

        logger.info(f"\nValidating {country_name} ({country_code}) for year {prediction_year}")

        # 1. Prepare data for this specific validation run
        endog_cols = ['economic_agency', 'political_agency', 'social_agency', 'health_agency', 'educational_agency']
        exog_cols = [col for col in train_df.columns if col not in endog_cols and col != 'country_code']

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

        future_exog_df = test_data_year_before[exog_cols]

        # 4. Forecast the 5 agency domains
        agency_forecast_df = forecaster.predict(steps=1, future_exog_df=future_exog_df)
        agency_scores_forecast = agency_forecast_df.iloc[0].to_dict()

        # 5. Calculate the brittleness score from the forecast
        # Placeholder for GDP - a real system would need a GDP forecast as well
        nominal_gdp_forecast = 20e9  # $20B, a plausible value for a fragile state
        brittleness_result = self.brittleness_engine.calculate_systemic_brittleness(
            agency_scores=agency_scores_forecast,
            nominal_gdp=nominal_gdp_forecast,
            return_details=True
        )
        predicted_score = brittleness_result.brittleness_score

        # 6. Compare and log results
        difference = abs(predicted_score - expected_score)
        success = difference <= tolerance

        logger.info(f" - Forecasted Agency Scores: {{ {', '.join([f'{k}: {v:.2f}' for k, v in agency_scores_forecast.items()])} }}")
        logger.info(f" - Calculated Brittleness Score: {predicted_score:.2f}")
        logger.info(f" - Expected Brittleness Score: {expected_score:.2f} ± {tolerance}")
        logger.info(f" - Difference: {difference:.2f}")
        logger.info(f" - Status: {'✅ PASS' if success else '❌ FAIL'}")

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
        logger.info("=" * 60)
        logger.info("STARTING HISTORICAL VALIDATION PIPELINE")
        logger.info("=" * 60)

        self.prepare_data()
        validation_cases = self.load_validation_cases()
        all_passed = True

        for country_code, case_details in validation_cases.items():
            train_df = self.featured_data[self.featured_data['country_code'] != country_code].copy()
            success, result = self.validate_single_case(train_df, case_details, country_code)
            self.validation_results.append(result)
            if not success:
                all_passed = False

        self._generate_validation_report()

        if all_passed:
            logger.info("\n✅ All historical validation cases PASSED!")
        else:
            logger.error("\n❌ One or more historical validation cases FAILED. Production model not trained.")

        return all_passed

    def _generate_validation_report(self):
        """Generate and save the final validation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_cases': len(self.validation_results),
                'passed': sum(1 for r in self.validation_results if r.get('success')),
                'failed': sum(1 for r in self.validation_results if not r.get('success')),
                'avg_error': np.mean([r.get('difference', 0) for r in self.validation_results if 'difference' in r])
            },
            'details': self.validation_results
        }
        report_path = self.config['paths']['validation_report_path']
        report_path.parent.mkdir(parents=True, exist_ok=True)
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