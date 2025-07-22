# scripts/run_historical_validation.py
"""
Historical Validation Pipeline for Brittleness Predictor
Validates model against known historical collapse cases before production deployment

Key Features:
- Leave-one-out cross-validation
- Validation against Haiti 2024, Soviet Union 1991, etc.
- Feature importance analysis
- Automated model deployment if all tests pass
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

from ai.training.feature_engineering import FeatureEngineer
from ai.brittleness_predictor import BrittlenessPredictor
from api.database import get_db, get_timeseries_data

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalValidator:
    """Orchestrates historical validation of brittleness predictions."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
        self.validation_results = []
        self.feature_importance_history = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Loads and validates configuration."""
        config_file = PROJECT_ROOT / config_path
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set default paths if not specified
            defaults = {
                'raw_historical_data_path': 'data/raw/historical_collapses.json',
                'featured_data_path': 'data/featured/historical_features.json',
                'validation_cases_path': 'data/validation/validation_cases.json',
                'production_model_path': 'models/brittleness_predictor_prod.pkl',
                'metrics_path': 'models/validation_metrics.json',
                'plot_path': 'plots/feature_importance.png',
                'validation_report_path': 'reports/validation_report.json'
            }
            
            paths = config.get('paths', {})
            for key, default in defaults.items():
                if key not in paths:
                    paths[key] = default
            
            # Convert to Path objects
            config['paths'] = {
                key: PROJECT_ROOT / value 
                for key, value in paths.items()
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            # Return minimal config
            return {'paths': {
                key: PROJECT_ROOT / value 
                for key, value in defaults.items()
            }}
    
    def load_validation_cases(self) -> Dict[str, Any]:
        """Load historical validation cases with enhanced validation."""
        
        # Default validation cases if file doesn't exist
        default_cases = {
            "HTI": {
                "name": "Haiti",
                "prediction_year": 2024,
                "expected_brittleness": 8.5,
                "tolerance": 1.0,
                "description": "Gang violence and state collapse",
                "key_indicators": ["political_agency", "social_agency", "violence_level"]
            },
            "SUN": {
                "name": "Soviet Union",
                "prediction_year": 1991,
                "expected_brittleness": 9.0,
                "tolerance": 1.0,
                "description": "Economic and political system collapse",
                "key_indicators": ["economic_agency", "political_agency", "systemic_stress"]
            },
            "RWA": {
                "name": "Rwanda",
                "prediction_year": 1994,
                "expected_brittleness": 9.5,
                "tolerance": 0.8,
                "description": "Ethnic violence and genocide",
                "key_indicators": ["social_agency", "polarization_index", "violence_level"]
            },
            "CHL": {
                "name": "Chile",
                "prediction_year": 1973,
                "expected_brittleness": 7.5,
                "tolerance": 1.2,
                "description": "Military coup and democratic breakdown",
                "key_indicators": ["political_agency", "polarization_index", "economic_agency"]
            },
            "IRN": {
                "name": "Iran",
                "prediction_year": 1979,
                "expected_brittleness": 8.0,
                "tolerance": 1.0,
                "description": "Revolution and regime change",
                "key_indicators": ["political_agency", "social_agency", "protest_intensity"]
            }
        }
        
        path = self.config['paths']['validation_cases_path']
        
        try:
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                cases = data.get('cases', data)
                
                # Validate structure
                for code, details in cases.items():
                    required = ['name', 'prediction_year', 'expected_brittleness', 'tolerance']
                    if not all(key in details for key in required):
                        logger.warning(f"Case {code} missing required fields, using defaults")
                        if code in default_cases:
                            cases[code] = default_cases[code]
                
                return cases
            else:
                logger.warning(f"Validation cases not found at {path}, using defaults")
                return default_cases
                
        except Exception as e:
            logger.error(f"Error loading validation cases: {e}")
            return default_cases
    
    def prepare_data(self) -> pd.DataFrame:
        """Load and prepare data, either from database or files."""
        
        featured_path = self.config['paths']['featured_data_path']
        
        # Try to load from featured data first
        if featured_path.exists():
            logger.info(f"Loading featured data from {featured_path}")
            with open(featured_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data.get('data', data))
        
        # Otherwise, generate features from raw data or database
        logger.info("Featured data not found, generating from raw data...")
        
        # Try database first
        try:
            return self._load_from_database()
        except Exception as e:
            logger.warning(f"Database load failed: {e}, trying raw files...")
            return self._load_from_files()
    
    def _load_from_database(self) -> pd.DataFrame:
        """Load data from database and engineer features."""
        
        all_data = []
        
        with get_db() as db:
            # Get all countries with historical collapse data
            countries = ['HTI', 'SUN', 'RWA', 'CHL', 'IRN', 'USA', 'GBR', 'FRA']
            
            for country_code in countries:
                # Get agency scores
                agency_scores = pd.read_sql_query(
                    """
                    SELECT year, economic_agency, political_agency, 
                           social_agency, health_agency, educational_agency
                    FROM agency_scores
                    WHERE country_code = %s
                    ORDER BY year
                    """,
                    db,
                    params=(country_code,)
                )
                
                if not agency_scores.empty:
                    agency_scores['country_code'] = country_code
                    agency_scores['total_agency'] = agency_scores[
                        ['economic_agency', 'political_agency', 'social_agency', 
                         'health_agency', 'educational_agency']
                    ].mean(axis=1)
                    
                    all_data.append(agency_scores)
        
        if not all_data:
            raise ValueError("No data found in database")
        
        # Combine and engineer features
        df = pd.concat(all_data, ignore_index=True)
        
        # Engineer features
        feature_engineer = FeatureEngineer()
        return feature_engineer.engineer_features_dataframe(df)
    
    def _load_from_files(self) -> pd.DataFrame:
        """Load and process raw historical data files."""
        
        raw_path = self.config['paths']['raw_historical_data_path']
        
        if not raw_path.exists():
            # Generate synthetic data for testing
            logger.warning("No raw data found, generating synthetic data for testing")
            return self._generate_synthetic_data()
        
        # Process raw data through feature engineering
        feature_engineer = FeatureEngineer(
            input_path=str(raw_path),
            output_path=str(self.config['paths']['featured_data_path'])
        )
        
        return feature_engineer.run_pipeline()
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic historical data for testing."""
        
        logger.warning("Generating synthetic data - replace with real data for production!")
        
        # Historical collapse cases with realistic patterns
        cases = {
            'HTI': {'base_agency': 0.4, 'decline_rate': -0.02, 'volatility': 0.15, 
                    'collapse_year': 2024, 'name': 'Haiti'},
            'SUN': {'base_agency': 0.5, 'decline_rate': -0.03, 'volatility': 0.12, 
                    'collapse_year': 1991, 'name': 'Soviet Union'},
            'RWA': {'base_agency': 0.45, 'decline_rate': -0.04, 'volatility': 0.18, 
                    'collapse_year': 1994, 'name': 'Rwanda'},
            'CHL': {'base_agency': 0.6, 'decline_rate': -0.015, 'volatility': 0.10, 
                    'collapse_year': 1973, 'name': 'Chile'},
            'IRN': {'base_agency': 0.55, 'decline_rate': -0.025, 'volatility': 0.14, 
                    'collapse_year': 1979, 'name': 'Iran'},
            # Stable countries for contrast
            'USA': {'base_agency': 0.75, 'decline_rate': 0.001, 'volatility': 0.05, 
                    'collapse_year': None, 'name': 'United States'},
            'GBR': {'base_agency': 0.72, 'decline_rate': 0.0005, 'volatility': 0.06, 
                    'collapse_year': None, 'name': 'United Kingdom'}
        }
        
        data = []
        
        for country_code, params in cases.items():
            years = range(1960, 2025)
            
            for i, year in enumerate(years):
                # Calculate agency scores with decline and volatility
                years_to_collapse = (params['collapse_year'] - year) if params['collapse_year'] else 100
                
                # Accelerate decline near collapse
                if params['collapse_year'] and 0 < years_to_collapse < 10:
                    decline_multiplier = (10 - years_to_collapse) / 5
                else:
                    decline_multiplier = 1.0
                
                base = params['base_agency'] + (i * params['decline_rate'] * decline_multiplier)
                noise = np.random.normal(0, params['volatility'])
                
                total_agency = np.clip(base + noise, 0.1, 0.9)
                
                # Domain-specific agencies
                domains = {
                    'economic_agency': total_agency + np.random.normal(0, 0.05),
                    'political_agency': total_agency + np.random.normal(-0.05, 0.05),
                    'social_agency': total_agency + np.random.normal(0, 0.05),
                    'health_agency': total_agency + np.random.normal(0.02, 0.05),
                    'educational_agency': total_agency + np.random.normal(0, 0.05)
                }
                
                # Clip all values
                for key in domains:
                    domains[key] = np.clip(domains[key], 0.05, 0.95)
                
                # Calculate brittleness (inverse of agency with adjustments)
                brittleness = (1 - total_agency) * 10
                
                # Add shock near collapse
                if params['collapse_year'] and 0 < years_to_collapse < 3:
                    brittleness += np.random.uniform(1, 2)
                
                record = {
                    'country_code': country_code,
                    'country_name': params['name'],
                    'year': year,
                    'total_agency': total_agency,
                    **domains,
                    'brittleness_score': np.clip(brittleness, 0, 10),
                    'polarization_index': 50 + (10 - years_to_collapse) if years_to_collapse < 10 else 50,
                    'systemic_stress': 0.1 + (0.8 * (10 - years_to_collapse) / 10) if years_to_collapse < 10 else 0.1
                }
                
                data.append(record)
        
        df = pd.DataFrame(data)
        
        # Add engineered features
        for col in ['total_agency', 'economic_agency', 'political_agency', 
                    'social_agency', 'health_agency', 'educational_agency']:
            # Volatility
            df[f'{col}_volatility'] = df.groupby('country_code')[col].transform(
                lambda x: x.rolling(5, min_periods=1).std()
            )
            
            # Trend
            df[f'{col}_trend'] = df.groupby('country_code')[col].transform(
                lambda x: x.rolling(5, min_periods=1).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
                )
            )
        
        return df
    
    def validate_single_case(self, 
                            predictor: BrittlenessPredictor,
                            train_df: pd.DataFrame,
                            test_case: Dict[str, Any],
                            country_code: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate a single historical case."""
        
        country_name = test_case['name']
        prediction_year = test_case['prediction_year']
        expected_score = test_case['expected_brittleness']
        tolerance = test_case['tolerance']
        
        logger.info(f"\nValidating {country_name} ({country_code}) for year {prediction_year}")
        
        # Train model excluding this country
        logger.info(f"Training model without {country_name} data...")
        predictor.train(train_data=train_df)
        
        # Get test data for prediction year
        test_data = self.featured_data[
            (self.featured_data['country_code'] == country_code) & 
            (self.featured_data['year'] == prediction_year)
        ]
        
        if test_data.empty:
            logger.error(f"No data found for {country_name} in {prediction_year}")
            return False, {
                'country': country_name,
                'year': prediction_year,
                'success': False,
                'error': 'No test data'
            }
        
        # Make prediction
        test_features = test_data.iloc[0].to_dict()
        prediction = predictor.predict(test_features, country_code)
        
        predicted_score = prediction.brittleness_score
        difference = abs(predicted_score - expected_score)
        success = difference <= tolerance
        
        # Log results
        logger.info(f"  Predicted: {predicted_score:.2f}")
        logger.info(f"  Expected: {expected_score:.2f} ± {tolerance}")
        logger.info(f"  Difference: {difference:.2f}")
        logger.info(f"  Status: {'✅ PASS' if success else '❌ FAIL'}")
        
        if prediction.top_risk_factors:
            logger.info("  Top risk factors:")
            for factor in prediction.top_risk_factors[:3]:
                logger.info(f"    - {factor['feature']}: {factor['contribution']:.3f}")
        
        # Store feature importance for this model
        if predictor.feature_importance:
            self.feature_importance_history[country_code] = predictor.feature_importance
        
        result = {
            'country': country_name,
            'country_code': country_code,
            'year': prediction_year,
            'predicted': round(predicted_score, 2),
            'expected': expected_score,
            'tolerance': tolerance,
            'difference': round(difference, 2),
            'success': success,
            'risk_level': prediction.risk_level,
            'trajectory': prediction.trajectory,
            'top_risk_factors': prediction.top_risk_factors[:5]
        }
        
        return success, result
    
    def run_validation(self) -> bool:
        """Run complete leave-one-out validation."""
        
        logger.info("="*60)
        logger.info("HISTORICAL VALIDATION PIPELINE")
        logger.info("="*60)
        
        # Setup
        self._setup_directories()
        
        # Load data
        logger.info("\n1. Loading and preparing data...")
        self.featured_data = self.prepare_data()
        logger.info(f"   Loaded {len(self.featured_data)} records for {self.featured_data['country_code'].nunique()} countries")
        
        # Load validation cases
        validation_cases = self.load_validation_cases()
        logger.info(f"\n2. Loaded {len(validation_cases)} validation cases")
        
        # Run leave-one-out validation
        logger.info("\n3. Running leave-one-out cross-validation...")
        all_passed = True
        
        for country_code, case_details in validation_cases.items():
            # Prepare training data (exclude test country)
            train_df = self.featured_data[
                self.featured_data['country_code'] != country_code
            ]
            
            # Create new predictor for each validation
            predictor = BrittlenessPredictor()
            
            # Validate
            success, result = self.validate_single_case(
                predictor, train_df, case_details, country_code
            )
            
            self.validation_results.append(result)
            if not success:
                all_passed = False
        
        # Generate report
        self._generate_validation_report()
        
        # Train final model if all passed
        if all_passed:
            logger.info("\n✅ All validations PASSED!")
            self._train_production_model()
            return True
        else:
            logger.error("\n❌ Some validations FAILED!")
            self._analyze_failures()
            return False
    
    def _train_production_model(self):
        """Train and save the production model using all data."""
        
        logger.info("\n4. Training production model on all data...")
        
        prod_predictor = BrittlenessPredictor()
        metrics = prod_predictor.train(train_data=self.featured_data)
        
        # Save metrics
        metrics_path = self.config['paths']['metrics_path']
        with open(metrics_path, 'w') as f:
            json.dump({
                'training_metrics': metrics,
                'validation_results': self.validation_results,
                'trained_at': datetime.now().isoformat(),
                'data_size': len(self.featured_data)
            }, f, indent=2)
        
        logger.info(f"   Metrics saved to {metrics_path}")
        
        # Save model
        model_path = self.config['paths']['production_model_path']
        prod_predictor.save_model(str(model_path))
        logger.info(f"   Model saved to {model_path}")
        
        # Generate plots
        self._generate_plots(prod_predictor)
    
    def _generate_plots(self, predictor: BrittlenessPredictor):
        """Generate analysis plots."""
        
        # Feature importance plot
        if predictor.feature_importance:
            plt.figure(figsize=(12, 8))
            
            # Get top 20 features
            top_features = list(predictor.feature_importance.items())[:20]
            features, importances = zip(*top_features)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            plt.barh(y_pos, importances, color='steelblue')
            plt.yticks(y_pos, features)
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Features for Brittleness Prediction')
            plt.tight_layout()
            
            plot_path = self.config['paths']['plot_path']
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"   Feature importance plot saved to {plot_path}")
            plt.close()
        
        # Validation results plot
        if self.validation_results:
            plt.figure(figsize=(10, 6))
            
            results_df = pd.DataFrame(self.validation_results)
            results_df['error'] = results_df['predicted'] - results_df['expected']
            
            # Bar plot of prediction errors
            colors = ['green' if s else 'red' for s in results_df['success']]
            plt.bar(results_df['country'], results_df['error'], color=colors)
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.xlabel('Country')
            plt.ylabel('Prediction Error (Predicted - Expected)')
            plt.title('Historical Validation: Prediction Errors')
            plt.xticks(rotation=45)
            
            # Add tolerance bands
            for i, row in results_df.iterrows():
                plt.axhspan(-row['tolerance'], row['tolerance'], 
                           alpha=0.1, color='gray')
            
            plt.tight_layout()
            validation_plot_path = self.config['paths']['plot_path'].parent / 'validation_errors.png'
            plt.savefig(validation_plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"   Validation plot saved to {validation_plot_path}")
            plt.close()
    
    def _generate_validation_report(self):
        """Generate detailed validation report."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_cases': len(self.validation_results),
                'passed': sum(1 for r in self.validation_results if r['success']),
                'failed': sum(1 for r in self.validation_results if not r['success']),
                'avg_error': np.mean([r['difference'] for r in self.validation_results])
            },
            'details': self.validation_results,
            'feature_importance_variations': self._analyze_feature_importance()
        }
        
        report_path = self.config['paths'].get(
            'validation_report_path',
            self.config['paths']['metrics_path'].parent / 'validation_report.json'
        )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nValidation report saved to {report_path}")
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze how feature importance varies across models."""
        
        if not self.feature_importance_history:
            return {}
        
        # Get all unique features
        all_features = set()
        for importances in self.feature_importance_history.values():
            all_features.update(importances.keys())
        
        # Calculate statistics
        feature_stats = {}
        for feature in all_features:
            values = [
                importances.get(feature, 0) 
                for importances in self.feature_importance_history.values()
            ]
            
            feature_stats[feature] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Sort by mean importance
        sorted_features = sorted(
            feature_stats.items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )
        
        return {
            'top_stable_features': [
                f for f, stats in sorted_features[:10]
                if stats['std'] < 0.1 * stats['mean']  # Low variation
            ],
            'top_variable_features': [
                f for f, stats in sorted_features
                if stats['std'] > 0.3 * stats['mean']  # High variation
            ][:5],
            'statistics': dict(sorted_features[:20])
        }
    
    def _analyze_failures(self):
        """Analyze validation failures in detail."""
        
        failures = [r for r in self.validation_results if not r['success']]
        
        if not failures:
            return
        
        logger.info("\n" + "="*60)
        logger.info("FAILURE ANALYSIS")
        logger.info("="*60)
        
        for failure in failures:
            logger.info(f"\n{failure['country']} ({failure['year']}):")
            logger.info(f"  Predicted: {failure['predicted']}")
            logger.info(f"  Expected: {failure['expected']}")
            logger.info(f"  Error: {failure['difference']} (tolerance: {failure['tolerance']})")
            
            if failure.get('top_risk_factors'):
                logger.info("  Risk factors that may have been misweighted:")
                for factor in failure['top_risk_factors']:
                    logger.info(f"    - {factor['feature']}: {factor.get('contribution', 'N/A')}")
    
    def _setup_directories(self):
        """Ensure all required directories exist."""
        
        for key, path in self.config['paths'].items():
            if isinstance(path, Path):
                path.parent.mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point."""
    
    validator = HistoricalValidator()
    
    try:
        success = validator.run_validation()
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Fatal error in validation pipeline: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())