# ai/hybrid_forecaster.py
"""
Hybrid Forecasting Model (VARX + XGBoost)
Implements Task 3.1 of the Agency Monitor project

This module fuses econometric and machine learning approaches to create
a robust forecasting system for the five agency domains.
"""

# --- Core Library Imports ---
import pandas as pd
import numpy as np
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# --- Time Series Modeling Imports ---
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import adfuller, coint_johansen

# --- Machine Learning Imports ---
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# --- SHAP for Interpretability ---
import shap

# --- Utility Imports ---
import joblib
import json

# Suppress convergence warnings from statsmodels for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridForecaster:
    """
    A hybrid time-series forecasting model combining VARX for linear patterns
    and XGBoost for non-linear residual patterns.

    This class encapsulates the entire workflow:
    1. Selects the optimal lag order for the VARX model using BIC.
    2. Fits a VARX model to capture linear trends and interdependencies.
    3. Fits a multi-output XGBoost model on the residuals of the VARX model.
    4. Generates combined forecasts by summing the predictions from both models.
    5. Provides SHAP-based interpretability for the XGBoost component.
    """
    
    def __init__(self, max_lags: int = 10, xgb_params: Optional[Dict] = None):
        """
        Initializes the HybridForecaster.

        Args:
            max_lags (int): The maximum number of lags to test for VAR order selection.
            xgb_params (dict, optional): Hyperparameters for the XGBRegressor. 
                                        If None, default parameters are used.
        """
        self.max_lags = max_lags
        
        # Default XGBoost parameters optimized for time series
        default_xgb_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,  # Feature subsampling to handle correlation
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.xgb_params = {**default_xgb_params, **(xgb_params or {})}
        
        # Model components
        self.varx_model_ = None
        self.xgb_model_ = None
        self.lag_order_ = None
        self.endog_columns_ = None
        self.exog_columns_ = None
        self.differenced_columns_ = {}  # Track which columns were differenced
        
        # SHAP explainers
        self.explainers_ = None
        self.base_values_ = None
        
        # Training metadata
        self._is_fitted = False
        self.training_endog_ = None  # Store original endogenous data
        self.training_residuals_ = None
        self.training_features_ = None

    def _check_stationarity(self, series: pd.Series, name: str) -> bool:
        """
        Performs Augmented Dickey-Fuller test for stationarity.
        
        Args:
            series: Time series to test
            name: Name of the series for logging
            
        Returns:
            bool: True if stationary, False otherwise
        """
        result = adfuller(series.dropna())
        p_value = result[1]
        is_stationary = p_value < 0.05
        
        logger.info(f"ADF test for {name}: p-value = {p_value:.4f}, "
                   f"{'stationary' if is_stationary else 'non-stationary'}")
        
        return is_stationary

    def _select_lag_order(self, endog_df: pd.DataFrame, exog_df: pd.DataFrame) -> int:
        """
        Selects the optimal VAR lag order using the Bayesian Information Criterion (BIC).
        
        Args:
            endog_df: DataFrame of endogenous variables
            exog_df: DataFrame of exogenous variables
            
        Returns:
            int: Optimal lag order
        """
        # Use VAR for lag selection as the more complex VARMAX model lacks a
        # direct .select_order() method. This is a standard and efficient
        # approach to estimate a reasonable lag order for the system.
        temp_model = VAR(endog_df, exog=exog_df)
        lag_selection_results = temp_model.select_order(maxlags=self.max_lags)
        
        # Get lag order that minimizes BIC
        selected_lag = lag_selection_results.bic
        
        logger.info(f"Lag order selection results:")
        logger.info(f"  AIC: {lag_selection_results.aic}")
        logger.info(f"  BIC: {lag_selection_results.bic}")
        logger.info(f"  Optimal lag order selected via BIC: p={selected_lag}")
        
        return selected_lag

    def fit(self, endog_df: pd.DataFrame, exog_df: pd.DataFrame) -> 'HybridForecaster':
        """
        Fits the hybrid VARX + XGBoost model and initializes the SHAP explainer.

        Args:
            endog_df (pd.DataFrame): DataFrame of endogenous variables (the 5 agency domains),
                                    with a DatetimeIndex.
            exog_df (pd.DataFrame): DataFrame of exogenous variables (shock features),
                                   with a DatetimeIndex aligned with endog_df.
                                   
        Returns:
            self: Fitted HybridForecaster instance
        """
        # --- Input Validation ---
        if not isinstance(endog_df, pd.DataFrame) or not isinstance(exog_df, pd.DataFrame):
            raise TypeError("endog_df and exog_df must be pandas DataFrames.")
        if not isinstance(endog_df.index, pd.DatetimeIndex):
            raise TypeError("endog_df must have a DatetimeIndex.")
        if not endog_df.index.equals(exog_df.index):
            raise ValueError("endog_df and exog_df must have aligned indices.")
        
        self.endog_columns_ = endog_df.columns.tolist()
        self.exog_columns_ = exog_df.columns.tolist()
        self.training_endog_ = endog_df.copy()  # Store original data for integration
        
        logger.info(f"Fitting hybrid model with {len(self.endog_columns_)} endogenous "
                   f"and {len(self.exog_columns_)} exogenous variables")

        # --- Handle Non-Stationarity ---
        logger.info("Checking and handling non-stationarity...")
        processed_endog = endog_df.copy()
        self.differenced_columns_ = {}
        
        for col in self.endog_columns_:
            if not self._check_stationarity(processed_endog[col], col):
                logger.info(f"Applying first-order differencing to non-stationary column: {col}")
                self.differenced_columns_[col] = True
                processed_endog[col] = processed_endog[col].diff()
            else:
                self.differenced_columns_[col] = False
        
        # Drop NaN values created by differencing
        processed_endog = processed_endog.dropna()
        aligned_exog = exog_df.loc[processed_endog.index]

        # --- Step 1: Select Optimal Lag Order ---
        self.lag_order_ = self._select_lag_order(processed_endog, aligned_exog)

        # --- Step 2: Fit VARX Model ---
        logger.info("Fitting VARX model on processed data...")
        varx_model = VARMAX(
            endog=processed_endog, 
            exog=aligned_exog, 
            order=(self.lag_order_, 0),
            enforce_stationarity=False,  # We've already handled stationarity
            enforce_invertibility=False
        )
        
        try:
            self.varx_model_ = varx_model.fit(disp=False)
            logger.info("VARX model fitting complete.")
            
            # Log model diagnostics
            logger.info(f"VARX Log-likelihood: {self.varx_model_.llf:.2f}")
            logger.info(f"VARX AIC: {self.varx_model_.aic:.2f}")
            logger.info(f"VARX BIC: {self.varx_model_.bic:.2f}")
            
        except Exception as e:
            logger.error(f"VARX fitting failed: {e}")
            raise

        # --- Step 3: Extract Residuals ---
        varx_residuals = self.varx_model_.resid.copy()
        self.training_residuals_ = varx_residuals
        
        logger.info(f"Residuals shape: {varx_residuals.shape}")
        logger.info(f"Residuals mean: {varx_residuals.mean().mean():.4f}")
        logger.info(f"Residuals std: {varx_residuals.std().mean():.4f}")
        
        # --- Step 4: Prepare Data for XGBoost ---
        # Enhanced feature engineering: include lagged residuals AND lagged exogenous
        lagged_residuals = varx_residuals.shift(1)
        lagged_residuals.columns = [f'{col}_resid_lag1' for col in varx_residuals.columns]
        
        lagged_exog = aligned_exog.shift(1)
        lagged_exog.columns = [f'{col}_exog_lag1' for col in aligned_exog.columns]
        
        # Combine all features: current exog + lagged exog + lagged residuals
        xgb_features = pd.concat([aligned_exog, lagged_exog, lagged_residuals], axis=1).dropna()
        xgb_target = varx_residuals.loc[xgb_features.index]
        
        self.training_features_ = xgb_features
        
        X_train_xgb = xgb_features.values
        y_train_xgb = xgb_target.values

        # --- Step 5: Fit Multi-Output XGBoost Model ---
        logger.info("Fitting Multi-Output XGBoost model on residuals...")
        
        # Create base estimator with our parameters
        base_xgb = xgb.XGBRegressor(**self.xgb_params)
        
        # Wrap in MultiOutputRegressor for multi-target prediction
        self.xgb_model_ = MultiOutputRegressor(estimator=base_xgb, n_jobs=1)
        
        # Fit the model
        self.xgb_model_.fit(X_train_xgb, y_train_xgb)
        
        # Calculate in-sample performance
        xgb_pred = self.xgb_model_.predict(X_train_xgb)
        mse_scores = [mean_squared_error(y_train_xgb[:, i], xgb_pred[:, i]) 
                      for i in range(y_train_xgb.shape[1])]
        
        logger.info("XGBoost model fitting complete.")
        for i, (col, mse) in enumerate(zip(self.endog_columns_, mse_scores)):
            logger.info(f"  {col} residual MSE: {mse:.6f}")

        # --- Step 6: Initialize SHAP Explainers ---
        logger.info("Initializing SHAP TreeExplainer...")
        
        # For multi-output models, create an explainer for each output
        self.explainers_ = []
        self.base_values_ = []
        
        for i, estimator in enumerate(self.xgb_model_.estimators_):
            explainer = shap.TreeExplainer(estimator)
            self.explainers_.append(explainer)
            self.base_values_.append(explainer.expected_value)
            
        logger.info("SHAP explainer initialization complete.")
        
        self._is_fitted = True
        return self

    def predict(self, steps: int, future_exog_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates multi-step ahead forecasts.

        Args:
            steps (int): The number of steps to forecast into the future.
            future_exog_df (pd.DataFrame): DataFrame containing future values of the
                                          exogenous variables for the forecast horizon.

        Returns:
            pd.DataFrame: A DataFrame containing the combined hybrid forecasts.
        """
        if not self._is_fitted:
            raise NotFittedError("This HybridForecaster instance is not fitted yet.")
        if len(future_exog_df) != steps:
            raise ValueError("Length of `future_exog_df` must be equal to `steps`.")

        # --- Step 1: Generate Base Forecast from VARX ---
        logger.info(f"Generating {steps}-step ahead base forecast from VARX...")
        
        varx_forecast = self.varx_model_.forecast(steps=steps, exog=future_exog_df.values)
        varx_forecast = pd.DataFrame(
            varx_forecast, 
            columns=self.endog_columns_,
            index=future_exog_df.index
        )

        # --- Step 2: Recursively Forecast Residuals with XGBoost ---
        logger.info("Recursively forecasting residuals with XGBoost...")
        
        # Initialize with the last known values
        last_residuals = self.training_residuals_.iloc[-1:].values.flatten()
        last_exog = self.training_features_[self.exog_columns_].iloc[-1:].values.flatten()
        xgb_forecasts = []
        
        for i in range(steps):
            # Prepare feature vector matching training features
            # Current exogenous
            current_exog = future_exog_df.iloc[i:i+1].values.flatten()
            
            # Lagged exogenous
            lagged_exog_features = last_exog
            
            # Lagged residuals
            lagged_resid_features = last_residuals
            
            # Combine features in the same order as training
            feature_vector = np.hstack([
                current_exog,
                lagged_exog_features,
                lagged_resid_features
            ]).reshape(1, -1)
            
            # Predict residuals for current step
            residual_pred = self.xgb_model_.predict(feature_vector)
            xgb_forecasts.append(residual_pred[0])
            
            # Update for next iteration
            last_residuals = residual_pred[0]
            last_exog = current_exog
        
        # Convert to DataFrame
        xgb_resid_forecast = pd.DataFrame(
            xgb_forecasts, 
            columns=self.endog_columns_, 
            index=varx_forecast.index
        )

        # --- Step 3: Combine Forecasts ---
        logger.info("Combining VARX and XGBoost forecasts...")
        hybrid_forecast_diff = varx_forecast + xgb_resid_forecast
        
        # --- Step 4: Integrate Differenced Forecasts ---
        logger.info("Reversing differencing where applied...")
        final_forecast = hybrid_forecast_diff.copy()
        
        for col in self.endog_columns_:
            if self.differenced_columns_.get(col, False):
                # Get the last actual value from original training data
                last_actual_value = self.training_endog_[col].iloc[-1]
                # Cumulatively sum the differenced forecast and add to last actual
                final_forecast[col] = last_actual_value + hybrid_forecast_diff[col].cumsum()
        
        # Log forecast summary
        logger.info("Forecast summary:")
        logger.info(f"  VARX contribution mean: {varx_forecast.mean().mean():.4f}")
        logger.info(f"  XGBoost contribution mean: {xgb_resid_forecast.mean().mean():.4f}")
        logger.info(f"  Final forecast mean: {final_forecast.mean().mean():.4f}")
        
        return final_forecast

    def explain(self, X_instance: pd.DataFrame) -> Tuple[List[float], List[np.ndarray]]:
        """
        Calculates SHAP values for a single data instance to explain the
        XGBoost model's contribution to the forecast.

        Args:
            X_instance (pd.DataFrame): A single row DataFrame representing the features
                                      for the prediction to be explained.

        Returns:
            tuple: A tuple containing (base_values, shap_values_list), where
                  base_values is a list of base values for each output, and
                  shap_values_list is a list of SHAP value arrays for the instance,
                  one array for each output.
        """
        if not self._is_fitted:
            raise NotFittedError("This HybridForecaster instance is not fitted yet.")
        if not hasattr(self, 'explainers_'):
            raise AttributeError("SHAP explainers were not initialized. Call fit() first.")
        
        # Calculate SHAP values for each output
        shap_values_list = []
        for explainer in self.explainers_:
            shap_vals = explainer.shap_values(X_instance.values)
            shap_values_list.append(shap_vals)
        
        return (self.base_values_, shap_values_list)

    def get_residual_diagnostics(self) -> Dict[str, Any]:
        """
        Performs diagnostic tests on the VARX residuals.
        
        Returns:
            dict: Dictionary containing diagnostic test results
        """
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted first.")
        
        diagnostics = {}
        
        # Residual statistics
        residuals = self.training_residuals_
        diagnostics['residual_means'] = residuals.mean().to_dict()
        diagnostics['residual_stds'] = residuals.std().to_dict()
        
        # Check for residual stationarity
        diagnostics['residual_stationarity'] = {}
        for col in residuals.columns:
            is_stationary = self._check_stationarity(residuals[col], f"Residual_{col}")
            diagnostics['residual_stationarity'][col] = is_stationary
        
        # Ljung-Box test for residual autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox
        diagnostics['ljung_box_tests'] = {}
        
        for col in residuals.columns:
            lb_result = acorr_ljungbox(residuals[col], lags=10, return_df=True)
            # Check if any p-values are below 0.05 (indicating autocorrelation)
            has_autocorr = (lb_result['lb_pvalue'] < 0.05).any()
            diagnostics['ljung_box_tests'][col] = {
                'has_autocorrelation': bool(has_autocorr),
                'min_pvalue': float(lb_result['lb_pvalue'].min())
            }
        
        return diagnostics

    def save_model(self, filepath: str):
        """
        Saves the fitted model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before saving.")
        
        model_data = {
            'varx_model': self.varx_model_,
            'xgb_model': self.xgb_model_,
            'lag_order': self.lag_order_,
            'endog_columns': self.endog_columns_,
            'exog_columns': self.exog_columns_,
            'differenced_columns': self.differenced_columns_,
            'xgb_params': self.xgb_params,
            'training_endog': self.training_endog_,
            'training_residuals': self.training_residuals_,
            'training_features': self.training_features_
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'HybridForecaster':
        """
        Loads a fitted model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            HybridForecaster: Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        forecaster = cls(xgb_params=model_data['xgb_params'])
        
        # Restore fitted components
        forecaster.varx_model_ = model_data['varx_model']
        forecaster.xgb_model_ = model_data['xgb_model']
        forecaster.lag_order_ = model_data['lag_order']
        forecaster.endog_columns_ = model_data['endog_columns']
        forecaster.exog_columns_ = model_data['exog_columns']
        forecaster.differenced_columns_ = model_data.get('differenced_columns', {})
        forecaster.training_endog_ = model_data.get('training_endog')
        forecaster.training_residuals_ = model_data['training_residuals']
        forecaster.training_features_ = model_data['training_features']
        forecaster._is_fitted = True
        
        # Re-initialize SHAP explainers
        forecaster.explainers_ = []
        forecaster.base_values_ = []
        
        for estimator in forecaster.xgb_model_.estimators_:
            explainer = shap.TreeExplainer(estimator)
            forecaster.explainers_.append(explainer)
            forecaster.base_values_.append(explainer.expected_value)
        
        logger.info(f"Model loaded from {filepath}")
        return forecaster


# --- Example Usage and Testing ---
def create_sample_data(n_years: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates sample data for testing the HybridForecaster.
    
    Returns:
        tuple: (endog_df, exog_df) DataFrames with synthetic agency scores and shock features
    """
    # Create date index
    dates = pd.date_range(start='2000-01-01', periods=n_years, freq='Y')
    
    # Create synthetic endogenous variables (5 agency domains)
    np.random.seed(42)
    
    # Base trends for each domain
    trends = {
        'economic_agency': 0.7 + 0.01 * np.arange(n_years),
        'political_agency': 0.6 - 0.005 * np.arange(n_years),
        'social_agency': 0.65 + 0.002 * np.arange(n_years),
        'health_agency': 0.8 + 0.008 * np.arange(n_years),
        'educational_agency': 0.75 + 0.006 * np.arange(n_years)
    }
    
    # Add noise and create DataFrame
    endog_data = {}
    for domain, trend in trends.items():
        noise = np.random.normal(0, 0.05, n_years)
        endog_data[domain] = np.clip(trend + noise, 0, 1)
    
    endog_df = pd.DataFrame(endog_data, index=dates)
    
    # Create synthetic exogenous variables (shock features)
    exog_data = {
        'shock_magnitude': np.random.exponential(0.1, n_years),
        'recovery_slope': np.random.uniform(-0.1, 0.1, n_years),
        'time_since_shock': np.random.randint(0, 10, n_years),
        'polarization_index': np.random.uniform(0.2, 0.8, n_years)
    }
    
    exog_df = pd.DataFrame(exog_data, index=dates)
    
    return endog_df, exog_df


def main():
    """
    Demonstrates the complete workflow of the HybridForecaster.
    """
    logger.info("=== HybridForecaster Demo ===")
    
    # Step 1: Create sample data
    logger.info("\nStep 1: Creating sample data...")
    endog_df, exog_df = create_sample_data(n_years=30)
    
    # Split into train and test
    split_date = '2025-01-01'
    train_endog = endog_df[endog_df.index < split_date]
    train_exog = exog_df[exog_df.index < split_date]
    
    test_endog = endog_df[endog_df.index >= split_date]
    test_exog = exog_df[exog_df.index >= split_date]
    
    logger.info(f"Training samples: {len(train_endog)}")
    logger.info(f"Test samples: {len(test_endog)}")
    
    # Step 2: Initialize and fit the model
    logger.info("\nStep 2: Fitting HybridForecaster...")
    forecaster = HybridForecaster(max_lags=5)
    forecaster.fit(train_endog, train_exog)
    
    # Step 3: Generate forecast
    logger.info("\nStep 3: Generating forecast...")
    forecast_steps = len(test_endog)
    forecast = forecaster.predict(steps=forecast_steps, future_exog_df=test_exog)
    
    # Step 4: Evaluate forecast
    logger.info("\nStep 4: Evaluating forecast...")
    for col in endog_df.columns:
        mse = mean_squared_error(test_endog[col], forecast[col])
        logger.info(f"{col} - Test MSE: {mse:.6f}")
    
    # Step 5: Get residual diagnostics
    logger.info("\nStep 5: Residual diagnostics...")
    diagnostics = forecaster.get_residual_diagnostics()
    logger.info(f"Residual stationarity: {diagnostics['residual_stationarity']}")
    
    # Step 6: Demonstrate SHAP explanation
    logger.info("\nStep 6: SHAP explanation for last training instance...")
    last_features = forecaster.training_features_.iloc[-1:]
    base_values, shap_values = forecaster.explain(last_features)
    
    for i, domain in enumerate(forecaster.endog_columns_):
        logger.info(f"\n{domain}:")
        logger.info(f"  Base value: {base_values[i]:.4f}")
        logger.info("  Top 3 feature impacts:")
        
        # Get feature names and SHAP values
        feature_names = last_features.columns.tolist()
        shap_vals = shap_values[i][0]
        
        # Sort by absolute impact
        sorted_impacts = sorted(
            zip(feature_names, shap_vals), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:3]
        
        for feat, impact in sorted_impacts:
            logger.info(f"    {feat}: {impact:+.4f}")
    
    # Step 7: Save the model
    logger.info("\nStep 7: Saving model...")
    forecaster.save_model('models/hybrid_forecaster.pkl')
    
    # Step 8: Load and verify
    logger.info("\nStep 8: Loading saved model...")
    loaded_forecaster = HybridForecaster.load_model('models/hybrid_forecaster.pkl')
    
    # Verify by making same prediction
    loaded_forecast = loaded_forecaster.predict(steps=1, future_exog_df=test_exog.iloc[:1])
    logger.info(f"Loaded model forecast matches: {np.allclose(forecast.iloc[0], loaded_forecast.iloc[0])}")
    
    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    main()