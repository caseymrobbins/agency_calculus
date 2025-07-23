# ai/hybrid_forecaster.py
"""
Hybrid Forecasting Model (VARX + XGBoost) - Production Version

Implements the state-of-the-art forecasting architecture mandated by the
project's research and refinement plan. This module is the core predictive
engine of the Agency Monitor system, integrating with feature_engineering.py
and the main training scripts.
"""

# --- Core Library Imports ---
import pandas as pd
import numpy as np
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any

# --- Time Series Modeling Imports ---
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import adfuller

# --- Machine Learning Imports ---
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error

# --- SHAP for Interpretability ---
import shap

# --- Utility Imports ---
import joblib

# Suppress convergence warnings from statsmodels for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridForecaster:
    """
    A hybrid time-series forecasting model combining VARX for linear patterns
    and XGBoost for non-linear residual patterns. This class encapsulates the
    entire workflow, including automated stationarity handling, model diagnostics,
    and SHAP-based interpretability.
    """
    def __init__(self, max_lags: int = 10, xgb_params: Optional] = None):
        self.max_lags = max_lags
        default_xgb_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        self.xgb_params = {**default_xgb_params, **(xgb_params or {})}
        
        # Attributes to be populated during fitting
        self.varx_model_ = None
        self.xgb_model_ = None
        self.lag_order_ = None
        self.endog_columns_ = None
        self.exog_columns_ = None
        self.differenced_columns_ = {}
        self.explainers_ = None
        self.base_values_ = None
        self._is_fitted = False
        self.training_endog_ = None
        self.training_residuals_ = None
        self.training_features_ = None

    def _check_stationarity(self, series: pd.Series, name: str) -> Tuple[bool, int]:
        """Performs ADF test, applying differencing if needed. Returns (is_stationary, diff_order)."""
        max_diff = 2
        for d in range(max_diff + 1):
            result = adfuller(series.dropna())
            p_value = result
            is_stationary = p_value < 0.05
            logger.info(f"ADF test for '{name}' (diff={d}): p-value={p_value:.4f} -> {'Stationary' if is_stationary else 'Non-Stationary'}")
            if is_stationary:
                return True, d
            series = series.diff()
        logger.warning(f"'{name}' remains non-stationary after {max_diff} differencings.")
        return False, max_diff

    def _select_lag_order(self, endog_df: pd.DataFrame, exog_df: pd.DataFrame) -> int:
        """Selects the optimal VAR lag order using BIC."""
        temp_model = VAR(endog_df, exog=exog_df)
        try:
            lag_selection = temp_model.select_order(maxlags=self.max_lags)
            selected_lag = lag_selection.bic
            logger.info(f"Optimal lag order selected via BIC: p={selected_lag}")
            return selected_lag
        except Exception as e:
            logger.warning(f"Lag order selection failed: {e}. Defaulting to lag order 1.")
            return 1

    def fit(self, endog_df: pd.DataFrame, exog_df: pd.DataFrame) -> 'HybridForecaster':
        """Fits the hybrid VARX + XGBoost model and initializes the SHAP explainer."""
        if not endog_df.index.equals(exog_df.index):
            raise ValueError("endog_df and exog_df must have aligned DatetimeIndex.")
        
        self.endog_columns_ = endog_df.columns.tolist()
        self.exog_columns_ = exog_df.columns.tolist()
        self.training_endog_ = endog_df.copy()

        logger.info("Checking and handling non-stationarity...")
        processed_endog = endog_df.copy()
        for col in self.endog_columns_:
            is_stationary, diff_order = self._check_stationarity(processed_endog[col], col)
            self.differenced_columns_[col] = diff_order
            temp_series = processed_endog[col]
            for _ in range(diff_order):
                temp_series = temp_series.diff()
            processed_endog[col] = temp_series
        
        processed_endog = processed_endog.dropna()
        aligned_exog = exog_df.loc[processed_endog.index]

        self.lag_order_ = self._select_lag_order(processed_endog, aligned_exog)

        logger.info("Fitting VARX model...")
        varx_model = VARMAX(endog=processed_endog, exog=aligned_exog, order=(self.lag_order_, 0))
        self.varx_model_ = varx_model.fit(disp=False)
        logger.info(f"VARX model fitting complete. BIC: {self.varx_model_.bic:.2f}")

        self.training_residuals_ = self.varx_model_.resid.copy()

        lagged_residuals = self.training_residuals_.shift(1).rename(columns=lambda c: f'{c}_resid_lag1')
        xgb_features = pd.concat([aligned_exog, lagged_residuals], axis=1).dropna()
        xgb_target = self.training_residuals_.loc[xgb_features.index]
        self.training_features_ = xgb_features
        
        X_train_xgb = xgb_features.values
        y_train_xgb = xgb_target.values

        logger.info("Fitting Multi-Output XGBoost model on residuals...")
        base_xgb = xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model_ = MultiOutputRegressor(estimator=base_xgb, n_jobs=-1)
        self.xgb_model_.fit(X_train_xgb, y_train_xgb)
        logger.info("XGBoost model fitting complete.")

        logger.info("Initializing SHAP TreeExplainer...")
        if not hasattr(self.xgb_model_, 'estimators_') or not self.xgb_model_.estimators_:
            raise ValueError("XGBoost model not properly fitted or estimators are missing.")
        self.explainers_ =
        self.base_values_ = [e.expected_value for e in self.explainers_]
        logger.info("SHAP explainer initialization complete.")

        self._is_fitted = True
        return self

    def predict(self, steps: int, future_exog_df: pd.DataFrame) -> pd.DataFrame:
        """Generates multi-step ahead forecasts."""
        if not self._is_fitted:
            raise NotFittedError("This instance is not fitted yet.")
        if len(future_exog_df)!= steps:
            raise ValueError(f"Length of `future_exog_df` ({len(future_exog_df)}) must be equal to `steps` ({steps}).")
        if not set(self.exog_columns_).issubset(future_exog_df.columns):
            raise ValueError(f"future_exog_df must contain all exogenous columns: {self.exog_columns_}")

        varx_forecast = self.varx_model_.forecast(steps=steps, exog=future_exog_df[self.exog_columns_].values)
        varx_forecast = pd.DataFrame(varx_forecast, columns=self.endog_columns_, index=future_exog_df.index)

        last_residuals = self.training_residuals_.iloc[-1:].copy()
        xgb_forecasts =

        for i in range(steps):
            current_exog = future_exog_df[self.exog_columns_].iloc[i:i+1]
            lagged_resid_features = last_residuals.rename(columns=lambda c: f'{c}_resid_lag1')
            
            # Ensure feature vector matches training structure exactly
            feature_vector_df = pd.concat(, axis=1)
            feature_vector = feature_vector_df[self.training_features_.columns].values

            residual_pred = self.xgb_model_.predict(feature_vector)
            xgb_forecasts.append(residual_pred)
            last_residuals = pd.DataFrame(residual_pred, columns=self.endog_columns_)

        xgb_resid_forecast = pd.DataFrame(np.array(xgb_forecasts), columns=self.endog_columns_, index=varx_forecast.index)

        hybrid_forecast_diff = varx_forecast + xgb_resid_forecast
        
        final_forecast = hybrid_forecast_diff.copy()
        for col, diff_order in self.differenced_columns_.items():
            if diff_order > 0:
                last_values = self.training_endog_[col].tail(diff_order).tolist()
                final_forecast[col] = hybrid_forecast_diff[col].copy()
                for _ in range(diff_order):
                    final_forecast[col] = final_forecast[col].cumsum() + last_values.pop()
        
        return final_forecast

    def explain(self, X_instance: pd.DataFrame) -> Tuple[List[float], List[np.ndarray]]:
        """Calculates SHAP values for a single data instance."""
        if not self._is_fitted:
            raise NotFittedError("This instance is not fitted yet.")
        
        # Ensure the instance has the correct columns in the correct order
        X_instance_ordered = X_instance[self.training_features_.columns]
        
        shap_values_list = [explainer.shap_values(X_instance_ordered.values) for explainer in self.explainers_]
        return (self.base_values_, shap_values_list)

    def save_model(self, filepath: str):
        """Saves the fitted model to disk."""
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before saving.")
        
        # SHAP explainers cannot be pickled, so we exclude them
        model_data = {
            'varx_model': self.varx_model_,
            'xgb_model': self.xgb_model_,
            'lag_order': self.lag_order_,
            'endog_columns': self.endog_columns_,
            'exog_columns': self.exog_columns_,
            'differenced_columns': self.differenced_columns_,
            'xgb_params': self.xgb_params,
            'training_endog_': self.training_endog_,
            'training_residuals_': self.training_residuals_,
            'training_features_': self.training_features_
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'HybridForecaster':
        """Loads a fitted model from disk."""
        model_data = joblib.load(filepath)
        
        forecaster = cls(max_lags=model_data['lag_order'], xgb_params=model_data['xgb_params'])
        
        # Load all saved attributes
        for attr, value in model_data.items():
            setattr(forecaster, f"{attr}_" if not attr.endswith('_') else attr, value)
            
        forecaster._is_fitted = True
        
        # Re-initialize SHAP explainers, which are not saved
        if not hasattr(forecaster.xgb_model_, 'estimators_') or not forecaster.xgb_model_.estimators_:
            raise ValueError("Loaded XGBoost model has no fitted estimators.")
        forecaster.explainers_ =
        forecaster.base_values_ = [e.expected_value for e in forecaster.explainers_]
        
        logger.info(f"Model loaded from {filepath} and SHAP explainers re-initialized.")
        return forecaster