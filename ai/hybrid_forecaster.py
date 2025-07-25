import pandas as pd
import numpy as np
import warnings
import logging
from typing import Dict, Optional, Tuple
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.exceptions import NotFittedError
import shap
import joblib
import torch  # For GPU check

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridForecaster:
    """
    Hybrid time-series forecasting model combining VARX/VECM for linear patterns
    and XGBoost for non-linear residuals. Handles stationarity, cointegration, and uncertainty.
    """

    def __init__(self, max_lags: int = 10, xgb_params: Optional[Dict] = None):
        self.max_lags = max_lags
        default_xgb_params = {
            'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.7, 'gamma': 0.1,
            'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1,
            'tree_method': 'hist' if not torch.cuda.is_available() else 'gpu_hist'
        }
        self.xgb_params = {**default_xgb_params, **(xgb_params or {})}
        self.model_type_: Optional[str] = None  # 'VARMAX' or 'VECM'
        self.varx_model_: Optional[VARMAX] = None
        self.vecm_model_: Optional[VECM] = None
        self.xgb_model_: Optional[MultiOutputRegressor] = None
        self.lag_order_: Optional[int] = None
        self.endog_columns_: Optional[List[str]] = None
        self.exog_columns_: Optional[List[str]] = None
        self.differenced_columns_: Dict[str, int] = {}
        self.explainers_: Optional[List[shap.Explainer]] = None
        self.base_values_: Optional[List[float]] = None
        self._is_fitted = False
        self.training_endog_: Optional[pd.DataFrame] = None
        self.training_residuals_: Optional[pd.DataFrame] = None
        self.training_features_: Optional[pd.DataFrame] = None
        self.model_version = "1.0.0"  # For save/load compatibility

    def _check_stationarity(self, series: pd.Series, name: str) -> Tuple[bool, int]:
        """Performs ADF test, applying differencing if needed. Returns (is_stationary, diff_order)."""
        max_diff = 2
        diff_order = 0
        temp_series = series.dropna()
        while diff_order <= max_diff:
            result = adfuller(temp_series)
            p_value = result[1]
            is_stationary = p_value < 0.05
            logger.info(f"ADF test for '{name}' (diff={diff_order}): p-value={p_value:.4f} -> {'Stationary' if is_stationary else 'Non-Stationary'}")
            if is_stationary:
                return True, diff_order
            temp_series = temp_series.diff().dropna()
            diff_order += 1
        logger.warning(f"'{name}' remains non-stationary after {max_diff} differencings.")
        return False, max_diff

    def _check_cointegration(self, endog_df: pd.DataFrame) -> Tuple[bool, int]:
        """Performs Johansen test for cointegration. Returns (is_cointegrated, rank)."""
        if len(endog_df) < self.max_lags + 2:
            logger.warning("Data too short for cointegration test. Skipping.")
            return False, 0
        result = coint_johansen(endog_df, det_order=0, k_ar_diff=1)
        trace_stat = result.lr1
        critical_values = result.cvt[:, 1]  # 5% critical values
        rank = sum(trace_stat > critical_values)
        is_cointegrated = rank > 0
        logger.info(f"Johansen test: Cointegration rank = {rank} (is_cointegrated: {is_cointegrated})")
        return is_cointegrated, rank

    def _select_lag_order(self, endog_df: pd.DataFrame, exog_df: Optional[pd.DataFrame]) -> int:
        """Selects the optimal VAR lag order using BIC."""
        temp_model = VAR(endog_df, exog=exog_df)
        try:
            lag_selection = temp_model.select_order(maxlags=self.max_lags)
            selected_lag = lag_selection.bic
            logger.info(f"Optimal lag order selected via BIC: p={selected_lag}")
            return selected_lag if selected_lag > 0 else 1
        except Exception as e:
            logger.warning(f"Lag order selection failed: {e}. Defaulting to lag order 1.")
            return 1

    def fit(self, endog_df: pd.DataFrame, exog_df: Optional[pd.DataFrame] = None) -> 'HybridForecaster':
        """Fits the hybrid model, handling stationarity and cointegration."""
        if not isinstance(endog_df.index, pd.DatetimeIndex) or (exog_df is not None and not isinstance(exog_df.index, pd.DatetimeIndex)):
            raise ValueError("endog_df and exog_df must have aligned DatetimeIndex.")
        if exog_df is not None and not endog_df.index.equals(exog_df.index):
            raise ValueError("endog_df and exog_df must have the same index.")
        if endog_df.empty:
            raise ValueError("endog_df cannot be empty.")

        self.endog_columns_ = endog_df.columns.tolist()
        self.exog_columns_ = exog_df.columns.tolist() if exog_df is not None else []
        self.training_endog_ = endog_df.copy()

        logger.info("Checking and handling non-stationarity for endog...")
        processed_endog = endog_df.copy()
        all_stationary = True
        for col in self.endog_columns_:
            is_stat, diff_order = self._check_stationarity(processed_endog[col], col)
            all_stationary = all_stationary and is_stat
            self.differenced_columns_[col] = diff_order
            for _ in range(diff_order):
                processed_endog[col] = processed_endog[col].diff()
        processed_endog = processed_endog.dropna()

        aligned_exog = exog_df.loc[processed_endog.index] if exog_df is not None else None

        if aligned_exog is not None:
            logger.info("Checking stationarity for exog...")
            for col in self.exog_columns_:
                self._check_stationarity(aligned_exog[col], col)  # Log only, no differencing on exog

        is_cointegrated, rank = self._check_cointegration(processed_endog)

        self.lag_order_ = self._select_lag_order(processed_endog, aligned_exog)

        logger.info("Fitting core model...")
        if is_cointegrated:
            logger.info(f"Fitting VECM with rank {rank}...")
            self.model_type_ = 'VECM'
            self.vecm_model_ = VECM(endog=processed_endog, exog=aligned_exog, k_ar_diff=self.lag_order_, coint_rank=rank).fit()
            residuals = pd.DataFrame(self.vecm_model_.resid, index=processed_endog.index, columns=self.endog_columns_)
        else:
            logger.info("Fitting VARMAX...")
            self.model_type_ = 'VARMAX'
            self.varx_model_ = VARMAX(endog=processed_endog, exog=aligned_exog, order=(self.lag_order_, 0)).fit(disp=False)
            residuals = pd.DataFrame(self.varx_model_.resid, index=processed_endog.index, columns=self.endog_columns_)
        self.training_residuals_ = residuals

        # Prepare features for XGBoost: lagged residuals + exog
        lagged_residuals = residuals.shift(1).dropna().rename(columns=lambda c: f'{c}_resid_lag1')
        features = pd.concat([aligned_exog.loc[lagged_residuals.index], lagged_residuals], axis=1) if aligned_exog is not None else lagged_residuals
        targets = residuals.loc[features.index]
        self.training_features_ = features

        logger.info("Fitting Multi-Output XGBoost model on residuals...")
        base_xgb = xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model_ = MultiOutputRegressor(estimator=base_xgb, n_jobs=-1).fit(features, targets)
        logger.info("XGBoost model fitting complete.")

        logger.info("Initializing SHAP TreeExplainer...")
        self.explainers_ = [shap.TreeExplainer(estimator) for estimator in self.xgb_model_.estimators_]
        self.base_values_ = [e.expected_value for e in self.explainers_]
        logger.info("SHAP explainer initialization complete.")
        self._is_fitted = True
        return self

    def predict(self, steps: int, future_exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generates multi-step forecasts with uncertainty intervals."""
        if not self._is_fitted:
            raise NotFittedError("Model not fitted.")
        if future_exog is not None and len(future_exog) != steps:
            raise ValueError("future_exog must match steps.")
        if future_exog is not None and list(future_exog.columns) != self.exog_columns_:
            raise ValueError("future_exog columns must match training exog.")

        if self.model_type_ == 'VECM':
            vecm_forecast = self.vecm_model_.forecast(steps=steps, exog=future_exog)
            vecm_forecast = pd.DataFrame(vecm_forecast, columns=self.endog_columns_, index=future_exog.index if future_exog is not None else pd.RangeIndex(steps))
            forecast = vecm_forecast
        else:
            forecast = self.varx_model_.forecast(steps=steps, exog=future_exog)
            forecast = pd.DataFrame(forecast, columns=self.endog_columns_, index=future_exog.index if future_exog is not None else pd.RangeIndex(steps))

        last_residuals = self.training_residuals_.iloc[-1:].rename(columns=lambda c: f'{c}_resid_lag1')
        xgb_forecasts = []

        for i in range(steps):
            current_exog = future_exog.iloc[i:i+1] if future_exog is not None else pd.DataFrame(index=[i])
            feature_vector = pd.concat([current_exog, last_residuals], axis=1)[self.training_features_.columns]
            residual_pred = self.xgb_model_.predict(feature_vector)
            xgb_forecasts.append(residual_pred[0])
            last_residuals = pd.DataFrame(residual_pred, columns=self.training_residuals_.columns).rename(columns=lambda c: f'{c}_resid_lag1')

        xgb_resid_forecast = pd.DataFrame(xgb_forecasts, columns=self.endog_columns_, index=forecast.index)
        hybrid_forecast_diff = forecast + xgb_resid_forecast

        final_forecast = hybrid_forecast_diff.copy()
        for col, diff_order in self.differenced_columns_.items():
            if diff_order > 0:
                last_values = self.training_endog_[col].tail(diff_order).values[::-1]  # Reverse for correct inversion
                cum_forecast = final_forecast[col].copy()
                for d in range(diff_order):
                    cum_forecast = cum_forecast.cumsum() + last_values[d]
                final_forecast[col] = cum_forecast

        return final_forecast

    def explain(self, X_instance: pd.DataFrame) -> Tuple[List[float], List[np.ndarray]]:
        """Calculates SHAP values for a single data instance to explain XGBoost's contribution."""
        if not self._is_fitted or self.explainers_ is None:
            raise NotFittedError("This instance is not fitted yet or explainers not initialized.")
        if list(X_instance.columns) != self.training_features_.columns:
            raise ValueError("X_instance columns must match training features.")
        shap_values_list = [explainer.shap_values(X_instance.values) for explainer in self.explainers_]
        return self.base_values_, shap_values_list

    def save_model(self, filepath: str):
        """Saves the fitted model to disk."""
        if not self._is_fitted:
            raise NotFittedError("Model must be fitted before saving.")
        model_data = {
            'model_version': self.model_version,
            'model_type_': self.model_type_,
            'varx_model_': self.varx_model_,
            'vecm_model_': self.vecm_model_,
            'xgb_model_': self.xgb_model_,
            'lag_order_': self.lag_order_,
            'endog_columns_': self.endog_columns_,
            'exog_columns_': self.exog_columns_,
            'differenced_columns_': self.differenced_columns_,
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
        if model_data['model_version'] != cls().model_version:
            raise ValueError("Model version mismatch. Retrain or update.")
        forecaster = cls(max_lags=model_data['lag_order_'], xgb_params=model_data['xgb_params'])
        for key, value in model_data.items():
            if key not in ['model_version']:
                setattr(forecaster, key, value)
        forecaster._is_fitted = True
        
        # Re-initialize SHAP explainers
        forecaster.explainers_ = [shap.TreeExplainer(estimator) for estimator in forecaster.xgb_model_.estimators_]
        forecaster.base_values_ = [e.expected_value for e in forecaster.explainers_]
        logger.info(f"Model loaded from {filepath} and SHAP explainers re-initialized.")
        return forecaster