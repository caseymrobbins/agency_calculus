# ai/changepoint_detector.py

"""
Component: Changepoint Detector (Refined and Production-Ready)
Purpose: Encapsulates logic for detecting abrupt shifts (Pelt) and gradual changes
         (Bayesian) in multivariate time-series data. This is a core component
         for identifying systemic shocks in the Agency Calculus framework.
Inputs:
- A pandas DataFrame, indexed by time, containing numerical data.
- A list of columns to treat as the multivariate signal.
Outputs:
- A list of discrete changepoint objects (from Pelt).
- A DataFrame of changepoint probabilities for each time step (from Bayesian).
"""

# --- Core Library Imports ---
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats

# --- Specialized AI Library Imports ---
import ruptures as rpt
# sdt-python is a specialized library for Bayesian changepoint detection, chosen for its
# direct ability to generate the posterior probability distribution required by the AI plan.
# To install: pip install sdt-python
import sdt

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Dataclass for Structured Output ---
@dataclass
class Changepoint:
    """Structured representation of a detected abrupt changepoint"""
    timestamp: pd.Timestamp
    index: int
    magnitude: float
    direction: str
    affected_domains: List[str]
    confidence: float
    pre_mean: Dict[str, float] = field(default_factory=dict)
    post_mean: Dict[str, float] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Validation for key fields
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

    def to_dict(self) -> Dict:
        """Converts the dataclass to a JSON-serializable dictionary."""
        return asdict(self)

# --- Main Detector Class ---
class ChangepointDetector:
    """
    A configurable detector for finding both abrupt (Pelt) and gradual
    (Bayesian) changes in time-series data.
    """
    
    def __init__(self, model: str = "rbf", min_size: int = 3, penalty_scale: float = 2.0, significance_threshold: float = 0.05):
        """
        Initializes the ChangepointDetector.
        Args:
            model (str): Cost model for the ruptures (Pelt) algorithm.
                         'rbf' detects changes in distribution (recommended).
                         'l2' detects changes in the mean.
            min_size (int): The minimum size of a segment for the Pelt algorithm.
            penalty_scale (float): Multiplier for the Pelt penalty term.
            significance_threshold (float): P-value threshold for statistical tests.
        """
        self.model = model
        self.min_size = min_size
        self.penalty_scale = penalty_scale
        self.significance_threshold = significance_threshold
        self.logger = logging.getLogger(__name__)
        self.last_detection_results: Optional[Dict] = None

    def detect_pelt(self, data: pd.DataFrame, signal_columns: List[str]) -> List[Changepoint]:
        """Detects abrupt, discrete changepoints using the Pelt algorithm."""
        # ... (Implementation from previous response, no changes needed)
        pass # Placeholder for brevity, the full code is below

    def detect_bayesian(self, data: pd.DataFrame, signal_columns: List[str]) -> pd.DataFrame:
        """Detects changepoint probabilities using a Bayesian approach via sdt-python."""
        # ... (Implementation from previous response, no changes needed)
        pass # Placeholder for brevity, the full code is below
    
    def clear_cache(self):
        """Clears the last_detection_results to manage memory."""
        self.logger.info("Clearing cached detection results.")
        self.last_detection_results = None

    def _calculate_adaptive_penalty(self, signal: np.ndarray) -> float:
        # ... (Implementation from previous response, no changes needed)
        pass # Placeholder for brevity, the full code is below

    def _characterize_changepoint(self, data: pd.DataFrame, signal_columns: List[str], changepoint_idx: int) -> Changepoint:
        """Characterize a discrete changepoint found by Pelt with enhanced robustness."""
        # ... (Implementation from previous response with suggested robustness improvements)
        pass # Placeholder for brevity, the full code is below


# --- Full Class Implementation (with your refinements included) ---
class ChangepointDetector:
    def __init__(self, model: str = "rbf", min_size: int = 3, penalty_scale: float = 2.0, significance_threshold: float = 0.05):
        self.model = model
        self.min_size = min_size
        self.penalty_scale = penalty_scale
        self.significance_threshold = significance_threshold
        self.logger = logging.getLogger(__name__)
        self.last_detection_results: Optional[Dict] = None

    def detect_pelt(self, data: pd.DataFrame, signal_columns: List[str]) -> List[Changepoint]:
        if data.empty or not all(col in data.columns for col in signal_columns):
            self.logger.warning("Input data is empty or missing signal columns for Pelt.")
            return []
        signal = data[signal_columns].values
        if len(signal) < (self.min_size * 2):
            self.logger.info(f"Signal length ({len(signal)}) too short for Pelt detection.")
            return []
        try:
            penalty = self._calculate_adaptive_penalty(signal)
            self.logger.info(f"Running Pelt detection with model='{self.model}' and pen={penalty:.2f}...")
            algo = rpt.Pelt(model=self.model, min_size=self.min_size).fit(signal)
            indices = algo.predict(pen=penalty)[:-1]
            if not indices:
                self.logger.info("No significant abrupt changepoints detected by Pelt.")
                return []
            self.last_detection_results = {'data': data, 'signal_columns': signal_columns, 'changepoints_indices': indices}
            changepoints = [self._characterize_changepoint(data, signal_columns, idx) for idx in indices]
            self.logger.info(f"Detected {len(changepoints)} abrupt changepoints with Pelt.")
            return changepoints
        except Exception as e:
            self.logger.error(f"Pelt detection failed: {e}", exc_info=True)
            return []

    def detect_bayesian(self, data: pd.DataFrame, signal_columns: List[str]) -> pd.DataFrame:
        self.logger.info("Running Bayesian Changepoint Detection...")
        if data.empty or not all(col in data.columns for col in signal_columns):
            self.logger.warning("Input data is empty or missing signal columns for Bayesian.")
            return pd.DataFrame()
        try:
            signal = data[signal_columns].values
            detector = sdt.BayesianDetector()
            probabilities = detector.find_changepoints(signal)
            result_df = pd.DataFrame({'shock_probability': probabilities}, index=data.index)
            self.logger.info("Bayesian changepoint detection complete.")
            return result_df
        except Exception as e:
            self.logger.error(f"Bayesian detection failed: {e}", exc_info=True)
            return pd.DataFrame()

    def clear_cache(self):
        """Clears the last_detection_results to manage memory, as suggested."""
        self.logger.info("Clearing cached detection results.")
        self.last_detection_results = None

    def _calculate_adaptive_penalty(self, signal: np.ndarray) -> float:
        n = len(signal)
        base_penalty = self.penalty_scale * np.log(n)
        signal_std = np.std(signal, axis=0).mean()
        # Ensure variance factor is non-negative
        variance_factor = max(1.0, 1.0 + (signal_std - 0.1) * 2)
        penalty = base_penalty * variance_factor
        return max(1.0, min(penalty, 100.0)) # Added bounds to prevent extreme penalties

    def _characterize_changepoint(self, data: pd.DataFrame, signal_columns: List[str], changepoint_idx: int) -> Changepoint:
        window_size = max(self.min_size, int(len(data) * 0.1))
        pre_start = max(0, changepoint_idx - window_size)
        post_end = min(len(data), changepoint_idx + window_size)
        
        pre_data = data.iloc[pre_start:changepoint_idx][signal_columns]
        post_data = data.iloc[changepoint_idx:post_end][signal_columns]

        # Fallback for insufficient data for characterization
        if pre_data.empty or post_data.empty:
            return Changepoint(timestamp=data.index[changepoint_idx], index=changepoint_idx, magnitude=0, direction='unknown', affected_domains=[], confidence=0)

        pre_mean = pre_data.mean().to_dict()
        post_mean = post_data.mean().to_dict()
        changes, significance, affected_domains = {}, {}, []

        for col in signal_columns:
            change = post_mean.get(col, 0) - pre_mean.get(col, 0)
            changes[col] = change
            # Note: A non-parametric test like Mann-Whitney U could be more robust here.
            if len(pre_data[col].dropna()) > 2 and len(post_data[col].dropna()) > 2:
                _, p_value = stats.ttest_ind(pre_data[col], post_data[col], equal_var=False, nan_policy='omit')
                significance[col] = p_value if not np.isnan(p_value) else 1.0
                if significance[col] < self.significance_threshold:
                    affected_domains.append(col)
            else:
                significance[col] = 1.0
        
        pre_std = pre_data.std()
        normalized_changes = [(changes[col] / pre_std[col]) if pre_std[col] > 0 else 0 for col in signal_columns]
        magnitude = np.linalg.norm(normalized_changes)
        avg_change = np.mean(list(changes.values()))
        direction = 'improvement' if avg_change > 0 else 'deterioration'
        confidence = 1.0 - np.mean([significance[d] for d in affected_domains]) if affected_domains else 0.0
        
        return Changepoint(timestamp=data.index[changepoint_idx], index=changepoint_idx,
                           magnitude=magnitude, direction=direction, affected_domains=affected_domains,
                           confidence=confidence, pre_mean=pre_mean, post_mean=post_mean,
                           statistical_significance=significance)