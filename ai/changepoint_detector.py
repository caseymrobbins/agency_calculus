import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from scipy import stats
import logging

logger = logging.getLogger(__name__)

@dataclass
class Changepoint:
    """Structured representation of a detected abrupt changepoint."""
    timestamp: pd.Timestamp
    index: int
    magnitude: float
    direction: str
    affected_domains: List[str]
    confidence: float
    pre_mean: Dict[str, float] = field(default_factory=dict)
    post_mean: Dict[str, float] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)

class ChangepointDetector:
    """Detects abrupt changepoints in multivariate time series data."""

    def __init__(self, cusum_threshold: float = 5.0, min_window_size: int = 10, bootstrap_samples: int = 1000):
        self.cusum_threshold = cusum_threshold
        self.min_window_size = min_window_size
        self.bootstrap_samples = bootstrap_samples

    def detect_changepoints(self, data: pd.DataFrame) -> List[Changepoint]:
        """Detect changepoints in multivariate time series."""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex.")

        signal_columns = [col for col in data.columns if col not in ['country_code', 'year']]
        cusum = self._compute_multivariate_cusum(data[signal_columns])
        potential_points = np.where(np.abs(cusum) > self.cusum_threshold)[0]

        changepoints = []
        for point in potential_points:
            if point < self.min_window_size or point > len(data) - self.min_window_size:
                continue

            pre_data = data.iloc[:point]
            post_data = data.iloc[point:]

            magnitude = np.mean(np.abs(post_data.mean() - pre_data.mean()))
            direction = 'increase' if np.mean(post_data.mean() - pre_data.mean()) > 0 else 'decrease'
            affected = [col for col in signal_columns if abs(post_data[col].mean() - pre_data[col].mean()) > 0.1 * pre_data[col].std()]

            confidence, pre_mean, post_mean, sig = self._bootstrap_significance(pre_data, post_data, signal_columns)

            changepoints.append(Changepoint(
                timestamp=data.index[point],
                index=point,
                magnitude=magnitude,
                direction=direction,
                affected_domains=affected,
                confidence=confidence,
                pre_mean=pre_mean,
                post_mean=post_mean,
                statistical_significance=sig
            ))

        return sorted(changepoints, key=lambda x: x.index)

    def _compute_multivariate_cusum(self, data: pd.DataFrame) -> np.ndarray:
        """Compute cumulative sum for multivariate data."""
        mean = data.mean()
        cusum = np.cumsum(data - mean, axis=0)
        return np.linalg.norm(cusum, axis=1)

    def _bootstrap_significance(self, pre_data: pd.DataFrame, post_data: pd.DataFrame, signal_columns: List[str]) -> tuple:
        """Bootstrap test for change significance."""
        diffs = []
        for _ in range(self.bootstrap_samples):
            pre_sample = pre_data[signal_columns].sample(frac=1, replace=True).mean()
            post_sample = post_data[signal_columns].sample(frac=1, replace=True).mean()
            diffs.append(post_sample - pre_sample)

        diffs = np.array(diffs)
        p_values = {col: stats.ttest_ind(pre_data[col], post_data[col]).pvalue for col in signal_columns}
        confidence = 1 - np.mean([p_values[col] for col in signal_columns])

        pre_mean = pre_data[signal_columns].mean().to_dict()
        post_mean = post_data[signal_columns].mean().to_dict()

        return confidence, pre_mean, post_mean, p_values