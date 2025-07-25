import json
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, Any

@dataclass
class BrittlenessResult:
    """Result of brittleness calculation."""
    brittleness_score: float
    raw_score: float
    total_agency: float
    percentile: float
    risk_level: str = "LOW"

    def to_dict(self) -> Dict:
        return asdict(self)

class BrittlenessEngine:
    """Engine for calculating systemic brittleness based on agency scores."""

    # Risk thresholds as per AC4.3 guidelines
    RISK_THRESHOLDS = {
        "LOW": 0,
        "MEDIUM": 5,
        "HIGH": 7,
        "CRITICAL": 9
    }

    def __init__(self, historical_calibration: Dict[int, float], max_gdp: float = 1e12, output_scale: float = 10.0):
        """
        Initializes the BrittlenessEngine.

        Args:
            historical_calibration: Dict of year to average brittleness score for percentile calc.
            max_gdp: Maximum GDP for normalization.
            output_scale: Scale for final brittleness score (default 10.0).
        """
        self._historical_calibration = historical_calibration
        self._max_gdp = max_gdp
        self._output_scale = output_scale
        # Sort calibration for percentile calculation
        self._calibration_sorted = sorted(self._historical_calibration.items())

    def calculate_brittleness(self, nominal_gdp: float, agency_scores: Dict[str, float], return_details: bool = False) -> BrittlenessResult:
        """
        Calculates brittleness score.

        Args:
            nominal_gdp: Nominal GDP for the country.
            agency_scores: Dict of domain to agency score.
            return_details: If True, returns full result object.

        Returns:
            BrittlenessResult object.
        """
        # Calculate total agency as average of domain scores
        total_agency = np.mean(list(agency_scores.values()))

        # Raw brittleness score: inverse relationship with agency, scaled by GDP
        raw_score = (1 - total_agency) * (nominal_gdp / self._max_gdp)

        # Calculate percentile based on historical calibration
        percentile = np.percentile([score for _, score in self._calibration_sorted], raw_score * 100)

        # Scale brittleness score based on percentile
        if percentile < 5:
            scaled_brittleness_score = (percentile / 5) * 0.5  # First 5% to 0.5 points
        elif percentile < 95:
            scaled_brittleness_score = 0.5 + ((percentile - 5) / 90) * 9  # Linear from 5th to 95th
        else:
            scaled_brittleness_score = 9.5 + ((percentile - 95) / 5) * 0.5  # Last 5% to 0.5 points

        scaled_brittleness_score = float(np.clip(scaled_brittleness_score, 0, self._output_scale))

        # Determine risk level
        risk_level = "LOW"
        for level, threshold in sorted(self.RISK_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if scaled_brittleness_score >= threshold:
                risk_level = level
                break

        return BrittlenessResult(
            brittleness_score=round(scaled_brittleness_score, 2),
            raw_score=round(raw_score, 2),
            total_agency=round(total_agency, 2),
            percentile=round(percentile, 2),
            risk_level=risk_level
        )