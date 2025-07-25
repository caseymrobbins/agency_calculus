import json
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict, field

@dataclass
class BrittlenessResult:
    """Structured result for brittleness calculation."""
    brittleness_score: float
    raw_score: float
    total_agency: float
    nominal_gdp: float
    risk_level: str
    percentile: float
    timestamp: str
    ideology: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

class BrittlenessEngine:
    """Calculates the Systemic Brittleness score (B_sys)."""
    # Formula: B_sys = Nominal_GDP / A_total
    # The raw score is then scaled to a 0-10 range.
    RISK_THRESHOLDS = {
        "LOW": 3.0,
        "MEDIUM": 5.0,
        "HIGH": 7.0,
        "CRITICAL": 8.0
    }
    # Historical calibration data (would be loaded from database in production)
    # Fixed typos: 20e9 -> 2e9, 5oe9 -> 50e9, 100e9 assumed correct.
    HISTORICAL_CALIBRATION = {
        5: 1e9,  # 5th percentile (very stable)
        25: 5e9,  # 25th percentile
        50: 2e9,  # Median
        75: 100e9,  # 75th percentile
        90: 50e9,  # 90th percentile
        95: 200e9  # 95th percentile (very brittle)
    }

    def __init__(self, agency_calculator, min_gdp=1e6, max_gdp=1e15, output_scale=10.0):
        self.agency_calculator = agency_calculator
        self._min_gdp = min_gdp
        self._max_gdp = max_gdp
        self.output_scale = output_scale
        # Sort calibration for percentile calculation
        self.calibration_sorted = sorted(self.HISTORICAL_CALIBRATION.items())

    def calculate_brittleness(self, nominal_gdp: float, agency_scores: dict, power_metrics: dict = None) -> BrittlenessResult:
        warnings = []
        # Validate GDP
        if nominal_gdp <= 0:
            warnings.append(f"Invalid GDP: {nominal_gdp}, using minimum")
            nominal_gdp = self._min_gdp
        elif nominal_gdp > self._max_gdp:
            warnings.append(f"GDP exceeds maximum: {nominal_gdp}, capping")
            nominal_gdp = self._max_gdp

        # Calculate Total Agency
        total_agency = self.agency_calculator.calculate_total_agency(agency_scores, power_metrics)

        # Calculate raw brittleness score
        if total_agency == 0:
            raw_brittleness_score = float('inf')  # Handle division by zero
            warnings.append("Total agency is zero, setting raw score to inf")
        else:
            raw_brittleness_score = nominal_gdp / total_agency

        # Get percentile
        percentile = self._get_percentile(raw_brittleness_score)

        # Scale to 0-10
        if percentile <= 5:
            scaled_brittleness_score = 0.0
        elif percentile <= 95:
            scaled_brittleness_score = ((percentile - 5) / 90) * 9.5  # Linear from 5th to 95th
        else:
            scaled_brittleness_score = 9.5 + ((percentile - 95) / 5) * 0.5  # Last 5% to 0.5 points

        scaled_brittleness_score = float(np.clip(scaled_brittleness_score, 0, self.output_scale))

        # Determine risk level
        risk_level = "LOW"
        for level, threshold in sorted(self.RISK_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if scaled_brittleness_score >= threshold:
                risk_level = level
                break

        return BrittlenessResult(
            brittleness_score=round(scaled_brittleness_score, 2),
            raw_score=raw_brittleness_score,
            total_agency=round(total_agency, 4),
            nominal_gdp=nominal_gdp,
            risk_level=risk_level,
            percentile=round(percentile, 1),
            timestamp=datetime.now().isoformat(),
            ideology=self.agency_calculator.ideology,
            warnings=warnings
        )

    def _get_percentile(self, raw_score: float) -> float:
        """Calculate percentile based on historical calibration."""
        if raw_score <= self.calibration_sorted[0][1]:
            return 5.0
        if raw_score >= self.calibration_sorted[-1][1]:
            return 95.0
        # Interpolate between calibration points
        for i in range(len(self.calibration_sorted) - 1):
            low_perc, low_val = self.calibration_sorted[i]
            high_perc, high_val = self.calibration_sorted[i + 1]
            if low_val <= raw_score < high_val:
                return low_perc + ((raw_score - low_val) / (high_val - low_val)) * (high_perc - low_perc)
        return 95.0  # Default to high if not found