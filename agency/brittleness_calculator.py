import json
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict, field
import warnings
from typing import Dict, Any
import yaml
import os

# Load config dynamically
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)

RISK_THRESHOLDS = CONFIG['brittleness']['risk_thresholds']
HISTORICAL_CALIBRATION = CONFIG['brittleness']['historical_calibration']

@dataclass
class BrittlenessResult:
    """Structured result for brittleness calculation."""
    brittleness_score: float
    raw_score: float
    total_agency: float
    nominal_gdp: float
    risk_level: str
    percentile: float
    timestamp: datetime = field(default_factory=datetime.now)
    iqa_notes: str = ""  # For IQA integration

@dataclass
class AgencyCalculation:
    """Result of agency calculation."""
    total_agency: float
    ideology: str
    domain_scores: Dict[str, float]
    domain_weights: Dict[str, float]
    domain_contributions: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ViolationImpactCalculation:
    """Detailed violation impact calculation."""
    violation_magnitude: float
    delta_agency: float
    attack_factor_c: float
    disparity_sensitivity_k: float
    disparity_amplifier: float
    power_ratio: float
    perpetrator_power: float
    victim_power: float

    def to_dict(self) -> Dict:
        return asdict(self)

class BrittlenessEngine:
    def __init__(self, agency_calculator, min_gdp=None, max_gdp=None, output_scale=None):
        self.agency_calculator = agency_calculator
        self.min_gdp = min_gdp or CONFIG['brittleness']['min_gdp']
        self.max_gdp = max_gdp or CONFIG['brittleness']['max_gdp']
        self.output_scale = output_scale or CONFIG['brittleness']['output_scale']
        self.calibration_sorted = sorted(HISTORICAL_CALIBRATION.items())

    def calculate_brittleness(self, nominal_gdp: float, agency_scores: dict, power_metrics: dict = None, iqa_notes: str = "") -> BrittlenessResult:
        if not (self.min_gdp <= nominal_gdp <= self.max_gdp):
            warnings.warn(f"GDP {nominal_gdp} clipped to range [{self.min_gdp}, {self.max_gdp}]")
            nominal_gdp = np.clip(nominal_gdp, self.min_gdp, self.max_gdp)

        required_domains = ["economic", "political", "social", "health", "educational"]
        if not all(domain in agency_scores for domain in required_domains):
            raise ValueError(f"Missing domains: {set(required_domains) - set(agency_scores)}")

        for domain, score in agency_scores.items():
            if not (0 <= score <= 1):
                raise ValueError(f"Score for {domain} must be in [0,1]: {score}")

        total_agency = self.agency_calculator.calculate_total_agency(agency_scores)
        if nominal_gdp <= 1:
            raw_brittleness_score = 0
        else:
            raw_brittleness_score = (1 - total_agency) / np.log(nominal_gdp)

        calibration_values = np.array([v for _, v in self.calibration_sorted])
        calibration_percentiles = np.array([p for p, _ in self.calibration_sorted])
        percentile = float(np.clip(np.interp(raw_brittleness_score, calibration_values, calibration_percentiles), 0, 100))

        if percentile <= 5:
            scaled_brittleness_score = (percentile / 5) * 0.5
        elif percentile <= 95:
            scaled_brittleness_score = 0.5 + ((percentile - 5) / 90) * 9
        else:
            scaled_brittleness_score = 9.5 + ((percentile - 95) / 5) * 0.5
        scaled_brittleness_score = float(np.clip(scaled_brittleness_score, 0, self.output_scale))

        risk_level = "LOW"
        for level, threshold in sorted(RISK_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if scaled_brittleness_score >= threshold:
                risk_level = level
                break

        return BrittlenessResult(
            brittleness_score=round(scaled_brittleness_score, 2),
            raw_score=raw_brittleness_score,
            total_agency=total_agency,
            nominal_gdp=nominal_gdp,
            risk_level=risk_level,
            percentile=percentile,
            iqa_notes=iqa_notes
        )

class AgencyCalculator:
    """Performs core calculations based on agency scores."""

    def __init__(self, ideology: str = "framework_average"):
        self.ideology = ideology
        self.domain_weights = CONFIG['agency']['domain_weights'].get(ideology, {"economic": 0.2, "political": 0.2, "social": 0.2, "health": 0.2, "educational": 0.2})

    def calculate_total_agency(self, agency_scores: Dict[str, float]) -> float:
        total_agency = sum(agency_scores.get(domain, 0) * weight for domain, weight in self.domain_weights.items())
        weight_sum = sum(self.domain_weights.values())
        return total_agency / weight_sum if weight_sum > 0 else 0

    def calculate_violation_impact(self, violation_magnitude: float, perpetrator_power: float, victim_power: float) -> ViolationImpactCalculation:
        attack_factor_c = CONFIG['violation_impact']['attack_factor_c']
        disparity_sensitivity_k = CONFIG['violation_impact']['disparity_sensitivity_k']
        power_ratio = perpetrator_power / max(victim_power, 1e-6)
        power_ratio = np.clip(power_ratio, CONFIG['violation_impact']['power_ratio_clip_min'], CONFIG['violation_impact']['power_ratio_clip_max'])
        disparity_amplifier = disparity_sensitivity_k * (power_ratio - 1)
        delta_agency = violation_magnitude * attack_factor_c * (1 + disparity_amplifier)
        return ViolationImpactCalculation(
            violation_magnitude=violation_magnitude,
            delta_agency=delta_agency,
            attack_factor_c=attack_factor_c,
            disparity_sensitivity_k=disparity_sensitivity_k,
            disparity_amplifier=disparity_amplifier,
            power_ratio=power_ratio,
            perpetrator_power=perpetrator_power,
            victim_power=victim_power
        )