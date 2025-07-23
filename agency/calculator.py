# agency/calculator.py
"""
Component: Agency Calculator (Enhanced)
Purpose: Implements the core mathematical formulas of Agency Calculus 4.3.
         Provides detailed, structured outputs for transparency and easy integration.
Integration: Consumes agency scores and power metrics. Its primary output,
             A_total, is the key input for the BrittlenessEngine.
"""
from typing import Dict, Optional, Tuple, List
import numpy as np
import logging
from dataclasses import dataclass, asdict
import json

from agency.power_calculator import PowerCalculator, IDEOLOGICAL_WEIGHTS

logger = logging.getLogger(__name__)

# --- Data Structures for Structured Output ---
@dataclass
class AgencyCalculation:
    """Structured result for Total Agency calculation."""
    total_agency: float
    ideology: str
    domain_scores: Dict[str, float]
    domain_weights: Dict[str, float]
    domain_contributions: Dict[str, float]

    def to_dict(self) -> Dict: return asdict(self)

@dataclass
class ViolationCalculation:
    """Structured result for Agency Violation calculation."""
    violation_magnitude: float
    delta_agency: float
    attack_factor_c: float
    disparity_sensitivity_k: float
    disparity_amplifier: float
    power_ratio: float
    perpetrator_power: float
    victim_power: float

    def to_dict(self) -> Dict: return asdict(self)

# --- Main Calculator Class ---
class AgencyCalculator:
    """Performs core calculations based on the Agency Calculus 4.3 framework."""

    AGENCY_WEIGHTS = {
        "framework_average": {'economic': 0.25, 'political': 0.20, 'social': 0.20, 'health': 0.20, 'educational': 0.15},
        "libertarian": {'economic': 0.5, 'political': 0.3, 'social': 0.05, 'health': 0.05, 'educational': 0.1},
        "socialist": {'economic': 0.15, 'political': 0.15, 'social': 0.25, 'health': 0.25, 'educational': 0.2},
        "communitarian": {'economic': 0.2, 'political': 0.2, 'social': 0.2, 'health': 0.2, 'educational': 0.2}
    }

    def __init__(self, ideology: str = "framework_average"):
        if ideology not in self.AGENCY_WEIGHTS:
            raise ValueError(f"Invalid ideology '{ideology}'. Supported: {list(self.AGENCY_WEIGHTS.keys())}")
        
        self.ideology = ideology
        self.agency_weights = self.AGENCY_WEIGHTS[ideology]
        self.power_calculator = PowerCalculator(ideology)
        
        if not np.isclose(sum(self.agency_weights.values()), 1.0):
            raise ValueError(f"Agency weights for '{ideology}' must sum to 1.")

    def calculate_total_agency(self, agency_scores: Dict[str, float], return_details: bool = True) -> float | AgencyCalculation:
        """Calculates Total Agency (A_total)."""
        contributions = {domain: agency_scores.get(domain, 0.0) * weight for domain, weight in self.agency_weights.items()}
        total_agency = sum(contributions.values())

        if not (0 <= total_agency <= 1):
            logger.warning(f"Calculated A_total ({total_agency:.3f}) is outside the expected [0, 1] range.")

        if return_details:
            return AgencyCalculation(
                total_agency=total_agency,
                ideology=self.ideology,
                domain_scores=agency_scores,
                domain_weights=self.agency_weights,
                domain_contributions=contributions
            )
        return total_agency

    def calculate_violation_magnitude(self, delta_agency: float, perpetrator_power_metrics: Dict, victim_power_metrics: Dict, attack_factor_c: float = 1.0, k: float = 1.0, return_details: bool = True) -> float | ViolationCalculation:
        """Calculates the full Agency Violation Magnitude (V_alpha)."""
        perp_power = self.power_calculator.calculate_power(perpetrator_power_metrics)
        vic_power = self.power_calculator.calculate_power(victim_power_metrics)
        power_ratio = perp_power / (vic_power + 1e-9)
        d_amp = power_ratio ** k
        v_alpha = attack_factor_c * abs(delta_agency) * d_amp

        if return_details:
            return ViolationCalculation(
                violation_magnitude=v_alpha, delta_agency=abs(delta_agency),
                attack_factor_c=attack_factor_c, disparity_sensitivity_k=k,
                disparity_amplifier=d_amp, power_ratio=power_ratio,
                perpetrator_power=perp_power, victim_power=vic_power
            )
        return v_alpha
    
    def analyze_agency_change(self, before_scores: Dict[str, float], after_scores: Dict[str, float]) -> Dict:
        """Analyzes the change between two sets of agency scores."""
        before_total = self.calculate_total_agency(before_scores, return_details=False)
        after_total = self.calculate_total_agency(after_scores, return_details=False)
        
        domain_changes = {domain: after_scores.get(domain, 0) - before_scores.get(domain, 0) for domain in self.agency_weights}
        
        most_decreased = min(domain_changes.items(), key=lambda item: item[1])
        most_increased = max(domain_changes.items(), key=lambda item: item[1])

        return {
            "total_change": after_total - before_total,
            "percentage_change": ((after_total - before_total) / (before_total + 1e-9)) * 100,
            "domain_changes": domain_changes,
            "most_decreased": most_decreased,
            "most_increased": most_increased
        }