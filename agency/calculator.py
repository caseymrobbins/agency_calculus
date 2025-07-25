from typing import Dict, Any, List
from dataclasses import dataclass, asdict

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

class AgencyCalculator:
    """Performs core calculations based on the Agency Calculus 4.3 framework."""
    AGENCY_WEIGHTS = {
        "framework_average": {"economic": 0.25, "political": 0.20, "social": 0.20, "health": 0.20, "educational": 0.15},
        # Add other ideologies with weights summing to 1.0
    }

    def __init__(self, ideology: str = "framework_average"):
        if ideology not in self.AGENCY_WEIGHTS:
            raise ValueError(f"Invalid ideology: {ideology}")
        self.ideology = ideology
        self.agency_weights = self.AGENCY_WEIGHTS[ideology]

    def calculate_total_agency(self, agency_scores: Dict[str, float], power_metrics: Dict[str, Any] = None) -> float:
        """Calculate weighted total agency."""
        contributions = {}
        total_agency = 0.0
        for domain, score in agency_scores.items():
            weight = self.agency_weights.get(domain, 0.0)
            contributions[domain] = score * weight
            total_agency += contributions[domain]
        # Ensure normalization if weights don't sum to 1
        weight_sum = sum(self.agency_weights.values())
        if weight_sum != 1.0:
            total_agency /= weight_sum
        return total_agency

    # Add other methods like calculate_violation_impact, but remove repetitions
    def calculate_violation_impact(self, violation_magnitude: float, perpetrator_power: float, victim_power: float) -> ViolationImpactCalculation:
        # Simplified; add full logic without repetitions
        attack_factor_c = 1.0  # Placeholder
        disparity_sensitivity_k = 1.0
        power_ratio = perpetrator_power / victim_power if victim_power != 0 else float('inf')
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