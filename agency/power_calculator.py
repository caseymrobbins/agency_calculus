# agency/power_calculator.py
"""
Component: Power Calculator (Enhanced)
Purpose: Calculates a composite Power score (P) from multiple metrics based on the 
         Agency Calculus 4.3 formula. It supports the Adversarial Weighting 
         Protocol with predefined ideological weights and configurable normalization.
Integration: Used by the AgencyCalculator to determine perpetrator and victim 
             power for the Disparity Amplifier calculation.
"""
from typing import Dict, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- Configuration ---
# Predefined ideological weights for the Adversarial Weighting Protocol
IDEOLOGICAL_WEIGHTS = {
    "framework_average": {
        "economic": 0.4,
        "political": 0.3,
        "social_reach": 0.15,
        "social_centrality": 0.15
    },
    "libertarian": {
        "economic": 0.6,
        "political": 0.2,
        "social_reach": 0.1,
        "social_centrality": 0.1
    },
    "socialist": {
        "economic": 0.15,
        "political": 0.25,
        "social_reach": 0.3,
        "social_centrality": 0.3
    },
    "communitarian": {
        "economic": 0.25,
        "political": 0.25,
        "social_reach": 0.25,
        "social_centrality": 0.25
    }
}

@dataclass
class NormalizationConfig:
    """Configuration for normalizing raw power metrics."""
    max_economic_revenue: float = 1_000_000_000_000  # 1 Trillion USD
    max_social_reach: float = 1_000_000_000      # 1 Billion people

class PowerCalculator:
    """
    Calculates a composite power score based on normalized metrics and weights.
    Implements the formula:
    P = (P'_econ^w_e) * (P'_poli^w_p) * (P'_soc_reach^w_s1) * (P'_soc_centrality^w_s2)
    """
    def __init__(self, ideology: str = "framework_average", norm_config: Optional[NormalizationConfig] = None):
        if ideology not in IDEOLOGICAL_WEIGHTS:
            raise ValueError(f"Invalid ideology '{ideology}'. Supported: {list(IDEOLOGICAL_WEIGHTS.keys())}")
        
        self.ideology = ideology
        self.weights = IDEOLOGICAL_WEIGHTS[ideology]
        self.norm_config = norm_config or NormalizationConfig()

        if not np.isclose(sum(self.weights.values()), 1.0):
            logger.warning(f"Weights for ideology '{ideology}' do not sum to 1. Normalizing them.")
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}

    def _normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Normalizes raw input metrics to a 0-1 scale using log scaling for large numbers."""
        normalized = {
            "economic": np.log1p(metrics.get("economic_revenue", 0)) / np.log1p(self.norm_config.max_economic_revenue),
            "political": metrics.get("political_authority", 0),
            "social_reach": np.log1p(metrics.get("social_reach", 0)) / np.log1p(self.norm_config.max_social_reach),
            "social_centrality": metrics.get("network_centrality", 0)
        }

        for key, value in normalized.items():
            if not (0 <= value <= 1):
                logger.warning(f"Normalized value for '{key}' is outside  range: {value}")
                normalized[key] = np.clip(value, 0, 1)
        
        return normalized

    def calculate_power(self, metrics: Dict[str, float], return_components: bool = False) -> float | Tuple:
        """Calculates the final composite power score."""
        if not metrics:
            return (0.0, {}) if return_components else 0.0

        p_prime = self._normalize_metrics(metrics)
        epsilon = 1e-9  # To avoid log(0)

        # Using logs for numerical stability: log(P) = sum(w_i * log(P'_i))
        component_contributions = {
            key: self.weights[key] * np.log(p_prime[key] + epsilon) for key in self.weights
        }
        log_power = sum(component_contributions.values())
        power_score = np.exp(log_power)
        
        clipped_score = float(np.clip(power_score, 0, 1))

        if return_components:
            # Return the contribution of each component to the final score
            return clipped_score, {k: np.exp(v) for k, v in component_contributions.items()}
        
        return clipped_score

    def get_power_ratio(self, perpetrator_metrics: Dict, victim_metrics: Dict) -> float:
        """Calculates the direct power ratio between two entities."""
        perp_power = self.calculate_power(perpetrator_metrics)
        vic_power = self.calculate_power(victim_metrics)
        return perp_power / (vic_power + 1e-9)