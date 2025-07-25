from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class NormalizationConfig:
    """Configuration for normalizing raw power metrics."""
    max_economic_revenue: float = 1_000_000_000_000  # 1 Trillion USD
    max_social_reach: float = 1_000_000_000  # 1 Billion people

@dataclass
class PowerCalculator:
    """Calculates composite power scores with ideological weighting."""

    IDEOLOGICAL_WEIGHTS = {
        "framework_average": {
            "economic": 0.3,
            "political": 0.3,
            "social_reach": 0.2,
            "social_centrality": 0.2
        },
        "libertarian": {
            "economic": 0.4,
            "political": 0.3,
            "social_reach": 0.15,
            "social_centrality": 0.15
        },
        "socialist": {
            "economic": 0.2,
            "political": 0.4,
            "social_reach": 0.2,
            "social_centrality": 0.2
        },
        "communitarian": {
            "economic": 0.25,
            "political": 0.25,
            "social_reach": 0.25,
            "social_centrality": 0.25
        }
    }

    def __init__(self, normalization_config: NormalizationConfig, ideology: str = "framework_average"):
        if ideology not in self.IDEOLOGICAL_WEIGHTS:
            raise ValueError(f"Invalid ideology: {ideology}")
        self.weights = self.IDEOLOGICAL_WEIGHTS[ideology]
        self.norm_config = normalization_config

    def _normalize_metric(self, value: float, max_value: float) -> float:
        """Normalize a single metric."""
        return min(value / max_value, 1.0) if max_value > 0 else 0.0

    def calculate_power(self, metrics: Dict[str, float], return_components: bool = False) -> float | Tuple[float, Dict[str, float]]:
        """Calculates the final composite power score."""
        if not metrics:
            return (0.0, {}) if return_components else 0.0

        normalized = {
            "economic": self._normalize_metric(metrics.get("economic", 0), self.norm_config.max_economic_revenue),
            "political": metrics.get("political", 0),  # Assuming already normalized 0-1
            "social_reach": self._normalize_metric(metrics.get("social_reach", 0), self.norm_config.max_social_reach),
            "social_centrality": metrics.get("social_centrality", 0)  # Assuming 0-1
        }

        components = {k: v * self.weights[k] for k, v in normalized.items()}
        power_score = sum(components.values())

        return (power_score, components) if return_components else power_score