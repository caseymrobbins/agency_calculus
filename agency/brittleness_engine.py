# agency/brittleness_engine.py
"""
Component: Brittleness Engine
Purpose: Calculates the Systemic Brittleness score (B_sys) based on the
         ratio of nominal wealth to multi-domain agency, as defined in
         the Agency Calculus 4.3 framework.
Integration: This is the final step in the core calculus chain. It consumes
             the A_total score from the AgencyCalculator and economic data
             to produce the final brittleness score for the API.
"""

from typing import Dict, Optional, List, Tuple
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import json

from agency.calculator import AgencyCalculator

logger = logging.getLogger(__name__)

@dataclass
class BrittlenessResult:
    """Structured output for brittleness calculations."""
    brittleness_score: float
    raw_score: float
    total_agency: float
    nominal_gdp: float
    risk_level: str
    percentile: float  # Historical percentile
    timestamp: str
    ideology: str
    warnings: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class BrittlenessEngine:
    """
    Calculates the Systemic Brittleness score (B_sys).
    Formula: B_sys = Nominal GDP / A_total
    The raw score is then scaled to a 0-10 range.
    """

    # Risk level thresholds (same as brittleness_predictor.py)
    RISK_THRESHOLDS = {
        'LOW': 3.0,
        'MEDIUM': 5.0,
        'HIGH': 7.0,
        'CRITICAL': 8.0
    }
    
    # Historical calibration data (would be loaded from database in production)
    # These represent historical B_sys values at different percentiles
    HISTORICAL_CALIBRATION = {
        5: 1e9,      # 5th percentile (very stable)
        25: 5e9,     # 25th percentile
        50: 20e9,    # Median
        75: 100e9,   # 75th percentile
        90: 500e9,   # 90th percentile
        95: 1e12,    # 95th percentile (pre-collapse)
        99: 5e12     # 99th percentile (active collapse)
    }

    def __init__(self, 
                 agency_calculator: AgencyCalculator,
                 calibration_data: Optional[Dict[float, float]] = None):
        """
        Initializes the BrittlenessEngine.

        Args:
            agency_calculator: AgencyCalculator instance for computing A_total
            calibration_data: Historical percentile calibration data
        """
        self.agency_calculator = agency_calculator
        self.calibration = calibration_data or self.HISTORICAL_CALIBRATION
        
        # Derive scaling parameters from calibration
        self._update_scaling_parameters()
        
        self.output_scale = 10
        self._min_gdp = 1e6  # $1M minimum to avoid numerical issues
        self._max_gdp = 1e15  # $1 Quadrillion maximum

    def _update_scaling_parameters(self):
        """Updates min/max raw scores based on calibration data."""
        if self.calibration:
            self.min_raw_score = self.calibration.get(5, 1e9)
            self.max_raw_score = self.calibration.get(95, 1e12)
        else:
            self.min_raw_score = 5e9
            self.max_raw_score = 50e12

    def _scale_score(self, raw_score: float) -> float:
        """
        Scales the raw B_sys score to a 0-10 range using logarithmic scaling.
        
        This uses a sigmoid-like transformation to ensure:
        - Scores cluster around 2-4 for stable countries
        - Rapid increase between 6-8 for deteriorating situations
        - Asymptotic approach to 10 for extreme cases
        """
        if raw_score <= self.min_raw_score:
            return 0.0
        
        # Use percentile-based scaling if calibration available
        if self.calibration:
            # Find which percentile bracket this falls into
            percentile = self._get_percentile(raw_score)
            
            # Map percentiles to 0-10 scale
            # 5th percentile -> 0, 95th percentile -> 9.5
            scaled_score = (percentile / 100) * self.output_scale
        else:
            # Fallback to logarithmic scaling
            log_min = np.log10(self.min_raw_score)
            log_max = np.log10(self.max_raw_score)
            log_raw = np.log10(raw_score)
            
            scaled_log = (log_raw - log_min) / (log_max - log_min)
            scaled_score = scaled_log * self.output_scale
        
        return float(np.clip(scaled_score, 0, self.output_scale))

    def _get_percentile(self, raw_score: float) -> float:
        """
        Estimates the historical percentile for a raw brittleness score.
        Uses linear interpolation between calibration points.
        """
        if not self.calibration:
            return 50.0  # Default to median if no calibration
        
        # Sort calibration points
        percentiles = sorted(self.calibration.keys())
        values = [self.calibration[p] for p in percentiles]
        
        # Handle edge cases
        if raw_score <= values[0]:
            return percentiles[0]
        if raw_score >= values[-1]:
            return percentiles[-1]
        
        # Linear interpolation
        for i in range(len(values) - 1):
            if values[i] <= raw_score <= values[i + 1]:
                # Interpolate between percentiles
                p1, p2 = percentiles[i], percentiles[i + 1]
                v1, v2 = values[i], values[i + 1]
                
                if v2 == v1:  # Avoid division by zero
                    return p1
                
                fraction = (raw_score - v1) / (v2 - v1)
                return p1 + fraction * (p2 - p1)
        
        return 50.0  # Fallback

    def _get_risk_level(self, brittleness_score: float) -> str:
        """Determines risk level from brittleness score."""
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if brittleness_score >= self.RISK_THRESHOLDS[level]:
                return level
        return 'LOW'

    def calculate_systemic_brittleness(self,
                                     agency_scores: Dict[str, float],
                                     nominal_gdp: float,
                                     return_details: bool = True) -> Union[float, BrittlenessResult]:
        """
        Calculates the final, scaled Systemic Brittleness score.

        Args:
            agency_scores: The five domain agency scores
            nominal_gdp: Country's nominal GDP in USD
            return_details: If True, returns detailed BrittlenessResult

        Returns:
            float or BrittlenessResult: Brittleness score (0-10) or detailed result
        """
        warnings = []
        
        # Validate GDP
        if nominal_gdp <= 0:
            warnings.append(f"Invalid GDP: {nominal_gdp}, using minimum")
            nominal_gdp = self._min_gdp
        elif nominal_gdp > self._max_gdp:
            warnings.append(f"GDP exceeds maximum: {nominal_gdp}, capping")
            nominal_gdp = self._max_gdp
        
        # Calculate Total Agency
        total_agency = self.agency_calculator.calculate_total_agency(agency_scores)
        
        if total_agency <= 0:
            warnings.append(f"Total agency <= 0: {total_agency}")
            scaled_brittleness_score = self.output_scale
            raw_brittleness_score = float('inf')
            percentile = 100.0
        else:
            # Calculate raw brittleness
            raw_brittleness_score = nominal_gdp / total_agency
            
            # Get percentile
            percentile = self._get_percentile(raw_brittleness_score)
            
            # Scale to 0-10
            scaled_brittleness_score = self._scale_score(raw_brittleness_score)
            
            # Log extreme values
            if scaled_brittleness_score >= 8.0:
                logger.warning(
                    f"High brittleness detected: {scaled_brittleness_score:.2f} "
                    f"(GDP: ${nominal_gdp/1e9:.1f}B, A_total: {total_agency:.3f})"
                )
        
        if not return_details:
            return scaled_brittleness_score
        
        # Determine risk level
        risk_level = self._get_risk_level(scaled_brittleness_score)
        
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

    def calculate_brittleness_trajectory(self,
                                       historical_data: List[Tuple[Dict[str, float], float]],
                                       window: int = 5) -> Dict[str, float]:
        """
        Calculates brittleness trajectory over time.
        
        Args:
            historical_data: List of (agency_scores, gdp) tuples
            window: Number of periods for trend calculation
            
        Returns:
            Dict with trajectory metrics
        """
        if len(historical_data) < 2:
            return {'trend': 0.0, 'volatility': 0.0, 'acceleration': 0.0}
        
        # Calculate brittleness for each period
        brittleness_series = []
        for agency_scores, gdp in historical_data:
            b_score = self.calculate_systemic_brittleness(
                agency_scores, gdp, return_details=False
            )
            brittleness_series.append(b_score)
        
        brittleness_array = np.array(brittleness_series)
        
        # Calculate metrics
        # 1. Trend (linear regression slope)
        x = np.arange(len(brittleness_array))
        coeffs = np.polyfit(x[-window:], brittleness_array[-window:], 1)
        trend = coeffs[0]
        
        # 2. Volatility (rolling std)
        if len(brittleness_array) >= window:
            volatility = np.std(brittleness_array[-window:])
        else:
            volatility = np.std(brittleness_array)
        
        # 3. Acceleration (change in trend)
        if len(brittleness_array) >= 2 * window:
            # Compare recent trend to previous trend
            prev_coeffs = np.polyfit(
                x[-2*window:-window], 
                brittleness_array[-2*window:-window], 
                1
            )
            acceleration = trend - prev_coeffs[0]
        else:
            acceleration = 0.0
        
        return {
            'current': brittleness_array[-1],
            'trend': trend,
            'volatility': volatility,
            'acceleration': acceleration,
            'trajectory': 'DETERIORATING' if trend > 0.1 else 'STABLE' if abs(trend) <= 0.1 else 'IMPROVING'
        }


# Integration with main system
def create_brittleness_engine(ideology: str = "framework_average") -> BrittlenessEngine:
    """Factory function to create a configured BrittlenessEngine."""
    calculator = AgencyCalculator(ideology=ideology)
    engine = BrittlenessEngine(calculator)
    return engine


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Enhanced BrittlenessEngine ===\n")
    
    # Create engines with different ideologies
    engines = {
        ideology: create_brittleness_engine(ideology)
        for ideology in ["framework_average", "libertarian", "socialist"]
    }
    
    # Test cases
    test_cases = [
        {
            "name": "Stable Developed Country",
            "scores": {'economic': 0.9, 'political': 0.95, 'social': 0.88, 
                      'health': 0.92, 'educational': 0.93},
            "gdp": 600e9  # $600B
        },
        {
            "name": "Pre-Crisis Country",
            "scores": {'economic': 0.65, 'political': 0.45, 'social': 0.52,
                      'health': 0.71, 'educational': 0.68},
            "gdp": 150e9  # $150B
        },
        {
            "name": "Fragile State",
            "scores": {'economic': 0.35, 'political': 0.28, 'social': 0.42,
                      'health': 0.36, 'educational': 0.34},
            "gdp": 20e9   # $20B
        }
    ]
    
    # Test each case under different ideologies
    for test_case in test_cases:
        print(f"\n{test_case['name']} (GDP: ${test_case['gdp']/1e9:.0f}B)")
        print("-" * 50)
        
        for ideology, engine in engines.items():
            result = engine.calculate_systemic_brittleness(
                test_case['scores'],
                test_case['gdp']
            )
            
            print(f"\n{ideology.upper()}:")
            print(f"  Brittleness Score: {result.brittleness_score}/10")
            print(f"  Risk Level: {result.risk_level}")
            print(f"  Percentile: {result.percentile}%")
            print(f"  Total Agency: {result.total_agency:.3f}")
            
            if result.warnings:
                print(f"  Warnings: {', '.join(result.warnings)}")
    
    # Test trajectory calculation
    print("\n\nTrajectory Analysis (Deteriorating Country):")
    print("-" * 50)
    
    # Simulate 10 years of decline
    historical = []
    base_scores = {'economic': 0.7, 'political': 0.7, 'social': 0.7,
                   'health': 0.7, 'educational': 0.7}
    gdp = 100e9
    
    for year in range(10):
        # Gradual decline
        scores = {k: v - (year * 0.02) for k, v in base_scores.items()}
        historical.append((scores, gdp))
    
    engine = engines["framework_average"]
    trajectory = engine.calculate_brittleness_trajectory(historical)
    
    print(f"Current Brittleness: {trajectory['current']:.2f}")
    print(f"Trend: {trajectory['trend']:+.3f} per period")
    print(f"Volatility: {trajectory['volatility']:.3f}")
    print(f"Acceleration: {trajectory['acceleration']:+.3f}")
    print(f"Trajectory: {trajectory['trajectory']}")