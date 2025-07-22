# etl/realtime_aggregator.py - Enhanced Version

"""
Component: etl/realtime_aggregator.py
Purpose: Extracts REAL-TIME data from various live sources (APIs, news feeds).
         Normalizes the data using pre-calculated historical statistics.
         Creates a daily "AgencySnapshot" for the live prediction engine.
Inputs: Live API responses, `normalization_stats.json`
Outputs: Standardized JSON snapshot for a given country and day.
Integration: Provides the live input vector for the trained brittleness_predictor.
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from collections import deque
import aioredis
import hashlib

# (Adapters would be in separate files)
# from etl.adapters.worldbank_adapter import WorldBankAdapter
# ... other adapters

@dataclass
class AgencySnapshot:
    """Represents agency state at a point in time"""
    timestamp: str
    country_code: str
    economic_agency: float
    political_agency: float
    social_agency: float
    health_agency: float
    educational_agency: float
    total_agency: float = 0.0
    
    # Add critical metrics for brittleness detection
    agency_delta: float = 0.0  # Change from previous snapshot
    volatility_7d: float = 0.0  # 7-day rolling volatility
    data_completeness: float = 1.0  # Percentage of successfully fetched metrics
    anomaly_flags: List[str] = None  # List of detected anomalies
    
    def __post_init__(self):
        if self.anomaly_flags is None:
            self.anomaly_flags = []

class RealTimeAggregator:
    def __init__(self, 
                 stats_path: str = "normalization_stats.json",
                 redis_url: str = "redis://localhost:6379",
                 cache_ttl: int = 300):  # 5 minute cache
        self.logger = logging.getLogger(__name__)
        self.norm_stats = self._load_stats(stats_path)
        self.cache_ttl = cache_ttl
        self.redis = None  # Initialize in async context
        
        # Historical buffer for volatility calculations
        self.history_buffer = {}  # country_code -> deque of recent snapshots
        self.max_history = 30  # Keep 30 days of history
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            'sudden_drop': 0.15,  # 15% drop in any agency score
            'high_volatility': 0.2,  # Volatility above 0.2
            'data_quality': 0.7,  # Less than 70% data completeness
        }
        
    async def initialize(self):
        """Async initialization for Redis connection"""
        try:
            self.redis = await aioredis.create_redis_pool(
                'redis://localhost:6379',
                encoding='utf-8'
            )
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.warning(f"Redis unavailable, using in-memory cache: {e}")
            
    def _load_stats(self, path: str) -> Dict:
        """Loads the pre-calculated normalization statistics."""
        try:
            with open(path, 'r') as f:
                stats = json.load(f)
                self.logger.info(f"Loaded normalization stats: {len(stats)} metrics")
                
                # Validate stats structure
                for metric, values in stats.items():
                    required_keys = ['p25', 'p75', 'mean', 'std']
                    if not all(k in values for k in required_keys):
                        self.logger.warning(f"Incomplete stats for {metric}")
                        
                return stats
        except Exception as e:
            self.logger.critical(f"FATAL: Could not load normalization stats: {e}")
            raise
            
    def _normalize_value(self, metric_name: str, value: float, invert: bool = False) -> float:
        """
        Normalizes a single value using pre-calculated historical stats.
        Enhanced with outlier detection and logging.
        """
        if metric_name not in self.norm_stats:
            self.logger.warning(f"No normalization stats for {metric_name}")
            return 0.5
            
        if value is None:
            return 0.5

        stats = self.norm_stats[metric_name]
        p25 = stats['p25']
        p75 = stats['p75']
        mean = stats.get('mean', (p25 + p75) / 2)
        std = stats.get('std', (p75 - p25) / 1.35)  # Approximate std from IQR
        
        # Check for outliers (beyond 3 standard deviations)
        if abs(value - mean) > 3 * std:
            self.logger.warning(f"Outlier detected for {metric_name}: {value} (mean: {mean}, std: {std})")
        
        # Robust scaling using IQR
        if (p75 - p25) > 0:
            score = (value - p25) / (p75 - p25)
        else:
            score = 0.5
            
        score = np.clip(score, 0, 1)
        
        return 1.0 - score if invert else score

    def _calculate_economic_agency(self, raw_data: Dict) -> Tuple[float, float]:
        """
        Calculate economic agency with data completeness tracking.
        Returns: (agency_score, completeness_ratio)
        """
        metrics = {
            'gdp': ('NY.GDP.PCAP.CD', 'gdp_per_capita_usd', False),
            'unemployment': ('SL.UEM.TOTL.ZS', 'unemployment_rate_percent', True),
            'gini': ('SI.POV.GINI', 'gini_coefficient', True),
            'inflation': ('FP.CPI.TOTL.ZG', 'inflation_rate_percent', True),
            'debt': ('GC.DOD.TOTL.GD.ZS', 'government_debt_percent_gdp', True)
        }
        
        scores = []
        available_count = 0
        
        for metric_key, (raw_key, norm_key, invert) in metrics.items():
            value = raw_data.get(raw_key)
            if value is not None:
                score = self._normalize_value(norm_key, value, invert=invert)
                scores.append(score)
                available_count += 1
            else:
                self.logger.debug(f"Missing economic metric: {metric_key}")
        
        completeness = available_count / len(metrics)
        agency_score = np.mean(scores) if scores else 0.5
        
        return agency_score, completeness

    def _calculate_political_agency(self, raw_data: Dict) -> Tuple[float, float]:
        """Calculate political agency from governance and freedom metrics."""
        metrics = {
            'democracy_index': ('democracy_index', 'democracy_index_score', False),
            'corruption': ('corruption_perception', 'corruption_perception_index', False),
            'press_freedom': ('press_freedom_index', 'press_freedom_rank', True),
            'rule_of_law': ('rule_of_law_index', 'rule_of_law_score', False)
        }
        
        scores = []
        available_count = 0
        
        for metric_key, (raw_key, norm_key, invert) in metrics.items():
            value = raw_data.get(raw_key)
            if value is not None:
                score = self._normalize_value(norm_key, value, invert=invert)
                scores.append(score)
                available_count += 1
        
        # Special handling for recent bills (if available)
        if 'restrictive_bills_count' in raw_data:
            bill_penalty = min(raw_data['restrictive_bills_count'] * 0.05, 0.3)
            scores.append(1.0 - bill_penalty)
            available_count += 1
        
        completeness = available_count / (len(metrics) + 1)
        agency_score = np.mean(scores) if scores else 0.5
        
        return agency_score, completeness

    def _calculate_social_agency(self, raw_data: Dict) -> Tuple[float, float]:
        """Calculate social agency from sentiment and cohesion metrics."""
        # Base metrics
        sentiment_score = 0.5
        polarization_score = 0.5
        protest_score = 0.5
        
        completeness_count = 0
        
        # Twitter sentiment
        if 'twitter_sentiment' in raw_data:
            # Convert -1 to 1 range to 0 to 1
            sentiment_score = (raw_data['twitter_sentiment'] + 1) / 2
            completeness_count += 1
        
        # Polarization index (higher = worse)
        if 'social_polarization_index' in raw_data:
            polarization_score = 1.0 - self._normalize_value(
                'polarization_index', 
                raw_data['social_polarization_index'], 
                invert=False
            )
            completeness_count += 1
        
        # Protest activity (normalized)
        if 'protest_events_monthly' in raw_data:
            protest_score = 1.0 - self._normalize_value(
                'protest_frequency',
                raw_data['protest_events_monthly'],
                invert=False
            )
            completeness_count += 1
        
        scores = [sentiment_score, polarization_score, protest_score]
        completeness = completeness_count / 3
        agency_score = np.mean(scores)
        
        return agency_score, completeness

    def _calculate_health_agency(self, raw_data: Dict) -> Tuple[float, float]:
        """Calculate health agency."""
        metrics = {
            'life_expectancy': ('SP.DYN.LE00.IN', 'life_expectancy_years', False),
            'health_expenditure': ('SH.XPD.CHEX.PC.CD', 'health_expenditure_per_capita', False),
            'hospital_beds': ('SH.MED.BEDS.ZS', 'hospital_beds_per_1000', False),
            'physician_density': ('SH.MED.PHYS.ZS', 'physicians_per_1000', False)
        }
        
        scores = []
        available_count = 0
        
        for metric_key, (raw_key, norm_key, invert) in metrics.items():
            value = raw_data.get(raw_key)
            if value is not None:
                score = self._normalize_value(norm_key, value, invert=invert)
                scores.append(score)
                available_count += 1
        
        completeness = available_count / len(metrics)
        agency_score = np.mean(scores) if scores else 0.5
        
        return agency_score, completeness

    def _calculate_educational_agency(self, raw_data: Dict) -> Tuple[float, float]:
        """Calculate educational agency."""
        metrics = {
            'literacy': ('SE.ADT.LITR.ZS', 'adult_literacy_rate', False),
            'enrollment': ('SE.TER.ENRR', 'tertiary_enrollment_rate', False),
            'expenditure': ('SE.XPD.TOTL.GD.ZS', 'education_expenditure_percent_gdp', False),
            'completion': ('SE.PRM.CMPT.ZS', 'primary_completion_rate', False)
        }
        
        scores = []
        available_count = 0
        
        for metric_key, (raw_key, norm_key, invert) in metrics.items():
            value = raw_data.get(raw_key)
            if value is not None:
                score = self._normalize_value(norm_key, value, invert=invert)
                scores.append(score)
                available_count += 1
        
        completeness = available_count / len(metrics)
        agency_score = np.mean(scores) if scores else 0.5
        
        return agency_score, completeness

    def _detect_anomalies(self, 
                         current: AgencySnapshot, 
                         previous: Optional[AgencySnapshot]) -> List[str]:
        """Detect anomalies in the current snapshot."""
        anomalies = []
        
        # Check data completeness
        if current.data_completeness < self.anomaly_thresholds['data_quality']:
            anomalies.append(f"LOW_DATA_QUALITY:{current.data_completeness:.2f}")
        
        # Check for sudden drops (if we have previous data)
        if previous:
            for domain in ['economic', 'political', 'social', 'health', 'educational']:
                current_val = getattr(current, f"{domain}_agency")
                previous_val = getattr(previous, f"{domain}_agency")
                
                if previous_val > 0:
                    drop_rate = (previous_val - current_val) / previous_val
                    if drop_rate > self.anomaly_thresholds['sudden_drop']:
                        anomalies.append(f"SUDDEN_DROP:{domain}:{drop_rate:.2%}")
        
        # Check volatility
        if current.volatility_7d > self.anomaly_thresholds['high_volatility']:
            anomalies.append(f"HIGH_VOLATILITY:{current.volatility_7d:.3f}")
        
        # Check for critically low agency scores
        if current.total_agency < 0.3:
            anomalies.append(f"CRITICAL_AGENCY:{current.total_agency:.3f}")
        
        return anomalies

    def _calculate_volatility(self, history: List[AgencySnapshot]) -> float:
        """Calculate 7-day rolling volatility of total agency."""
        if len(history) < 7:
            return 0.0
        
        recent_scores = [s.total_agency for s in history[-7:]]
        return float(np.std(recent_scores))

    async def _get_cached_data(self, cache_key: str) -> Optional[Dict]:
        """Get data from Redis cache."""
        if not self.redis:
            return None
        
        try:
            data = await self.redis.get(cache_key)
            if data:
                return json.loads(data)
        except Exception as e:
            self.logger.error(f"Cache read error: {e}")
        
        return None

    async def _set_cached_data(self, cache_key: str, data: Dict):
        """Set data in Redis cache."""
        if not self.redis:
            return
        
        try:
            await self.redis.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(data)
            )
        except Exception as e:
            self.logger.error(f"Cache write error: {e}")

    async def _fetch_raw_data(self, country_code: str) -> Dict:
        """
        Fetch raw data from all sources with caching.
        In production, this would call actual API adapters.
        """
        cache_key = f"raw_data:{country_code}:{datetime.now().strftime('%Y%m%d%H')}"
        
        # Check cache first
        cached_data = await self._get_cached_data(cache_key)
        if cached_data:
            self.logger.info(f"Using cached data for {country_code}")
            return cached_data
        
        # Simulate fetching from multiple sources
        # In production, these would be real API calls
        raw_data = {
            # Economic indicators
            'NY.GDP.PCAP.CD': 800.0 if country_code == 'HT' else 65000.0,
            'SL.UEM.TOTL.ZS': 14.5 if country_code == 'HT' else 3.5,
            'SI.POV.GINI': 60.1 if country_code == 'HT' else 35.0,
            'FP.CPI.TOTL.ZG': 20.0 if country_code == 'HT' else 2.5,
            'GC.DOD.TOTL.GD.ZS': 30.0 if country_code == 'HT' else 80.0,
            
            # Political indicators
            'democracy_index': 4.2 if country_code == 'HT' else 7.8,
            'corruption_perception': 18 if country_code == 'HT' else 65,
            'press_freedom_index': 85 if country_code == 'HT' else 25,
            'rule_of_law_index': 0.3 if country_code == 'HT' else 0.75,
            
            # Social indicators
            'twitter_sentiment': -0.4 if country_code == 'HT' else 0.1,
            'social_polarization_index': 0.8 if country_code == 'HT' else 0.3,
            'protest_events_monthly': 15 if country_code == 'HT' else 2,
            
            # Health indicators
            'SP.DYN.LE00.IN': 64.0 if country_code == 'HT' else 78.0,
            'SH.XPD.CHEX.PC.CD': 70.0 if country_code == 'HT' else 5000.0,
            'SH.MED.BEDS.ZS': 0.7 if country_code == 'HT' else 3.5,
            
            # Educational indicators
            'SE.ADT.LITR.ZS': 61.7 if country_code == 'HT' else 99.0,
            'SE.TER.ENRR': 15.0 if country_code == 'HT' else 65.0,
            'SE.XPD.TOTL.GD.ZS': 2.0 if country_code == 'HT' else 5.0,
        }
        
        # Cache the data
        await self._set_cached_data(cache_key, raw_data)
        
        return raw_data

    async def get_latest_snapshot(self, country_code: str) -> Optional[AgencySnapshot]:
        """
        Main function to get the latest agency snapshot for a country.
        Enhanced with anomaly detection and historical context.
        """
        self.logger.info(f"Generating real-time snapshot for {country_code}")
        
        try:
            # 1. Fetch raw data
            raw_data = await self._fetch_raw_data(country_code)
            
            # 2. Calculate agency scores with completeness tracking
            economic, econ_complete = self._calculate_economic_agency(raw_data)
            political, poli_complete = self._calculate_political_agency(raw_data)
            social, soc_complete = self._calculate_social_agency(raw_data)
            health, health_complete = self._calculate_health_agency(raw_data)
            educational, edu_complete = self._calculate_educational_agency(raw_data)
            
            # 3. Calculate overall data completeness
            completeness_scores = [econ_complete, poli_complete, soc_complete, 
                                 health_complete, edu_complete]
            data_completeness = np.mean(completeness_scores)
            
            # 4. Create snapshot
            snapshot = AgencySnapshot(
                timestamp=datetime.now().isoformat(),
                country_code=country_code,
                economic_agency=economic,
                political_agency=political,
                social_agency=social,
                health_agency=health,
                educational_agency=educational,
                data_completeness=data_completeness
            )
            
            # 5. Calculate total agency with weights
            weights = {
                'economic': 0.25,
                'political': 0.20,
                'social': 0.20,
                'health': 0.20,
                'educational': 0.15
            }
            
            snapshot.total_agency = sum(
                getattr(snapshot, f"{domain}_agency") * weight
                for domain, weight in weights.items()
            )
            
            # 6. Get historical context
            if country_code not in self.history_buffer:
                self.history_buffer[country_code] = deque(maxlen=self.max_history)
            
            history = list(self.history_buffer[country_code])
            
            # 7. Calculate delta and volatility
            if history:
                previous = history[-1]
                snapshot.agency_delta = snapshot.total_agency - previous.total_agency
                snapshot.volatility_7d = self._calculate_volatility(history + [snapshot])
            
            # 8. Detect anomalies
            previous = history[-1] if history else None
            snapshot.anomaly_flags = self._detect_anomalies(snapshot, previous)
            
            # 9. Update history
            self.history_buffer[country_code].append(snapshot)
            
            # 10. Log critical information
            if snapshot.anomaly_flags:
                self.logger.warning(
                    f"Anomalies detected for {country_code}: {snapshot.anomaly_flags}"
                )
            
            self.logger.info(
                f"Snapshot generated for {country_code}: "
                f"Total Agency={snapshot.total_agency:.3f}, "
                f"Delta={snapshot.agency_delta:+.3f}, "
                f"Volatility={snapshot.volatility_7d:.3f}, "
                f"Completeness={snapshot.data_completeness:.1%}"
            )
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to generate snapshot for {country_code}: {e}")
            return None

    async def get_multiple_snapshots(self, country_codes: List[str]) -> Dict[str, AgencySnapshot]:
        """Get snapshots for multiple countries in parallel."""
        tasks = [self.get_latest_snapshot(code) for code in country_codes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        snapshots = {}
        for code, result in zip(country_codes, results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to get snapshot for {code}: {result}")
            elif result:
                snapshots[code] = result
        
        return snapshots

    def get_historical_snapshots(self, country_code: str, days: int = 7) -> List[AgencySnapshot]:
        """Get historical snapshots from buffer."""
        if country_code not in self.history_buffer:
            return []
        
        history = list(self.history_buffer[country_code])
        return history[-days:] if len(history) >= days else history


# --- Example Usage ---
async def main():
    # Create comprehensive normalization stats
    norm_stats = {
        # Economic metrics
        "gdp_per_capita_usd": {"p25": 1000, "p75": 45000, "mean": 15000, "std": 18000},
        "unemployment_rate_percent": {"p25": 3.0, "p75": 10.0, "mean": 6.5, "std": 3.2},
        "gini_coefficient": {"p25": 30, "p75": 50, "mean": 40, "std": 8},
        "inflation_rate_percent": {"p25": 1.0, "p75": 5.0, "mean": 3.0, "std": 2.5},
        "government_debt_percent_gdp": {"p25": 30, "p75": 80, "mean": 55, "std": 25},
        
        # Political metrics
        "democracy_index_score": {"p25": 3.0, "p75": 8.0, "mean": 5.5, "std": 2.0},
        "corruption_perception_index": {"p25": 30, "p75": 70, "mean": 50, "std": 15},
        "press_freedom_rank": {"p25": 20, "p75": 120, "mean": 70, "std": 40},
        "rule_of_law_score": {"p25": 0.3, "p75": 0.8, "mean": 0.55, "std": 0.2},
        
        # Social metrics
        "polarization_index": {"p25": 0.2, "p75": 0.6, "mean": 0.4, "std": 0.15},
        "protest_frequency": {"p25": 1, "p75": 10, "mean": 4, "std": 3},
        
        # Health metrics
        "life_expectancy_years": {"p25": 65, "p75": 80, "mean": 72, "std": 6},
        "health_expenditure_per_capita": {"p25": 200, "p75": 3000, "mean": 1200, "std": 1000},
        "hospital_beds_per_1000": {"p25": 1.0, "p75": 5.0, "mean": 3.0, "std": 1.5},
        "physicians_per_1000": {"p25": 0.5, "p75": 3.5, "mean": 2.0, "std": 1.2},
        
        # Educational metrics
        "adult_literacy_rate": {"p25": 70, "p75": 98, "mean": 85, "std": 12},
        "tertiary_enrollment_rate": {"p25": 20, "p75": 70, "mean": 45, "std": 20},
        "education_expenditure_percent_gdp": {"p25": 3.0, "p75": 6.0, "mean": 4.5, "std": 1.2},
        "primary_completion_rate": {"p25": 80, "p75": 100, "mean": 90, "std": 8}
    }
    
    with open("normalization_stats.json", 'w') as f:
        json.dump(norm_stats, f, indent=2)

    # Initialize aggregator
    aggregator = RealTimeAggregator(stats_path="normalization_stats.json")
    await aggregator.initialize()
    
    # Test with Haiti and comparison country
    print("\n=== REAL-TIME AGENCY MONITOR ===")
    
    # Get single snapshot for Haiti
    haiti_snapshot = await aggregator.get_latest_snapshot('HT')
    if haiti_snapshot:
        print(f"\n--- Haiti Current Status ---")
        print(f"Total Agency: {haiti_snapshot.total_agency:.3f}")
        print(f"Volatility (7d): {haiti_snapshot.volatility_7d:.3f}")
        print(f"Data Quality: {haiti_snapshot.data_completeness:.1%}")
        if haiti_snapshot.anomaly_flags:
            print(f"⚠️  ANOMALIES: {', '.join(haiti_snapshot.anomaly_flags)}")
        
        print(f"\nDomain Breakdown:")
        for domain in ['economic', 'political', 'social', 'health', 'educational']:
            score = getattr(haiti_snapshot, f"{domain}_agency")
            print(f"  {domain.capitalize()}: {score:.3f}")
    
    # Simulate historical data for volatility calculation
    print("\n--- Simulating 7-day history ---")
    for i in range(7):
        await aggregator.get_latest_snapshot('HT')
        await asyncio.sleep(0.1)  # Small delay to simulate time passing
    
    # Get final snapshot with volatility
    final_snapshot = await aggregator.get_latest_snapshot('HT')
    if final_snapshot:
        print(f"\nAfter 7 days of monitoring:")
        print(f"Current Agency: {final_snapshot.total_agency:.3f}")
        print(f"7-day Volatility: {final_snapshot.volatility_7d:.3f}")
        print(f"Latest Delta: {final_snapshot.agency_delta:+.3f}")
    
    # Compare multiple countries
    print("\n--- Multi-Country Comparison ---")
    countries = ['HT', 'US', 'CL']  # Haiti, USA, Chile
    snapshots = await aggregator.get_multiple_snapshots(countries)
    
    for code, snapshot in snapshots.items():
        print(f"\n{code}: Total Agency = {snapshot.total_agency:.3f}, "
              f"Anomalies = {len(snapshot.anomaly_flags)}")
    
    # Export snapshot for downstream processing
    if haiti_snapshot:
        output_data = {
            "snapshot": asdict(haiti_snapshot),
            "metadata": {
                "model_version": "4.3",
                "normalization_method": "IQR",
                "generated_at": datetime.now().isoformat()
            }
        }
        
        with open("data/realtime/latest_haiti_snapshot.json", 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Snapshot exported to data/realtime/latest_haiti_snapshot.json")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())