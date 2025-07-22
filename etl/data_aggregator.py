"""
Component: etl/data_aggregator.py
Purpose: Extract data from primary sources, transform into unified time-series format,
         calculate required indices, and load clean JSON for AI layers
Inputs: Various API responses, CSV files, unstructured text
Outputs: Standardized JSON with agency scores, power metrics, and metadata
Integration: Feeds directly into ai/training/feature_engineering.py
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

# Import adapters (these would be implemented separately)
from etl.adapters.worldbank_adapter import WorldBankAdapter
from etl.adapters.twitter_adapter import TwitterAdapter
from etl.adapters.news_adapter import NewsAdapter
from etl.adapters.bills_adapter import BillsAdapter


@dataclass
class AgencySnapshot:
    """Represents agency state at a point in time"""
    timestamp: str
    country_code: str
    country_name: str
    
    # Core agency scores (0-1 scale)
    economic_agency: float
    political_agency: float
    social_agency: float
    health_agency: float
    educational_agency: float
    
    # Power metrics for disparity calculation
    power_metrics: Dict[str, float]
    
    # Metadata
    data_quality: float  # 0-1, percentage of available data
    sources: List[str]
    events: List[str]  # Notable events for context
    
    # Calculated indices
    total_agency: float = 0.0
    agency_volatility: float = 0.0
    polarization_index: float = 0.0
    
    def __post_init__(self):
        """Calculate composite indices"""
        # Default weights (can be overridden by config)
        weights = {
            'economic': 0.25,
            'political': 0.20,
            'social': 0.20,
            'health': 0.20,
            'educational': 0.15
        }
        
        self.total_agency = (
            self.economic_agency * weights['economic'] +
            self.political_agency * weights['political'] +
            self.social_agency * weights['social'] +
            self.health_agency * weights['health'] +
            self.educational_agency * weights['educational']
        )


class DataAggregator:
    """Main ETL aggregator for Agency Monitor"""
    
    def __init__(self, config_path: str = "config/data_sources.json"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize adapters
        self.worldbank = WorldBankAdapter(self.config.get('worldbank', {}))
        self.twitter = TwitterAdapter(self.config.get('twitter', {}))
        self.news = NewsAdapter(self.config.get('news', {}))
        self.bills = BillsAdapter(self.config.get('bills', {}))
        
        # Thread pool for parallel API calls
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def _load_config(self, path: str) -> Dict:
        """Load configuration from JSON"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    async def aggregate_country_data(
        self, 
        country_code: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = 'daily'
    ) -> List[Dict[str, Any]]:
        """
        Main aggregation function for a single country
        Returns time series of agency snapshots
        """
        self.logger.info(f"Aggregating data for {country_code} from {start_date} to {end_date}")
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        snapshots = []
        
        # Fetch all data sources in parallel
        economic_data = await self._fetch_economic_data(country_code, start_date, end_date)
        political_data = await self._fetch_political_data(country_code, start_date, end_date)
        social_data = await self._fetch_social_data(country_code, start_date, end_date)
        health_data = await self._fetch_health_data(country_code, start_date, end_date)
        education_data = await self._fetch_education_data(country_code, start_date, end_date)
        
        # Process each date
        for date in dates:
            snapshot = self._create_snapshot(
                country_code=country_code,
                date=date,
                economic_data=economic_data,
                political_data=political_data,
                social_data=social_data,
                health_data=health_data,
                education_data=education_data
            )
            
            if snapshot:
                snapshots.append(asdict(snapshot))
        
        # Calculate volatility and trends
        snapshots = self._calculate_derived_metrics(snapshots)
        
        return snapshots
    
    async def _fetch_economic_data(
        self, 
        country_code: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch economic indicators from World Bank and other sources"""
        indicators = [
            'NY.GDP.PCAP.CD',  # GDP per capita
            'FP.CPI.TOTL.ZG',  # Inflation
            'SL.UEM.TOTL.ZS',  # Unemployment
            'SI.POV.GINI',     # GINI index
            'GC.DOD.TOTL.GD.ZS' # Government debt
        ]
        
        # Fetch from World Bank API
        wb_data = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.worldbank.fetch_indicators,
            country_code,
            indicators,
            start_date.year,
            end_date.year
        )
        
        return wb_data
    
    async def _fetch_political_data(
        self, 
        country_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Fetch political indicators including bills and governance metrics"""
        # Democracy indices, corruption perception, bills passed
        political_data = {}
        
        # Fetch recent bills
        bills = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.bills.fetch_recent_bills,
            country_code,
            start_date,
            end_date
        )
        
        political_data['bills'] = bills
        political_data['democracy_index'] = 0.7  # Placeholder - would fetch from V-Dem or similar
        
        return political_data
    
    async def _fetch_social_data(
        self,
        country_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Fetch social indicators including sentiment and polarization"""
        # Social media sentiment, news sentiment, protest data
        tasks = [
            self._fetch_twitter_sentiment(country_code, start_date, end_date),
            self._fetch_news_sentiment(country_code, start_date, end_date)
        ]
        
        sentiment_data = await asyncio.gather(*tasks)
        
        return {
            'twitter_sentiment': sentiment_data[0],
            'news_sentiment': sentiment_data[1],
            'social_cohesion_index': self._calculate_cohesion(sentiment_data)
        }
    
    async def _fetch_health_data(
        self,
        country_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch health indicators"""
        indicators = [
            'SP.DYN.LE00.IN',  # Life expectancy
            'SH.XPD.CHEX.PC.CD', # Health expenditure per capita
            'SH.MED.BEDS.ZS'   # Hospital beds per 1000
        ]
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.worldbank.fetch_indicators,
            country_code,
            indicators,
            start_date.year,
            end_date.year
        )
    
    async def _fetch_education_data(
        self,
        country_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch education indicators"""
        indicators = [
            'SE.ADT.LITR.ZS',  # Literacy rate
            'SE.XPD.TOTL.GD.ZS', # Education expenditure
            'SE.TER.ENRR'      # Tertiary enrollment
        ]
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.worldbank.fetch_indicators,
            country_code,
            indicators,
            start_date.year,
            end_date.year
        )
    
    def _create_snapshot(
        self,
        country_code: str,
        date: datetime,
        **data_sources
    ) -> Optional[AgencySnapshot]:
        """Create a single agency snapshot from various data sources"""
        try:
            # Calculate agency scores from raw data
            economic_agency = self._calculate_economic_agency(
                data_sources['economic_data'], date
            )
            political_agency = self._calculate_political_agency(
                data_sources['political_data'], date
            )
            social_agency = self._calculate_social_agency(
                data_sources['social_data'], date
            )
            health_agency = self._calculate_health_agency(
                data_sources['health_data'], date
            )
            educational_agency = self._calculate_educational_agency(
                data_sources['education_data'], date
            )
            
            # Calculate power metrics
            power_metrics = self._calculate_power_metrics(
                country_code, date, **data_sources
            )
            
            # Identify notable events
            events = self._extract_events(data_sources, date)
            
            # Calculate data quality
            data_quality = self._calculate_data_quality(data_sources)
            
            return AgencySnapshot(
                timestamp=date.isoformat(),
                country_code=country_code,
                country_name=self._get_country_name(country_code),
                economic_agency=economic_agency,
                political_agency=political_agency,
                social_agency=social_agency,
                health_agency=health_agency,
                educational_agency=educational_agency,
                power_metrics=power_metrics,
                data_quality=data_quality,
                sources=list(data_sources.keys()),
                events=events
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create snapshot for {date}: {e}")
            return None
    
    def _calculate_economic_agency(self, data: pd.DataFrame, date: datetime) -> float:
        """Calculate economic agency score from indicators"""
        # Normalize and weight economic indicators
        # Higher GDP per capita, lower unemployment, lower inequality = higher agency
        if data.empty:
            return 0.5  # Default neutral score
        
        # Example calculation (would be more sophisticated)
        gdp_score = min(data.get('NY.GDP.PCAP.CD', 0) / 50000, 1.0)
        unemployment_score = 1.0 - min(data.get('SL.UEM.TOTL.ZS', 0) / 100, 1.0)
        gini_score = 1.0 - min(data.get('SI.POV.GINI', 0) / 100, 1.0)
        
        return np.mean([gdp_score, unemployment_score, gini_score])
    
    def _calculate_political_agency(self, data: Dict, date: datetime) -> float:
        """Calculate political agency from governance metrics"""
        # Democracy index, corruption levels, bill restrictiveness
        democracy_score = data.get('democracy_index', 0.5)
        
        # Analyze bills for agency-reducing content
        restrictive_bills = 0
        total_bills = len(data.get('bills', []))
        if total_bills > 0:
            # This would use NLP to analyze bill content
            restrictive_ratio = restrictive_bills / total_bills
            bill_score = 1.0 - restrictive_ratio
        else:
            bill_score = 0.7  # Neutral if no bills
        
        return np.mean([democracy_score, bill_score])
    
    def _calculate_social_agency(self, data: Dict, date: datetime) -> float:
        """Calculate social agency from sentiment and cohesion"""
        twitter_sentiment = data.get('twitter_sentiment', {}).get('average', 0)
        news_sentiment = data.get('news_sentiment', {}).get('average', 0)
        cohesion = data.get('social_cohesion_index', 0.5)
        
        # Convert sentiment (-1 to 1) to agency score (0 to 1)
        sentiment_score = (np.mean([twitter_sentiment, news_sentiment]) + 1) / 2
        
        return np.mean([sentiment_score, cohesion])
    
    def _calculate_health_agency(self, data: pd.DataFrame, date: datetime) -> float:
        """Calculate health agency from health indicators"""
        if data.empty:
            return 0.5
        
        # Normalize health indicators
        life_expectancy_score = min(data.get('SP.DYN.LE00.IN', 0) / 85, 1.0)
        health_spending_score = min(data.get('SH.XPD.CHEX.PC.CD', 0) / 5000, 1.0)
        
        return np.mean([life_expectancy_score, health_spending_score])
    
    def _calculate_educational_agency(self, data: pd.DataFrame, date: datetime) -> float:
        """Calculate educational agency from education indicators"""
        if data.empty:
            return 0.5
        
        literacy_score = data.get('SE.ADT.LITR.ZS', 0) / 100
        enrollment_score = min(data.get('SE.TER.ENRR', 0) / 100, 1.0)
        
        return np.mean([literacy_score, enrollment_score])
    
    def _calculate_power_metrics(
        self, 
        country_code: str,
        date: datetime,
        **data_sources
    ) -> Dict[str, float]:
        """Calculate power distribution metrics"""
        return {
            'government': 0.7,  # Placeholder - would calculate from actual data
            'opposition': 0.3,
            'military': 0.5,
            'economic_elite': 0.8,
            'civil_society': 0.4
        }
    
    def _calculate_derived_metrics(self, snapshots: List[Dict]) -> List[Dict]:
        """Calculate volatility, trends, and other derived metrics"""
        if len(snapshots) < 2:
            return snapshots
        
        df = pd.DataFrame(snapshots)
        
        # Calculate rolling volatility
        for agency_type in ['economic', 'political', 'social', 'health', 'educational']:
            col = f'{agency_type}_agency'
            df[f'{agency_type}_volatility'] = df[col].rolling(window=7).std()
        
        # Calculate overall volatility
        df['agency_volatility'] = df[[
            'economic_volatility', 'political_volatility', 'social_volatility',
            'health_volatility', 'educational_volatility'
        ]].mean(axis=1)
        
        # Calculate polarization index (simplified)
        df['polarization_index'] = (
            df['political_agency'].rolling(window=7).std() * 2 +
            df['social_agency'].rolling(window=7).std()
        ) / 3
        
        return df.fillna(0).to_dict('records')
    
    async def _fetch_twitter_sentiment(
        self,
        country_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Fetch Twitter sentiment data"""
        # Placeholder implementation
        return {
            'average': 0.1,
            'volume': 10000,
            'volatility': 0.3
        }
    
    async def _fetch_news_sentiment(
        self,
        country_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Fetch news sentiment data"""
        # Placeholder implementation
        return {
            'average': -0.2,
            'volume': 500,
            'sources': ['Reuters', 'AP', 'Local']
        }
    
    def _calculate_cohesion(self, sentiment_data: List) -> float:
        """Calculate social cohesion from various sentiment sources"""
        # Low variance in sentiment = high cohesion
        sentiments = [s.get('average', 0) for s in sentiment_data if isinstance(s, dict)]
        if sentiments:
            variance = np.var(sentiments)
            return 1.0 - min(variance * 2, 1.0)  # Scale variance to 0-1
        return 0.5
    
    def _extract_events(self, data_sources: Dict, date: datetime) -> List[str]:
        """Extract notable events from various data sources"""
        events = []
        
        # Check for significant bills
        bills = data_sources.get('political_data', {}).get('bills', [])
        for bill in bills:
            if bill.get('significance', 0) > 0.7:
                events.append(f"Bill: {bill.get('title', 'Unknown')}")
        
        # Check for social unrest indicators
        social_data = data_sources.get('social_data', {})
        if social_data.get('twitter_sentiment', {}).get('average', 0) < -0.5:
            events.append("High social media negativity")
        
        return events[:5]  # Limit to top 5 events
    
    def _calculate_data_quality(self, data_sources: Dict) -> float:
        """Calculate percentage of available data"""
        total_sources = len(data_sources)
        available_sources = sum(
            1 for source in data_sources.values()
            if source is not None and (
                (isinstance(source, pd.DataFrame) and not source.empty) or
                (isinstance(source, dict) and source) or
                (isinstance(source, list) and source)
            )
        )
        
        return available_sources / total_sources if total_sources > 0 else 0.0
    
    def _get_country_name(self, country_code: str) -> str:
        """Get country name from code"""
        # Would use a proper country code library
        country_map = {
            'US': 'United States',
            'HT': 'Haiti',
            'CL': 'Chile',
            'IR': 'Iran'
        }
        return country_map.get(country_code, country_code)
    
    async def aggregate_multiple_countries(
        self,
        country_codes: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, List[Dict]]:
        """Aggregate data for multiple countries in parallel"""
        tasks = [
            self.aggregate_country_data(code, start_date, end_date)
            for code in country_codes
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            code: result 
            for code, result in zip(country_codes, results)
        }
    
    def save_to_json(self, data: Dict, output_path: str):
        """Save aggregated data to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved aggregated data to {output_path}")


# Example usage
async def main():
    aggregator = DataAggregator()
    
    # Test with Haiti data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    haiti_data = await aggregator.aggregate_country_data(
        'HT', 
        start_date, 
        end_date
    )
    
    # Save for AI processing
    aggregator.save_to_json(
        {'HT': haiti_data},
        'data/processed/haiti_2024.json'
    )
    
    # Print sample output
    if haiti_data:
        print(f"Aggregated {len(haiti_data)} snapshots for Haiti")
        print(f"First snapshot: {json.dumps(haiti_data[0], indent=2)}")
        print(f"Average brittleness indicators:")
        print(f"  Total Agency: {np.mean([s['total_agency'] for s in haiti_data]):.3f}")
        print(f"  Volatility: {np.mean([s['agency_volatility'] for s in haiti_data]):.3f}")
        print(f"  Polarization: {np.mean([s['polarization_index'] for s in haiti_data]):.3f}")


if __name__ == "__main__":
    asyncio.run(main())