# api/main_integrated.py
"""
Updated main.py with real database integration
This shows how to replace the mock functions with real database calls
"""

# Add this to your imports in main.py
from api.database import (
    get_db, 
    get_timeseries_data as db_get_timeseries,
    get_latest_agency_scores,
    save_brittleness_prediction,
    get_iqa_notes as db_get_iqa_notes,
    save_iqa_note as db_save_iqa_note,
    Observation, AgencyScore, BrittlenessPrediction
)

# Replace the mock query functions with these:

def query_db_for_timeseries(country_code: str, start_year: Optional[int], end_year: Optional[int]) -> List[Dict]:
    """Query real database for timeseries data."""
    logger.info(f"Querying database for {country_code} from {start_year} to {end_year}")
    
    try:
        # Get all data for the country
        data = db_get_timeseries(
            country_code=country_code,
            start_year=start_year,
            end_year=end_year
        )
        
        # Also get calculated agency scores
        with get_db() as db:
            query = db.query(AgencyScore).filter(
                AgencyScore.country_code == country_code
            )
            
            if start_year:
                query = query.filter(AgencyScore.year >= start_year)
            if end_year:
                query = query.filter(AgencyScore.year <= end_year)
            
            agency_scores = query.all()
            
            # Add agency scores to data
            for score in agency_scores:
                data.extend([
                    {"indicator_code": "A_econ", "year": score.year, 
                     "value": float(score.economic_agency) if score.economic_agency else None},
                    {"indicator_code": "A_poli", "year": score.year, 
                     "value": float(score.political_agency) if score.political_agency else None},
                    {"indicator_code": "A_soc", "year": score.year, 
                     "value": float(score.social_agency) if score.social_agency else None},
                    {"indicator_code": "A_health", "year": score.year, 
                     "value": float(score.health_agency) if score.health_agency else None},
                    {"indicator_code": "A_edu", "year": score.year, 
                     "value": float(score.educational_agency) if score.educational_agency else None},
                ])
        
        return data
        
    except Exception as e:
        logger.error(f"Database query error: {e}")
        # Fallback to mock data if database fails
        return []


def query_db_for_iqa(country_code: str, year: Optional[int]) -> List[Dict[str, Any]]:
    """Query real database for IQA notes."""
    logger.info(f"Querying database for IQA notes: {country_code}, year: {year}")
    
    try:
        return db_get_iqa_notes(country_code, year)
    except Exception as e:
        logger.error(f"IQA query error: {e}")
        return []


def store_iqa_note(country_code: str, note_data: Dict) -> Dict:
    """Store IQA note in real database."""
    logger.info(f"Storing IQA note for {country_code}: {note_data}")
    
    try:
        db_save_iqa_note({
            "country_code": country_code,
            **note_data
        })
        return {"status": "success", "message": "IQA note stored"}
    except Exception as e:
        logger.error(f"IQA storage error: {e}")
        raise HTTPException(status_code=500, detail="Failed to store IQA note")


# Update the HybridForecaster to load real models and save predictions

class RealHybridForecaster:
    """Real forecaster that loads trained models and saves predictions."""
    
    def __init__(self, country_code: str):
        logger.info(f"Loading model for {country_code}...")
        self.country = country_code
        
        # Load the actual model
        model_path = f"models/{country_code.lower()}_brittleness_model.pkl"
        if os.path.exists(model_path):
            from ai.brittleness_predictor import BrittlenessPredictor
            self.model = BrittlenessPredictor(model_path)
        else:
            logger.warning(f"Model not found at {model_path}, using mock")
            self.model = None

    def predict(self, steps: int = 10, weighting: str = "Communitarian") -> List[Dict[str, Any]]:
        """Generate real predictions using the trained model."""
        
        if not self.model:
            # Fallback to mock if no model
            return MockHybridForecaster(self.country).predict(steps, weighting)
        
        try:
            # Get latest data for features
            latest_data = get_latest_agency_scores(self.country)
            if not latest_data:
                raise ValueError("No historical data available")
            
            # Create feature vector (simplified - you'd engineer full features)
            features = {
                'total_agency': latest_data.get('total_agency', 0.5),
                'economic': latest_data.get('economic_agency', 0.5),
                'political': latest_data.get('political_agency', 0.5),
                'social': latest_data.get('social_agency', 0.5),
                'health': latest_data.get('health_agency', 0.5),
                'educational': latest_data.get('educational_agency', 0.5),
                # Add more engineered features as needed
            }
            
            # Get prediction
            prediction = self.model.predict(features, self.country)
            
            # Save to database
            save_brittleness_prediction({
                'country_code': self.country,
                'prediction_date': datetime.now().date(),
                'target_year': latest_data['year'] + 1,
                'brittleness_score': prediction.brittleness_score,
                'confidence_lower': prediction.confidence_interval[0],
                'confidence_upper': prediction.confidence_interval[1],
                'risk_level': prediction.risk_level,
                'model_version': '1.0',
                'weighting_scheme': weighting
            })
            
            # Format response
            base_year = latest_data['year'] + 1
            data = []
            
            # For now, return single prediction
            # In production, you'd generate multi-step forecasts
            data.append({
                "indicator_code": "brittleness_score",
                "year": base_year,
                "value": round(prediction.brittleness_score, 2),
                "confidence_lower": round(prediction.confidence_interval[0], 2),
                "confidence_upper": round(prediction.confidence_interval[1], 2),
                "risk_level": prediction.risk_level
            })
            
            return data
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Fallback to mock
            return MockHybridForecaster(self.country).predict(steps, weighting)

    def explain(self, features: Dict) -> Dict:
        """Generate real SHAP explanations."""
        if not self.model:
            return MockHybridForecaster(self.country).explain(features)
        
        try:
            explanation = self.model.explain_prediction(features, plot=False)
            return {
                "base_value": explanation['base_value'],
                "shap_values": explanation['feature_impacts']
            }
        except Exception as e:
            logger.error(f"Explanation error: {e}")
            return MockHybridForecaster(self.country).explain(features)


# In your endpoints, you can now use RealHybridForecaster instead of MockHybridForecaster
# The rest of your API remains the same!