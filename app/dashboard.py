import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import yaml
import os

# Load config
def load_config(config_path='../ingestion/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
engine = create_engine(config['database']['url'])
Session = sessionmaker(bind=engine)

# Sidebar for selection
st.sidebar.title("Agency Calculus Dashboard")
country = st.sidebar.selectbox("Select Country", config['etl']['countries'])
indicator = st.sidebar.selectbox("Select Indicator", config['vdem']['indicators'])
forecast_steps = st.sidebar.slider("Forecast Steps", min_value=1, max_value=10, value=5)

# Main content
st.title(f"Agency Metrics for {country}")

# Fetch and display historical data
db = Session()
try:
    stmt = text("""
        SELECT year, value 
        FROM observations 
        WHERE country_code = :country AND indicator_code = :indicator
        ORDER BY year
    """)
    df = pd.read_sql(stmt, db.bind, params={'country': country, 'indicator': indicator})
finally:
    db.close()

if df.empty:
    st.error(f"No data available for {country} - {indicator}")
else:
    fig = px.line(df, x='year', y='value', title=f"Historical {indicator} for {country}")
    st.plotly_chart(fig)

    # Load model and forecast
    model_path = f"../models/{country}_hybrid_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        # Dummy exog for forecast (in real use, prepare future exog based on trends or inputs)
        exog_future = pd.DataFrame(0, index=range(forecast_steps), columns=['v2x_freexp_altinf'])  # Example; adjust to actual exog columns
        
        try:
            predictions = model.predict(steps=forecast_steps, exog=exog_future)
            if isinstance(predictions, pd.DataFrame) and indicator in predictions.columns:
                pred_df = pd.DataFrame({
                    'year': range(df['year'].max() + 1, df['year'].max() + forecast_steps + 1),
                    'value': predictions[indicator]
                })
                st.subheader(f"Forecast for Next {forecast_steps} Years")
                fig_pred = px.line(pred_df, x='year', y='value', title=f"Forecasted {indicator}")
                st.plotly_chart(fig_pred)
            else:
                st.warning("Forecast not available for this indicator.")
        except Exception as e:
            st.error(f"Forecast error: {str(e)}")
    else:
        st.warning("Trained model not found. Run train_models.py first.")