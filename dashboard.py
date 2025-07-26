"""
Streamlit Dashboard for the Agency Monitor Project - Production Version
This dashboard provides a user-friendly interface to visualize historical data, 
AI-powered forecasts, and model explanations by interacting with the live 
FastAPI backend.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="Agency Monitor", layout="wide", initial_sidebar_state="expanded")

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")
try:
    API_KEY = st.secrets["API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("API key not found. Please create a .streamlit/secrets.toml file with an API_KEY.")
    st.stop()

# --- API Interaction Layer ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def api_request(method: str, endpoint: str, **kwargs) -> Any:
    """Generic function for making API requests with retry logic."""
    headers = {"X-API-Key": API_KEY}
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, params=kwargs.get("params"), timeout=15)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=kwargs.get("json"), timeout=15)
        else:
            raise ValueError("Unsupported HTTP method")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error ({e.response.status_code}): {e.response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Network Error: Could not connect to the API at {API_BASE_URL}. Is it running?")
    return None

@st.cache_data(ttl=600)
def get_forecast_data(country_code: str, weighting: str) -> pd.DataFrame:
    """Fetches and processes forecast data from the API."""
    response = api_request('GET', f"country/{country_code}/forecast", params={"weighting_scheme": weighting, "steps": 10})
    if not response or 'forecast' not in response:
        return pd.DataFrame()
    
    # Process the forecast data
    records = []
    for item in response['forecast']:
        year = item['year']
        # Add agency scores
        for domain, score in item['agency_scores'].items():
            records.append({'year': year, 'indicator_code': domain, 'value': score})
        # Add brittleness score
        records.append({'year': year, 'indicator_code': 'brittleness_score', 'value': item['brittleness_score']})
    
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    return df.pivot(index='year', columns='indicator_code', values='value')

# --- UI Rendering Functions ---
def render_forecast_view(historical_df: pd.DataFrame):
    st.header("Brittleness Forecast")
    
    # Only show available weighting options
    weighting_options = ["framework_average"]  # Only use what's implemented
    weighting = st.selectbox("Select Adversarial Weighting Scheme", options=weighting_options)

    if weighting:
        with st.spinner("Fetching forecast data..."):
            forecast_df = get_forecast_data(st.session_state.country_code, weighting)
        
        if forecast_df.empty:
            st.warning("Unable to generate forecast. Please check if the API is running and models are loaded.")
            return

        # Create visualization
        if 'brittleness_score' in forecast_df.columns:
            # Using historical proxy for visualization
            hist_data = (1 - historical_df[st.session_state.agency_cols].mean(axis=1)) * 10 if not historical_df.empty else pd.Series(name='brittleness_score_proxy')
            fore_data = forecast_df['brittleness_score'].dropna()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data, mode='lines', name='Historical (Proxy)'))
            fig.add_trace(go.Scatter(x=fore_data.index, y=fore_data, mode='lines', name='Forecast', line=dict(dash='dot')))
            
            fig.update_layout(
                title=f"Brittleness Score Forecast ({weighting})",
                yaxis_title="Brittleness Score (0-10)",
                xaxis_title="Year",
                yaxis_range=[0, 10]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show agency scores breakdown
            agency_cols = [col for col in forecast_df.columns if 'agency' in col]
            if agency_cols:
                st.subheader("Agency Scores Breakdown")
                agency_fig = go.Figure()
                for col in agency_cols:
                    agency_fig.add_trace(go.Scatter(
                        x=forecast_df.index, 
                        y=forecast_df[col], 
                        mode='lines', 
                        name=col.replace('_agency', '').title()
                    ))
                agency_fig.update_layout(
                    title="Forecasted Agency Scores by Domain",
                    yaxis_title="Agency Score",
                    xaxis_title="Year",
                    yaxis_range=[0, 1]
                )
                st.plotly_chart(agency_fig, use_container_width=True)

def render_explanation_view():
    st.header("Model Explanation")
    
    explain_year = st.number_input("Select Year to Explain Forecast For", min_value=2000, max_value=2030, value=2024)

    if st.button(f"Explain Forecast for {explain_year}"):
        with st.spinner("Generating explanation..."):
            explanation = api_request('POST', f"country/{st.session_state.country_code}/explain", json={'year': explain_year})
            if explanation and 'explanation' in explanation:
                st.subheader(f"Feature Impact on Domain Forecasts for {explain_year}")
                domain_to_show = st.selectbox("Select Domain to Explain", options=list(explanation['explanation'].keys()))
                
                if domain_to_show:
                    domain_explanation = explanation['explanation'][domain_to_show]
                    st.metric("Predicted Residual", f"{domain_explanation['predicted_residual']:.4f}", 
                              help=f"Base (average) residual was {domain_explanation['base_value']:.4f}")
                    
                    impacts = domain_explanation['top_feature_impacts']
                    impact_df = pd.DataFrame.from_dict(impacts, orient='index', columns=['Impact']).reset_index()
                    impact_df.columns = ['Feature', 'Impact']
                    
                    fig = px.bar(impact_df.sort_values(by='Impact', key=abs), x='Impact', y='Feature', orientation='h',
                                 title=f"Top Feature Contributions for {domain_to_show}", color='Impact', color_continuous_scale='RdBu')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not generate explanation. Please check if the year has available data.")

# --- Main App Logic ---
st.title("🚨 Agency Monitor")
st.sidebar.title("Controls")

# Country selection
country_name = st.sidebar.selectbox("Select Country", options=["United States", "Haiti"])
st.session_state.country_code = "USA" if country_name == "United States" else "HTI"
st.session_state.agency_cols = ['economic_agency', 'political_agency', 'social_agency', 'health_agency', 'educational_agency']

# Generate placeholder historical data
historical_df = pd.DataFrame(
    np.random.rand(20, 5) * 0.4 + 0.5,
    columns=st.session_state.agency_cols,
    index=pd.RangeIndex(start=2004, stop=2024, name="year")
)

# Tabs
tab1, tab2, tab3 = st.tabs(["Forecast", "Explanation", "Transparency"])

with tab1:
    render_forecast_view(historical_df)
with tab2:
    render_explanation_view()
with tab3:
    st.header("Radical Transparency Protocol")
    st.markdown("In accordance with the Agency Calculus 4.3 'Bill of Rights for the Analyzed,' this project adheres to a radical transparency mandate.")
    st.subheader("Core Formulas")
    st.latex(r"B_{sys} = \frac{\text{Nominal GDP}}{A_{total}}")
    st.latex(r"V(\alpha) = C \times |\Delta A| \times D_{amp}")
    st.latex(r"D_{amp} = \left(\frac{P_{perp}}{P_{vic}}\right)^k")