# dashboard.py
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

# --- Configuration ---
st.set_page_config(page_title="Agency Monitor", layout="wide", initial_sidebar_state="expanded")

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")
try:
    # FIX: Retrieve the specific API_KEY from secrets, not the whole dictionary
    API_KEY = st.secrets["API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("API key not found. Please create a .streamlit/secrets.toml file with an API_KEY.")
    st.stop()

# --- API Interaction Layer ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def api_request(method: str, endpoint: str, **kwargs) -> Any:
    """Generic function for making API requests."""
    headers = {"X-API-Key": API_KEY}
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, params=kwargs.get("params"))
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=kwargs.get("json"))
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
def get_timeseries_data(country_code: str) -> pd.DataFrame:
    response = api_request('GET', f"country/{country_code}/timeseries")
    if response and 'data' in response:
        df = pd.DataFrame(response['data'])
        return df.pivot(index='year', columns='indicator_code', values='value')
    return pd.DataFrame()

@st.cache_data(ttl=600)
def get_forecast_data(country_code: str, weighting: str) -> pd.DataFrame:
    response = api_request('GET', f"country/{country_code}/forecast", params={"weighting": weighting})
    if response and 'forecast' in response:
        # FIX: Initialize records as an empty list
        records = []
        for item in response['forecast']:
            year = item['year']
            for domain, score in item['agency_scores'].items():
                records.append({'year': year, 'indicator_code': domain, 'value': score})
            records.append({'year': year, 'indicator_code': 'brittleness_score', 'value': item['brittleness_score']})
        df = pd.DataFrame(records)
        return df.pivot(index='year', columns='indicator_code', values='value')
    return pd.DataFrame()

# --- UI Rendering Functions ---
def render_forecast_view(historical_df: pd.DataFrame):
    st.header("Brittleness Forecast")
    # FIX: Added options for the selectbox based on the Adversarial Weighting Protocol. [cite: 778]
    weighting_options = ["Communitarian", "Libertarian", "Socialist"]
    weighting = st.selectbox("Select Adversarial Weighting Scheme", options=weighting_options)
    
    if weighting:
        forecast_df = get_forecast_data(st.session_state.country_code, weighting)
        if historical_df.empty or forecast_df.empty:
            st.warning("Data not available to display forecast.")
            return

        # Using a proxy for historical brittleness for visualization purposes
        hist_data = historical_df.get('brittleness_score_proxy', pd.Series(name='brittleness_score_proxy')).dropna()
        fore_data = forecast_df['brittleness_score'].dropna()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data, mode='lines', name='Historical (Proxy)'))
        fig.add_trace(go.Scatter(x=fore_data.index, y=fore_data, mode='lines', name='Forecast', line=dict(dash='dot')))
        # FIX: Provided a value for yaxis_range for the 0-10 brittleness score. [cite: 231]
        fig.update_layout(title=f"Brittleness Score Forecast ({weighting})", yaxis_range=[0,10])
        st.plotly_chart(fig, use_container_width=True)

def render_explanation_view(historical_df: pd.DataFrame):
    st.header("Model Explanation")
    if historical_df.empty:
        st.warning("Historical data needed for explanations.")
        return

    # Allow user to select a year from the available historical data
    explain_year = st.selectbox("Select Year to Explain Forecast For", options=sorted(historical_df.index.unique(), reverse=True)[1:])
    
    if st.button(f"Explain Forecast for {explain_year + 1}"):
        with st.spinner("Generating explanation..."):
            explanation = api_request('POST', f"country/{st.session_state.country_code}/explain", json={'year': explain_year + 1})
            if explanation:
                st.subheader(f"Feature Impact on Domain Forecasts")
                domain_to_show = st.selectbox("Select Domain to Explain", options=list(explanation.keys()))
                domain_explanation = explanation[domain_to_show]
                
                st.metric("Predicted Residual", f"{domain_explanation['predicted_residual']:.4f}",
                          help=f"Base (average) residual was {domain_explanation['base_value_residual']:.4f}")

                impacts = domain_explanation['top_feature_impacts']
                impact_df = pd.DataFrame.from_dict(impacts, orient='index', columns=['Impact']).reset_index()
                impact_df.columns = ['Feature', 'Impact']
                
                fig = px.bar(impact_df.sort_values(by='Impact', key=abs), 
                             x='Impact', y='Feature', orientation='h',
                             title=f"Top Feature Contributions for {domain_to_show}",
                             color='Impact', color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)

# --- Main App Logic ---
st.title("🚨 Agency Monitor")
st.sidebar.title("Controls")
# FIX: Added options for the selectbox based on project validation targets. 
country_name = st.sidebar.selectbox("Select Country", ["United States", "Haiti"])
st.session_state.country_code = "USA" if country_name == "United States" else "HTI"

historical_df = get_timeseries_data(st.session_state.country_code)
# Create a proxy for historical brittleness for visualization
if not historical_df.empty and 'total_agency' in historical_df.columns:
    historical_df['brittleness_score_proxy'] = (1 - historical_df['total_agency']) * 10

# FIX: Added titles for the tabs
tab1, tab2, tab3 = st.tabs(["Brittleness Forecast", "Model Explanation", "Transparency Protocol"])

with tab1:
    render_forecast_view(historical_df)
with tab2:
    render_explanation_view(historical_df)
with tab3:
    st.header("Radical Transparency Protocol")
    st.markdown("In accordance with the Agency Calculus 4.3 'Bill of Rights for the Analyzed,' this project adheres to a radical transparency mandate. [cite: 783, 784]")
    st.subheader("Core Formulas")
    st.latex(r"B_{sys} = \frac{\text{Nominal GDP}}{A_{total}}")
    st.latex(r"V(\alpha) = C \times |\Delta A| \times D_{amp}")