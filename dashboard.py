# agency-monitor/dashboard.py

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import Dict, List, Any
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000/api/v1"
try:
    # Use Streamlit's secrets management for the API key. This is a security best practice.
    API_KEY = st.secrets["API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("API key not configured. Please create a .streamlit/secrets.toml file with an API_KEY.")
    st.stop()

INDICATOR_MAP = {
    "gdp_per_capita_usd": "GDP per Capita (USD)",
    "unemployment_rate_percent": "Unemployment Rate (%)",
    "gini_coefficient": "Gini Coefficient",
    "life_expectancy_years": "Life Expectancy (Years)",
    "infant_mortality_rate_per_1000": "Infant Mortality (per 1000 births)",
    "public_health_spending_percent_gdp": "Public Health Spending (% of GDP)",
    "political_freedom_index": "Political Freedom Index",
    "voter_turnout_percent": "Voter Turnout (%)",
    "polarization_index": "Polarization Index",
    "social_trust_index": "Social Trust Index",
    "mean_years_of_schooling": "Mean Years of Schooling",
    "public_education_spending_percent_gdp": "Public Education Spending (% of GDP)",
    "A_econ": "Economic Agency Score",
    "A_poli": "Political Agency Score",
    "A_soc": "Social Agency Score",
    "A_health": "Health Agency Score",
    "A_edu": "Educational Agency Score",
}
AGENCY_DOMAINS = ["A_econ", "A_poli", "A_soc", "A_health", "A_edu"]

# --- Mock API Functions (for standalone development) ---
def mock_get_timeseries(country_code: str) -> List[Dict[str, Any]]:
    st.warning("Using MOCKED API data for timeseries.")
    base_year = 2000 if country_code == "HTI" else 1980
    start_val = 50 if country_code == "HTI" else 60
    factor = -1.5 if country_code == "HTI" else 0.5
    data = []
    for year in range(base_year, 2024):
        for code in INDICATOR_MAP.keys():
            value = start_val + (year - base_year) * factor + (hash(code) % 10)
            data.append({"indicator_code": code, "year": year, "value": max(0, value)})
    return data

def mock_get_forecast(country_code: str, weighting: str) -> Dict[str, List[Dict[str, Any]]]:
    st.warning(f"Using MOCKED API data for forecast with '{weighting}' weighting.")
    forecast_data = {}
    base_value = 20 if country_code == "HTI" else 75
    factors = {"Libertarian": 0.8, "Socialist": 1.2, "Communitarian": 1.0}
    factor = factors.get(weighting, 1.0)
    for domain in AGENCY_DOMAINS:
        forecast_data[domain] = []
        for i in range(1, 11):
            val = base_value + (i * (hash(domain) % 3 - 1) * factor)
            forecast_data[domain].append({"year": 2023 + i, "value": max(0, val)})
    return forecast_data

def mock_post_explain(payload: dict) -> Dict[str, Any]:
    st.warning("Using MOCKED API data for explanation.")
    return {"base_value": 45.5, "shap_values": {"polarization_index": -5.1, "gini_coefficient": -2.4, "A_econ_lag1": 4.1}}

def mock_get_iqa(country_code: str, year: int) -> List[Dict[str, str]]:
    if year == 2008 and country_code == "USA":
        return [{"analyst": "Dr. Smith", "note": "The 2008 financial crisis represents a clear systemic shock, primarily in the economic domain, with cascading effects on social trust."}]
    return []

def mock_post_iqa(payload: dict) -> Dict[str, str]:
    return {"status": "success", "message": "IQA note submitted successfully (mocked)."}


# --- API Interaction Layer ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_api_data(endpoint_path: str, params: Dict = None) -> Any:
    """Generic function to handle GET requests with robust error handling."""
    if not API_KEY:
        st.error("API key not configured.")
        st.stop()
    try:
        headers = {"X-API-Key": API_KEY}
        response = requests.get(f"{API_BASE_URL}/{endpoint_path}", headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # CORRECTED: Placed st.error on a new line
        if e.response.status_code == 401:
            st.error("Authentication failed: Invalid API key.")
        elif e.response.status_code == 404:
            st.error(f"Data not found at endpoint: {endpoint_path}.")
        else:
            st.error(f"API Error ({e.response.status_code}): {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network Error: {e}")
        return None

@st.cache_data
def get_timeseries_data(country_code: str) -> pd.DataFrame:
    # data = get_api_data(f"country/{country_code}/timeseries")
    data = mock_get_timeseries(country_code)
    if not data or not isinstance(data, list) or not all('indicator_code' in d for d in data):
        st.error(f"Invalid timeseries data format for {country_code}.")
        return pd.DataFrame()
    df = pd.DataFrame(data).pivot(index='year', columns='indicator_code', values='value')
    return df.apply(pd.to_numeric, errors='coerce')

@st.cache_data
def get_forecast_data(country_code: str, weighting: str) -> pd.DataFrame:
    # data = get_api_data(f"country/{country_code}/forecast", params={"weighting": weighting})
    data = mock_get_forecast(country_code, weighting)
    # CORRECTED: Placed return on a new line
    if not data:
        return pd.DataFrame()
    all_forecasts = [pd.DataFrame(values).assign(indicator_code=code) for code, values in data.items()]
    df = pd.concat(all_forecasts).pivot(index='year', columns='indicator_code', values='value')
    return df.apply(pd.to_numeric, errors='coerce')

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: df.to_json()})
def get_explanation_data(country_code: str, year: int, country_df: pd.DataFrame) -> Dict[str, Any]:
    feature_payload = country_df.loc[year].to_dict()
    return mock_post_explain(feature_payload)

@st.cache_data
def get_iqa_notes(country_code: str, year: int) -> List[Dict[str, str]]:
    return mock_get_iqa(country_code, year)

def submit_iqa_note(country_code: str, year: int, note: str, analyst: str):
    payload = {"country_code": country_code, "year": year, "note": note, "analyst": analyst}
    return mock_post_iqa(payload)


# --- UI Rendering Functions ---
def render_historical_view(country_df: pd.DataFrame):
    st.header("Historical Data Explorer")
    if country_df.empty:
        st.error("No historical data available to display.")
        return

    options = [(code, name) for code, name in INDICATOR_MAP.items() if code in country_df.columns]
    selected = st.multiselect("Select Indicators", options=options, format_func=lambda x: x[1], default=[o for o in options if o[0] in ["gdp_per_capita_usd", "political_freedom_index"]])
    
    if not selected:
        st.info("Please select at least one indicator to plot.")
        return

    plot_data = country_df[[code for code, name in selected]].dropna(how='all')
    if plot_data.isna().any().any():
        st.warning("Some selected indicators contain missing data, which may affect the plot.")

    fig = px.line(plot_data, labels={"variable": "Indicator", "value": "Value", "year": "Year"}, color_discrete_sequence=px.colors.qualitative.Vivid)

    # Integrate IQA notes directly into the chart with hover annotations.
    for year in plot_data.index:
        iqa_notes = get_iqa_notes(st.session_state.country_code, year)
        if iqa_notes:
            fig.add_annotation(
                x=year, y=plot_data.loc[year].mean(), text="📝", showarrow=False,
                hovertext="<br>".join([f"<b>{n['analyst']}:</b> {n['note']}" for n in iqa_notes]),
                font=dict(size=16)
            )
    st.plotly_chart(fig, use_container_width=True)

def render_prediction_view(country_df: pd.DataFrame):
    st.header("Brittleness Forecast")
    st.markdown("This view shows the AI-powered multi-year forecast. As required by the **Adversarial Weighting Protocol**, you can select different ideological weighting schemes to see how they affect the outcome.")
    
    weighting_scheme = st.selectbox("Select Adversarial Weighting Scheme", ["Communitarian", "Libertarian", "Socialist"])
    with st.spinner("Fetching forecast data..."):
        forecast_df = get_forecast_data(st.session_state.country_code, weighting_scheme)

    if country_df.empty or forecast_df.empty:
        st.error("Historical or forecast data is missing.")
        return

    selected_domain_code, selected_domain_name = st.selectbox("Select Agency Domain", options=[(c, n) for c, n in INDICATOR_MAP.items() if c in AGENCY_DOMAINS], format_func=lambda x: x[1])
    
    hist_data = country_df[[selected_domain_code]].dropna().reset_index()
    fore_data = forecast_df[[selected_domain_code]].dropna().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_data['year'], y=hist_data[selected_domain_code], mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=fore_data['year'], y=fore_data[selected_domain_code], mode='lines', name='Forecast', line=dict(dash='dot')))
    fig.add_vline(x=hist_data['year'].max(), line_width=2, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"Forecast for {selected_domain_name} ({weighting_scheme})", xaxis_title="Year", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

def render_explanation_iqa_view(country_df: pd.DataFrame):
    st.header("Model Explanation & IQA")
    st.markdown("This view provides explanations for model predictions and allows for **Integrated Qualitative Analysis (IQA)**, fulfilling the 'Right to Context' mandate.")
    if country_df.empty:
        st.error("No data available.")
        return

    year_to_explain = st.selectbox("Select a year:", options=sorted(country_df.index.unique(), reverse=True))

    st.subheader(f"IQA Notes for {year_to_explain}")
    iqa_notes = get_iqa_notes(st.session_state.country_code, year_to_explain)
    if iqa_notes:
        for note in iqa_notes:
            st.info(f"**Analyst: {note['analyst']}**\n\n{note['note']}")
    else:
        st.write("No IQA notes for this year.")

    st.subheader(f"Explanation for {year_to_explain + 1} Prediction")
    if st.button("Generate Explanation"):
        with st.spinner("Fetching explanation..."):
            explanation = get_explanation_data(st.session_state.country_code, year_to_explain, country_df)
        if explanation:
            prediction = explanation['base_value'] + sum(explanation['shap_values'].values())
            st.metric(label=f"Predicted Score for {year_to_explain + 1}", value=f"{prediction:.2f}")
            
            # Create a more visual Plotly bar plot for SHAP values
            shap_df = pd.DataFrame(explanation['shap_values'].items(), columns=['Feature', 'SHAP Value']).sort_values('SHAP Value')
            fig = px.bar(
                shap_df, x='SHAP Value', y='Feature', orientation='h',
                title=f"Feature Contributions to Prediction",
                color='SHAP Value', color_continuous_scale='RdBu',
                range_color=[-max(abs(shap_df['SHAP Value'])), max(abs(shap_df['SHAP Value']))]
            )
            fig.add_vline(x=0, line_color="black")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Submit New IQA Note"):
        analyst_name = st.text_input("Your Name/ID")
        note_text = st.text_area("Enter qualitative note:")
        if st.button("Submit Note"):
            if analyst_name and note_text:
                response = submit_iqa_note(st.session_state.country_code, year_to_explain, note_text, analyst_name)
                st.success(response['message'])
            else:
                st.error("Please provide both an analyst name and a note.")

def render_transparency_view():
    st.header("Radical Transparency Protocol")
    st.markdown("In accordance with the Agency Calculus 4.3 'Bill of Rights for the Analyzed,' this project adheres to a radical transparency mandate. All source code, formulas, and data inputs must be public and auditable.")
    
    st.subheader("Source Code")
    st.markdown("- The complete source for the Agency Monitor is available on our public repository: [AC4 Monitor on GitHub](https://github.com/agency-calculus/agency-monitor)")
    
    st.subheader("Core Formulas")
    st.latex(r"V(\alpha) = C \cdot |\Delta A| \cdot D_{amp}")
    st.latex(r"D_{amp} = \left(\frac{P_{perp}}{P_{vic}}\right)^k")
    st.latex(r"B_{sys} = \frac{\text{Nominal GDP}}{A_{total}}")

    st.subheader("Data Sources")
    st.markdown("- A catalog of primary data sources is available in our [Data Sourcing Document](https://github.com/agency-calculus/docs/blob/main/AC4_Data_Source_Research.pdf).")

# --- Main App Logic ---
st.set_page_config(page_title="Agency Monitor", layout="wide", initial_sidebar_state="expanded")
st.title("🚨 Agency Monitor")

st.sidebar.title("Controls")
country_name = st.sidebar.selectbox("Select Country", ["United States", "Haiti"])
st.session_state.country_code = "USA" if country_name == "United States" else "HTI"
st.session_state.country_name = country_name

with st.spinner("Fetching initial data..."):
    country_df = get_timeseries_data(st.session_state.country_code)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Historical View", "📈 Prediction View", "🔬 Explanation & IQA", "📜 Transparency"])

with tab1:
    render_historical_view(country_df)
with tab2:
    render_prediction_view(country_df)
with tab3:
    render_explanation_iqa_view(country_df)
with tab4:
    render_transparency_view()