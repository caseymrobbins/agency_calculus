Agency Monitor
An early warning system for societal brittleness based on the Agency Calculus 4.3 framework. This system is designed to provide a quantitative measure of a society's resilience or fragility by analyzing its "choice space" across five key domains.

Overview
The Agency Monitor ingests a wide range of global indicators, processes them through the lens of Agency Calculus, and produces a Systemic Brittleness Score (B_sys) on a scale of 0-10. This score represents the ratio of a society's nominal wealth (e.g., GDP) to its actual agency or freedom of choice.

The core of the project is the Agency Calculus 4.3, a framework that defines and quantifies "agency" across five domains:

Economic: The ability of individuals to participate in the economy and improve their financial well-being.

Political: The freedom to participate in the political process and hold power accountable.

Social: The level of trust, cohesion, and freedom of association within a society.

Health: Access to healthcare and the ability to make choices that lead to a healthy life.

Educational: Access to quality education and the ability to acquire knowledge and skills.

Current Status
The system currently measures US brittleness at a concerning 9+/10.

It is undergoing validation using a case study of Haiti.

Historical validation has been performed against several known societal collapses, including 

Chile (1973), Iran (1979), the Soviet Union (1991), and Rwanda (1994).

Architecture
The Agency Monitor is built with a modern Python-based architecture, designed for scalability and maintainability.

ETL Layer: Scripts in the etl/ directory handle Extract, Transform, and Load operations. This includes 

etl_world_bank.py for API-based data ingestion and etl_bulk.py for processing large datasets like V-Dem. A real-time aggregator (


etl/realtime_aggregator.py) provides live data for the prediction engine.

AI Layer: Housed in the ai/ directory, this layer contains the core machine learning models. A 

HybridForecaster combines VARX and XGBoost models for time-series predictions , while a 

BrittlenessPredictor uses XGBoost to calculate the final brittleness score.


Agency Calculator: The agency/ directory contains the implementation of the Agency Calculus 4.3 framework, including the BrittlenessEngine which calculates the final score.



API: A FastAPI backend provides a secure and efficient interface to the system's data and predictions.


Dashboard: A Streamlit frontend offers a user-friendly interface for visualizing data and model outputs.


Database: The system uses a PostgreSQL database, with the schema defined in database/schema.sql. It includes tables for countries, indicators, observations, predictions, and more.



Installation and Setup
Prerequisites
Python 3.8+

PostgreSQL

An API key (for accessing the API)

1. Clone the Repository
Bash

git clone <repository-url>
cd agency-monitor
2. Install Dependencies
It's recommended to use a virtual environment.

Bash

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
3. Database Setup
Create a PostgreSQL database:

SQL

CREATE DATABASE agency_monitor;
Set the DATABASE_URL environment variable. You can create a .env file in the project root:

DATABASE_URL="postgresql://<user>:<password>@<host>:<port>/agency_monitor"
API_KEY="your_secret_api_key"
Initialize the database schema:

Bash

python api/database.py
This will create all the necessary tables as defined in 

database/schema.sql.

4. Data Ingestion
You'll need to populate the database with data from various sources.

Populate Metadata: This step populates the countries and indicators tables.

Bash

python etl/populate_metadata.py
Run ETL Scripts:

For World Bank data:

Bash

python etl/etl_world_bank.py
For bulk data sources like V-Dem:

Bash

python etl/etl_bulk.py v_dem
5. Training the Models
To train the forecasting and brittleness models, run the training script:

Bash

python scripts/train_models.py
This will create model artifacts in the 

models/ directory, which are then used by the API.

6. Run the Application
You can run the FastAPI server and the Streamlit dashboard separately.

Run the API:

Bash

uvicorn api.main:app --reload
Run the Dashboard:

Bash

streamlit run dashboard.py


Overview of the Repository
The GitHub repository at https://github.com/caseymrobbins/agency_calculus/ is owned by Casey M. Robbins (author of Agency Calculus 4.3 framework). It implements "Agency Monitor," a practical software system based on AC4.3 principles for measuring societal health through "agency flow" and systemic brittleness (B_sys). The project quantifies risks like inequality, polarization, and institutional fragility across domains (economic, political, social, health, educational), producing a brittleness score (0-10 scale). It's described as a tool to diagnose societal resilience, with historical validation (e.g., against collapses like Chile 1973 or Rwanda 1994) and current US analysis showing high brittleness (9+/10).github.com

Project Structure and Files
The structure is modular and production-oriented, divided into directories for data processing, models, API, and frontend. 

etl/: Data ingestion layer.
etl_world_bank.py: Fetches indicators from World Bank API (e.g., GDP, inequality metrics).
etl_bulk.py: Handles large datasets like V-Dem (Varieties of Democracy) for political/social scores.
realtime_aggregator.py: Pulls live data for predictions (e.g., news/economic feeds).
ai/: Machine learning core.
HybridForecaster: Time-series model (VARX/XGBoost hybrid) for trend forecasting—matches your code from earlier chats.
BrittlenessPredictor: XGBoost-based final scorer for B_sys, integrating agency calculations.
agency/: AC4.3 implementation.
BrittlenessEngine: Computes B_sys = Nominal_GDP / Total_Agency (scaled 0-10); includes risk levels (LOW/MEDIUM/HIGH/CRITICAL) and percentile calibration.
api/: Backend service.
main.py: FastAPI app entry (run via uvicorn api.main:app).
database.py: Sets up PostgreSQL schema.
database/:
schema.sql: Defines tables (e.g., countries, indicators, observations, predictions)—stores agency scores, GDP, brittleness results.
models/: Stores trained artifacts (e.g., pickled XGBoost models).
scripts/:
train_models.py: Trains forecaster/predictor, saves to models/.
dashboard.py: Streamlit UI for visualizations (e.g., brittleness dashboards, forecasts).
requirements.txt: Dependencies (inferred: pandas, numpy, statsmodels, xgboost, shap, fastapi, streamlit, psycopg2 for DB).

Key Features and Tie to AC4.3
Core Implementation: Directly embodies AC4.3—calculates Total Agency as weighted sum (e.g., economic:0.25, political:0.20 per "framework_average"); Violation Magnitude via V(a)=C*|ΔA|*Damp (with k sensitivity). Brittleness as GDP/Agency proxy measures "fragility."
Ideological Perspectives: Supports "framework_average," "libertarian," etc., with custom weights—enables adversarial comparisons (e.g., summary stats on brittleness variance).
Data-Driven: ETL from sources like World Bank/V-Dem for real-time/historical agency metrics.
Predictive: HybridForecaster for trends (e.g., Gini forecasts); BrittlenessPredictor for scores.
API/Dashboard: Process JSON requests (e.g., agency_scores + GDP → brittleness/violation dict); visualize via Streamlit.
Validation: Tests against historical collapses;
This repo turns AC4.3 from theory (PDFs) into software—great for simulations like US forecasts 