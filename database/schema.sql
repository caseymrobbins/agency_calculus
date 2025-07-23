-- Agency Monitor Database Schema - Final Production Version
-- PostgreSQL implementation for Task 1.1, supporting all project phases.
-- This schema is well-designed and requires no major changes. It is presented here for completeness.

-- Enable UUID extension for primary keys if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table: regions (for geographical context)
CREATE TABLE IF NOT EXISTS regions (
    region_code VARCHAR(50) PRIMARY KEY,
    region_name VARCHAR(100) NOT NULL UNIQUE
);

-- Table: countries
CREATE TABLE IF NOT EXISTS countries (
    country_code CHAR(3) PRIMARY KEY,
    country_name VARCHAR(100) NOT NULL UNIQUE,
    region_code VARCHAR(50) REFERENCES regions(region_code) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: indicators
CREATE TABLE IF NOT EXISTS indicators (
    indicator_code VARCHAR(50) PRIMARY KEY,
    indicator_name VARCHAR(255) NOT NULL,
    description TEXT,
    source VARCHAR(100) NOT NULL,
    access_method VARCHAR(10) NOT NULL CHECK (access_method IN ('API', 'Bulk')),
    domain VARCHAR(20) CHECK (domain IN ('economic', 'political', 'social', 'health', 'educational', 'composite', 'exogenous')),
    unit_of_measure VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: observations (partitioned for performance)
CREATE TABLE IF NOT EXISTS observations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    country_code CHAR(3) NOT NULL REFERENCES countries(country_code) ON DELETE CASCADE,
    indicator_code VARCHAR(50) NOT NULL REFERENCES indicators(indicator_code) ON DELETE CASCADE,
    year INTEGER NOT NULL CHECK (year BETWEEN 1900 AND 2100),
    value NUMERIC, -- Allow NULL for missing data
    dataset_version VARCHAR(50) NOT NULL,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_observation UNIQUE (country_code, indicator_code, year, dataset_version)
) PARTITION BY RANGE (year);

-- Create partitions for different time ranges
CREATE TABLE IF NOT EXISTS observations_y1900_y1999 PARTITION OF observations FOR VALUES FROM (1900) TO (2000);
CREATE TABLE IF NOT EXISTS observations_y2000_y2009 PARTITION OF observations FOR VALUES FROM (2000) TO (2010);
CREATE TABLE IF NOT EXISTS observations_y2010_y2019 PARTITION OF observations FOR VALUES FROM (2010) TO (2020);
CREATE TABLE IF NOT EXISTS observations_y2020_y2029 PARTITION OF observations FOR VALUES FROM (2020) TO (2030);

-- Table: agency_scores (processed data, input for AI models)
CREATE TABLE IF NOT EXISTS agency_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    country_code CHAR(3) NOT NULL REFERENCES countries(country_code) ON DELETE CASCADE,
    year INTEGER NOT NULL CHECK (year BETWEEN 1900 AND 2100),
    economic_agency NUMERIC CHECK (economic_agency >= 0 AND economic_agency <= 1),
    political_agency NUMERIC CHECK (political_agency >= 0 AND political_agency <= 1),
    social_agency NUMERIC CHECK (social_agency >= 0 AND social_agency <= 1),
    health_agency NUMERIC CHECK (health_agency >= 0 AND health_agency <= 1),
    educational_agency NUMERIC CHECK (educational_agency >= 0 AND educational_agency <= 1),
    calculation_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_agency_score UNIQUE (country_code, year, calculation_version)
);

-- Table: brittleness_predictions (final output of the system)
CREATE TABLE IF NOT EXISTS brittleness_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    country_code CHAR(3) NOT NULL REFERENCES countries(country_code) ON DELETE CASCADE,
    prediction_date DATE NOT NULL,
    target_year INTEGER NOT NULL CHECK (target_year BETWEEN 1900 AND 2100),
    brittleness_score NUMERIC CHECK (brittleness_score >= 0 AND brittleness_score <= 10),
    confidence_lower NUMERIC,
    confidence_upper NUMERIC,
    risk_level VARCHAR(20) CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    model_version VARCHAR(20) NOT NULL,
    weighting_scheme VARCHAR(20) NOT NULL CHECK (weighting_scheme IN ('Libertarian', 'Socialist', 'Communitarian', 'Framework Average')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_brittleness_prediction UNIQUE (country_code, target_year, model_version, weighting_scheme)
);

-- Table: iqa_notes (for Integrated Qualitative Analysis)
CREATE TABLE IF NOT EXISTS iqa_notes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    country_code CHAR(3) NOT NULL REFERENCES countries(country_code) ON DELETE CASCADE,
    year INTEGER NOT NULL CHECK (year BETWEEN 1900 AND 2100),
    analyst VARCHAR(100) NOT NULL,
    note TEXT NOT NULL,
    category VARCHAR(50) CHECK (category IN ('Economic', 'Political', 'Social', 'Health', 'Educational', 'Other')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: ingestion_logs (for ETL monitoring)
CREATE TABLE IF NOT EXISTS ingestion_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('SUCCESS', 'FAILURE', 'RUNNING')),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    records_processed INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Trigger function to update 'updated_at' columns automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to all relevant tables
CREATE OR REPLACE TRIGGER set_countries_updated_at BEFORE UPDATE ON countries FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER set_indicators_updated_at BEFORE UPDATE ON indicators FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER set_observations_updated_at BEFORE UPDATE ON observations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER set_agency_scores_updated_at BEFORE UPDATE ON agency_scores FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER set_brittleness_predictions_updated_at BEFORE UPDATE ON brittleness_predictions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER set_iqa_notes_updated_at BEFORE UPDATE ON iqa_notes FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_observations_country_year ON observations (country_code, year);
CREATE INDEX IF NOT EXISTS idx_agency_scores_country_year ON agency_scores (country_code, year);
CREATE INDEX IF NOT EXISTS idx_brittleness_predictions_country_year ON brittleness_predictions (country_code, target_year);
CREATE INDEX IF NOT EXISTS idx_iqa_notes_country_year ON iqa_notes (country_code, year);