-- Complete Database Schema for Agency Calculus
-- File: scripts/create_schema.sql

-- Drop tables if they exist (in reverse dependency order)
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS observations;
DROP TABLE IF EXISTS indicators;
DROP TABLE IF EXISTS countries;

-- Countries table
CREATE TABLE countries (
    country_code VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    region VARCHAR(255),
    income_level VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on country name for faster lookups
CREATE INDEX idx_countries_name ON countries(name);

-- Indicators table
CREATE TABLE indicators (
    indicator_code VARCHAR(100) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    unit VARCHAR(255),
    source VARCHAR(255),
    topic VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes on indicators for faster queries
CREATE INDEX idx_indicators_source ON indicators(source);
CREATE INDEX idx_indicators_topic ON indicators(topic);

-- Observations table (main data table)
CREATE TABLE observations (
    id SERIAL PRIMARY KEY,
    country_code VARCHAR(10) NOT NULL REFERENCES countries(country_code),
    indicator_code VARCHAR(100) NOT NULL REFERENCES indicators(indicator_code),
    year INTEGER NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    dataset_version VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique combination of country, indicator, and year
    UNIQUE(country_code, indicator_code, year)
);

-- Create indexes on observations for faster queries
CREATE INDEX idx_observations_country ON observations(country_code);
CREATE INDEX idx_observations_indicator ON observations(indicator_code);
CREATE INDEX idx_observations_year ON observations(year);
CREATE INDEX idx_observations_country_year ON observations(country_code, year);
CREATE INDEX idx_observations_indicator_year ON observations(indicator_code, year);
CREATE INDEX idx_observations_value ON observations(value);

-- Predictions table (for Agency Calculus forecasting)
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    country_code VARCHAR(10) NOT NULL REFERENCES countries(country_code),
    indicator_code VARCHAR(100) NOT NULL REFERENCES indicators(indicator_code),
    year INTEGER NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    confidence_interval_lower DOUBLE PRECISION,
    confidence_interval_upper DOUBLE PRECISION,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    prediction_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique combination for each prediction
    UNIQUE(country_code, indicator_code, year, model_name, model_version)
);

-- Create indexes on predictions
CREATE INDEX idx_predictions_country ON predictions(country_code);
CREATE INDEX idx_predictions_indicator ON predictions(indicator_code);
CREATE INDEX idx_predictions_year ON predictions(year);
CREATE INDEX idx_predictions_model ON predictions(model_name);

-- Insert sample countries (your focus countries)
INSERT INTO countries (country_code, name, region, income_level) VALUES
('USA', 'United States', 'North America', 'High income'),
('CHN', 'China', 'East Asia & Pacific', 'Upper middle income'),
('JPN', 'Japan', 'East Asia & Pacific', 'High income'),
('DEU', 'Germany', 'Europe & Central Asia', 'High income'),
('IND', 'India', 'South Asia', 'Lower middle income'),
('GBR', 'United Kingdom', 'Europe & Central Asia', 'High income'),
('FRA', 'France', 'Europe & Central Asia', 'High income'),
('ITA', 'Italy', 'Europe & Central Asia', 'High income'),
('CAN', 'Canada', 'North America', 'High income'),
('BRA', 'Brazil', 'Latin America & Caribbean', 'Upper middle income'),
('HTI', 'Haiti', 'Latin America & Caribbean', 'Low income'),
('SDN', 'Sudan', 'Sub-Saharan Africa', 'Low income'),
('MMR', 'Myanmar', 'East Asia & Pacific', 'Lower middle income')
ON CONFLICT (country_code) DO NOTHING;

-- Add comments for documentation
COMMENT ON TABLE countries IS 'Country metadata and classifications';
COMMENT ON TABLE indicators IS 'Indicator definitions and metadata from various sources';
COMMENT ON TABLE observations IS 'Time series data for country-indicator combinations';
COMMENT ON TABLE predictions IS 'Model predictions and forecasts for Agency Calculus';

COMMENT ON COLUMN observations.value IS 'Numeric value for the indicator measurement';
COMMENT ON COLUMN observations.dataset_version IS 'Version/date of source dataset';
COMMENT ON COLUMN predictions.predicted_value IS 'Forecasted value from Agency Calculus models';
COMMENT ON COLUMN predictions.confidence_interval_lower IS 'Lower bound of prediction confidence interval';
COMMENT ON COLUMN predictions.confidence_interval_upper IS 'Upper bound of prediction confidence interval';