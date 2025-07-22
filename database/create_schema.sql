-- database/schema.sql
-- Agency Monitor Database Schema - Final Production Version
-- PostgreSQL implementation for Task 1.1, supporting all project phases.

-- Enable UUID extension for primary keys
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table: regions
CREATE TABLE IF NOT EXISTS regions (
    region_code VARCHAR(50) PRIMARY KEY,
    region_name VARCHAR(100) NOT NULL
);

-- Table: countries
CREATE TABLE IF NOT EXISTS countries (
    country_code CHAR(3) PRIMARY KEY,
    country_name VARCHAR(100) NOT NULL,
    region VARCHAR(50) REFERENCES regions(region_code) ON DELETE SET NULL,
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
    domain VARCHAR(20) CHECK (domain IN ('economic', 'political', 'social', 'health', 'educational', 'composite')),
    unit_of_measure VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: observations (partitioned)
CREATE TABLE IF NOT EXISTS observations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    country_code CHAR(3) NOT NULL REFERENCES countries(country_code) ON DELETE CASCADE,
    indicator_code VARCHAR(50) NOT NULL REFERENCES indicators(indicator_code) ON DELETE CASCADE,
    year INTEGER NOT NULL CHECK (year BETWEEN 1900 AND 2100),
    value NUMERIC CHECK (value >= 0 OR value IS NULL),
    dataset_version VARCHAR(50) NOT NULL,
    data_quality NUMERIC CHECK (data_quality IS NULL OR (data_quality >= 0 AND data_quality <= 1)),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_observation UNIQUE (country_code, indicator_code, year, dataset_version)
) PARTITION BY RANGE (year);

CREATE TABLE observations_y1900_y1999 PARTITION OF observations FOR VALUES FROM (1900) TO (2000);
CREATE TABLE observations_y2000_y2009 PARTITION OF observations FOR VALUES FROM (2000) TO (2010);
CREATE TABLE observations_y2010_y2019 PARTITION OF observations FOR VALUES FROM (2010) TO (2020);
CREATE TABLE observations_y2020_y2029 PARTITION OF observations FOR VALUES FROM (2020) TO (2030);

-- Table: agency_scores
CREATE TABLE IF NOT EXISTS agency_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    country_code CHAR(3) NOT NULL REFERENCES countries(country_code) ON DELETE CASCADE,
    year INTEGER NOT NULL CHECK (year BETWEEN 1900 AND 2100),
    indicator_code VARCHAR(50) NOT NULL REFERENCES indicators(indicator_code) ON DELETE CASCADE,
    score NUMERIC CHECK (score >= 0 AND score <= 1),
    calculation_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_agency_score UNIQUE (country_code, indicator_code, year, calculation_version)
);

-- Table: brittleness_predictions
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

-- Table: changepoints
CREATE TABLE IF NOT EXISTS changepoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    country_code CHAR(3) NOT NULL REFERENCES countries(country_code) ON DELETE CASCADE,
    indicator_code VARCHAR(50) NOT NULL REFERENCES indicators(indicator_code) ON DELETE CASCADE,
    year INTEGER NOT NULL CHECK (year BETWEEN 1900 AND 2100),
    change_magnitude NUMERIC,
    model_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_changepoint UNIQUE (country_code, indicator_code, year, model_version)
);

-- Table: policy_analyses
CREATE TABLE IF NOT EXISTS policy_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    country_code CHAR(3) NOT NULL REFERENCES countries(country_code) ON DELETE CASCADE,
    policy_id VARCHAR(50) NOT NULL,
    target_year INTEGER NOT NULL CHECK (target_year BETWEEN 1900 AND 2100),
    impact_score NUMERIC CHECK (impact_score >= -1 AND impact_score <= 1),
    confidence_lower NUMERIC,
    confidence_upper NUMERIC,
    model_version VARCHAR(50) NOT NULL,
    weighting_scheme VARCHAR(20) NOT NULL CHECK (weighting_scheme IN ('Libertarian', 'Socialist', 'Communitarian', 'Framework Average')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_policy_analysis UNIQUE (country_code, policy_id, target_year, model_version, weighting_scheme)
);

-- Table: iqa_notes
CREATE TABLE IF NOT EXISTS iqa_notes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    country_code CHAR(3) NOT NULL REFERENCES countries(country_code) ON DELETE CASCADE,
    year INTEGER NOT NULL CHECK (year BETWEEN 1900 AND 2100),
    analyst VARCHAR(100) NOT NULL,
    note TEXT NOT NULL,
    category VARCHAR(50) CHECK (category IN ('Economic', 'Political', 'Social', 'Health', 'Educational', 'Other')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_iqa_note UNIQUE (country_code, year, analyst, created_at)
);

-- Table: ingestion_logs
CREATE TABLE IF NOT EXISTS ingestion_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source VARCHAR(100) NOT NULL,
    access_method VARCHAR(10) NOT NULL CHECK (access_method IN ('API', 'Bulk')),
    dataset_version VARCHAR(50) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL CHECK (status IN ('SUCCESS', 'FAILURE', 'RUNNING')),
    records_processed INTEGER CHECK (records_processed >= 0 OR records_processed IS NULL),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: schema_version
CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_observations_y1900_y1999_country_year ON observations_y1900_y1999 (country_code, year);
CREATE INDEX IF NOT EXISTS idx_observations_y2000_y2009_country_year ON observations_y2000_y2009 (country_code, year);
CREATE INDEX IF NOT EXISTS idx_observations_y2010_y2019_country_year ON observations_y2010_y2019 (country_code, year);
CREATE INDEX IF NOT EXISTS idx_observations_y2020_y2029_country_year ON observations_y2020_y2029 (country_code, year);
CREATE INDEX IF NOT EXISTS idx_observations_indicator ON observations (indicator_code);
CREATE INDEX IF NOT EXISTS idx_agency_scores_country_year ON agency_scores (country_code, year);
CREATE INDEX IF NOT EXISTS idx_brittleness_predictions_country_year ON brittleness_predictions (country_code, target_year);
CREATE INDEX IF NOT EXISTS idx_changepoints_country_year ON changepoints (country_code, year);
CREATE INDEX IF NOT EXISTS idx_policy_analyses_country_year ON policy_analyses (country_code, target_year);
CREATE INDEX IF NOT EXISTS idx_iqa_notes_country_year_category ON iqa_notes (country_code, year, category);
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_source ON ingestion_logs (source);

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Data quality default trigger
CREATE OR REPLACE FUNCTION set_data_quality_default()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.data_quality IS NULL THEN
        NEW.data_quality = CASE
            WHEN NEW.indicator_code LIKE 'A_%' THEN 0.9
            ELSE 1.0
        END;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers
CREATE OR REPLACE TRIGGER observations_update_trigger BEFORE UPDATE ON observations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER observations_data_quality_trigger BEFORE INSERT ON observations FOR EACH ROW EXECUTE FUNCTION set_data_quality_default();
CREATE OR REPLACE TRIGGER countries_update_trigger BEFORE UPDATE ON countries FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER indicators_update_trigger BEFORE UPDATE ON indicators FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER agency_scores_update_trigger BEFORE UPDATE ON agency_scores FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER brittleness_predictions_update_trigger BEFORE UPDATE ON brittleness_predictions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER changepoints_update_trigger BEFORE UPDATE ON changepoints FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER policy_analyses_update_trigger BEFORE UPDATE ON policy_analyses FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER iqa_notes_update_trigger BEFORE UPDATE ON iqa_notes FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE OR REPLACE TRIGGER ingestion_logs_update_trigger BEFORE UPDATE ON ingestion_logs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Sample data
INSERT INTO regions (region_code, region_name) VALUES
('NA', 'North America'),
('CAR', 'Caribbean');

INSERT INTO countries (country_code, country_name, region) VALUES
('USA', 'United States', 'NA'),
('HTI', 'Haiti', 'CAR');

INSERT INTO indicators (indicator_code, indicator_name, description, source, access_method, domain, unit_of_measure) VALUES
('NY.GDP.MKTP.CD', 'GDP (current US$)', 'Gross Domestic Product in current US dollars', 'World Bank', 'API', 'economic', 'USD'),
('v2eltrnout', 'Voter Turnout', 'Percentage of eligible voters who voted', 'V-Dem', 'Bulk', 'political', 'Percent'),
('A_econ', 'Economic Agency Score', 'Composite score for economic agency', 'Agency Calculus', 'API', 'composite', 'Index');

INSERT INTO observations (country_code, indicator_code, year, value, dataset_version, notes) VALUES
('USA', 'NY.GDP.MKTP.CD', 2022, 25462700000000, '2025-07-21', 'API-sourced GDP data'),
('HTI', 'v2eltrnout', 2016, 28.8, 'V-Dem-v14', 'Bulk import from V-Dem'),
('USA', 'A_econ', 2022, 0.75, '2025-07-21', 'Calculated economic agency score');

INSERT INTO iqa_notes (country_code, year, analyst, note, category) VALUES
('USA', 2008, 'Dr. Smith', '2008 financial crisis impacted social trust.', 'Social'),
('USA', 2020, 'Dr. Jones', 'COVID-19 pandemic response led to health agency decline.', 'Health');

INSERT INTO schema_version (version, description) VALUES
('1.0.0', 'Initial Agency Monitor schema with partitioning and full lifecycle support') ON CONFLICT (version) DO NOTHING;