-- Create extension for UUID if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Function to update 'updated_at' timestamp (reusable for triggers)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = CURRENT_TIMESTAMP;
   RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Table: regions (e.g., Americas, Europe, etc.)
-- Stores geographic regions for country grouping.
CREATE TABLE IF NOT EXISTS regions (
  region_code VARCHAR(50) PRIMARY KEY,  -- e.g., 'AMER'
  region_name VARCHAR(100) NOT NULL UNIQUE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Trigger for auto-updating updated_at
CREATE TRIGGER trg_regions_updated_at
BEFORE UPDATE ON regions
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Table: countries
-- Stores country metadata with region linkage.
CREATE TABLE IF NOT EXISTS countries (
  country_code CHAR(3) PRIMARY KEY,  -- ISO 3-letter code, e.g., 'USA'
  country_name VARCHAR(100) NOT NULL UNIQUE,
  region_code VARCHAR(50) REFERENCES regions(region_code) ON DELETE SET NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Trigger for auto-updating updated_at
CREATE TRIGGER trg_countries_updated_at
BEFORE UPDATE ON countries
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- ENUM Types for consistency (replaces CHECK constraints)
CREATE TYPE indicator_domain AS ENUM ('economic', 'political', 'social', 'health', 'educational', 'composite', 'exogenous');
CREATE TYPE risk_level_enum AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL');
CREATE TYPE note_category AS ENUM ('economic', 'political', 'social', 'health', 'educational', 'other');
CREATE TYPE ingestion_status AS ENUM ('SUCCESS', 'FAILURE', 'PARTIAL');

-- Table: indicators
-- Stores metadata for data indicators/sources.
CREATE TABLE IF NOT EXISTS indicators (
  indicator_code VARCHAR(50) PRIMARY KEY,
  indicator_name VARCHAR(255) NOT NULL,
  description TEXT,
  source VARCHAR(100) NOT NULL,
  access_method VARCHAR(10) NOT NULL CHECK (access_method IN ('API', 'Bulk')),
  domain indicator_domain,
  unit_of_measure VARCHAR(50),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Trigger for auto-updating updated_at
CREATE TRIGGER trg_indicators_updated_at
BEFORE UPDATE ON indicators
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Table: observations (raw data points)
-- Suggested partitioning by year for large datasets (PostgreSQL 10+): PARTITION BY RANGE (year);
CREATE TABLE IF NOT EXISTS observations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  country_code CHAR(3) REFERENCES countries(country_code) ON DELETE CASCADE,
  indicator_code VARCHAR(50) REFERENCES indicators(indicator_code) ON DELETE CASCADE,
  year INTEGER NOT NULL,
  value NUMERIC,
  dataset_version VARCHAR(50) NOT NULL,
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (country_code, indicator_code, year, dataset_version)
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_observations_country_year ON observations (country_code, year);

-- Trigger for auto-updating updated_at
CREATE TRIGGER trg_observations_updated_at
BEFORE UPDATE ON observations
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Table: agency_scores (wide format for AI input)
-- Normalized scores per domain (0-1 scale).
CREATE TABLE IF NOT EXISTS agency_scores (
  country_code CHAR(3) REFERENCES countries(country_code) ON DELETE CASCADE,
  year INTEGER NOT NULL,
  economic_agency NUMERIC CHECK (economic_agency IS NULL OR (economic_agency >= 0 AND economic_agency <= 1)),
  political_agency NUMERIC CHECK (political_agency IS NULL OR (political_agency >= 0 AND political_agency <= 1)),
  social_agency NUMERIC CHECK (social_agency IS NULL OR (social_agency >= 0 AND social_agency <= 1)),
  health_agency NUMERIC CHECK (health_agency IS NULL OR (health_agency >= 0 AND health_agency <= 1)),
  educational_agency NUMERIC CHECK (educational_agency IS NULL OR (educational_agency >= 0 AND educational_agency <= 1)),
  calculation_version VARCHAR(50) NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (country_code, year, calculation_version)
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_agency_scores_country_year ON agency_scores (country_code, year);

-- Trigger for auto-updating updated_at
CREATE TRIGGER trg_agency_scores_updated_at
BEFORE UPDATE ON agency_scores
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Table: brittleness_predictions
-- ML-generated predictions with metadata.
CREATE TABLE IF NOT EXISTS brittleness_predictions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  country_code CHAR(3) REFERENCES countries(country_code) ON DELETE CASCADE,
  target_year INTEGER NOT NULL,
  brittleness_score NUMERIC NOT NULL,
  confidence_interval_low NUMERIC,
  confidence_interval_high NUMERIC,
  risk_level risk_level_enum,
  trajectory VARCHAR(50),
  days_to_critical INTEGER,
  top_risk_factors JSONB,
  model_version VARCHAR(50) NOT NULL,
  weighting_scheme VARCHAR(50) NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (country_code, target_year, model_version, weighting_scheme)
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_brittleness_predictions_country_year ON brittleness_predictions (country_code, target_year);

-- Trigger for auto-updating updated_at
CREATE TRIGGER trg_brittleness_predictions_updated_at
BEFORE UPDATE ON brittleness_predictions
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Table: iqa_notes (Interpretive Qualitative Annotations)
-- Human annotations for qualitative context.
CREATE TABLE IF NOT EXISTS iqa_notes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  country_code CHAR(3) REFERENCES countries(country_code) ON DELETE CASCADE,
  year INTEGER NOT NULL,
  analyst VARCHAR(100) NOT NULL,
  note TEXT NOT NULL,
  category note_category NOT NULL DEFAULT 'other',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_iqa_notes_country_year ON iqa_notes (country_code, year);

-- Trigger for auto-updating updated_at
CREATE TRIGGER trg_iqa_notes_updated_at
BEFORE UPDATE ON iqa_notes
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Table: ingestion_logs (for ETL monitoring)
-- Logs data ingestion processes for auditing.
CREATE TABLE IF NOT EXISTS ingestion_logs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  source VARCHAR(100) NOT NULL,
  status ingestion_status NOT NULL,
  records_processed INTEGER,
  records_inserted INTEGER,
  records_updated INTEGER,
  error_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- No updated_at needed as logs are immutable.
-- Index for querying by source/status
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_source_status ON ingestion_logs (source, status);