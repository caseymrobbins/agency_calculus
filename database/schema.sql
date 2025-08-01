-- Drop tables if they exist (in correct order to handle foreign keys)
DROP TABLE IF EXISTS agency_scores CASCADE;
DROP TABLE IF EXISTS observations CASCADE;
DROP TABLE IF EXISTS indicators CASCADE;
DROP TABLE IF EXISTS countries CASCADE;

-- Create countries table
CREATE TABLE countries (
    country_code VARCHAR(3) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    region VARCHAR(255),
    income_level VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indicators table
CREATE TABLE indicators (
    indicator_code VARCHAR(50) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    unit VARCHAR(100),
    source VARCHAR(255),
    topic VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create observations table
CREATE TABLE observations (
    id SERIAL PRIMARY KEY,
    country_code VARCHAR(3) NOT NULL,
    indicator_code VARCHAR(50) NOT NULL,
    year INTEGER NOT NULL,
    value DOUBLE PRECISION,
    dataset_version VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (country_code) REFERENCES countries(country_code),
    FOREIGN KEY (indicator_code) REFERENCES indicators(indicator_code),
    UNIQUE(country_code, indicator_code, year, dataset_version)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_observations_country_indicator ON observations(country_code, indicator_code);
CREATE INDEX IF NOT EXISTS idx_observations_year ON observations(year);
CREATE INDEX IF NOT EXISTS idx_observations_value ON observations(value);

-- Create agency_scores table for Agency Calculus
CREATE TABLE agency_scores (
    id SERIAL PRIMARY KEY,
    country_code VARCHAR(3) NOT NULL,
    year INTEGER NOT NULL,
    economic_agency DOUBLE PRECISION,
    political_agency DOUBLE PRECISION,
    social_agency DOUBLE PRECISION,
    health_agency DOUBLE PRECISION,
    educational_agency DOUBLE PRECISION,
    total_agency DOUBLE PRECISION,
    brittleness_score DOUBLE PRECISION,
    framework_version VARCHAR(50) DEFAULT 'AC4.3',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (country_code) REFERENCES countries(country_code),
    UNIQUE(country_code, year, framework_version)
);

-- Show created tables
\dt
