#!/bin/bash
# File: scripts/run_etl_complete.sh
# Complete ETL pipeline runner for Agency Calculus (1960-2024)

set -e  # Exit on any error

echo "🚀 Starting Complete ETL Pipeline for Agency Monitor (1960-2024)..."

# Check if dry run
DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "🧪 Running in DRY RUN mode - no data will be written"
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Create logs directory
mkdir -p logs

echo "📊 Starting ETL pipeline at $(date)"
echo "📅 Data coverage: 1960-2024 (65 years)"

# Check database connection
echo "🔍 Checking database connection..."
DB_NAME=$(echo $DATABASE_URL | sed 's/.*\///')
if ! psql $DB_NAME -c "SELECT 1;" > /dev/null 2>&1; then
    echo "❌ Cannot connect to database. Run scripts/setup_complete.sh first."
    exit 1
fi

echo "✅ Database connection OK"

# Step 1: World Bank ETL (Economic, Health, Educational domains)
echo ""
echo "🏦 Step 1: Running World Bank ETL (1960-2024)..."
echo "   Expected indicators: ~18 (GDP, Gini, unemployment, life expectancy, etc.)"
echo "   Expected observations: ~6,000+ (vs 3,571 from 2000 start)"

python etl/etl_world_bank.py --config ingestion/config.yaml $DRY_RUN

if [[ -z "$DRY_RUN" ]]; then
    # Check World Bank data loaded
    WB_COUNT=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM indicators WHERE source LIKE '%World Bank%';" | xargs)
    WB_OBS=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM observations WHERE dataset_version LIKE 'WB-API%';" | xargs)
    echo "✅ World Bank indicators loaded: $WB_COUNT"
    echo "✅ World Bank observations loaded: $WB_OBS"
    
    # Check year coverage
    WB_YEARS=$(psql $DB_NAME -t -c "SELECT MIN(year) || '-' || MAX(year) FROM observations WHERE dataset_version LIKE 'WB-API%';" | xargs)
    echo "📅 World Bank year coverage: $WB_YEARS"
fi

# Step 2: V-Dem ETL (Political domain)
echo ""
echo "🗳️  Step 2: Running V-Dem ETL (1960-2024)..."
echo "   Expected indicators: ~11 (democracy indices, political rights, etc.)"
echo "   Expected observations: ~8,000+ from V-Dem v15"

python etl/etl_vdem.py --config-path ingestion/config.yaml $DRY_RUN

if [[ -z "$DRY_RUN" ]]; then
    # Check V-Dem data loaded
    VDEM_COUNT=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM indicators WHERE source LIKE '%V-Dem%';" | xargs)
    VDEM_OBS=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM observations WHERE dataset_version LIKE 'V-Dem%';" | xargs)
    echo "✅ V-Dem indicators loaded: $VDEM_COUNT"
    echo "✅ V-Dem observations loaded: $VDEM_OBS"
    
    # Check year coverage
    VDEM_YEARS=$(psql $DB_NAME -t -c "SELECT MIN(year) || '-' || MAX(year) FROM observations WHERE dataset_version LIKE 'V-Dem%';" | xargs)
    echo "📅 V-Dem year coverage: $VDEM_YEARS"
fi

# Step 3: Summary and Analysis
echo ""
echo "📊 ETL Pipeline Summary:"
if [[ -z "$DRY_RUN" ]]; then
    echo "📈 Data Sources Loaded:"
    psql $DB_NAME -c "
    SELECT 
        source,
        COUNT(*) as indicator_count
    FROM indicators 
    GROUP BY source 
    ORDER BY source;
    "
    
    echo ""
    echo "📊 Data Coverage Analysis:"
    psql $DB_NAME -c "
    SELECT 
        MIN(year) as earliest_year,
        MAX(year) as latest_year,
        MAX(year) - MIN(year) + 1 as years_covered,
        COUNT(DISTINCT country_code) as countries,
        COUNT(DISTINCT indicator_code) as indicators,
        COUNT(*) as total_observations
    FROM observations;
    "
    
    echo ""
    echo "🏛️ Agency Calculus Domain Coverage:"
    psql $DB_NAME -c "
    SELECT 
        CASE 
            WHEN source LIKE '%World Bank%' AND (
                topic LIKE '%Economic%' OR 
                indicator_code LIKE 'NY.%' OR 
                indicator_code LIKE 'SI.%'
            ) THEN 'Economic Domain'
            WHEN source LIKE '%V-Dem%' OR source LIKE '%Freedom%' THEN 'Political Domain'
            WHEN source LIKE '%World Bank%' AND (
                indicator_code LIKE 'SP.DYN%' OR 
                indicator_code LIKE 'SH.%'
            ) THEN 'Health Domain'  
            WHEN source LIKE '%World Bank%' AND indicator_code LIKE 'SE.%' THEN 'Educational Domain'
            WHEN source LIKE '%UNDP%' OR topic LIKE '%Social%' THEN 'Social Domain'
            ELSE 'Other'
        END as agency_domain,
        COUNT(*) as indicator_count
    FROM indicators 
    GROUP BY 1
    ORDER BY 2 DESC;
    "
    
    echo ""
    echo "🎯 Historical Validation Points Available:"
    psql $DB_NAME -c "
    SELECT 
        'Chile Coup (1973)' as event,
        COUNT(*) as observations_available
    FROM observations 
    WHERE year = 1973 AND country_code = 'CHL'
    UNION ALL
    SELECT 
        'Iran Revolution (1979)' as event,  
        COUNT(*) as observations_available
    FROM observations 
    WHERE year = 1979 AND country_code = 'IRN'
    UNION ALL
    SELECT 
        'Soviet Collapse (1991)' as event,
        COUNT(*) as observations_available  
    FROM observations
    WHERE year = 1991 AND country_code = 'RUS'
    UNION ALL
    SELECT
        'Rwanda Genocide (1994)' as event,
        COUNT(*) as observations_available
    FROM observations 
    WHERE year = 1994 AND country_code = 'RWA';
    "
    
    echo ""
    echo "🇺🇸 US Brittleness Analysis Ready:"
    USA_OBS=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM observations WHERE country_code = 'USA';" | xargs)
    USA_INDICATORS=$(psql $DB_NAME -t -c "SELECT COUNT(DISTINCT indicator_code) FROM observations WHERE country_code = 'USA';" | xargs) 
    USA_YEARS=$(psql $DB_NAME -t -c "SELECT MIN(year) || '-' || MAX(year) FROM observations WHERE country_code = 'USA';" | xargs)
    
    echo "   • Observations: $USA_OBS"
    echo "   • Indicators: $USA_INDICATORS"  
    echo "   • Year coverage: $USA_YEARS"
    echo "   • Ready for Agency Calculus brittleness score calculation"
    
else
    echo "🧪 Dry run completed - no data was written to database"
fi

echo ""
echo "✅ ETL Pipeline completed at $(date)"

# Performance metrics
if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "📈 Performance Metrics:"
    TOTAL_OBS=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM observations;" | xargs)
    TOTAL_COUNTRIES=$(psql $DB_NAME -t -c "SELECT COUNT(DISTINCT country_code) FROM observations;" | xargs)
    TOTAL_INDICATORS=$(psql $DB_NAME -t -c "SELECT COUNT(DISTINCT indicator_code) FROM observations;" | xargs)
    
    echo "   • Total observations: $TOTAL_OBS"
    echo "   • Countries covered: $TOTAL_COUNTRIES" 
    echo "   • Indicators available: $TOTAL_INDICATORS"
    echo "   • Average obs per country: $((TOTAL_OBS / TOTAL_COUNTRIES))"
fi

echo ""
echo "🎯 Ready for Agency Calculus!"
echo ""
echo "Next steps:"
echo "1. 🧮 Calculate domain-specific agency scores"
echo "2. 📊 Compute Total Agency = Σ(weighted domain scores)"  
echo "3. 🚨 Calculate Brittleness = GDP / Total Agency"
echo "4. 🔮 Run HybridForecaster for trend predictions"
echo "5. ⚡ Generate early warning alerts"
echo ""
echo "Historical validation available for:"
echo "   • Chile (1973), Iran (1979), Soviet Union (1991), Rwanda (1994)"
echo ""
echo "🎉 All done!"