#!/bin/bash
# File: scripts/run_etl.sh
# ETL pipeline runner for Agency Calculus

set -e  # Exit on any error

echo "🚀 Starting ETL Pipeline for Agency Monitor..."

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

# Check database connection
echo "🔍 Checking database connection..."
DB_NAME=$(echo $DATABASE_URL | sed 's/.*\///')
if ! psql $DB_NAME -c "SELECT 1;" > /dev/null 2>&1; then
    echo "❌ Cannot connect to database. Run setup_database.sh first."
    exit 1
fi

echo "✅ Database connection OK"

# Step 1: World Bank ETL
echo ""
echo "🏦 Step 1: Running World Bank ETL..."
python etl/etl_world_bank.py --config ingestion/config.yaml $DRY_RUN

if [[ -z "$DRY_RUN" ]]; then
    # Check World Bank data loaded
    WB_COUNT=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM indicators WHERE source LIKE '%World Bank%';")
    echo "✅ World Bank indicators loaded: $WB_COUNT"
fi

# Step 2: V-Dem ETL
echo ""
echo "🗳️  Step 2: Running V-Dem ETL..."
python etl/etl_vdem.py --config-path ingestion/config.yaml $DRY_RUN

if [[ -z "$DRY_RUN" ]]; then
    # Check V-Dem data loaded
    VDEM_COUNT=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM indicators WHERE source LIKE '%V-Dem%';")
    echo "✅ V-Dem indicators loaded: $VDEM_COUNT"
fi

# Step 3: Summary
echo ""
echo "📊 ETL Pipeline Summary:"
if [[ -z "$DRY_RUN" ]]; then
    psql $DB_NAME -c "
    SELECT 
        source,
        COUNT(*) as indicator_count
    FROM indicators 
    GROUP BY source 
    ORDER BY source;
    "
    
    TOTAL_OBS=$(psql $DB_NAME -t -c "SELECT COUNT(*) FROM observations;")
    echo "📈 Total observations: $TOTAL_OBS"
else
    echo "🧪 Dry run completed - no data was written to database"
fi

echo ""
echo "✅ ETL Pipeline completed at $(date)"

# Optional: Run bulk ingestion for other sources
if [[ -z "$DRY_RUN" ]] && [[ "$2" == "--include-bulk" ]]; then
    echo ""
    echo "🔄 Running additional bulk ingestion sources..."
    
    # Freedom House (if file exists)
    if [[ -f "data/raw/FIW_2025_AllData.xlsx" ]]; then
        echo "🏛️  Running Freedom House ETL..."
        python etl/bulk_ingestion.py freedom_house
    else
        echo "⚠️  Freedom House file not found, skipping"
    fi
fi

echo "🎉 All done!"