#!/bin/bash
# File: scripts/setup_complete.sh
# Complete setup script for Agency Calculus ETL with 1960 start year

set -e  # Exit on any error

echo "🚀 Setting up Agency Calculus ETL Pipeline (1960-2024)..."

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Please activate your virtual environment first:"
    echo "   source venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment detected: $(basename $VIRTUAL_ENV)"

# Install/upgrade dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/raw data/processed logs scripts ingestion

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔧 Creating .env file..."
    cat > .env << 'EOF'
DATABASE_URL=postgresql://caseyrobbins@localhost:5432/agency_monitor
DATA_ROOT=data/raw
LOG_LEVEL=INFO
EOF
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check PostgreSQL
echo "🔍 Checking PostgreSQL..."
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL not found. Please install:"
    echo "   brew install postgresql"
    exit 1
fi

if ! pg_isready -q 2>/dev/null; then
    echo "⚠️  PostgreSQL not running. Starting..."
    brew services start postgresql || echo "Please start PostgreSQL manually"
    sleep 3
fi

# Setup database
echo "🗄️  Setting up database..."
DB_NAME=$(echo $DATABASE_URL | sed 's/.*\///')

# Create database if it doesn't exist
if ! psql -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    echo "Creating database: $DB_NAME"
    createdb $DB_NAME
fi

# Create database schema
echo "📋 Creating database schema..."
psql $DB_NAME -f scripts/create_schema.sql

# Verify setup
echo "✅ Verifying setup..."
echo "📊 Database tables:"
psql $DB_NAME -c "\dt"

echo "📁 Data files found:"
find data/raw -name "*.csv" -o -name "*.xlsx" | wc -l | xargs echo "   Files:"

# Check V-Dem file specifically
if [ -f "data/raw/V-Dem-CY-Full+Others-v15.csv" ]; then
    echo "✅ V-Dem file found"
else
    echo "⚠️  V-Dem file not found at data/raw/V-Dem-CY-Full+Others-v15.csv"
fi

# Test database connection
echo "🔗 Testing database connection..."
psql $DB_NAME -c "SELECT COUNT(*) as country_count FROM countries;" || echo "Database connection issue"

echo ""
echo "🎉 Setup complete! Your Agency Calculus ETL is ready for 1960-2024 data."
echo ""
echo "📊 Expected data coverage:"
echo "   • World Bank: 1960-2024 (65 years)"
echo "   • V-Dem: 1960-2024 (65 years)"
echo "   • Total: ~15,000+ observations"
echo ""
echo "Next steps:"
echo ""
echo "1. 🧪 Test World Bank ETL (dry run):"
echo "   python etl/etl_world_bank.py --config ingestion/config.yaml --dry-run"
echo ""
echo "2. 🧪 Test V-Dem ETL (dry run):"
echo "   python etl/etl_vdem.py --config-path ingestion/config.yaml --dry-run"
echo ""
echo "3. 🚀 Run real ETL pipeline:"
echo "   python etl/etl_world_bank.py --config ingestion/config.yaml"
echo "   python etl/etl_vdem.py --config-path ingestion/config.yaml"
echo ""
echo "4. ✅ Verify data loaded:"
echo "   psql $DB_NAME -c \"SELECT source, COUNT(*) FROM indicators GROUP BY source;\""
echo "   psql $DB_NAME -c \"SELECT MIN(year), MAX(year), COUNT(*) FROM observations;\""
echo ""
echo "5. 🎯 Check Agency Calculus domains:"
echo "   psql $DB_NAME -c \"SELECT topic, COUNT(*) FROM indicators GROUP BY topic;\""
echo ""
echo "Historical validation points available:"
echo "   • 1960-1973: Pre-Chile coup baseline"  
echo "   • 1973-1979: Chile → Iran period"
echo "   • 1979-1991: Iran → Soviet collapse"
echo "   • 1991-1994: Soviet → Rwanda period"
echo "   • 1994-2024: Post-Rwanda modern era"
echo ""
echo "Ready for brittleness calculations! 🎯"