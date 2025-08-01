#!/bin/bash
# File: setup.sh
# Complete setup script for Agency Calculus ETL

set -e
echo "🚀 Setting up Agency Calculus ETL Pipeline..."

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

# Make scripts executable
chmod +x scripts/*.sh 2>/dev/null || echo "Scripts directory will be created"

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
fi

# Setup database
echo "🗄️  Setting up database..."
export $(cat .env | grep -v '^#' | xargs)
DB_NAME=$(echo $DATABASE_URL | sed 's/.*\///')

# Create database if it doesn't exist
if ! psql -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    echo "Creating database: $DB_NAME"
    createdb $DB_NAME
fi

# Initialize schema
echo "📋 Initializing database schema..."
python api/database.py

# Verify setup
echo "✅ Verifying setup..."
echo "📊 Database tables:"
psql $DB_NAME -c "\dt" 2>/dev/null || echo "Database connection issues"

echo "📁 Data files found:"
ls -la data/raw/ | grep -E '\.(csv|xlsx)$' | wc -l | xargs echo "   Files:"

echo ""
echo "🎉 Setup complete! Next steps:"
echo ""
echo "1. Test ETL pipeline:"
echo "   python etl/etl_world_bank.py --config ingestion/config.yaml --dry-run"
echo "   python etl/etl_vdem.py --config-path ingestion/config.yaml --dry-run"
echo ""
echo "2. Run real ETL:"
echo "   python etl/etl_world_bank.py --config ingestion/config.yaml"
echo "   python etl/etl_vdem.py --config-path ingestion/config.yaml"
echo ""
echo "3. Add Freedom House and UNDP:"
echo "   python etl/bulk_ingestion.py freedom_house"
echo "   python etl/bulk_ingestion.py undp"
echo ""
echo "4. Verify data loaded:"
echo "   psql $DB_NAME -c \"SELECT source, COUNT(*) FROM indicators GROUP BY source;\""