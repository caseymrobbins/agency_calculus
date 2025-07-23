#!/bin/bash
# Agency Monitor Docker Setup Script

set -e  # Exit on error

echo "🚀 Agency Monitor - Docker Setup"
echo "================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before proceeding."
    echo "Press Enter when ready..."
    read
fi

# Build images
echo "🔨 Building Docker images..."
docker-compose build

# Start database
echo "🗄️  Starting database..."
docker-compose up -d db

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Initialize database schema
echo "📊 Initializing database schema..."
docker-compose exec db psql -U postgres -d agency_monitor -c "SELECT 1;" || {
    echo "❌ Database not ready. Waiting more..."
    sleep 5
}

# Populate metadata
echo "📋 Populating metadata..."
docker-compose run --rm etl python -m etl.populate_metadata

# Process raw data files
echo "📁 Processing raw data files..."
echo "  - Processing Freedom House data..."
docker-compose run --rm etl python -m etl.parsers.freedom_house_parser || echo "⚠️  Freedom House parser failed"

echo "  - Processing UNDP data..."
docker-compose run --rm etl python -m etl.parsers.undp_parser || echo "⚠️  UNDP parser failed"

echo "  - Processing V-Dem data..."
docker-compose run --rm etl python -m etl.parsers.vdem_parser || echo "⚠️  V-Dem parser failed"

# Run bulk ETL
echo "🔄 Running bulk ETL..."
docker-compose run --rm etl python -m etl.etl_bulk || echo "⚠️  Bulk ETL failed"

# Train models
echo "🤖 Training models..."
docker-compose run --rm trainer python scripts/train_models.py || {
    echo "⚠️  Model training failed. Check if data was processed correctly."
    echo "You can check the database with: make db-shell"
}

# Start all services
echo "🚀 Starting all services..."
docker-compose up -d

# Wait for services to start
sleep 5

# Check service status
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Access points:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Dashboard: http://localhost:8501"
echo ""
echo "📝 Useful commands:"
echo "  - View logs: make logs"
echo "  - Stop services: make down"
echo "  - Access API shell: make shell"
echo "  - Access database: make db-shell"
echo ""
echo "⚠️  If you encounter issues, check the logs with 'make logs'"