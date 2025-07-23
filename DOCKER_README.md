# Agency Monitor - Docker Setup

This guide helps you run the Agency Monitor system using Docker containers.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB+ RAM recommended
- 10GB+ free disk space

## Quick Start

1. **Clone and setup environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

2. **Build and start everything:**
   ```bash
   make full-setup
   ```

   This will:
   - Build all Docker images
   - Initialize the PostgreSQL database
   - Populate metadata tables
   - Process raw data files
   - Train the ML models

3. **Access the applications:**
   - API: http://localhost:8000
   - Dashboard: http://localhost:8501
   - API Docs: http://localhost:8000/docs

## Service Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Dashboard     │────▶│      API        │
│  (Streamlit)    │     │   (FastAPI)     │
│   Port: 8501    │     │   Port: 8000    │
└─────────────────┘     └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │   PostgreSQL    │
                        │   Port: 5432    │
                        └─────────────────┘
```

## Common Operations

### Starting Services
```bash
# Start all services
make up

# Start with logs
make dev

# View logs
make logs
```

### Data Processing
```bash
# Process raw data files
make process-raw-data

# Run World Bank ETL
make etl-world-bank

# Run bulk ETL
make etl-bulk
```

### Model Training
```bash
# Train models
make train-models

# Run historical validation
make validate-historical
```

### Database Operations
```bash
# Access database shell
make db-shell

# Reset database
make init-db
```

### Development
```bash
# Access API container shell
make shell

# Run tests
make test

# View API logs only
make api-logs
```

## Troubleshooting

### Database Connection Issues
If you see database connection errors:
```bash
# Ensure database is healthy
docker-compose ps db
docker-compose logs db

# Restart database
docker-compose restart db
```

### Model Training Failures
If model training fails:
```bash
# Check if data is populated
make db-shell
# Then run: SELECT COUNT(*) FROM observations;

# Re-run data processing
make process-raw-data
```

### Port Conflicts
If ports are already in use:
```bash
# Change ports in docker-compose.yml
# Or stop conflicting services
```

## Data Volume Management

Your data is persisted in:
- PostgreSQL data: Docker volume `postgres_data`
- Raw data files: `./data/raw/` (mounted)
- Trained models: `./models/` (mounted)
- Configuration: `./config/` (mounted)

## Production Deployment

For production:

1. **Update .env file:**
   - Set strong `SECRET_API_KEY`
   - Use production database credentials
   - Set `DEBUG=false`

2. **Use production compose file:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

3. **Enable HTTPS:**
   - Add reverse proxy (nginx/traefik)
   - Configure SSL certificates

## Backup and Restore

### Backup Database
```bash
docker-compose exec db pg_dump -U postgres agency_monitor > backup.sql
```

### Restore Database
```bash
docker-compose exec -T db psql -U postgres agency_monitor < backup.sql
```

## Monitoring

Consider adding:
- Prometheus for metrics
- Grafana for visualization
- Health check endpoints
- Log aggregation (ELK stack)

## Security Notes

- Change default passwords in production
- Use secrets management for API keys
- Enable firewall rules
- Regular security updates

## Support

For issues:
1. Check logs: `make logs`
2. Verify services: `docker-compose ps`
3. Check data: `make db-shell`
4. Run tests: `make test`