.PHONY: help build up down logs shell db-shell clean reset init-data train-models test

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build all Docker images
	docker-compose build

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

logs: ## View logs for all services
	docker-compose logs -f

api-logs: ## View API logs
	docker-compose logs -f api

dashboard-logs: ## View dashboard logs
	docker-compose logs -f dashboard

shell: ## Open a shell in the API container
	docker-compose exec api bash

db-shell: ## Open PostgreSQL shell
	docker-compose exec db psql -U postgres agency_monitor

clean: ## Remove all containers and volumes
	docker-compose down -v

reset: clean build ## Reset everything and rebuild

init-db: ## Initialize database schema
	docker-compose up -d db
	sleep 5
	docker-compose exec db psql -U postgres agency_monitor -f /docker-entrypoint-initdb.d/01-schema.sql

populate-metadata: ## Populate metadata tables
	docker-compose run --rm etl python -m etl.populate_metadata

etl-world-bank: ## Run World Bank ETL
	docker-compose run --rm etl python -m etl.etl_world_bank

etl-bulk: ## Run bulk data ETL
	docker-compose run --rm etl python -m etl.etl_bulk

process-raw-data: ## Process all raw data files
	docker-compose run --rm etl python -m etl.parsers.freedom_house_parser
	docker-compose run --rm etl python -m etl.parsers.undp_parser
	docker-compose run --rm etl python -m etl.parsers.vdem_parser

train-models: ## Train ML models
	docker-compose run --rm trainer python scripts/train_models.py

validate-historical: ## Run historical validation
	docker-compose run --rm trainer python scripts/run_historical_validation.py

test: ## Run tests
	docker-compose run --rm trainer pytest tests/

full-setup: build init-db populate-metadata process-raw-data train-models ## Complete setup from scratch

dev: ## Start services in development mode with logs
	docker-compose up

prod: ## Start services in production mode
	docker-compose up -d --scale api=2