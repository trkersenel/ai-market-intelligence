# Developer entrypoints. Every command a contributor needs lives here, so the
# README never drifts from what CI actually runs.

.DEFAULT_GOAL := help
SHELL := /bin/bash
BACKEND := backend
COMPOSE := docker compose

.PHONY: help
help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- Environment -----------------------------------------------------------
.PHONY: env
env: ## Create .env from the template if it does not exist
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

.PHONY: install-frontend
install-frontend: ## Install frontend dependencies
	cd frontend && npm install

.PHONY: dev-frontend
dev-frontend: ## Run the Vite dev server against a local API
	cd frontend && npm run dev

.PHONY: check-frontend
check-frontend: ## Typecheck, lint and build the frontend
	cd frontend && npx tsc --noEmit && npx eslint . --max-warnings 0 && npm run build

.PHONY: install
install: ## Install backend dependencies into a local virtualenv
	cd $(BACKEND) && python3 -m venv .venv \
		&& .venv/bin/pip install --upgrade pip \
		&& .venv/bin/pip install -e ".[dev]"

# --- Local stack -----------------------------------------------------------
.PHONY: up
up: env ## Start the full stack (postgres, mongo, api)
	$(COMPOSE) up -d --build

.PHONY: down
down: ## Stop the stack, keeping volumes
	$(COMPOSE) down

.PHONY: clean
clean: ## Stop the stack and delete all data volumes
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Tail API logs
	$(COMPOSE) logs -f api

.PHONY: shell
shell: ## Open a shell inside the API container
	$(COMPOSE) exec api /bin/bash

# --- Database --------------------------------------------------------------
.PHONY: migrate
migrate: ## Apply all pending migrations
	$(COMPOSE) exec api alembic upgrade head

.PHONY: migration
migration: ## Autogenerate a migration: make migration m="add companies"
	$(COMPOSE) exec api alembic revision --autogenerate -m "$(m)"

.PHONY: downgrade
downgrade: ## Roll back the most recent migration
	$(COMPOSE) exec api alembic downgrade -1

.PHONY: migration-check
migration-check: ## Fail if the models have drifted from the migrations
	$(COMPOSE) exec api alembic check

.PHONY: seed
seed: ## Load the tracked universe and ensure MongoDB indexes
	$(COMPOSE) exec api python -m app.db.seed

# --- Quality gates ---------------------------------------------------------
.PHONY: lint
lint: ## Run ruff checks and formatting verification
	cd $(BACKEND) && ruff check . && ruff format --check .

.PHONY: format
format: ## Auto-fix lint issues and format code
	cd $(BACKEND) && ruff check --fix . && ruff format .

.PHONY: typecheck
typecheck: ## Run mypy in strict mode
	cd $(BACKEND) && mypy app

.PHONY: test
test: ## Run the unit test suite (no database required)
	cd $(BACKEND) && pytest --cov --cov-report=term-missing -m "not integration"

.PHONY: test-integration
test-integration: ## Run tests against live databases (needs `make up`)
	cd $(BACKEND) && pytest -m integration

.PHONY: test-all
test-all: ## Run every test with coverage
	cd $(BACKEND) && pytest --cov --cov-report=term-missing

.PHONY: check
check: lint typecheck test ## Run every gate CI runs
