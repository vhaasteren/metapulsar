.PHONY: datetime-date datetime-full datetime-filename datetime test fast help install install-dev test-cov lint format clean docs check ci dev example

# Date-time outputs for different use cases
datetime-date:
	@date '+%Y-%m-%d'

datetime-full:
	@date '+%Y-%m-%d %H:%M:%S'

datetime-filename:
	@date '+%Y-%m-%d-%H-%M-%S'

# Convenience target (defaults to full format)
datetime: datetime-full

# --- Testing -----------------------------------------------------------------
test:
	@pytest tests/

fast:
	@pytest tests/ -m "not slow"

test-integration:
	@pytest tests/ -m "integration" -v

test-real-data:
	@pytest tests/ -m "real_data" -v

test-slow:
	@pytest tests/ -m "slow" -v

# --- Development -------------------------------------------------------------
install:  ## Install package in production mode
	pip install -e .

install-dev:  ## Install package in development mode with all dependencies
	pip install -e ".[dev,libstempo]"
	pre-commit install

test-cov:  ## Run tests with coverage
	pytest --cov=src/metapulsar --cov-report=html --cov-report=term

lint:  ## Run linting
	ruff check src/ tests/

format:  ## Format code
	black src/ tests/
	ruff check --fix src/ tests/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:  ## Build documentation
	cd docs && make html

check:  ## Run all checks (lint, format, test)
	black --check src/ tests/
	ruff check src/ tests/
	pytest

ci:  ## Run CI pipeline locally
	pre-commit run --all-files
	pytest --cov=src/metapulsar --cov-report=xml

dev: install-dev  ## Quick development setup
	@echo "Development environment ready!"
	@echo "Run 'make test' to verify installation"

example:  ## Run example scripts
	python examples/factory_usage_example.py

# --- Help -------------------------------------------------------------------
help:
	@echo "Available targets:"
	@echo ""
	@echo "Testing targets:"
	@echo "  test             - run all tests with pytest"
	@echo "  fast             - run fast tests only (excludes slow tests)"
	@echo "  test-integration - run integration tests only"
	@echo "  test-real-data   - run real data tests"
	@echo "  test-slow        - run slow tests only"
	@echo "  test-cov         - run tests with coverage report"
	@echo ""
	@echo "Development targets:"
	@echo "  install          - install package in production mode"
	@echo "  install-dev      - install package in development mode"
	@echo "  dev              - quick development setup"
	@echo "  lint             - run code linting"
	@echo "  format           - format code with black and ruff"
	@echo "  clean            - clean build artifacts"
	@echo "  docs             - build documentation"
	@echo "  check            - run all checks (lint, format, test)"
	@echo "  ci               - run CI pipeline locally"
	@echo "  example          - run example scripts"
	@echo ""
	@echo "Date-time targets:"
	@echo "  datetime-date     - output current date in YYYY-MM-DD format"
	@echo "  datetime-full     - output current date-time in YYYY-MM-DD HH:MM:SS format"
	@echo "  datetime-filename - output current date-time in YYYY-MM-DD-HH-MM-SS format"
	@echo "  datetime          - convenience target (defaults to datetime-full)"