# Makefile for NLP Microservice

.PHONY: help install dev-install generate-proto clean test lint format type-check run-server run-client docker-build docker-run

# Variables
PROTO_DIR = proto
SRC_DIR = src/nlpmicroservice
PYTHON = python
POETRY = poetry

# Help
help:
	@echo "Available commands:"
	@echo "  install        - Install dependencies"
	@echo "  dev-install    - Install development dependencies"
	@echo "  generate-proto - Generate gRPC Python code from proto files"
	@echo "  clean          - Remove generated files and cache"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linting"
	@echo "  format         - Format code"
	@echo "  type-check     - Run type checking"
	@echo "  run-server     - Start the gRPC server"
	@echo "  run-client     - Run the example client"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"

# Installation
install:
	$(POETRY) install --only=main

dev-install:
	$(POETRY) install

# Generate gRPC code
generate-proto:
	@echo "Generating gRPC code from proto files..."
	$(POETRY) run python -m grpc_tools.protoc \
		--proto_path=$(PROTO_DIR) \
		--python_out=$(SRC_DIR) \
		--grpc_python_out=$(SRC_DIR) \
		$(PROTO_DIR)/nlp.proto
	@echo "gRPC code generated successfully"

# Clean
clean:
	@echo "Cleaning generated files and cache..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*_pb2.py" -delete
	find . -type f -name "*_pb2_grpc.py" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	@echo "Cleanup completed"

# Testing
test:
	$(POETRY) run pytest

test-verbose:
	$(POETRY) run pytest -v

test-coverage:
	$(POETRY) run pytest --cov=nlpmicroservice --cov-report=html --cov-report=term

# Code quality
lint:
	$(POETRY) run flake8 src/ tests/

format:
	$(POETRY) run black src/ tests/
	$(POETRY) run isort src/ tests/

type-check:
	$(POETRY) run mypy src/

# Quality checks (all)
quality: format lint type-check

# Run services
run-server:
	$(POETRY) run nlp-server

run-client:
	$(POETRY) run python -m nlpmicroservice.client

# Development
dev-setup: dev-install generate-proto

# Build and run
build: generate-proto
	$(POETRY) build

# Docker
docker-build:
	docker build -t nlpmicroservice .

docker-run:
	docker run -p 8161:8161 nlpmicroservice

# Development server with auto-reload
dev-server:
	$(POETRY) run watchmedo auto-restart --directory=src --pattern="*.py" --recursive -- $(POETRY) run nlp-server

# Install pre-commit hooks
install-hooks:
	$(POETRY) run pre-commit install

# Run pre-commit on all files
pre-commit:
	$(POETRY) run pre-commit run --all-files

# Check dependencies
deps-check:
	$(POETRY) check
	$(POETRY) show --outdated

# Export requirements
export-requirements:
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes
	$(POETRY) export -f requirements.txt --output requirements-dev.txt --without-hashes --with dev
