.PHONY: help install test lint format check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install: ## Install the package + dev deps
	uv sync --group dev

test: ## Run unit tests
	uv run pytest -q

lint: ## Lint with ruff
	uv run ruff check .

format: ## Format + autofix with ruff
	uv run ruff format .
	uv run ruff check . --fix

check: lint test ## Lint then test
