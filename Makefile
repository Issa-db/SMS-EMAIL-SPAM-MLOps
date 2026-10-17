.PHONY: install install-tracking lint test check train api clean
.DEFAULT_GOAL := check

install:
	poetry install --with test,lint

install-tracking:
	poetry install --with test,lint,tracking

lint:
	poetry run flake8 src tests

test:
	poetry run pytest

check: lint test

train:
	poetry run python -m src.runner_builder

api:
	poetry run python -m src.runner_inference

clean:
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
