.PHONY: run install check clean runner
_DEFAULT_GOAL := runner

run:
	cd Development; poetry run python3 runner.py
install: pyproject.toml
	cd Development; poetry install
check:
	poetry run flake8
clean:
	rm -rf `find . -name "__pycache__" -type d`
	rm -rf `find . -name "*.pyc" -type f`
	rm -rf .ruff_cache
runner: check run clean