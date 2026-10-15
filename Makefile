.PHONY: run install check test clean runner
.DEFAULT_GOAL := runner

run:
	poetry run python3 runner.py
install: pyproject.toml poetry.lock
	poetry install
check:
	poetry run flake8 config db model runner.py
test:
	poetry run pytest --cov

clean:
	rm -rf `find . -name "__pycache__" -type d`
	rm -rf `find . -name "*.pyc" -type f`
	rm -rf .ruff_cache
runner: check test run clean
