.PHONY: install dev lint format test test-unit test-integration clean

install:
	uv sync

dev:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/infer_check/

test: test-unit

test-unit:
	uv run pytest tests/unit -v --tb=short

test-integration:
	uv run pytest tests/integration -v --tb=short

clean:
	rm -rf build/ dist/ *.egg-info .mypy_cache .ruff_cache .pytest_cache results/
	find . -type d -name __pycache__ -exec rm -rf {} +
