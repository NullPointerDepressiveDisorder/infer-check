.PHONY: install dev lint format test test-unit test-integration test-mlx-cpu clean

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
	uv run pytest tests/integration -m slow -v --tb=short

# Run the MLX unit tests on the CPU, as Linux CI does.
test-mlx-cpu:
	INFER_CHECK_MLX_DEVICE=cpu uv run pytest tests/unit/test_mlx_generate.py tests/unit/test_mlx_score.py -v --tb=short

clean:
	rm -rf build/ dist/ *.egg-info .mypy_cache .ruff_cache .pytest_cache results/
	find . -type d -name __pycache__ -exec rm -rf {} +
