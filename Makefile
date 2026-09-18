.PHONY: sync test lint format

sync:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts kaggle

format:
	uv run ruff format src tests scripts kaggle
