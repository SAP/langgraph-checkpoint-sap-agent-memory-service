lint:
	uv run ruff check langgraph/ tests/
	uv run ruff format --check langgraph/ tests/

format:
	uv run ruff format langgraph/ tests/

typecheck:
	uv run ty check langgraph/

test:
	uv run pytest tests/ -k "not conformance" -v

test-conformance:
	uv run pytest tests/test_conformance.py -v

check: lint typecheck test

.PHONY: lint format typecheck test test-conformance check
