PYTHON ?= python

.PHONY: test lint clean

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m black --check src tests
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy src tests

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
