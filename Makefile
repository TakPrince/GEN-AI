PYTHON ?= python

.PHONY: test coverage lint clean app eval-smoke eval eval-data report

test:
	$(PYTHON) -m pytest -q

coverage:
	PYTHONPATH=src $(PYTHON) -m pytest tests/unit -q --cov=safety_platform --cov-fail-under=80

lint:
	$(PYTHON) -m black --check src metrics tests
	$(PYTHON) -m ruff check src metrics tests
	$(PYTHON) -m mypy src metrics tests

app:
	PYTHONPATH=src:. $(PYTHON) -m streamlit run src/safety_platform/app.py

eval-data:
	PYTHONPATH=src:. $(PYTHON) -m safety_platform.evaluation fetch --out evaluation/data/jbb-v2

eval-smoke:
	PYTHONPATH=src:. $(PYTHON) -m safety_platform.evaluation run --dataset evaluation/data/jbb-v2/cases.jsonl --db evaluation/results/jbb-v2.sqlite3 --per-label 2

eval:
	PYTHONPATH=src:. $(PYTHON) -m safety_platform.evaluation run --dataset evaluation/data/jbb-v2/cases.jsonl --db evaluation/results/jbb-v2.sqlite3 --per-label 30 --report-json evaluation/results/jbb-v2.json

report:
	PYTHONPATH=src:. $(PYTHON) -m safety_platform.evaluation report --db evaluation/results/jbb-v2.sqlite3

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
