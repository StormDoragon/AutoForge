PYTHONPATH := src

.PHONY: test-local lint

test-local:
	PYTHONPATH=$(PYTHONPATH) python -m pytest -q

lint:
	ruff check src tests
