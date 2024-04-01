DIRS_PYTHON := .

.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo
	@echo "Targets:"
	@echo "  help    This help (default target)"
	@echo "  format  Format source code"
	@echo "  lint    Run lint checks"
	@echo "  test    Run tests"
	@echo "  ci      Install dependencies, run lints and tests"
	@echo "  docs    Generate the documentation"
	@echo "  jupyterlab Run jupyterlab"
	@echo ""

.PHONY: format
format:	\
	format-isort \
	format-black

.PHONY: format-isort
format-isort:
	mamba run isort --profile=black $(DIRS_PYTHON)

.PHONY: format-black
format-black:
	mamba run black --line-length 100 $(DIRS_PYTHON)

.PHONY: lint
lint: \
	lint-isort \
	lint-black \
	lint-flake8 \
	lint-mypy

.PHONY: lint-isort
lint-isort:
	mamba run isort --profile=black --check-only --diff $(DIRS_PYTHON)

.PHONY: lint-black
lint-black:
	mamba run black --check --line-length 100 --diff $(DIRS_PYTHON)

.PHONY: lint-flake8
flake8:
	mamba run flake8 --max-line-length 100 $(DIRS_PYTHON)

.PHONY: lint-mypy
lint-mypy:
	mamba run mypy $(DIRS_PYTHON)

.PHONY: test
test:
	mamba run pytest \
		-n auto \
		--cov-report term-missing \
		--cov-report lcov \
		--cov=app \
		tests/

.PHONY: test-ci
test-ci:
	pytest \
		--cov-report term-missing \
		--cov-report lcov \
		--cov=app \
		tests/

.PHONY: ci
ci: \
	deps \
	lint \
	test-ci

.PHONY: docs
docs:
	PYTHONPATH=$(PWD) -- make -C ../docs clean html

.PHONY: jupyterlab
jupyterlab:
	mamba run jupyter lab \
	    --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://127.0.0.1:8888
