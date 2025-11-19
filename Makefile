#################################################################################
#
# Makefile to build the project
#
#################################################################################

PROJECT_NAME = used-car-price-predictor
PYTHON_INTERPRETER = python
# WD=$(shell pwd)
# PYTHONPATH=${WD}
SHELL := /bin/bash

## Create python interpreter environment.
create-environment:
	@echo ">>> About to create environment: $(PROJECT_NAME)..."
	$(PYTHON_INTERPRETER) --version
	@echo ">>> Setting up VirtualEnv."
	$(PYTHON_INTERPRETER) -m venv venv

# Define utility variable to help calling Python from the virtual environment
ACTIVATE_ENV := source "venv/bin/activate"

# Execute python related functionalities from within the project's environment
define execute_in_env
	$(ACTIVATE_ENV) && $1
endef

## Build the developer environment requirements
dev-requirements: create-environment
	$(call execute_in_env, pip install pip-tools)
	@if [ ! -f requirements-dev.txt ]; then \
		echo "Compiling dev requirements..."; \
		$(call execute_in_env, pip-compile --output-file=requirements-dev.txt requirements-dev.in); \
	else \
		echo "Dev requirements already compiled."; \
	fi
	$(call execute_in_env, pip install -r ./requirements-dev.txt)

################################################################################################################

# Build / Run

## Run the security test (bandit + safety)
security-test:
	$(call execute_in_env, safety scan -r ./requirements.txt)
	$(call execute_in_env, bandit -lll */*.py *c/*.py)

## Run the black code check
run-black:
	$(call execute_in_env, black  ./src/*.py ./test/*.py)

## Run the unit tests
unit-test:
	$(call execute_in_env, PYTHONPATH=${CURDIR} pytest --testdox -v)

## Run the coverage check
check-coverage:
	$(call execute_in_env, PYTHONPATH=${CURDIR} pytest --cov=src test/)

## Run all checks
run-checks: security-test run-black unit-test check-coverage