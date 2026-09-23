# SenseCast - common tasks. `make help` lists them.
PY      ?= python
PROFILE ?=
PROF_ARG = $(if $(PROFILE),--profile $(PROFILE),)

.PHONY: help install data backtest faults simulate monitor report all small test lint audit api dashboard loadtest openapi docker-build up down clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:         ## install pinned runtime + dev dependencies and the package
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pip install --no-deps -e .

data:            ## generate the synthetic world and ingest it (validation + quarantine)
	$(PY) -m sensecast.cli generate $(PROF_ARG)
	$(PY) -m sensecast.cli ingest $(PROF_ARG)

backtest:        ## train, route, calibrate, reconcile, score the test block
	$(PY) -m sensecast.cli backtest $(PROF_ARG)

faults:          ## run the failure-mode suite
	$(PY) -m sensecast.cli faults $(PROF_ARG)

simulate:        ## inventory policy simulation + production order suggestions
	$(PY) -m sensecast.cli simulate $(PROF_ARG)

monitor:         ## drift (PSI), weekly accuracy, freshness
	$(PY) -m sensecast.cli monitor $(PROF_ARG)

report:          ## gates + figures + docs/03-evaluation/results.md
	$(PY) -m sensecast.cli report $(PROF_ARG)

all:             ## the whole pipeline end to end
	$(PY) -m sensecast.cli all $(PROF_ARG)

small:           ## whole pipeline on the small profile (~15 s)
	$(PY) -m sensecast.cli all --profile small

test:            ## unit, leakage, fault and integration tests
	$(PY) -m pytest

lint:            ## static checks
	ruff check src tests dashboard

audit:           ## dependency vulnerability scan
	pip-audit -r requirements.txt

ART = $(if $(PROFILE),artifacts/$(PROFILE),artifacts)

api:             ## run the API on :8000 (PROFILE=small serves the small run)
	SENSECAST_ARTIFACTS=$(ART) uvicorn sensecast.api.app:app --host 0.0.0.0 --port 8000

dashboard:       ## run the planner dashboard on :8501
	streamlit run dashboard/app.py

loadtest:        ## latency test against a running API (writes artifacts/loadtest.json)
	SENSECAST_ARTIFACTS=$(ART) $(PY) -m sensecast.evaluation.loadtest --requests 400 --concurrency 8 $(PROF_ARG)

openapi:         ## export the OpenAPI spec to docs/02-design/openapi.json
	$(PY) scripts/export_openapi.py

docker-build:    ## build the container image
	docker build -t sensecast:1.0.0 .

up:              ## pipeline + API + dashboard with docker compose
	docker compose up --build

down:
	docker compose down

clean:           ## remove generated data and artifacts
	rm -rf data/raw data/processed data/quarantine data/ground_truth artifacts mlruns
