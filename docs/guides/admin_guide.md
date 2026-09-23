# Administrator guide

## Requirements

Docker 24+ with Compose v2, or Python 3.11 with 4 GB RAM free. The full run takes under 10 minutes on
a 2-core laptop; the `small` profile about 15 seconds.

## Run with Docker (recommended)

```bash
cp .env.example .env            # optional
docker compose up --build       # pipeline runs once, then API :8000 and dashboard :8501 start
docker compose --profile tracking up mlflow   # optional MLflow UI on :5000
```

Outputs land in bind-mounted folders: `data/`, `artifacts/`, `mlruns/`, `docs/03-evaluation/`.

## Run without Docker

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install
make all          # or: make small
make api          # terminal 1
make dashboard    # terminal 2
make loadtest     # optional, with the API running
```

## Configuration

`configs/default.yaml` holds every setting that changes a result (world size, model parameters,
backtest blocks, simulation, thresholds). A profile file overrides parts of it:
`python -m sensecast.cli all --profile small`. Environment overrides:

| Variable | Default | Purpose |
|---|---|---|
| `SENSECAST_CONFIG` | (none) | profile name or path |
| `SENSECAST_DATA_DIR` / `SENSECAST_ARTIFACTS` / `SENSECAST_REPORTS` | repo folders | output locations |
| `SENSECAST_AUTH_MODE` | `off` | `header` = identity from a trusted gateway (X-User, X-Role, X-Store) |
| `SENSECAST_RATE_LIMIT_PER_MIN` | 600 | per-client request budget |
| `SENSECAST_AUDIT_DB` | `artifacts/audit.db` | decision log location |
| `API_URL` | `http://localhost:8000` | used by the dashboard |

## Using real weather

`weather.source: auto` tries the Open-Meteo archive and caches the result in
`data/external/weather_openmeteo.parquet`; with no network it falls back to the synthetic
climatology. The source used is recorded in `data/processed/manifest.json`.

## Operations

* **Health:** `GET /health` (Docker health check uses it).
* **Metrics:** `GET /metrics` in Prometheus format: request latency by route and status, decisions by
  action, rejected requests by reason. Point a Prometheus scrape job at `api:8000/metrics`.
* **Logs:** every component writes one JSON object per line to stdout (request_id, route, status,
  latency; pipeline steps with durations). `docker compose logs -f api`.
* **Retraining:** rerun the pipeline (`docker compose run --rm pipeline`), then restart the API
  (`docker compose restart api`) to load new artifacts.
* **Backups:** the only state that is not reproducible is `artifacts/audit.db`. Back it up.
* **Resetting:** `make clean` deletes generated data and artifacts (not the audit log backup).

## Troubleshooting

| Symptom | Fix |
|---|---|
| API returns 503 "Artifact missing" | the pipeline has not finished: `docker compose logs pipeline` |
| Dashboard "Cannot reach the API" | check `API_URL`, and that the API container is healthy |
| 429 responses | raise `SENSECAST_RATE_LIMIT_PER_MIN` or slow the client |
| Pipeline killed (out of memory) | use `--profile small`, or give Docker at least 4 GB |
| `pip-audit` fails in CI | upgrade the flagged package in `requirements.txt`, run tests, commit |
