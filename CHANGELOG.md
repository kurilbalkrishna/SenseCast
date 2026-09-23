# Release notes

## v1.0.0 (2026-09-23)

First complete release for the BDS-32 capstone evaluation.

**Added**
* Synthetic retail world with known weather, festival, promotion, search-lead and price effects;
  stockout censoring through a historical ordering policy; injected feed defects.
* Open-Meteo loader with cache and climatology fallback; Maharashtra festival calendar.
* Data contracts (pandera), quarantine with reasons, SHA-256 data manifest.
* Point-in-time feature builder with leakage guards; weather as a horizon-scaled forecast.
* Model ladder B0 / B1 / M1 / M2; SBA / TSB routing; cold-start comparison; conformal intervals.
* Aggregate model and reconciliation (bottom-up, WLS, MinT-shrink) over 1,554 nodes.
* Inventory simulator (P0 / P1 / P2) with service-vs-inventory frontier; production order suggestions.
* FastAPI service: forecasts, orders, decisions, append-only audit, RBAC hook, rate limiting,
  Prometheus metrics, JSON logs, OpenAPI.
* Streamlit dashboard: overview, forecasts with drivers, orders workflow, simulation, monitoring, audit.
* Fault suite, drift monitoring, load test, generated evaluation report with go/no-go gates.
* Docker image, compose stack, GitHub Actions (lint, tests, smoke run, pip-audit, gitleaks, docker build).

**Results:** 14 of 15 gates pass (see `docs/03-evaluation/`).

**Known issues**
* P10 to P90 coverage 85.6% (conservative; integer intervals for slow movers).
* Production inventory position ignores open purchase orders.
* Authentication is a hook; connect a real identity provider before any shared deployment.
