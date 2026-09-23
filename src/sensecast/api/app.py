"""SenseCast REST API.

Run:  uvicorn sensecast.api.app:app --port 8000
Docs: http://localhost:8000/docs   (OpenAPI spec at /openapi.json)
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import threading
import time
import uuid
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator, model_validator

from sensecast import __version__
from sensecast.api.audit import AuditLog
from sensecast.api.auth import User, current_user, ensure_can_decide, ensure_store_access
from sensecast.config import REPO_ROOT
from sensecast.logging_utils import get_logger

log = get_logger("sensecast.api")

REQUESTS = Histogram("sensecast_request_seconds", "Request latency", ["method", "route", "status"],
                     buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
DECISIONS = Counter("sensecast_decisions_total", "Order decisions recorded", ["action"])
REJECTED = Counter("sensecast_rejected_requests_total", "Requests rejected", ["reason"])

REASON_CODES = {"LOCAL_EVENT", "SUPPLIER_ISSUE", "SHELF_SPACE", "PROMO_CHANGE", "DATA_ERROR",
                "KNOWN_DEMAND_SHIFT", "OTHER"}
# Drivers shown to planners: the demand-sensing signals. Recent sales and item/store identity
# set the item's usual level (they dominate SHAP by construction) and are not shown as drivers.
FAMILY_LABEL = {"drv_weather": "weather", "drv_events": "festivals & holidays", "drv_promotion": "promotion",
                "drv_search": "search interest", "drv_calendar": "day of week / season"}


def artifacts_dir() -> Path:
    return Path(os.getenv("SENSECAST_ARTIFACTS", REPO_ROOT / "artifacts"))


class State:
    """Lazy, cached access to pipeline outputs."""

    def __init__(self, root: Path):
        self.root = root

    def _pq(self, name: str) -> pd.DataFrame:
        p = self.root / name
        if not p.exists():
            raise HTTPException(503, detail=f"Artifact {name} missing - run the pipeline first (make all)")
        return pd.read_parquet(p)

    def _js(self, name: str) -> dict:
        p = self.root / name
        return json.loads(p.read_text()) if p.exists() else {}

    @cached_property
    def series(self) -> pd.DataFrame:
        return self._pq("series.parquet")

    @cached_property
    def production(self) -> pd.DataFrame:
        return self._pq("forecasts/production.parquet")

    @cached_property
    def aggregates(self) -> pd.DataFrame:
        return self._pq("forecasts/aggregates_production.parquet")

    @cached_property
    def orders(self) -> pd.DataFrame:
        return self._pq("orders_production.parquet")

    @cached_property
    def history(self) -> pd.DataFrame:
        return self._pq("history_recent.parquet")

    @cached_property
    def production_by_series(self) -> dict:
        """(store_id, item_id) -> forecast rows, so a request is a dict lookup, not a scan."""
        return {k: g.sort_values("h") for k, g in self.production.groupby(["store_id", "item_id"])}

    @cached_property
    def history_by_node(self) -> dict:
        return {k: g.tail(60) for k, g in self.history.groupby("node_id")}

    @cached_property
    def metrics(self) -> dict:
        return self._js("metrics.json")

    @cached_property
    def model_version(self) -> str:
        return f"{__version__}+{self.metrics.get('config_hash', 'unknown')}"


class RateLimiter:
    """Token bucket per client address."""

    def __init__(self, per_minute: int):
        self.rate = per_minute / 60.0
        self.cap = per_minute
        self.buckets: dict[str, list[float]] = defaultdict(lambda: [float(per_minute), time.monotonic()])
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self.lock:
            tokens, last = self.buckets[key]
            now = time.monotonic()
            tokens = min(self.cap, tokens + (now - last) * self.rate)
            ok = tokens >= 1
            self.buckets[key] = [tokens - 1 if ok else tokens, now]
            return ok


def escape_csv_cell(v):
    """Neutralise spreadsheet formula injection (=, +, -, @, tab, CR at the start)."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else round(float(v), 3)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.date().isoformat()
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict(orient="records")]


# ---------------------------------------------------------------------------
class Decision(BaseModel):
    store_id: str = Field(pattern=r"^S\d{2}$")
    item_id: str = Field(pattern=r"^I\d{3}$")
    order_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    action: Literal["approve", "override", "reject"]
    final_qty: int = Field(ge=0, le=100_000)
    reason_code: str | None = None
    note: str | None = Field(default=None, max_length=280)

    @field_validator("note")
    @classmethod
    def strip_note(cls, v):
        return v.strip() if v else v

    @model_validator(mode="after")
    def reason_needed(self):
        if self.action in ("override", "reject") and self.reason_code not in REASON_CODES:
            raise ValueError(f"reason_code required for {self.action}; one of {sorted(REASON_CODES)}")
        if self.reason_code is not None and self.reason_code not in REASON_CODES:
            raise ValueError(f"unknown reason_code; one of {sorted(REASON_CODES)}")
        return self


def create_app(root: Path | None = None) -> FastAPI:
    app = FastAPI(title="SenseCast API", version=__version__,
                  description="Demand forecasts, suggested orders and an auditable approval workflow.")
    root = root or artifacts_dir()
    state = State(root)
    audit = AuditLog(Path(os.getenv("SENSECAST_AUDIT_DB", root / "audit.db")))
    limiter = RateLimiter(int(os.getenv("SENSECAST_RATE_LIMIT_PER_MIN", "600")))
    app.state.sc = state
    app.state.audit = audit

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        client = request.client.host if request.client else "unknown"
        t0 = time.perf_counter()
        if request.url.path not in ("/health", "/metrics") and not limiter.allow(client):
            REJECTED.labels("rate_limit").inc()
            response = JSONResponse({"detail": "Rate limit exceeded, retry shortly"}, status_code=429)
        elif int(request.headers.get("content-length", "0") or 0) > 64_000:
            REJECTED.labels("payload_too_large").inc()
            response = JSONResponse({"detail": "Payload too large"}, status_code=413)
        else:
            response = await call_next(request)
        dt = time.perf_counter() - t0
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        REQUESTS.labels(request.method, path, str(response.status_code)).observe(dt)
        response.headers["X-Request-ID"] = rid
        log.info("request", extra=dict(request_id=rid, method=request.method, path=path,
                                       status=response.status_code, ms=round(dt * 1000, 2)))
        return response

    @app.get("/health", tags=["ops"])
    def health():
        ok = (root / "metrics.json").exists()
        return {"status": "ok" if ok else "degraded", "version": __version__,
                "model_version": state.model_version if ok else None,
                "last_observed_day": state.metrics.get("dates", {}).get("last_obs") if ok else None}

    @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/stores", tags=["catalogue"])
    def stores(user: User = Depends(current_user)):
        s = state.series.groupby(["store_id", "city"]).size().rename("items").reset_index()
        if user.role == "store_manager":
            s = s[s.store_id == user.store_id]
        return records(s)

    @app.get("/v1/items", tags=["catalogue"])
    def items(store_id: str = Query(pattern=r"^S\d{2}$"), user: User = Depends(current_user)):
        ensure_store_access(user, store_id)
        s = state.series[state.series.store_id == store_id]
        return records(s[["item_id", "category", "demand_class", "price", "pack_size", "on_hand"]])

    @app.get("/v1/forecast/{store_id}/{item_id}", tags=["forecasts"])
    def forecast(store_id: str, item_id: str, user: User = Depends(current_user)):
        ensure_store_access(user, store_id)
        f = state.production_by_series.get((store_id, item_id))
        if f is None:
            raise HTTPException(404, detail="No forecast for this store and item")
        hist = state.history_by_node.get(f"{store_id}|{item_id}", state.history.iloc[:0])
        drv_cols = [c for c in f.columns if c in FAMILY_LABEL]
        eff = (np.exp(f[drv_cols].mean()) - 1).sort_values(key=np.abs, ascending=False)
        drivers = [{"factor": FAMILY_LABEL[c], "effect_pct": round(100 * float(v), 1)} for c, v in eff.items()]
        route = str(f.route.iloc[0])
        return {
            "store_id": store_id, "item_id": item_id, "category": f.category.iloc[0],
            "origin_date": pd.Timestamp(f.origin_date.iloc[0]).date().isoformat(),
            "demand_class": str(f.demand_class.iloc[0]), "model": route,
            "model_version": state.model_version,
            "note": None if route == "m2" else f"Forecast from {route.upper()}; drivers describe the multisource model.",
            "history": records(hist[["date", "units"]]),
            "forecast": records(f[["date", "h", "p10c", "final", "p90c"]].rename(
                columns={"p10c": "p10", "final": "p50", "p90c": "p90"})),
            "drivers": drivers,
        }

    @app.get("/v1/stores/{store_id}/forecast", tags=["forecasts"])
    def store_forecast(store_id: str, user: User = Depends(current_user)):
        ensure_store_access(user, store_id)
        a = state.aggregates[state.aggregates.node_id.isin([f"S:{store_id}"]) |
                             state.aggregates.node_id.str.startswith(f"SC:{store_id}|")]
        if a.empty:
            raise HTTPException(404, detail="Unknown store")
        a = a.assign(node=a.node_id.str.replace(f"SC:{store_id}|", "", regex=False).str.replace(f"S:{store_id}", "store total", regex=False))
        return records(a[["node", "level", "date", "base", "reconciled"]])

    def _orders_view(store_id: str | None) -> pd.DataFrame:
        o = state.orders
        if store_id:
            o = o[o.store_id == store_id]
        order_date = str(o.order_date.iloc[0]) if len(o) else ""
        latest = audit.latest_by_item(store_id, order_date)
        o = o.copy()
        o["status"] = [latest.get((s, i), {}).get("action", "pending") for s, i in zip(o.store_id, o.item_id, strict=False)]
        o["final_qty"] = [latest.get((s, i), {}).get("final_qty") for s, i in zip(o.store_id, o.item_id, strict=False)]
        return o

    @app.get("/v1/orders", tags=["orders"])
    def orders(store_id: str | None = Query(default=None, pattern=r"^S\d{2}$"),
               status: Literal["pending", "approve", "override", "reject"] | None = None,
               user: User = Depends(current_user)):
        if user.role == "store_manager":
            store_id = user.store_id
        if store_id:
            ensure_store_access(user, store_id)
        o = _orders_view(store_id)
        if status:
            o = o[o.status == status]
        cols = ["store_id", "item_id", "category", "demand_class", "order_date", "on_hand", "mu", "p10", "p90",
                "order_up_to", "suggested_qty", "status", "final_qty"]
        # p10 / p90 here are sums of the daily bands over the protection period (a conservative
        # range for the 4-day total, not its exact quantiles)
        return records(o[cols].rename(columns={"mu": "expected_demand_4d", "p10": "low_4d", "p90": "high_4d"}))

    @app.get("/v1/orders/export.csv", tags=["orders"], response_class=PlainTextResponse)
    def export_orders(store_id: str = Query(pattern=r"^S\d{2}$"), user: User = Depends(current_user)):
        ensure_store_access(user, store_id)
        o = _orders_view(store_id)
        buf = io.StringIO()
        w = csv.writer(buf)
        cols = ["store_id", "item_id", "category", "order_date", "suggested_qty", "status", "final_qty"]
        w.writerow(cols)
        for r in o[cols].itertuples(index=False):
            w.writerow([escape_csv_cell(v) for v in r])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=orders_{store_id}.csv"})

    @app.post("/v1/orders/decision", tags=["orders"], status_code=201)
    def decide(d: Decision, request: Request, user: User = Depends(current_user)):
        ensure_can_decide(user)
        ensure_store_access(user, d.store_id)
        o = state.orders
        row = o[(o.store_id == d.store_id) & (o.item_id == d.item_id)]
        if row.empty:
            raise HTTPException(404, detail="No suggested order for this store and item")
        if str(row.order_date.iloc[0]) != d.order_date:
            raise HTTPException(409, detail=f"Suggested orders are for {row.order_date.iloc[0]}")
        suggested = int(row.suggested_qty.iloc[0])
        final = suggested if d.action == "approve" else (0 if d.action == "reject" else d.final_qty)
        guard = max(10 * suggested, 50)
        if final > guard:
            REJECTED.labels("override_guard").inc()
            raise HTTPException(422, detail=f"final_qty {final} exceeds the guard of {guard} (10x suggestion)")
        rid = audit.record(user_name=user.name, user_role=user.role, store_id=d.store_id, item_id=d.item_id,
                           order_date=d.order_date, action=d.action, suggested_qty=suggested, final_qty=final,
                           reason_code=d.reason_code, note=d.note, model_version=state.model_version,
                           request_id=getattr(request.state, "request_id", None))
        DECISIONS.labels(d.action).inc()
        return {"id": rid, "status": d.action, "suggested_qty": suggested, "final_qty": final}

    @app.get("/v1/audit", tags=["orders"])
    def audit_log(store_id: str | None = Query(default=None, pattern=r"^S\d{2}$"),
                  limit: int = Query(default=200, ge=1, le=5000), user: User = Depends(current_user)):
        if user.role == "store_manager":
            store_id = user.store_id
        return audit.query(store_id, limit)

    @app.get("/v1/evaluation", tags=["evaluation"])
    def evaluation():
        m = state.metrics
        keep = ["overall", "pinball", "coverage", "coverage_by_group", "stability", "coldstart", "sensing_days",
                "store_mase", "vs_true_demand", "mase_intermittent", "routing", "reconciliation",
                "importance_by_family", "mean_abs_log_driver", "by_day_type", "by_class", "dates", "runtime_seconds"]
        out = {k: m.get(k) for k in keep}
        out["gates"] = state._js("gates.json")
        return out

    @app.get("/v1/simulation", tags=["evaluation"])
    def simulation():
        return state._js("simulation.json")

    @app.get("/v1/simulation/trace", tags=["evaluation"])
    def simulation_trace():
        return records(state._pq("simulation_trace.parquet"))

    @app.get("/v1/monitoring", tags=["ops"])
    def monitoring():
        return state._js("monitoring.json") | {"faults": state._js("faults.json")}

    return app


app = create_app()
