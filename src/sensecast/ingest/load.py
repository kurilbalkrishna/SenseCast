"""Ingestion: validate raw feeds, quarantine bad rows, derive stockout flags,
write processed Parquet plus a manifest with content hashes (data versioning).
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors

from sensecast.config import resolve_path
from sensecast.ingest import schemas
from sensecast.logging_utils import get_logger, timed

log = get_logger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_sales(raw: pd.DataFrame, max_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean, quarantined). Every quarantined row carries a ``reason``."""
    df = raw.copy()
    reasons = pd.Series("", index=df.index, dtype=object)

    def flag(mask, why):
        nonlocal reasons
        mask = mask & (reasons == "")
        reasons = reasons.mask(mask, why)

    flag(df.duplicated(["date", "store_id", "item_id"], keep="first"), "duplicate_key")
    flag(df["date"] > max_date, "date_in_future")
    flag(df["units_sold"] < 0, "negative_units")
    flag(df["unit_price"].isna() | (df["unit_price"] <= 0), "bad_price")
    flag(df[["on_hand_open", "receipts", "on_hand_close"]].lt(0).any(axis=1), "negative_stock")
    balance = df.on_hand_open + df.receipts - df.units_sold.clip(lower=0) != df.on_hand_close
    flag(balance & (df.units_sold >= 0), "stock_balance")

    quarantine = df[reasons != ""].assign(reason=reasons[reasons != ""])
    clean = df[reasons == ""]
    # schema is the final gate; lazy=True reports every failure at once
    try:
        clean = schemas.sales_schema(max_date).validate(clean, lazy=True)
    except pandera.errors.SchemaErrors as err:  # pragma: no cover - defensive
        bad_idx = err.failure_cases["index"].dropna().unique()
        quarantine = pd.concat([quarantine, clean.loc[bad_idx].assign(reason="schema")])
        clean = schemas.sales_schema(max_date).validate(clean.drop(index=bad_idx))
    return clean.reset_index(drop=True), quarantine.reset_index(drop=True)


def ingest(cfg: dict) -> dict:
    data_dir = resolve_path(cfg, "data")
    raw, proc, quar = data_dir / "raw", data_dir / "processed", data_dir / "quarantine"
    proc.mkdir(exist_ok=True)
    quar.mkdir(exist_ok=True)
    end = pd.Timestamp(cfg["world"]["end"])

    with timed(log, "ingest"):
        stores = schemas.STORES.validate(pd.read_csv(raw / "stores.csv"))
        items = schemas.ITEMS.validate(pd.read_csv(raw / "items.csv", parse_dates=["launch_date"]))
        events = schemas.EVENTS.validate(pd.read_csv(raw / "events.csv", parse_dates=["date"]))
        promos = schemas.PROMOTIONS.validate(pd.read_csv(raw / "promotions.csv", parse_dates=["date_start", "date_end"]))
        weather = schemas.WEATHER.validate(pd.read_parquet(raw / "weather.parquet"))
        wfc = pd.read_parquet(raw / "weather_forecast.parquet")
        search = schemas.SEARCH.validate(pd.read_csv(raw / "search.csv", parse_dates=["date"]))

        sales_raw = pd.read_parquet(raw / "sales.parquet")
        sales, bad = validate_sales(sales_raw, end)

        # referential integrity
        orphan = ~sales.item_id.isin(items.item_id) | ~sales.store_id.isin(stores.store_id)
        if orphan.any():
            bad = pd.concat([bad, sales[orphan].assign(reason="unknown_key")], ignore_index=True)
            sales = sales[~orphan]

        # Stockout = shelf empty at close. On these days sales under-count demand
        # (censoring): history features mask them and training uses the configured
        # censoring treatment (default: demand unconstraining, see evaluation/backtest.py).
        sales["stockout"] = (sales.on_hand_close == 0).astype(np.int8)
        sales = sales.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

        outputs = {
            "sales.parquet": sales, "stores.parquet": stores, "items.parquet": items, "events.parquet": events,
            "promotions.parquet": promos, "weather.parquet": weather, "weather_forecast.parquet": wfc,
            "search.parquet": search,
        }
        for name, df in outputs.items():
            df.to_parquet(proc / name, index=False)
        bad.to_parquet(quar / "sales_quarantine.parquet", index=False)

    report = dict(
        ingested_at=datetime.now(UTC).isoformat(),
        rows_raw=int(len(sales_raw)),
        rows_clean=int(len(sales)),
        rows_quarantined=int(len(bad)),
        quarantine_reasons={k: int(v) for k, v in bad.reason.value_counts().items()},
        stockout_share=float(sales.stockout.mean()),
        date_range=[str(sales.date.min().date()), str(sales.date.max().date())],
        weather_rows_missing=int(weather[["temp_max", "rain_mm"]].isna().any(axis=1).sum()),
    )
    manifest = {
        "files": {p.name: sha256_file(p) for p in sorted(proc.glob("*.parquet"))},
        "raw_files": {p.name: sha256_file(p) for p in sorted(raw.glob("*")) if p.is_file()},
        "dq": report,
    }
    gen = raw / "_generation.json"
    if gen.exists():
        manifest["generation"] = json.loads(gen.read_text())
    (proc / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("ingest_done", extra=report)
    return report
