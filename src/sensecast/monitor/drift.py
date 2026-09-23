"""Model and data monitoring.

* PSI (population stability index) per model feature: reference = training block,
  current = test block. PSI > 0.2 is the usual 'investigate' threshold.
* Rolling accuracy: WAPE and bias per forecast origin (weekly), overall and per store.
* Data freshness: last date seen per source.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sensecast.config import resolve_path
from sensecast.evaluation.backtest import Context
from sensecast.features.build import FEATURES_M2
from sensecast.logging_utils import get_logger, timed
from sensecast.models import metrics as M

log = get_logger(__name__)


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def run_monitoring(cfg: dict) -> dict:
    ctx = Context(cfg)
    art = resolve_path(cfg, "artifacts")
    mon = cfg["monitoring"]
    with timed(log, "monitor"):
        ref = ctx.fb.build(ctx.layout["train"][-26:], ctx.H).sample(frac=0.5, random_state=cfg["seed"])
        cur = ctx.fb.build(ctx.layout["test"], ctx.H)
        # calendar and event features follow a known schedule (a July test window SHOULD differ
        # from a winter reference), so drift is measured on data-driven features only
        skip = {"store_idx", "cat_idx", "region_idx", "h", "dow", "dom", "month", "woy",
                "days_to_event", "event_id", "in_festival", "is_holiday", "age"}
        drift = {f: round(psi(ref[f].values, cur[f].values), 4) for f in FEATURES_M2 if f not in skip}
        alerts = sorted([f for f, v in drift.items() if v > mon["psi_alert"]])

        te = pd.read_parquet(art / "forecasts" / "test.parquet")
        v = te[te.y.notna() & ~te.censored]
        weekly = (v.groupby("origin_date")
                  .apply(lambda g: pd.Series(dict(wape_final=M.wape(g.y, g.final), wape_b0=M.wape(g.y, g.b0),
                                                  bias_final=M.bias(g.y, g.final))), include_groups=False)
                  .reset_index())
        weekly["origin_date"] = weekly.origin_date.astype(str)
        store_bias = (v.groupby(["store_id", "origin_date"])
                      .apply(lambda g: M.bias(g.y, g.final), include_groups=False).rename("bias").reset_index())
        store_bias["alert"] = store_bias.bias.abs() > mon["bias_alert"]
        store_bias["origin_date"] = store_bias.origin_date.astype(str)

        proc = ctx.data_dir / "processed"
        freshness = {}
        for name, col in (("sales", "date"), ("weather", "date"), ("search", "date")):
            freshness[name] = str(pd.read_parquet(proc / f"{name}.parquet", columns=[col])[col].max().date())

    out = dict(psi=drift, psi_alerts=alerts, psi_threshold=mon["psi_alert"], weekly=weekly.to_dict(orient="records"),
               store_bias=store_bias.to_dict(orient="records"),
               store_bias_alerts=int(store_bias.alert.sum()), freshness=freshness)
    (art / "monitoring.json").write_text(json.dumps(out, indent=2, default=str))
    log.info("monitoring_done", extra=dict(psi_alerts=alerts, store_bias_alerts=out["store_bias_alerts"]))
    return out
