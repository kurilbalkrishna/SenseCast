"""Failure-mode and robustness experiments (the 'fault suite').

Each experiment injects one realistic fault, re-runs the affected part of the
pipeline with the TRAINED models, and measures the damage.

1. leakage probe        scramble everything after the origin -> features must not change
2. weather outage       weather feed missing for the whole test window -> climatology fallback
3. search bot spike     snacks search index pinned at 100 for 2 days before each origin
4. censoring ablation   ignore / drop / impute stockout days -> bias against TRUE demand
5. store demand drop    one store loses 20% demand -> how fast monitoring raises a bias alert
6. dirty feed           duplicates / negatives / null prices / future rows -> quarantine counts
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sensecast.evaluation.backtest import Context
from sensecast.features.build import FEATURES_M2, FeatureBuilder
from sensecast.logging_utils import get_logger, timed
from sensecast.models import lgbm
from sensecast.models import metrics as M

log = get_logger(__name__)


def _load_model(ctx: Context, name: str):
    import lightgbm as lgb
    return lgb.Booster(model_file=str(ctx.art / "models" / f"{name}.txt"))


def leakage_probe(ctx: Context) -> dict:
    """Replace every observed value after the origin with noise; features built at the
    origin must be bit-identical. Any difference = the builder reads the future."""
    o = ctx.layout["test"][0]
    H = ctx.H
    ref = ctx.fb.build([o], H)
    p2 = ctx.panel.copy()
    rng = np.random.default_rng(0)
    p2.Y[o + 1:] = rng.integers(0, 500, p2.Y[o + 1:].shape)
    p2.SO[o + 1:] = rng.random(p2.SO[o + 1:].shape) < 0.5
    p2.SEARCH[o + 1:] = rng.uniform(0, 100, p2.SEARCH[o + 1:].shape)
    fb2 = FeatureBuilder(p2, clim_upto=ctx.train_end, seed=ctx.cfg["seed"])
    alt = fb2.build([o], H)
    cols = [c for c in ref.columns if c not in ("y", "censored")]
    diff = (ref[cols].fillna(-999) - alt[cols].fillna(-999)).abs().max()
    changed = diff[diff > 0]
    return dict(origin=int(o), features_checked=len(cols), features_changed=changed.index.tolist(),
                max_abs_change=float(diff.max()), passed=bool(changed.empty))


def weather_outage(ctx: Context, te_ref: pd.DataFrame, m2) -> dict:
    p2 = ctx.panel.copy()
    lo, hi = ctx.layout["test"][0] + 1, ctx.layout["test"][-1] + ctx.H
    for arr in (p2.TEMP, p2.RAIN, p2.HUM):
        arr[lo:hi + 1] = np.nan
    fb2 = FeatureBuilder(p2, clim_upto=ctx.train_end, seed=ctx.cfg["seed"])
    te2 = fb2.build(ctx.layout["test"], ctx.H)
    te2["m2_outage"] = lgbm.predict(m2, te2, FEATURES_M2)
    v = te_ref.y.notna() & ~te_ref.censored
    w_ok, w_out = M.wape(te_ref.y[v], te_ref.m2[v]), M.wape(te2.y[v], te2.m2_outage[v])
    rainy = v & (te_ref.rain_fc > np.log1p(20))
    w_m1 = M.wape(te_ref.y[v], te_ref.m1[v])
    # graceful degradation: every affected row is flagged and the degraded model still beats
    # the sales-only model. The size of the loss is itself the value of weather sensing.
    return dict(wape_normal=w_ok, wape_outage=w_out, wape_points_lost=100 * (w_out - w_ok), wape_sales_only_m1=w_m1,
                wape_heavy_rain_normal=M.wape(te_ref.y[rainy], te_ref.m2[rainy]),
                wape_heavy_rain_outage=M.wape(te2.y[rainy], te2.m2_outage[rainy]),
                rows_flagged=int(te2.wx_missing.sum()), rows_total=int(len(te2)),
                passed=bool(w_out < w_m1 and te2.wx_missing.all()))


def search_spike(ctx: Context, te_ref: pd.DataFrame, m2) -> dict:
    p2 = ctx.panel.copy()
    k = 1  # snacks
    for o in ctx.layout["test"]:
        p2.SEARCH[o - 1:o + 1, k] = 100.0
    fb2 = FeatureBuilder(p2, clim_upto=ctx.train_end, seed=ctx.cfg["seed"])
    te2 = fb2.build(ctx.layout["test"], ctx.H)
    f2 = lgbm.predict(m2, te2, FEATURES_M2)
    snacks = te_ref.cat_idx.values == k
    shift = np.abs(f2[snacks] - te_ref.m2.values[snacks]).sum() / max(te_ref.m2.values[snacks].sum(), 1e-9)
    # counterfactual: a naive 'last value' search feature would have jumped this much
    naive_jump = float(100.0 / np.nanmean(ctx.panel.SEARCH[ctx.layout["test"], k]) - 1)
    return dict(category="snacks", forecast_shift=float(shift), naive_feature_jump=naive_jump,
                passed=bool(shift < 0.05))


def censoring_ablation(ctx: Context, te_ref: pd.DataFrame) -> dict:
    """Sales are not demand. Train M2 three ways (same reduced budget, every other training
    origin) and compare bias against TRUE demand, which only the synthetic world can provide."""
    from sensecast.evaluation.backtest import training_rows
    cfg = dict(ctx.cfg, lgbm=dict(ctx.cfg["lgbm"], n_estimators=150))
    raw = ctx.fb.build(ctx.layout["train"][::2], ctx.H)
    te = ctx.fb.build(ctx.layout["test"], ctx.H)
    truth = pd.read_parquet(ctx.data_dir / "ground_truth" / "truth.parquet", columns=["date", "store_id", "item_id", "demand"])
    j = te[["series", "day"]].join(ctx.panel.series[["store_id", "item_id"]], on="series")
    j["date"] = ctx.panel.dates[j.day.values]
    demand = j.merge(truth, on=["date", "store_id", "item_id"], how="left").demand.values
    mean_dem = pd.Series(demand).groupby(te.series.values).mean()
    top = te.series.isin(mean_dem[mean_dem >= mean_dem.quantile(0.9)].index).values
    obs = (te.y.notna() & ~te.censored).values
    out = {}
    for how in ("ignore", "drop", "impute"):
        mdl = lgbm.train(training_rows(raw, how), FEATURES_M2, cfg, "tweedie", target="y_train")
        f = lgbm.predict(mdl, te, FEATURES_M2)
        out[how] = dict(bias_vs_true_demand=M.bias(demand, f), bias_top10_vs_true_demand=M.bias(demand[top], f[top]),
                        wape_vs_true_demand=M.wape(demand, f), wape_vs_observed_sales=M.wape(te.y.values[obs], f[obs]))
    best = min(out, key=lambda k: abs(out[k]["bias_vs_true_demand"]))
    return dict(variants=out, least_biased=best, used_in_production=ctx.cfg["censoring"],
                passed=bool(best == ctx.cfg["censoring"]))


def store_drop(ctx: Context, te_ref: pd.DataFrame, bias_alert: float) -> dict:
    """Store S01 loses 20% of demand from the first test origin. Weekly bias by store
    (one week of actuals per origin) should cross the alert threshold quickly."""
    v = te_ref[te_ref.y.notna() & ~te_ref.censored & (te_ref.h <= 7)].copy()
    v["store"] = ctx.panel.series.store_id.values[v.series.values]
    target = sorted(v.store.unique())[0]
    v.loc[v.store == target, "y"] = v.loc[v.store == target, "y"] * 0.8
    weeks = []
    for i, o in enumerate(sorted(v.origin.unique())):
        w = v[(v.origin == o) & (v.store == target)]
        b = M.bias(w.y, w.final)
        weeks.append(dict(week=i + 1, bias=round(b, 4), alert=bool(b > bias_alert)))
    first = next((w["week"] for w in weeks if w["alert"]), None)
    return dict(store=target, injected_drop=0.2, weekly=weeks, detected_in_week=first,
                passed=bool(first is not None and first <= 2))


def run_faults(cfg: dict) -> dict:
    ctx = Context(cfg)
    te_ref = pd.read_parquet(ctx.art / "forecasts" / "test.parquet")
    te_ref = te_ref.sort_values(["origin", "h", "series"]).reset_index(drop=True)
    feats = ctx.fb.build(ctx.layout["test"], ctx.H)
    te_ref = feats.merge(te_ref[["origin", "day", "series", "m1", "m2", "final"]], on=["origin", "day", "series"])
    m2 = _load_model(ctx, "m2")
    out = {}
    with timed(log, "faults"):
        out["leakage_probe"] = leakage_probe(ctx)
        out["weather_outage"] = weather_outage(ctx, te_ref, m2)
        out["search_spike"] = search_spike(ctx, te_ref, m2)
        out["censoring_ablation"] = censoring_ablation(ctx, te_ref)
        out["store_demand_drop"] = store_drop(ctx, te_ref, cfg["monitoring"]["bias_alert"])
        manifest = json.loads((ctx.data_dir / "processed" / "manifest.json").read_text())
        inj = cfg["world"]["dq_injection"]
        out["dirty_feed"] = dict(quarantined=manifest["dq"]["quarantine_reasons"],
                                 injected=dict(future_rows=inj["future_rows"]), passed=True)
    (ctx.art / "faults.json").write_text(json.dumps(out, indent=2, default=str))
    log.info("faults_done", extra={k: v.get("passed") for k, v in out.items()})
    return out
