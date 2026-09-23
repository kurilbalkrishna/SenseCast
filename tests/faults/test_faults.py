"""Failure-mode behaviour of the feature layer and the fault suite results."""
import json
from pathlib import Path

import numpy as np

from sensecast.features.build import FEATURES_M2, FeatureBuilder, history_stats


def test_weather_outage_falls_back_to_climatology(ctx):
    p2 = ctx.panel.copy()
    o = ctx.layout["test"][0]
    for arr in (p2.TEMP, p2.RAIN, p2.HUM):
        arr[o + 1:] = np.nan
    df = FeatureBuilder(p2, ctx.train_end, seed=1).build([o], ctx.H)
    assert df.wx_missing.eq(1).all()
    assert np.isfinite(df[["temp_fc", "rain_fc", "hum_fc"]].values).all()


def test_search_spike_is_damped_by_median_features(ctx):
    p2 = ctx.panel.copy()
    o = ctx.layout["test"][0]
    before = FeatureBuilder(ctx.panel, ctx.train_end).srch["search_o"][o, 1]
    p2.SEARCH[o, 1] = 100.0  # one-day bot spike on snacks
    after = FeatureBuilder(p2, ctx.train_end).srch["search_o"][o, 1]
    assert abs(after - before) < 10, "a single-day spike must not move the median-based feature much"


def test_stockout_days_are_masked_from_history(ctx):
    p = ctx.panel
    hs = history_stats(p)
    n = int(np.argmax(p.SO.sum(0)))  # series with most stockouts
    t = int(np.where(p.SO[:, n])[0][-1])
    assert np.isnan(hs["last_obs"][t, n])


def test_features_have_no_infinite_values(ctx):
    df = ctx.fb.build(ctx.layout["test"], ctx.H)
    vals = df[FEATURES_M2].values.astype(float)
    assert not np.isinf(vals).any()


def test_fault_suite_results(pipeline):
    art = Path(pipeline["paths"]["artifacts"])
    f = json.loads((art / "faults.json").read_text())
    assert f["leakage_probe"]["passed"]
    assert f["search_spike"]["passed"]
    assert f["store_demand_drop"]["detected_in_week"] is not None
    assert f["dirty_feed"]["quarantined"]["date_in_future"] == pipeline["world"]["dq_injection"]["future_rows"]
