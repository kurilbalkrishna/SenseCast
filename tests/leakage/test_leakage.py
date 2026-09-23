"""Point-in-time guarantees. If any of these fail, every accuracy number is suspect."""
import numpy as np

from sensecast.features.build import FeatureBuilder

META = {"y", "censored"}


def _features(ctx, panel, origin):
    fb = FeatureBuilder(panel, clim_upto=ctx.train_end, seed=ctx.cfg["seed"])
    return fb.build([origin], ctx.H)


def test_scrambling_the_future_changes_no_feature(ctx):
    for o in (ctx.layout["cal"][0], ctx.layout["test"][-1]):
        ref = _features(ctx, ctx.panel, o)
        p2 = ctx.panel.copy()
        rng = np.random.default_rng(o)
        p2.Y[o + 1:] = rng.integers(0, 999, p2.Y[o + 1:].shape)
        p2.SO[o + 1:] = True
        p2.SEARCH[o + 1:] = 100
        alt = _features(ctx, p2, o)
        cols = [c for c in ref.columns if c not in META]
        diff = (ref[cols].fillna(-1) - alt[cols].fillna(-1)).abs().max()
        assert diff.max() == 0, f"features read the future: {diff[diff > 0].index.tolist()}"


def test_target_is_not_a_feature(ctx):
    o = ctx.layout["test"][0]
    ref = _features(ctx, ctx.panel, o)
    p2 = ctx.panel.copy()
    p2.Y[o + 1:o + ctx.H + 1] *= 10
    alt = _features(ctx, p2, o)
    assert np.allclose(ref.drop(columns=list(META)).fillna(-1).values, alt.drop(columns=list(META)).fillna(-1).values)
    assert not np.allclose(ref.y.fillna(0), alt.y.fillna(0))


def test_weather_is_a_forecast_whose_error_grows_with_horizon(ctx):
    o = ctx.layout["test"][0]
    df = _features(ctx, ctx.panel, o)
    city = ctx.panel.series.city_idx.values[df.series.values]
    actual = ctx.panel.TEMP[df.day.values, city]
    err = df.temp_fc.values - actual
    sd = {h: err[df.h.values == h].std() for h in (1, 14)}
    assert sd[1] > 0, "day-ahead weather must not equal the observed value"
    assert sd[14] > 2 * sd[1]


def test_same_weekday_lag_never_after_origin(ctx):
    df = _features(ctx, ctx.panel, ctx.layout["test"][0])
    k0 = np.ceil(df.h.values / 7)
    assert (df.day.values - 7 * k0 <= df.origin.values).all()


def test_training_rows_do_not_overlap_later_blocks(ctx):
    H = ctx.H
    assert max(ctx.layout["train"]) + H < min(ctx.layout["cal"])
    assert max(ctx.layout["cal"]) + H < min(ctx.layout["test"])
