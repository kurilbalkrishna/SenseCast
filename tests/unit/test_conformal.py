import numpy as np
import pandas as pd
import pytest

from sensecast.models.conformal import CQR, horizon_bucket
from sensecast.models.metrics import coverage


def _data(n, rng):
    y = rng.normal(0, 1, n)
    # deliberately too-narrow raw interval (~50% coverage instead of 80%)
    return y, np.full(n, -0.67), np.full(n, 0.67)


def test_cqr_restores_nominal_coverage():
    rng = np.random.default_rng(0)
    y_cal, lo_cal, hi_cal = _data(5000, rng)
    y_te, lo_te, hi_te = _data(5000, rng)
    g_cal = pd.DataFrame(dict(g=["a"] * 5000))
    raw = coverage(y_te, lo_te, hi_te)
    cqr = CQR(alpha=0.2).fit(y_cal, lo_cal, hi_cal, g_cal)
    lo, hi = cqr.apply(lo_te, hi_te, g_cal)
    assert raw < 0.55
    # apply() clips at 0 (demand cannot be negative); undo for this symmetric test
    lo = lo_te - cqr.q["a",]
    assert coverage(y_te, lo, hi) == pytest.approx(0.80, abs=0.02)


def test_small_groups_fall_back_to_global():
    rng = np.random.default_rng(1)
    y, lo, hi = _data(200, rng)
    groups = pd.DataFrame(dict(g=["big"] * 190 + ["tiny"] * 10))
    cqr = CQR(0.2).fit(y, lo, hi, groups)
    assert cqr.q[("tiny",)] == cqr.q_global


def test_horizon_bucket():
    assert list(horizon_bucket(np.array([1, 3, 4, 7, 8, 14]))) == ["h1-3", "h1-3", "h4-7", "h4-7", "h8-14", "h8-14"]
