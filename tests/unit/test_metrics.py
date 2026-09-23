import numpy as np
import pandas as pd
import pytest

from sensecast.models import metrics as M


def test_wape_known_value():
    assert M.wape([10, 0, 10], [8, 1, 12]) == pytest.approx(5 / 20)


def test_wape_all_zero_is_nan():
    assert np.isnan(M.wape([0, 0], [1, 1]))


def test_bias_sign():
    assert M.bias([10, 10], [12, 12]) == pytest.approx(0.2)
    assert M.bias([10, 10], [8, 8]) == pytest.approx(-0.2)


def test_pinball_asymmetry():
    # under-forecast at tau=0.9 costs 9x an equal over-forecast
    under = M.pinball([10], [9], 0.9)
    over = M.pinball([10], [11], 0.9)
    assert under == pytest.approx(0.9)
    assert over == pytest.approx(0.1)


def test_coverage():
    assert M.coverage([1, 5, 9], [0, 0, 0], [5, 5, 5]) == pytest.approx(2 / 3)


def test_mase_perfect_forecast_is_zero():
    df = pd.DataFrame(dict(series=[0, 0, 1, 1], y=[1.0, 2, 3, 4], f=[1.0, 2, 3, 4]))
    assert M.mase(df, "f", np.array([1.0, 1.0])) == 0


def test_mase_scale_uses_weekly_naive():
    Y = np.tile(np.arange(7, dtype=float), 4)[:, None]  # perfectly weekly -> naive error 0 -> fallback
    Y2 = np.arange(28, dtype=float)[:, None]              # trend: |y_t - y_t-7| = 7
    sc = M.mase_scale(np.hstack([Y, Y2]), upto=27)
    assert sc[1] == pytest.approx(7.0)
    assert np.isfinite(sc[0])


def test_stability_zero_when_forecasts_do_not_change():
    df = pd.DataFrame(dict(series=[0, 0], day=[5, 5], origin=[1, 2], f=[3.0, 3.0]))
    assert M.stability(df, "f") == 0
