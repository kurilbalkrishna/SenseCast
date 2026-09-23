import numpy as np
import pytest

from sensecast.models.intermittent import classify, croston_family


def test_classification_quadrants():
    T = 200
    smooth = np.full(T, 10.0)
    erratic = np.where(np.arange(T) % 2, 2.0, 30.0)
    intermittent = np.where(np.arange(T) % 4 == 0, 5.0, 0.0)
    lumpy = np.where(np.arange(T) % 4 == 0, np.where(np.arange(T) % 8 == 0, 1.0, 40.0), 0.0)
    Y = np.column_stack([smooth, erratic, intermittent, lumpy])
    cls = classify(Y, upto=T - 1).demand_class.tolist()
    assert cls == ["smooth", "erratic", "intermittent", "lumpy"]


def test_new_series_flagged():
    Y = np.full((100, 1), np.nan)
    Y[90:] = 3
    assert classify(Y, upto=99).demand_class.iloc[0] == "new"


def test_croston_and_sba_relationship():
    y = np.array([0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4], dtype=float)[:, None]
    out = croston_family(y, alpha=0.1)
    # steady pattern: size 4 every 3 days -> Croston ~ 4/3
    assert out["croston"][-1, 0] == pytest.approx(4 / 3, rel=0.05)
    assert out["sba"][-1, 0] == pytest.approx(out["croston"][-1, 0] * 0.95)


def test_tsb_decays_when_item_dies():
    y = np.concatenate([np.full(50, 2.0), np.zeros(100)])[:, None]
    out = croston_family(y, alpha=0.1, beta=0.1)
    assert out["tsb"][49, 0] > 1.5
    assert out["tsb"][-1, 0] < 0.01          # TSB fades to zero
    assert out["croston"][-1, 0] > 1.0       # Croston never updates on zeros: stale


def test_nan_days_do_not_update_state():
    y = np.array([2, 2, np.nan, np.nan, 2], dtype=float)[:, None]
    out = croston_family(y)
    assert out["tsb"][2, 0] == out["tsb"][1, 0]
