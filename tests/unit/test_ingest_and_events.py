import numpy as np
import pandas as pd
import pytest

from sensecast.generate.events import EVENT_NAMES, build_event_calendar, event_day_table
from sensecast.ingest.load import validate_sales


def _feed():
    return pd.DataFrame(dict(
        date=pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-02-01", "2026-01-05"]),
        store_id=["S01"] * 7,
        item_id=["I001", "I001", "I001", "I001", "I001", "I001", "I001"],
        units_sold=[3, 3, -2, 4, 2, 1, 5],
        unit_price=[10.0, 10.0, 10.0, np.nan, 10.0, 10.0, 10.0],
        on_hand_open=[10, 10, 7, 7, 3, 1, 5],
        receipts=[0, 0, 0, 0, 0, 0, 0],
        on_hand_close=[7, 7, 7, 3, 1, 0, 9],   # last row breaks the stock balance
    ))


def test_validate_sales_quarantines_every_defect():
    clean, bad = validate_sales(_feed(), max_date=pd.Timestamp("2026-01-31"))
    reasons = bad.reason.value_counts().to_dict()
    assert reasons == {"duplicate_key": 1, "negative_units": 1, "bad_price": 1, "date_in_future": 1, "stock_balance": 1}
    assert len(clean) == 2
    assert (clean.units_sold >= 0).all()


def test_event_calendar_contains_regional_festivals():
    cal = build_event_calendar("2025-01-01", "2026-12-31")
    names = set(cal.event)
    assert {"Diwali", "Ganesh Chaturthi", "Navratri", "Holi", "Raksha Bandhan"} <= names


def test_event_features_follow_the_date():
    """If a festival moves (e.g. Diwali in Oct one year, Nov the next), features move with it."""
    dates = pd.date_range("2025-10-01", "2025-11-30")
    cal = pd.DataFrame(dict(date=[pd.Timestamp("2025-10-20")], event=["Diwali"], kind=["festival"],
                            source=["test"], span_days=[5], buildup_days=[7]))
    t1 = event_day_table(cal, dates)
    cal2 = cal.assign(date=[pd.Timestamp("2025-11-08")])
    t2 = event_day_table(cal2, dates)
    assert t1.loc["2025-10-17", "days_to_event"] == 3
    assert t2.loc["2025-11-05", "days_to_event"] == 3
    assert t2.loc["2025-10-17", "days_to_event"] == 99
    assert t1.loc["2025-10-21", "in_festival"] == 1
    assert t1.loc["2025-10-20", "event_id"] == EVENT_NAMES.index("Diwali")


@pytest.mark.parametrize("bad_mix", [{"beverages": 1}])
def test_config_rejects_bad_category_mix(bad_mix):
    from sensecast.config import load_config

    with pytest.raises(ValueError):
        load_config(None, world={"category_mix": bad_mix})
