"""Festival and public-holiday calendar for Maharashtra retail.

Public holidays come from the ``holidays`` package (India). Regional festivals the
package does not cover (Ganesh Chaturthi, Navratri, Raksha Bandhan, Eid al-Fitr,
Gudi Padwa, Makar Sankranti) are curated below. Verify future years against an
official panchang before relying on them.
"""
from __future__ import annotations

import holidays
import pandas as pd

# Curated regional festival dates (first day of the festival).
CURATED = {
    "Ganesh Chaturthi": ["2023-09-19", "2024-09-07", "2025-08-27", "2026-09-14"],
    "Navratri": ["2023-10-15", "2024-10-03", "2025-09-22", "2026-10-11"],
    "Raksha Bandhan": ["2023-08-30", "2024-08-19", "2025-08-09", "2026-08-28"],
    "Eid al-Fitr": ["2023-04-22", "2024-04-11", "2025-03-31", "2026-03-21"],
    "Gudi Padwa": ["2023-03-22", "2024-04-09", "2025-03-30", "2026-03-19"],
    "Makar Sankranti": ["2023-01-15", "2024-01-15", "2025-01-14", "2026-01-14"],
}

# Names from the holidays package we keep, mapped to our canonical event names.
FROM_LIBRARY = {
    "Diwali": "Diwali",
    "Holi": "Holi",
    "Dussehra": "Dussehra",
    "Christmas": "Christmas",
    "Independence Day": "Independence Day",
    "Republic Day": "Republic Day",
    "Janmashtami": "Janmashtami",
}

# Event metadata: festival span (days the festival lasts), buildup (shopping days
# before), and ground-truth log-uplift per category. Ground truth is used ONLY by
# the synthetic world generator; models never read it.
EVENT_META = {
    "Diwali": dict(span=5, buildup=7, uplift=dict(beverages=0.40, snacks=0.90, personal_care=0.50, monsoon_seasonal=0.0)),
    "Ganesh Chaturthi": dict(span=10, buildup=4, uplift=dict(beverages=0.40, snacks=0.70, personal_care=0.20, monsoon_seasonal=0.05)),
    "Navratri": dict(span=9, buildup=3, uplift=dict(beverages=0.20, snacks=0.30, personal_care=0.30, monsoon_seasonal=0.0)),
    "Holi": dict(span=2, buildup=4, uplift=dict(beverages=0.40, snacks=0.30, personal_care=0.40, monsoon_seasonal=0.0)),
    "Raksha Bandhan": dict(span=1, buildup=4, uplift=dict(beverages=0.10, snacks=0.50, personal_care=0.10, monsoon_seasonal=0.0)),
    "Eid al-Fitr": dict(span=2, buildup=4, uplift=dict(beverages=0.30, snacks=0.40, personal_care=0.15, monsoon_seasonal=0.0)),
    "Christmas": dict(span=2, buildup=5, uplift=dict(beverages=0.30, snacks=0.30, personal_care=0.10, monsoon_seasonal=0.0)),
    "Gudi Padwa": dict(span=1, buildup=3, uplift=dict(beverages=0.10, snacks=0.30, personal_care=0.10, monsoon_seasonal=0.0)),
    "Makar Sankranti": dict(span=1, buildup=3, uplift=dict(beverages=0.05, snacks=0.30, personal_care=0.05, monsoon_seasonal=0.0)),
    "Dussehra": dict(span=1, buildup=3, uplift=dict(beverages=0.10, snacks=0.30, personal_care=0.10, monsoon_seasonal=0.0)),
    "Janmashtami": dict(span=1, buildup=2, uplift=dict(beverages=0.05, snacks=0.20, personal_care=0.05, monsoon_seasonal=0.0)),
    "Independence Day": dict(span=1, buildup=1, uplift=dict(beverages=0.10, snacks=0.10, personal_care=0.0, monsoon_seasonal=0.0)),
    "Republic Day": dict(span=1, buildup=1, uplift=dict(beverages=0.10, snacks=0.10, personal_care=0.0, monsoon_seasonal=0.0)),
}

# Regional intensity: Ganesh Chaturthi is much bigger in Mumbai and Pune.
REGIONAL = {("Ganesh Chaturthi", "Mumbai"): 1.3, ("Ganesh Chaturthi", "Pune"): 1.3}

EVENT_NAMES = sorted(EVENT_META)


def build_event_calendar(start: str, end: str) -> pd.DataFrame:
    """Return one row per event start date inside [start - 30d, end + 30d]."""
    s, e = pd.Timestamp(start) - pd.Timedelta(days=30), pd.Timestamp(end) + pd.Timedelta(days=30)
    rows = []
    years = list(range(s.year, e.year + 1))
    for day, name in holidays.India(years=years).items():
        for key, canon in FROM_LIBRARY.items():
            if key in name:
                rows.append((pd.Timestamp(day), canon, "public_holiday", "holidays-lib"))
    for canon, dates in CURATED.items():
        for d in dates:
            rows.append((pd.Timestamp(d), canon, "festival", "curated"))
    cal = pd.DataFrame(rows, columns=["date", "event", "kind", "source"]).drop_duplicates(["date", "event"])
    cal = cal[(cal.date >= s) & (cal.date <= e)].sort_values("date").reset_index(drop=True)
    cal["span_days"] = cal.event.map(lambda n: EVENT_META[n]["span"])
    cal["buildup_days"] = cal.event.map(lambda n: EVENT_META[n]["buildup"])
    return cal


def event_day_table(cal: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-date known-in-advance event features.

    Columns: ``days_to_event`` (signed days to the nearest event start within
    [-span, +14], else 99), ``event_id`` (index into EVENT_NAMES or -1),
    ``in_festival`` (1 while the festival runs), ``is_holiday``.
    """
    n = len(dates)
    out = pd.DataFrame(index=dates)
    out["days_to_event"] = 99
    out["event_id"] = -1
    out["in_festival"] = 0
    out["is_holiday"] = 0
    best = pd.Series(999, index=dates)
    for _, ev in cal.iterrows():
        delta = (ev.date - dates).days.values  # >0: event in future
        window = (delta <= 14) & (delta >= -(ev.span_days - 1))
        closer = window & (abs(delta) < best.values)
        out.loc[closer, "days_to_event"] = delta[closer]
        out.loc[closer, "event_id"] = EVENT_NAMES.index(ev.event)
        best[closer] = abs(delta[closer])
        running = (delta <= 0) & (delta >= -(ev.span_days - 1))
        out.loc[running, "in_festival"] = 1
        if ev.kind == "public_holiday":
            out.loc[delta == 0, "is_holiday"] = 1
    assert len(out) == n
    return out
