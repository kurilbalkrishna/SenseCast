"""Dense time x series arrays (the 'feature store' in memory).

Everything downstream reads from a Panel, so point-in-time rules are enforced in
one place (features/build.py) instead of scattered across joins.

Layout: rows are days (index 0 = world start), columns are series. The panel
extends H days past the last observed day so known-future inputs (promotions,
events, issued weather forecasts) can be read for the production origin; the
target and search arrays are NaN there.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sensecast.generate.events import event_day_table

CATEGORIES = ["beverages", "snacks", "personal_care", "monsoon_seasonal"]
LEVELS = ["total", "region", "store", "store_category", "item_store"]


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    n_obs: int
    series: pd.DataFrame           # one row per series (see build_bottom_panel)
    Y: np.ndarray                  # [T, N] observed sales (NaN pre-launch / future)
    SO: np.ndarray                 # [T, N] stockout flag (censored day)
    DISC: np.ndarray               # [T, N] promo discount (known in advance)
    POST: np.ndarray               # [T, N] 1..4 days after a promo ended, else 0
    TEMP: np.ndarray               # [T, C+1] daily max temp; col C = mean over cities
    RAIN: np.ndarray               # [T, C+1]
    HUM: np.ndarray                # [T, C+1]
    SEARCH: np.ndarray             # [T, K+1] search index; col K = mean over categories
    EV: pd.DataFrame               # per-day event features (known in advance)
    cities: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return self.Y.shape[1]

    def copy(self) -> Panel:
        import copy
        return copy.deepcopy(self)


def build_bottom_panel(proc_dir: Path, horizon: int) -> Panel:
    sales = pd.read_parquet(proc_dir / "sales.parquet")
    stores = pd.read_parquet(proc_dir / "stores.parquet")
    items = pd.read_parquet(proc_dir / "items.parquet")
    events = pd.read_parquet(proc_dir / "events.parquet")
    promos = pd.read_parquet(proc_dir / "promotions.parquet")
    weather = pd.read_parquet(proc_dir / "weather.parquet")
    wfc = pd.read_parquet(proc_dir / "weather_forecast.parquet")
    search = pd.read_parquet(proc_dir / "search.parquet")

    start = min(sales.date.min(), items.launch_date.min())
    last_obs = sales.date.max()
    dates = pd.date_range(start, last_obs + pd.Timedelta(days=horizon), freq="D")
    n_obs = int((dates <= last_obs).sum())
    T = len(dates)

    series = stores.assign(_k=1).merge(items.assign(_k=1), on="_k").drop(columns="_k")
    series = series.sort_values(["store_id", "item_id"]).reset_index(drop=True)
    cities = sorted(stores.city.unique())
    series["cat_idx"] = series.category.map(CATEGORIES.index)
    series["city_idx"] = series.city.map(cities.index)
    series["region_idx"] = series["city_idx"]
    series["store_idx"] = series.store_id.map({s: i for i, s in enumerate(sorted(stores.store_id))})
    series["launch_idx"] = ((series.launch_date - start).dt.days).clip(lower=0).astype(int)
    series["level"] = "item_store"
    series["node_id"] = series.store_id + "|" + series.item_id
    N = len(series)

    key = pd.MultiIndex.from_frame(series[["store_id", "item_id"]])
    col = pd.Series(np.arange(N), index=key)
    ti = ((sales.date - start).dt.days).values
    ni = col.reindex(pd.MultiIndex.from_frame(sales[["store_id", "item_id"]])).values
    Y = np.full((T, N), np.nan, dtype=np.float32)
    SO = np.zeros((T, N), dtype=bool)
    Y[ti, ni] = sales.units_sold.values
    SO[ti, ni] = sales.stockout.values.astype(bool)

    DISC = np.zeros((T, N), dtype=np.float32)
    POST = np.zeros((T, N), dtype=np.float32)
    by_item_region = series.groupby(["item_id", "region"]).indices
    for p in promos.itertuples():
        cols = by_item_region.get((p.item_id, p.region))
        if cols is None:
            continue
        a = (p.date_start - start).days
        b = (p.date_end - start).days
        if a >= T:
            continue
        DISC[max(a, 0):min(b, T - 1) + 1, cols] = p.discount
        for j in range(1, 5):
            if 0 <= b + j < T:
                POST[b + j, cols] = np.where(DISC[b + j, cols] == 0, j, POST[b + j, cols])

    def wx(col_name):
        obs = weather.pivot(index="date", columns="city", values=col_name)
        fc = wfc.pivot(index="date", columns="city", values=col_name) if col_name in wfc else None
        full = obs if fc is None else pd.concat([obs, fc[~fc.index.isin(obs.index)]])
        arr = full.reindex(dates)[cities].values.astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.column_stack([arr, np.nanmean(arr, axis=1)])

    TEMP, RAIN, HUM = wx("temp_max"), wx("rain_mm"), wx("humidity")
    sp = search.pivot(index="date", columns="category", values="search_index").reindex(dates)[CATEGORIES]
    sa = np.clip(sp.values.astype(np.float32), 0, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        SEARCH = np.column_stack([sa, np.nanmean(sa, axis=1)])

    EV = event_day_table(events, dates).reset_index(drop=True)
    return Panel(dates, n_obs, series, Y, SO, DISC, POST, TEMP, RAIN, HUM, SEARCH, EV, cities,
                 meta=dict(start=str(start.date()), last_obs=str(last_obs.date())))


def aggregate_nodes(series: pd.DataFrame) -> list[tuple[str, str, np.ndarray, dict]]:
    """Return (level, node_id, bottom column indices, attrs) for every aggregate node,
    ordered total -> region -> store -> store_category."""
    nodes = [("total", "total", np.arange(len(series)), dict(city_idx=-1, cat_idx=-1, store_idx=-1))]
    for r, idx in sorted(series.groupby("region").indices.items()):
        nodes.append(("region", f"R:{r}", idx, dict(city_idx=int(series.city_idx[idx[0]]), cat_idx=-1, store_idx=-1)))
    for s, idx in sorted(series.groupby("store_id").indices.items()):
        nodes.append(("store", f"S:{s}", idx, dict(city_idx=int(series.city_idx[idx[0]]), cat_idx=-1,
                                                   store_idx=int(series.store_idx[idx[0]]))))
    for (s, c), idx in sorted(series.groupby(["store_id", "category"]).indices.items()):
        nodes.append(("store_category", f"SC:{s}|{c}", idx, dict(city_idx=int(series.city_idx[idx[0]]),
                                                                 cat_idx=int(series.cat_idx[idx[0]]),
                                                                 store_idx=int(series.store_idx[idx[0]]))))
    return nodes


def summing_matrix(series: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """S (n_total x n_bottom), node ids (aggregates then bottom), and their levels."""
    nodes = aggregate_nodes(series)
    n_b = len(series)
    S = np.zeros((len(nodes) + n_b, n_b), dtype=np.float64)
    ids, levels = [], []
    for i, (lvl, nid, idx, _) in enumerate(nodes):
        S[i, idx] = 1.0
        ids.append(nid)
        levels.append(lvl)
    S[len(nodes):, :] = np.eye(n_b)
    ids += list(series.node_id)
    levels += ["item_store"] * n_b
    return S, ids, levels


def build_aggregate_panel(bottom: Panel) -> Panel:
    """Panel whose series are the aggregate nodes (sums of bottom series)."""
    nodes = aggregate_nodes(bottom.series)
    T = bottom.T
    Y = np.full((T, len(nodes)), np.nan, dtype=np.float32)
    DISC = np.zeros((T, len(nodes)), dtype=np.float32)
    rows = []
    C = len(bottom.cities)
    for j, (lvl, nid, idx, attrs) in enumerate(nodes):
        sub = bottom.Y[:, idx]
        any_obs = ~np.all(np.isnan(sub), axis=1)
        Y[:, j] = np.where(any_obs, np.nansum(sub, axis=1), np.nan)
        act = ~np.isnan(sub)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            DISC[:, j] = np.nan_to_num(np.nanmean(np.where(act, bottom.DISC[:, idx], np.nan), axis=1))
        rows.append(dict(
            node_id=nid, level=lvl, level_idx=LEVELS.index(lvl),
            city_idx=attrs["city_idx"] if attrs["city_idx"] >= 0 else C,
            region_idx=attrs["city_idx"] if attrs["city_idx"] >= 0 else C,
            cat_idx=attrs["cat_idx"] if attrs["cat_idx"] >= 0 else len(CATEGORIES),
            store_idx=attrs["store_idx"] if attrs["store_idx"] >= 0 else bottom.series.store_idx.max() + 1,
            price=float(bottom.series.price.values[idx].mean()), pack_size=float(bottom.series.pack_size.values[idx].mean()),
            launch_idx=0, n_children=len(idx),
        ))
    series = pd.DataFrame(rows)
    return Panel(bottom.dates, bottom.n_obs, series, Y, np.zeros_like(Y, dtype=bool), DISC,
                 np.zeros_like(DISC), bottom.TEMP, bottom.RAIN, bottom.HUM, bottom.SEARCH, bottom.EV,
                 bottom.cities, meta=dict(bottom.meta, aggregate=True))
