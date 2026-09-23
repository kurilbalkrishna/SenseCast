"""Cold-start forecasting for items with < 28 days of history.

C1 (analog items): pick the k most similar ESTABLISHED items in the same store and
category (by log price and log pack size), take the median of their multisource forecasts for
the target day, and scale by a launch-ramp factor learned from earlier launches.
Because the analogs' forecasts already include weather, festival and search
effects, the new item inherits demand sensing on day one.

Baseline: category mean - average daily sales per established item of the same
category in the same store over the last 28 days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGE_BUCKETS = [0, 7, 14, 21, 28]


def analog_table(series: pd.DataFrame, k: int) -> dict[int, np.ndarray]:
    """For every series, indices of up to k nearest other series in the same store and
    category (candidates are filtered by age at forecast time)."""
    x = np.column_stack([np.log(series.price.values), np.log(series.pack_size.values + 1)])
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    out = {}
    groups = series.groupby(["store_id", "category"]).indices
    for _, idx in groups.items():
        for i in idx:
            others = idx[idx != i]
            dist = np.linalg.norm(x[others] - x[i], axis=1)
            out[i] = others[np.argsort(dist)]
    return out


def launch_ramp(Y: np.ndarray, series: pd.DataFrame, upto: int, analogs: dict, k: int) -> np.ndarray:
    """Median ratio of new-item sales to analog sales, per age bucket, over launches whose
    first 28 days finished by ``upto``. Medians keep one freak launch from dominating.
    Falls back to a generic ramp when fewer than 3 launches are available."""
    launch = series.launch_idx.values
    ratios: list[list[float]] = [[] for _ in range(len(AGE_BUCKETS) - 1)]
    for i in np.where((launch > 0) & (launch + 28 <= upto))[0]:
        cands = [j for j in analogs[i] if launch[j] + 90 <= launch[i]][:k]
        if not cands:
            continue
        for b in range(len(AGE_BUCKETS) - 1):
            days = np.arange(launch[i] + AGE_BUCKETS[b], launch[i] + AGE_BUCKETS[b + 1])
            den = np.nanmedian(np.nansum(Y[days][:, cands], axis=0))
            if den > 0:
                ratios[b].append(np.nansum(Y[days, i]) / den)
    generic = np.array([0.35, 0.6, 0.75, 0.85])
    out = np.array([np.median(r) if len(r) >= 3 else g for r, g in zip(ratios, generic, strict=False)])
    return np.clip(out, 0.05, 3.0)


def _candidates(analogs: dict, launch: np.ndarray, i: int, origin: int, k: int) -> list[int]:
    return [j for j in analogs[i] if launch[j] + 90 <= origin][:k]


def forecast_c1(df: pd.DataFrame, point_col: str, series: pd.DataFrame, analogs: dict, ramp: np.ndarray,
                k: int, young_age: int) -> pd.Series:
    """Cold-start forecast for rows with age < young_age: mean of the analog series'
    ``point_col`` forecasts for the same (origin, day), times the launch ramp."""
    launch = series.launch_idx.values
    young = df[df.age < young_age]
    if young.empty:
        return pd.Series(dtype=np.float32)
    pairs = []
    for (i, o) in young[["series", "origin"]].drop_duplicates().itertuples(index=False):
        for j in _candidates(analogs, launch, i, o, k):
            pairs.append((i, o, j))
    if not pairs:
        return pd.Series(np.zeros(len(young), dtype=np.float32), index=young.index)
    pr = pd.DataFrame(pairs, columns=["series", "origin", "analog"])
    rows = young[["series", "origin", "day"]].reset_index().merge(pr, on=["series", "origin"], how="left")
    ref = df[["origin", "day", "series", point_col]].rename(columns={"series": "analog", point_col: "f"})
    rows = rows.merge(ref, on=["origin", "day", "analog"], how="left")
    base = rows.groupby("index").f.median().reindex(young.index).fillna(0.0)
    b = np.minimum(young.age.values.astype(int) // 7, len(ramp) - 1)
    return pd.Series((base.values * ramp[b]).astype(np.float32), index=young.index)


def category_mean(df: pd.DataFrame, hist_mean28: np.ndarray, series: pd.DataFrame, young_age: int) -> pd.Series:
    launch = series.launch_idx.values
    young = df[df.age < young_age]
    if young.empty:
        return pd.Series(dtype=np.float32)
    groups = series.groupby(["store_id", "category"]).indices
    store, cat = series.store_id.values, series.category.values
    val = {}
    for (i, o) in young[["series", "origin"]].drop_duplicates().itertuples(index=False):
        idx = groups[(store[i], cat[i])]
        est = idx[launch[idx] + 90 <= o]
        m = hist_mean28[o, est] if len(est) else np.array([])
        val[(i, o)] = float(np.nanmean(m)) if len(m) and np.isfinite(m).any() else 0.0
    keys = list(zip(young.series.values, young.origin.values, strict=False))
    return pd.Series(np.array([val[k_] for k_ in keys], dtype=np.float32), index=young.index)
