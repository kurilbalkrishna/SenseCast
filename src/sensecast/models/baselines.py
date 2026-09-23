"""Baselines.

B0  seasonal naive: the same weekday's sales from the most recent week <= origin.
B1  'planner' baseline: exponentially smoothed level x shrunk weekday index, with
    negative-binomial intervals from recent mean/variance. A stand-in for a
    spreadsheet process a store planner might run today.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def ses_level(Ym: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Simple exponential smoothing level after observing day t (NaNs skip the update)."""
    T, N = Ym.shape
    L = np.full((T, N), np.nan, dtype=np.float32)
    lvl = np.full(N, np.nan)
    for t in range(T):
        y = Ym[t]
        obs = ~np.isnan(y)
        start = obs & np.isnan(lvl)
        lvl = np.where(start, y, lvl)
        upd = obs & ~start
        lvl = np.where(upd, lvl + alpha * (y - lvl), lvl)
        L[t] = lvl
    return L


def b0(df: pd.DataFrame, Ym: np.ndarray) -> np.ndarray:
    h = (df.day - df.origin).values
    k0 = np.ceil(h / 7).astype(int)
    lag = df.day.values - 7 * k0
    v = np.where(lag >= 0, Ym[np.clip(lag, 0, None), df.series.values], np.nan)
    fb = df.mean7.values
    return np.nan_to_num(np.where(np.isnan(v), fb, v), nan=0.0)


def b1(df: pd.DataFrame, level: np.ndarray) -> np.ndarray:
    lvl = level[df.origin.values, df.series.values]
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = df.sdw_mean4.values / df.mean28.values
    idx = np.where(np.isfinite(idx), 1 + 0.5 * (idx - 1), 1.0)
    return np.nan_to_num(lvl * idx, nan=0.0).clip(min=0)


def nb_quantiles(mean: np.ndarray, std: np.ndarray, qs=(0.1, 0.5, 0.9)) -> list[np.ndarray]:
    """Negative-binomial (or Poisson when under-dispersed) quantiles by moments."""
    m = np.maximum(np.nan_to_num(mean), 1e-6)
    var = np.nan_to_num(std) ** 2
    over = var > m * 1.05
    k = np.where(over, m**2 / np.maximum(var - m, 1e-6), 1e6)
    p = k / (k + m)
    return [np.where(over, stats.nbinom.ppf(q, k, p), stats.poisson.ppf(q, m)).astype(np.float32) for q in qs]
