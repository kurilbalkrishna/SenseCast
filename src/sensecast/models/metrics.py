"""Forecast and inventory metrics.

WAPE  = sum|y - f| / sum y            volume-weighted, safe with zero-sales days
MASE  = mean|y - f| / mean|y_t - y_t-7|  per series, scale-free vs weekly naive
Pinball(tau) = mean(max(tau (y-q), (tau-1)(y-q)))   scores a quantile forecast
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def wape(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    den = np.abs(y).sum()
    return float(np.abs(y - f).sum() / den) if den > 0 else float("nan")


def bias(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    den = np.abs(y).sum()
    return float((f - y).sum() / den) if den > 0 else float("nan")


def pinball(y, q, tau: float) -> float:
    y, q = np.asarray(y, float), np.asarray(q, float)
    diff = y - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def mean_pinball(y, q10, q50, q90) -> float:
    return float(np.mean([pinball(y, q10, 0.1), pinball(y, q50, 0.5), pinball(y, q90, 0.9)]))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def mase_scale(Ym: np.ndarray, upto: int, m: int = 7) -> np.ndarray:
    """Per-series in-sample MAE of the seasonal naive forecast over days <= upto."""
    hist = Ym[: upto + 1]
    diff = np.abs(hist[m:] - hist[:-m])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sc = np.nanmean(diff, axis=0)
    sc = np.where(np.isnan(sc) | (sc <= 0), np.nan, sc)
    fallback = np.nanmedian(sc) if np.isfinite(np.nanmedian(sc)) else 1.0
    return np.where(np.isnan(sc), fallback, sc)


def mase(df: pd.DataFrame, fcol: str, scale: np.ndarray, series_col: str = "series") -> float:
    """Mean over series of (MAE / scale)."""
    err = (df["y"] - df[fcol]).abs() / scale[df[series_col].values]
    return float(err.groupby(df[series_col]).mean().mean())


def stability(df: pd.DataFrame, fcol: str) -> float:
    """Median over series of sum|F_o(d) - F_o'(d)| / sum F, for consecutive origins o < o'
    that both forecast the same day d. Lower = forecasts do not flip-flop."""
    g = df[["series", "day", "origin", fcol]].sort_values(["series", "day", "origin"])
    g["prev"] = g.groupby(["series", "day"])[fcol].shift(1)
    g = g.dropna(subset=["prev"])
    if g.empty:
        return float("nan")
    per = g.groupby("series").apply(
        lambda x: np.abs(x[fcol] - x["prev"]).sum() / max(x[fcol].abs().sum(), 1e-9), include_groups=False
    )
    return float(per.median())


def score_block(df: pd.DataFrame, fcols: list[str], scale: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in fcols:
        rows.append(dict(model=c, wape=wape(df.y, df[c]), mase=mase(df, c, scale), bias=bias(df.y, df[c])))
    return pd.DataFrame(rows)
