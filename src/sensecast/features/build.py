"""Point-in-time feature construction.

One row = (series n, origin o, horizon h), target day d = o + h.

The leakage rule, enforced with asserts:
* anything observed (sales, stockouts, search, weather actuals) is read at day <= o;
* known-in-advance inputs (calendar, events, planned promotions) may be read at d;
* weather at d is a FORECAST: actual + noise whose spread grows with h, because on
  day o nobody knows what the weather on day d will be.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from sensecast.features.panel import Panel

WX_TEMP_SD = 0.8      # degC per sqrt(day of lead)
WX_RAIN_LOGSD = 0.35  # multiplicative noise per sqrt(day of lead)

STATIC = ["cat_idx", "store_idx", "region_idx", "price_log", "pack_size"]
HISTORY = ["last_obs", "mean7", "mean28", "mean56", "std28", "zero28", "sdw_mean4", "days_since_sale", "age"]
CALENDAR = ["h", "dow", "dom", "month", "woy"]
EVENTS = ["days_to_event", "event_id", "in_festival", "is_holiday"]
PROMO = ["disc", "post_promo"]
WEATHER = ["temp_fc", "rain_fc", "hum_fc", "wx_missing"]
SEARCH = ["search_o", "search_mean7", "search_ratio", "search_trend"]

FEATURES_M1 = STATIC + HISTORY + CALENDAR
FEATURES_M2 = FEATURES_M1 + EVENTS + PROMO + WEATHER + SEARCH
CATEGORICAL = ["cat_idx", "store_idx", "region_idx", "event_id", "level_idx"]

FAMILY = {**{f: "history" for f in HISTORY}, **{f: "calendar" for f in CALENDAR},
          **{f: "item/store" for f in STATIC}, **{f: "events" for f in EVENTS},
          **{f: "promotion" for f in PROMO}, **{f: "weather" for f in WEATHER},
          **{f: "search" for f in SEARCH}, "level_idx": "item/store"}


# ---------------------------------------------------------------------------
# rolling statistics at every day t, using only days <= t
# ---------------------------------------------------------------------------
def _rolling(Ym: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling sum, sum of squares and count of non-NaN values over window w ending at t."""
    X = np.nan_to_num(Ym)
    M = (~np.isnan(Ym)).astype(np.float64)
    cs = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), X]), axis=0)
    cs2 = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), X.astype(np.float64) ** 2]), axis=0)
    cm = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), M]), axis=0)
    idx = np.arange(1, X.shape[0] + 1)
    lo = np.maximum(idx - w, 0)
    return cs[idx] - cs[lo], cs2[idx] - cs2[lo], cm[idx] - cm[lo]


def history_stats(panel: Panel) -> dict[str, np.ndarray]:
    """Per-day history statistics; stockout days are masked (censored demand)."""
    Ym = np.where(panel.SO, np.nan, panel.Y).astype(np.float64)
    out = {}
    for w in (7, 28, 56):
        s, _, c = _rolling(Ym, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"mean{w}"] = np.where(c > 0, s / c, np.nan)
    s, s2, c = _rolling(Ym, 28)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(c > 1, (s2 - s * s / np.maximum(c, 1)) / np.maximum(c - 1, 1), np.nan)
        out["std28"] = np.sqrt(np.maximum(var, 0))
        z, _, cz = _rolling(np.where(np.isnan(Ym), np.nan, (Ym == 0).astype(float)), 28)
        out["zero28"] = np.where(cz > 0, z / cz, np.nan)
    out["last_obs"] = Ym
    # days since last positive sale
    T, N = Ym.shape
    dss = np.full((T, N), np.nan)
    last = np.full(N, -10_000)
    obs_any = ~np.isnan(panel.Y)
    first = np.where(obs_any.any(0), obs_any.argmax(0), T)
    for t in range(T):
        pos = np.nan_to_num(panel.Y[t]) > 0
        last = np.where(pos, t, last)
        dss[t] = np.where(t >= first, np.minimum(t - last, 90), np.nan)
    out["days_since_sale"] = dss
    out["Ym"] = Ym
    return {k: v.astype(np.float32) for k, v in out.items()}


def search_stats(panel: Panel) -> dict[str, np.ndarray]:
    S = pd.DataFrame(panel.SEARCH)
    med5 = S.rolling(5, min_periods=1).median().values
    med28 = S.rolling(28, min_periods=7).median().values
    mean7 = S.rolling(7, min_periods=1).mean().values
    mean28 = S.rolling(28, min_periods=7).mean().values
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.clip(med5 / np.maximum(med28, 1e-3), 0, 5)  # medians resist bot spikes (up to 2 days)
    return dict(search_o=med5.astype(np.float32), search_mean7=mean7.astype(np.float32),
                search_ratio=ratio.astype(np.float32), search_trend=(mean7 - mean28).astype(np.float32))


def climatology(panel: Panel, upto: int) -> dict[str, np.ndarray]:
    """Monthly mean weather per city column from days <= upto (fallback for outages)."""
    month = panel.dates.month.values[: upto + 1] - 1
    out = {}
    for name, arr in (("temp", panel.TEMP), ("rain", panel.RAIN), ("hum", panel.HUM)):
        clim = np.zeros((12, arr.shape[1]), dtype=np.float32)
        for m in range(12):
            sel = arr[: upto + 1][month == m]
            clim[m] = np.nanmean(sel, axis=0) if len(sel) else np.nanmean(arr[: upto + 1], axis=0)
        out[name] = clim
    return out


class FeatureBuilder:
    """Builds feature rows for a panel. Cache history/search stats once per panel."""

    def __init__(self, panel: Panel, clim_upto: int, seed: int = 0, aggregate: bool = False):
        self.p = panel
        self.seed = seed
        self.aggregate = aggregate
        self.hist = history_stats(panel)
        self.srch = search_stats(panel)
        self.clim = climatology(panel, clim_upto)
        cal = pd.DataFrame(dict(dow=panel.dates.dayofweek, dom=panel.dates.day, month=panel.dates.month,
                                woy=panel.dates.isocalendar().week.values.astype(int)))
        self.cal = {c: cal[c].values.astype(np.float32) for c in cal}
        self.ev = {c: panel.EV[c].values.astype(np.float32) for c in panel.EV.columns}
        self.ev["event_id"] = self.ev["event_id"] + 1  # -1 (none) -> 0 for LightGBM categorical

    def _wx_noise(self, o: int, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Deterministic forecast-error draws for origin o: [H, C+1] per variable."""
        rng = np.random.default_rng(self.seed * 1_000_003 + o)
        C1 = self.p.TEMP.shape[1]
        return (rng.normal(0, 1, (H, C1)), rng.normal(0, 1, (H, C1)), rng.normal(0, 1, (H, C1)))

    def build(self, origins: list[int], horizon: int, with_target: bool = True) -> pd.DataFrame:
        p, hs = self.p, self.hist
        N = p.N
        H = horizon
        o = np.repeat(np.asarray(origins), H * N)
        h = np.tile(np.repeat(np.arange(1, H + 1), N), len(origins))
        n = np.tile(np.arange(N), len(origins) * H)
        d = o + h
        launch = p.series.launch_idx.values
        keep = (d < p.T) & (launch[n] <= d)
        o, h, n, d = o[keep], h[keep], n[keep], d[keep]

        # ---------------- leakage guards ---------------------------------
        assert np.all(o < p.n_obs), "origin must be an observed day"
        f: dict[str, np.ndarray] = {}
        f["h"] = h.astype(np.float32)
        s = p.series
        f["cat_idx"] = s.cat_idx.values[n].astype(np.float32)
        f["store_idx"] = s.store_idx.values[n].astype(np.float32)
        f["region_idx"] = s.region_idx.values[n].astype(np.float32)
        f["price_log"] = np.log(s.price.values[n]).astype(np.float32)
        f["pack_size"] = s.pack_size.values[n].astype(np.float32)
        if "level_idx" in s:
            f["level_idx"] = s.level_idx.values[n].astype(np.float32)

        for name in ("last_obs", "mean7", "mean28", "mean56", "std28", "zero28", "days_since_sale"):
            f[name] = hs[name][o, n]                                   # read at origin only
        # same weekday as target, the last 4 occurrences on or before the origin
        k0 = np.ceil(h / 7).astype(int)
        lags = np.stack([d - 7 * (k0 + j) for j in range(4)])
        assert np.all(lags <= o), "same-weekday lag reads the future"
        vals = hs["Ym"][np.clip(lags, 0, None), n]
        vals[lags < 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            f["sdw_mean4"] = np.nanmean(vals, axis=0).astype(np.float32)
        f["age"] = np.minimum(d - launch[n], 365).astype(np.float32)

        for c in ("dow", "dom", "month", "woy"):
            f[c] = self.cal[c][d]
        for c in ("days_to_event", "event_id", "in_festival", "is_holiday"):
            f[c] = self.ev[c][d]
        f["disc"] = p.DISC[d, n]
        f["post_promo"] = p.POST[d, n]

        # weather forecast proxy at target day
        city = s.city_idx.values[n]
        temp_a, rain_a, hum_a = p.TEMP[d, city], p.RAIN[d, city], p.HUM[d, city]
        missing = np.isnan(temp_a) | np.isnan(rain_a)
        month = self.cal["month"][d].astype(int) - 1
        temp_a = np.where(np.isnan(temp_a), self.clim["temp"][month, city], temp_a)
        rain_a = np.where(np.isnan(rain_a), self.clim["rain"][month, city], rain_a)
        hum_a = np.where(np.isnan(hum_a), self.clim["hum"][month, city], hum_a)
        is_obs_day = d < p.n_obs  # beyond n_obs the arrays already hold issued forecasts
        zt = np.zeros_like(temp_a)
        zr = np.zeros_like(rain_a)
        zh = np.zeros_like(hum_a)
        for oo in np.unique(o):
            m = o == oo
            nt, nr, nh = self._wx_noise(int(oo), H)
            zt[m], zr[m], zh[m] = nt[h[m] - 1, city[m]], nr[h[m] - 1, city[m]], nh[h[m] - 1, city[m]]
        sq = np.sqrt(h)
        noisy = is_obs_day & ~missing
        f["temp_fc"] = np.where(noisy, temp_a + WX_TEMP_SD * sq * zt, temp_a).astype(np.float32)
        f["rain_fc"] = np.log1p(np.where(noisy, rain_a * np.exp(WX_RAIN_LOGSD * sq * zr), rain_a)).astype(np.float32)
        f["hum_fc"] = np.clip(np.where(noisy, hum_a + 3.0 * sq * zh, hum_a), 0, 100).astype(np.float32)
        f["wx_missing"] = missing.astype(np.float32)

        cat = s.cat_idx.values[n]
        for name, arr in self.srch.items():
            f[name] = arr[o, cat]                                      # read at origin only

        df = pd.DataFrame(f)
        df["origin"] = o.astype(np.int32)
        df["day"] = d.astype(np.int32)
        df["series"] = n.astype(np.int32)
        if with_target:
            y = p.Y[np.minimum(d, p.T - 1), n]
            df["y"] = np.where(d < p.n_obs, y, np.nan).astype(np.float32)
            df["censored"] = p.SO[d, n]
        return df


def scale_for_aggregate(df: pd.DataFrame) -> np.ndarray:
    """Per-row scale used to make aggregate series comparable (mean28 at origin)."""
    return np.maximum(np.nan_to_num(df["mean28"].values, nan=1.0), 1e-3)


def normalise_history(df: pd.DataFrame, scale: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    for c in ("last_obs", "mean7", "mean28", "mean56", "std28", "sdw_mean4"):
        out[c] = out[c] / scale
    return out
