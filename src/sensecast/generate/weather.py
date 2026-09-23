"""Daily weather per city.

Primary source: Open-Meteo historical archive (free for non-commercial use, no key).
Fallback: a synthetic climatology calibrated to Mumbai / Pune / Nashik monthly
normals, used when the network is unavailable (CI, sandboxed graders).
The provenance of whichever source was used is written to the data manifest.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from sensecast.logging_utils import get_logger

log = get_logger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Monthly normals (Jan..Dec): mean daily max temperature (C) and total rain (mm).
NORMALS = {
    "Mumbai": dict(
        tmax=[31.0, 32.0, 33.0, 33.5, 34.0, 32.0, 30.0, 29.8, 30.8, 33.0, 34.0, 32.5],
        rain=[1, 1, 1, 2, 15, 500, 840, 580, 330, 90, 15, 3],
        wet_p=[0.02, 0.02, 0.02, 0.03, 0.10, 0.65, 0.90, 0.85, 0.65, 0.25, 0.08, 0.03],
    ),
    "Pune": dict(
        tmax=[30.0, 32.0, 35.0, 37.0, 36.5, 31.0, 28.0, 27.5, 29.0, 31.0, 30.0, 29.0],
        rain=[1, 1, 2, 10, 35, 140, 180, 130, 130, 90, 25, 5],
        wet_p=[0.02, 0.02, 0.03, 0.08, 0.15, 0.55, 0.80, 0.75, 0.55, 0.30, 0.10, 0.03],
    ),
    "Nashik": dict(
        tmax=[29.5, 31.5, 35.0, 37.5, 37.0, 32.0, 28.5, 28.0, 29.5, 31.0, 30.0, 29.0],
        rain=[1, 1, 2, 5, 20, 130, 190, 150, 130, 60, 15, 3],
        wet_p=[0.02, 0.02, 0.03, 0.05, 0.12, 0.55, 0.80, 0.75, 0.55, 0.25, 0.08, 0.03],
    ),
}


def fetch_openmeteo(cities: dict, start: str, end: str, timeout: int = 20) -> pd.DataFrame:
    frames = []
    for city, c in cities.items():
        q = urllib.parse.urlencode(
            dict(
                latitude=c["lat"], longitude=c["lon"], start_date=start, end_date=end,
                daily="temperature_2m_max,precipitation_sum,relative_humidity_2m_mean",
                timezone="Asia/Kolkata",
            )
        )
        with urllib.request.urlopen(f"{OPEN_METEO_URL}?{q}", timeout=timeout) as resp:  # noqa: S310 - fixed https URL
            payload = json.loads(resp.read())
        d = payload["daily"]
        frames.append(
            pd.DataFrame(
                dict(
                    date=pd.to_datetime(d["time"]),
                    city=city,
                    temp_max=d["temperature_2m_max"],
                    rain_mm=d["precipitation_sum"],
                    humidity=d["relative_humidity_2m_mean"],
                )
            )
        )
    df = pd.concat(frames, ignore_index=True)
    # Open-Meteo can return nulls for the most recent days; fill from climatology later.
    return df


def synthetic_weather(cities: list[str], dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Markov wet/dry days + gamma rain amounts + AR(1) temperature anomalies."""
    frames = []
    doy_month = dates.month.values - 1
    days_in_month = dates.days_in_month.values
    for city in cities:
        nm = NORMALS[city]
        tmax_clim = np.array(nm["tmax"])[doy_month]
        wet_p = np.array(nm["wet_p"])[doy_month]
        rain_total = np.array(nm["rain"])[doy_month]
        mean_wet_amount = rain_total / np.maximum(days_in_month * wet_p, 1e-6)

        n = len(dates)
        wet = np.zeros(n, dtype=bool)
        anom = np.zeros(n)
        for t in range(n):
            # persistence: wet yesterday -> higher chance today
            p = wet_p[t]
            if t > 0:
                p = min(0.97, p + 0.25 * (wet[t - 1] - p))
            wet[t] = rng.random() < p
            anom[t] = (0.75 * anom[t - 1] if t else 0) + rng.normal(0, 0.9)
        shape = 0.8
        rain = np.where(wet, rng.gamma(shape, mean_wet_amount / shape), 0.0)
        temp = tmax_clim + anom - 1.2 * np.log1p(rain) / 2
        hum = np.clip(55 + 30 * wet_p + 5 * wet + rng.normal(0, 4, n), 25, 99)
        frames.append(
            pd.DataFrame(dict(date=dates, city=city, temp_max=temp.round(1), rain_mm=rain.round(1), humidity=hum.round(0)))
        )
    return pd.concat(frames, ignore_index=True)


def get_weather(cfg: dict, dates: pd.DatetimeIndex, external_dir: Path, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Return weather for all configured cities over ``dates`` plus a provenance record."""
    cities = cfg["weather"]["cities"]
    city_names = sorted({s["city"] for s in cfg["world"]["stores"]})
    cities = {c: cities[c] for c in city_names}
    source = cfg["weather"]["source"]
    cache = external_dir / "weather_openmeteo.parquet"
    # the archive lags real time by a few days and cannot serve future dates: fetch what exists,
    # the synthetic climatology fills the remainder (counted in the provenance record)
    last_available = min(dates.max(), pd.Timestamp.today().normalize() - pd.Timedelta(days=6))
    start, end = dates.min().strftime("%Y-%m-%d"), last_available.strftime("%Y-%m-%d")

    synth = synthetic_weather(city_names, dates, rng)  # always drawn: keeps RNG stream stable

    if source in ("auto", "openmeteo"):
        try:
            if cache.exists():
                real = pd.read_parquet(cache)
                covered = real.date.min() <= dates.min() and real.date.max() >= last_available
                if not covered or set(real.city) < set(city_names):
                    raise FileNotFoundError("cache does not cover range")
                how = "cache"
            else:
                real = fetch_openmeteo(cities, start, end)
                real.to_parquet(cache, index=False)
                how = "api"
            real = real[real.city.isin(city_names) & real.date.between(dates.min(), dates.max())]
            merged = synth.merge(real, on=["date", "city"], how="left", suffixes=("_syn", ""))
            n_filled = int(merged.temp_max.isna().sum())
            for col in ["temp_max", "rain_mm", "humidity"]:
                merged[col] = merged[col].fillna(merged[f"{col}_syn"])
            df = merged[["date", "city", "temp_max", "rain_mm", "humidity"]]
            prov = dict(source="open-meteo archive", url=OPEN_METEO_URL, retrieved_via=how,
                        licence="CC BY 4.0 (Open-Meteo), non-commercial API tier",
                        rows_filled_from_climatology=n_filled)
            log.info("weather_loaded", extra=prov)
            return df, prov
        except Exception as exc:  # noqa: BLE001 - any network/parse failure -> fallback
            if source == "openmeteo":
                raise
            log.warning("weather_fallback_synthetic", extra={"reason": str(exc)[:200]})

    prov = dict(source="synthetic climatology", calibrated_to="IMD-style monthly normals for Mumbai, Pune, Nashik",
                licence="generated by this project")
    return synth, prov
