"""Synthetic retail world with KNOWN demand effects.

Why synthetic: because every effect (weather elasticity, festival uplift, promo
elasticity, search lead) is set here, the evaluation can check whether the models
recover them - something real data can never prove.

Outputs
-------
data/raw/          what a retailer would actually have (feeds)
    stores.csv, items.csv, sales.parquet, promotions.csv, events.csv,
    weather.parquet, weather_forecast.parquet, search.csv
data/ground_truth/ what only the simulator knows (never read by models)
    truth.parquet  (true mean demand, dispersion, uncensored demand)
    items_truth.csv, effects.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sensecast.config import resolve_path
from sensecast.evaluation.layout import origin_layout
from sensecast.generate.events import EVENT_META, REGIONAL, build_event_calendar
from sensecast.generate.weather import get_weather
from sensecast.logging_utils import get_logger, timed

log = get_logger(__name__)

CATEGORIES = ["beverages", "snacks", "personal_care", "monsoon_seasonal"]
CLASSES = ["smooth", "erratic", "intermittent", "lumpy"]

# ---- ground-truth effect sizes (log scale) --------------------------------
BETA_TEMP = dict(beverages=0.060, snacks=0.0, personal_care=0.010, monsoon_seasonal=-0.030)   # per degC above 30
BETA_RAIN = dict(beverages=-0.100, snacks=0.040, personal_care=-0.020, monsoon_seasonal=0.350)  # per log1p(mm)
WEEKLY = dict(  # Mon..Sun
    beverages=[-0.05, -0.08, -0.06, -0.03, 0.05, 0.16, 0.12],
    snacks=[-0.06, -0.08, -0.05, -0.02, 0.06, 0.15, 0.10],
    personal_care=[0.00, -0.02, -0.02, 0.00, 0.02, 0.06, 0.04],
    monsoon_seasonal=[-0.03, -0.03, -0.02, 0.00, 0.02, 0.08, 0.06],
)
MONSOON_SEASON_AMP = 0.8
SEARCH_GAMMA = dict(beverages=0.5, snacks=0.5, personal_care=0.4, monsoon_seasonal=0.6)
PRICE_MEAN = dict(beverages=60, snacks=40, personal_care=150, monsoon_seasonal=350)
PACK_SIZES = dict(beverages=[250, 500, 1000, 2000], snacks=[50, 100, 200, 400],
                  personal_care=[100, 200, 400], monsoon_seasonal=[1, 2])
POST_PROMO_DIP = -0.20
PRICE_VOLUME_ELASTICITY = -0.7
NOISE_SD = 0.10


def _item_table(cfg: dict, rng: np.random.Generator, layout: dict, n_obs: int, dates: pd.DatetimeIndex) -> pd.DataFrame:
    w = cfg["world"]
    mix = w["demand_class_mix"]
    rows = []
    idx = 0
    for cat in CATEGORIES:
        for _ in range(w["category_mix"][cat]):
            idx += 1
            cls = rng.choice(CLASSES, p=[mix[c] for c in CLASSES])
            if cls == "smooth":
                base, k, pi = rng.uniform(4, 30), 20.0, 0.0
            elif cls == "erratic":
                base, k, pi = rng.uniform(3, 15), 1.5, 0.0
            elif cls == "intermittent":
                base, k, pi = rng.uniform(0.15, 0.6), 5.0, 0.0
            else:  # lumpy
                base, k, pi = rng.uniform(0.3, 1.0), 0.4, 0.6
            price = float(np.round(PRICE_MEAN[cat] * np.exp(rng.normal(0, 0.35)), 0))
            base *= (max(price, 10.0) / PRICE_MEAN[cat]) ** PRICE_VOLUME_ELASTICITY  # cheaper items sell more
            rows.append(dict(
                item_id=f"I{idx:03d}", category=cat, demand_class=cls, base=base, k=k, pi=pi,
                price=max(price, 10.0), pack_size=int(rng.choice(PACK_SIZES[cat])),
                elasticity=rng.uniform(2.0, 4.0), year_amp=rng.uniform(0.0, 0.12),
                year_phase=rng.uniform(0, 2 * np.pi), drift=rng.normal(0, 0.08),
            ))
    items = pd.DataFrame(rows)

    # cold-start launches: 40% train block, 20% calibration block, 40% test block
    n_new = int(round(w["new_item_share"] * len(items)))
    new_idx = rng.choice(len(items), size=n_new, replace=False)
    H = layout["horizon"]
    windows = [
        (int(n_obs * 0.35), layout["train"][-1]),
        (layout["cal"][0] - 3, layout["cal"][-1] + 3),
        (layout["test"][0] - 10, layout["test"][-1] + 3),
    ]
    shares = np.array([0.4, 0.2, 0.4])
    counts = np.floor(shares * n_new).astype(int)
    counts[-1] += n_new - counts.sum()
    launch = np.zeros(len(items), dtype=int)
    j = 0
    for (lo, hi), c in zip(windows, counts, strict=False):
        for _ in range(c):
            launch[new_idx[j]] = rng.integers(lo, max(lo + 1, hi))
            j += 1
    items["launch_idx"] = launch
    items["launch_date"] = dates[launch]
    items["is_new"] = items.index.isin(new_idx)
    assert H > 0
    return items


def _event_effect(cal: pd.DataFrame, dates: pd.DatetimeIndex, cities: list[str]) -> np.ndarray:
    """E[t, cat, city] = summed log-uplift of all events on date t."""
    T = len(dates)
    E = np.zeros((T, len(CATEGORIES), len(cities)))
    for _, ev in cal.iterrows():
        meta = EVENT_META[ev.event]
        delta = (dates - ev.date).days.values  # 0 on festival day, negative before
        shape = np.zeros(T)
        b, s = meta["buildup"], meta["span"]
        pre = (delta >= -b) & (delta < 0)
        shape[pre] = 0.6 * (b + delta[pre] + 1) / (b + 1)
        on = (delta >= 0) & (delta < s)
        shape[on] = 1.0 - 0.5 * delta[on] / max(s, 1)
        post = (delta >= s) & (delta < s + 2)
        shape[post] = -0.10
        for ci, cat in enumerate(CATEGORIES):
            for ki, city in enumerate(cities):
                E[:, ci, ki] += meta["uplift"][cat] * REGIONAL.get((ev.event, city), 1.0) * shape
    return E


def _promotions(items: pd.DataFrame, regions: list[str], T: int, rng: np.random.Generator, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    years = T / 365.0
    for _, it in items.iterrows():
        for region in regions:
            n = rng.poisson(3 * years)
            for _ in range(n):
                start = int(rng.integers(0, T - 3))
                dur = int(rng.integers(4, 11))
                end = min(T - 1, start + dur - 1)
                rows.append(dict(item_id=it.item_id, region=region, date_start=dates[start], date_end=dates[end],
                                 discount=float(rng.choice([0.10, 0.15, 0.20, 0.30]))))
    promos = pd.DataFrame(rows).sort_values(["item_id", "region", "date_start"]).reset_index(drop=True)
    # drop overlaps within the same item-region: keep the earlier promo
    keep = []
    last_end = {}
    for i, r in promos.iterrows():
        key = (r.item_id, r.region)
        if key in last_end and r.date_start <= last_end[key]:
            continue
        keep.append(i)
        last_end[key] = r.date_end
    return promos.loc[keep].reset_index(drop=True)


def _buzz(T: int, n_cat: int, rng: np.random.Generator) -> np.ndarray:
    """Latent category 'buzz' (AR(1) with occasional viral jumps)."""
    B = np.zeros((T, n_cat))
    for t in range(1, T):
        jump = (rng.random(n_cat) < 0.006) * rng.uniform(0.4, 0.8, n_cat)
        B[t] = 0.95 * B[t - 1] + rng.normal(0, 0.06, n_cat) + jump
    return B


def generate_world(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    w = cfg["world"]
    H = cfg["backtest"]["horizon"]
    data_dir = resolve_path(cfg, "data")
    raw_dir, truth_dir, ext_dir = (data_dir / d for d in ("raw", "ground_truth", "external"))
    for d in (raw_dir, truth_dir, ext_dir):
        d.mkdir(parents=True, exist_ok=True)

    start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
    dates = pd.date_range(start, end + pd.Timedelta(days=H), freq="D")  # world runs H days past `end`
    n_obs = int((dates <= end).sum())
    T = len(dates)
    layout = origin_layout(n_obs, cfg)

    with timed(log, "generate_world", profile=cfg["profile"]):
        stores = pd.DataFrame(w["stores"])
        stores["region"] = stores["city"]
        stores["size_factor"] = np.clip(np.exp(rng.normal(0, 0.3, len(stores))), 0.6, 1.7).round(2)
        cities = sorted(stores.city.unique())
        items = _item_table(cfg, rng, layout, n_obs, dates)

        # series = store x item, store-major order
        S = stores.assign(_k=1).merge(items.assign(_k=1), on="_k").drop(columns="_k")
        N = len(S)
        cat_idx = S.category.map(CATEGORIES.index).values
        city_idx = S.city.map(cities.index).values

        weather, w_prov = get_weather(cfg, dates, ext_dir, rng)
        temp = weather.pivot(index="date", columns="city", values="temp_max").loc[dates, cities].values
        rain = weather.pivot(index="date", columns="city", values="rain_mm").loc[dates, cities].values

        cal = build_event_calendar(w["start"], str((end + pd.Timedelta(days=H)).date()))
        E = _event_effect(cal, dates, cities)

        promos = _promotions(items, cities, T, rng, dates)
        disc = np.zeros((T, N))
        post = np.zeros((T, N), dtype=bool)
        s_index = {(r.item_id, r.region): [] for r in S.itertuples()}
        for n, r in enumerate(S.itertuples()):
            s_index[(r.item_id, r.region)].append(n)
        for p in promos.itertuples():
            cols = s_index[(p.item_id, p.region)]
            a, b = dates.get_loc(p.date_start), dates.get_loc(p.date_end)
            disc[a:b + 1, cols] = p.discount
            post[b + 1:b + 5, cols] = True
        post &= disc == 0

        buzz = _buzz(T, len(CATEGORIES), rng)
        lead = w["search_lead_days"]
        buzz_lag = np.vstack([np.repeat(buzz[:1], lead, 0), buzz[:-lead]])

        dow = dates.dayofweek.values
        doy = dates.dayofyear.values
        tt = np.arange(T)[:, None]

        Wk = np.array([WEEKLY[c] for c in CATEGORIES])
        log_lam = Wk[cat_idx[None, :], dow[:, None]]
        log_lam += S.year_amp.values * np.sin(2 * np.pi * doy[:, None] / 365.25 + S.year_phase.values)
        season = np.exp(-(((doy - 205) / 45.0) ** 2))
        log_lam += (cat_idx == CATEGORIES.index("monsoon_seasonal")) * MONSOON_SEASON_AMP * season[:, None]
        bt = np.array([BETA_TEMP[c] for c in CATEGORIES])[cat_idx]
        br = np.array([BETA_RAIN[c] for c in CATEGORIES])[cat_idx]
        log_lam += bt * (temp[:, city_idx] - 30.0) + br * np.log1p(rain[:, city_idx])
        log_lam += E[:, cat_idx, city_idx]
        log_lam += S.elasticity.values * disc + POST_PROMO_DIP * post * (cat_idx != 3)
        gam = np.array([SEARCH_GAMMA[c] for c in CATEGORIES])[cat_idx]
        log_lam += gam * buzz_lag[:, cat_idx]
        log_lam += S.drift.values * tt / 365.25
        log_lam += rng.normal(0, NOISE_SD, (T, N))

        age = tt - S.launch_idx.values
        active = age >= 0
        ramp = np.where(active, 1 - np.exp(-(np.maximum(age, 0) + 1) / 10.0), 0.0)
        ramp = np.where(S.is_new.values, ramp, active.astype(float))
        lam = S.base.values * S.size_factor.values * np.exp(log_lam) * ramp
        lam = np.where(active, lam, np.nan)

        # draw demand: zero-inflated negative binomial
        k = S.k.values
        pi = S.pi.values
        mean_nz = np.nan_to_num(lam) / (1 - pi)
        p = k / (k + mean_nz)
        demand = rng.negative_binomial(np.broadcast_to(k, (T, N)), np.clip(p, 1e-9, 1))
        demand = np.where(rng.random((T, N)) < pi, 0, demand)
        demand = np.where(active, demand, 0)

        # ---- historical inventory under a simple manual policy -> censored sales
        inv = cfg["inventory"]
        L, R, mult = inv["lead_time_days"], inv["review_days"], inv["hist_cover_multiplier"]
        on_hand = np.zeros(N)
        arrivals = np.zeros((T + L + 1, N))
        ema = np.nan_to_num(lam[0]).copy()
        sales = np.zeros((n_obs, N), dtype=np.int64)
        oh_open = np.zeros((n_obs, N), dtype=np.int64)
        receipts = np.zeros((n_obs, N), dtype=np.int64)
        oh_close = np.zeros((n_obs, N), dtype=np.int64)
        init = active[0]
        on_hand[init] = np.ceil(np.nan_to_num(lam[0][init]) * (L + R) * mult) + 2
        for t in range(n_obs):
            launching = (S.launch_idx.values == t) & (t > 0)
            on_hand[launching] = np.ceil(S.base.values[launching] * S.size_factor.values[launching] * 3) + 1
            ema[launching] = S.base.values[launching] * S.size_factor.values[launching] * 0.4
            oh_open[t] = on_hand
            receipts[t] = arrivals[t]
            avail = on_hand + arrivals[t]
            sold = np.minimum(demand[t], avail)
            sales[t] = sold
            on_hand = avail - sold
            oh_close[t] = on_hand
            ema = np.where(active[t], 0.87 * ema + 0.13 * sold, ema)
            target = np.ceil(mult * ema * (L + R)) + 1
            position = on_hand + arrivals[t + 1:t + L + 1].sum(0)
            order = np.maximum(0, target - position) * active[t]
            order *= rng.random(N) >= inv["supplier_miss_rate"]
            arrivals[t + L] += order

        # ---- raw feeds ------------------------------------------------------
        obs_dates = dates[:n_obs]
        store_ids = S.store_id.values
        item_ids = S.item_id.values
        tt_idx, nn_idx = np.nonzero(active[:n_obs])
        disc_obs = disc[:n_obs]
        feed = pd.DataFrame(dict(
            date=obs_dates[tt_idx],
            store_id=store_ids[nn_idx],
            item_id=item_ids[nn_idx],
            units_sold=sales[tt_idx, nn_idx],
            unit_price=np.round(S.price.values[nn_idx] * (1 - disc_obs[tt_idx, nn_idx]), 2),
            on_hand_open=oh_open[tt_idx, nn_idx],
            receipts=receipts[tt_idx, nn_idx],
            on_hand_close=oh_close[tt_idx, nn_idx],
        ))
        feed = _inject_defects(feed, cfg, rng, end)

        items_pub = items[["item_id", "category", "price", "pack_size", "launch_date"]]
        stores_pub = stores[["store_id", "city", "region"]]
        obs_weather = weather[weather.date <= end].reset_index(drop=True)
        fc_rows = weather[weather.date > end].copy()
        h = (fc_rows.date - end).dt.days.values
        fc_rows["temp_max"] = (fc_rows.temp_max + rng.normal(0, 0.8 * np.sqrt(h))).round(1)
        fc_rows["rain_mm"] = (fc_rows.rain_mm * np.exp(rng.normal(0, 0.35 * np.sqrt(h)))).round(1)
        fc_rows.insert(0, "issued", end)
        search = pd.DataFrame(
            np.clip(50 * np.exp(0.9 * buzz + 0.15 * season[:, None]) + rng.normal(0, 2.5, buzz.shape), 0, 100),
            index=dates, columns=CATEGORIES,
        ).round(1)
        search = search.loc[:end].rename_axis("date").reset_index().melt("date", var_name="category", value_name="search_index")
        promos_pub = promos[promos.date_start <= end + pd.Timedelta(days=H)]

        stores_pub.to_csv(raw_dir / "stores.csv", index=False)
        items_pub.to_csv(raw_dir / "items.csv", index=False)
        feed.to_parquet(raw_dir / "sales.parquet", index=False)
        promos_pub.to_csv(raw_dir / "promotions.csv", index=False)
        cal.to_csv(raw_dir / "events.csv", index=False)
        obs_weather.to_parquet(raw_dir / "weather.parquet", index=False)
        fc_rows.to_parquet(raw_dir / "weather_forecast.parquet", index=False)
        search.to_csv(raw_dir / "search.csv", index=False)

        tt_all, nn_all = np.nonzero(active)
        truth = pd.DataFrame(dict(
            date=dates[tt_all], store_id=store_ids[nn_all], item_id=item_ids[nn_all],
            lam=lam[tt_all, nn_all].astype(np.float32), demand=demand[tt_all, nn_all].astype(np.int32),
        ))
        truth.to_parquet(truth_dir / "truth.parquet", index=False)
        items.to_csv(truth_dir / "items_truth.csv", index=False)
        stores.to_csv(truth_dir / "stores_truth.csv", index=False)
        effects = dict(
            beta_temp=BETA_TEMP, beta_rain=BETA_RAIN, weekly=WEEKLY, monsoon_season_amp=MONSOON_SEASON_AMP,
            search_gamma=SEARCH_GAMMA, search_lead_days=lead, post_promo_dip=POST_PROMO_DIP,
            price_volume_elasticity=PRICE_VOLUME_ELASTICITY,
            event_uplift={k: v["uplift"] for k, v in EVENT_META.items()},
            regional={f"{a}|{b}": v for (a, b), v in REGIONAL.items()}, noise_sd=NOISE_SD,
        )
        (truth_dir / "effects.json").write_text(json.dumps(effects, indent=2))

    stats = dict(
        n_series=N, n_days_observed=n_obs, rows_feed=len(feed),
        stockout_share=float((oh_close[active[:n_obs]] == 0).mean()),
        weather=w_prov, layout={k: (v if isinstance(v, int) else [int(x) for x in v]) for k, v in layout.items()},
    )
    (raw_dir / "_generation.json").write_text(json.dumps(stats, indent=2, default=str))
    log.info("world_generated", extra={k: v for k, v in stats.items() if k != "layout"})
    return stats


def _inject_defects(feed: pd.DataFrame, cfg: dict, rng: np.random.Generator, end: pd.Timestamp) -> pd.DataFrame:
    dq = cfg["world"]["dq_injection"]
    n = len(feed)
    dup = feed.sample(n=max(1, int(n * dq["duplicate_rate"])), random_state=int(rng.integers(1e9)))
    neg_idx = rng.choice(n, size=max(1, int(n * dq["negative_rate"])), replace=False)
    feed.loc[neg_idx, "units_sold"] = -feed.loc[neg_idx, "units_sold"].clip(lower=1)
    null_idx = rng.choice(n, size=max(1, int(n * dq["null_price_rate"])), replace=False)
    feed["unit_price"] = feed["unit_price"].astype(float)
    feed.loc[null_idx, "unit_price"] = np.nan
    fut = feed.sample(n=dq["future_rows"], random_state=int(rng.integers(1e9))).copy()
    fut["date"] = end + pd.to_timedelta(rng.integers(1, 30, len(fut)), unit="D")
    out = pd.concat([feed, dup, fut], ignore_index=True)
    return out.sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)


if __name__ == "__main__":  # pragma: no cover
    from sensecast.config import load_config

    generate_world(load_config())
    _ = Path
