"""Uncertainty-aware replenishment simulation.

Daily review, fixed lead time L, lost sales (no backorders). Every evening each
policy sets an order-up-to level S for the protection period (L + R days) and
orders S minus the inventory position.

    P0 manual       S = (1 + buffer) * sum(B1 forecast)
    P1 point ML     S = sum(final forecast) + z * sd_resid * sqrt(L + R)
    P2 uncertainty  S = sum(final forecast) + z * sqrt(sum(sd_day^2)),
                    sd_day from the conformal P10-P90 width of each day

Demand paths are drawn from the generator's TRUE demand distribution for the test
window, so the simulator measures outcomes rather than forecast opinions.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from sensecast.config import resolve_path
from sensecast.logging_utils import get_logger, timed

log = get_logger(__name__)
POLICIES = ["P0_manual", "P1_point_ml", "P2_uncertainty"]
Z_WIDTH = stats.norm.ppf(0.9) - stats.norm.ppf(0.1)  # 2.563: P10-P90 width in sd units


def _forecast_cube(fc: pd.DataFrame, origins: list[int], series: np.ndarray, col: str, H: int) -> np.ndarray:
    """[n_origin, H, n_series] array of a forecast column."""
    oi = {o: i for i, o in enumerate(origins)}
    si = {s: i for i, s in enumerate(series)}
    cube = np.zeros((len(origins), H, len(series)), dtype=np.float64)
    sub = fc[fc.series.isin(si)]
    cube[sub.origin.map(oi).values, sub.h.values.astype(int) - 1, sub.series.map(si).values] = sub[col].values
    return cube


def order_up_to(policy: str, cubes: dict, o_i: int, h0: int, n_days: int, resid_sd: np.ndarray,
                z: float, buffer: float) -> np.ndarray:
    """Order-up-to level for protection days h0 .. h0+n_days-1 (horizons from origin o_i)."""
    sl = slice(h0 - 1, h0 - 1 + n_days)
    if policy == "P0_manual":
        return np.ceil((1 + buffer) * cubes["b1"][o_i, sl].sum(0))
    mu = cubes["final"][o_i, sl].sum(0)
    if policy == "P1_point_ml":
        return np.ceil(mu + z * resid_sd * np.sqrt(n_days))
    sd_day = np.maximum(cubes["p90c"][o_i, sl] - cubes["p10c"][o_i, sl], 0) / Z_WIDTH
    return np.ceil(mu + z * np.sqrt((sd_day**2).sum(0)))


def run_simulation(cfg: dict) -> dict:
    art = resolve_path(cfg, "artifacts")
    data = resolve_path(cfg, "data")
    sim = cfg["simulation"]
    inv = cfg["inventory"]
    L, R = inv["lead_time_days"], inv["review_days"]
    P = L + R
    H = cfg["backtest"]["horizon"]
    z = stats.norm.ppf(sim["service_target"])
    rng = np.random.default_rng(cfg["seed"] + 7)
    metrics = json.loads((art / "metrics.json").read_text())
    test_origins = metrics["layout"]["test"]

    with timed(log, "simulate"):
        fc = pd.read_parquet(art / "forecasts" / "test.parquet")
        series_tbl = pd.read_parquet(art / "series.parquet")
        truth = pd.read_parquet(data / "ground_truth" / "truth.parquet")
        items_truth = pd.read_csv(data / "ground_truth" / "items_truth.csv")

        first_day, last_day = test_origins[0] + 1, test_origins[-1] + 7
        # series active for the whole window (launches inside it are cold-start, simulated separately in reports)
        age_ok = fc[fc.origin == test_origins[0]].groupby("series").age.min()
        series = np.sort(age_ok[age_ok >= 28].index.values)
        n_s = len(series)
        cubes = {c: _forecast_cube(fc, test_origins, series, c, H) for c in ("b1", "final", "p10c", "p90c")}
        resid_sd = series_tbl.resid_sd.values[series]

        start_date = pd.Timestamp(metrics["dates"]["start"])
        days = np.arange(first_day, last_day + 1)
        dates = start_date + pd.to_timedelta(days, unit="D")
        keys = series_tbl.iloc[series][["store_id", "item_id"]]
        tr = truth[truth.date.isin(dates)].merge(keys.reset_index(), on=["store_id", "item_id"])
        lam = np.zeros((len(days), n_s))
        lam[(tr.date - dates[0]).dt.days.values, tr["index"].map({s: i for i, s in enumerate(series)}).values] = tr.lam.values
        it = items_truth.set_index("item_id").loc[keys.item_id.values]
        k, pi = it.k.values, it.pi.values
        price = series_tbl.price.values[series]
        n_paths = sim["n_paths"]

        # demand paths [paths, days, series] from the true zero-inflated negative binomial
        mean_nz = lam / (1 - pi)
        p = k / (k + mean_nz)
        demand = rng.negative_binomial(np.broadcast_to(k, (n_paths, len(days), n_s)),
                                       np.broadcast_to(np.clip(p, 1e-9, 1), (n_paths, len(days), n_s)))
        demand = np.where(rng.random(demand.shape) < pi, 0, demand)
        mean_daily = np.maximum(lam.mean(0), 1e-6)

        results, traces = {}, {}
        for pol in POLICIES:
            results[pol], traces[pol] = simulate_policy(
                pol, cubes, demand, days, test_origins, resid_sd, z, price, mean_daily, L, P, sim)

        # service-vs-inventory frontier: sweep each policy's knob on a subset of paths, so
        # policies are compared at EQUAL inventory rather than at their default settings
        fp = min(sim["frontier_paths"], n_paths)
        frontier = []
        for pol in POLICIES:
            knobs = sim["frontier_p0_buffers"] if pol == "P0_manual" else sim["frontier_service_targets"]
            for kv in knobs:
                sim_k = dict(sim, p0_buffer=kv) if pol == "P0_manual" else sim
                z_k = z if pol == "P0_manual" else stats.norm.ppf(kv)
                k_res, _ = simulate_policy(pol, cubes, demand[:fp], days, test_origins, resid_sd, z_k, price,
                                           mean_daily, L, P, sim_k)
                frontier.append(dict(policy=pol, knob=kv, fill_rate=k_res["fill_rate"],
                                     days_of_cover=k_res["avg_days_of_cover"],
                                     stockout_day_rate=k_res["stockout_day_rate"],
                                     total_cost_inr_per_day=k_res["total_cost_inr_per_day"]))

    base = results["P0_manual"]
    summary = dict(
        policies=results,
        stockout_reduction_P2_vs_P0=1 - results["P2_uncertainty"]["stockout_day_rate"] / max(base["stockout_day_rate"], 1e-9),
        inventory_change_P2_vs_P0=results["P2_uncertainty"]["avg_on_hand_units"] / max(base["avg_on_hand_units"], 1e-9) - 1,
        cost_change_P2_vs_P0=results["P2_uncertainty"]["total_cost_inr_per_day"] / max(base["total_cost_inr_per_day"], 1e-9) - 1,
        frontier=dict(points=frontier, cover_days_at_95_fill=cover_at_fill(frontier, 0.95)),
        n_series=int(n_s), n_days=int(len(days)), n_paths=int(n_paths), lead_time=L, review=R,
        service_target=sim["service_target"], window=[str(dates[0].date()), str(dates[-1].date())],
    )
    trace_df = pd.concat([pd.DataFrame(v, columns=["on_hand", "lost", "demand"]).assign(policy=k, date=dates)
                          for k, v in traces.items()])
    trace_df.to_parquet(art / "simulation_trace.parquet", index=False)
    (art / "simulation.json").write_text(json.dumps(summary, indent=2))
    orders = production_orders(cfg, art, z, P)
    log.info("simulation_done", extra={k: v for k, v in summary.items() if k != "policies"} | {"orders": len(orders)})
    return summary


def cover_at_fill(points: list[dict], target: float) -> dict:
    """Days of cover each policy needs to reach ``target`` fill rate (linear interpolation
    along its frontier; None if the sweep never reaches the target)."""
    out = {}
    df = pd.DataFrame(points)
    for pol, g in df.groupby("policy"):
        g = g.sort_values("fill_rate")
        if g.fill_rate.max() < target or g.fill_rate.min() > target:
            out[pol] = None
            continue
        out[pol] = float(np.interp(target, g.fill_rate.values, g.days_of_cover.values))
    return out


def simulate_policy(pol, cubes, demand, days, test_origins, resid_sd, z, price, mean_daily, L, P, sim):
    """Run one policy over all demand paths. Returns (kpis, daily trace)."""
    n_paths, n_days, n_s = demand.shape
    on_hand = np.broadcast_to(order_up_to(pol, cubes, 0, 1, P, resid_sd, z, sim["p0_buffer"]), (n_paths, n_s)).copy()
    arrivals = np.zeros((n_days + L + 1, n_paths, n_s))
    sold_t = lost_t = oh_t = over_t = ordered = hold = lost_margin = 0.0
    so_days = np.zeros(n_s)
    trace = np.zeros((n_days, 3))
    for t, d in enumerate(days):
        on_hand += arrivals[t]
        dem = demand[:, t]
        sold = np.minimum(dem, on_hand)
        lost = dem - sold
        on_hand -= sold
        sold_t += sold.sum()
        lost_t += lost.sum()
        oh_t += on_hand.sum()
        over_t += np.maximum(on_hand - sim["overstock_cover_days"] * mean_daily, 0).sum()
        hold += (on_hand * price).sum() * sim["holding_rate_daily"]
        lost_margin += (lost * price).sum() * sim["margin"]
        so_days += (lost > 0).mean(0)
        trace[t] = [on_hand.sum(1).mean(), lost.sum(1).mean(), dem.sum(1).mean()]
        # evening decision with the latest forecast origin <= d
        o_i = max(i for i, o in enumerate(test_origins) if o <= d)
        S = order_up_to(pol, cubes, o_i, d - test_origins[o_i] + 1, P, resid_sd, z, sim["p0_buffer"])
        order = np.maximum(S - (on_hand + arrivals[t + 1:t + L + 1].sum(0)), 0)
        arrivals[t + L] += order
        ordered += order.sum()
    pd_ = n_paths * n_days
    kpis = dict(
        fill_rate=float(sold_t / max(sold_t + lost_t, 1)),
        stockout_day_rate=float(so_days.sum() / (n_days * n_s)),
        stockout_days_per_series=float(so_days.sum() / n_s),
        avg_on_hand_units=float(oh_t / pd_),
        avg_days_of_cover=float((oh_t / pd_) / mean_daily.sum()),
        overstock_units_per_day=float(over_t / pd_),
        lost_units_per_day=float(lost_t / pd_),
        units_ordered_per_day=float(ordered / pd_),
        holding_cost_inr_per_day=float(hold / pd_),
        lost_margin_inr_per_day=float(lost_margin / pd_),
        total_cost_inr_per_day=float((hold + lost_margin) / pd_),
    )
    return kpis, trace


def production_orders(cfg: dict, art, z: float, P: int) -> pd.DataFrame:
    """Suggested orders for the production origin under policy P2 (what the planner reviews)."""
    prod = pd.read_parquet(art / "forecasts" / "production.parquet")
    series_tbl = pd.read_parquet(art / "series.parquet")
    win = prod[prod.h <= P]
    sd_day = np.maximum(win.p90c - win.p10c, 0) / Z_WIDTH
    agg = win.assign(var=sd_day**2).groupby("series").agg(mu=("final", "sum"), var=("var", "sum"),
                                                          p10=("p10c", "sum"), p90=("p90c", "sum"))
    agg["order_up_to"] = np.ceil(agg.mu + z * np.sqrt(agg["var"]))
    t = series_tbl.iloc[agg.index]
    agg["on_hand"] = t.on_hand.values
    agg["suggested_qty"] = np.maximum(agg.order_up_to - agg.on_hand, 0).astype(int)
    drv = [c for c in prod.columns if c.startswith("drv_") and c != "drv_baseline"]
    top = prod[prod.h <= P].groupby("series")[drv].mean()
    agg = agg.join(top)
    out = pd.concat([t[["store_id", "item_id", "category", "demand_class"]].reset_index(drop=True),
                     agg.reset_index()], axis=1)
    out["order_date"] = pd.Timestamp(prod.origin_date.iloc[0]).date().isoformat()
    out["policy"] = "P2_uncertainty"
    out.to_parquet(art / "orders_production.parquet", index=False)
    return out
