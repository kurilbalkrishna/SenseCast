"""End-to-end backtest: train on the train block, decide routing and calibrate on the
calibration block, score everything once on the test block.

Nothing in the test block influences any choice (model routing, cold-start method,
conformal widths, reconciliation weights).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sensecast.config import config_hash, resolve_path
from sensecast.evaluation.layout import origin_layout
from sensecast.features.build import FEATURES_M1, FEATURES_M2, FeatureBuilder, normalise_history, scale_for_aggregate
from sensecast.features.panel import LEVELS, build_aggregate_panel, build_bottom_panel, summing_matrix
from sensecast.logging_utils import get_logger, timed
from sensecast.models import baselines, coldstart, intermittent, lgbm
from sensecast.models import metrics as M
from sensecast.models.conformal import CQR, horizon_bucket
from sensecast.reconcile import mint

log = get_logger(__name__)
QS = (0.1, 0.5, 0.9)


class Context:
    """Everything later stages (faults, simulation, API) need from the backtest."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.H = cfg["backtest"]["horizon"]
        self.data_dir = resolve_path(cfg, "data")
        self.art = resolve_path(cfg, "artifacts")
        (self.art / "models").mkdir(exist_ok=True)
        self.panel = build_bottom_panel(self.data_dir / "processed", self.H)
        self.layout = origin_layout(self.panel.n_obs, cfg)
        self.train_end = self.layout["train"][-1] + self.H
        self.fb = FeatureBuilder(self.panel, clim_upto=self.train_end, seed=cfg["seed"])
        Ym = self.fb.hist["Ym"]
        ic = cfg["intermittent"]
        self.classes = intermittent.classify(Ym, self.train_end, ic["adi_cut"], ic["cv2_cut"])
        self.level = baselines.ses_level(Ym)
        self.cro = intermittent.croston_family(Ym, ic["alpha"], ic["beta"])
        self.scale = M.mase_scale(Ym, self.train_end)


def add_base_predictions(ctx: Context, df: pd.DataFrame, models: dict) -> pd.DataFrame:
    Ym = ctx.fb.hist["Ym"]
    df["b0"] = baselines.b0(df, Ym)
    df["b1"] = baselines.b1(df, ctx.level)
    b1q = baselines.nb_quantiles(df.b1.values, df.std28.values)
    df["b1_p10"], df["b1_p50"], df["b1_p90"] = b1q
    df["m1"] = lgbm.predict(models["m1"], df, FEATURES_M1)
    df["m2"] = lgbm.predict(models["m2"], df, FEATURES_M2)
    q = np.sort(np.column_stack([lgbm.predict(models[f"q{int(t * 100)}"], df, FEATURES_M2) for t in QS]), axis=1)
    df["m2_p10"], df["m2_p50"], df["m2_p90"] = np.maximum(q, 0).T
    for k in ("croston", "sba", "tsb"):
        df[k] = ctx.cro[k][df.origin.values, df.series.values]
    df["demand_class"] = ctx.classes.demand_class.values[df.series.values]
    return df


def training_rows(df: pd.DataFrame, censoring: str) -> pd.DataFrame:
    """Build the training target ``y_train`` from observed sales.

    Sales on a stockout day under-count demand (the shelf was empty). Three treatments:
      ignore  train on raw sales (biased low)
      drop    remove censored days (still biased low: the days left are the calmer ones)
      impute  'demand unconstraining': on censored days use max(sales, typical level from
              uncensored history at the origin) - a lower-bound-aware estimate of demand
    """
    df = df[df.y.notna()].copy()
    if censoring == "drop":
        df = df[~df.censored]
        df["y_train"] = df.y
    elif censoring == "ignore":
        df["y_train"] = df.y
    elif censoring == "impute":
        typical = np.fmax(df.sdw_mean4.fillna(df.mean28), df.mean7).fillna(df.y)
        df["y_train"] = np.where(df.censored, np.fmax(df.y, typical), df.y).astype(np.float32)
    else:
        raise ValueError(f"unknown censoring treatment {censoring}")
    return df.reset_index(drop=True)


def poisson_q(mean: np.ndarray) -> list[np.ndarray]:
    m = np.maximum(np.nan_to_num(mean), 1e-6)
    return [stats.poisson.ppf(q, m).astype(np.float32) for q in QS]


def apply_routing(df: pd.DataFrame, route: dict, young_route: str, young_age: int) -> pd.DataFrame:
    df["route"] = df.demand_class.map(lambda c: route.get(c, "m2"))
    young = df.age < young_age
    df.loc[young, "route"] = young_route
    df["final"] = np.float32(0)
    for r in df.route.unique():
        m = df.route == r
        df.loc[m, "final"] = df.loc[m, r].values
    df["p10"], df["p50"], df["p90"] = df.m2_p10, df.m2_p50, df.m2_p90
    non_m2 = df.route != "m2"
    if non_m2.any():
        q = poisson_q(df.loc[non_m2, "final"].values)
        df.loc[non_m2, "p10"], df.loc[non_m2, "p50"], df.loc[non_m2, "p90"] = q
    df["cls_group"] = np.where(young, "new", df.demand_class)
    return df


def day_type(df: pd.DataFrame, panel) -> np.ndarray:
    rain = panel.RAIN[df.day.values, panel.series.city_idx.values[df.series.values]]
    ev = (df.in_festival.values == 1) | ((df.days_to_event.values >= 0) & (df.days_to_event.values <= 4))
    return np.select([ev, rain >= 20, df.disc.values > 0], ["festival", "heavy_rain", "promotion"], "normal")


def run_backtest(cfg: dict) -> dict:
    t_start = time.perf_counter()
    ctx = Context(cfg)
    p, lay, H = ctx.panel, ctx.layout, ctx.H
    art = ctx.art

    with timed(log, "features"):
        tr = training_rows(ctx.fb.build(lay["train"], H), cfg["censoring"])
        cal = ctx.fb.build(lay["cal"], H)
        te = ctx.fb.build(lay["test"], H)
        prod = ctx.fb.build([lay["production"]], H, with_target=False)
    log.info("feature_rows", extra=dict(train=len(tr), cal=len(cal), test=len(te), prod=len(prod)))

    models = {}
    with timed(log, "train_models", rows=len(tr)):
        models["m1"] = lgbm.train(tr, FEATURES_M1, cfg, "tweedie", target="y_train")
        models["m2"] = lgbm.train(tr, FEATURES_M2, cfg, "tweedie", target="y_train")
        for t in QS:
            models[f"q{int(t * 100)}"] = lgbm.train(tr, FEATURES_M2, cfg, "quantile", alpha=t, target="y_train",
                                                    n_estimators=cfg["lgbm"]["quantile_estimators"])
    for name, mdl in models.items():
        mdl.save_model(str(art / "models" / f"{name}.txt"))

    with timed(log, "predict"):
        for df in (cal, te, prod):
            add_base_predictions(ctx, df, models)

    # ---------------- routing decided on the calibration block ---------------
    calv = cal[cal.y.notna() & ~cal.censored]
    route, route_scores = {}, {}
    for cls in ("intermittent", "lumpy"):
        sub = calv[(calv.demand_class == cls) & (calv.age >= cfg["coldstart"]["young_age_days"])]
        if len(sub) < 50:
            continue
        sc = {m: M.mase(sub, m, ctx.scale) for m in ("m2", "tsb", "sba", "croston")}
        route[cls] = min(sc, key=sc.get)
        route_scores[cls] = sc

    # ---------------- cold start --------------------------------------------
    cs = cfg["coldstart"]
    analogs = coldstart.analog_table(p.series, cs["k_analogs"])
    young_age = cs["young_age_days"]
    blocks = {"cal": (cal, lay["cal"][0]), "test": (te, lay["test"][0]), "prod": (prod, lay["production"])}
    ramps = {}
    t_cs = time.perf_counter()
    for name, (df, first_origin) in blocks.items():
        apply_routing(df, route, "m2", young_age)   # provisional so analogs have a 'final'
        ramp = coldstart.launch_ramp(p.Y, p.series, first_origin, analogs, cs["k_analogs"])
        ramps[name] = ramp.round(3).tolist()
        c1 = coldstart.forecast_c1(df, "final", p.series, analogs, ramp, cs["k_analogs"], young_age)
        cm = coldstart.category_mean(df, ctx.fb.hist["mean28"], p.series, young_age)
        df["c1"] = df["final"]
        df["catmean"] = df["final"]
        df.loc[c1.index, "c1"] = c1
        df.loc[cm.index, "catmean"] = cm

    log.info("coldstart_done", extra={"seconds": round(time.perf_counter() - t_cs, 2)})
    young_cal = cal[(cal.age < young_age) & cal.y.notna() & ~cal.censored]
    if len(young_cal) >= 30:
        ysc = {m: M.wape(young_cal.y, young_cal[m]) for m in ("m2", "c1", "catmean")}
        young_route = min(ysc, key=ysc.get)
    else:
        ysc, young_route = {}, "c1"
    for df, _ in blocks.values():
        apply_routing(df, route, young_route, young_age)

    # ---------------- conformal calibration ---------------------------------
    calv = cal[cal.y.notna() & ~cal.censored]
    cqr = CQR(cfg["conformal"]["alpha"])
    g = lambda d: pd.DataFrame(dict(hb=horizon_bucket(d.h.values), c=d.cls_group.values))  # noqa: E731
    cqr.fit(calv.y.values, calv.p10.values, calv.p90.values, g(calv))
    for df, _ in blocks.values():
        df["p10c"], df["p90c"] = cqr.apply(df.p10.values, df.p90.values, g(df))

    # per-series residual sd on calibration (used by inventory policy P1)
    resid_sd = (calv.assign(e=calv.y - calv.final).groupby("series").e.std()
                .reindex(range(p.N)).fillna(calv.assign(e=calv.y - calv.final).e.std()))

    # ---------------- explainability ----------------------------------------
    # TreeSHAP is expensive: precompute drivers for production rows (served by the API)
    # and a fixed random sample of test rows (global explanation + error analysis).
    with timed(log, "explain"):
        # drivers are shown as the average effect over the horizon: explain a subset of horizons
        sel = prod[prod.h.isin(cfg["explain"]["production_horizons"])]
        fam = lgbm.driver_families(models["m2"], sel, FEATURES_M2)
        fam.index = sel.index
        for c in fam.columns:
            prod[f"drv_{c.replace('/', '_')}"] = np.nan
            prod.loc[sel.index, f"drv_{c.replace('/', '_')}"] = fam[c].values
        n_s = min(len(te), cfg["explain"]["test_sample_rows"])
        sample = te.sample(n=n_s, random_state=cfg["seed"])
        fam_t = lgbm.driver_families(models["m2"], sample, FEATURES_M2)
        fam_t.index = sample.index
        for c in fam_t.columns:
            te[f"drv_{c.replace('/', '_')}"] = np.nan
            te.loc[sample.index, f"drv_{c.replace('/', '_')}"] = fam_t[c].values
    imp = lgbm.importance(models["m2"], FEATURES_M2)
    mean_abs_driver = fam_t.drop(columns="baseline").abs().mean().sort_values(ascending=False).round(4).to_dict()

    # ---------------- aggregates + reconciliation ---------------------------
    with timed(log, "reconcile"):
        rec = reconcile_block(ctx, cal, te, prod)

    # ---------------- evaluation --------------------------------------------
    tev = te[te.y.notna() & ~te.censored].copy()
    tev["day_type"] = day_type(tev, p)
    truth_path = ctx.data_dir / "ground_truth" / "truth.parquet"
    results = evaluate(ctx, tev, te, cfg, truth_path)
    results.update(
        routing=dict(route=route, scores=route_scores, young_route=young_route, young_scores=ysc),
        conformal=cqr.to_dict(), launch_ramp=ramps, reconciliation=rec["metrics"],
        importance_by_family=imp.groupby("family").gain.sum().sort_values(ascending=False).round(4).to_dict(),
        mean_abs_log_driver=mean_abs_driver,
        rows=dict(train=len(tr), cal=len(cal), test=len(te), prod=len(prod)),
        layout={k: v for k, v in lay.items()}, config_hash=config_hash(cfg), profile=cfg["profile"],
        censoring=cfg["censoring"],
        dates=dict(start=p.meta["start"], last_obs=p.meta["last_obs"],
                   test_first_origin=str(p.dates[lay["test"][0]].date()),
                   test_last_day=str(p.dates[lay["test"][-1] + H].date())),
    )
    results["runtime_seconds"] = round(time.perf_counter() - t_start, 1)

    # ---------------- persist ------------------------------------------------
    keys = p.series[["store_id", "item_id", "category", "city"]]
    def with_keys(df):
        out = df.join(keys, on="series")
        out["date"] = p.dates[out.day.values]
        out["origin_date"] = p.dates[out.origin.values]
        return out
    fcols = ["origin", "day", "series", "h", "y", "censored", "b0", "b1", "b1_p10", "b1_p90", "m1", "m2",
             "m2_p10", "m2_p50", "m2_p90", "tsb", "sba", "croston", "c1", "catmean", "final", "p10", "p50", "p90",
             "p10c", "p90c", "route", "demand_class", "cls_group", "age", "disc", "days_to_event", "event_id",
             "in_festival", "temp_fc", "rain_fc", "search_ratio", "mean28"]
    drv = [c for c in te.columns if c.startswith("drv_")]
    fdir = art / "forecasts"
    fdir.mkdir(exist_ok=True)
    with_keys(te[fcols + drv]).to_parquet(fdir / "test.parquet", index=False)
    with_keys(cal[[c for c in fcols if c in cal]]).to_parquet(fdir / "cal.parquet", index=False)
    with_keys(prod[[c for c in fcols if c in prod] + drv]).to_parquet(fdir / "production.parquet", index=False)
    rec["agg_test"].to_parquet(fdir / "aggregates_test.parquet", index=False)
    rec["agg_prod"].to_parquet(fdir / "aggregates_production.parquet", index=False)
    series_tbl = p.series[["store_id", "item_id", "category", "city", "price", "pack_size", "launch_date"]].copy()
    series_tbl["demand_class"] = ctx.classes.demand_class.values
    series_tbl["resid_sd"] = resid_sd.values
    series_tbl["mase_scale"] = ctx.scale
    last = p.n_obs - 1
    sales = pd.read_parquet(ctx.data_dir / "processed" / "sales.parquet", columns=["date", "store_id", "item_id", "on_hand_close"])
    oh = sales[sales.date == p.dates[last]].set_index(["store_id", "item_id"]).on_hand_close
    series_tbl["on_hand"] = oh.reindex(pd.MultiIndex.from_frame(series_tbl[["store_id", "item_id"]])).fillna(0).values
    series_tbl.to_parquet(art / "series.parquet", index=False)
    imp.to_csv(art / "feature_importance.csv", index=False)
    hist_days = p.dates[max(0, last - 90): last + 1]
    hist = pd.DataFrame(p.Y[max(0, last - 90): last + 1], index=hist_days, columns=p.series.node_id)
    hist.rename_axis("date").reset_index().melt("date", var_name="node_id", value_name="units").dropna().to_parquet(
        art / "history_recent.parquet", index=False)
    (art / "metrics.json").write_text(json.dumps(results, indent=2, default=_json_default))
    log_mlflow(cfg, results, art)
    log.info("backtest_done", extra=dict(runtime=results["runtime_seconds"], wape_final=results["overall"]["final"]["wape"]))
    return results


def reconcile_block(ctx: Context, cal: pd.DataFrame, te: pd.DataFrame, prod: pd.DataFrame) -> dict:
    cfg, p, lay, H = ctx.cfg, ctx.panel, ctx.layout, ctx.H
    ap = build_aggregate_panel(p)
    fba = FeatureBuilder(ap, clim_upto=ctx.train_end, seed=cfg["seed"])
    feats = FEATURES_M2 + ["level_idx"]

    def prep(origins, with_target=True):
        df = fba.build(origins, H, with_target=with_target)
        sc = scale_for_aggregate(df)
        df = normalise_history(df, sc)
        df["scale"] = sc
        if with_target:
            df["ratio"] = df.y / sc
        return df

    atr = prep(lay["train"])
    atr = atr[atr.ratio.notna()]
    agg_model = lgbm.train(atr, feats, cfg, "regression", target="ratio")
    agg_model.save_model(str(ctx.art / "models" / "aggregate.txt"))

    S, ids, levels = summing_matrix(p.series)
    n_agg = len(ids) - p.N

    def matrices(bottom_df, origins, with_target=True):
        adf = prep(origins, with_target)
        adf["base"] = np.maximum(lgbm.predict(agg_model, adf, feats) * adf.scale.values, 0)
        cases = [(o, o + h) for o in origins for h in range(1, H + 1)]
        cidx = {c: i for i, c in enumerate(cases)}
        Yh = np.zeros((len(ids), len(cases)))
        Ya = np.full((len(ids), len(cases)), np.nan)
        ci = np.array([cidx[(o, d)] for o, d in zip(adf.origin.values, adf.day.values, strict=False)])
        Yh[adf.series.values, ci] = adf.base.values
        if with_target:
            Ya[adf.series.values, ci] = adf.y.values
        bi = np.array([cidx[(o, d)] for o, d in zip(bottom_df.origin.values, bottom_df.day.values, strict=False)])
        Yh[n_agg + bottom_df.series.values, bi] = bottom_df.final.values
        if with_target:
            Ya[n_agg + bottom_df.series.values, bi] = bottom_df.y.values
        return Yh, Ya, cases

    Yh_cal, Ya_cal, _ = matrices(cal, lay["cal"])
    E = np.nan_to_num(Ya_cal - Yh_cal).T
    W, lam = mint.shrunk_covariance(E)
    G_mint = mint.mint_projection(S, W)
    G_wls = mint.mint_projection(S, np.diag(np.diag(W)))
    G_bu = mint.bottom_up_projection(S)

    Yh, Ya, cases = matrices(te, lay["test"])
    Y_bu = mint.reconcile(G_bu, Yh, S)
    Y_wls = mint.reconcile(G_wls, Yh, S)
    Y_mt = mint.reconcile(G_mint, Yh, S)
    lv = np.array(levels)
    rows = []
    for L in LEVELS:
        m = lv == L
        y = Ya[m].ravel()
        ok = ~np.isnan(y)
        for name, Yx in (("base", Yh), ("bottom_up", Y_bu), ("wls_var", Y_wls), ("mint_shrink", Y_mt)):
            rows.append(dict(level=L, method=name, wape=M.wape(y[ok], Yx[m].ravel()[ok])))
    lvl_tbl = pd.DataFrame(rows).pivot(index="level", columns="method", values="wape").reindex(LEVELS)

    def agg_frame(Yx, Yb, cases_, extra=None):
        recs = []
        for j, (o, d) in enumerate(cases_):
            for i in range(n_agg):
                recs.append((ids[i], levels[i], o, d, Yb[i, j], Yx[i, j]))
        df = pd.DataFrame(recs, columns=["node_id", "level", "origin", "day", "base", "reconciled"])
        df["date"] = p.dates[df.day.values]
        if extra is not None:
            df["y"] = [extra[i, j] for j in range(len(cases_)) for i in range(n_agg)]
        return df

    served = cfg["reconciliation"]["method"]
    G_served = {"bottom_up": G_bu, "wls_var": G_wls, "mint_shrink": G_mint}[served]
    Y_served = {"bottom_up": Y_bu, "wls_var": Y_wls, "mint_shrink": Y_mt}[served]
    Yh_p, _, cases_p = matrices(prod, [lay["production"]], with_target=False)
    Y_mt_p = mint.reconcile(G_served, Yh_p, S)
    metrics_ = dict(
        served_method=served,
        coherence_error_served=mint.coherence_error(Y_served, S),
        shrinkage_lambda=round(lam, 4),
        coherence_error_base=mint.coherence_error(Yh, S),
        coherence_error_mint=mint.coherence_error(Y_mt, S),
        coherence_error_bottom_up=mint.coherence_error(Y_bu, S),
        coherence_error_wls=mint.coherence_error(Y_wls, S),
        wape_by_level=lvl_tbl.round(4).to_dict(orient="index"),
        n_nodes=len(ids), n_aggregate_nodes=n_agg,
    )
    return dict(metrics=metrics_, agg_test=agg_frame(Y_served, Yh, cases, Ya), agg_prod=agg_frame(Y_mt_p, Yh_p, cases_p))


def evaluate(ctx: Context, tev: pd.DataFrame, te: pd.DataFrame, cfg: dict, truth_path: Path) -> dict:
    sc = ctx.scale
    models = ["b0", "b1", "m1", "m2", "final"]
    overall = {m: dict(wape=M.wape(tev.y, tev[m]), mase=M.mase(tev, m, sc), bias=M.bias(tev.y, tev[m])) for m in models}
    pin = dict(
        b1=M.mean_pinball(tev.y, tev.b1_p10, tev.b1_p50, tev.b1_p90),
        final_raw=M.mean_pinball(tev.y, tev.p10, tev.p50, tev.p90),
        final_conformal=M.mean_pinball(tev.y, tev.p10c, tev.p50, tev.p90c),
    )
    cov = dict(
        b1=M.coverage(tev.y, tev.b1_p10, tev.b1_p90),
        final_raw=M.coverage(tev.y, tev.p10, tev.p90),
        final_conformal=M.coverage(tev.y, tev.p10c, tev.p90c),
    )
    cov_by_group = (tev.assign(inside=(tev.y >= tev.p10c) & (tev.y <= tev.p90c))
                    .groupby("cls_group").inside.mean().round(4).to_dict())

    def cut(col):
        out = {}
        for k, g in tev.groupby(col):
            out[str(k)] = {m: round(M.wape(g.y, g[m]), 4) for m in models} | {"rows": int(len(g))}
        return out

    young = tev[tev.age < cfg["coldstart"]["young_age_days"]]
    cold = {m: round(M.wape(young.y, young[m]), 4) for m in ("catmean", "m2", "c1", "final")} | {"rows": int(len(young))}
    sens_days = tev[tev.day_type.isin(["festival", "heavy_rain"])]
    sensing = dict(m1=M.wape(sens_days.y, sens_days.m1), m2=M.wape(sens_days.y, sens_days.m2), rows=int(len(sens_days)))

    # store-level MASE: aggregate item forecasts to store-day per origin
    store_of = ctx.panel.series.store_idx.values
    Ym0 = np.nan_to_num(ctx.fb.hist["Ym"][: ctx.train_end + 1])
    n_st = int(store_of.max()) + 1
    s_scale = np.array([np.mean(np.abs(Ym0[7:, store_of == s].sum(1) - Ym0[:-7, store_of == s].sum(1)))
                        for s in range(n_st)])
    st = tev.assign(store=store_of[tev.series.values])
    store_mase = {}
    for m in ("b0", "b1", "m2", "final"):
        agg = st.groupby(["store", "origin", "day"])[["y", m]].sum().reset_index().rename(columns={"store": "series"})
        store_mase[m] = M.mase(agg, m, s_scale)

    # accuracy against TRUE demand (possible only because the world is synthetic)
    vs_truth = {}
    if truth_path.exists():
        truth = pd.read_parquet(truth_path, columns=["date", "store_id", "item_id", "demand"])
        keys = ctx.panel.series[["store_id", "item_id"]]
        t2 = te.join(keys, on="series")
        t2["date"] = ctx.panel.dates[t2.day.values]
        t2 = t2.merge(truth, on=["date", "store_id", "item_id"], how="left")
        vs_truth = {m: round(M.wape(t2.demand, t2[m]), 4) for m in ("b0", "b1", "m2", "final")}
        top = t2.groupby("series").demand.mean()
        top_ids = top[top >= top.quantile(0.9)].index
        tt = t2[t2.series.isin(top_ids)]
        vs_truth["bias_top10pct_final"] = round(M.bias(tt.demand, tt.final), 4)
        vs_truth["bias_final"] = round(M.bias(t2.demand, t2.final), 4)

    tev["store_id"] = ctx.panel.series.store_id.values[tev.series.values]
    per_class = cut("cls_group")
    return dict(
        overall=overall, pinball=pin, coverage=cov, coverage_by_group=cov_by_group,
        stability=dict(final=M.stability(te[te.y.notna()], "final"), m2=M.stability(te[te.y.notna()], "m2"),
                       b0=M.stability(te[te.y.notna()], "b0")),
        by_class=per_class, by_day_type=cut("day_type"), by_store=cut("store_id"),
        coldstart=cold, sensing_days=sensing, store_mase=store_mase, vs_true_demand=vs_truth,
        mase_intermittent={m: M.mase(tev[tev.cls_group.isin(["intermittent", "lumpy"])], m, sc)
                           for m in ("b0", "m2", "tsb", "sba", "final")},
    )


def log_mlflow(cfg: dict, results: dict, art: Path) -> None:
    if not cfg["mlflow"]["enabled"]:
        return
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        log.warning("mlflow_missing")
        return
    try:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        with mlflow.start_run(run_name=f"backtest-{results['config_hash']}"):
            mlflow.log_params({f"lgbm.{k}": v for k, v in cfg["lgbm"].items()})
            mlflow.log_params({"profile": cfg["profile"], "seed": cfg["seed"], "config_hash": results["config_hash"]})
            for m, v in results["overall"].items():
                mlflow.log_metrics({f"{m}.wape": v["wape"], f"{m}.mase": v["mase"]})
            mlflow.log_metrics({f"pinball.{k}": v for k, v in results["pinball"].items()})
            mlflow.log_metrics({f"coverage.{k}": v for k, v in results["coverage"].items()})
            mlflow.log_artifact(str(art / "metrics.json"))
            mlflow.log_artifact(str(art / "feature_importance.csv"))
            for f in (art / "models").glob("*.txt"):
                mlflow.log_artifact(str(f), artifact_path="models")
    except Exception as exc:  # noqa: BLE001 - tracking must never break the pipeline
        log.warning("mlflow_failed", extra={"err": str(exc)[:200]})


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
