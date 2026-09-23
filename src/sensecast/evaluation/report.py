"""Evaluation dossier: go/no-go gates, figures and a generated Markdown report.

Output: artifacts/gates.json, docs/03-evaluation/results.md, docs/03-evaluation/figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sensecast.config import resolve_path  # noqa: E402
from sensecast.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)

INK, ACCENT, MUTED, WARN, BLUE, PINK = "#14202E", "#0C7A68", "#8A94A0", "#B8710C", "#2C68CF", "#A8347F"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9AA3AD", "axes.titleweight": "bold", "figure.dpi": 130})


def _load(art: Path, name: str) -> dict:
    p = art / name
    return json.loads(p.read_text()) if p.exists() else {}


def evaluate_gates(cfg: dict, m: dict, sim: dict, faults: dict, load: dict, run: dict) -> list[dict]:
    th = cfg["thresholds"]
    ov = m["overall"]
    g = []

    def add(name, value, target, ok, detail=""):
        g.append(dict(gate=name, value=value, target=target, result="PASS" if ok else ("N/A" if ok is None else "FAIL"),
                      detail=detail))

    gain = 1 - ov["final"]["wape"] / ov["b0"]["wape"]
    add("WAPE vs seasonal naive", f"{gain:.1%} better", f">= {th['wape_gain_vs_b0']:.0%} better", gain >= th["wape_gain_vs_b0"],
        f"final {ov['final']['wape']:.3f} vs B0 {ov['b0']['wape']:.3f}")
    sm = m["store_mase"]["final"]
    add("MASE, store level", f"{sm:.3f}", f"< {th['mase_store_max']}", sm < th["mase_store_max"])
    sd = m["sensing_days"]
    sg = 1 - sd["m2"] / sd["m1"]
    add("Demand-sensing gain (festival + heavy-rain days)", f"{sg:.1%}", f">= {th['sensing_gain_event_days']:.0%}",
        sg >= th["sensing_gain_event_days"], f"M2 {sd['m2']:.3f} vs M1 {sd['m1']:.3f} on {sd['rows']:,} rows")
    pg = 1 - m["pinball"]["final_conformal"] / m["pinball"]["b1"]
    add("Pinball loss vs planner quantiles", f"{pg:.1%} better", f">= {th['pinball_gain_vs_b1']:.0%} better",
        pg >= th["pinball_gain_vs_b1"])
    cov = m["coverage"]["final_conformal"]
    add("P10-P90 coverage", f"{cov:.1%}", f"{th['coverage_low']:.0%}-{th['coverage_high']:.0%}",
        th["coverage_low"] <= cov <= th["coverage_high"],
        "by class: " + ", ".join(f"{k} {v:.0%}" for k, v in m["coverage_by_group"].items()))
    mi = m["mase_intermittent"]
    add("Intermittent + lumpy MASE", f"{mi['final']:.3f}", f"<= B0 ({mi['b0']:.3f})", mi["final"] <= mi["b0"])
    cs = m["coldstart"]
    cg = 1 - cs["final"] / cs["catmean"] if cs.get("catmean") else float("nan")
    add("Cold-start WAPE, first 28 days", f"{cg:.1%} better", f">= {th['coldstart_gain_vs_catmean']:.0%} better",
        cg >= th["coldstart_gain_vs_catmean"], f"final {cs['final']:.3f} vs category mean {cs['catmean']:.3f} ({cs['rows']:,} rows)")
    ce = m["reconciliation"]["coherence_error_served"]
    add("Hierarchy coherence error", f"{ce:.1e}", f"< {th['coherence_max_error']:.0e}", ce < th["coherence_max_error"],
        f"served method: {m['reconciliation']['served_method']}")
    if sim:
        p2 = sim["policies"]["P2_uncertainty"]
        add("Fill rate at 95% service target (P2)", f"{p2['fill_rate']:.1%}", f">= {th['fill_rate_min']:.0%}",
            p2["fill_rate"] >= th["fill_rate_min"])
        fr = sim.get("frontier", {}).get("cover_days_at_95_fill", {})
        if fr.get("P0_manual") and fr.get("P2_uncertainty"):
            chg = fr["P2_uncertainty"] / fr["P0_manual"] - 1
            add("Inventory needed for a 95% fill rate, P2 vs P0", f"{chg:+.1%}",
                f"<= {th['inventory_at_95_fill_vs_p0']:+.0%}", chg <= th["inventory_at_95_fill_vs_p0"],
                f"P0 {fr['P0_manual']:.2f} vs P2 {fr['P2_uncertainty']:.2f} days of cover")
        red = sim["stockout_reduction_P2_vs_P0"]
        add("Stockout days vs manual policy", f"{red:.1%} fewer", f">= {th['stockout_reduction_vs_p0']:.0%} fewer",
            red >= th["stockout_reduction_vs_p0"])
    st = m["stability"]["final"]
    add("Forecast stability", f"{st:.3f}", f"<= {th['stability_max']}", st <= th["stability_max"])
    if faults:
        lp = faults["leakage_probe"]
        add("Leakage probe (future scrambled)", f"{len(lp['features_changed'])} features changed", "0", lp["passed"])
    if load:
        add("API latency p95", f"{load['p95_ms']:.0f} ms", f"< {th['api_p95_ms']} ms", load["p95_ms"] < th["api_p95_ms"],
            f"{load['requests']} requests, {load['concurrency']} concurrent")
    else:
        add("API latency p95", "not measured", f"< {th['api_p95_ms']} ms", None, "run `make loadtest`")
    if run:
        add("Full batch run", f"{run['minutes']:.1f} min", f"< {th['batch_minutes_max']} min",
            run["minutes"] < th["batch_minutes_max"])
    return g


# ------------------------------------------------------------------ figures
def fig_ladder(m, out):
    ov = m["overall"]
    names = {"b0": "B0 seasonal\nnaive", "b1": "B1 planner", "m1": "M1 LightGBM\nsales only",
             "m2": "M2 LightGBM\nmultisource", "final": "SenseCast\nrouted"}
    vals = [ov[k]["wape"] for k in names]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(list(names.values()), vals, color=[MUTED, MUTED, BLUE, ACCENT, INK])
    ax.bar_label(bars, fmt="%.3f", padding=2)
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_title("Model ladder on the test block")
    fig.tight_layout()
    fig.savefig(out / "ladder.png")
    plt.close(fig)


def fig_day_type(m, out):
    d = pd.DataFrame(m["by_day_type"]).T
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(d))
    for i, (col, c) in enumerate([("b0", MUTED), ("m1", BLUE), ("m2", ACCENT)]):
        ax.bar(x + (i - 1) * 0.26, d[col], 0.26, label=col.upper(), color=c)
    ax.set_xticks(x, [f"{k}\n({int(r):,} rows)" for k, r in zip(d.index, d.rows, strict=False)])
    ax.set_ylabel("WAPE")
    ax.set_title("Where multisource sensing helps")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "day_type.png")
    plt.close(fig)


def fig_recon(m, out):
    d = pd.DataFrame(m["reconciliation"]["wape_by_level"]).T
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(d))
    cols = [c for c in ("base", "bottom_up", "wls_var", "mint_shrink") if c in d]
    colors = [MUTED, BLUE, WARN, ACCENT]
    for i, c in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * 0.2, d[c], 0.2, label=c, color=colors[i])
    ax.set_xticks(x, d.index)
    ax.set_ylabel("WAPE")
    ax.set_title("Reconciliation by hierarchy level")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "reconciliation.png")
    plt.close(fig)


def fig_sim(sim, out):
    if not sim:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    pts = pd.DataFrame(sim.get("frontier", {}).get("points", []))
    colors = {"P0_manual": MUTED, "P1_point_ml": BLUE, "P2_uncertainty": ACCENT}
    for pol, g in pts.groupby("policy") if not pts.empty else []:
        g = g.sort_values("days_of_cover")
        ax.plot(g.days_of_cover, g.fill_rate * 100, marker="o", color=colors[pol], label=pol.replace("_", " "))
    for pol, r in sim["policies"].items():
        ax.scatter(r["avg_days_of_cover"], r["fill_rate"] * 100, s=140, facecolors="none", edgecolors=colors[pol], lw=2)
    ax.axhline(95, color=INK, lw=0.8, ls="--")
    ax.set_xlabel("Average days of cover held")
    ax.set_ylabel("Fill rate (%)")
    ax.set_title("Service vs inventory frontier (up-left is better)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "simulation.png")
    plt.close(fig)


def fig_coverage(m, out):
    d = pd.Series(m["coverage_by_group"])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(d.index, d.values * 100, color=ACCENT)
    ax.axhspan(75, 85, color=ACCENT, alpha=0.1)
    ax.axhline(80, color=INK, lw=0.8, ls="--")
    ax.set_ylabel("Coverage of P10-P90 (%)")
    ax.set_ylim(50, 100)
    ax.set_title("Calibrated interval coverage by demand class")
    fig.tight_layout()
    fig.savefig(out / "coverage.png")
    plt.close(fig)


def fig_drivers(m, out):
    d = pd.Series(m.get("mean_abs_log_driver", {}))
    if d.empty:
        return
    d = d.sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(d.index, (np.exp(d.values) - 1) * 100, color=ACCENT)
    ax.set_xlabel("Mean absolute effect on forecast (%)")
    ax.set_title("What moves forecasts (SHAP, test sample)")
    fig.tight_layout()
    fig.savefig(out / "drivers.png")
    plt.close(fig)


def fig_recovery(cfg, out) -> dict:
    """Partial dependence of M2 on rain and temperature vs the TRUE effect in the generator."""
    import lightgbm as lgb

    from sensecast.evaluation.backtest import Context
    from sensecast.features.build import FEATURES_M2
    ctx = Context(cfg)
    eff = json.loads((ctx.data_dir / "ground_truth" / "effects.json").read_text())
    m2 = lgb.Booster(model_file=str(ctx.art / "models" / "m2.txt"))
    te = ctx.fb.build(ctx.layout["test"], ctx.H)
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, (feat, cat, grid, label, truth_fn) in zip(axes, [
        ("rain_fc", 3, np.log1p(np.array([0, 2, 5, 10, 20, 40, 80])), "rain forecast (mm)",
         lambda g: eff["beta_rain"]["monsoon_seasonal"] * g),
        ("temp_fc", 0, np.array([27, 29, 31, 33, 35, 37]), "max temperature (degC)",
         lambda g: eff["beta_temp"]["beverages"] * (g - 30)),
    ], strict=False):
        pool = te[(te.cat_idx == cat) & (te.h <= 2)]  # shortest horizons: least forecast noise
        sub = pool.sample(n=min(4000, len(pool)), random_state=0)
        pd_vals = []
        for gv in grid:
            s2 = sub.copy()
            s2[feat] = gv
            pd_vals.append(np.log(np.maximum(m2.predict(s2[FEATURES_M2]), 1e-6)).mean())
        pd_vals = np.array(pd_vals) - pd_vals[0]
        true = truth_fn(grid) - truth_fn(grid)[0]
        xs = np.expm1(grid) if feat == "rain_fc" else grid
        ax.plot(xs, (np.exp(true) - 1) * 100, color=MUTED, ls="--", lw=2, label="true effect (generator)")
        ax.plot(xs, (np.exp(pd_vals) - 1) * 100, color=ACCENT, lw=2.4, marker="o", label="learned by M2")
        ax.set_xlabel(label)
        ax.set_ylabel("% change in demand")
        ax.set_title("Monsoon items vs rain" if cat == 3 else "Beverages vs heat")
        ax.legend(frameon=False, fontsize=8)
        res[feat] = dict(grid=xs.round(1).tolist(), learned_pct=((np.exp(pd_vals) - 1) * 100).round(1).tolist(),
                         true_pct=((np.exp(true) - 1) * 100).round(1).tolist())
    fig.tight_layout()
    fig.savefig(out / "effect_recovery.png")
    plt.close(fig)
    return res


def fig_example(art, out) -> None:
    """One Mumbai snacks item through Ganesh Chaturthi, forecast from the last test origin."""
    from sensecast.generate.events import EVENT_NAMES

    te = pd.read_parquet(art / "forecasts" / "test.parquet")
    ganesh = EVENT_NAMES.index("Ganesh Chaturthi") + 1  # +1: event ids are shifted for LightGBM
    last_o = te.origin.max()
    cand = te[(te.origin == last_o) & (te.category == "snacks") & (te.city == "Mumbai") &
              (te.demand_class == "smooth") & (te.event_id == ganesh)]
    if cand.empty:
        cand = te[(te.origin == last_o) & (te.category == "snacks") & (te.in_festival == 1)]
    if cand.empty:
        return
    s = cand.groupby("series").y.sum().idxmax()
    d = te[(te.series == s) & (te.origin == last_o)].sort_values("day")
    node = f"{d.store_id.iloc[0]}|{d.item_id.iloc[0]}"
    hist = pd.read_parquet(art / "history_recent.parquet")
    hist = hist[(hist.node_id == node) & (hist.date < d.date.min())].tail(28)
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    ax.plot(hist.date, hist.units, color=INK, lw=1.4, label="actual")
    ax.plot(d.date, d.y, color=INK, lw=1.4)
    ax.fill_between(d.date, d.p10c, d.p90c, color=ACCENT, alpha=0.18, label="P10-P90 (calibrated)")
    ax.plot(d.date, d.final, color=ACCENT, lw=2.4, label="SenseCast")
    ax.plot(d.date, d.m1, color=MUTED, lw=1.6, ls="--", label="M1 sales-only")
    ax.axvline(d.date.min() - pd.Timedelta(hours=12), color=INK, lw=0.8)
    for x in d[d.in_festival == 1].date:
        ax.axvspan(x - pd.Timedelta(hours=12), x + pd.Timedelta(hours=12), color=WARN, alpha=0.10, lw=0)
    ax.set_title(f"{d.store_id.iloc[0]} {d.item_id.iloc[0]} (snacks, Mumbai) through Ganesh Chaturthi")
    ax.set_ylabel("units / day")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "example_festival.png")
    plt.close(fig)


# ------------------------------------------------------------------ report
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def build_report(cfg: dict) -> dict:
    art = resolve_path(cfg, "artifacts")
    docs = resolve_path(cfg, "reports")
    figs = docs / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    m = _load(art, "metrics.json")
    sim, faults, mon = _load(art, "simulation.json"), _load(art, "faults.json"), _load(art, "monitoring.json")
    load, run = _load(art, "loadtest.json"), _load(art, "run_summary.json")
    gates = evaluate_gates(cfg, m, sim, faults, load, run)
    (art / "gates.json").write_text(json.dumps({"gates": gates}, indent=2))

    fig_ladder(m, figs)
    fig_day_type(m, figs)
    fig_recon(m, figs)
    fig_sim(sim, figs)
    fig_coverage(m, figs)
    fig_drivers(m, figs)
    fig_example(art, figs)
    recovery = fig_recovery(cfg, figs)
    (art / "effect_recovery.json").write_text(json.dumps(recovery, indent=2))

    ov = pd.DataFrame(m["overall"]).T.round(4).reset_index().rename(columns={"index": "model"})
    g = pd.DataFrame(gates)[["gate", "value", "target", "result"]]
    n_pass = int((g.result == "PASS").sum())
    lines = [
        "# Evaluation results (generated)",
        "",
        f"Generated by `sensecast report` from run `{m.get('config_hash')}` (profile `{m.get('profile')}`).",
        f"Test block: origins from {m['dates']['test_first_origin']} to the last target day {m['dates']['test_last_day']}.",
        "Do not edit by hand; rerun the pipeline instead. The narrative lives in `evaluation_report.md`.",
        "",
        f"## Go / no-go gates: {n_pass} of {len(g)} pass",
        "",
        _md_table(g),
        "",
        "## Model ladder (item-store-day, test block, non-censored days)",
        "",
        _md_table(ov),
        "",
        "![ladder](figures/ladder.png)",
        "",
        "## Accuracy by day type",
        "",
        _md_table(pd.DataFrame(m["by_day_type"]).T.reset_index().rename(columns={"index": "day type"})),
        "",
        "![day type](figures/day_type.png)",
        "",
        "## Accuracy by demand class",
        "",
        _md_table(pd.DataFrame(m["by_class"]).T.reset_index().rename(columns={"index": "class"})),
        "",
        "## Intervals",
        "",
        _md_table(pd.DataFrame(dict(pinball=m["pinball"], coverage=m["coverage"])).round(4).reset_index()
                  .rename(columns={"index": "forecast"})),
        "",
        "![coverage](figures/coverage.png)",
        "",
        "## Reconciliation (WAPE by level)",
        "",
        _md_table(pd.DataFrame(m["reconciliation"]["wape_by_level"]).T.reset_index().rename(columns={"index": "level"})),
        "",
        f"Shrinkage intensity lambda = {m['reconciliation']['shrinkage_lambda']}; base forecasts were incoherent by up to "
        f"{m['reconciliation']['coherence_error_base']:.0f} units; reconciled error {m['reconciliation']['coherence_error_mint']:.1e}.",
        "",
        "![reconciliation](figures/reconciliation.png)",
        "",
        "## Cold start (first 28 days of an item's life)",
        "",
        _md_table(pd.DataFrame([m["coldstart"]])),
        "",
        f"Routing chosen on the calibration block: {json.dumps(m['routing']['route'])}; new items -> `{m['routing']['young_route']}`.",
        "",
        "## Inventory simulation",
        "",
    ]
    if sim:
        p = pd.DataFrame(sim["policies"]).T.round(4).reset_index().rename(columns={"index": "policy"})
        lines += [_md_table(p), "", "![simulation](figures/simulation.png)", ""]
    lines += ["## Does the model recover the true effects?", "", "![recovery](figures/effect_recovery.png)", ""]
    for _feat, r in recovery.items():
        lines.append(_md_table(pd.DataFrame(r)))
        lines.append("")
    lines += ["## Explainability", "", "![drivers](figures/drivers.png)", "", "## Example", "",
              "![example](figures/example_festival.png)", ""]
    if faults:
        lines += ["## Fault suite", "", "```json", json.dumps(faults, indent=2), "```", ""]
    if mon:
        lines += ["## Monitoring", "", f"PSI alerts (> {mon['psi_threshold']}): {', '.join(mon['psi_alerts']) or 'none'}",
                  f"Store bias alerts: {mon['store_bias_alerts']}", ""]
    (docs / "results.md").write_text("\n".join(lines))
    log.info("report_done", extra={"gates_passed": n_pass, "gates_total": len(g)})
    return {"gates": gates}
