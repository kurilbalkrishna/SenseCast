"""End-to-end run of the small profile."""
import json
from pathlib import Path

import pandas as pd


def test_artifacts_exist(pipeline):
    art = Path(pipeline["paths"]["artifacts"])
    for name in ("metrics.json", "simulation.json", "faults.json", "monitoring.json", "gates.json",
                 "series.parquet", "orders_production.parquet", "forecasts/test.parquet",
                 "forecasts/production.parquet", "forecasts/aggregates_production.parquet", "models/m2.txt"):
        assert (art / name).exists(), name


def test_processed_data_has_manifest_with_hashes(pipeline):
    man = json.loads((Path(pipeline["paths"]["data"]) / "processed" / "manifest.json").read_text())
    assert len(man["files"]["sales.parquet"]) == 64
    assert man["dq"]["rows_quarantined"] > 0


def test_forecasts_are_sane(pipeline):
    art = Path(pipeline["paths"]["artifacts"])
    te = pd.read_parquet(art / "forecasts" / "test.parquet")
    assert (te.final >= 0).all()
    assert (te.p10c <= te.p90c).all()
    assert te.route.isin(["m2", "tsb", "sba", "croston", "c1", "catmean"]).all()
    m = json.loads((art / "metrics.json").read_text())
    assert m["reconciliation"]["coherence_error_mint"] < 1e-6
    # the global model must beat the naive floor even on the tiny profile
    assert m["overall"]["final"]["wape"] < m["overall"]["b0"]["wape"]


def test_production_orders(pipeline):
    o = pd.read_parquet(Path(pipeline["paths"]["artifacts"]) / "orders_production.parquet")
    assert (o.suggested_qty >= 0).all()
    assert o.order_date.nunique() == 1
    assert {"store_id", "item_id", "suggested_qty", "order_up_to", "on_hand"} <= set(o.columns)


def test_report_generated(pipeline):
    rep = Path(pipeline["paths"]["reports"])
    assert (rep / "results.md").exists()
    assert (rep / "figures" / "ladder.png").exists()


def test_simulation_policies(pipeline):
    s = json.loads((Path(pipeline["paths"]["artifacts"]) / "simulation.json").read_text())
    for pol in ("P0_manual", "P1_point_ml", "P2_uncertainty"):
        assert 0 < s["policies"][pol]["fill_rate"] <= 1
