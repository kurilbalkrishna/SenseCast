"""API contract, validation, RBAC and audit tests."""
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sensecast.api.app import create_app, escape_csv_cell


@pytest.fixture(scope="module")
def client(pipeline, tmp_path_factory, monkeypatch_module):
    art = Path(pipeline["paths"]["artifacts"])
    monkeypatch_module.setenv("SENSECAST_AUDIT_DB", str(tmp_path_factory.mktemp("audit") / "audit.db"))
    monkeypatch_module.setenv("SENSECAST_AUTH_MODE", "off")
    return TestClient(create_app(art))


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _first(client):
    store = client.get("/v1/stores").json()[0]["store_id"]
    item = client.get("/v1/items", params={"store_id": store}).json()[0]["item_id"]
    return store, item


def test_health_and_metrics(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_version"]
    client.get("/v1/stores")
    m = client.get("/metrics").text
    assert "sensecast_request_seconds" in m


def test_request_id_header(client):
    r = client.get("/v1/stores", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_forecast_contract(client):
    store, item = _first(client)
    body = client.get(f"/v1/forecast/{store}/{item}").json()
    assert {"history", "forecast", "drivers", "model", "model_version"} <= set(body)
    f = body["forecast"]
    assert len(f) == 14
    assert all(r["p10"] <= r["p90"] for r in f)


def test_unknown_item_404(client):
    assert client.get("/v1/forecast/S01/I999").status_code == 404


def test_store_forecast_is_reconciled(client):
    store, _ = _first(client)
    rows = client.get(f"/v1/stores/{store}/forecast").json()
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], {"total": 0, "cats": 0})
        if r["level"] == "store":
            by_date[r["date"]]["total"] = r["reconciled"]
        else:
            by_date[r["date"]]["cats"] += r["reconciled"]
    for v in by_date.values():
        assert v["total"] == pytest.approx(v["cats"], abs=0.01)


def test_decision_flow_and_audit(client):
    store, _ = _first(client)
    orders = client.get("/v1/orders", params={"store_id": store}).json()
    o = orders[0]
    ok = client.post("/v1/orders/decision", json=dict(store_id=store, item_id=o["item_id"], order_date=o["order_date"],
                                                      action="approve", final_qty=0))
    assert ok.status_code == 201 and ok.json()["final_qty"] == o["suggested_qty"]
    # override without a reason is rejected
    bad = client.post("/v1/orders/decision", json=dict(store_id=store, item_id=o["item_id"], order_date=o["order_date"],
                                                       action="override", final_qty=3))
    assert bad.status_code == 422
    # absurd override is blocked by the guard
    guard = client.post("/v1/orders/decision", json=dict(store_id=store, item_id=o["item_id"], order_date=o["order_date"],
                                                         action="override", final_qty=99_999, reason_code="LOCAL_EVENT"))
    assert guard.status_code == 422
    good = client.post("/v1/orders/decision", json=dict(store_id=store, item_id=o["item_id"], order_date=o["order_date"],
                                                        action="override", final_qty=3, reason_code="LOCAL_EVENT",
                                                        note="school fete on Saturday"))
    assert good.status_code == 201
    log = client.get("/v1/audit", params={"store_id": store}).json()
    assert log[0]["action"] == "override" and log[0]["reason_code"] == "LOCAL_EVENT"
    status = {r["item_id"]: r["status"] for r in client.get("/v1/orders", params={"store_id": store}).json()}
    assert status[o["item_id"]] == "override"


def test_wrong_order_date_conflict(client):
    store, _ = _first(client)
    o = client.get("/v1/orders", params={"store_id": store}).json()[0]
    r = client.post("/v1/orders/decision", json=dict(store_id=store, item_id=o["item_id"], order_date="2000-01-01",
                                                     action="approve", final_qty=0))
    assert r.status_code == 409


def test_input_validation(client):
    r = client.post("/v1/orders/decision", json=dict(store_id="S01; DROP TABLE", item_id="I001",
                                                     order_date="2026-01-01", action="approve", final_qty=1))
    assert r.status_code == 422
    r = client.post("/v1/orders/decision", json=dict(store_id="S01", item_id="I001", order_date="2026-01-01",
                                                     action="approve", final_qty=-5))
    assert r.status_code == 422


def test_audit_log_is_append_only(client):
    db = client.app.state.audit.path
    with sqlite3.connect(db) as c, pytest.raises(sqlite3.DatabaseError):
        c.execute("UPDATE decisions SET final_qty = 0")
    with sqlite3.connect(db) as c, pytest.raises(sqlite3.DatabaseError):
        c.execute("DELETE FROM decisions")


def test_csv_export_escapes_formulas(client):
    store, _ = _first(client)
    r = client.get("/v1/orders/export.csv", params={"store_id": store})
    assert r.status_code == 200 and r.text.startswith("store_id,")
    assert escape_csv_cell("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert escape_csv_cell("@SUM") == "'@SUM"
    assert escape_csv_cell("S01") == "S01"


def test_rbac_in_header_mode(pipeline, monkeypatch, tmp_path):
    monkeypatch.setenv("SENSECAST_AUTH_MODE", "header")
    monkeypatch.setenv("SENSECAST_AUDIT_DB", str(tmp_path / "a.db"))
    c = TestClient(create_app(Path(pipeline["paths"]["artifacts"])))
    stores = [s["store_id"] for s in c.get("/v1/stores", headers={"X-User": "p", "X-Role": "planner"}).json()]
    mine, other = stores[0], stores[1]
    assert c.get("/v1/stores").status_code == 401
    assert c.get("/v1/stores", headers={"X-User": "x", "X-Role": "superuser"}).status_code == 403
    mgr = {"X-User": "m", "X-Role": "store_manager", "X-Store": mine}
    assert c.get("/v1/items", params={"store_id": other}, headers=mgr).status_code == 403
    assert c.get("/v1/items", params={"store_id": mine}, headers=mgr).status_code == 200
    o = c.get("/v1/orders", headers=mgr).json()
    assert {r["store_id"] for r in o} == {mine}
    analyst = {"X-User": "a", "X-Role": "analyst"}
    r = c.post("/v1/orders/decision", headers=analyst, json=dict(store_id=mine, item_id=o[0]["item_id"],
                                                                  order_date=o[0]["order_date"], action="approve", final_qty=0))
    assert r.status_code == 403


def test_rate_limit(pipeline, monkeypatch, tmp_path):
    monkeypatch.setenv("SENSECAST_RATE_LIMIT_PER_MIN", "5")
    monkeypatch.setenv("SENSECAST_AUDIT_DB", str(tmp_path / "b.db"))
    monkeypatch.setenv("SENSECAST_AUTH_MODE", "off")
    c = TestClient(create_app(Path(pipeline["paths"]["artifacts"])))
    codes = [c.get("/v1/stores").status_code for _ in range(8)]
    assert 429 in codes
    assert c.get("/health").status_code == 200  # health is never rate limited
