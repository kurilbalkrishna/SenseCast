"""SenseCast planner dashboard (Streamlit). Talks to the API only.

Run:  streamlit run dashboard/app.py      (API_URL defaults to http://localhost:8000)
"""
from __future__ import annotations

import os

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REASONS = ["LOCAL_EVENT", "SUPPLIER_ISSUE", "SHELF_SPACE", "PROMO_CHANGE", "DATA_ERROR", "KNOWN_DEMAND_SHIFT", "OTHER"]

st.set_page_config(page_title="SenseCast", layout="wide")

with st.sidebar:
    st.markdown("### SenseCast")
    st.caption("Multimodal retail demand sensing")
    api = st.text_input("API URL", API_URL)
    with st.expander("Identity (header auth mode)"):
        who = st.text_input("User", "demo")
        role = st.selectbox("Role", ["planner", "store_manager", "analyst", "admin"])
        my_store = st.text_input("Store (store managers)", "S01")
HEADERS = {"X-User": who, "X-Role": role, "X-Store": my_store}


def get(path: str, **params):
    r = httpx.get(f"{api}{path}", params=params, headers=HEADERS, timeout=30)
    if r.status_code >= 400:
        st.error(f"{path}: {r.status_code} {r.json().get('detail') if r.headers.get('content-type','').startswith('application/json') else r.text}")
        st.stop()
    return r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text


try:
    health = get("/health")
except httpx.HTTPError:
    st.error(f"Cannot reach the API at {api}. Start it with `make api` or `docker compose up`.")
    st.stop()

st.caption(f"Model {health.get('model_version')} · data through {health.get('last_observed_day')}")
tabs = st.tabs(["Overview", "Forecasts", "Orders", "Inventory simulation", "Monitoring", "Audit log"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    ev = get("/v1/evaluation")
    sim = get("/v1/simulation")
    ov = ev["overall"]
    c = st.columns(4)
    c[0].metric("WAPE (SenseCast)", f"{ov['final']['wape']:.3f}", f"{(ov['final']['wape'] / ov['b0']['wape'] - 1) * 100:.1f}% vs seasonal naive", delta_color="inverse")
    c[1].metric("P10–P90 coverage", f"{ev['coverage']['final_conformal'] * 100:.1f}%", "target 75–85%", delta_color="off")
    if sim:
        p0, p2 = sim["policies"]["P0_manual"], sim["policies"]["P2_uncertainty"]
        c[2].metric("Fill rate (P2)", f"{p2['fill_rate'] * 100:.1f}%", f"{(p2['fill_rate'] - p0['fill_rate']) * 100:+.1f} pts vs manual")
        c[3].metric("Stockout days vs manual policy", f"{-sim['stockout_reduction_P2_vs_P0'] * 100:+.0f}%")
    st.subheader("Go / no-go gates")
    gates = ev.get("gates") or {}
    if gates:
        g = pd.DataFrame(gates["gates"])
        st.dataframe(g, hide_index=True, use_container_width=True)
    st.subheader("Model ladder (test block)")
    ladder = pd.DataFrame(ov).T.rename(index={"b0": "B0 seasonal naive", "b1": "B1 planner", "m1": "M1 LightGBM sales-only",
                                              "m2": "M2 LightGBM multisource", "final": "SenseCast (routed)"})
    st.dataframe(ladder.style.format("{:.3f}"), use_container_width=True)

# ---------------------------------------------------------------- forecasts
with tabs[1]:
    stores = pd.DataFrame(get("/v1/stores"))
    col1, col2 = st.columns([1, 3])
    store = col1.selectbox("Store", stores.store_id)
    items = pd.DataFrame(get("/v1/items", store_id=store))
    cat = col1.selectbox("Category", sorted(items.category.unique()))
    sub = items[items.category == cat]
    item = col1.selectbox("Item", sub.item_id, format_func=lambda i: f"{i} · {sub.set_index('item_id').demand_class[i]}")
    fc = get(f"/v1/forecast/{store}/{item}")
    hist = pd.DataFrame(fc["history"]).assign(kind="actual")
    f = pd.DataFrame(fc["forecast"])
    band = alt.Chart(f).mark_area(opacity=0.25).encode(x="date:T", y=alt.Y("p10:Q", title="units / day"), y2="p90:Q")
    p50 = alt.Chart(f).mark_line(strokeWidth=2.5).encode(x="date:T", y="p50:Q")
    act = alt.Chart(hist).mark_line(color="#555").encode(x="date:T", y="units:Q")
    col2.altair_chart((band + p50 + act).properties(height=320), use_container_width=True)
    col2.caption(f"Model: {fc['model']} · class: {fc['demand_class']} · shaded band = calibrated P10–P90")
    if fc.get("note"):
        col2.info(fc["note"])
    d = pd.DataFrame(fc["drivers"])
    col2.markdown("**What is moving this forecast** (average effect of each signal over the next 14 days, "
                  "relative to this item's usual level)")
    col2.altair_chart(alt.Chart(d).mark_bar().encode(
        x=alt.X("effect_pct:Q", title="% effect vs typical"), y=alt.Y("factor:N", sort="-x", title=None),
        color=alt.condition("datum.effect_pct > 0", alt.value("#0C7A68"), alt.value("#B83A30"))).properties(height=220),
        use_container_width=True)
    st.subheader(f"Store {store}: reconciled category forecast")
    agg = pd.DataFrame(get(f"/v1/stores/{store}/forecast"))
    st.altair_chart(alt.Chart(agg[agg.level == "store_category"]).mark_line().encode(
        x="date:T", y=alt.Y("reconciled:Q", title="units / day"), color="node:N").properties(height=260),
        use_container_width=True)

# ---------------------------------------------------------------- orders
with tabs[2]:
    store_o = st.selectbox("Store", stores.store_id, key="ostore")
    orders = pd.DataFrame(get("/v1/orders", store_id=store_o))
    if orders.empty:
        st.info("No suggested orders.")
    else:
        st.caption(f"Suggested orders for {orders.order_date.iloc[0]} · policy: uncertainty-aware order-up-to (P2)")
        counts = orders.status.value_counts()
        st.write(" · ".join(f"**{k}**: {v}" for k, v in counts.items()))
        st.dataframe(orders, hide_index=True, use_container_width=True, height=330)
        with st.form("decision"):
            st.markdown("**Record a decision**")
            a, b, c3 = st.columns(3)
            it = a.selectbox("Item", orders.item_id)
            row = orders.set_index("item_id").loc[it]
            action = b.selectbox("Action", ["approve", "override", "reject"])
            qty = c3.number_input("Final quantity", min_value=0, value=int(row.suggested_qty), step=1)
            reason = a.selectbox("Reason (required for override / reject)", ["", *REASONS])
            note = b.text_input("Note", max_chars=280)
            if st.form_submit_button("Save decision"):
                body = dict(store_id=store_o, item_id=it, order_date=str(row.order_date), action=action,
                            final_qty=int(qty), reason_code=reason or None, note=note or None)
                r = httpx.post(f"{api}/v1/orders/decision", json=body, headers=HEADERS, timeout=30)
                if r.status_code == 201:
                    st.success(f"Saved: {action} {it}, final quantity {r.json()['final_qty']}")
                else:
                    st.error(f"Not saved ({r.status_code}): {r.json().get('detail')}")
        csv_text = httpx.get(f"{api}/v1/orders/export.csv", params={"store_id": store_o}, headers=HEADERS, timeout=30).text
        st.download_button("Download orders CSV", csv_text, file_name=f"orders_{store_o}.csv", mime="text/csv")

# ---------------------------------------------------------------- simulation
with tabs[3]:
    sim = get("/v1/simulation")
    if not sim:
        st.info("Run `sensecast simulate` first.")
    else:
        pol = pd.DataFrame(sim["policies"]).T
        st.caption(f"{sim['n_series']} item-stores · {sim['n_days']} days ({sim['window'][0]} to {sim['window'][1]}) · "
                   f"{sim['n_paths']} demand paths · lead time {sim['lead_time']} d · target service {sim['service_target']:.0%}")
        show = pol[["fill_rate", "stockout_day_rate", "avg_on_hand_units", "avg_days_of_cover", "overstock_units_per_day",
                    "total_cost_inr_per_day"]]
        st.dataframe(show.style.format({"fill_rate": "{:.2%}", "stockout_day_rate": "{:.2%}", "avg_on_hand_units": "{:,.0f}",
                                        "avg_days_of_cover": "{:.1f}", "overstock_units_per_day": "{:,.0f}",
                                        "total_cost_inr_per_day": "₹{:,.0f}"}), use_container_width=True)
        fr = pd.DataFrame(sim.get("frontier", {}).get("points", []))
        if not fr.empty:
            st.markdown("**Service vs inventory frontier** (each point = one setting of the policy's knob)")
            st.altair_chart(alt.Chart(fr).mark_line(point=True).encode(
                x=alt.X("days_of_cover:Q", title="average days of cover", scale=alt.Scale(zero=False)),
                y=alt.Y("fill_rate:Q", title="fill rate", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
                color="policy:N", tooltip=["policy", "knob", "fill_rate", "days_of_cover"]).properties(height=280),
                use_container_width=True)
            need = sim["frontier"]["cover_days_at_95_fill"]
            st.caption("Days of cover needed for a 95% fill rate: " +
                       ", ".join(f"{k}: {v:.2f}" if v else f"{k}: not reached" for k, v in need.items()))
        tr = pd.DataFrame(get("/v1/simulation/trace"))
        st.altair_chart(alt.Chart(tr).mark_line().encode(x="date:T", y=alt.Y("lost:Q", title="lost units / day"),
                                                         color="policy:N").properties(height=260), use_container_width=True)

# ---------------------------------------------------------------- monitoring
with tabs[4]:
    mon = get("/v1/monitoring")
    if "psi" in mon:
        psi = pd.DataFrame(sorted(mon["psi"].items(), key=lambda x: -x[1]), columns=["feature", "psi"])
        st.subheader("Feature drift (PSI, train vs test)")
        rule = alt.Chart(pd.DataFrame({"t": [mon["psi_threshold"]]})).mark_rule(color="#B83A30", strokeDash=[4, 4]).encode(x="t:Q")
        st.altair_chart(alt.Chart(psi).mark_bar().encode(x="psi:Q", y=alt.Y("feature:N", sort="-x")).properties(height=420) + rule,
                        use_container_width=True)
        st.caption("Seasonal features (month, week, weather) drift by design between a training year and a monsoon test window.")
        w = pd.DataFrame(mon["weekly"])
        st.subheader("Weekly accuracy")
        st.altair_chart(alt.Chart(w.melt("origin_date", ["wape_final", "wape_b0"])).mark_line(point=True).encode(
            x="origin_date:T", y=alt.Y("value:Q", title="WAPE"), color="variable:N").properties(height=240), use_container_width=True)
        sb = pd.DataFrame(mon["store_bias"])
        st.subheader(f"Store bias alerts: {mon['store_bias_alerts']}")
        st.dataframe(sb[sb.alert], hide_index=True, use_container_width=True)
        st.subheader("Data freshness")
        st.json(mon["freshness"])
    if mon.get("faults"):
        st.subheader("Fault suite")
        st.dataframe(pd.DataFrame([{"experiment": k, "passed": v.get("passed")} for k, v in mon["faults"].items()]),
                     hide_index=True)

# ---------------------------------------------------------------- audit
with tabs[5]:
    log = pd.DataFrame(get("/v1/audit", limit=500))
    st.caption("Append-only: updates and deletes are blocked in the database.")
    st.dataframe(log, hide_index=True, use_container_width=True)
