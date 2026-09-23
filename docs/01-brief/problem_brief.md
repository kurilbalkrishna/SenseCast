# Industry problem brief: BDS-32 Multimodal Retail Demand Sensing

## 1. Problem in one paragraph

Grocery and FMCG stores in Maharashtra see demand swing with things that are not in last month's
sales: a heavy monsoon day (umbrellas up, cold drinks down), a heat wave, Ganesh Chaturthi, a
promotion, a product going viral online. Replenishment that only looks backwards reacts a week late,
so shelves run empty on the days that matter most and overstock builds up afterwards. New and
slow-moving items are the worst case: there is little or no history to average. SenseCast forecasts
every item in every store 1 to 14 days ahead from **sales, weather, festivals, promotions and search
interest**, attaches an honest uncertainty band, and turns that into order suggestions a planner
approves, overrides or rejects, with every decision audited.

## 2. Stakeholder map

| Stakeholder | Interest | Influence | What they need from SenseCast |
|---|---|---|---|
| Replenishment planner (primary user) | High | High | Daily order suggestions they can trust, and the reason behind each one |
| Store manager | High | Medium | Their store's orders and the ability to flag local events |
| Inventory / supply-chain analyst | Medium | High | Accuracy tracking, service vs inventory trade-off, drift alerts |
| Category manager | Medium | Medium | Promotion and festival uplift by category |
| Finance | Low | High | Working capital tied up in stock; cost of lost sales |
| IT / platform team | Low | High | Containerised service, logs, metrics, no secrets in code |
| Customers (indirect) | High | Low | Product on the shelf when they want it |

## 3. Current process (baseline we must beat)

A typical store-level process in small and mid-sized chains:

1. Every evening the planner exports the last 4 to 8 weeks of sales per item into a spreadsheet.
2. They take an average, apply a rough day-of-week adjustment, and add a fixed safety buffer (often ~20%).
3. They bump orders by feel before a big festival, and forget to bring them back down afterwards.
4. New items get "the same as a similar item", chosen from memory.

SenseCast reproduces this as two baselines so the comparison is fair and repeatable:
**B1** (smoothed level x weekday index) and **policy P0** (B1 x lead time x 1.2 buffer).

> **Team task:** replace or confirm this section with evidence you collect yourselves. Record at
> least three short conversations (a kirana owner, a supermarket shift manager, a retail-analytics
> professional or faculty member) in the log below. Do not invent entries.

| # | Date | Role of person | How they order today | Biggest pain | Quote (with permission) |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

## 4. Personas

**Priya, replenishment planner (central team, Mumbai).** Orders for 10 stores and ~1,500 item-stores
every evening in about 90 minutes. Measured on availability and on stock value. Distrusts black
boxes: "If the system says order 40 cases, I need to know why."

**Rahul, store manager (Pune).** Knows local events the head office does not: a society fete, a
cricket screening, road works outside the store. Wants to adjust his store's order and have that
remembered.

**Meera, inventory analyst.** Owns the monthly review. Needs to see whether forecasts are drifting,
which categories lose money to stockouts, and whether human overrides help or hurt.

## 5. User stories (with acceptance criteria)

| ID | As a... | I want... | So that... | Accepted when |
|---|---|---|---|---|
| US-01 | planner | a suggested order per item-store every evening | I stop building spreadsheets | `/v1/orders` returns every active item-store with qty >= 0 |
| US-02 | planner | to see the forecast band and the top drivers | I can trust or challenge a suggestion | forecast page shows P10/P50/P90 and % effect of weather, festivals, promo, search |
| US-03 | planner | to approve, override or reject with a reason | the system learns where it is wrong | override without a reason code is refused; decision stored with user and time |
| US-04 | store manager | to see and act on my own store only | other stores' data stays private | RBAC test: manager of S01 gets 403 on S02 |
| US-05 | analyst | weekly accuracy and drift alerts | I catch problems before stores do | monitoring page shows weekly WAPE, PSI per feature, store bias alerts |
| US-06 | analyst | a service-vs-inventory comparison of policies | I can argue for the change with finance | simulation reports fill rate, stockout days, days of cover and cost in INR |
| US-07 | planner | forecasts for newly launched items | launches do not start empty or overstocked | cold-start rows served; accuracy reported for first 28 days |
| US-08 | IT | one command to run everything | the system is reproducible | `docker compose up` runs pipeline, API and dashboard from a clean clone |
| US-09 | planner | an export of approved orders | I can send them to suppliers | CSV export with formula-injection protection |
| US-10 | analyst | every number traceable to data and config | audits are possible | manifest hashes + MLflow run with config hash |

## 6. Misuse and abuse cases

| Misuse case | Actor | Harm | Control in SenseCast |
|---|---|---|---|
| Manager inflates overrides to hoard stock | Insider | Overstock, write-offs | Override needs a reason code; guard blocks > 10x suggestion; audit log is append-only; override accuracy can be reviewed per user |
| Poisoned or glitching search feed (bot spike) | External / data | Wrong orders across a category | Median-based search features; fault test shows < 5% forecast shift for a 2-day spike |
| Weather feed outage | Supplier | Model silently uses garbage | Missing weather replaced by climatology and flagged (`wx_missing`); outage cost measured |
| Formula injection in exported CSV | External | Code runs in the planner's spreadsheet | Cells starting with `= + - @` are escaped |
| Store manager reads another store's data | Insider | Commercial leak | Store-scoped RBAC on every endpoint |
| Flooding the API | External | Denial of service | Per-client rate limit, payload size cap |
| Editing past decisions to hide a mistake | Insider | Lost accountability | SQLite triggers block UPDATE and DELETE on the audit table |

## 7. Scope

**In scope:** daily item x store forecasts 1 to 14 days ahead, 5-level hierarchy, intermittent and
cold-start handling, calibrated intervals, order-up-to suggestions, approval workflow, monitoring,
containerised deployment.

**Out of scope (and why):** real POS integration (no partner data available; the feed contract is
defined so a real feed can replace the generator), supplier ordering (orders are exported, not
sent), pricing and promotion optimisation, multi-echelon (warehouse) inventory, real Google Trends
scraping (terms of service), production-grade identity provider (hook provided in `api/auth.py`).

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Synthetic data too easy, results look better than reality | Medium | High | Effects set to realistic sizes; noise, censoring and dirty rows added; gains reported against strong baselines, not just naive |
| Leakage inflates accuracy | Medium | High | Point-in-time builder with asserts; leakage probe scrambles the future; blocked train/cal/test split |
| Weather API unavailable at demo time | High | Low | Cached pull + synthetic climatology fallback, provenance recorded |
| Laptop too slow for full run | Medium | Medium | `small` profile runs in ~15 s; full run gated at < 10 minutes |
| Only two team members for a three-person brief | Certain | Medium | Scope kept to one global model family; ownership split in CODEOWNERS; automation over manual steps |

## 9. Success metrics

See `configs/default.yaml -> thresholds` and the generated gate table in
`docs/03-evaluation/results.md`. Headline: WAPE at least 10% better than seasonal naive, calibrated
P10 to P90 coverage of 75 to 85%, at least 20% fewer stockout days than the manual policy.

## 10. Prioritised backlog

MoSCoW priorities; the importable issue list is `backlog.csv`.

| Priority | Item |
|---|---|
| Must | Synthetic world + feed contracts, ingestion with quarantine, point-in-time features, B0/B1/M1/M2 ladder, backtest with blocked split, calibrated intervals, reconciliation, inventory simulation, API with audit, dashboard, tests, Docker, CI |
| Should | Intermittent routing (TSB/SBA), cold-start module, fault suite, drift monitoring, SHAP drivers, MLflow tracking, load test |
| Could | Real Open-Meteo pull (implemented, used when network allows), override-accuracy report, Postgres audit store |
| Won't (this release) | Real POS feed, supplier EDI, price optimisation, multi-echelon inventory |
