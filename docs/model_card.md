# Model card: SenseCast demand forecaster v1.0.0

## Model details

* **Type:** global gradient-boosted trees (LightGBM 4.7), one model across all item-stores and
  horizons. Point model: Tweedie objective (variance power 1.2). Interval models: quantile objective
  at 0.1, 0.5 and 0.9, then conformalized quantile regression (CQR).
* **Routing:** intermittent items -> SBA, lumpy items -> TSB, chosen per class on the calibration
  block. New items (first 28 days) -> M2, chosen over the analog method and category mean.
* **Aggregate model:** LightGBM on ratio-to-recent-level targets for 54 aggregate nodes; forecasts
  reconciled with WLS (variance scaling).
* **Version:** 1.0.0, config hash `46cb3732ab4d`. Artefacts in `artifacts/models/*.txt`; every run in MLflow.
* **Owners:** Member A (model), Member B (serving). Contact through the repository.

## Intended use

Daily replenishment support for grocery / FMCG stores: forecast 1 to 14 days ahead per item and
store, and suggest orders a human approves. **Not for:** automatic ordering without human review,
pricing decisions, staff performance evaluation, or any decision about individual people.

## Inputs

Sales history (stockout days masked in history features; lower-bound imputation on stockout days in
the training target), item price and pack size, store, calendar, Indian festivals and public
holidays, planned promotions, weather forecast (max temperature, rain, humidity) and category search
interest. 36 features; see `docs/02-design/architecture.md` section 5.

## Training and evaluation data

Synthetic retail world (`data/README.md`): 1,500 item-stores, 3 years daily, known effects, 6.5% of
item-days censored by stockouts, injected data defects. Train: 78 weekly origins (1.51 M rows).
Calibration: 8 origins. Test: 8 origins, 19 Jul to 20 Sep 2026 (167,100 rows).

## Metrics (test block)

| Metric | SenseCast | Best baseline |
|---|---|---|
| WAPE | 0.407 | 0.481 (B1) |
| MASE | 0.895 | 0.987 (B1) |
| Pinball loss (P10/P50/P90) | 1.477 | 1.714 (B1) |
| P10 to P90 coverage | 85.6% | 83.5% (B1) |
| WAPE on festival + heavy-rain days, M2 vs M1 | 0.414 vs 0.506 | |
| Bias vs true demand (all items / top 10%) | -9.0% / -11.9% | |

By group (WAPE): smooth 0.29, erratic 0.49, new 0.51, lumpy 0.97, intermittent 1.25; stores 0.37 to 0.43.

## Explainability

TreeSHAP contributions grouped into families. Planners see the demand-sensing families (festivals,
weather, promotion, search, day of week / season) as "% effect vs the item's usual level". Recent
sales dominate raw SHAP by construction (they set the level), so they are not shown as a driver.

## Ethical considerations and risks

No personal data. Main risks are operational: systematic under-forecasting of fast sellers (bias
-11.9% on the top 10% of items against true demand), over-trust in driver percentages (associations,
attenuated by forecast noise), and override misuse (mitigated by reason codes, guard and audit log).

## Caveats and recommendations

* Validated only on a synthetic world. Re-run the full gate suite on real POS data before relying on it.
* Retrain at least monthly and whenever the store-bias alert fires two weeks in a row.
* Treat slow movers' intervals as conservative; review coverage by class, not only overall.
* Keep humans in the loop: suggestions, not orders.
