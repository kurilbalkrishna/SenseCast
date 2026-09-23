# Viva preparation

Each question lists who answers first. Both members should be able to answer every question and
point to the file that implements the answer.

**1. Why WAPE and MASE, not MAPE?** (A) About a third of item-stores sell 0 on many days; MAPE divides
by actual sales and explodes. WAPE weights errors by volume; MASE compares against the weekly naive
forecast so it is scale-free across items. `models/metrics.py`.

**2. How do you stop weather from leaking?** (A) On the forecast day you only know a weather
forecast. The feature is actual + noise whose spread grows with the horizon (0.8 degC x sqrt(h)); a
leakage test checks the error at h=14 is larger than at h=1, and the scrambled-future probe proves no
feature reads after the origin. `features/build.py`, `tests/leakage/`.

**3. Why are sales not demand, and what did you do about it?** (A) When the shelf is empty, sales
stop but demand does not. Dropping those days is not enough: the days that remain are the calmer
ones, so the model still learns low. We impute a lower-bound demand on stockout days ("demand
unconstraining") and the censoring ablation compares ignore / drop / impute against the true demand,
which only the synthetic world can provide. `evaluation/backtest.py::training_rows`, `evaluation/faults.py`.

**4. Isn't synthetic data cheating?** (A) The opposite: every effect size is known, so we can check
whether the model recovers it (effect-recovery figure for rain and heat). The world includes
censoring, dirty rows, cold starts and noise so it is not easy. The loader also pulls real
Open-Meteo weather when the network allows.

**5. What does reconciliation do, and why WLS rather than MinT?** (A) Forecasts made independently at
store, category and item level do not add up. Reconciliation finds the closest coherent set. MinT
needs the full error covariance of 1,554 series from 112 calibration cases, which is badly
estimated even with shrinkage; WLS uses only the variances. In our test WLS was as good or better at
every level and did not degrade item-level accuracy. `reconcile/mint.py`.

**6. Why Croston, SBA, TSB for slow movers?** (A) They model how big a sale is and how often it
happens separately. SBA removes Croston's upward bias; TSB lets the probability decay so a dying
item fades to zero. The router picks per demand class on the calibration block, never on test.

**7. What is pinball loss?** (A) The loss for a quantile: under-forecasting a P90 costs 9 times more
than over-forecasting it. It is how you score P10 and P90, which WAPE cannot.

**8. Why conformal calibration?** (A) Quantile models are often over- or under-confident. CQR widens
or narrows every interval by an amount learned on a calibration block the model never trained on,
per horizon bucket and demand class. Coverage is then checked on test.

**9. Why is coverage slightly above 85% for slow movers?** (A) Counts are integers. For an item that
sells 0 or 1 unit a day, the interval [0, 1] already contains about 95% of days; no integer interval
can hit 80% exactly. Continuous-demand classes land at 78 to 81%.

**10. How is an order quantity calculated?** (B) Order-up-to level S = expected demand over lead time
+ review period + z x combined uncertainty from the calibrated daily bands; order = S minus stock on
hand and on order. `simulate/inventory.py::order_up_to`.

**11. How do you know the policy is better and not just holding more stock?** (B) The frontier
sweeps each policy's knob and compares them at equal inventory: how many days of cover each needs to
reach a 95% fill rate.

**12. How does a human stay in control?** (B) Every order is a suggestion; approve / override / reject
with a reason code, guard at 10x, append-only audit with user, time, before/after, model version and
request id. `api/app.py`, `api/audit.py`.

**13. What happens if the weather feed dies?** (B) Missing values fall back to monthly climatology and
every affected row is flagged. The fault suite measures the cost; the degraded model still beats
the sales-only model.

**14. What are the security controls?** (B) STRIDE table in `docs/02-design/architecture.md`: RBAC,
rate limiting, payload cap, validation, append-only audit, CSV escaping, no secrets in the repo
(gitleaks), pinned dependencies (pip-audit), non-root container.

**15. How do you monitor it in production?** (B) JSON logs with request ids, Prometheus metrics,
weekly WAPE vs baseline, store bias alerts at 15%, PSI per feature at 0.2, data freshness.

**16. What would you change with real data?** (both) Replace the generator with the POS feed (same
contract), retrain, re-run the gates; add promotion effects learned per item; store outstanding
purchase orders so the inventory position is exact; add a real identity provider.
