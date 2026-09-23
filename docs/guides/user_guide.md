# User guide (planners, store managers, analysts)

Open the dashboard at http://localhost:8501 (or the address your admin gives you).

## Every evening: review suggested orders

1. **Orders** tab, pick your store. Each row is one item: stock on hand, expected demand over the
   next 4 days (lead time 3 + review 1), the P10 to P90 range, the order-up-to level and the
   **suggested quantity**.
2. For most rows, choose **approve**.
3. If you know something the system does not (a local event, shelf space, a supplier problem),
   choose **override**, enter your quantity and pick a **reason code**. A reason is required;
   quantities above 10x the suggestion are blocked as likely typing errors.
4. **reject** means "do not order this item today" (reason required).
5. **Download orders CSV** for your supplier.

Every decision is saved with your name, the time, the suggestion and your change. It cannot be edited
afterwards; if you made a mistake, record a new decision (the newest one counts).

## Understanding a forecast

**Forecasts** tab, pick store, category, item.

* Grey line: recent sales. Green line: expected demand. Shaded band: there is roughly an 80% chance
  demand lands inside it.
* **What is moving this forecast** shows how much each factor pushes demand up or down compared with
  a typical day over the next two weeks: for example *festivals & holidays +22%*, *weather -6%*.
* If the note says the forecast comes from **SBA** or **TSB**, the item sells rarely and a
  specialised slow-mover method is used instead of the main model.

## Reason codes

| Code | Use when |
|---|---|
| LOCAL_EVENT | fete, match screening, wedding season in the area |
| SUPPLIER_ISSUE | supplier short or delayed |
| SHELF_SPACE | not enough space for the suggested quantity |
| PROMO_CHANGE | promotion added, cancelled or changed at short notice |
| DATA_ERROR | the stock figure or history looks wrong |
| KNOWN_DEMAND_SHIFT | new competitor, road closure, store refit |
| OTHER | anything else (add a note) |

## For analysts

* **Overview:** go/no-go gates and the model ladder.
* **Inventory simulation:** how the three ordering policies trade stock held against fill rate.
* **Monitoring:** feature drift (PSI), weekly accuracy against the seasonal-naive baseline, store bias
  alerts (a store consistently over- or under-forecast by more than 15%), data freshness and the
  fault-suite results.
* **Audit log:** every decision, filterable by store.
