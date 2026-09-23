# Demo video script (7 minutes)

Record at 1080p with the dashboard full-screen. Both members speak: A covers data and models,
B covers the product and operations. Rehearse twice; keep each segment to its time.

| Time | Who | Screen | Say |
|---|---|---|---|
| 0:00 - 0:40 | A | Title slide, then a monsoon-day photo of an empty shelf | The problem: demand moves with rain, heat, festivals and online buzz; backward-looking orders arrive a week late. |
| 0:40 - 1:30 | B | Terminal: `docker compose up --build`; logs scrolling | One command runs the whole pipeline, then the API and dashboard. Point at a JSON log line with a step duration. |
| 1:30 - 2:30 | A | Architecture diagram (`docs/02-design/architecture.md`) | Four sources, point-in-time features, model ladder, calibration, reconciliation, order-up-to. Say the leakage rule out loud. |
| 2:30 - 3:40 | A | Dashboard > Forecasts: a snacks item in Mumbai around Ganesh Chaturthi | Band = 80% range. Drivers chart: festival +x%, rain -y%. Switch to a slow mover: routed to SBA/TSB. |
| 3:40 - 4:40 | B | Dashboard > Orders | Approve two rows, override one with LOCAL_EVENT and a note, try an override without a reason (refused), export CSV. Show the audit log tab. |
| 4:40 - 5:40 | B | Dashboard > Inventory simulation | Frontier chart: same inventory, higher fill rate. Quote the gate numbers. |
| 5:40 - 6:30 | A | Dashboard > Monitoring, then `docs/03-evaluation/results.md` | PSI alerts and why seasonal features drift; fault suite (weather outage, search spike, censoring ablation). One honest failure and what we learned. |
| 6:30 - 7:00 | both | Gates table | What passed, what did not, limitations, what we would do with real POS data. |

Checklist before recording: fresh `make all`; `make loadtest` so the latency gate has a value;
clear the audit DB (`rm artifacts/audit.db`) so the demo starts clean.
