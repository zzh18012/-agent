# Monthly demand forecast followed by production planning

A factory records monthly demand in `data.csv` (columns `period` and `demand`) for the last
24 months.

Two linked tasks must be completed in order:

1. Forecast monthly demand for the next three months (2026-01, 2026-02, 2026-03) from the
   historical monthly series. Temporal order must be preserved and the forecast must be
   validated with rolling-origin splits.
2. Use that forecast as the demand input to a production-planning model. For each of the
   three planning months choose how much to produce and how much inventory to carry so that
   demand is met, monthly production capacities are respected, and total production plus
   holding cost is minimized.

The second task depends on the first: the production plan must consume the validated
forecast rather than a separately supplied demand number. Report the forecast, the
production plan, and the audit trail linking them.

The forecast horizon is three months and the production plan covers exactly the same three
months. Point forecasts are acceptable; do not claim to model demand uncertainty.
