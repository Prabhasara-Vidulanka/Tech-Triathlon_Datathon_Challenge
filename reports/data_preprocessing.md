# Data Preprocessing and Modelling Rationale

## Sources and integrity

The solution reads the official order, route, reference, test, and template CSVs without modifying them. SHA-256 hashes, schemas, missingness, duplicate counts, date ranges, and template identities are recorded in `initial_audit.json`. All dispatched training orders join exactly once to a route leg. Not-run orders remain part of Task 2A demand but are excluded from Task 1 because no actual route outcome exists.

## Task 1 labels and time handling

Times are parsed as Asia/Colombo minutes after midnight. The supplied data has no midnight-crossing windows or routes, but the label code can roll a close or completion time into the next day when required. Service begins at the later of actual arrival and window opening, because an early vehicle waits. Service time is outlet leave time minus that service start. Lateness is one only when arrival is strictly later than window close. This removes waiting time from 4,023 early arrivals and preserves 279 arrivals exactly at close as non-late.

The 45 service observations above 180 minutes are retained. They are valid positive completions concentrated in larger Style/Tech jobs rather than timestamp failures. Tree models limit their influence without arbitrary clipping.

## Task 1 features

Only planning-time information is used. Features cover order size and density, outlet and dock access, vehicle capacity, route position and planned totals, window width and planned margins, calendar effects, traffic speed, date-specific disruption, district free-flow context, and service allowance. Clock time receives both minute and cyclic representations. Ratios use safe denominators. Categories use fold-fitted encoders with explicit unknown handling.

Actual departure, actual travel duration, arrival, leave time, and every current-row target derivative are absent from predictors. Preprocessing is fitted inside each chronological training fold. The final GPU XGBoost models use one-hot categories and median-imputed numeric features. No pretrained model is involved.

## Task 1 validation and model choice

Three expanding 41-day blocks reproduce the later-period deployment setting. Global/group baselines and regularized linear models precede boosting. GPU XGBoost configurations are compared on the two most recent folds against the saved CPU booster without retraining it. Depthwise XGBoost produces the best service MAE (4.044-minute mean; 3.788 latest fold). Loss-guided XGBoost produces the best lateness log loss (0.14245 mean; 0.13799 latest fold). The raw probability model is retained because earlier fold-safe Platt and isotonic tests did not improve the latest pseudo-test.

Permutation analysis confirms that service depends mainly on outlet, festival ramp, and order size, while lateness depends mainly on the planned margin to close, outlet, disruption, route sequence, and festival context. These relationships are operationally plausible and provide a leakage check.

## Task 2A construction

Demand combines `deliveries_train.csv` and `task1_test_inputs.csv`; their delivery IDs do not overlap. Every attempted, deferred, and not-run order is counted once in the week of `order_date`. Calendar ISO year/week fields define 117 complete observations for each depot-brand series. Total volume is the weekly sum. Chilled volume is the weekly sum of chilled Fresh orders and exactly zero for Style and Tech.

Features include shifted lags 1-6, 8, 13, 26, and 52; shifted rolling means and standard deviations; recent trends; cyclic week; time index; and future-known weekly calendar aggregates for operating days, payday, holiday, weekend, festival ramp, and monsoon share. Recursive backtests expose each horizon to earlier predictions, matching final inference.

Three rolling 10-week origins compare historical, recent, seasonal, regularized, and boosting forecasts. Gradient boosting wins for total volume (5.32% mean WAPE) and Fresh chilled volume (3.97%). Final post-processing clips negatives, forces non-Fresh chilled to zero, and caps Fresh chilled at Fresh total.

## Task 2B allocation

The allocation model reads the scenario, fleet, vehicle, district, and service-allowance files directly. Refrigerated orders are the scarce subproblem and are solved exactly with SciPy MILP. Ambient orders are packed deterministically with full capacity and time checks while reserving reefers. The official trip-time formula is used unchanged. A sensitivity comparison quantifies the cost of fairness versus maximum order count. The unchanged official checker reports `FEASIBILITY: PASSED`.

## Limitations

The organizers do not disclose the exact prediction metric, so selection uses multiple transparent surrogates. The service target has a long tail, and Task 2A contains only 117 weekly observations per series. Temporal validation reduces optimism but cannot reveal hidden-period shocks. Outlet identity is predictive and legitimate because all test outlets are historical, but performance would be less certain for a newly opened outlet.
