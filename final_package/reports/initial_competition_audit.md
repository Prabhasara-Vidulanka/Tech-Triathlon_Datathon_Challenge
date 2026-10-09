# Initial Competition Audit

## A. Files discovered

The official material is present and internally consistent:

- `Challenge Booklet.pdf` (33 pages) and the 40-slide kick-off deck.
- Seven general-data CSVs, two training CSVs, five test CSVs, and three submission templates.
- `Data/check_allocation.py`, the official Task 2B feasibility checker.
- The original nested data archive and an extracted copy. Raw files were not modified.
- A separate Hackathon application under `waypoint-logistics/`; Datathon integration is not required by the booklet.

The detailed inventory, schemas, missingness, hashes, and template checks are in `initial_audit.json`.

## B. Dataset shapes

| File | Rows | Columns |
|---|---:|---:|
| deliveries_train.csv | 92,307 | 20 |
| route_legs_train.csv | 91,894 | 22 |
| task1_test_inputs.csv | 5,014 | 20 |
| route_legs_test.csv | 5,014 | 18 |
| task2a_test_inputs.csv | 60 | 5 |
| task2b_peak_day_scenarios.csv | 85 | 17 |
| task2b_peak_day_fleet.csv | 38 | 3 |
| outlets.csv | 120 | 9 |
| vehicles.csv | 60 | 9 |
| calendar.csv | 910 | 12 |
| district_travel.csv | 12 | 8 |
| service_allowance.csv | 9 | 3 |
| traffic_speed.csv | 576 | 4 |
| road_conditions.csv | 10,920 | 3 |

## C. Date ranges

- Task 1 training orders and route legs: 2024-01-01 through 2026-02-14.
- Task 1 test orders and route legs: 2026-02-16 through 2026-03-28.
- Task 2A combined order history: 2024-01-01 through 2026-03-28, covering 117 complete weekly observations for each of six depot-brand series.
- Task 2A forecast: ISO weeks 14-23 of 2026, from 2026-03-30 through 2026-06-07.
- Calendar and road data: 2024-01-01 through 2026-06-28.

## D. Key joins

- Training contains 90,351 attempted, 1,543 deferred, and 413 not-run orders.
- All 91,894 dispatched orders join one-to-one to a unique route leg on `(route_id, seq_in_route) = (route_id, seq)`.
- Outlet, vehicle, brand, district, depot, and dispatch-date/leg-date consistency all have zero mismatches.
- The 413 not-run rows correctly have no dispatch or route fields.
- All 5,014 Task 1 test orders match one unique test leg, with zero outlet mismatches.
- Submission templates match source identifiers, row counts, and required order exactly.

## E. Task 1 label formulas

For each dispatched order:

```text
service_start = max(actual_arrival, window_open)
service_min   = leave_outlet_time - service_start
late          = 1(actual_arrival > window_close)
```

The strict lateness inequality follows the booklet. A total of 4,023 rows (4.38%) arrived before opening and would have had waiting time incorrectly included by a raw leave-minus-arrival label. No supplied window or route sequence crosses midnight. The final service label has no missing or negative values; its median is 15 minutes and mean is 19.00 minutes. The late rate is 19.58%; 279 arrivals exactly at closing are correctly non-late. All 1,267 mall rows have order windows identical to the stated mall window.

## F. Leakage risks

The feature pipeline explicitly rejects actual departure, actual travel duration, arrival, outlet-leave time, current-row targets, and target-derived current-row features. It fits preprocessing separately inside every chronological fold. Pure identifiers are excluded except outlet and vehicle identities, which are legitimate plan-time entities with repeated historical observations. Permutation checks show plausible dominance by outlet, festival context, order size, planned closing margin, and disruption rather than actual outcomes.

## G. Train/test distribution concerns

- The test period is strictly later than training and must not be evaluated with random folds.
- Test means are moderately higher for units, weight, and volume, but all brands, districts, depots, outlets, temperature classes, and vehicle classes occur in training.
- Five historical vehicle IDs do not appear in test; test has no unseen vehicles.
- Lateness prevalence varies materially over time in pseudo-test folds, so probability calibration is monitored on time blocks rather than assumed stable.

## H. Validation schemes

- Task 1: three expanding-window, 41-day chronological validation blocks. The latest block, 2026-01-05 through 2026-02-14, mirrors the 41-day hidden test period.
- Task 2A: three rolling origins, each forecasting the next 10 weeks recursively.
- Task 2B: exact MILP for refrigerated scarcity, deterministic capacity/time-aware packing for ambient work, policy sensitivity runs, a maximum-coverage bound, and the unchanged official validator.

## I. Baselines built first

- Service: global median, brand-dock median, regularized ridge regression.
- Lateness: global prevalence and regularized logistic regression.
- Demand: historical mean, rolling means, seasonal lag 52, seasonal/recent blend, and ridge regression.
- Allocation: maximum-order coverage versus fairness-first and cold-chain policies.

## J. Task 2B optimization strategy

The scarce refrigerated subproblem is solved exactly with binary order-trip and trip-group variables. Constraints encode refrigeration, van access, home depot, whole orders, capacity, two trips, brand/district purity, and the official trip-time equation. Ambient demand is then packed using only ambient vehicles, preserving reefer capacity. The selected lexicographic policy protects prior deferrals first, then stale outlets, chilled/Fresh urgency, and coverage.

## K. Competition-rule compliance risks

No proprietary modelling API, pretrained model, external dataset, AutoML system, or low-code modeller is used. All data remains local. XGBoost is trained from scratch on the supplied synthetic data. AI assistance is disclosed. The exact prediction metric remains unspecified: the booklet and kick-off deck state only “Performance score (Task 1, Task 2A).” MAE/RMSE and log loss/Brier/AUC are therefore development surrogates, not claimed official metrics.

## L. Immediate experiments completed

Task 1 compared linear, CPU boosting, and GPU XGBoost candidates. GPU XGBoost won on the two most recent folds: service MAE 4.044 minutes and lateness log loss 0.14245. Task 2A gradient boosting reached 5.32% mean WAPE for total volume and 3.97% for Fresh chilled volume. The Task 2B allocation passes the official validator.
