# Datathon Demo Video Outline (3-5 minutes)

## 0:00-0:25 — Problem and outputs

Introduce Waypoint's shared distribution network and the three deliverables: outlet service/lateness prediction, ten-week depot-brand demand forecasts, and a feasible peak-day fleet allocation.

## 0:25-0:55 — Data and label construction

Show the one-to-one order/route join. Explain that early vehicles wait, so service starts at the later of arrival and window opening. Define lateness as arrival strictly after closing. Mention the 4,023 early arrivals that make this distinction material.

## 0:55-1:25 — Preprocessing and leakage control

Show plan-time order, outlet, route, calendar, traffic, road, vehicle, and allowance features. Explicitly exclude actual journey and completion fields from predictors. Explain the later-period train/test split.

## 1:25-2:05 — Task 1 models and results

Show the 41-day expanding-window validation. Compare baselines with GPU XGBoost. Report service MAE of 4.044 minutes across the two most recent folds and lateness log loss of 0.14245, with latest-fold ROC-AUC about 0.973. Mention that fold-safe calibration did not improve the raw probabilities.

## 2:05-2:40 — Task 2A forecasting

Show the six weekly series and explain requested-week aggregation, lags, rolling statistics, and future-known calendar features. Describe recursive 10-week backtests. Report 5.32% mean WAPE for total volume and 3.97% for chilled Fresh volume. Show the chilled-volume constraints.

## 2:40-3:30 — Task 2B allocation

Explain the four-reefer bottleneck and the reefer van's two-trip requirement. Show the fairness-first policy, 77 served orders, all 10 prior deferrals protected, and the 79-order maximum-coverage comparison. Point out the individually impossible 40.66 m³ Style order. Display the official validator pass.

## 3:30-4:10 — Architecture and reproducibility

Show the three architecture diagrams, saved `.pkl` files, experiment log, model reload tests, exact template audits, and final notebook inference cell.

## 4:10-4:30 — Challenges and close

Summarize the main challenge: avoiding temporal leakage while balancing scarce reefers against fairness. State that the exact competition metric was not disclosed, so multiple transparent validation metrics were used. End with the final submission artifacts and truthful AI disclosure.
