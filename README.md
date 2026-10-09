# Waypoint Group Datathon — Final Submission

This package contains the complete solution for Task 1, Task 2A, and Task 2B.

## Required submissions

- `submission_task1.csv` — predicted service minutes and late probabilities for 5,014 deliveries.
- `submission_task2a.csv` — 10-week total and chilled volume forecasts for all depot/brand series.
- `submission_task2b.csv` — feasible vehicle/trip allocation with explicit deferrals.
- `TeamName_FinalNotebook.ipynb` — executed solution notebook; its final cell reloads the saved models and demonstrates inference without retraining.
- `models/task1_models.pkl` and `models/task2a_models.pkl` — final fitted model bundles.

## Run the notebook

1. Extract this package next to the organiser-provided `Data` folder.
2. Install the versions in `requirements.txt`.
3. Start Jupyter in this extracted folder and open `TeamName_FinalNotebook.ipynb`.
4. Run all cells. The last cell loads the saved model files and reproduces representative predictions.

Task 1 was trained with XGBoost 3.4.1 using CUDA. A compatible NVIDIA driver and CUDA-capable GPU are required only for retraining; loading the saved model and running inference can use the installed XGBoost package without retraining.

## Method summary

- Task 1 uses leakage-safe plan-time features, chronological validation, GPU XGBoost regression for service time, and GPU XGBoost classification for lateness.
- Task 2A aggregates all known demand by requested ISO week and uses shifted lag/rolling features with recursive 10-week backtesting.
- Task 2B combines an exact mixed-integer optimization for scarce refrigerated capacity with deterministic feasible packing for ambient orders, then validates every official rule.

## Quality checks

`reports/final_audit.json` records the final automated checks. The official Task 2B validator result is: `FEASIBILITY: PASSED - every rule satisfied.`

The competition booklet does not name an exact prediction metric for Task 1 or Task 2A. The notebook therefore reports multiple standard chronological metrics and does not present a surrogate metric as official.

## Supporting material

- `reports/data_preprocessing.md`
- `reports/architecture_diagrams.md`
- `reports/task2b_prioritization_policy.md`
- `reports/ai_tool_disclosure.md`
- `reports/demo_video_outline.md`
- `figures/` and `reports/` for validation evidence and diagnostics

Before upload, replace the literal `TeamName` in the ZIP and notebook filenames with the registered team name if the organisers require it.
