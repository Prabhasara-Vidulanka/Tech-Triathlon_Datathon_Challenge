# Waypoint Group Datathon — Final Submission

This package contains the deliverables for Task 1, Task 2A, and Task 2B.

## Required submissions

- `submission_task1.csv` — predicted service minutes and late probabilities for 5,014 deliveries.
- `submission_task2a.csv` — 10-week total and chilled volume forecasts for all depot/brand series.
- `submission_task2b.csv` — feasible vehicle/trip allocation with explicit deferrals.
- `DevOps_FinalNotebook.ipynb` — executed solution notebook; its final cell reloads the saved models and demonstrates inference without retraining.
- `models/task1_models.pkl` and `models/task2a_models.pkl` — final fitted model bundles.

## Run the notebook

1. Extract this package next to the organiser-provided `Data` folder.
2. Install the versions in `requirements.txt`.
3. Start Jupyter in this extracted folder and open `DevOps_FinalNotebook.ipynb`.
4. The notebook is saved with all 11 code cells executed. Running all cells again repeats label construction, feature generation, model training, evaluation, and allocation in a separate `work` directory. The final cell loads the submitted model files and demonstrates inference without retraining inside that cell.

Task 1 was trained with XGBoost 3.4.1 using CUDA. A compatible NVIDIA driver and CUDA-capable GPU are required only for retraining; loading the saved model and running inference can use the installed XGBoost package without retraining.

## Method summary

- Task 1 uses leakage-safe plan-time features, chronological validation, GPU XGBoost regression for service time, and GPU XGBoost classification for lateness.
- Task 2A aggregates all known demand by requested ISO week and uses shifted lag/rolling features with recursive 10-week backtesting.
- Task 2B combines an exact mixed-integer optimization for scarce refrigerated capacity with deterministic feasible packing for ambient orders, then validates every official rule.

## Quality checks

`reports/final_audit.json` records the submitted artifact checks. A fresh full notebook run also passed the official Task 2B validator and all quality gates; those regenerated outputs are in the sibling `work` directory.

The competition booklet does not name an exact prediction metric for Task 1 or Task 2A. The notebook therefore reports multiple standard chronological metrics and does not present a surrogate metric as official.

## Written deliverables

- `reports/data_preprocessing.pdf`
- `reports/architecture_diagrams.pdf`
- `reports/task2b_prioritization_policy.pdf`
- `reports/ai_tool_disclosure.pdf`

The remaining `reports/*.json` files supply the notebook's displayed audit and validation results. The `src/` modules contain the full pipeline and are required for the notebook's inference and reproduction cells.

The ZIP and notebook use `DevOps` as the team name. Confirm it matches the registered team name before upload. The required 3–5 minute unlisted YouTube demo video is submitted by URL through the organiser form; its URL is not included in this package.
