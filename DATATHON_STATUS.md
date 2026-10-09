# Datathon progress and deliverables

Status reviewed: 9 October 2026 (Sri Lanka time). This inventory is based on the cloned repository and the Datathon requirements on pages 15–23 of the local `Challenge Booklet.pdf`. “Complete” means the artifact exists and the repository records its checks; it does not mean a competition submission has been uploaded or scored.

## What we have completed

| Workstream | Completed work | Evidence |
| --- | --- | --- |
| Data preparation | Audited the supplied data, checked joins and template identifiers, constructed Task 1 labels, and built planning-time features without actual journey outcomes. Task 2A demand includes attempted, deferred, and not-run orders by requested ISO week. | [Preprocessing write-up](reports/data_preprocessing.md), [initial audit](reports/initial_competition_audit.md) |
| Task 1: service time and lateness | Trained and selected GPU XGBoost models; produced 5,014 service-minute and late-probability predictions. The recorded mean on the two most recent chronological folds is 4.044 minutes MAE for service and 0.14245 log loss for lateness. These are validation metrics, not official scores. | [Submission](submissions/submission_task1.csv), [model selection](reports/best_models.json), [saved model](models/task1_models.pkl) |
| Task 2A: depot demand | Built six depot/brand weekly demand series and recursive ten-week forecasts; produced 60 total-volume and chilled-volume predictions. Recorded mean backtest WAPE is 5.32% for total volume and 3.97% for chilled volume. These are validation metrics, not official scores. | [Submission](submissions/submission_task2a.csv), [model selection](reports/best_models.json), [saved model](models/task2a_models.pkl) |
| Task 2B: peak-day fleet | Produced decisions for all 85 orders using a fairness-first policy: 77 served, 8 deferred, and all 10 orders deferred the previous day served. The official feasibility checker is recorded as passing every rule. | [Submission](submissions/submission_task2b.csv), [policy](reports/task2b_prioritization_policy.md), [optimization report](reports/task2b_optimization_report.json) |
| Reproducibility and explanation | Created an executed notebook with label construction, preprocessing, training, evaluation, and a final saved-model reload and inference demonstration. Prepared architecture diagrams, experiment records, diagnostics, figures, and AI disclosure. | [Packaged notebook](final_package/DevOps_FinalNotebook.ipynb), [architecture](reports/architecture_diagrams.md), [AI disclosure](reports/ai_tool_disclosure.md), [experiments](experiments/experiments.csv) |

## Deliverable checklist

| Booklet deliverable | Current status | Location or action |
| --- | --- | --- |
| `submission_task1.csv` | Complete | [Packaged CSV](final_package/submission_task1.csv): 5,014 rows, required columns and identifiers checked. |
| `submission_task2a.csv` | Complete | [Packaged CSV](final_package/submission_task2a.csv): 60 rows, required columns and identifiers checked. |
| `submission_task2b.csv` | Complete | [Packaged CSV](final_package/submission_task2b.csv): 85 rows, official feasibility pass recorded. |
| Written Task 2B priority and deferral policy | Complete | [Policy PDF](final_package/reports/task2b_prioritization_policy.pdf) and [Markdown source](final_package/reports/task2b_prioritization_policy.md). |
| Final notebook | Complete; confirm registered team name | [Packaged notebook](final_package/DevOps_FinalNotebook.ipynb): 7 of 7 code cells executed, with no recorded error outputs. |
| Final model files | Complete | [Task 1 model](final_package/models/task1_models.pkl) and [Task 2A model](final_package/models/task2a_models.pkl). |
| Architecture diagrams | Complete | [Diagram PDF](final_package/reports/architecture_diagrams.pdf) and [Markdown source](final_package/reports/architecture_diagrams.md) cover both models, preprocessing, Task 2B, and a proposed deployment approach. |
| Data preprocessing document | Complete | [Write-up PDF](final_package/reports/data_preprocessing.pdf) and [Markdown source](final_package/reports/data_preprocessing.md) cover data preparation, labels, cleaning, features, and rationale. |
| AI tool disclosure | Complete | [Disclosure PDF](final_package/reports/ai_tool_disclosure.pdf) and [Markdown source](final_package/reports/ai_tool_disclosure.md). |
| Demo video, 3–5 minutes, unlisted on YouTube | Pending / not found in this repository | A [video outline](final_package/reports/demo_video_outline.md) exists, but no Datathon video file or YouTube URL is present here. The `DEMO.mp4` in the parent workspace is outside this repository and is not identified as the Datathon demo. |
| `DevOps_Datathon.zip` upload package | Complete; confirm registered team name | [ZIP archive](DevOps_Datathon.zip) contains 51 files. ZIP integrity passed; all required file deliverables are present and no raw `Data` folder is included. |
| Submission form upload | Not verifiable from local files | No upload receipt or submission confirmation is present in the repository. |

## Checks and limits

- The recorded [final audit](reports/final_audit.json) passes the CSV schema, row order, value, model reload, and Task 2B validator checks. The [package audit](final_package/reports/package_audit.json) was rerun in the pinned dependency environment and passed file presence, CSV shapes, notebook execution metadata, saved-model loading, and raw-data exclusion. The ZIP was reopened and its internal CRC integrity passed.
- The organiser's raw `Data` directory and official checker are not in this repository, so training, notebook execution, and the official Task 2B checker were not rerun here. The notebook's full run requires the organiser data alongside the package, as described in the [package README](final_package/README.md).
- The booklet does not disclose an exact scoring metric for Task 1 or Task 2A. The validation figures above should not be presented as leaderboard results.
- The Datathon is judged separately from the Hackathon. Integration with the `waypoint-logistics` application is not a required Datathon deliverable.

## Remaining submission steps

1. Confirm `DevOps` is the registered team name on the notebook and ZIP.
2. Record a 3–5 minute Datathon walkthrough, upload it as an unlisted YouTube video, and include its URL in the submission form.
3. Upload [DevOps_Datathon.zip](DevOps_Datathon.zip) through the organiser's submission form and retain the confirmation.

The booklet states a deadline of **9 October 2026 at 11:59 PM Sri Lanka time**.
