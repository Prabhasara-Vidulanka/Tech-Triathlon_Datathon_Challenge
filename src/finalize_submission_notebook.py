"""Add the reproducible pipeline cells to the packaged Datathon notebook."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "final_package" / "DevOps_FinalNotebook.ipynb"
MARKER = "## Reproduce the modelling pipeline"


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    if any(cell.cell_type == "markdown" and MARKER in cell.source for cell in notebook.cells):
        for cell in notebook.cells:
            if cell.cell_type == "code" and "validate_outputs()" in cell.source and "original_inference_work" not in cell.source:
                cell.source = cell.source.replace(
                    "    from audit import main as validate_outputs",
                    "    import inference\n    from audit import main as validate_outputs",
                ).replace(
                    "    validate_outputs()      # template checks, model reload and official Task 2B checker",
                    "    original_inference_work = inference.WORK\n"
                    "    inference.WORK = REPRO\n"
                    "    try:\n"
                    "        validate_outputs()  # template checks, model reload and official Task 2B checker\n"
                    "    finally:\n"
                    "        inference.WORK = original_inference_work",
                )
        nbformat.validate(notebook)
        nbformat.write(notebook, NOTEBOOK)
        print("Updated reproduction cells")
        return

    section = [
        nbformat.v4.new_markdown_cell(
            """## Reproduce the modelling pipeline

The cells below call the same local source modules used to create the submitted models and predictions. They document label construction, preprocessing, fitting, chronological evaluation, allocation, and the final audit. The organiser's `Data` folder is required. Set `RUN_FULL_REPRODUCTION = True` to regenerate artifacts in a separate `work` directory; Task 1 retraining also requires a CUDA-capable GPU. The switch is off in this submitted copy so the saved-model inference cell remains quick to run. The training cells have not been re-executed in this package because the raw competition data is excluded."""
        ),
        nbformat.v4.new_code_cell(
            """# Reproduction controls: outputs go to PROJECT/work, leaving the submitted files intact.
RUN_FULL_REPRODUCTION = False
REPRO = PROJECT / "work"
if RUN_FULL_REPRODUCTION:
    for folder in ("features", "models", "submissions", "reports", "experiments", "figures"):
        (REPRO / folder).mkdir(parents=True, exist_ok=True)
print("Full reproduction enabled:", RUN_FULL_REPRODUCTION)"""
        ),
        nbformat.v4.new_markdown_cell("### Label construction and preprocessing"),
        nbformat.v4.new_code_cell(
            """if RUN_FULL_REPRODUCTION:
    from audit_data import main as audit_raw_data
    from labels import main as construct_task1_labels
    from features_task1 import build_task1_feature_frame
    from features_task2a import build_weekly_history

    audit_raw_data()
    construct_task1_labels()  # service starts at max(arrival, window open); late means arrival > close
    task1_train_features = build_task1_feature_frame("train")
    task1_test_features = build_task1_feature_frame("test")
    weekly_demand = build_weekly_history()  # includes deferred and not-run orders
    print("Task 1 train/test feature rows:", len(task1_train_features), len(task1_test_features))
    print("Task 2A depot/brand/week rows:", len(weekly_demand))"""
        ),
        nbformat.v4.new_markdown_cell("### Fit and compare Task 1 and Task 2A models"),
        nbformat.v4.new_code_cell(
            """if RUN_FULL_REPRODUCTION:
    import shutil
    from train_task1 import main as train_task1_baselines
    from train_task1_gpu import main as train_task1_final
    from train_task2a import main as train_task2a_final

    train_task1_baselines()  # chronological CPU baselines and evaluation report
    shutil.copy2(REPRO / "models" / "task1_models.pkl", REPRO / "models" / "task1_models_cpu.pkl")
    train_task1_final()      # chronological GPU model comparison and final Task 1 outputs
    train_task2a_final()     # rolling ten-week backtests and final Task 2A outputs"""
        ),
        nbformat.v4.new_markdown_cell("### Held-out analysis, peak-day allocation and output validation"),
        nbformat.v4.new_code_cell(
            """if RUN_FULL_REPRODUCTION:
    from evaluate import main as evaluate_holdout
    from optimize_task2b import main as allocate_peak_day
    import inference
    from audit import main as validate_outputs

    evaluate_holdout()      # subgroup errors, diagnostics and feature importance
    allocate_peak_day()     # Task 2B allocation and prioritization comparison
    original_inference_work = inference.WORK
    inference.WORK = REPRO
    try:
        validate_outputs()  # template checks, model reload and official Task 2B checker
    finally:
        inference.WORK = original_inference_work
    print("Reproduced artifacts:", REPRO)"""
        ),
    ]
    # The saved-model inference demonstration remains the final cell, as required.
    notebook.cells[-1:-1] = section
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    print(f"Added {len(section)} cells to {NOTEBOOK}")


if __name__ == "__main__":
    main()
