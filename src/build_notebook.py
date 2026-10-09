"""Build and execute the required final Datathon notebook in a clean kernel."""

from __future__ import annotations

import queue
from pathlib import Path

import nbformat as nbf
from jupyter_client import KernelManager


SCRIPT_PACKAGE = Path(__file__).resolve().parent.parent
WORK = SCRIPT_PACKAGE if (SCRIPT_PACKAGE / "models").is_dir() else Path(__file__).resolve().parents[2] / "work"
NOTEBOOK = WORK / "TeamName_FinalNotebook.ipynb"


def build() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            """# Tech-Triathlon 2026 — Waypoint Group Datathon

This notebook documents the reproducible final solution for Task 1, Task 2A, and Task 2B. The exact prediction metric is not stated in the official booklet, so the workflow reports multiple transparent chronological metrics and does not describe any surrogate as official.

All predictive models were trained from scratch on the supplied synthetic data. No proprietary modelling API, pretrained predictive model, external dataset, AutoML platform, or low-code/no-code modeller was used."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json, os, sys, warnings
import joblib
import numpy as np
import pandas as pd

os.environ["LOKY_MAX_CPU_COUNT"] = "16"
warnings.filterwarnings("ignore", message=".*mismatched devices.*")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

WORK = Path.cwd().resolve()
if not ((WORK / "src").is_dir() and (WORK / "models").is_dir()):
    candidates = [p / "work" for p in [WORK, *WORK.parents] if (p / "work" / "src").exists()]
    if not candidates:
        raise FileNotFoundError("Run this notebook from the extracted submission folder or project work directory")
    WORK = candidates[0]
data_candidates = [p / "Data" for p in [WORK, *WORK.parents] if (p / "Data").is_dir()]
if not data_candidates:
    raise FileNotFoundError("Place the supplied Data folder beside the submission folder")
DATA = data_candidates[0]
PROJECT = DATA.parent
sys.path.insert(0, str(WORK / "src"))

print("Project:", PROJECT)
print("Data:", DATA)
print("Python:", sys.version.split()[0])"""
        ),
        nbf.v4.new_markdown_cell("## Data inventory and integrity"),
        nbf.v4.new_code_cell(
            """audit = json.loads((WORK / "reports" / "initial_audit.json").read_text(encoding="utf-8"))
inventory = pd.DataFrame([
    {"file": item["path"], "rows": item["rows"], "columns": len(item["columns"]), "duplicate_rows": item["duplicate_rows"]}
    for item in audit["files"]
])
display(inventory)
print("Join audit:", json.dumps(audit["joins"], indent=2))
print("Template audit:", json.dumps(audit["templates"], indent=2))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Task 1 label construction

For dispatched orders, `(route_id, seq_in_route)` matches `(route_id, seq)` exactly once. Service begins at `max(actual_arrival, window_open)` so early waiting is excluded. Lateness is a strict comparison: `actual_arrival > window_close`. Actual-time fields are used only here and never enter the model feature matrix."""
        ),
        nbf.v4.new_code_cell(
            """label_audit = json.loads((WORK / "reports" / "task1_label_audit.json").read_text(encoding="utf-8"))
print("Formula:", label_audit["formula"])
print("Rows:", label_audit["rows"])
print("Early arrivals:", label_audit["early_arrivals"], f"({label_audit['early_arrival_rate']:.2%})")
print("Service median / mean:", label_audit["service_summary"]["50%"], "/", round(label_audit["service_summary"]["mean"], 3))
print("Late rate:", f"{label_audit['late_rate']:.2%}")
print("Arrival exactly at close (non-late):", label_audit["arrival_equal_close_rows"])
assert label_audit["window_cross_midnight_rows"] == 0
assert label_audit["leave_before_arrival_raw_rows"] == 0"""
        ),
        nbf.v4.new_markdown_cell("## Task 1 chronological validation and final GPU models"),
        nbf.v4.new_code_cell(
            """gpu = json.loads((WORK / "reports" / "task1_gpu_benchmark.json").read_text(encoding="utf-8"))
comparison = pd.DataFrame({
    "model": ["GPU XGBoost depth-6", "Saved CPU booster", "GPU XGBoost loss-guided", "Saved CPU booster"],
    "target": ["service", "service", "lateness", "lateness"],
    "two-fold mean": [
        gpu["scores"]["service"]["xgb_depth6"]["mean_mae"],
        gpu["scores"]["service"]["cpu_existing"]["mean_mae"],
        gpu["scores"]["late"]["xgb_lossguide"]["mean_log_loss"],
        gpu["scores"]["late"]["cpu_existing"]["mean_log_loss"],
    ],
    "metric": ["MAE minutes", "MAE minutes", "log loss", "log loss"],
})
display(comparison)
print("CUDA training confirmed:", gpu["cuda_training_confirmed"])
print("Selected:", gpu["selected_service_model"], "/", gpu["selected_late_model"])

errors = json.loads((WORK / "reports" / "error_analysis.json").read_text(encoding="utf-8"))
display(pd.DataFrame(errors["top_service_features"][:8]).rename(columns={"service_importance_mean": "permutation importance"}))
display(pd.DataFrame(errors["top_late_features"][:8]).rename(columns={"late_importance_mean": "permutation importance"}))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Task 2A weekly demand forecasting

Training demand combines every order from the training file and Task 1 test inputs, including deferred and not-run orders, and assigns volume to the requested ISO week. Lag and rolling features are shifted. Future calendar fields are known from the supplied calendar. Every backtest recursively predicts the next 10 weeks."""
        ),
        nbf.v4.new_code_cell(
            """task2a_report = json.loads((WORK / "reports" / "task2a_model_report.json").read_text(encoding="utf-8"))
rows = []
for target, detail in task2a_report["targets"].items():
    for model, score in detail["candidates"].items():
        rows.append({"target": target, "model": model, "mean WAPE": score["mean_wape"], "latest WAPE": score["latest_wape"], "selected": model == detail["selected"]})
display(pd.DataFrame(rows).sort_values(["target", "mean WAPE"]))
print("Forecast horizon:", task2a_report["forecast_start"], "to", task2a_report["forecast_end"])
print("Selected models:", task2a_report["selected"])
assert task2a_report["missing_series_weeks"] == 0"""
        ),
        nbf.v4.new_markdown_cell("## Task 2B allocation and official validation"),
        nbf.v4.new_code_cell(
            """allocation = json.loads((WORK / "reports" / "task2b_optimization_report.json").read_text(encoding="utf-8"))
final_audit = json.loads((WORK / "reports" / "final_audit.json").read_text(encoding="utf-8"))
display(pd.DataFrame(allocation["sensitivity"]).T[["served_orders", "deferred_orders", "deferred_yesterday_served", "days_since_last_served_sum", "chilled_orders_served"]])
print("Selected policy:", allocation["selected_policy"])
print("Maximum feasible order count:", allocation["selected_summary"]["maximum_feasible_order_count"])
print("Official validator:", final_audit["task2b_validator_output"])
assert final_audit["task2b_official_validator"]"""
        ),
        nbf.v4.new_markdown_cell("## Final saved-model inference demonstration"),
        nbf.v4.new_code_cell(
            """# FINAL CELL: load the saved models and run inference without retraining.
from inference import predict_task1, predict_task2a

task1_bundle = joblib.load(WORK / "models" / "task1_models.pkl")
task2a_bundle = joblib.load(WORK / "models" / "task2a_models.pkl")

task1_inputs = pd.read_csv(DATA / "Test Data" / "task1_test_inputs.csv").head(5)
task1_predictions = predict_task1().head(5)
print("TASK 1 REPRESENTATIVE INPUTS")
display(task1_inputs[["delivery_id", "outlet_id", "brand", "order_volume_m3", "planned_arrival_time", "window_close_time"]])
print("TASK 1 PREDICTIONS")
display(task1_predictions)

task2a_inputs = pd.read_csv(DATA / "Test Data" / "task2a_test_inputs.csv").head(6)
task2a_predictions = predict_task2a().head(6)
print("TASK 2A REPRESENTATIVE INPUTS")
display(task2a_inputs)
print("TASK 2A PREDICTIONS")
display(task2a_predictions)

print("Loaded Task 1 models:", task1_bundle["selected_service_model"], "/", task1_bundle["selected_late_model"])
print("Loaded Task 2A models:", task2a_bundle["selected"])
print("Inference completed from saved model files.")"""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": sys.version.split()[0] if False else "3.14.4"}
    return notebook


def execute(notebook: nbf.NotebookNode) -> None:
    km = KernelManager(kernel_name="python3")
    km.start_kernel(cwd=str(WORK))
    client = km.client()
    client.start_channels()
    client.wait_for_ready(timeout=60)
    execution_count = 0
    try:
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            execution_count += 1
            cell.execution_count = execution_count
            cell.outputs = []
            message_id = client.execute(cell.source, stop_on_error=True)
            while True:
                try:
                    message = client.get_iopub_msg(timeout=300)
                except queue.Empty as exc:
                    raise TimeoutError(f"Notebook cell {execution_count} timed out") from exc
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                message_type = message["msg_type"]
                content = message["content"]
                if message_type == "status" and content.get("execution_state") == "idle":
                    break
                if message_type == "stream":
                    cell.outputs.append(nbf.v4.new_output("stream", name=content["name"], text=content["text"]))
                elif message_type == "display_data":
                    cell.outputs.append(
                        nbf.v4.new_output(
                            "display_data",
                            data=content.get("data", {}),
                            metadata=content.get("metadata", {}),
                        )
                    )
                elif message_type == "execute_result":
                    cell.outputs.append(
                        nbf.v4.new_output(
                            "execute_result",
                            data=content.get("data", {}),
                            metadata=content.get("metadata", {}),
                            execution_count=execution_count,
                        )
                    )
                elif message_type == "error":
                    cell.outputs.append(
                        nbf.v4.new_output(
                            "error",
                            ename=content["ename"],
                            evalue=content["evalue"],
                            traceback=content["traceback"],
                        )
                    )
                    raise RuntimeError(f"Notebook cell {execution_count} failed: {content['ename']}: {content['evalue']}")
    finally:
        client.stop_channels()
        km.shutdown_kernel(now=True)


def main() -> None:
    notebook = build()
    execute(notebook)
    nbf.write(notebook, NOTEBOOK)
    loaded = nbf.read(NOTEBOOK, as_version=4)
    errors = [output for cell in loaded.cells if cell.cell_type == "code" for output in cell.get("outputs", []) if output.output_type == "error"]
    if errors:
        raise RuntimeError(f"Executed notebook contains {len(errors)} error output(s)")
    if any(cell.cell_type == "code" and cell.execution_count is None for cell in loaded.cells):
        raise RuntimeError("Executed notebook contains an unexecuted code cell")
    print(f"WROTE AND EXECUTED {NOTEBOOK}")


if __name__ == "__main__":
    main()
