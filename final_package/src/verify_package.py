"""Verify the final submission folder without retraining any model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


PACKAGE = Path(__file__).resolve().parent.parent


def main() -> None:
    required = [
        "DevOps_FinalNotebook.ipynb",
        "submission_task1.csv",
        "submission_task2a.csv",
        "submission_task2b.csv",
        "models/task1_models.pkl",
        "models/task2a_models.pkl",
        "README.md",
        "requirements.txt",
        "reports/final_audit.json",
        "reports/ai_tool_disclosure.pdf",
        "reports/architecture_diagrams.pdf",
        "reports/data_preprocessing.pdf",
        "reports/task2b_prioritization_policy.pdf",
        "reports/initial_audit.json",
        "reports/task1_label_audit.json",
        "reports/task1_model_report.json",
        "reports/task1_gpu_benchmark.json",
        "reports/task2a_model_report.json",
        "reports/task2b_optimization_report.json",
        "reports/error_analysis.json",
        "src/inference.py",
        "src/labels.py",
        "src/features_task1.py",
        "src/features_task2a.py",
        "src/train_task1.py",
        "src/train_task1_gpu.py",
        "src/train_task2a.py",
        "src/evaluate.py",
        "src/optimize_task2b.py",
        "src/audit.py",
    ]
    missing = [path for path in required if not (PACKAGE / path).is_file()]

    notebook = json.loads((PACKAGE / "DevOps_FinalNotebook.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    reproduction_steps = [
        "construct_task1_labels()",
        'build_task1_feature_frame("train")',
        "build_weekly_history()",
        "train_task1_baselines()",
        "train_task1_final()",
        "train_task2a_final()",
        "evaluate_holdout()",
        "validate_outputs()",
    ]
    notebook_errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]

    task1 = pd.read_csv(PACKAGE / "submission_task1.csv")
    task2a = pd.read_csv(PACKAGE / "submission_task2a.csv")
    task2b = pd.read_csv(PACKAGE / "submission_task2b.csv")
    final_audit = json.loads((PACKAGE / "reports" / "final_audit.json").read_text(encoding="utf-8"))
    task1_model = joblib.load(PACKAGE / "models" / "task1_models.pkl")
    task2a_model = joblib.load(PACKAGE / "models" / "task2a_models.pkl")

    checks = {
        "required_files_present": not missing,
        "missing_files": missing,
        "raw_data_excluded": not (PACKAGE / "Data").exists(),
        "notebook_code_cells": len(code_cells),
        "notebook_all_code_cells_executed": all(cell.get("execution_count") is not None for cell in code_cells),
        "notebook_reproduction_steps_present": all(step in source for step in reproduction_steps),
        "notebook_unexecuted_reproduction_cells": sum(cell.get("execution_count") is None for cell in code_cells),
        "notebook_final_inference_executed": bool(code_cells[-1].get("execution_count") is not None and "joblib.load" in "".join(code_cells[-1]["source"])),
        "notebook_error_outputs": len(notebook_errors),
        "task1_shape": list(task1.shape),
        "task2a_shape": list(task2a.shape),
        "task2b_shape": list(task2b.shape),
        "task1_columns": list(task1.columns),
        "task2a_columns": list(task2a.columns),
        "task2b_columns": list(task2b.columns),
        "task1_model_loaded": bool(task1_model.get("service_components")) and bool(task1_model.get("late_components")),
        "task2a_model_loaded": set(task2a_model.get("models", {})) == {"total_volume_m3", "chilled_volume_m3"},
        "all_quality_gates_passed": final_audit.get("all_quality_gates_passed", False),
        "official_validator": final_audit.get("task2b_validator_output"),
    }
    checks["package_ready"] = all(
        [
            checks["required_files_present"],
            checks["raw_data_excluded"],
            checks["notebook_reproduction_steps_present"],
            checks["notebook_final_inference_executed"],
            checks["notebook_error_outputs"] == 0,
            checks["task1_shape"] == [5014, 3],
            checks["task2a_shape"] == [60, 3],
            checks["task2b_shape"] == [85, 6],
            checks["task1_model_loaded"],
            checks["task2a_model_loaded"],
            checks["all_quality_gates_passed"],
        ]
    )
    output = PACKAGE / "reports" / "package_audit.json"
    output.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    if not checks["package_ready"]:
        raise SystemExit("PACKAGE VERIFICATION FAILED")
    print("PACKAGE VERIFICATION PASSED")


if __name__ == "__main__":
    main()
