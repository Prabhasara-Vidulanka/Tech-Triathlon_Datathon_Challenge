"""Final model, submission, and official-validator quality gates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from inference import predict_task1, predict_task2a


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
WORK = ROOT / "work"


def main() -> None:
    checks: dict[str, object] = {}

    task1_template = pd.read_csv(DATA / "Submission Templates" / "submission_task1.csv")
    task1 = pd.read_csv(WORK / "submissions" / "submission_task1.csv")
    task1_rebuilt = predict_task1()
    checks["task1_columns"] = list(task1.columns) == list(task1_template.columns)
    checks["task1_rows"] = len(task1) == len(task1_template)
    checks["task1_ids_and_order"] = task1["delivery_id"].equals(task1_template["delivery_id"])
    checks["task1_no_duplicates"] = not task1["delivery_id"].duplicated().any()
    checks["task1_finite"] = bool(np.isfinite(task1.iloc[:, 1:].to_numpy()).all())
    checks["task1_service_nonnegative"] = bool(task1["pred_service_min"].ge(0).all())
    checks["task1_probability_bounds"] = bool(task1["pred_late_prob"].between(0, 1).all())
    checks["task1_reload_reproduces"] = bool(
        np.allclose(task1.iloc[:, 1:].to_numpy(), task1_rebuilt.iloc[:, 1:].to_numpy())
    )

    task2a_template = pd.read_csv(DATA / "Submission Templates" / "submission_task2a.csv")
    task2a_inputs = pd.read_csv(DATA / "Test Data" / "task2a_test_inputs.csv")
    task2a = pd.read_csv(WORK / "submissions" / "submission_task2a.csv")
    task2a_rebuilt = predict_task2a()
    joined_2a = task2a_inputs.merge(task2a, on="row_id", validate="one_to_one")
    checks["task2a_columns"] = list(task2a.columns) == list(task2a_template.columns)
    checks["task2a_rows"] = len(task2a) == len(task2a_template)
    checks["task2a_ids_and_order"] = task2a["row_id"].equals(task2a_template["row_id"])
    checks["task2a_nonnegative"] = bool(task2a.iloc[:, 1:].ge(0).all().all())
    checks["task2a_nonfresh_chilled_zero"] = bool(
        joined_2a.loc[joined_2a["brand"].ne("Fresh"), "pred_chilled_volume_m3"].eq(0).all()
    )
    checks["task2a_fresh_chilled_within_total"] = bool(
        (
            joined_2a.loc[joined_2a["brand"].eq("Fresh"), "pred_chilled_volume_m3"]
            <= joined_2a.loc[joined_2a["brand"].eq("Fresh"), "pred_total_volume_m3"]
        ).all()
    )
    checks["task2a_reload_reproduces"] = bool(
        np.allclose(task2a.iloc[:, 1:].to_numpy(), task2a_rebuilt.iloc[:, 1:].to_numpy())
    )

    task2b_template = pd.read_csv(DATA / "Submission Templates" / "submission_task2b.csv")
    task2b = pd.read_csv(WORK / "submissions" / "submission_task2b.csv")
    checks["task2b_columns"] = list(task2b.columns) == list(task2b_template.columns)
    checks["task2b_rows"] = len(task2b) == len(task2b_template)
    checks["task2b_ids_and_order"] = task2b.iloc[:, :3].equals(task2b_template.iloc[:, :3])
    checks["task2b_decisions"] = set(task2b["decision"]) <= {"served", "deferred"}
    deferred = task2b["decision"].eq("deferred")
    served = task2b["decision"].eq("served")
    checks["task2b_deferred_blanks"] = bool(task2b.loc[deferred, ["vehicle_id", "trip_id"]].isna().all().all())
    checks["task2b_served_assignments"] = bool(task2b.loc[served, ["vehicle_id", "trip_id"]].notna().all().all())
    validator = subprocess.run(
        [sys.executable, str(DATA / "check_allocation.py"), str(WORK / "submissions" / "submission_task2b.csv")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    checks["task2b_official_validator"] = validator.returncode == 0 and "FEASIBILITY: PASSED" in validator.stdout
    checks["task2b_validator_output"] = validator.stdout.strip()

    task1_bundle = joblib.load(WORK / "models" / "task1_models.pkl")
    task2a_bundle = joblib.load(WORK / "models" / "task2a_models.pkl")
    checks["task1_model_reload"] = bool(task1_bundle["service_components"] and task1_bundle["late_components"])
    checks["task2a_model_reload"] = bool(task2a_bundle["models"] and len(task2a_bundle["history"]) == 702)

    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    checks["all_quality_gates_passed"] = all(boolean_checks)
    path = WORK / "reports" / "final_audit.json"
    path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    if not checks["all_quality_gates_passed"]:
        raise SystemExit("One or more final quality gates failed")


if __name__ == "__main__":
    main()
