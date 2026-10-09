"""Reload saved models and reproduce Task 1 and Task 2A predictions."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import joblib
import numpy as np
import pandas as pd

from features_task1 import build_task1_feature_frame
from features_task2a import forecast_rows
from project_paths import find_project_root
from train_task1_gpu import predict_ensemble
from train_task2a import recursive_baseline, recursive_model_forecast


ROOT = find_project_root(__file__)
DATA = ROOT / "Data"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
WORK = PACKAGE_ROOT if (PACKAGE_ROOT / "models").is_dir() else ROOT / "work"


def predict_task1() -> pd.DataFrame:
    bundle = joblib.load(WORK / "models" / "task1_models.pkl")
    features = build_task1_feature_frame("test")
    columns = bundle["categorical_features"] + bundle["numeric_features"]
    X = features[columns]
    service = np.maximum(
        0.0,
        predict_ensemble(bundle["service_components"], bundle["service_weights"], X, probability=False),
    )
    late = np.clip(
        predict_ensemble(bundle["late_components"], bundle["late_weights"], X, probability=True),
        0.0,
        1.0,
    )
    predictions = pd.DataFrame(
        {"delivery_id": features["delivery_id"], "pred_service_min": service, "pred_late_prob": late}
    )
    template = pd.read_csv(DATA / "Submission Templates" / "submission_task1.csv")
    return template[["delivery_id"]].merge(predictions, on="delivery_id", validate="one_to_one")


def predict_task2a() -> pd.DataFrame:
    bundle = joblib.load(WORK / "models" / "task2a_models.pkl")
    history = bundle["history"]
    future = forecast_rows()
    predictions: dict[str, pd.DataFrame] = {}
    for target in ["total_volume_m3", "chilled_volume_m3"]:
        target_history = history if target == "total_volume_m3" else history.loc[history["brand"].eq("Fresh")].copy()
        target_future = future if target == "total_volume_m3" else future.loc[future["brand"].eq("Fresh")].copy()
        specification = bundle["models"][target]
        if specification["model"] is None:
            predictions[target] = recursive_baseline(
                target_history, target_future, target, specification["method"]
            )
        else:
            predictions[target] = recursive_model_forecast(
                target_history,
                target_future,
                target,
                specification["model"],
                specification["log_target"],
            )
    keys = ["depot", "brand", "iso_year", "iso_week"]
    total = predictions["total_volume_m3"][keys + ["prediction"]].rename(
        columns={"prediction": "pred_total_volume_m3"}
    )
    chilled = predictions["chilled_volume_m3"][keys + ["prediction"]].rename(
        columns={"prediction": "pred_chilled_volume_m3"}
    )
    output = future[["row_id", *keys]].merge(total, on=keys, validate="one_to_one")
    output = output.merge(chilled, on=keys, how="left", validate="one_to_one")
    output["pred_chilled_volume_m3"] = output["pred_chilled_volume_m3"].fillna(0.0).clip(lower=0)
    output["pred_total_volume_m3"] = output["pred_total_volume_m3"].clip(lower=0)
    fresh = output["brand"].eq("Fresh")
    output.loc[fresh, "pred_chilled_volume_m3"] = np.minimum(
        output.loc[fresh, "pred_chilled_volume_m3"], output.loc[fresh, "pred_total_volume_m3"]
    )
    output.loc[~fresh, "pred_chilled_volume_m3"] = 0.0
    template = pd.read_csv(DATA / "Submission Templates" / "submission_task2a.csv")
    return template[["row_id"]].merge(
        output[["row_id", "pred_total_volume_m3", "pred_chilled_volume_m3"]],
        on="row_id",
        validate="one_to_one",
    )


def main() -> None:
    task1 = predict_task1()
    task2a = predict_task2a()
    output_dir = WORK / "submissions" if (WORK / "submissions").is_dir() else WORK
    task1.to_csv(output_dir / "submission_task1.csv", index=False)
    task2a.to_csv(output_dir / "submission_task2a.csv", index=False)
    print("Task 1 sample")
    print(task1.head().to_string(index=False))
    print("\nTask 2A sample")
    print(task2a.head().to_string(index=False))


if __name__ == "__main__":
    main()
