"""GPU XGBoost benchmark under the existing chronological validation protocol."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features_task1 import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_task1_feature_frame
from labels import build_task1_labels
from train_task1 import (
    classification_metrics,
    chronological_splits,
    regression_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
WORK = ROOT / "work"
SEED = 20261006


def xgb_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=2), CATEGORICAL_FEATURES),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ]
    )


CONFIGS = {
    "xgb_depth6": {
        "max_depth": 6,
        "min_child_weight": 12,
        "grow_policy": "depthwise",
        "max_leaves": 0,
    },
    "xgb_lossguide": {
        "max_depth": 0,
        "min_child_weight": 12,
        "grow_policy": "lossguide",
        "max_leaves": 31,
    },
}


def regressor(config: dict, n_estimators: int = 1200, early_stopping_rounds: int | None = 60) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=n_estimators,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=8.0,
        max_bin=256,
        tree_method="hist",
        device="cuda",
        random_state=SEED,
        n_jobs=4,
        early_stopping_rounds=early_stopping_rounds,
        **config,
    )


def classifier(config: dict, n_estimators: int = 1200, early_stopping_rounds: int | None = 60) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=8.0,
        max_bin=256,
        tree_method="hist",
        device="cuda",
        random_state=SEED,
        n_jobs=4,
        early_stopping_rounds=early_stopping_rounds,
        **config,
    )


def predict_component(component: dict, X: pd.DataFrame, probability: bool = False) -> np.ndarray:
    if component["kind"] == "hgb":
        return component["model"].predict_proba(X)[:, 1] if probability else component["model"].predict(X)
    transformed = component["preprocess"].transform(X)
    # The explicit DMatrix path lets a CUDA booster ingest host-side sparse features
    # without sklearn's harmless device-mismatch warning during saved-model inference.
    return component["model"].get_booster().predict(xgb.DMatrix(transformed))


def predict_ensemble(components: list[dict], weights: list[float], X: pd.DataFrame, probability: bool = False) -> np.ndarray:
    predictions = [predict_component(component, X, probability) for component in components]
    return np.average(np.vstack(predictions), axis=0, weights=np.asarray(weights))


def main() -> None:
    started = time.perf_counter()
    features = build_task1_feature_frame("train")
    test_features = build_task1_feature_frame("test")
    labels, label_report = build_task1_labels()
    data = features.merge(labels[["delivery_id", "service_min", "late"]], on="delivery_id", validate="one_to_one")
    data = data.sort_values(["dispatch_date", "delivery_id"]).reset_index(drop=True)
    columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = data[columns]
    y_service = data["service_min"].to_numpy(float)
    y_late = data["late"].to_numpy(int)
    splits = chronological_splits(data["dispatch_date"])[-2:]

    fold_predictions = {"service": {}, "late": {}}
    fold_targets = {"service": [], "late": []}
    best_iterations: dict[str, dict[str, list[int]]] = {
        "service": {name: [] for name in CONFIGS},
        "late": {name: [] for name in CONFIGS},
    }
    cuda_configs: list[str] = []

    for fold, (train_idx, valid_idx, _) in enumerate(splits, start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        service_train, service_valid = y_service[train_idx], y_service[valid_idx]
        late_train, late_valid = y_late[train_idx], y_late[valid_idx]
        fold_targets["service"].append(service_valid)
        fold_targets["late"].append(late_valid)

        for name, config in CONFIGS.items():
            preprocessor = xgb_preprocessor()
            transformed_train = preprocessor.fit_transform(X_train)
            transformed_valid = preprocessor.transform(X_valid)

            service_model = regressor(config)
            service_model.fit(transformed_train, service_train, eval_set=[(transformed_valid, service_valid)], verbose=False)
            service_prediction = service_model.predict(transformed_valid)
            fold_predictions["service"].setdefault(name, []).append(service_prediction)
            best_iterations["service"][name].append(int(service_model.best_iteration) + 1)
            cuda_configs.append(service_model.get_booster().save_config())

            late_model = classifier(config)
            late_model.fit(transformed_train, late_train, eval_set=[(transformed_valid, late_valid)], verbose=False)
            late_prediction = late_model.predict_proba(transformed_valid)[:, 1]
            fold_predictions["late"].setdefault(name, []).append(late_prediction)
            best_iterations["late"][name].append(int(late_model.best_iteration) + 1)

    scores: dict[str, dict] = {"service": {}, "late": {}}
    for name, predictions in fold_predictions["service"].items():
        fold_metrics = [regression_metrics(y, pred) for y, pred in zip(fold_targets["service"], predictions)]
        scores["service"][name] = {
            "folds": fold_metrics,
            "mean_mae": float(np.mean([item["mae"] for item in fold_metrics])),
            "latest_mae": fold_metrics[-1]["mae"],
        }
    for name, predictions in fold_predictions["late"].items():
        fold_metrics = [classification_metrics(y, pred) for y, pred in zip(fold_targets["late"], predictions)]
        scores["late"][name] = {
            "folds": fold_metrics,
            "mean_log_loss": float(np.mean([item["log_loss"] for item in fold_metrics])),
            "latest_log_loss": fold_metrics[-1]["log_loss"],
        }

    cpu_report = json.loads((WORK / "reports" / "task1_model_report.json").read_text(encoding="utf-8"))
    cpu_service_folds = cpu_report["service"]["hgb_squared"]["folds"][-2:]
    cpu_late_folds = cpu_report["late"]["hgb"]["folds"][-2:]
    scores["service"]["cpu_existing"] = {
        "folds": cpu_service_folds,
        "mean_mae": float(np.mean([item["mae"] for item in cpu_service_folds])),
        "latest_mae": cpu_service_folds[-1]["mae"],
        "note": "Saved benchmark only; no CPU retraining in this run",
    }
    scores["late"]["cpu_existing"] = {
        "folds": cpu_late_folds,
        "mean_log_loss": float(np.mean([item["log_loss"] for item in cpu_late_folds])),
        "latest_log_loss": cpu_late_folds[-1]["log_loss"],
        "note": "Saved benchmark only; no CPU retraining in this run",
    }
    service_winner = min(scores["service"], key=lambda name: (scores["service"][name]["mean_mae"], scores["service"][name]["latest_mae"]))
    late_winner = min(scores["late"], key=lambda name: (scores["late"][name]["mean_log_loss"], scores["late"][name]["latest_log_loss"]))
    cpu_bundle = joblib.load(WORK / "models" / "task1_models_cpu.pkl")

    def fit_components(winner: str, target: str) -> tuple[list[dict], list[float]]:
        values = y_service if target == "service" else y_late
        components: list[dict] = []
        weights: list[float] = []
        if winner == "cpu_existing":
            model_key = "service_model" if target == "service" else "late_model"
            components.append({"kind": "hgb", "model": cpu_bundle[model_key]})
            weights.append(1.0)
        if winner.startswith("xgb_"):
            config_name = winner
            config = CONFIGS[config_name]
            preprocessor = xgb_preprocessor()
            transformed = preprocessor.fit_transform(X)
            iterations = max(50, int(np.median(best_iterations[target][config_name])))
            model = regressor(config, iterations, None) if target == "service" else classifier(config, iterations, None)
            model.fit(transformed, values, verbose=False)
            components.append({"kind": "xgb", "preprocess": preprocessor, "model": model})
            weights.append(1.0)
        return components, weights

    service_components, service_weights = fit_components(service_winner, "service")
    late_components, late_weights = fit_components(late_winner, "late")
    test_X = test_features[columns]
    service_prediction = np.maximum(0.0, predict_ensemble(service_components, service_weights, test_X, False))
    late_probability = np.clip(predict_ensemble(late_components, late_weights, test_X, True), 0.0, 1.0)

    bundle = {
        "service_components": service_components,
        "service_weights": service_weights,
        "late_components": late_components,
        "late_weights": late_weights,
        "late_calibrator": None,
        "calibration_method": "none",
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "seed": SEED,
        "label_formula": label_report["formula"],
        "selected_service_model": service_winner,
        "selected_late_model": late_winner,
    }
    model_path = WORK / "models" / "task1_models.pkl"
    joblib.dump(bundle, model_path)
    template = pd.read_csv(DATA / "Submission Templates" / "submission_task1.csv")
    predictions = pd.DataFrame(
        {
            "delivery_id": test_features["delivery_id"],
            "pred_service_min": service_prediction,
            "pred_late_prob": late_probability,
        }
    )
    submission = template[["delivery_id"]].merge(predictions, on="delivery_id", validate="one_to_one")
    submission.to_csv(WORK / "submissions" / "submission_task1.csv", index=False)

    reloaded = joblib.load(model_path)
    assert np.allclose(service_prediction, predict_ensemble(reloaded["service_components"], reloaded["service_weights"], test_X))
    assert np.allclose(late_probability, predict_ensemble(reloaded["late_components"], reloaded["late_weights"], test_X, True))
    cuda_confirmed = any('"device":"cuda:0"' in config.replace(" ", "") for config in cuda_configs)
    report = {
        "xgboost_version": xgb.__version__,
        "build_info": xgb.build_info(),
        "cuda_training_confirmed": cuda_confirmed,
        "validation_folds": [metadata for _, _, metadata in splits],
        "scores": scores,
        "best_iterations": best_iterations,
        "selected_service_model": service_winner,
        "selected_late_model": late_winner,
        "prediction_summary": submission.iloc[:, 1:].describe().to_dict(),
        "runtime_seconds": time.perf_counter() - started,
    }
    (WORK / "reports" / "task1_gpu_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
