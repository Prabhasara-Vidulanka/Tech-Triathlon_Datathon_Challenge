"""Chronological validation, model selection, and final inference for Task 1."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from features_task1 import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_task1_feature_frame
from labels import build_task1_labels


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
WORK = ROOT / "work"
SEED = 20261006


def chronological_splits(dates: pd.Series, n_splits: int = 3, horizon_days: int = 41) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    parsed = pd.to_datetime(dates)
    last_end = parsed.max()
    starts = [last_end - pd.Timedelta(days=horizon_days - 1) - pd.Timedelta(days=42 * offset) for offset in reversed(range(n_splits))]
    splits = []
    for start in starts:
        end = start + pd.Timedelta(days=horizon_days - 1)
        train_idx = np.flatnonzero(parsed.lt(start).to_numpy())
        valid_idx = np.flatnonzero(parsed.between(start, end).to_numpy())
        if not len(train_idx) or not len(valid_idx):
            raise ValueError(f"Empty chronological split {start.date()} to {end.date()}")
        splits.append(
            (
                train_idx,
                valid_idx,
                {
                    "train_start": str(parsed.iloc[train_idx].min().date()),
                    "train_end": str(parsed.iloc[train_idx].max().date()),
                    "valid_start": str(parsed.iloc[valid_idx].min().date()),
                    "valid_end": str(parsed.iloc[valid_idx].max().date()),
                    "train_rows": int(len(train_idx)),
                    "valid_rows": int(len(valid_idx)),
                },
            )
        )
    return splits


def ordinal_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def onehot_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
        ]
    )


def service_hgb(loss: str = "absolute_error", max_leaf_nodes: int = 31, min_samples_leaf: int = 30) -> Pipeline:
    mask = [True] * len(CATEGORICAL_FEATURES) + [False] * len(NUMERIC_FEATURES)
    return Pipeline(
        [
            ("preprocess", ordinal_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss=loss,
                    learning_rate=0.07,
                    max_iter=350,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=1.0,
                    categorical_features=mask,
                    early_stopping=True,
                    random_state=SEED,
                ),
            ),
        ]
    )


def late_hgb(max_leaf_nodes: int = 31, min_samples_leaf: int = 40) -> Pipeline:
    mask = [True] * len(CATEGORICAL_FEATURES) + [False] * len(NUMERIC_FEATURES)
    return Pipeline(
        [
            ("preprocess", ordinal_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=0.06,
                    max_iter=350,
                    max_leaf_nodes=max_leaf_nodes,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=2.0,
                    categorical_features=mask,
                    early_stopping=True,
                    random_state=SEED,
                ),
            ),
        ]
    )


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "median_ae": float(median_absolute_error(y, pred)),
        "bias": float(np.mean(pred - y)),
    }


def classification_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    clipped = np.clip(prob, 1e-6, 1 - 1e-6)
    fraction, predicted = calibration_curve(y, clipped, n_bins=10, strategy="quantile")
    return {
        "log_loss": float(log_loss(y, clipped)),
        "brier": float(brier_score_loss(y, clipped)),
        "roc_auc": float(roc_auc_score(y, clipped)),
        "pr_auc": float(average_precision_score(y, clipped)),
        "calibration_error": float(np.mean(np.abs(fraction - predicted))),
        "mean_probability": float(clipped.mean()),
        "positive_rate": float(np.mean(y)),
    }


def group_service_baseline(train: pd.DataFrame, y: np.ndarray, valid: pd.DataFrame) -> np.ndarray:
    table = train.assign(target=y).groupby(["brand", "dock_type"])["target"].median()
    global_median = float(np.median(y))
    index = pd.MultiIndex.from_frame(valid[["brand", "dock_type"]])
    return table.reindex(index).fillna(global_median).to_numpy()


def append_experiments(rows: list[dict]) -> None:
    path = WORK / "experiments" / "experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def main() -> None:
    started = time.perf_counter()
    features = build_task1_feature_frame("train")
    test_features = build_task1_feature_frame("test")
    labels, label_report = build_task1_labels()
    dataset = features.merge(labels[["delivery_id", "service_min", "late"]], on="delivery_id", validate="one_to_one")
    dataset = dataset.sort_values(["dispatch_date", "delivery_id"]).reset_index(drop=True)
    X = dataset[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_service = dataset["service_min"].to_numpy(float)
    y_late = dataset["late"].to_numpy(int)
    splits = chronological_splits(dataset["dispatch_date"])

    experiment_rows: list[dict] = []
    summary: dict = {"splits": [metadata for _, _, metadata in splits], "service": {}, "late": {}}
    service_candidates = {
        "global_median": None,
        "brand_dock_median": None,
        "ridge": Pipeline([("preprocess", onehot_preprocessor()), ("model", Ridge(alpha=10.0))]),
        "hgb_absolute": service_hgb("absolute_error", 31, 30),
        "hgb_squared": service_hgb("squared_error", 31, 30),
    }
    late_candidates = {
        "global_rate": None,
        "logistic": Pipeline(
            [
                ("preprocess", onehot_preprocessor()),
                ("model", LogisticRegression(C=0.3, max_iter=500, solver="lbfgs", random_state=SEED)),
            ]
        ),
        "hgb": late_hgb(),
    }

    service_fold_predictions: dict[str, list[np.ndarray]] = {name: [] for name in service_candidates}
    late_fold_predictions: dict[str, list[np.ndarray]] = {name: [] for name in late_candidates}
    late_fold_targets: list[np.ndarray] = []

    for fold, (train_idx, valid_idx, metadata) in enumerate(splits, start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        service_train, service_valid = y_service[train_idx], y_service[valid_idx]
        late_train, late_valid = y_late[train_idx], y_late[valid_idx]
        late_fold_targets.append(late_valid)

        for name, candidate in service_candidates.items():
            tic = time.perf_counter()
            if name == "global_median":
                prediction = np.full(len(valid_idx), np.median(service_train))
                train_prediction = np.full(len(train_idx), np.median(service_train))
            elif name == "brand_dock_median":
                prediction = group_service_baseline(X_train, service_train, X_valid)
                train_prediction = group_service_baseline(X_train, service_train, X_train)
            else:
                candidate.fit(X_train, service_train)
                prediction = candidate.predict(X_valid)
                train_prediction = candidate.predict(X_train)
            valid_metrics = regression_metrics(service_valid, prediction)
            train_metrics = regression_metrics(service_train, train_prediction)
            service_fold_predictions[name].append(prediction)
            experiment_rows.append(
                {
                    "experiment_id": f"T1S-{name}-F{fold}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "task": "Task1",
                    "target": "service_min",
                    "feature_set": "planned_reference_v1",
                    "model": name,
                    "hyperparameters": "see train_task1.py",
                    "validation_scheme": f"chronological_{metadata['valid_start']}_{metadata['valid_end']}",
                    "fold_scores": json.dumps(valid_metrics),
                    "mean_score": valid_metrics["mae"],
                    "std_score": np.nan,
                    "chronological_holdout_score": valid_metrics["mae"],
                    "training_score": train_metrics["mae"],
                    "generalization_gap": valid_metrics["mae"] - train_metrics["mae"],
                    "runtime": time.perf_counter() - tic,
                    "seed": SEED,
                    "notes": json.dumps(metadata),
                    "leakage_risk": "low: planned/reference features only",
                    "improved_best": False,
                    "artifact_path": "",
                }
            )

        for name, candidate in late_candidates.items():
            tic = time.perf_counter()
            if name == "global_rate":
                probability = np.full(len(valid_idx), late_train.mean())
                train_probability = np.full(len(train_idx), late_train.mean())
            else:
                candidate.fit(X_train, late_train)
                probability = candidate.predict_proba(X_valid)[:, 1]
                train_probability = candidate.predict_proba(X_train)[:, 1]
            valid_metrics = classification_metrics(late_valid, probability)
            train_metrics = classification_metrics(late_train, train_probability)
            late_fold_predictions[name].append(probability)
            experiment_rows.append(
                {
                    "experiment_id": f"T1L-{name}-F{fold}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "task": "Task1",
                    "target": "late",
                    "feature_set": "planned_reference_v1",
                    "model": name,
                    "hyperparameters": "see train_task1.py",
                    "validation_scheme": f"chronological_{metadata['valid_start']}_{metadata['valid_end']}",
                    "fold_scores": json.dumps(valid_metrics),
                    "mean_score": valid_metrics["log_loss"],
                    "std_score": np.nan,
                    "chronological_holdout_score": valid_metrics["log_loss"],
                    "training_score": train_metrics["log_loss"],
                    "generalization_gap": valid_metrics["log_loss"] - train_metrics["log_loss"],
                    "runtime": time.perf_counter() - tic,
                    "seed": SEED,
                    "notes": json.dumps(metadata),
                    "leakage_risk": "low: planned/reference features only",
                    "improved_best": False,
                    "artifact_path": "",
                }
            )

    for name, folds in service_fold_predictions.items():
        metrics = [regression_metrics(y_service[valid_idx], prediction) for (_, valid_idx, _), prediction in zip(splits, folds)]
        summary["service"][name] = {
            "folds": metrics,
            "mean_mae": float(np.mean([metric["mae"] for metric in metrics])),
            "std_mae": float(np.std([metric["mae"] for metric in metrics])),
            "pseudo_test_mae": metrics[-1]["mae"],
        }
    for name, folds in late_fold_predictions.items():
        metrics = [classification_metrics(target, prediction) for target, prediction in zip(late_fold_targets, folds)]
        summary["late"][name] = {
            "folds": metrics,
            "mean_log_loss": float(np.mean([metric["log_loss"] for metric in metrics])),
            "std_log_loss": float(np.std([metric["log_loss"] for metric in metrics])),
            "pseudo_test_log_loss": metrics[-1]["log_loss"],
        }

    service_name = min(summary["service"], key=lambda name: (summary["service"][name]["mean_mae"], summary["service"][name]["pseudo_test_mae"]))
    late_name = min(summary["late"], key=lambda name: (summary["late"][name]["mean_log_loss"], summary["late"][name]["pseudo_test_log_loss"]))
    if service_name not in {"hgb_absolute", "hgb_squared", "ridge"}:
        service_name = "hgb_absolute"
    if late_name not in {"hgb", "logistic"}:
        late_name = "hgb"

    # Fold-safe calibration diagnostic: learn on the first two time folds and
    # evaluate only on the latest pseudo-test fold.
    calibration_method = "none"
    calibration_model = None
    raw_oof = late_fold_predictions[late_name]
    calibration_x = np.concatenate(raw_oof[:2])
    calibration_y = np.concatenate(late_fold_targets[:2])
    pseudo_x = raw_oof[-1]
    pseudo_y = late_fold_targets[-1]
    raw_metrics = classification_metrics(pseudo_y, pseudo_x)
    platt = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED).fit(calibration_x.reshape(-1, 1), calibration_y)
    platt_metrics = classification_metrics(pseudo_y, platt.predict_proba(pseudo_x.reshape(-1, 1))[:, 1])
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(calibration_x, calibration_y)
    isotonic_metrics = classification_metrics(pseudo_y, isotonic.predict(pseudo_x))
    calibration_comparison = {"none": raw_metrics, "platt": platt_metrics, "isotonic": isotonic_metrics}
    winner = min(calibration_comparison, key=lambda key: (calibration_comparison[key]["log_loss"], calibration_comparison[key]["brier"]))
    if winner == "platt":
        calibration_method = "platt"
        calibration_model = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED).fit(
            np.concatenate(raw_oof).reshape(-1, 1), np.concatenate(late_fold_targets)
        )
    elif winner == "isotonic":
        calibration_method = "isotonic"
        calibration_model = IsotonicRegression(out_of_bounds="clip").fit(
            np.concatenate(raw_oof), np.concatenate(late_fold_targets)
        )

    service_model = service_candidates[service_name]
    late_model = late_candidates[late_name]
    service_model.fit(X, y_service)
    late_model.fit(X, y_late)

    test_X = test_features[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    service_prediction = np.maximum(0.0, service_model.predict(test_X))
    late_probability = late_model.predict_proba(test_X)[:, 1]
    if calibration_method == "platt":
        late_probability = calibration_model.predict_proba(late_probability.reshape(-1, 1))[:, 1]
    elif calibration_method == "isotonic":
        late_probability = calibration_model.predict(late_probability)
    late_probability = np.clip(late_probability, 0.0, 1.0)

    model_dir = WORK / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = model_dir / "task1_models.pkl"
    joblib.dump(
        {
            "service_model": service_model,
            "late_model": late_model,
            "late_calibrator": calibration_model,
            "calibration_method": calibration_method,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "seed": SEED,
            "label_formula": label_report["formula"],
        },
        bundle_path,
    )

    template = pd.read_csv(DATA / "Submission Templates" / "submission_task1.csv")
    prediction_frame = pd.DataFrame(
        {
            "delivery_id": test_features["delivery_id"],
            "pred_service_min": service_prediction,
            "pred_late_prob": late_probability,
        }
    )
    submission = template[["delivery_id"]].merge(prediction_frame, on="delivery_id", how="left", validate="one_to_one")
    if not submission["delivery_id"].equals(template["delivery_id"]):
        raise AssertionError("Task 1 row order changed")
    if submission.isna().any().any() or not np.isfinite(submission.iloc[:, 1:].to_numpy()).all():
        raise AssertionError("Task 1 predictions contain missing/non-finite values")
    if not submission["pred_late_prob"].between(0, 1).all() or submission["pred_service_min"].lt(0).any():
        raise AssertionError("Task 1 predictions violate output bounds")
    submission_path = WORK / "submissions" / "submission_task1.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)

    # Prove that the serialized objects reproduce the generated predictions.
    reloaded = joblib.load(bundle_path)
    reloaded_service = np.maximum(0.0, reloaded["service_model"].predict(test_X))
    reloaded_late = reloaded["late_model"].predict_proba(test_X)[:, 1]
    if reloaded["calibration_method"] == "platt":
        reloaded_late = reloaded["late_calibrator"].predict_proba(reloaded_late.reshape(-1, 1))[:, 1]
    elif reloaded["calibration_method"] == "isotonic":
        reloaded_late = reloaded["late_calibrator"].predict(reloaded_late)
    assert np.allclose(service_prediction, reloaded_service)
    assert np.allclose(late_probability, reloaded_late)

    summary.update(
        {
            "selected_service_model": service_name,
            "selected_late_model": late_name,
            "calibration_method": calibration_method,
            "calibration_comparison_on_pseudo_test": calibration_comparison,
            "prediction_summary": submission.iloc[:, 1:].describe().to_dict(),
            "runtime_seconds": time.perf_counter() - started,
            "artifacts": {"model": str(bundle_path), "submission": str(submission_path)},
        }
    )
    report_path = WORK / "reports" / "task1_model_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_experiments(experiment_rows)
    print(json.dumps(summary, indent=2))
    print(f"WROTE {bundle_path}")
    print(f"WROTE {submission_path}")
    print(f"WROTE {report_path}")


if __name__ == "__main__":
    main()
