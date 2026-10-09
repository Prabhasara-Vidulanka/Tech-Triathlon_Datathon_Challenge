"""Rolling-origin evaluation and final 10-week forecasts for Task 2A."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from features_task2a import (
    MODEL_CATEGORICAL,
    MODEL_NUMERIC,
    build_weekly_history,
    forecast_rows,
    supervised_features,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
WORK = ROOT / "work"
SEED = 20261006


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    denominator = np.abs(y).sum()
    smape_denominator = np.abs(y) + np.abs(pred)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "wape": float(np.abs(y - pred).sum() / denominator) if denominator else 0.0,
        "smape": float(np.mean(np.divide(2 * np.abs(y - pred), smape_denominator, out=np.zeros_like(y), where=smape_denominator > 0))),
        "bias": float(np.mean(pred - y)),
    }


def model_candidates() -> dict[str, tuple[Pipeline, bool]]:
    onehot = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), MODEL_CATEGORICAL),
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                MODEL_NUMERIC,
            ),
        ]
    )
    ordinal = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
                MODEL_CATEGORICAL,
            ),
            ("numeric", SimpleImputer(strategy="median"), MODEL_NUMERIC),
        ],
        verbose_feature_names_out=False,
    )
    mask = [True] * len(MODEL_CATEGORICAL) + [False] * len(MODEL_NUMERIC)
    ridge = Pipeline([("preprocess", onehot), ("model", Ridge(alpha=20.0))])
    hgb = Pipeline(
        [
            ("preprocess", ordinal),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=15,
                    min_samples_leaf=12,
                    l2_regularization=5.0,
                    categorical_features=mask,
                    early_stopping=True,
                    random_state=SEED,
                ),
            ),
        ]
    )
    return {"ridge": (ridge, False), "hgb": (hgb, False), "hgb_log": (clone(hgb), True)}


def train_model(history: pd.DataFrame, target: str, template: Pipeline, log_target: bool) -> Pipeline:
    train = supervised_features(history, target)
    # Thirteen weeks of history are required for stable short/medium-term lags;
    # lag-26 and lag-52 are imputed until they become available.
    train = train.loc[train["lag_13"].notna() & train[target].notna()].copy()
    X = train[MODEL_CATEGORICAL + MODEL_NUMERIC]
    y = train[target].to_numpy(float)
    model = clone(template)
    model.fit(X, np.log1p(y) if log_target else y)
    return model


def recursive_model_forecast(
    history: pd.DataFrame,
    future: pd.DataFrame,
    target: str,
    model: Pipeline,
    log_target: bool,
) -> pd.DataFrame:
    future_panel = future.drop(columns=["row_id"], errors="ignore").copy()
    future_panel[target] = np.nan
    for other in ["total_volume_m3", "chilled_volume_m3", "order_count"]:
        if other not in future_panel:
            future_panel[other] = np.nan
    working = pd.concat([history, future_panel], ignore_index=True, sort=False)
    working = working.sort_values(["week_start", "depot", "brand"]).reset_index(drop=True)
    prediction_rows = []
    for week in sorted(pd.to_datetime(future_panel["week_start"]).unique()):
        featured = supervised_features(working, target)
        mask = pd.to_datetime(featured["week_start"]).eq(week) & featured[target].isna()
        X_week = featured.loc[mask, MODEL_CATEGORICAL + MODEL_NUMERIC]
        predicted = model.predict(X_week)
        if log_target:
            predicted = np.expm1(predicted)
        predicted = np.maximum(0.0, predicted)
        keys = featured.loc[mask, ["depot", "brand", "iso_year", "iso_week", "week_start"]].copy()
        keys["prediction"] = predicted
        prediction_rows.append(keys)
        lookup = keys.set_index(["depot", "brand", "iso_year", "iso_week"])["prediction"]
        working_index = pd.MultiIndex.from_frame(working[["depot", "brand", "iso_year", "iso_week"]])
        fill_mask = working_index.isin(lookup.index) & working[target].isna().to_numpy()
        working.loc[fill_mask, target] = working_index[fill_mask].map(lookup).to_numpy()
    return pd.concat(prediction_rows, ignore_index=True)


def recursive_baseline(history: pd.DataFrame, future: pd.DataFrame, target: str, method: str) -> pd.DataFrame:
    future_panel = future.drop(columns=["row_id"], errors="ignore").copy()
    future_panel[target] = np.nan
    for other in ["total_volume_m3", "chilled_volume_m3", "order_count"]:
        if other not in future_panel:
            future_panel[other] = np.nan
    working = pd.concat([history, future_panel], ignore_index=True, sort=False)
    working = working.sort_values(["week_start", "depot", "brand"]).reset_index(drop=True)
    output = []
    for week in sorted(pd.to_datetime(future_panel["week_start"]).unique()):
        featured = supervised_features(working, target)
        mask = pd.to_datetime(featured["week_start"]).eq(week) & featured[target].isna()
        rows = featured.loc[mask].copy()
        if method == "historical_mean":
            means = history.groupby(["depot", "brand"])[target].mean()
            pred = pd.MultiIndex.from_frame(rows[["depot", "brand"]]).map(means).to_numpy()
        elif method == "rolling4":
            pred = rows["rolling_mean_4"].to_numpy(float)
        elif method == "rolling13":
            pred = rows["rolling_mean_13"].to_numpy(float)
        elif method == "seasonal52":
            pred = rows["lag_52"].fillna(rows["rolling_mean_13"]).to_numpy(float)
        elif method == "seasonal_recent_blend":
            seasonal = rows["lag_52"].fillna(rows["rolling_mean_13"]).to_numpy(float)
            pred = 0.6 * seasonal + 0.4 * rows["rolling_mean_4"].to_numpy(float)
        else:
            raise ValueError(method)
        pred = np.maximum(0.0, pred)
        keys = rows[["depot", "brand", "iso_year", "iso_week", "week_start"]].copy()
        keys["prediction"] = pred
        output.append(keys)
        lookup = keys.set_index(["depot", "brand", "iso_year", "iso_week"])["prediction"]
        working_index = pd.MultiIndex.from_frame(working[["depot", "brand", "iso_year", "iso_week"]])
        fill_mask = working_index.isin(lookup.index) & working[target].isna().to_numpy()
        working.loc[fill_mask, target] = working_index[fill_mask].map(lookup).to_numpy()
    return pd.concat(output, ignore_index=True)


def evaluate_method(
    history: pd.DataFrame,
    target: str,
    validation_weeks: list[pd.Timestamp],
    method: str,
    model_spec: tuple[Pipeline, bool] | None,
) -> tuple[pd.DataFrame, dict]:
    start = min(validation_weeks)
    train = history.loc[pd.to_datetime(history["week_start"]).lt(start)].copy()
    actual = history.loc[pd.to_datetime(history["week_start"]).isin(validation_weeks)].copy()
    future = actual.drop(columns=["total_volume_m3", "chilled_volume_m3", "order_count"], errors="ignore")
    if model_spec is None:
        predicted = recursive_baseline(train, future, target, method)
    else:
        template, log_target = model_spec
        fitted = train_model(train, target, template, log_target)
        predicted = recursive_model_forecast(train, future, target, fitted, log_target)
    keys = ["depot", "brand", "iso_year", "iso_week"]
    scored = actual[keys + [target]].merge(predicted[keys + ["prediction"]], on=keys, validate="one_to_one")
    scored["horizon"] = scored.sort_values("iso_week").groupby(["depot", "brand"]).cumcount() + 1
    result = metrics(scored[target].to_numpy(float), scored["prediction"].to_numpy(float))
    result["by_brand_wape"] = {
        brand: metrics(group[target].to_numpy(float), group["prediction"].to_numpy(float))["wape"]
        for brand, group in scored.groupby("brand")
    }
    result["by_depot_wape"] = {
        depot: metrics(group[target].to_numpy(float), group["prediction"].to_numpy(float))["wape"]
        for depot, group in scored.groupby("depot")
    }
    return scored, result


def append_experiments(rows: list[dict]) -> None:
    path = WORK / "experiments" / "experiments.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)


def main() -> None:
    started = time.perf_counter()
    history = build_weekly_history()
    future = forecast_rows()
    all_weeks = sorted(pd.to_datetime(history["week_start"]).unique())
    origins = [all_weeks[-30:-20], all_weeks[-20:-10], all_weeks[-10:]]
    baseline_methods = ["historical_mean", "rolling4", "rolling13", "seasonal52", "seasonal_recent_blend"]
    models = model_candidates()
    methods = {name: None for name in baseline_methods} | models
    report: dict = {
        "history_start": str(pd.to_datetime(history["week_start"]).min().date()),
        "history_end": str(pd.to_datetime(history["week_start"]).max().date()),
        "forecast_start": str(pd.to_datetime(future["week_start"]).min().date()),
        "forecast_end": str(pd.to_datetime(future["week_start"]).max().date()),
        "series_rows": {f"{d}|{b}": int(n) for (d, b), n in history.groupby(["depot", "brand"]).size().items()},
        "missing_series_weeks": int(history["order_count"].eq(0).sum()),
        "targets": {},
    }
    experiment_rows: list[dict] = []
    selected: dict[str, str] = {}
    fitted_models: dict[str, dict] = {}
    final_predictions: dict[str, pd.DataFrame] = {}

    for target in ["total_volume_m3", "chilled_volume_m3"]:
        target_history = history if target == "total_volume_m3" else history.loc[history["brand"].eq("Fresh")].copy()
        target_future = future if target == "total_volume_m3" else future.loc[future["brand"].eq("Fresh")].copy()
        target_results: dict[str, dict] = {}
        for method, spec in methods.items():
            fold_results = []
            for fold, weeks in enumerate(origins, start=1):
                tic = time.perf_counter()
                weeks = [pd.Timestamp(week) for week in weeks]
                _, score = evaluate_method(target_history, target, weeks, method, spec)
                fold_results.append(score)
                experiment_rows.append(
                    {
                        "experiment_id": f"T2A-{target}-{method}-F{fold}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "task": "Task2A",
                        "target": target,
                        "feature_set": "weekly_calendar_lags_v1",
                        "model": method,
                        "hyperparameters": "see train_task2a.py",
                        "validation_scheme": f"rolling_origin_10w_{weeks[0].date()}_{weeks[-1].date()}",
                        "fold_scores": json.dumps(score),
                        "mean_score": score["wape"],
                        "std_score": np.nan,
                        "chronological_holdout_score": score["wape"],
                        "training_score": np.nan,
                        "generalization_gap": np.nan,
                        "runtime": time.perf_counter() - tic,
                        "seed": SEED,
                        "notes": "recursive 10-week forecast",
                        "leakage_risk": "low: lags shifted; future calendar only",
                        "improved_best": False,
                        "artifact_path": "",
                    }
                )
            target_results[method] = {
                "folds": fold_results,
                "mean_wape": float(np.mean([item["wape"] for item in fold_results])),
                "std_wape": float(np.std([item["wape"] for item in fold_results])),
                "mean_mae": float(np.mean([item["mae"] for item in fold_results])),
                "latest_wape": fold_results[-1]["wape"],
            }
        # Selection balances average performance with the latest pseudo-test.
        winner = min(target_results, key=lambda name: (target_results[name]["mean_wape"] + target_results[name]["latest_wape"], target_results[name]["std_wape"]))
        selected[target] = winner
        spec = methods[winner]
        if spec is None:
            prediction = recursive_baseline(target_history, target_future, target, winner)
            fitted_models[target] = {"method": winner, "model": None, "log_target": False}
        else:
            template, log_target = spec
            fitted = train_model(target_history, target, template, log_target)
            prediction = recursive_model_forecast(target_history, target_future, target, fitted, log_target)
            fitted_models[target] = {"method": winner, "model": fitted, "log_target": log_target}
        final_predictions[target] = prediction
        report["targets"][target] = {"candidates": target_results, "selected": winner}

    keys = ["depot", "brand", "iso_year", "iso_week"]
    total = final_predictions["total_volume_m3"][keys + ["prediction"]].rename(columns={"prediction": "pred_total_volume_m3"})
    chilled_fresh = final_predictions["chilled_volume_m3"][keys + ["prediction"]].rename(columns={"prediction": "pred_chilled_volume_m3"})
    output = future[["row_id", *keys]].merge(total, on=keys, how="left", validate="one_to_one")
    output = output.merge(chilled_fresh, on=keys, how="left", validate="one_to_one")
    output["pred_chilled_volume_m3"] = output["pred_chilled_volume_m3"].fillna(0.0)
    output["pred_total_volume_m3"] = output["pred_total_volume_m3"].clip(lower=0)
    output["pred_chilled_volume_m3"] = output["pred_chilled_volume_m3"].clip(lower=0)
    fresh = output["brand"].eq("Fresh")
    output.loc[fresh, "pred_chilled_volume_m3"] = np.minimum(
        output.loc[fresh, "pred_chilled_volume_m3"], output.loc[fresh, "pred_total_volume_m3"]
    )
    output.loc[~fresh, "pred_chilled_volume_m3"] = 0.0

    template = pd.read_csv(DATA / "Submission Templates" / "submission_task2a.csv")
    submission = template[["row_id"]].merge(
        output[["row_id", "pred_total_volume_m3", "pred_chilled_volume_m3"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if not submission["row_id"].equals(template["row_id"]):
        raise AssertionError("Task 2A row order changed")
    if submission.isna().any().any() or submission.iloc[:, 1:].lt(0).any().any():
        raise AssertionError("Task 2A output has missing or negative predictions")
    check = output.set_index("row_id").loc[submission["row_id"]]
    if (check.loc[check["brand"].ne("Fresh"), "pred_chilled_volume_m3"] != 0).any():
        raise AssertionError("Non-Fresh chilled prediction is nonzero")
    if (check.loc[check["brand"].eq("Fresh"), "pred_chilled_volume_m3"] > check.loc[check["brand"].eq("Fresh"), "pred_total_volume_m3"]).any():
        raise AssertionError("Fresh chilled prediction exceeds total")

    submission_path = WORK / "submissions" / "submission_task2a.csv"
    model_path = WORK / "models" / "task2a_models.pkl"
    submission.to_csv(submission_path, index=False)
    joblib.dump(
        {
            "models": fitted_models,
            "selected": selected,
            "history": history,
            "model_categorical": MODEL_CATEGORICAL,
            "model_numeric": MODEL_NUMERIC,
            "seed": SEED,
        },
        model_path,
    )
    reloaded = joblib.load(model_path)
    if reloaded["selected"] != selected or len(reloaded["history"]) != len(history):
        raise AssertionError("Task 2A model reload check failed")

    report.update(
        {
            "selected": selected,
            "prediction_summary": submission.iloc[:, 1:].describe().to_dict(),
            "runtime_seconds": time.perf_counter() - started,
            "artifacts": {"model": str(model_path), "submission": str(submission_path)},
        }
    )
    report_path = WORK / "reports" / "task2a_model_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    append_experiments(experiment_rows)
    print(json.dumps(report, indent=2))
    print(f"WROTE {model_path}")
    print(f"WROTE {submission_path}")
    print(f"WROTE {report_path}")


if __name__ == "__main__":
    main()
