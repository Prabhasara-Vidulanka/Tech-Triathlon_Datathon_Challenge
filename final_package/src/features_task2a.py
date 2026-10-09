"""Weekly panel construction and leakage-safe lag features for Task 2A."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import find_project_root


ROOT = find_project_root(__file__)
DATA = ROOT / "Data"

LAGS = [1, 2, 3, 4, 5, 6, 8, 13, 26, 52]
ROLLING_WINDOWS = [4, 8, 13, 26, 52]

CALENDAR_FEATURES = [
    "operating_days",
    "payday_days",
    "holiday_days",
    "weekend_days",
    "festival_days",
    "festival_ramp_max",
    "festival_ramp_mean",
    "festival_ramp_sum",
    "monsoon_share",
]


def weekly_calendar() -> pd.DataFrame:
    calendar = pd.read_csv(DATA / "General Data" / "calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["festival_present"] = calendar["festival"].notna().astype(int)
    weekly = calendar.groupby(["iso_year", "iso_week"], as_index=False).agg(
        week_start=("date", "min"),
        week_end=("date", "max"),
        operating_days=("is_operating", "sum"),
        payday_days=("is_payday", "sum"),
        holiday_days=("is_holiday", "sum"),
        weekend_days=("is_weekend", "sum"),
        festival_days=("festival_present", "sum"),
        festival_ramp_max=("festival_ramp", "max"),
        festival_ramp_mean=("festival_ramp", "mean"),
        festival_ramp_sum=("festival_ramp", "sum"),
        monsoon_share=("monsoon", "mean"),
    )
    return weekly


def build_weekly_history() -> pd.DataFrame:
    train = pd.read_csv(DATA / "Training Data" / "deliveries_train.csv")
    task1 = pd.read_csv(DATA / "Test Data" / "task1_test_inputs.csv")
    orders = pd.concat([train, task1], ignore_index=True)
    if orders["delivery_id"].duplicated().any():
        raise ValueError("Duplicate delivery_id after combining Task 2A history")
    calendar = pd.read_csv(DATA / "General Data" / "calendar.csv")[["date", "iso_year", "iso_week"]]
    orders = orders.merge(calendar, left_on="order_date", right_on="date", how="left", validate="many_to_one")
    if orders[["iso_year", "iso_week"]].isna().any().any():
        raise ValueError("Some order dates are absent from calendar.csv")
    orders["chilled_volume_m3"] = np.where(
        orders["brand"].eq("Fresh") & orders["temp_requirement"].eq("chilled"),
        orders["order_volume_m3"],
        0.0,
    )
    observed = orders.groupby(["depot", "brand", "iso_year", "iso_week"], as_index=False).agg(
        total_volume_m3=("order_volume_m3", "sum"),
        chilled_volume_m3=("chilled_volume_m3", "sum"),
        order_count=("delivery_id", "size"),
    )
    weekly = weekly_calendar()
    observed = observed.merge(weekly, on=["iso_year", "iso_week"], how="left", validate="many_to_one")
    series = orders[["depot", "brand"]].drop_duplicates().sort_values(["depot", "brand"])
    weeks = weekly.loc[
        weekly["week_start"].between(observed["week_start"].min(), observed["week_start"].max())
    ]
    grid = series.merge(weeks, how="cross")
    history = grid.merge(
        observed[["depot", "brand", "iso_year", "iso_week", "total_volume_m3", "chilled_volume_m3", "order_count"]],
        on=["depot", "brand", "iso_year", "iso_week"],
        how="left",
        validate="one_to_one",
    )
    history[["total_volume_m3", "chilled_volume_m3", "order_count"]] = history[
        ["total_volume_m3", "chilled_volume_m3", "order_count"]
    ].fillna(0.0)
    if (history.loc[~history["brand"].eq("Fresh"), "chilled_volume_m3"] != 0).any():
        raise AssertionError("Non-Fresh chilled demand must be zero")
    return history.sort_values(["week_start", "depot", "brand"]).reset_index(drop=True)


def forecast_rows() -> pd.DataFrame:
    requested = pd.read_csv(DATA / "Test Data" / "task2a_test_inputs.csv")
    weekly = weekly_calendar()
    result = requested.merge(weekly, on=["iso_year", "iso_week"], how="left", validate="many_to_one")
    if result["week_start"].isna().any():
        raise ValueError("Forecast periods absent from calendar.csv")
    return result


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    week_start = pd.to_datetime(result["week_start"])
    result["year"] = week_start.dt.year
    result["week_of_year"] = result["iso_week"].astype(int)
    result["week_sin"] = np.sin(2 * np.pi * result["week_of_year"] / 52.1775)
    result["week_cos"] = np.cos(2 * np.pi * result["week_of_year"] / 52.1775)
    result["time_index"] = ((week_start - pd.Timestamp("2024-01-01")).dt.days / 7).astype(int)
    result["brand_fresh"] = result["brand"].eq("Fresh").astype(int)
    result["brand_style"] = result["brand"].eq("Style").astype(int)
    result["brand_tech"] = result["brand"].eq("Tech").astype(int)
    return result


def supervised_features(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    result = add_time_features(panel)
    result = result.sort_values(["depot", "brand", "week_start"]).copy()
    grouped = result.groupby(["depot", "brand"], sort=False)[target]
    for lag in LAGS:
        result[f"lag_{lag}"] = grouped.shift(lag)
    for window in ROLLING_WINDOWS:
        result[f"rolling_mean_{window}"] = grouped.transform(lambda values: values.shift(1).rolling(window, min_periods=1).mean())
        result[f"rolling_std_{window}"] = grouped.transform(lambda values: values.shift(1).rolling(window, min_periods=2).std())
    result["recent_trend_4"] = result["lag_1"] - result["lag_4"]
    result["recent_trend_13"] = result["lag_1"] - result["lag_13"]
    return result.sort_values(["week_start", "depot", "brand"]).reset_index(drop=True)


MODEL_CATEGORICAL = ["depot", "brand"]
MODEL_NUMERIC = [
    "year",
    "week_of_year",
    "week_sin",
    "week_cos",
    "time_index",
    "brand_fresh",
    "brand_style",
    "brand_tech",
    *CALENDAR_FEATURES,
    *[f"lag_{lag}" for lag in LAGS],
    *[name for window in ROLLING_WINDOWS for name in (f"rolling_mean_{window}", f"rolling_std_{window}")],
    "recent_trend_4",
    "recent_trend_13",
]


if __name__ == "__main__":
    history = build_weekly_history()
    future = forecast_rows()
    print("history", history.shape, history["week_start"].min(), history["week_start"].max())
    print("series", history.groupby(["depot", "brand"]).size().to_dict())
    print("future", future.shape, future["week_start"].min(), future["week_start"].max())
    print("missing weeks", int(history["order_count"].eq(0).sum()))
