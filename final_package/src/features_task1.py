"""Inference-available feature construction for Task 1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import find_project_root

from labels import clock_minutes


ROOT = find_project_root(__file__)
DATA = ROOT / "Data"

PROHIBITED_COLUMNS = {
    "actual_depart_time",
    "actual_travel_duration_min",
    "arrival_time",
    "leave_outlet_time",
    "actual_arrival_min",
    "service_start_min",
    "actual_leave_min",
    "service_min",
    "late",
}

CATEGORICAL_FEATURES = [
    "outlet_id",
    "brand",
    "district",
    "depot",
    "temp_requirement",
    "vehicle_id",
    "vehicle_type",
    "vehicle_temp",
    "dock_type",
    "parking_constraint",
    "road_class",
    "festival",
]

NUMERIC_FEATURES = [
    "order_units",
    "order_weight_kg",
    "order_volume_m3",
    "weight_per_unit",
    "volume_per_unit",
    "density_kg_m3",
    "log_order_units",
    "log_order_weight",
    "log_order_volume",
    "seq_in_route",
    "is_first_stop",
    "distance_km",
    "planned_travel_duration_min",
    "planned_speed_kmh",
    "route_stop_count",
    "route_total_distance_km",
    "route_total_planned_travel_min",
    "planned_depart_min",
    "planned_arrival_min",
    "window_open_min",
    "window_close_min",
    "window_width_min",
    "planned_after_open_min",
    "planned_margin_to_close_min",
    "planned_from_window_mid_min",
    "planned_arrival_sin",
    "planned_arrival_cos",
    "depart_hour",
    "year",
    "month",
    "day_of_year",
    "dow",
    "is_weekend",
    "iso_week",
    "is_payday",
    "festival_ramp",
    "is_holiday",
    "monsoon",
    "is_operating",
    "speed_index",
    "disruption_index",
    "free_flow_kmh",
    "depot_to_district_km",
    "depot_to_district_freeflow_min",
    "inter_stop_km",
    "inter_stop_freeflow_min",
    "traffic_delay_factor",
    "disruption_factor",
    "distance_congestion",
    "service_allowance_min",
    "order_volume_per_allowance",
    "vehicle_weight_cap_kg",
    "vehicle_volume_cap_m3",
    "vehicle_weight_load_ratio",
    "vehicle_volume_load_ratio",
    "km_per_l",
    "weekly_fuel_quota_l",
    "has_mall_window",
]


def _load_reference_tables() -> dict[str, pd.DataFrame]:
    general = DATA / "General Data"
    return {
        "outlets": pd.read_csv(general / "outlets.csv"),
        "vehicles": pd.read_csv(general / "vehicles.csv"),
        "calendar": pd.read_csv(general / "calendar.csv"),
        "district": pd.read_csv(general / "district_travel.csv"),
        "service": pd.read_csv(general / "service_allowance.csv"),
        "traffic": pd.read_csv(general / "traffic_speed.csv"),
        "roads": pd.read_csv(general / "road_conditions.csv"),
    }


def build_task1_feature_frame(split: str) -> pd.DataFrame:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    refs = _load_reference_tables()
    if split == "train":
        orders = pd.read_csv(DATA / "Training Data" / "deliveries_train.csv")
        orders = orders.loc[orders["dispatch_status"].isin(["attempted", "deferred"])].copy()
        legs = pd.read_csv(DATA / "Training Data" / "route_legs_train.csv")
    else:
        orders = pd.read_csv(DATA / "Test Data" / "task1_test_inputs.csv")
        legs = pd.read_csv(DATA / "Test Data" / "route_legs_test.csv")

    if legs.duplicated(["route_id", "seq"]).any():
        raise ValueError("Duplicate route_id/seq keys in route legs")
    route_stats = legs.groupby("route_id", as_index=False).agg(
        route_stop_count=("seq", "size"),
        route_total_distance_km=("distance_km", "sum"),
        route_total_planned_travel_min=("planned_travel_duration_min", "sum"),
    )
    plan_columns = [
        "route_id",
        "seq",
        "date",
        "from_point",
        "to_outlet",
        "distance_km",
        "planned_depart_time",
        "planned_travel_duration_min",
        "planned_arrival_time",
    ]
    frame = orders.merge(
        legs[plan_columns],
        left_on=["route_id", "seq_in_route"],
        right_on=["route_id", "seq"],
        suffixes=("_order", "_leg"),
        validate="one_to_one",
    )
    if len(frame) != len(orders) or not frame["outlet_id"].eq(frame["to_outlet"]).all():
        raise ValueError("Order-to-leg relationship failed")
    frame = frame.merge(route_stats, on="route_id", how="left", validate="many_to_one")
    frame = frame.merge(refs["outlets"], on="outlet_id", how="left", suffixes=("", "_ref"), validate="many_to_one")

    vehicle_columns = refs["vehicles"].rename(
        columns={
            "type": "vehicle_type_ref",
            "temp": "vehicle_temp_ref",
            "depot": "vehicle_depot_ref",
            "weight_cap_kg": "vehicle_weight_cap_kg",
            "volume_cap_m3": "vehicle_volume_cap_m3",
        }
    )
    frame = frame.merge(vehicle_columns, on="vehicle_id", how="left", validate="many_to_one")

    date = pd.to_datetime(frame["dispatch_date"])
    frame["date_key"] = date.dt.strftime("%Y-%m-%d")
    calendar = refs["calendar"].rename(columns={"date": "date_key", "dow": "calendar_dow", "monsoon": "calendar_monsoon"})
    frame = frame.merge(calendar, on="date_key", how="left", validate="many_to_one")
    district = refs["district"].drop(columns=["depot"]).copy()
    frame = frame.merge(district, on="district", how="left", validate="many_to_one")
    frame = frame.merge(refs["service"], on=["brand", "dock_type"], how="left", validate="many_to_one")
    roads = refs["roads"].rename(columns={"date": "date_key"})
    frame = frame.merge(roads, on=["district", "date_key"], how="left", validate="many_to_one")

    frame["planned_depart_min"] = clock_minutes(frame["planned_depart_time"]).astype(float)
    frame["planned_arrival_min"] = clock_minutes(frame["planned_arrival_time_order"]).astype(float)
    frame["window_open_min"] = clock_minutes(frame["window_open_time"]).astype(float)
    frame["window_close_min"] = clock_minutes(frame["window_close_time"]).astype(float)
    crosses = frame["window_close_min"].lt(frame["window_open_min"])
    frame.loc[crosses, "window_close_min"] += 1440
    frame["depart_hour"] = (frame["planned_depart_min"] // 60).astype("int64")
    traffic = refs["traffic"].rename(columns={"monsoon": "calendar_monsoon"})
    frame = frame.merge(
        traffic,
        left_on=["district", "depart_hour", "calendar_monsoon"],
        right_on=["district", "hour", "calendar_monsoon"],
        how="left",
        validate="many_to_one",
    )

    units = frame["order_units"].replace(0, np.nan)
    volume = frame["order_volume_m3"].replace(0, np.nan)
    frame["weight_per_unit"] = frame["order_weight_kg"] / units
    frame["volume_per_unit"] = frame["order_volume_m3"] / units
    frame["density_kg_m3"] = frame["order_weight_kg"] / volume
    frame["log_order_units"] = np.log1p(frame["order_units"])
    frame["log_order_weight"] = np.log1p(frame["order_weight_kg"])
    frame["log_order_volume"] = np.log1p(frame["order_volume_m3"])
    frame["is_first_stop"] = frame["seq_in_route"].eq(0).astype("int8")
    frame["planned_speed_kmh"] = frame["distance_km"] / frame["planned_travel_duration_min"].replace(0, np.nan) * 60
    frame["window_width_min"] = frame["window_close_min"] - frame["window_open_min"]
    frame["planned_after_open_min"] = frame["planned_arrival_min"] - frame["window_open_min"]
    frame["planned_margin_to_close_min"] = frame["window_close_min"] - frame["planned_arrival_min"]
    frame["planned_from_window_mid_min"] = frame["planned_arrival_min"] - (
        frame["window_open_min"] + frame["window_close_min"]
    ) / 2
    angle = 2 * np.pi * frame["planned_arrival_min"] / 1440
    frame["planned_arrival_sin"] = np.sin(angle)
    frame["planned_arrival_cos"] = np.cos(angle)
    frame["year"] = date.dt.year
    frame["month"] = date.dt.month
    frame["day_of_year"] = date.dt.dayofyear
    frame["dow"] = frame["calendar_dow"]
    frame["monsoon"] = frame["calendar_monsoon"]
    frame["traffic_delay_factor"] = 100 / frame["speed_index"].replace(0, np.nan)
    frame["disruption_factor"] = 100 / frame["disruption_index"].replace(0, np.nan)
    frame["distance_congestion"] = frame["distance_km"] * frame["traffic_delay_factor"] * frame["disruption_factor"]
    frame["order_volume_per_allowance"] = frame["order_volume_m3"] / frame["service_allowance_min"].replace(0, np.nan)
    frame["vehicle_weight_load_ratio"] = frame["order_weight_kg"] / frame["vehicle_weight_cap_kg"].replace(0, np.nan)
    frame["vehicle_volume_load_ratio"] = frame["order_volume_m3"] / frame["vehicle_volume_cap_m3"].replace(0, np.nan)
    frame["has_mall_window"] = frame["mall_window"].notna().astype("int8")
    frame["festival"] = frame["festival"].fillna("none")

    result = frame[["delivery_id", "dispatch_date", *CATEGORICAL_FEATURES, *NUMERIC_FEATURES]].copy()
    result[CATEGORICAL_FEATURES] = result[CATEGORICAL_FEATURES].fillna("missing").astype(str)
    if PROHIBITED_COLUMNS.intersection(result.columns):
        raise AssertionError("Training-only actual outcome leaked into features")
    if result[NUMERIC_FEATURES].isna().any().any():
        missing = result[NUMERIC_FEATURES].isna().sum()
        raise ValueError(f"Missing numeric features: {missing[missing.gt(0)].to_dict()}")
    return result


if __name__ == "__main__":
    train = build_task1_feature_frame("train")
    test = build_task1_feature_frame("test")
    print("train", train.shape, train["dispatch_date"].min(), train["dispatch_date"].max())
    print("test", test.shape, test["dispatch_date"].min(), test["dispatch_date"].max())
    print("columns", list(train.columns))
