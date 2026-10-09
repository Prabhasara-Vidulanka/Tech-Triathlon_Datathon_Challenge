"""Leakage-safe Task 1 label construction with audited time semantics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
WORK = ROOT / "work"


def clock_minutes(values: pd.Series) -> pd.Series:
    """Convert HH:MM strings to numeric minutes after midnight."""
    text = values.astype("string")
    result = pd.to_numeric(text.str.slice(0, 2), errors="coerce") * 60
    result += pd.to_numeric(text.str.slice(3, 5), errors="coerce")
    return result.astype("Float64")


def _roll_forward(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Move a clock value to the next day only when it precedes its reference."""
    return values + 1440 * (values < reference).astype("int64")


def build_task1_labels() -> tuple[pd.DataFrame, dict]:
    orders = pd.read_csv(DATA / "Training Data" / "deliveries_train.csv")
    legs = pd.read_csv(DATA / "Training Data" / "route_legs_train.csv")
    outlets = pd.read_csv(DATA / "General Data" / "outlets.csv")

    dispatched = orders.loc[orders["dispatch_status"].isin(["attempted", "deferred"])].copy()
    assert not dispatched.duplicated("delivery_id").any()
    assert not legs.duplicated(["route_id", "seq"]).any()

    merged = dispatched.merge(
        legs,
        left_on=["route_id", "seq_in_route"],
        right_on=["route_id", "seq"],
        how="left",
        suffixes=("_order", "_leg"),
        validate="one_to_one",
        indicator=True,
    )
    assert merged["_merge"].eq("both").all()
    assert merged["outlet_id"].eq(merged["to_outlet"]).all()
    assert merged["vehicle_id_order"].eq(merged["vehicle_id_leg"]).all()
    assert merged["brand_order"].eq(merged["brand_leg"]).all()
    assert merged["district_order"].eq(merged["district_leg"]).all()
    assert merged["depot_order"].eq(merged["depot_leg"]).all()
    assert pd.to_datetime(merged["dispatch_date"]).eq(pd.to_datetime(merged["date"])).all()

    arrival = clock_minutes(merged["arrival_time"])
    leave = _roll_forward(clock_minutes(merged["leave_outlet_time"]), arrival)
    window_open = clock_minutes(merged["window_open_time"])
    window_close = _roll_forward(clock_minutes(merged["window_close_time"]), window_open)

    # No supplied row crosses midnight. These adjustments keep the label
    # definition safe if a future data refresh contains an overnight window.
    crosses_midnight = window_close.ge(1440)
    aligned_arrival = arrival + 1440 * (crosses_midnight & arrival.lt(window_open)).astype("int64")
    aligned_leave = leave + 1440 * (aligned_arrival.gt(leave)).astype("int64")

    service_start = np.maximum(aligned_arrival.astype(float), window_open.astype(float))
    service = aligned_leave.astype(float) - service_start
    late = aligned_arrival.astype(float) > window_close.astype(float)

    if service.isna().any():
        raise ValueError(f"Missing service labels: {int(service.isna().sum())}")
    if (service < 0).any():
        raise ValueError(f"Negative service labels: {int((service < 0).sum())}")

    labels = pd.DataFrame(
        {
            "delivery_id": merged["delivery_id"],
            "service_min": service,
            "late": late.astype("int8"),
            "actual_arrival_min": aligned_arrival.astype(float),
            "service_start_min": service_start,
            "actual_leave_min": aligned_leave.astype(float),
            "arrived_early": aligned_arrival.astype(float).lt(window_open.astype(float)).astype("int8"),
        }
    )

    with_outlets = merged.merge(
        outlets[["outlet_id", "dock_type", "parking_constraint", "mall_window"]],
        on="outlet_id",
        how="left",
        validate="many_to_one",
    )
    mall_rows = with_outlets["mall_window"].notna()
    encoded_window = (
        with_outlets["window_open_time"].astype(str)
        + "-"
        + with_outlets["window_close_time"].astype(str)
    )

    groups: dict[str, dict] = {}
    audit_frame = pd.DataFrame(
        {
            "brand": merged["brand_order"],
            "depot": merged["depot_order"],
            "district": merged["district_order"],
            "outlet_id": merged["outlet_id"],
            "dow": merged["dow"],
            "monsoon": merged["monsoon"],
            "seq": merged["seq"],
            "planned_margin_min": clock_minutes(merged["window_close_time"]).astype(float)
            - clock_minutes(merged["planned_arrival_time_order"]).astype(float),
            "late": late.astype(int),
            "service_min": service,
        }
    )
    for column in ["brand", "depot", "district", "dow", "monsoon"]:
        table = audit_frame.groupby(column, dropna=False).agg(
            rows=("late", "size"),
            late_rate=("late", "mean"),
            service_mean=("service_min", "mean"),
            service_median=("service_min", "median"),
        )
        groups[column] = {
            str(index): {key: float(value) for key, value in row.items()}
            for index, row in table.iterrows()
        }

    report = {
        "formula": {
            "service_start": "max(actual_arrival, window_open)",
            "service_min": "leave_outlet - service_start",
            "late": "actual_arrival > window_close (strict inequality)",
        },
        "rows": int(len(labels)),
        "early_arrivals": int(labels["arrived_early"].sum()),
        "early_arrival_rate": float(labels["arrived_early"].mean()),
        "window_cross_midnight_rows": int(crosses_midnight.sum()),
        "leave_before_arrival_raw_rows": int((clock_minutes(merged["leave_outlet_time"]) < arrival).sum()),
        "arrival_equal_close_rows": int((aligned_arrival.astype(float) == window_close.astype(float)).sum()),
        "late_rows": int(labels["late"].sum()),
        "late_rate": float(labels["late"].mean()),
        "service_summary": {
            str(key): float(value)
            for key, value in labels["service_min"].describe(
                percentiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
            ).items()
        },
        "service_over_180_min": int(labels["service_min"].gt(180).sum()),
        "mall_rows": int(mall_rows.sum()),
        "mall_window_matches_order_window": int((encoded_window[mall_rows] == with_outlets.loc[mall_rows, "mall_window"]).sum()),
        "mall_window_mismatches": int((encoded_window[mall_rows] != with_outlets.loc[mall_rows, "mall_window"]).sum()),
        "group_summaries": groups,
    }
    return labels, report


def main() -> None:
    labels, report = build_task1_labels()
    feature_path = WORK / "features" / "task1_labels.csv"
    report_path = WORK / "reports" / "task1_label_audit.json"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(feature_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"WROTE {feature_path}")
    print(f"WROTE {report_path}")


if __name__ == "__main__":
    main()
