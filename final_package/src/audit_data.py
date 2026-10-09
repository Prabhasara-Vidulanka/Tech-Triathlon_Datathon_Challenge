"""Reproducible inventory and integrity audit for the Tech-Triathlon data."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
WORK = ROOT / "work"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_csvs() -> list[Path]:
    return sorted(DATA.rglob("*.csv"), key=lambda p: str(p).lower())


def file_summary(path: Path) -> dict:
    frame = pd.read_csv(path)
    result = {
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing": {column: int(value) for column, value in frame.isna().sum().items() if value},
        "duplicate_rows": int(frame.duplicated().sum()),
        "sha256": sha256(path),
    }
    for column in frame.columns:
        if "date" in column.lower() or column in {"week_start"}:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().any():
                result.setdefault("date_ranges", {})[column] = {
                    "min": str(parsed.min().date()),
                    "max": str(parsed.max().date()),
                    "unparsed_nonmissing": int((frame[column].notna() & parsed.isna()).sum()),
                }
    for column in [
        "delivery_id",
        "leg_id",
        "row_id",
        "vehicle_id",
        "outlet_id",
        "order_ref",
    ]:
        if column in frame:
            result.setdefault("keys", {})[column] = {
                "unique": int(frame[column].nunique(dropna=True)),
                "duplicates": int(frame[column].duplicated(keep=False).sum()),
            }
    return result


def join_audit() -> dict:
    deliveries = pd.read_csv(DATA / "Training Data" / "deliveries_train.csv")
    legs = pd.read_csv(DATA / "Training Data" / "route_legs_train.csv")
    attempted = deliveries.loc[deliveries["dispatch_status"].eq("attempted")].copy()
    deferred = deliveries.loc[deliveries["dispatch_status"].eq("deferred")].copy()
    dispatched = deliveries.loc[deliveries["dispatch_status"].isin(["attempted", "deferred"])].copy()
    leg_key_dupes = int(legs.duplicated(["route_id", "seq"], keep=False).sum())
    merged = dispatched.merge(
        legs,
        left_on=["route_id", "seq_in_route"],
        right_on=["route_id", "seq"],
        how="left",
        suffixes=("_order", "_leg"),
        validate="one_to_one",
        indicator=True,
    )
    comparisons = {
        "outlet": (merged["outlet_id"] == merged["to_outlet"]),
        "vehicle": (merged["vehicle_id_order"] == merged["vehicle_id_leg"]),
        "brand": (merged["brand_order"] == merged["brand_leg"]),
        "district": (merged["district_order"] == merged["district_leg"]),
        "depot": (merged["depot_order"] == merged["depot_leg"]),
        "dispatch_date_vs_leg_date": (
            pd.to_datetime(merged["dispatch_date"]) == pd.to_datetime(merged["date"])
        ),
    }
    test_orders = pd.read_csv(DATA / "Test Data" / "task1_test_inputs.csv")
    test_legs = pd.read_csv(DATA / "Test Data" / "route_legs_test.csv")
    test_merged = test_orders.merge(
        test_legs,
        left_on=["route_id", "seq_in_route"],
        right_on=["route_id", "seq"],
        how="left",
        suffixes=("_order", "_leg"),
        validate="one_to_one",
        indicator=True,
    )
    return {
        "status_counts": {str(k): int(v) for k, v in deliveries["dispatch_status"].value_counts().items()},
        "attempted_rows": int(len(attempted)),
        "deferred_rows": int(len(deferred)),
        "dispatched_rows": int(len(dispatched)),
        "train_leg_key_duplicate_rows": leg_key_dupes,
        "train_dispatched_unmatched": int(merged["_merge"].ne("both").sum()),
        "train_consistency_mismatches": {
            name: int((~series.fillna(False)).sum()) for name, series in comparisons.items()
        },
        "test_order_rows": int(len(test_orders)),
        "test_leg_rows": int(len(test_legs)),
        "test_leg_key_duplicate_rows": int(test_legs.duplicated(["route_id", "seq"], keep=False).sum()),
        "test_unmatched": int(test_merged["_merge"].ne("both").sum()),
        "test_outlet_mismatches": int((test_merged["outlet_id"] != test_merged["to_outlet"]).sum()),
    }


def distribution_audit() -> dict:
    train = pd.read_csv(DATA / "Training Data" / "deliveries_train.csv")
    test = pd.read_csv(DATA / "Test Data" / "task1_test_inputs.csv")
    columns = [
        "brand",
        "district",
        "depot",
        "temp_requirement",
        "vehicle_type",
        "vehicle_temp",
        "outlet_id",
        "vehicle_id",
    ]
    result = {}
    for column in columns:
        train_values = set(train[column].dropna().astype(str))
        test_values = set(test[column].dropna().astype(str))
        result[column] = {
            "train_unique": len(train_values),
            "test_unique": len(test_values),
            "unseen_in_test": sorted(test_values - train_values),
            "missing_from_test": sorted(train_values - test_values),
        }
    numeric = ["order_units", "order_weight_kg", "order_volume_m3", "seq_in_route"]
    result["numeric"] = {}
    for column in numeric:
        result["numeric"][column] = {
            "train_mean": float(train[column].mean()),
            "test_mean": float(test[column].mean()),
            "train_p99": float(train[column].quantile(0.99)),
            "test_p99": float(test[column].quantile(0.99)),
        }
    return result


def template_audit() -> dict:
    task1 = pd.read_csv(DATA / "Submission Templates" / "submission_task1.csv")
    task1_input = pd.read_csv(DATA / "Test Data" / "task1_test_inputs.csv")
    task2a = pd.read_csv(DATA / "Submission Templates" / "submission_task2a.csv")
    task2a_input = pd.read_csv(DATA / "Test Data" / "task2a_test_inputs.csv")
    task2b = pd.read_csv(DATA / "Submission Templates" / "submission_task2b.csv")
    task2b_input = pd.read_csv(DATA / "Test Data" / "task2b_peak_day_scenarios.csv")
    return {
        "task1_columns": list(task1.columns),
        "task1_row_count_match": len(task1) == len(task1_input),
        "task1_order_match": task1["delivery_id"].equals(task1_input["delivery_id"]),
        "task2a_columns": list(task2a.columns),
        "task2a_row_count_match": len(task2a) == len(task2a_input),
        "task2a_order_match": task2a["row_id"].equals(task2a_input["row_id"]),
        "task2b_columns": list(task2b.columns),
        "task2b_row_count_match": len(task2b) == len(task2b_input),
        "task2b_order_match": task2b[["scenario", "order_ref", "outlet_id"]].equals(
            task2b_input[["scenario", "order_ref", "outlet_id"]]
        ),
    }


def main() -> None:
    summaries = [file_summary(path) for path in find_csvs()]
    report = {
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "files": summaries,
        "joins": join_audit(),
        "distribution": distribution_audit(),
        "templates": template_audit(),
    }
    target = WORK / "reports" / "initial_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["runtime"], indent=2))
    for item in summaries:
        print(f"{item['path']}: {item['rows']} rows x {len(item['columns'])} cols")
        print(f"  columns={item['columns']}")
        if item.get("date_ranges"):
            print(f"  date_ranges={item['date_ranges']}")
        if item["missing"]:
            print(f"  missing={item['missing']}")
    print("JOINS", json.dumps(report["joins"], indent=2))
    print("DISTRIBUTION", json.dumps(report["distribution"], indent=2))
    print("TEMPLATES", json.dumps(report["templates"], indent=2))
    print(f"WROTE {target}")


if __name__ == "__main__":
    main()
