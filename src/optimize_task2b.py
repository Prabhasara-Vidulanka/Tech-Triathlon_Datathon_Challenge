"""Exact SciPy MILP allocation for the Task 2B peak-day scenario."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
WORK = ROOT / "work"


@dataclass(frozen=True)
class Policy:
    name: str
    deferred: float
    days: float
    base: float
    fresh: float
    chilled: float
    volume: float


POLICIES = [
    Policy("fairness_first", 100_000_000, 100_000, 1_000, 5_000, 5_000, 1),
    Policy("cold_chain", 100_000_000, 100_000, 1_000, 8_000, 12_000, 1),
    Policy("coverage", 100_000_000, 100_000, 8_000, 1_000, 1_000, 0.1),
    Policy("max_orders", 0, 0, 1_000_000, 0, 0, 1),
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    scenarios = pd.read_csv(DATA / "Test Data" / "task2b_peak_day_scenarios.csv")
    fleet = pd.read_csv(DATA / "Test Data" / "task2b_peak_day_fleet.csv")
    vehicles = pd.read_csv(DATA / "General Data" / "vehicles.csv")
    available = fleet.loc[fleet["status"].eq("available")].merge(vehicles, on="vehicle_id", validate="one_to_one")
    travel = pd.read_csv(DATA / "General Data" / "district_travel.csv").set_index("district").to_dict("index")
    allowance_frame = pd.read_csv(DATA / "General Data" / "service_allowance.csv")
    allowance = {(row.brand, row.dock_type): float(row.service_allowance_min) for row in allowance_frame.itertuples()}
    return scenarios.reset_index(drop=True), fleet, available.reset_index(drop=True), travel, allowance


def compatible(order: pd.Series, vehicle: pd.Series) -> bool:
    if order["depot"] != vehicle["depot"]:
        return False
    if order["temp_requirement"] == "chilled" and vehicle["temp"] != "reefer":
        return False
    if order["parking_constraint"] == "van_only" and vehicle["type"] != "van":
        return False
    if order["order_volume_m3"] > vehicle["volume_cap_m3"] + 1e-9:
        return False
    if order["order_weight_kg"] > vehicle["weight_cap_kg"] + 1e-9:
        return False
    return True


def priority_score(order: pd.Series, policy: Policy) -> float:
    return (
        policy.deferred * float(order["deferred_yesterday"])
        + policy.days * float(order["days_since_last_served"])
        + policy.base
        + policy.fresh * float(order["brand"] == "Fresh")
        + policy.chilled * float(order["temp_requirement"] == "chilled")
        + policy.volume * float(order["order_volume_m3"])
    )


def trip_minutes(group: pd.DataFrame, travel: dict, allowance: dict) -> float:
    if group.empty:
        return 0.0
    brand = group["brand"].iloc[0]
    district = group["district"].iloc[0]
    return (
        float(travel[district]["depot_to_district_freeflow_min"])
        + (len(group) - 1) * float(travel[district]["inter_stop_freeflow_min"])
        + sum(allowance[(brand, dock)] for dock in group["dock_type"])
    )


def assign_ambient_orders(
    orders: pd.DataFrame,
    vehicles: pd.DataFrame,
    allocation: pd.DataFrame,
    travel: dict,
    allowance: dict,
) -> pd.DataFrame:
    """Pack the non-scarce ambient workload without consuming reefer capacity."""
    ambient = orders.loc[orders["temp_requirement"].eq("ambient")].copy()
    ambient_vehicles = vehicles.loc[vehicles["temp"].eq("ambient")].copy().reset_index(drop=True)
    state = {
        row.vehicle_id: {"trips": 0, "fresh_minutes": 0.0, "daytime_minutes": 0.0}
        for row in ambient_vehicles.itertuples()
    }
    group_keys = ambient[["brand", "district", "parking_constraint"]].drop_duplicates()
    group_keys["van_first"] = group_keys["parking_constraint"].eq("van_only").astype(int)
    volume_by_group = ambient.groupby(["brand", "district", "parking_constraint"])["order_volume_m3"].sum()
    group_keys["volume"] = pd.MultiIndex.from_frame(
        group_keys[["brand", "district", "parking_constraint"]]
    ).map(volume_by_group)
    group_keys = group_keys.sort_values(["van_first", "volume"], ascending=[False, False])

    for key in group_keys.itertuples(index=False):
        subset = ambient.loc[
            ambient["brand"].eq(key.brand)
            & ambient["district"].eq(key.district)
            & ambient["parking_constraint"].eq(key.parking_constraint)
        ].copy()
        subset["size_rank"] = np.maximum(
            subset["order_volume_m3"] / ambient_vehicles["volume_cap_m3"].max(),
            subset["order_weight_kg"] / ambient_vehicles["weight_cap_kg"].max(),
        )
        remaining = list(subset.sort_values(["size_rank", "order_volume_m3"], ascending=False).index)
        while remaining:
            candidates = ambient_vehicles.copy()
            if key.parking_constraint == "van_only":
                candidates = candidates.loc[candidates["type"].eq("van")]
            else:
                candidates = candidates.loc[candidates["type"].eq("truck")]
            candidates = candidates.loc[candidates["vehicle_id"].map(lambda value: state[value]["trips"] < 2)]
            candidates = candidates.sort_values(["volume_cap_m3", "weight_cap_kg"], ascending=False)
            chosen_vehicle = None
            chosen_orders: list[int] = []
            for vehicle in candidates.itertuples():
                packed: list[int] = []
                for order_index in remaining:
                    trial_indices = packed + [order_index]
                    trial = orders.loc[trial_indices]
                    if trial["order_volume_m3"].sum() > vehicle.volume_cap_m3 + 1e-9:
                        continue
                    if trial["order_weight_kg"].sum() > vehicle.weight_cap_kg + 1e-9:
                        continue
                    minutes = trip_minutes(trial, travel, allowance)
                    budget_used = state[vehicle.vehicle_id]["fresh_minutes" if key.brand == "Fresh" else "daytime_minutes"]
                    budget = 270.0 if key.brand == "Fresh" else 480.0
                    if budget_used + minutes <= budget + 1e-9:
                        packed.append(order_index)
                if packed:
                    chosen_vehicle = vehicle
                    chosen_orders = packed
                    break
            if chosen_vehicle is None:
                # Some whole orders are larger than every compatible vehicle.
                # They are structurally infeasible and must remain deferred.
                impossible = []
                for order_index in remaining:
                    order = orders.loc[order_index]
                    compatible_rows = ambient_vehicles.loc[
                        ambient_vehicles["type"].eq("van")
                        if key.parking_constraint == "van_only"
                        else ambient_vehicles["type"].eq("truck")
                    ]
                    fits = (
                        compatible_rows["volume_cap_m3"].ge(order["order_volume_m3"])
                        & compatible_rows["weight_cap_kg"].ge(order["order_weight_kg"])
                    ).any()
                    if not fits:
                        impossible.append(order_index)
                if impossible:
                    remaining = [order_index for order_index in remaining if order_index not in impossible]
                    continue
                raise RuntimeError(f"Ambient packing failed for {key.brand}/{key.district}/{key.parking_constraint}")
            vehicle_id = chosen_vehicle.vehicle_id
            state[vehicle_id]["trips"] += 1
            trip_id = state[vehicle_id]["trips"]
            minutes = trip_minutes(orders.loc[chosen_orders], travel, allowance)
            budget_key = "fresh_minutes" if key.brand == "Fresh" else "daytime_minutes"
            state[vehicle_id][budget_key] += minutes
            for order_index in chosen_orders:
                allocation.at[order_index, "decision"] = "served"
                allocation.at[order_index, "vehicle_id"] = vehicle_id
                allocation.at[order_index, "trip_id"] = trip_id
            remaining = [order_index for order_index in remaining if order_index not in chosen_orders]
    return allocation


def solve(policy: Policy) -> tuple[pd.DataFrame, dict]:
    all_orders, _, all_vehicles, travel, allowance = load_data()
    orders = all_orders.loc[all_orders["temp_requirement"].eq("chilled")].copy()
    orders["source_index"] = orders.index
    orders = orders.reset_index(drop=True)
    vehicles = all_vehicles.loc[all_vehicles["temp"].eq("reefer")].reset_index(drop=True)
    slots = [(v, trip) for v in range(len(vehicles)) for trip in (1, 2)]
    groups = sorted({(row.brand, row.district) for row in orders.itertuples()})

    x_keys: list[tuple[int, int]] = []
    for order_index, order in orders.iterrows():
        for slot_index, (vehicle_index, _) in enumerate(slots):
            if compatible(order, vehicles.iloc[vehicle_index]):
                x_keys.append((order_index, slot_index))
    x_index = {key: index for index, key in enumerate(x_keys)}

    feasible_by_slot_group: dict[tuple[int, tuple[str, str]], list[int]] = {}
    for order_index, slot_index in x_keys:
        group = (orders.at[order_index, "brand"], orders.at[order_index, "district"])
        feasible_by_slot_group.setdefault((slot_index, group), []).append(order_index)
    y_keys = sorted(feasible_by_slot_group, key=lambda key: (key[0], key[1][0], key[1][1]))
    y_index = {key: len(x_keys) + index for index, key in enumerate(y_keys)}
    n_variables = len(x_keys) + len(y_keys)

    objective = np.zeros(n_variables)
    for key, index in x_index.items():
        order_index, slot_index = key
        vehicle_index, _ = slots[slot_index]
        order = orders.iloc[order_index]
        vehicle = vehicles.iloc[vehicle_index]
        scarcity_penalty = 0.0
        if vehicle["temp"] == "reefer" and order["temp_requirement"] == "ambient":
            scarcity_penalty += 10.0
        if vehicle["type"] == "van" and order["parking_constraint"] != "van_only":
            scarcity_penalty += 10.0
        objective[index] = -priority_score(order, policy) + scarcity_penalty
    for index in y_index.values():
        objective[index] = 0.01

    constraints: list[tuple[dict[int, float], float, float]] = []

    def add(coefficients: dict[int, float], lower: float = -np.inf, upper: float = np.inf) -> None:
        constraints.append((coefficients, lower, upper))

    # Whole order: an order is assigned to at most one trip.
    for order_index in range(len(orders)):
        coefficients = {index: 1.0 for (candidate, _), index in x_index.items() if candidate == order_index}
        add(coefficients, 0.0, 1.0)

    # One brand/district group per trip and no empty active groups.
    for slot_index in range(len(slots)):
        add({index: 1.0 for (slot, _), index in y_index.items() if slot == slot_index}, 0.0, 1.0)
    for key, y_variable in y_index.items():
        slot_index, group = key
        order_indices = feasible_by_slot_group[key]
        add({y_variable: 1.0, **{x_index[(i, slot_index)]: -1.0 for i in order_indices}}, -np.inf, 0.0)
        for order_index in order_indices:
            add({x_index[(order_index, slot_index)]: 1.0, y_variable: -1.0}, -np.inf, 0.0)

    # Capacity per trip.
    for slot_index, (vehicle_index, _) in enumerate(slots):
        vehicle = vehicles.iloc[vehicle_index]
        volume = {
            index: float(orders.at[order_index, "order_volume_m3"])
            for (order_index, slot), index in x_index.items()
            if slot == slot_index
        }
        weight = {
            index: float(orders.at[order_index, "order_weight_kg"])
            for (order_index, slot), index in x_index.items()
            if slot == slot_index
        }
        add(volume, 0.0, float(vehicle["volume_cap_m3"]))
        add(weight, 0.0, float(vehicle["weight_cap_kg"]))

    # Use trip 1 before trip 2 to remove equivalent slot permutations.
    for vehicle_index in range(len(vehicles)):
        slot_one = slots.index((vehicle_index, 1))
        slot_two = slots.index((vehicle_index, 2))
        coefficients = {
            **{index: 1.0 for (slot, _), index in y_index.items() if slot == slot_two},
            **{index: -1.0 for (slot, _), index in y_index.items() if slot == slot_one},
        }
        add(coefficients, -np.inf, 0.0)

    # Exact published trip-time formula, accumulated into each vehicle's
    # separate Fresh and daytime budgets.
    for vehicle_index in range(len(vehicles)):
        vehicle_slots = [index for index, (v, _) in enumerate(slots) if v == vehicle_index]
        for budget_name, brand_predicate, budget in [
            ("fresh", lambda brand: brand == "Fresh", 270.0),
            ("daytime", lambda brand: brand != "Fresh", 480.0),
        ]:
            coefficients: dict[int, float] = {}
            for slot_index in vehicle_slots:
                for group in groups:
                    brand, district = group
                    if not brand_predicate(brand) or (slot_index, group) not in y_index:
                        continue
                    inter = float(travel[district]["inter_stop_freeflow_min"])
                    outbound = float(travel[district]["depot_to_district_freeflow_min"])
                    y_variable = y_index[(slot_index, group)]
                    coefficients[y_variable] = coefficients.get(y_variable, 0.0) + outbound - inter
                    for order_index in feasible_by_slot_group[(slot_index, group)]:
                        x_variable = x_index[(order_index, slot_index)]
                        handle = allowance[(brand, orders.at[order_index, "dock_type"])]
                        coefficients[x_variable] = coefficients.get(x_variable, 0.0) + inter + handle
            add(coefficients, 0.0, budget)

    matrix = lil_matrix((len(constraints), n_variables), dtype=float)
    lower = np.empty(len(constraints))
    upper = np.empty(len(constraints))
    for row_index, (coefficients, low, high) in enumerate(constraints):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
        lower[row_index] = low
        upper[row_index] = high

    result = milp(
        c=objective,
        integrality=np.ones(n_variables),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 120, "mip_rel_gap": 0.0, "presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"MILP failed for {policy.name}: {result.message}")

    allocation = all_orders[["scenario", "order_ref", "outlet_id"]].copy()
    allocation["decision"] = "deferred"
    allocation["vehicle_id"] = pd.NA
    allocation["trip_id"] = pd.NA
    for (order_index, slot_index), variable_index in x_index.items():
        if result.x[variable_index] > 0.5:
            vehicle_index, trip_id = slots[slot_index]
            source_index = int(orders.at[order_index, "source_index"])
            allocation.at[source_index, "decision"] = "served"
            allocation.at[source_index, "vehicle_id"] = vehicles.at[vehicle_index, "vehicle_id"]
            allocation.at[source_index, "trip_id"] = trip_id

    allocation = assign_ambient_orders(all_orders, all_vehicles, allocation, travel, allowance)
    merged = all_orders.merge(allocation, on=["scenario", "order_ref", "outlet_id"], validate="one_to_one")
    served = merged.loc[merged["decision"].eq("served")]
    summary = {
        "policy": policy.name,
        "solver_status": int(result.status),
        "solver_message": result.message,
        "objective_minimized": float(result.fun),
        "served_orders": int(len(served)),
        "deferred_orders": int(len(merged) - len(served)),
        "served_volume_m3": float(served["order_volume_m3"].sum()),
        "served_weight_kg": float(served["order_weight_kg"].sum()),
        "deferred_yesterday_served": int(served["deferred_yesterday"].sum()),
        "deferred_yesterday_total": int(all_orders["deferred_yesterday"].sum()),
        "days_since_last_served_sum": int(served["days_since_last_served"].sum()),
        "fresh_orders_served": int(served["brand"].eq("Fresh").sum()),
        "chilled_orders_served": int(served["temp_requirement"].eq("chilled").sum()),
        "van_only_served": int(served["parking_constraint"].eq("van_only").sum()),
        "variables": n_variables,
        "constraints": len(constraints),
    }
    return allocation, summary


def trip_analysis(allocation: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    orders, _, vehicles, travel, allowance = load_data()
    vehicle_lookup = vehicles.set_index("vehicle_id")
    merged = orders.merge(allocation, on=["scenario", "order_ref", "outlet_id"], validate="one_to_one")
    served = merged.loc[merged["decision"].eq("served")].copy()
    trips = []
    for (vehicle_id, trip_id), group in served.groupby(["vehicle_id", "trip_id"]):
        vehicle = vehicle_lookup.loc[vehicle_id]
        brand = group["brand"].iloc[0]
        district = group["district"].iloc[0]
        n = len(group)
        minutes = (
            float(travel[district]["depot_to_district_freeflow_min"])
            + (n - 1) * float(travel[district]["inter_stop_freeflow_min"])
            + sum(allowance[(brand, dock)] for dock in group["dock_type"])
        )
        trips.append(
            {
                "vehicle_id": vehicle_id,
                "trip_id": int(trip_id),
                "vehicle_type": vehicle["type"],
                "vehicle_temp": vehicle["temp"],
                "brand": brand,
                "district": district,
                "orders": n,
                "chilled_orders": int(group["temp_requirement"].eq("chilled").sum()),
                "volume_m3": float(group["order_volume_m3"].sum()),
                "volume_cap_m3": float(vehicle["volume_cap_m3"]),
                "weight_kg": float(group["order_weight_kg"].sum()),
                "weight_cap_kg": float(vehicle["weight_cap_kg"]),
                "trip_minutes": minutes,
            }
        )
    trip_frame = pd.DataFrame(trips).sort_values(["vehicle_id", "trip_id"]).reset_index(drop=True)
    deferred = merged.loc[merged["decision"].eq("deferred")]
    report = {
        "trip_count": int(len(trip_frame)),
        "vehicles_used": int(trip_frame["vehicle_id"].nunique()),
        "reefer_trips": int(trip_frame["vehicle_temp"].eq("reefer").sum()),
        "van_trips": int(trip_frame["vehicle_type"].eq("van").sum()),
        "fresh_minutes_by_vehicle": {
            vehicle: float(value)
            for vehicle, value in trip_frame.loc[trip_frame["brand"].eq("Fresh")].groupby("vehicle_id")["trip_minutes"].sum().items()
        },
        "daytime_minutes_by_vehicle": {
            vehicle: float(value)
            for vehicle, value in trip_frame.loc[trip_frame["brand"].ne("Fresh")].groupby("vehicle_id")["trip_minutes"].sum().items()
        },
        "deferred_by_brand_temp": {
            f"{brand}|{temp}": int(count)
            for (brand, temp), count in deferred.groupby(["brand", "temp_requirement"]).size().items()
        },
        "deferred_orders": deferred[
            [
                "order_ref",
                "outlet_id",
                "brand",
                "district",
                "temp_requirement",
                "parking_constraint",
                "order_volume_m3",
                "order_weight_kg",
                "deferred_yesterday",
                "days_since_last_served",
            ]
        ].to_dict("records"),
    }
    return trip_frame, report


def main() -> None:
    started = time.perf_counter()
    solutions: dict[str, tuple[pd.DataFrame, dict]] = {}
    for policy in POLICIES:
        allocation, summary = solve(policy)
        solutions[policy.name] = (allocation, summary)
        print(policy.name, summary)

    # The final choice is explicit and lexicographic: protect yesterday's
    # deferrals, then stale outlets, Fresh/chilled demand, and coverage.
    candidate_names = ["fairness_first", "cold_chain", "coverage"]
    selected_name = max(
        candidate_names,
        key=lambda name: (
            solutions[name][1]["deferred_yesterday_served"],
            solutions[name][1]["days_since_last_served_sum"],
            solutions[name][1]["chilled_orders_served"],
            solutions[name][1]["fresh_orders_served"],
            solutions[name][1]["served_orders"],
            solutions[name][1]["served_volume_m3"],
        ),
    )
    allocation, selected_summary = solutions[selected_name]
    trip_frame, detail = trip_analysis(allocation)
    maximum_orders = solutions["max_orders"][1]["served_orders"]
    selected_summary["maximum_feasible_order_count"] = maximum_orders
    selected_summary["minimum_unavoidable_deferral_count"] = 85 - maximum_orders
    selected_summary["selected_count_gap_to_maximum"] = maximum_orders - selected_summary["served_orders"]

    template = pd.read_csv(DATA / "Submission Templates" / "submission_task2b.csv")
    submission = template[["scenario", "order_ref", "outlet_id"]].merge(
        allocation,
        on=["scenario", "order_ref", "outlet_id"],
        how="left",
        validate="one_to_one",
    )
    if not submission[["scenario", "order_ref", "outlet_id"]].equals(template[["scenario", "order_ref", "outlet_id"]]):
        raise AssertionError("Task 2B identifiers/order changed")
    deferred = submission["decision"].eq("deferred")
    submission.loc[deferred, ["vehicle_id", "trip_id"]] = pd.NA
    submission_path = WORK / "submissions" / "submission_task2b.csv"
    trip_path = WORK / "reports" / "task2b_trip_summary.csv"
    report_path = WORK / "reports" / "task2b_optimization_report.json"
    submission.to_csv(submission_path, index=False)
    trip_frame.to_csv(trip_path, index=False)
    report = {
        "selected_policy": selected_name,
        "selected_summary": selected_summary,
        "sensitivity": {name: summary for name, (_, summary) in solutions.items()},
        "allocation_detail": detail,
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {"submission": str(submission_path), "trip_summary": str(trip_path)},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"WROTE {submission_path}")
    print(f"WROTE {trip_path}")
    print(f"WROTE {report_path}")


if __name__ == "__main__":
    main()
