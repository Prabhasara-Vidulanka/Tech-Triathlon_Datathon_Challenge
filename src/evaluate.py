"""Held-out error analysis, interpretability checks, and competition figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from features_task1 import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_task1_feature_frame
from features_task2a import build_weekly_history
from labels import build_task1_labels
from train_task1 import classification_metrics, late_hgb, regression_metrics, service_hgb


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
DATA = ROOT / "Data"
SEED = 20261006


def main() -> None:
    figures = WORK / "figures"
    reports = WORK / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    features = build_task1_feature_frame("train")
    labels, _ = build_task1_labels()
    data = features.merge(labels[["delivery_id", "service_min", "late"]], on="delivery_id", validate="one_to_one")
    data["dispatch_date"] = pd.to_datetime(data["dispatch_date"])
    valid_start = pd.Timestamp("2026-01-05")
    train = data.loc[data["dispatch_date"].lt(valid_start)].copy()
    valid = data.loc[data["dispatch_date"].ge(valid_start)].copy()
    columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    service_model = service_hgb("squared_error", 31, 30).fit(train[columns], train["service_min"])
    late_model = late_hgb().fit(train[columns], train["late"])
    valid["service_pred"] = service_model.predict(valid[columns])
    valid["late_prob"] = late_model.predict_proba(valid[columns])[:, 1]
    valid["service_abs_error"] = np.abs(valid["service_pred"] - valid["service_min"])

    subgroup_rows = []
    for dimension in ["brand", "district", "depot", "dock_type", "monsoon", "seq_in_route"]:
        for group, frame in valid.groupby(dimension):
            subgroup_rows.append(
                {
                    "dimension": dimension,
                    "group": str(group),
                    "rows": len(frame),
                    "service_mae": regression_metrics(frame["service_min"].to_numpy(), frame["service_pred"].to_numpy())["mae"],
                    "late_log_loss": classification_metrics(frame["late"].to_numpy(), frame["late_prob"].to_numpy())["log_loss"]
                    if frame["late"].nunique() > 1
                    else np.nan,
                    "late_rate": frame["late"].mean(),
                    "mean_probability": frame["late_prob"].mean(),
                }
            )
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(reports / "task1_subgroup_errors.csv", index=False)
    worst = valid.nlargest(100, "service_abs_error")[
        [
            "delivery_id",
            "dispatch_date",
            "outlet_id",
            "brand",
            "district",
            "dock_type",
            "order_volume_m3",
            "seq_in_route",
            "service_min",
            "service_pred",
            "service_abs_error",
            "late",
            "late_prob",
        ]
    ]
    worst.to_csv(reports / "task1_largest_service_errors.csv", index=False)

    sample = valid.sample(min(2000, len(valid)), random_state=SEED)
    service_importance = permutation_importance(
        service_model,
        sample[columns],
        sample["service_min"],
        scoring="neg_mean_absolute_error",
        n_repeats=2,
        random_state=SEED,
        n_jobs=1,
    )
    late_importance = permutation_importance(
        late_model,
        sample[columns],
        sample["late"],
        scoring="neg_log_loss",
        n_repeats=2,
        random_state=SEED,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": columns,
            "service_importance_mean": service_importance.importances_mean,
            "service_importance_std": service_importance.importances_std,
            "late_importance_mean": late_importance.importances_mean,
            "late_importance_std": late_importance.importances_std,
        }
    ).sort_values("service_importance_mean", ascending=False)
    importance.to_csv(reports / "task1_permutation_importance.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for brand, frame in data.groupby("brand"):
        axes[0].hist(frame["service_min"], bins=np.arange(0, 161, 5), alpha=0.55, density=True, label=brand)
    axes[0].set_xlim(0, 160)
    axes[0].set_xlabel("Observed service time (minutes)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Service-time distribution")
    axes[0].legend(frameon=False)
    axes[1].scatter(valid["service_pred"], valid["service_min"], s=7, alpha=0.18)
    bound = np.percentile(np.r_[valid["service_pred"], valid["service_min"]], 99)
    axes[1].plot([0, bound], [0, bound], color="black", linewidth=1)
    axes[1].set_xlim(0, bound)
    axes[1].set_ylim(0, bound)
    axes[1].set_xlabel("Predicted service time (minutes)")
    axes[1].set_ylabel("Observed service time (minutes)")
    axes[1].set_title("Pseudo-test predictions")
    frac, mean_pred = calibration_curve(valid["late"], valid["late_prob"], n_bins=10, strategy="quantile")
    axes[2].plot([0, 1], [0, 1], color="black", linewidth=1)
    axes[2].plot(mean_pred, frac, marker="o")
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Observed late rate")
    axes[2].set_title("Lateness calibration")
    fig.tight_layout()
    fig.savefig(figures / "task1_validation_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    history = build_weekly_history()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, brand in zip(axes, ["Fresh", "Style", "Tech"]):
        for depot, frame in history.loc[history["brand"].eq(brand)].groupby("depot"):
            axis.plot(frame["week_start"], frame["total_volume_m3"], label=depot, linewidth=1.25)
        axis.set_ylabel("Volume (m³)")
        axis.set_title(brand)
        axis.legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Requested-order week")
    fig.suptitle("Weekly demand history by brand and depot", y=0.995)
    fig.tight_layout()
    fig.savefig(figures / "task2a_weekly_series.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    allocation = pd.read_csv(WORK / "submissions" / "submission_task2b.csv")
    scenario = pd.read_csv(DATA / "Test Data" / "task2b_peak_day_scenarios.csv")
    allocation = scenario.merge(allocation, on=["scenario", "order_ref", "outlet_id"], validate="one_to_one")
    counts = allocation.groupby(["brand", "decision"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["served", "deferred"], fill_value=0)
    ax = counts.plot(kind="bar", stacked=True, figsize=(8, 4), color=["#2f7d67", "#b95050"])
    ax.set_ylabel("Orders")
    ax.set_xlabel("Brand")
    ax.set_title("Peak-day allocation decisions")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figures / "task2b_allocation_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    report = {
        "pseudo_test_period": [str(valid["dispatch_date"].min().date()), str(valid["dispatch_date"].max().date())],
        "service_metrics": regression_metrics(valid["service_min"].to_numpy(), valid["service_pred"].to_numpy()),
        "late_metrics": classification_metrics(valid["late"].to_numpy(), valid["late_prob"].to_numpy()),
        "top_service_features": importance.nlargest(12, "service_importance_mean")[["feature", "service_importance_mean"]].to_dict("records"),
        "top_late_features": importance.nlargest(12, "late_importance_mean")[["feature", "late_importance_mean"]].to_dict("records"),
        "largest_error_summary": worst["service_abs_error"].describe().to_dict(),
    }
    (reports / "error_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
