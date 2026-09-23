"""Evaluation harness for DonorCast.

Implements:
1. Split assignment by TARGET date (train, validation, test). Drops predictions where target exceeds split end.
2. Regression metrics: WAPE, MASE (scaled by in-sample seasonal-naive-7 on training data), pinball loss at 0.1 and 0.9.
3. Metric slicing: overall, by horizon, by blood group, by facility size tier (top 5, middle, bottom 7), and by holiday window.
4. Shortfall flag classification: forecast 7-day sum vs typical (median 7-day total for the same facility, group,
   and ISO week over previous TYPICAL_YEARS years before origin date) < SHORTFALL_RATIO x typical.
   Reports Precision, Recall, and F1.
5. Single entrypoint `evaluate(predictions_df, split)` writing reports/results_<model>_<split>.md.
6. Guarded test set access locked strictly to final evaluation with `reports/final_run.json` timestamp tracking.
"""

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from donorcast.calendar import build_calendar_dataframe, load_facility_state_mapping
from donorcast.config import (
    CALENDAR_PROCESSED_FILE,
    DATA_PROCESSED_DIR,
    FACILITY_STATE_FILE,
    FINAL_RUN_FILE,
    MASE_WINDOW_END,
    MASE_WINDOW_START,
    REPORTS_DIR,
    SHORTFALL_RATIO,
    TEST_END,
    TRAIN_END,
    TRAIN_START,
    TYPICAL_YEARS,
    VAL_END,
)


def filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter prediction DataFrame to only rows belonging to the split by TARGET date.

    Splits:
    - 'train': target_date <= TRAIN_END (and >= TRAIN_START if present)
    - 'val' or 'validation': TRAIN_END < target_date <= VAL_END
    - 'test': VAL_END < target_date <= TEST_END

    Rows where target_date is past the split end or before split start are dropped.
    """
    split_norm = split.lower().strip()
    df_filtered = df.copy()

    # Ensure target_date is string/comparable
    df_filtered["target_date"] = df_filtered["target_date"].astype(str)

    if split_norm == "train":
        mask = (df_filtered["target_date"] >= TRAIN_START) & (
            df_filtered["target_date"] <= TRAIN_END
        )
    elif split_norm in ("val", "validation"):
        mask = (df_filtered["target_date"] > TRAIN_END) & (df_filtered["target_date"] <= VAL_END)
    elif split_norm == "test":
        mask = (df_filtered["target_date"] > VAL_END) & (df_filtered["target_date"] <= TEST_END)
    else:
        raise ValueError(
            f"Unknown split: '{split}'. Must be one of 'train', 'val'/'validation', or 'test'."
        )

    return df_filtered[mask].copy()


def compute_mase_scales(
    train_long_df: pd.DataFrame,
    window_start: str = MASE_WINDOW_START,
    window_end: str = MASE_WINDOW_END,
) -> dict[tuple[str, str], float]:
    """Compute in-sample Seasonal Naive 7 MASE denominator per (facility, group).

    Computed over the specified window (default: last 3 years before TRAIN_END, 2020-01-01 to 2022-12-31).
    Formula: (1 / (N - 7)) * sum_{t=8}^N | y_t - y_{t-7} |
    """
    df_window = train_long_df[
        (train_long_df["date"] >= window_start) & (train_long_df["date"] <= window_end)
    ].sort_values(["facility", "group", "date"])
    scales: dict[tuple[str, str], float] = {}

    for (fac, grp), group_df in df_window.groupby(["facility", "group"], observed=False):
        y = group_df["donations"].values.astype(np.float64)
        if len(y) > 7:
            diffs = np.abs(y[7:] - y[:-7])
            scale = float(np.mean(diffs))
            scales[(str(fac), str(grp))] = scale if scale > 1e-6 else 1.0
        else:
            scales[(str(fac), str(grp))] = 1.0

    return scales


def compute_facility_tiers(
    train_long_df: pd.DataFrame, train_end: str = TRAIN_END
) -> dict[str, str]:
    """Rank facilities by mean daily donations over the training set into 3 tiers:

    - 'top_5' (top 5 facilities by volume)
    - 'middle' (next 10 facilities)
    - 'bottom_7' (bottom 7 facilities)
    """
    df_train = train_long_df[train_long_df["date"] <= train_end]
    # Sum across groups per facility-day to get facility daily total, then mean
    fac_daily = df_train.groupby(["facility", "date"], observed=False)["donations"].sum()
    fac_mean = fac_daily.groupby("facility", observed=False).mean().sort_values(ascending=False)

    ranked_facilities = list(fac_mean.index)
    top_5 = set(ranked_facilities[:5])
    bottom_7 = set(ranked_facilities[-7:])

    tiers = {}
    for fac in ranked_facilities:
        if fac in top_5:
            tiers[str(fac)] = "top_5"
        elif fac in bottom_7:
            tiers[str(fac)] = "bottom_7"
        else:
            tiers[str(fac)] = "middle"

    return tiers


def compute_wape_7d(df: pd.DataFrame) -> float:
    """Compute WAPE_7D across all complete (facility, group, origin_date) 7-day windows.

    Formula: sum(|sum_{h=1..7} y - sum_{h=1..7} y_hat|) / sum(sum_{h=1..7} y)
    """
    h7_df = df[df["horizon"].isin(range(1, 8))]
    if len(h7_df) == 0:
        return np.nan

    grouped = h7_df.groupby(["facility", "group", "origin_date"], observed=False).agg(
        pred_7d=("prediction", "sum"),
        true_7d=("target", "sum"),
        count=("horizon", "count"),
    )
    # Only keep complete 7-day forecast windows
    grouped = grouped[grouped["count"] == 7]
    if len(grouped) == 0:
        return np.nan

    total_actual_7d = grouped["true_7d"].sum()
    total_abs_error_7d = np.abs(grouped["true_7d"] - grouped["pred_7d"]).sum()

    return float(total_abs_error_7d / total_actual_7d) if total_actual_7d > 0 else np.nan


def compute_regression_metrics(
    df: pd.DataFrame, mase_scales: dict[tuple[str, str], float]
) -> dict[str, float]:
    """Compute WAPE, MASE, WAPE_7D, and pinball losses (if prediction intervals provided)."""
    if len(df) == 0:
        return {
            "wape": np.nan,
            "wape_7d": np.nan,
            "mase": np.nan,
            "pinball_10": np.nan,
            "pinball_90": np.nan,
            "count": 0,
        }

    y_true = df["target"].values.astype(np.float64)
    y_pred = df["prediction"].values.astype(np.float64)

    abs_errors = np.abs(y_true - y_pred)
    total_abs_error = np.sum(abs_errors)
    total_actual = np.sum(y_true)

    # WAPE = sum(|y - y_hat|) / sum(y)
    wape = float(total_abs_error / total_actual) if total_actual > 0 else np.nan

    # WAPE_7D = sum(|sum_7d y - sum_7d y_hat|) / sum(sum_7d y)
    wape_7d = compute_wape_7d(df)

    # MASE = mean( |y - y_hat| / scale_(fac, grp) )
    scales = np.array(
        [
            mase_scales.get((str(f), str(g)), 1.0)
            for f, g in zip(df["facility"], df["group"], strict=False)
        ],
        dtype=np.float64,
    )
    mase = float(np.mean(abs_errors / scales))

    metrics: dict[str, float] = {
        "wape": wape,
        "wape_7d": wape_7d,
        "mase": mase,
        "count": len(df),
    }

    # Pinball loss at 0.1 and 0.9 if columns exist
    if "pred_p10" in df.columns:
        p10 = df["pred_p10"].values.astype(np.float64)
        diff_10 = y_true - p10
        pinball_10 = np.maximum(0.1 * diff_10, (0.1 - 1.0) * diff_10)
        metrics["pinball_10"] = float(np.mean(pinball_10))

    if "pred_p90" in df.columns:
        p90 = df["pred_p90"].values.astype(np.float64)
        diff_90 = y_true - p90
        pinball_90 = np.maximum(0.9 * diff_90, (0.9 - 1.0) * diff_90)
        metrics["pinball_90"] = float(np.mean(pinball_90))

    return metrics


class HistoricalTypicalLookup:
    """Precomputes 7-day sums by (facility, group, year, iso_week) to evaluate shortfall thresholds without future leakage."""

    def __init__(self, long_df: pd.DataFrame):
        df = long_df.copy()
        df["dt"] = pd.to_datetime(df["date"])
        df["year"] = df["dt"].dt.isocalendar().year
        df["week"] = df["dt"].dt.isocalendar().week

        # Group by facility, group, year, week to get 7-day totals
        weekly = (
            df.groupby(["facility", "group", "year", "week"], observed=False)["donations"]
            .sum()
            .reset_index()
        )
        self.weekly = weekly

    def get_typical_7d(
        self,
        facility: str,
        group: str,
        origin_date_str: str,
        typical_years: int = TYPICAL_YEARS,
    ) -> float:
        """Compute median 7-day total for the same facility, group, and ISO week

        over the previous `typical_years` years strictly before origin date.
        """
        origin_dt = datetime.date.fromisoformat(origin_date_str)
        # Next 7-day target window corresponds to the ISO week of (origin + 1 day) or origin
        target_start_dt = origin_dt + datetime.timedelta(days=1)
        iso_year, iso_week, _ = target_start_dt.isocalendar()

        prev_years = [iso_year - i for i in range(1, typical_years + 1)]

        mask = (
            (self.weekly["facility"] == facility)
            & (self.weekly["group"] == group)
            & (self.weekly["week"] == iso_week)
            & (self.weekly["year"].isin(prev_years))
        )
        subset = self.weekly[mask]
        if len(subset) > 0:
            return float(subset["donations"].median())
        return 0.0


def _evaluate_shortfall_subset(
    grouped_subset: pd.DataFrame,
    threshold_col: str = "threshold",
) -> dict[str, Any]:
    """Helper to evaluate confusion matrix, prevalence, precision, recall, and F1 on a grouped DataFrame."""
    if len(grouped_subset) == 0:
        return {
            "prevalence": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "actual_shortfalls": 0,
            "predicted_shortfalls": 0,
            "total_windows": 0,
        }

    pred_flag = grouped_subset["pred_7d"] < grouped_subset[threshold_col]
    true_flag = grouped_subset["true_7d"] < grouped_subset[threshold_col]

    tp = int(np.sum(pred_flag & true_flag))
    fp = int(np.sum(pred_flag & (~true_flag)))
    fn = int(np.sum((~pred_flag) & true_flag))
    tn = int(np.sum((~pred_flag) & (~true_flag)))

    total_windows = len(grouped_subset)
    actual_shortfalls = int(np.sum(true_flag))
    predicted_shortfalls = int(np.sum(pred_flag))

    prevalence = float(actual_shortfalls / total_windows) if total_windows > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "prevalence": prevalence,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "actual_shortfalls": actual_shortfalls,
        "predicted_shortfalls": predicted_shortfalls,
        "total_windows": total_windows,
    }


def compute_shortfall_metrics(
    predictions_df: pd.DataFrame,
    long_df: pd.DataFrame,
    shortfall_ratio: float = SHORTFALL_RATIO,
    typical_years: int = TYPICAL_YEARS,
    facility_tiers: dict[str, str] | None = None,
    typical_lookup: HistoricalTypicalLookup | None = None,
) -> dict[str, Any]:
    """Evaluate 7-day shortfall alert classification for horizons 1..7.

    For each (facility, group, origin_date):
    - sum_pred_7d = sum(prediction for h in 1..7)
    - sum_true_7d = sum(target for h in 1..7)
    - typical_7d = median 7-day total for same (facility, group, ISO week) in past `typical_years`
    - pred_shortfall = (sum_pred_7d < shortfall_ratio * typical_7d)
    - actual_shortfall = (sum_true_7d < shortfall_ratio * typical_7d)

    Returns overall and tier-level prevalence, precision, recall, f1, support, and confusion matrix counts.
    """
    if typical_lookup is None:
        typical_lookup = HistoricalTypicalLookup(long_df)

    if facility_tiers is None:
        facility_tiers = compute_facility_tiers(long_df, train_end=TRAIN_END)

    # Restrict to horizons 1..7
    h7_df = predictions_df[predictions_df["horizon"].isin(range(1, 8))].copy()

    # Aggregate by facility, group, origin_date
    grouped = (
        h7_df.groupby(["facility", "group", "origin_date"], observed=False)
        .agg(
            pred_7d=("prediction", "sum"),
            true_7d=("target", "sum"),
            count=("horizon", "count"),
        )
        .reset_index()
    )

    # Only evaluate origins that have all 7 horizons present
    grouped = grouped[grouped["count"] == 7].copy()

    if len(grouped) == 0:
        empty_res = _evaluate_shortfall_subset(grouped)
        empty_res["by_tier"] = {
            tier: _evaluate_shortfall_subset(pd.DataFrame())
            for tier in ["top_5", "middle", "bottom_7"]
        }
        return empty_res

    # Vectorized / memoized typical calculation
    typical_cache: dict[tuple[str, str, str], float] = {}

    def _get_typical(row):
        key = (str(row["facility"]), str(row["group"]), str(row["origin_date"]))
        if key not in typical_cache:
            typical_cache[key] = typical_lookup.get_typical_7d(
                facility=key[0],
                group=key[1],
                origin_date_str=key[2],
                typical_years=typical_years,
            )
        return typical_cache[key]

    grouped["typical"] = grouped.apply(_get_typical, axis=1)
    grouped["threshold"] = shortfall_ratio * grouped["typical"]
    grouped["facility_tier"] = grouped["facility"].map(
        lambda f: facility_tiers.get(str(f), "middle")
    )

    overall_metrics = _evaluate_shortfall_subset(grouped, threshold_col="threshold")

    # Breakdown by facility tier
    by_tier = {}
    for tier in ["top_5", "middle", "bottom_7"]:
        tier_sub = grouped[grouped["facility_tier"] == tier]
        by_tier[tier] = _evaluate_shortfall_subset(tier_sub, threshold_col="threshold")

    overall_metrics["by_tier"] = by_tier
    return overall_metrics


def evaluate_breakdowns(
    df: pd.DataFrame,
    mase_scales: dict[tuple[str, str], float],
    facility_tiers: dict[str, str],
    calendar_df: pd.DataFrame,
    facility_state_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute detailed metric breakdowns across:

    - overall
    - by horizon (1..14)
    - by blood group (A, B, O, AB)
    - by facility tier (top_5, middle, bottom_7)
    - in holiday windows vs not (is_public_holiday / is_school_holiday / is_weekend)
    """
    results: dict[str, Any] = {}

    # 1. Overall
    results["overall"] = compute_regression_metrics(df, mase_scales)

    # 2. By Horizon
    results["by_horizon"] = {}
    for h, h_df in df.groupby("horizon", observed=False):
        results["by_horizon"][int(h)] = compute_regression_metrics(h_df, mase_scales)

    # 3. By Group
    results["by_group"] = {}
    for grp, g_df in df.groupby("group", observed=False):
        results["by_group"][str(grp)] = compute_regression_metrics(g_df, mase_scales)

    # 4. By Facility Tier
    df_with_tier = df.copy()
    df_with_tier["facility_tier"] = df_with_tier["facility"].map(
        lambda f: facility_tiers.get(str(f), "middle")
    )
    results["by_tier"] = {}
    for tier in ["top_5", "middle", "bottom_7"]:
        t_df = df_with_tier[df_with_tier["facility_tier"] == tier]
        results["by_tier"][tier] = compute_regression_metrics(t_df, mase_scales)

    # 5. Holiday vs Non-Holiday
    # Check if df has holiday columns, otherwise merge from calendar
    if "is_public_holiday" not in df.columns:
        # Join state from facility_state_df, then calendar features
        fac_to_state = dict(
            zip(facility_state_df["facility"], facility_state_df["state"], strict=False)
        )
        df_cal = df.copy()
        df_cal["state"] = df_cal["facility"].map(fac_to_state)
        merged = df_cal.merge(
            calendar_df[["state", "date", "is_public_holiday"]].rename(
                columns={"date": "target_date"}
            ),
            on=["state", "target_date"],
            how="left",
        )
    else:
        merged = df

    # For 7D holiday aggregation: a 7-day window is a holiday window if any target day in h=1..7 is a holiday
    # Group at origin-facility-group level to assign window-level holiday flag
    h7_df = merged[merged["horizon"].isin(range(1, 8))].copy()
    window_holidays = (
        h7_df.groupby(["facility", "group", "origin_date"], observed=False)["is_public_holiday"]
        .max()
        .reset_index()
        .rename(columns={"is_public_holiday": "window_has_holiday"})
    )
    merged_with_wh = merged.merge(
        window_holidays, on=["facility", "group", "origin_date"], how="left"
    )

    holiday_df = merged_with_wh[merged_with_wh["is_public_holiday"] == 1]
    non_holiday_df = merged_with_wh[merged_with_wh["is_public_holiday"] == 0]

    holiday_metrics = compute_regression_metrics(holiday_df, mase_scales)
    non_holiday_metrics = compute_regression_metrics(non_holiday_df, mase_scales)

    # Calculate WAPE_7D partitioned by whether the 7-day window contains a holiday
    w_hol_df = merged_with_wh[merged_with_wh["window_has_holiday"] == 1]
    w_non_hol_df = merged_with_wh[merged_with_wh["window_has_holiday"] == 0]
    holiday_metrics["wape_7d"] = compute_wape_7d(w_hol_df)
    non_holiday_metrics["wape_7d"] = compute_wape_7d(w_non_hol_df)

    results["by_holiday"] = {
        "holiday": holiday_metrics,
        "non_holiday": non_holiday_metrics,
    }

    return results


def format_results_markdown(
    model_name: str,
    split: str,
    results: dict[str, Any],
    shortfall_metrics: dict[str, Any],
) -> str:
    """Format evaluation results as a comprehensive markdown report."""
    overall = results["overall"]
    pinball_10_str = (
        f"{overall['pinball_10']:.2f}"
        if "pinball_10" in overall and not np.isnan(overall["pinball_10"])
        else "N/A"
    )
    pinball_90_str = (
        f"{overall['pinball_90']:.2f}"
        if "pinball_90" in overall and not np.isnan(overall["pinball_90"])
        else "N/A"
    )

    md = [
        f"# Evaluation Report: {model_name} ({split.upper()} Split)",
        "",
        f"- **Model**: `{model_name}`",
        f"- **Split**: `{split}`",
        f"- **Evaluated at**: `{datetime.datetime.now(datetime.UTC).isoformat()}`",
        f"- **Total Rows Evaluated**: {overall['count']:,}",
        "",
        "---",
        "",
        "## 1. Headline Regression Metrics",
        "",
        "| Metric | Value | Description |",
        "|---|---|---|",
        f"| **WAPE_7D (Primary)** | **{overall['wape_7d'] * 100:.1f}%** | 7-day cumulative sum WAPE |",
        f"| **WAPE (Daily)** | **{overall['wape'] * 100:.1f}%** | Daily Weighted Absolute Percentage Error |",
        f"| **MASE** | **{overall['mase']:.2f}** | Mean Absolute Scaled Error (vs 2020–2022 seasonal naive 7) |",
        f"| **Pinball Loss (p10)** | {pinball_10_str} | Quantile loss at 10th percentile |",
        f"| **Pinball Loss (p90)** | {pinball_90_str} | Quantile loss at 90th percentile |",
        "",
        "---",
        "",
        "## 2. Shortfall Classification Metrics (7-Day Ahead)",
        "",
        f"- **Shortfall Definition**: 7-day predicted sum < {SHORTFALL_RATIO:.1f} × historical 3-year median for ISO week",
        f"- **Evaluated Windows**: {shortfall_metrics['total_windows']:,}",
        f"- **Actual Shortfall Windows**: {shortfall_metrics['actual_shortfalls']:,} (Prevalence: **{shortfall_metrics['prevalence'] * 100:.1f}%**)",
        f"- **Predicted Shortfall Windows**: {shortfall_metrics['predicted_shortfalls']:,}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Prevalence** | **{shortfall_metrics['prevalence'] * 100:.1f}%** |",
        f"| **Precision** | **{shortfall_metrics['precision'] * 100:.1f}%** |",
        f"| **Recall** | **{shortfall_metrics['recall'] * 100:.1f}%** |",
        f"| **F1 Score** | **{shortfall_metrics['f1']:.2f}** |",
        f"| True Positives (TP) | {shortfall_metrics['tp']} |",
        f"| False Positives (FP) | {shortfall_metrics['fp']} |",
        f"| False Negatives (FN) | {shortfall_metrics['fn']} |",
        f"| True Negatives (TN) | {shortfall_metrics['tn']} |",
        "",
        "### Shortfall Metrics by Facility Tier",
        "",
        "| Tier | Description | Prevalence | Precision | Recall | F1 Score | Windows |",
        "|---|---|---|---|---|---|---|",
    ]

    tier_desc = {
        "top_5": "Top 5 High-Volume Sites",
        "middle": "Middle 10 Sites",
        "bottom_7": "Bottom 7 Small Sites",
    }
    for tier in ["top_5", "middle", "bottom_7"]:
        t_sf = shortfall_metrics["by_tier"][tier]
        md.append(
            f"| **{tier}** | {tier_desc[tier]} | {t_sf['prevalence'] * 100:.1f}% | {t_sf['precision'] * 100:.1f}% | {t_sf['recall'] * 100:.1f}% | {t_sf['f1']:.2f} | {t_sf['total_windows']:,} |"
        )

    md.extend(
        [
            "",
            "---",
            "",
            "## 3. Breakdown by Horizon (h = 1..14)",
            "",
            "| Horizon | WAPE | MASE | Count |",
            "|---|---|---|---|",
        ]
    )

    for h in sorted(results["by_horizon"].keys()):
        h_met = results["by_horizon"][h]
        md.append(
            f"| Day {h:02d} | {h_met['wape'] * 100:.1f}% | {h_met['mase']:.2f} | {h_met['count']:,} |"
        )

    md.extend(
        [
            "",
            "---",
            "",
            "## 4. Breakdown by Blood Group",
            "",
            "| Group | WAPE_7D | Daily WAPE | MASE | Count |",
            "|---|---|---|---|---|",
        ]
    )

    for grp in ["A", "B", "O", "AB"]:
        if grp in results["by_group"]:
            g_met = results["by_group"][grp]
            md.append(
                f"| **{grp}** | {g_met['wape_7d'] * 100:.1f}% | {g_met['wape'] * 100:.1f}% | {g_met['mase']:.2f} | {g_met['count']:,} |"
            )

    md.extend(
        [
            "",
            "---",
            "",
            "## 5. Breakdown by Facility Tier",
            "",
            "| Tier | Description | WAPE_7D | Daily WAPE | MASE | Count |",
            "|---|---|---|---|---|---|",
            f"| **top_5** | PDN, Penang, JB, Ipoh, Melaka | {results['by_tier']['top_5']['wape_7d'] * 100:.1f}% | {results['by_tier']['top_5']['wape'] * 100:.1f}% | {results['by_tier']['top_5']['mase']:.2f} | {results['by_tier']['top_5']['count']:,} |",
            f"| **middle** | Mid-size hospitals | {results['by_tier']['middle']['wape_7d'] * 100:.1f}% | {results['by_tier']['middle']['wape'] * 100:.1f}% | {results['by_tier']['middle']['mase']:.2f} | {results['by_tier']['middle']['count']:,} |",
            f"| **bottom_7** | Small / intermittent hospitals | {results['by_tier']['bottom_7']['wape_7d'] * 100:.1f}% | {results['by_tier']['bottom_7']['wape'] * 100:.1f}% | {results['by_tier']['bottom_7']['mase']:.2f} | {results['by_tier']['bottom_7']['count']:,} |",
            "",
            "---",
            "",
            "## 6. Holiday Window Slices",
            "",
            "| Window | WAPE_7D | Daily WAPE | MASE | Count |",
            "|---|---|---|---|---|",
            f"| **Public Holiday** | {results['by_holiday']['holiday']['wape_7d'] * 100:.1f}% | {results['by_holiday']['holiday']['wape'] * 100:.1f}% | {results['by_holiday']['holiday']['mase']:.2f} | {results['by_holiday']['holiday']['count']:,} |",
            f"| **Non-Holiday** | {results['by_holiday']['non_holiday']['wape_7d'] * 100:.1f}% | {results['by_holiday']['non_holiday']['wape'] * 100:.1f}% | {results['by_holiday']['non_holiday']['mase']:.2f} | {results['by_holiday']['non_holiday']['count']:,} |",
            "",
        ]
    )

    return "\n".join(md)


def evaluate(
    predictions_df: pd.DataFrame,
    split: str,
    model_name: str = "model",
    reports_dir: Path = REPORTS_DIR,
    allow_test: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Single entrypoint to evaluate predictions on a given split.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Dataframe containing at minimum:
        ['facility', 'group', 'origin_date', 'horizon', 'target_date', 'target', 'prediction']
        Optional: ['pred_p10', 'pred_p90']
    split : str
        Split to evaluate: 'train', 'val' / 'validation', or 'test'.
    model_name : str
        Name of model for reporting.
    reports_dir : Path
        Directory to write markdown report.
    allow_test : bool
        Must be True to allow test split evaluation (passed only by final CLI command).
    force : bool
        Override previous final evaluation lock if True.

    Returns
    -------
    dict[str, Any]
        Dictionary with regression and shortfall results.
    """
    split_norm = split.lower().strip()

    # Guard test split evaluation
    if split_norm == "test":
        if not allow_test:
            raise PermissionError(
                "Invariant TEST_SET_TOUCHED_ONCE: Evaluating on the test split is restricted to 'donorcast final'."
            )

        if FINAL_RUN_FILE.exists() and not force:
            raise RuntimeError(
                f"Invariant TEST_SET_TOUCHED_ONCE: Test set was already evaluated and recorded at {FINAL_RUN_FILE}. "
                "Use --force to override."
            )

        # Record test run timestamp
        FINAL_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
        final_info = {
            "model_name": model_name,
            "split": split_norm,
            "evaluated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": "completed",
        }
        with open(FINAL_RUN_FILE, "w") as f:
            json.dump(final_info, f, indent=2)

    # 1. Filter predictions strictly to the split by target_date
    df_split = filter_split(predictions_df, split_norm)
    if len(df_split) == 0:
        raise ValueError(f"No prediction rows matched split '{split}'. Please check target dates.")

    # 2. Load necessary reference data for scaling and features
    long_path = DATA_PROCESSED_DIR / "long.parquet"
    if not long_path.exists():
        from donorcast.clean import clean_data

        clean_data()
    long_df = pd.read_parquet(long_path)

    cal_path = CALENDAR_PROCESSED_FILE
    if not cal_path.exists():
        calendar_df = build_calendar_dataframe()
        calendar_df.to_parquet(cal_path, index=False)
    else:
        calendar_df = pd.read_parquet(cal_path)

    fac_state_df = load_facility_state_mapping(FACILITY_STATE_FILE)

    # 3. Compute in-sample MASE scaling factors on training data (2020-2022 window)
    mase_scales = compute_mase_scales(
        long_df, window_start=MASE_WINDOW_START, window_end=MASE_WINDOW_END
    )

    # 4. Facility tiers
    facility_tiers = compute_facility_tiers(long_df, train_end=TRAIN_END)

    # 5. Regression & Breakdown metrics
    breakdown_results = evaluate_breakdowns(
        df=df_split,
        mase_scales=mase_scales,
        facility_tiers=facility_tiers,
        calendar_df=calendar_df,
        facility_state_df=fac_state_df,
    )

    # 6. Shortfall classification metrics
    shortfall_results = compute_shortfall_metrics(
        predictions_df=df_split,
        long_df=long_df,
        shortfall_ratio=SHORTFALL_RATIO,
        typical_years=TYPICAL_YEARS,
        facility_tiers=facility_tiers,
    )

    # 7. Write Markdown Report
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_md = format_results_markdown(
        model_name=model_name,
        split=split_norm,
        results=breakdown_results,
        shortfall_metrics=shortfall_results,
    )
    report_file = reports_dir / f"results_{model_name}_{split_norm}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Saved evaluation report: {report_file}")

    return {
        "model_name": model_name,
        "split": split_norm,
        "overall": breakdown_results["overall"],
        "shortfall": shortfall_results,
        "by_horizon": breakdown_results["by_horizon"],
        "by_group": breakdown_results["by_group"],
        "by_tier": breakdown_results["by_tier"],
        "by_holiday": breakdown_results["by_holiday"],
        "report_file": str(report_file),
    }
