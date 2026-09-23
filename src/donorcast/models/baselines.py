"""Baseline forecasting models for DonorCast.

Implements:
- M0 (Seasonal Naive): Forecast for t+h = donations on the most recent same weekday <= t.
- M0b (Weekday Moving Average): Forecast for t+h = mean of donations across the last 4 occurrences of the same weekday <= t.

Both models operate on the long-format DataFrame and produce predictions adhering to the evaluate.py schema:
['facility', 'group', 'origin_date', 'horizon', 'target_date', 'target', 'prediction'].
"""

import datetime
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from donorcast.config import (
    DATA_PROCESSED_DIR,
    EVAL_ORIGIN_WEEKDAY,
    HORIZON,
    REPORTS_DIR,
    VAL_END,
    VAL_START,
)
from donorcast.evaluate import evaluate


class BaselinePrecomputer:
    """Precomputes matrix indexing for high-speed baseline forecasting."""

    def __init__(self, long_df: pd.DataFrame):
        self.long_df = long_df.copy()
        self.facilities = sorted(long_df["facility"].unique())
        self.groups = sorted(long_df["group"].unique())
        self.all_dates = sorted(long_df["date"].unique())
        self.date_to_idx = {d: i for i, d in enumerate(self.all_dates)}
        self.n_dates = len(self.all_dates)

        # Pivot donations: date x (facility, group)
        piv = (
            self.long_df.pivot(index="date", columns=["facility", "group"], values="donations")
            .reindex(self.all_dates)
            .fillna(0)
        )
        self.donations_matrix = piv.values.astype(np.float32)
        self.series_cols = list(piv.columns)  # list of (facility, group)
        self.facility_arr = np.array([s[0] for s in self.series_cols])
        self.group_arr = np.array([s[1] for s in self.series_cols])
        self.n_series = len(self.series_cols)


def predict_seasonal_naive(
    long_df: pd.DataFrame,
    origin_dates: Sequence[str],
    horizon: int = HORIZON,
    precomputer: BaselinePrecomputer | None = None,
) -> pd.DataFrame:
    """M0 Baseline: Seasonal Naive.

    For origin t and horizon h (target date t+h with weekday w):
    Forecast is the actual donations on the most recent date <= t having weekday w.
    """
    if precomputer is None:
        precomputer = BaselinePrecomputer(long_df)

    valid_origins = [o for o in origin_dates if o in precomputer.date_to_idx]
    if not valid_origins:
        return pd.DataFrame()

    n_series = precomputer.n_series
    total_blocks = len(valid_origins) * horizon
    total_rows = total_blocks * n_series

    out_facility = np.tile(precomputer.facility_arr, total_blocks)
    out_group = np.tile(precomputer.group_arr, total_blocks)
    out_origin_date = np.empty(total_rows, dtype=object)
    out_target_date = np.empty(total_rows, dtype=object)
    out_horizon = np.empty(total_rows, dtype=np.int16)
    out_target = np.empty(total_rows, dtype=np.float32)
    out_prediction = np.empty(total_rows, dtype=np.float32)

    block_idx = 0
    for origin in valid_origins:
        t_idx = precomputer.date_to_idx[origin]
        origin_dt = datetime.date.fromisoformat(origin)

        for h in range(1, horizon + 1):
            target_dt = origin_dt + datetime.timedelta(days=h)
            target_str = target_dt.isoformat()
            target_weekday = target_dt.weekday()

            # Most recent same weekday <= t
            # Difference in days from origin to target weekday looking backwards
            offset = (origin_dt.weekday() - target_weekday) % 7
            lag_idx = t_idx - offset

            if lag_idx >= 0:
                pred_vals = precomputer.donations_matrix[lag_idx]
            else:
                pred_vals = np.zeros(n_series, dtype=np.float32)

            if target_str in precomputer.date_to_idx:
                tgt_idx = precomputer.date_to_idx[target_str]
                tgt_vals = precomputer.donations_matrix[tgt_idx]
            else:
                tgt_vals = np.full(n_series, np.nan, dtype=np.float32)

            start = block_idx * n_series
            end = start + n_series

            out_origin_date[start:end] = origin
            out_target_date[start:end] = target_str
            out_horizon[start:end] = h
            out_target[start:end] = tgt_vals
            out_prediction[start:end] = pred_vals

            block_idx += 1

    df = pd.DataFrame(
        {
            "facility": out_facility,
            "group": out_group,
            "origin_date": out_origin_date,
            "horizon": out_horizon,
            "target_date": out_target_date,
            "target": out_target,
            "prediction": out_prediction,
        }
    )
    return df.dropna(subset=["target"]).copy()


def predict_weekday_moving_average(
    long_df: pd.DataFrame,
    origin_dates: Sequence[str],
    horizon: int = HORIZON,
    window: int = 4,
    precomputer: BaselinePrecomputer | None = None,
) -> pd.DataFrame:
    """M0b Baseline: Weekday Moving Average.

    For origin t and horizon h (target date t+h with weekday w):
    Forecast is the arithmetic mean of donations across the last `window` (default 4)
    occurrences of weekday w on or before t.
    """
    if precomputer is None:
        precomputer = BaselinePrecomputer(long_df)

    valid_origins = [o for o in origin_dates if o in precomputer.date_to_idx]
    if not valid_origins:
        return pd.DataFrame()

    n_series = precomputer.n_series
    total_blocks = len(valid_origins) * horizon
    total_rows = total_blocks * n_series

    out_facility = np.tile(precomputer.facility_arr, total_blocks)
    out_group = np.tile(precomputer.group_arr, total_blocks)
    out_origin_date = np.empty(total_rows, dtype=object)
    out_target_date = np.empty(total_rows, dtype=object)
    out_horizon = np.empty(total_rows, dtype=np.int16)
    out_target = np.empty(total_rows, dtype=np.float32)
    out_prediction = np.empty(total_rows, dtype=np.float32)

    block_idx = 0
    for origin in valid_origins:
        t_idx = precomputer.date_to_idx[origin]
        origin_dt = datetime.date.fromisoformat(origin)

        # Precompute mean across last `window` occurrences for each weekday (0..6)
        weekday_means = {}
        for w in range(7):
            offset = (origin_dt.weekday() - w) % 7
            lag_indices = [t_idx - offset - 7 * k for k in range(window)]
            valid_lag_indices = [idx for idx in lag_indices if idx >= 0]
            if valid_lag_indices:
                weekday_means[w] = np.mean(
                    precomputer.donations_matrix[valid_lag_indices], axis=0
                ).astype(np.float32)
            else:
                weekday_means[w] = np.zeros(n_series, dtype=np.float32)

        for h in range(1, horizon + 1):
            target_dt = origin_dt + datetime.timedelta(days=h)
            target_str = target_dt.isoformat()
            target_weekday = target_dt.weekday()

            pred_vals = weekday_means[target_weekday]

            if target_str in precomputer.date_to_idx:
                tgt_idx = precomputer.date_to_idx[target_str]
                tgt_vals = precomputer.donations_matrix[tgt_idx]
            else:
                tgt_vals = np.full(n_series, np.nan, dtype=np.float32)

            start = block_idx * n_series
            end = start + n_series

            out_origin_date[start:end] = origin
            out_target_date[start:end] = target_str
            out_horizon[start:end] = h
            out_target[start:end] = tgt_vals
            out_prediction[start:end] = pred_vals

            block_idx += 1

    df = pd.DataFrame(
        {
            "facility": out_facility,
            "group": out_group,
            "origin_date": out_origin_date,
            "horizon": out_horizon,
            "target_date": out_target_date,
            "target": out_target,
            "prediction": out_prediction,
        }
    )
    return df.dropna(subset=["target"]).copy()


def get_validation_origins(
    val_start: str = VAL_START,
    val_end: str = VAL_END,
    weekday: int = EVAL_ORIGIN_WEEKDAY,
) -> list[str]:
    """Generate evaluation origin dates on the specified weekday within the validation range."""
    all_dates = pd.date_range(start=val_start, end=val_end, freq="D")
    eval_dates = [d.strftime("%Y-%m-%d") for d in all_dates if d.weekday() == weekday]
    return eval_dates


def generate_baseline_comparison_markdown(
    results_m0: dict[str, Any],
    results_m0b: dict[str, Any],
    split: str = "val",
) -> str:
    """Format comparative baseline evaluation report in markdown."""
    m0_ov = results_m0["overall"]
    m0b_ov = results_m0b["overall"]
    m0_sf = results_m0["shortfall"]
    m0b_sf = results_m0b["shortfall"]

    md = [
        f"# Baseline Models Evaluation Report ({split.upper()} Split)",
        "",
        f"- **Evaluation Period**: `{VAL_START}` to `{VAL_END}` (Validation Split)",
        "- **Evaluation Frequency**: Weekly (every Monday origin)",
        "- **Forecast Horizons**: 1 to 14 days ahead",
        "- **Primary Metric**: **WAPE_7D** (7-day cumulative sum error)",
        "- **Models Evaluated**:",
        "  - `M0`: **Seasonal Naive** (most recent same weekday $\\le t$)",
        "  - `M0b`: **Weekday Moving Average** (mean of last 4 same weekdays $\\le t$)",
        "",
        "---",
        "",
        "## 1. Headline Regression Metrics Comparison",
        "",
        "| Model | Description | WAPE_7D (Primary) | Daily WAPE | MASE | Rows Evaluated |",
        "|---|---|---|---|---|---|",
        f"| **M0** | Seasonal Naive (last week) | **{m0_ov['wape_7d'] * 100:.1f}%** | **{m0_ov['wape'] * 100:.1f}%** | **{m0_ov['mase']:.2f}** | {m0_ov['count']:,} |",
        f"| **M0b** | Weekday Moving Avg (4-wk) | **{m0b_ov['wape_7d'] * 100:.1f}%** | **{m0b_ov['wape'] * 100:.1f}%** | **{m0b_ov['mase']:.2f}** | {m0b_ov['count']:,} |",
        "",
        (
            f"> **Winner on Validation**: **`{'M0b' if m0b_ov['wape_7d'] < m0_ov['wape_7d'] else 'M0'}`** "
            f"(Lower WAPE_7D: {min(m0_ov['wape_7d'], m0b_ov['wape_7d']) * 100:.1f}% vs {max(m0_ov['wape_7d'], m0b_ov['wape_7d']) * 100:.1f}%)"
        ),
        "",
        "---",
        "",
        "## 2. Shortfall Classification Metrics (7-Day Ahead)",
        "",
        f"- **Actual Shortfall Prevalence**: **{m0_sf['prevalence'] * 100:.1f}%** ({m0_sf['actual_shortfalls']:,} of {m0_sf['total_windows']:,} windows)",
        "",
        "| Model | Prevalence | Precision | Recall | F1 Score | TP | FP | FN | TN | Total Windows |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| **M0** | {m0_sf['prevalence'] * 100:.1f}% | {m0_sf['precision'] * 100:.1f}% | {m0_sf['recall'] * 100:.1f}% | {m0_sf['f1']:.2f} | {m0_sf['tp']} | {m0_sf['fp']} | {m0_sf['fn']} | {m0_sf['tn']} | {m0_sf['total_windows']:,} |",
        f"| **M0b** | {m0b_sf['prevalence'] * 100:.1f}% | {m0b_sf['precision'] * 100:.1f}% | {m0b_sf['recall'] * 100:.1f}% | {m0b_sf['f1']:.2f} | {m0b_sf['tp']} | {m0b_sf['fp']} | {m0b_sf['fn']} | {m0b_sf['tn']} | {m0b_sf['total_windows']:,} |",
        "",
        "### Shortfall Metrics by Facility Tier",
        "",
        "| Tier | Model | Prevalence | Precision | Recall | F1 Score | Windows |",
        "|---|---|---|---|---|---|---|",
    ]

    tier_desc = {
        "top_5": "Top 5 High-Volume Sites",
        "middle": "Middle 10 Sites",
        "bottom_7": "Bottom 7 Small Sites",
    }
    for tier in ["top_5", "middle", "bottom_7"]:
        m0_t_sf = m0_sf["by_tier"][tier]
        m0b_t_sf = m0b_sf["by_tier"][tier]
        md.append(
            f"| **{tier}** ({tier_desc[tier]}) | M0 | {m0_t_sf['prevalence'] * 100:.1f}% | {m0_t_sf['precision'] * 100:.1f}% | {m0_t_sf['recall'] * 100:.1f}% | {m0_t_sf['f1']:.2f} | {m0_t_sf['total_windows']:,} |"
        )
        md.append(
            f"| | **M0b** | {m0b_t_sf['prevalence'] * 100:.1f}% | {m0b_t_sf['precision'] * 100:.1f}% | {m0b_t_sf['recall'] * 100:.1f}% | {m0b_t_sf['f1']:.2f} | {m0b_t_sf['total_windows']:,} |"
        )

    md.extend(
        [
            "",
            "---",
            "",
            "## 3. Breakdown by Horizon (h = 1..14)",
            "",
            "| Horizon | M0 WAPE | M0 MASE | M0b WAPE | M0b MASE | Better Model |",
            "|---|---|---|---|---|---|",
        ]
    )

    all_horizons = sorted(results_m0["by_horizon"].keys())
    for h in all_horizons:
        m0_h = results_m0["by_horizon"][h]
        m0b_h = results_m0b["by_horizon"][h]
        better = "M0b" if m0b_h["wape"] < m0_h["wape"] else "M0"
        md.append(
            f"| Day {h:02d} | {m0_h['wape'] * 100:.1f}% | {m0_h['mase']:.2f} | {m0b_h['wape'] * 100:.1f}% | {m0b_h['mase']:.2f} | **{better}** |"
        )

    md.extend(
        [
            "",
            "---",
            "",
            "## 4. Breakdown by Blood Group",
            "",
            "| Group | M0 WAPE_7D | M0 Daily WAPE | M0 MASE | M0b WAPE_7D | M0b Daily WAPE | M0b MASE | Better (7D) |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )

    for grp in ["A", "B", "O", "AB"]:
        if grp in results_m0["by_group"] and grp in results_m0b["by_group"]:
            m0_g = results_m0["by_group"][grp]
            m0b_g = results_m0b["by_group"][grp]
            better = "M0b" if m0b_g["wape_7d"] < m0_g["wape_7d"] else "M0"
            md.append(
                f"| **{grp}** | {m0_g['wape_7d'] * 100:.1f}% | {m0_g['wape'] * 100:.1f}% | {m0_g['mase']:.2f} | {m0b_g['wape_7d'] * 100:.1f}% | {m0b_g['wape'] * 100:.1f}% | {m0b_g['mase']:.2f} | **{better}** |"
            )

    md.extend(
        [
            "",
            "---",
            "",
            "## 5. Breakdown by Facility Tier",
            "",
            "| Tier | Description | M0 WAPE_7D | M0 Daily WAPE | M0 MASE | M0b WAPE_7D | M0b Daily WAPE | M0b MASE | Better (7D) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )

    for tier in ["top_5", "middle", "bottom_7"]:
        m0_t = results_m0["by_tier"][tier]
        m0b_t = results_m0b["by_tier"][tier]
        better = "M0b" if m0b_t["wape_7d"] < m0_t["wape_7d"] else "M0"
        md.append(
            f"| **{tier}** | {tier_desc[tier]} | {m0_t['wape_7d'] * 100:.1f}% | {m0_t['wape'] * 100:.1f}% | {m0_t['mase']:.2f} | {m0b_t['wape_7d'] * 100:.1f}% | {m0b_t['wape'] * 100:.1f}% | {m0b_t['mase']:.2f} | **{better}** |"
        )

    md.extend(
        [
            "",
            "---",
            "",
            "## 6. Holiday Window Performance",
            "",
            "| Window | M0 WAPE_7D | M0 Daily WAPE | M0 MASE | M0b WAPE_7D | M0b Daily WAPE | M0b MASE | Better (7D) |",
            "|---|---|---|---|---|---|---|---|",
            f"| **Public Holiday** | {results_m0['by_holiday']['holiday']['wape_7d'] * 100:.1f}% | {results_m0['by_holiday']['holiday']['wape'] * 100:.1f}% | {results_m0['by_holiday']['holiday']['mase']:.2f} | {results_m0b['by_holiday']['holiday']['wape_7d'] * 100:.1f}% | {results_m0b['by_holiday']['holiday']['wape'] * 100:.1f}% | {results_m0b['by_holiday']['holiday']['mase']:.2f} | **{'M0b' if results_m0b['by_holiday']['holiday']['wape_7d'] < results_m0['by_holiday']['holiday']['wape_7d'] else 'M0'}** |",
            f"| **Non-Holiday** | {results_m0['by_holiday']['non_holiday']['wape_7d'] * 100:.1f}% | {results_m0['by_holiday']['non_holiday']['wape'] * 100:.1f}% | {results_m0['by_holiday']['non_holiday']['mase']:.2f} | {results_m0b['by_holiday']['non_holiday']['wape_7d'] * 100:.1f}% | {results_m0b['by_holiday']['non_holiday']['wape'] * 100:.1f}% | {results_m0b['by_holiday']['non_holiday']['mase']:.2f} | **{'M0b' if results_m0b['by_holiday']['non_holiday']['wape_7d'] < results_m0['by_holiday']['non_holiday']['wape_7d'] else 'M0'}** |",
            "",
        ]
    )

    return "\n".join(md)


def run_baselines_evaluation(
    split: str = "val",
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Generate predictions for M0 and M0b on the validation split and evaluate them."""
    long_path = DATA_PROCESSED_DIR / "long.parquet"
    if not long_path.exists():
        from donorcast.clean import clean_data

        clean_data()
    long_df = pd.read_parquet(long_path)

    origin_dates = get_validation_origins()
    precomputer = BaselinePrecomputer(long_df)

    print(
        f"Generating predictions for {len(origin_dates)} origin dates ({origin_dates[0]} to {origin_dates[-1]})..."
    )

    # M0
    print("Running M0 (Seasonal Naive)...")
    m0_preds = predict_seasonal_naive(
        long_df=long_df,
        origin_dates=origin_dates,
        horizon=HORIZON,
        precomputer=precomputer,
    )
    print(f"Evaluating M0 on {split} split...")
    results_m0 = evaluate(
        predictions_df=m0_preds,
        split=split,
        model_name="M0_seasonal_naive",
        reports_dir=reports_dir,
    )

    # M0b
    print("Running M0b (Weekday Moving Average)...")
    m0b_preds = predict_weekday_moving_average(
        long_df=long_df,
        origin_dates=origin_dates,
        horizon=HORIZON,
        window=4,
        precomputer=precomputer,
    )
    print(f"Evaluating M0b on {split} split...")
    results_m0b = evaluate(
        predictions_df=m0b_preds,
        split=split,
        model_name="M0b_weekday_ma4",
        reports_dir=reports_dir,
    )

    # Combined summary report
    summary_md = generate_baseline_comparison_markdown(results_m0, results_m0b, split=split)
    summary_file = reports_dir / "results_baselines.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print("\n=======================================================")
    print(f"Baseline Evaluation Complete! Saved report to: {summary_file}")
    print(
        f"M0  WAPE: {results_m0['overall']['wape']:.4%} | MASE: {results_m0['overall']['mase']:.4f}"
    )
    print(
        f"M0b WAPE: {results_m0b['overall']['wape']:.4%} | MASE: {results_m0b['overall']['mase']:.4f}"
    )
    print("=======================================================\n")

    return {
        "m0": results_m0,
        "m0b": results_m0b,
        "summary_file": str(summary_file),
    }
