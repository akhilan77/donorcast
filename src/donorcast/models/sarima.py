"""SARIMAX forecasting model for DonorCast.

Implements M1:
- One SARIMAX per (facility, group) series.
- Weekly seasonality (s = 7).
- Exogenous variables for facility's state: is_public_holiday, is_school_holiday, is_ramadan, is_mco.
- Fitted ONCE per series on the last 3 years of training data (2020-01-01 to 2022-12-31).
- For each validation origin t, updates model state with new observations without re-estimating
  parameters (using statsmodels `extend(..., refit=False)`) and forecasts 14 steps.
- Parallelised across series with joblib.
- Evaluated on validation split via evaluate.py.
"""

import datetime
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from donorcast.calendar import (
    build_calendar_dataframe,
    load_facility_state_mapping,
)
from donorcast.config import (
    CALENDAR_PROCESSED_FILE,
    DATA_PROCESSED_DIR,
    FACILITY_STATE_FILE,
    HORIZON,
    MASE_WINDOW_START,
    REPORTS_DIR,
    TRAIN_END,
    VAL_END,
    VAL_START,
)
from donorcast.evaluate import evaluate
from donorcast.models.baselines import get_validation_origins

# Exogenous feature columns used in SARIMAX
EXOG_COLS = ["is_public_holiday", "is_school_holiday", "is_ramadan", "is_mco"]

# Representative 5 series for order selection grid
REPRESENTATIVE_SERIES = [
    ("Pusat Darah Negara", "O"),  # Tier 1 (top), Major blood group
    ("Hospital Melaka", "A"),  # Tier 1/2, Group A
    ("Hospital Sultanah Bahiyah", "B"),  # Tier 2 (middle), Group B
    ("Hospital Duchess Of Kent", "O"),  # Tier 3 (bottom/small), Group O
    ("Hospital Queen Elizabeth II", "AB"),  # Tier 3, Rare blood group AB
]

# Candidate (order, seasonal_order) configs for grid search
CANDIDATE_ORDERS = [
    ((1, 0, 1), (1, 0, 1, 7)),
    ((1, 0, 0), (1, 0, 0, 7)),
    ((2, 0, 1), (1, 0, 1, 7)),
    ((1, 1, 1), (0, 1, 1, 7)),
    ((1, 0, 1), (0, 1, 1, 7)),
    ((2, 0, 0), (1, 0, 0, 7)),
]

# Default chosen order after empirical grid selection
CHOSEN_ORDER = (1, 0, 1)
CHOSEN_SEASONAL_ORDER = (1, 0, 1, 7)


def prepare_data_and_exog(
    long_df: pd.DataFrame,
    calendar_df: pd.DataFrame | None = None,
    facility_state_df: pd.DataFrame | None = None,
) -> tuple[
    dict[tuple[str, str], pd.Series],
    dict[str, pd.DataFrame],
    dict[str, str],
    list[str],
]:
    """Prepare series dictionary and facility-level exogenous DataFrames.

    Returns
    -------
    series_dict : dict[tuple[facility, group], pd.Series]
        Daily donations indexed by date string.
    facility_exog_dict : dict[facility, pd.DataFrame]
        Exogenous calendar variables indexed by date string for each facility's state.
    facility_to_state : dict[facility, state]
    all_dates : list[str]
        Sorted list of all calendar dates available.
    """
    if facility_state_df is None:
        facility_state_df = load_facility_state_mapping(FACILITY_STATE_FILE)
    fac_col = "facility" if "facility" in facility_state_df.columns else "hospital"
    facility_to_state = dict(zip(facility_state_df[fac_col], facility_state_df["state"]))

    if calendar_df is None:
        if CALENDAR_PROCESSED_FILE.exists():
            calendar_df = pd.read_parquet(CALENDAR_PROCESSED_FILE)
        else:
            calendar_df = build_calendar_dataframe()

    all_dates = sorted(calendar_df["date"].unique())

    # Convert all_dates to pd.DatetimeIndex with freq='D'
    dt_index = pd.date_range(start=all_dates[0], end=all_dates[-1], freq="D")
    str_dates = [d.strftime("%Y-%m-%d") for d in dt_index]

    # Build facility-specific exog DataFrames
    facility_exog_dict: dict[str, pd.DataFrame] = {}
    for fac, st in facility_to_state.items():
        st_cal = (
            calendar_df[calendar_df["state"] == st]
            .set_index("date")[EXOG_COLS]
            .reindex(str_dates)
            .fillna(0)
            .astype(float)
        )
        st_cal.index = dt_index
        facility_exog_dict[fac] = st_cal

    # Build series dictionary
    series_dict: dict[tuple[str, str], pd.Series] = {}
    grouped = long_df.groupby(["facility", "group"], observed=False)
    for (fac, grp), grp_df in grouped:
        s = grp_df.set_index("date")["donations"].reindex(str_dates).fillna(0.0).astype(float)
        s.index = dt_index
        series_dict[(str(fac), str(grp))] = s

    return series_dict, facility_exog_dict, facility_to_state, str_dates


def fit_and_forecast_single_series(
    facility: str,
    group: str,
    series: pd.Series,
    exog_df: pd.DataFrame,
    origin_dates: list[str],
    train_start: str = MASE_WINDOW_START,
    train_end: str = TRAIN_END,
    horizon: int = HORIZON,
    order: tuple[int, int, int] = CHOSEN_ORDER,
    seasonal_order: tuple[int, int, int, int] = CHOSEN_SEASONAL_ORDER,
) -> pd.DataFrame:
    """Fit SARIMAX on train data for one series, and forecast across validation origins via extend."""
    # Slice initial training data (e.g. 2020-01-01 to 2022-12-31)
    y_train = series.loc[train_start:train_end]
    X_train = exog_df.loc[train_start:train_end]

    rows: list[dict[str, Any]] = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            model = SARIMAX(
                y_train,
                exog=X_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False, maxiter=100)
        except Exception:
            # Fallback to simpler AR(1) order if optimization fails
            try:
                model = SARIMAX(
                    y_train,
                    exog=X_train,
                    order=(1, 0, 0),
                    seasonal_order=(1, 0, 0, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False, maxiter=100)
            except Exception:
                # Ultimate fallback to naive mean
                res = None

    last_observed_date = train_end
    current_res = res

    for origin in origin_dates:
        t_ts = pd.Timestamp(origin)

        # If origin > last_observed_date and we have a valid model result, extend state with new observations
        if current_res is not None and origin > last_observed_date:
            last_ts = pd.Timestamp(last_observed_date)
            # Slice observations between last_observed_date (exclusive) and origin (inclusive)
            next_start_ts = last_ts + pd.Timedelta(days=1)
            if next_start_ts <= t_ts:
                y_new = series.loc[next_start_ts:t_ts]
                X_new = exog_df.loc[next_start_ts:t_ts]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    current_res = current_res.extend(
                        endog=y_new,
                        exog=X_new,
                    )
            last_observed_date = origin

        # Exogenous inputs for future 14 days
        future_ts = pd.date_range(start=t_ts + pd.Timedelta(days=1), periods=horizon, freq="D")
        X_future = exog_df.reindex(future_ts).fillna(0.0)

        # Forecast 14 steps ahead
        if current_res is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    forecast_vals = current_res.forecast(steps=horizon, exog=X_future).values
                    # Clip to non-negative
                    forecast_vals = np.maximum(0.0, forecast_vals)
                except (ValueError, np.linalg.LinAlgError, RuntimeError):
                    # Fallback to mean of last 28 days
                    recent_slice = series.loc[max(series.index[0], t_ts - pd.Timedelta(days=27)) : t_ts]
                    forecast_vals = np.full(horizon, max(0.0, float(recent_slice.mean())))
        else:
            recent_slice = series.loc[max(series.index[0], t_ts - pd.Timedelta(days=27)) : t_ts]
            forecast_vals = np.full(horizon, max(0.0, float(recent_slice.mean())))

        for h, (tgt_ts, pred_val) in enumerate(zip(future_ts, forecast_vals), start=1):
            tgt_date = tgt_ts.strftime("%Y-%m-%d")
            if tgt_ts in series.index:
                tgt_val = float(series.loc[tgt_ts])
            else:
                tgt_val = np.nan

            rows.append(
                {
                    "facility": facility,
                    "group": group,
                    "origin_date": origin,
                    "horizon": h,
                    "target_date": tgt_date,
                    "target": tgt_val,
                    "prediction": float(pred_val),
                }
            )

    return pd.DataFrame(rows)


def run_order_selection_grid(
    series_dict: dict[tuple[str, str], pd.Series],
    facility_exog_dict: dict[str, pd.DataFrame],
    sample_series: list[tuple[str, str]] = REPRESENTATIVE_SERIES,
    candidate_orders: list[
        tuple[tuple[int, int, int], tuple[int, int, int, int]]
    ] = CANDIDATE_ORDERS,
    train_start: str = MASE_WINDOW_START,
    train_end: str = TRAIN_END,
) -> dict[str, Any]:
    """Run grid search over candidate orders on representative series."""
    grid_results = []

    for order, s_order in candidate_orders:
        aics = []
        bics = []
        fit_times = []
        converged_count = 0

        for fac, grp in sample_series:
            s = series_dict.get((fac, grp))
            if s is None:
                continue
            exog = facility_exog_dict[fac]
            y_tr = s.loc[train_start:train_end]
            X_tr = exog.loc[train_start:train_end]

            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    mod = SARIMAX(
                        y_tr,
                        exog=X_tr,
                        order=order,
                        seasonal_order=s_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = mod.fit(disp=False, maxiter=50)
                    aics.append(res.aic)
                    bics.append(res.bic)
                    converged_count += 1
                except Exception:
                    pass
            t1 = time.perf_counter()
            fit_times.append(t1 - t0)

        mean_aic = float(np.mean(aics)) if aics else float("inf")
        mean_bic = float(np.mean(bics)) if bics else float("inf")
        mean_time = float(np.mean(fit_times)) if fit_times else 0.0

        grid_results.append(
            {
                "order": order,
                "seasonal_order": s_order,
                "order_str": f"SARIMAX{order}x{s_order}",
                "mean_aic": mean_aic,
                "mean_bic": mean_bic,
                "mean_fit_time_sec": mean_time,
                "converged": f"{converged_count}/{len(sample_series)}",
            }
        )

    grid_df = pd.DataFrame(grid_results).sort_values("mean_aic")
    best_row = grid_df.iloc[0]

    return {
        "grid_df": grid_df,
        "best_order": best_row["order"],
        "best_seasonal_order": best_row["seasonal_order"],
    }


def generate_sarima_comparison_markdown(
    results_sarima: dict[str, Any],
    results_m0: dict[str, Any] | None = None,
    results_m0b: dict[str, Any] | None = None,
    runtime_sec: float = 0.0,
    grid_summary: pd.DataFrame | None = None,
    split: str = "val",
) -> str:
    """Format SARIMAX evaluation and baseline comparison markdown report."""
    sarima_ov = results_sarima["overall"]
    sarima_sf = results_sarima["shortfall"]

    md = [
        f"# SARIMAX (M1) Model Evaluation Report ({split.upper()} Split)",
        "",
        "- **Model**: `M1_sarimax` (SARIMAX with weekly seasonality $s=7$ and 4 exogenous calendar dummies)",
        f"- **Order Specification**: `SARIMAX{CHOSEN_ORDER}x{CHOSEN_SEASONAL_ORDER}`",
        f"- **Exogenous Variables**: `{', '.join(EXOG_COLS)}` (Facility's State)",
        f"- **Estimation Strategy**: Fit once on last 3 years of train (`{MASE_WINDOW_START}` to `{TRAIN_END}`), rolling state update via `extend(refit=False)` across all validation origins",
        f"- **Total Runtime (Fit + State Updating + 14D Forecast)**: **{runtime_sec:.2f} seconds** ({runtime_sec / 60:.2f} minutes)",
        f"- **Evaluation Period**: `{VAL_START}` to `{VAL_END}` (104 weekly Monday origins)",
        "- **Forecast Horizons**: 1 to 14 days ahead",
        "",
        "---",
        "",
    ]

    if grid_summary is not None:
        md.extend(
            [
                "## 1. Order Selection Grid Search (5 Representative Series)",
                "",
                "Evaluated across representative facilities (Pusat Darah Negara, Melaka, Sultanah Bahiyah, Duchess of Kent, Queen Elizabeth II):",
                "",
                "| Order | Seasonal Order | Mean AIC | Mean BIC | Avg Fit Time (s) | Converged |",
                "|---|---|---|---|---|---|",
            ]
        )
        for _, row in grid_summary.iterrows():
            md.append(
                f"| `{row['order']}` | `{row['seasonal_order']}` | {row['mean_aic']:.1f} | {row['mean_bic']:.1f} | {row['mean_fit_time_sec']:.2f}s | {row['converged']} |"
            )
        md.extend(["", "---", ""])

    md.extend(
        [
            "## 2. Headline Regression Metrics Comparison",
            "",
            "| Model | Description | WAPE_7D (Primary) | Daily WAPE | MASE | Rows Evaluated |",
            "|---|---|---|---|---|---|",
        ]
    )

    if results_m0 is not None:
        m0_ov = results_m0["overall"]
        md.append(
            f"| **M0** | Seasonal Naive (last week) | {m0_ov['wape_7d'] * 100:.1f}% | {m0_ov['wape'] * 100:.1f}% | {m0_ov['mase']:.2f} | {m0_ov['count']:,} |"
        )
    if results_m0b is not None:
        m0b_ov = results_m0b["overall"]
        md.append(
            f"| **M0b** | Weekday Moving Avg (4-wk) | {m0b_ov['wape_7d'] * 100:.1f}% | {m0b_ov['wape'] * 100:.1f}% | {m0b_ov['mase']:.2f} | {m0b_ov['count']:,} |"
        )

    md.append(
        f"| **M1 (SARIMAX)** | SARIMAX(1,0,1)x(1,0,1)7 + Exog | **{sarima_ov['wape_7d'] * 100:.1f}%** | **{sarima_ov['wape'] * 100:.1f}%** | **{sarima_ov['mase']:.2f}** | {sarima_ov['count']:,} |"
    )

    if results_m0b is not None:
        diff_7d = (sarima_ov["wape_7d"] - m0b_ov["wape_7d"]) * 100
        better_text = (
            f"beats M0b by {abs(diff_7d):.1f}%"
            if diff_7d < 0
            else f"is {diff_7d:.1f}% behind M0b baseline"
        )
        md.extend(
            [
                "",
                f"> **Validation Comparison vs M0b Baseline**: SARIMAX WAPE_7D is **{sarima_ov['wape_7d'] * 100:.1f}%** vs M0b **{m0b_ov['wape_7d'] * 100:.1f}%** ({better_text}).",
            ]
        )

    md.extend(
        [
            "",
            "---",
            "",
            "## 3. Shortfall Classification Metrics (7-Day Ahead)",
            "",
            f"- **Actual Shortfall Prevalence**: **{sarima_sf['prevalence'] * 100:.1f}%** ({sarima_sf['actual_shortfalls']:,} of {sarima_sf['total_windows']:,} windows)",
            "",
            "| Model | Prevalence | Precision | Recall | F1 Score | TP | FP | FN | TN |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )

    if results_m0 is not None:
        m0_sf = results_m0["shortfall"]
        md.append(
            f"| **M0** | {m0_sf['prevalence'] * 100:.1f}% | {m0_sf['precision'] * 100:.1f}% | {m0_sf['recall'] * 100:.1f}% | {m0_sf['f1']:.2f} | {m0_sf['tp']} | {m0_sf['fp']} | {m0_sf['fn']} | {m0_sf['tn']} |"
        )
    if results_m0b is not None:
        m0b_sf = results_m0b["shortfall"]
        md.append(
            f"| **M0b** | {m0b_sf['prevalence'] * 100:.1f}% | {m0b_sf['precision'] * 100:.1f}% | {m0b_sf['recall'] * 100:.1f}% | {m0b_sf['f1']:.2f} | {m0b_sf['tp']} | {m0b_sf['fp']} | {m0b_sf['fn']} | {m0b_sf['tn']} |"
        )

    md.append(
        f"| **M1 (SARIMAX)** | {sarima_sf['prevalence'] * 100:.1f}% | {sarima_sf['precision'] * 100:.1f}% | {sarima_sf['recall'] * 100:.1f}% | {sarima_sf['f1']:.2f} | {sarima_sf['tp']} | {sarima_sf['fp']} | {sarima_sf['fn']} | {sarima_sf['tn']} |"
    )

    md.extend(
        [
            "",
            "### Shortfall Metrics by Facility Tier",
            "",
            "| Tier | Model | Precision | Recall | F1 Score | Total Windows |",
            "|---|---|---|---|---|---|",
        ]
    )
    tier_desc = {
        "top_5": "Top 5 High-Volume Sites",
        "middle": "Middle 10 Sites",
        "bottom_7": "Bottom 7 Small Sites",
    }
    for tier in ["top_5", "middle", "bottom_7"]:
        s_t = sarima_sf["by_tier"][tier]
        md.append(
            f"| **{tier}** ({tier_desc[tier]}) | M1 (SARIMAX) | {s_t['precision'] * 100:.1f}% | {s_t['recall'] * 100:.1f}% | {s_t['f1']:.2f} | {s_t['total_windows']:,} |"
        )

    md.extend(
        [
            "",
            "---",
            "",
            "## 4. Breakdown by Horizon (h = 1..14)",
            "",
            "| Horizon | SARIMAX WAPE | SARIMAX MASE |",
            "|---|---|---|",
        ]
    )
    for h in sorted(results_sarima["by_horizon"].keys()):
        h_res = results_sarima["by_horizon"][h]
        md.append(f"| Day {h:02d} | {h_res['wape'] * 100:.1f}% | {h_res['mase']:.2f} |")

    md.extend(
        [
            "",
            "---",
            "",
            "## 5. Breakdown by Blood Group",
            "",
            "| Group | SARIMAX WAPE_7D | SARIMAX Daily WAPE | SARIMAX MASE |",
            "|---|---|---|---|",
        ]
    )
    for grp in ["A", "B", "O", "AB"]:
        if grp in results_sarima["by_group"]:
            g_res = results_sarima["by_group"][grp]
            md.append(
                f"| **{grp}** | {g_res['wape_7d'] * 100:.1f}% | {g_res['wape'] * 100:.1f}% | {g_res['mase']:.2f} |"
            )

    md.extend(
        [
            "",
            "---",
            "",
            "## 6. Breakdown by Facility Tier",
            "",
            "| Tier | Description | SARIMAX WAPE_7D | SARIMAX Daily WAPE | SARIMAX MASE |",
            "|---|---|---|---|---|",
        ]
    )
    for tier in ["top_5", "middle", "bottom_7"]:
        t_res = results_sarima["by_tier"][tier]
        md.append(
            f"| **{tier}** | {tier_desc[tier]} | {t_res['wape_7d'] * 100:.1f}% | {t_res['wape'] * 100:.1f}% | {t_res['mase']:.2f} |"
        )

    md.extend(
        [
            "",
            "---",
            "",
            "## 7. Holiday Window Performance",
            "",
            "| Window | SARIMAX WAPE_7D | SARIMAX Daily WAPE | SARIMAX MASE |",
            "|---|---|---|---|",
            f"| **Public Holiday** | {results_sarima['by_holiday']['holiday']['wape_7d'] * 100:.1f}% | {results_sarima['by_holiday']['holiday']['wape'] * 100:.1f}% | {results_sarima['by_holiday']['holiday']['mase']:.2f} |",
            f"| **Non-Holiday** | {results_sarima['by_holiday']['non_holiday']['wape_7d'] * 100:.1f}% | {results_sarima['by_holiday']['non_holiday']['wape'] * 100:.1f}% | {results_sarima['by_holiday']['non_holiday']['mase']:.2f} |",
            "",
        ]
    )

    return "\n".join(md)


def run_sarima_evaluation(
    split: str = "val",
    n_jobs: int = -1,
    order: tuple[int, int, int] = CHOSEN_ORDER,
    seasonal_order: tuple[int, int, int, int] = CHOSEN_SEASONAL_ORDER,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Train SARIMAX across all series and evaluate on the validation split."""
    long_path = DATA_PROCESSED_DIR / "long.parquet"
    if not long_path.exists():
        from donorcast.clean import clean_data

        clean_data()
    long_df = pd.read_parquet(long_path)

    print("Preparing series and exogenous features for SARIMAX...")
    series_dict, facility_exog_dict, _facility_to_state, _all_dates = prepare_data_and_exog(
        long_df
    )

    # Optional order selection grid on representative series
    print("\nRunning Order Selection Grid Search on 5 representative series...")
    grid_out = run_order_selection_grid(series_dict, facility_exog_dict)
    grid_df = grid_out["grid_df"]
    print("Grid Search Results:")
    print(grid_df[["order_str", "mean_aic", "mean_bic", "mean_fit_time_sec", "converged"]])

    origin_dates = get_validation_origins(val_start=VAL_START, val_end=VAL_END)
    all_series_keys = sorted(series_dict.keys())

    print(
        f"\nFitting SARIMAX{order}x{seasonal_order} on {len(all_series_keys)} series "
        f"and generating 14-step forecasts for {len(origin_dates)} validation origins (n_jobs={n_jobs})..."
    )

    t_start = time.perf_counter()

    # Parallelize across all 88 (facility, group) series
    delayed_jobs = [
        joblib.delayed(fit_and_forecast_single_series)(
            facility=fac,
            group=grp,
            series=series_dict[(fac, grp)],
            exog_df=facility_exog_dict[fac],
            origin_dates=origin_dates,
            train_start=MASE_WINDOW_START,
            train_end=TRAIN_END,
            horizon=HORIZON,
            order=order,
            seasonal_order=seasonal_order,
        )
        for (fac, grp) in all_series_keys
    ]

    results_dfs = joblib.Parallel(n_jobs=n_jobs, verbose=5)(delayed_jobs)
    t_end = time.perf_counter()
    total_runtime = t_end - t_start

    print(
        f"Forecasting complete in {total_runtime:.2f} seconds ({total_runtime / 60:.2f} minutes)."
    )

    # Concatenate all prediction rows
    predictions_df = pd.concat(results_dfs, ignore_index=True)
    predictions_df = predictions_df.dropna(subset=["target"]).copy()

    print(f"Total prediction rows generated: {len(predictions_df):,}")
    print(f"Evaluating M1_sarimax on {split} split...")

    results_sarima = evaluate(
        predictions_df=predictions_df,
        split=split,
        model_name="M1_sarimax",
        reports_dir=reports_dir,
    )

    # Load baseline results if available for comparison
    from donorcast.models.baselines import (
        predict_seasonal_naive,
        predict_weekday_moving_average,
    )

    m0_preds = predict_seasonal_naive(long_df, origin_dates=origin_dates, horizon=HORIZON)
    results_m0 = evaluate(
        m0_preds, split=split, model_name="M0_seasonal_naive", reports_dir=reports_dir
    )

    m0b_preds = predict_weekday_moving_average(
        long_df, origin_dates=origin_dates, horizon=HORIZON, window=4
    )
    results_m0b = evaluate(
        m0b_preds, split=split, model_name="M0b_weekday_ma4", reports_dir=reports_dir
    )

    # Generate and write report
    report_md = generate_sarima_comparison_markdown(
        results_sarima=results_sarima,
        results_m0=results_m0,
        results_m0b=results_m0b,
        runtime_sec=total_runtime,
        grid_summary=grid_df,
        split=split,
    )

    out_file = reports_dir / "results_sarima_val.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n=======================================================")
    print(f"SARIMAX Evaluation Complete! Report written to: {out_file}")
    print(
        f"SARIMAX WAPE_7D: {results_sarima['overall']['wape_7d']:.2%} | Daily WAPE: {results_sarima['overall']['wape']:.2%} | MASE: {results_sarima['overall']['mase']:.2f}"
    )
    print(
        f"M0b     WAPE_7D: {results_m0b['overall']['wape_7d']:.2%} | Daily WAPE: {results_m0b['overall']['wape']:.2%} | MASE: {results_m0b['overall']['mase']:.2f}"
    )
    print(
        f"Shortfall Recall: SARIMAX {results_sarima['shortfall']['recall']:.2%} vs M0b {results_m0b['shortfall']['recall']:.2%}"
    )
    print("=======================================================\n")

    return {
        "sarima": results_sarima,
        "m0": results_m0,
        "m0b": results_m0b,
        "runtime_sec": total_runtime,
        "report_file": str(out_file),
    }
