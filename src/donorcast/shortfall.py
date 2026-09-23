"""Forecasting and Shortfall Alerting Module for DonorCast (Task 4.3).

Implements:
1. Multi-horizon 14-day forecasts (p10, p50, p90) for all 88 series (facility x group) from any origin.
2. Shortfall detection for the next 7 days using HistoricalTypicalLookup imported from evaluate.py.
3. Severity classification:
   - HIGH if forecast < 0.7 x typical
   - MEDIUM if forecast < 0.8 x typical
4. Top 3 plain-English reasons for each alert using reasons() from explain.py.
5. Saving outputs to outputs/forecasts_<origin>.parquet and outputs/alerts_<origin>.parquet.
6. Precomputing outputs for 8 historical replay origins across the test period.
"""

import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from donorcast.calendar import load_facility_state_mapping
from donorcast.config import (
    CALENDAR_PROCESSED_FILE,
    DATA_CUTOFF,
    DATA_PROCESSED_DIR,
    FACILITY_STATE_FILE,
    HORIZON,
    OUTPUTS_DIR,
    REPLAY_ORIGINS,
    SHORTFALL_HIGH_RATIO,
    SHORTFALL_MEDIUM_RATIO,
)
from donorcast.evaluate import HistoricalTypicalLookup
from donorcast.explain import (
    generate_action_prescription,
    get_latest_lgbm_model_dir,
    reasons,
)
from donorcast.features import FeaturePrecomputer, build_features_for_origins
from donorcast.models.lgbm import CATEGORICAL_COLS


def load_final_models(
    model_dir: Path | None = None,
) -> tuple[lgb.Booster, lgb.Booster, lgb.Booster, list[str]]:
    """Load the final LightGBM point model and quantile models (p10, p90).

    Parameters
    ----------
    model_dir : Path | None
        Directory containing model artifacts. If None, finds the latest version (e.g. models/lgbm/v003).

    Returns
    -------
    tuple[lgb.Booster, lgb.Booster, lgb.Booster, list[str]]
        (point_booster, p10_booster, p90_booster, feature_names)
    """
    target_dir = model_dir if model_dir is not None else get_latest_lgbm_model_dir()

    model_point_file = target_dir / "model.txt"
    model_p10_file = target_dir / "model_p10.txt"
    model_p90_file = target_dir / "model_p90.txt"

    if not model_point_file.exists():
        raise FileNotFoundError(f"Final point model not found: {model_point_file}")
    if not model_p10_file.exists():
        raise FileNotFoundError(f"Final p10 model not found: {model_p10_file}")
    if not model_p90_file.exists():
        raise FileNotFoundError(f"Final p90 model not found: {model_p90_file}")

    b_point = lgb.Booster(model_file=str(model_point_file))
    b_p10 = lgb.Booster(model_file=str(model_p10_file))
    b_p90 = lgb.Booster(model_file=str(model_p90_file))

    feature_names = b_point.feature_name()
    return b_point, b_p10, b_p90, feature_names


def forecast_facility_groups(
    origin_date: str,
    long_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    fac_state_df: pd.DataFrame,
    precomputer: FeaturePrecomputer | None = None,
    models: tuple[lgb.Booster, lgb.Booster, lgb.Booster, list[str]] | None = None,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate 14-day forecasts (p10, p50, p90) for all facility-group pairs from an origin date.

    Parameters
    ----------
    origin_date : str
        Origin date in 'YYYY-MM-DD' format.
    long_df : pd.DataFrame
        Historical cleaned long donation dataset.
    calendar_df : pd.DataFrame
        Precomputed calendar dataset.
    fac_state_df : pd.DataFrame
        Facility to state mapping table.
    precomputer : FeaturePrecomputer | None
        Optional cached feature precomputer.
    models : tuple | None
        Optional preloaded tuple of (b_point, b_p10, b_p90, feature_names).
    horizon : int
        Number of forecast horizons ahead (default: 14).

    Returns
    -------
    forecasts_df : pd.DataFrame
        DataFrame with predictions for all 88 series across horizons 1..14.
    feats_df : pd.DataFrame
        Input feature dataframe for all horizons, used for explanation and inspection.
    """
    if precomputer is None:
        precomputer = FeaturePrecomputer(long_df, fac_state_df)

    if models is None:
        models = load_final_models()

    b_point, b_p10, b_p90, feature_names = models

    # Build features as of origin date strictly using historical data <= origin_date
    feats_df = build_features_for_origins(
        origin_dates=[origin_date],
        long_df=long_df,
        calendar_df=calendar_df,
        facility_state_df=fac_state_df,
        horizon=horizon,
        precomputer=precomputer,
    )

    if len(feats_df) == 0:
        raise ValueError(
            f"Could not build features for origin '{origin_date}'. Origin date may be outside historical data range."
        )

    # Prepare features matrix with categorical dtypes
    X = feats_df[feature_names].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # Generate predictions
    raw_pred = b_point.predict(X)
    raw_p10 = b_p10.predict(X)
    raw_p90 = b_p90.predict(X)

    # Post-processing: non-negativity and quantile monotonicity
    pred_p50 = np.maximum(0.0, raw_pred)
    pred_p10 = np.maximum(0.0, raw_p10)
    pred_p90 = np.maximum(0.0, raw_p90)
    pred_p90 = np.maximum(pred_p90, pred_p10)

    # Assemble forecast dataframe
    forecasts_df = pd.DataFrame(
        {
            "facility": feats_df["facility"].values,
            "group": feats_df["group"].values,
            "origin_date": feats_df["origin_date"].values,
            "horizon": feats_df["horizon"].values.astype(int),
            "target_date": feats_df["target_date"].values,
            "target": feats_df["target"].values,
            "prediction": pred_p50,
            "pred_p50": pred_p50,
            "pred_p10": pred_p10,
            "pred_p90": pred_p90,
        }
    )

    return forecasts_df, feats_df


def compute_alerts_table(
    forecasts_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    long_df: pd.DataFrame,
    shortfall_ratio: float = SHORTFALL_MEDIUM_RATIO,
    shortfall_high_ratio: float = SHORTFALL_HIGH_RATIO,
    typical_lookup: HistoricalTypicalLookup | None = None,
    only_flagged: bool = True,
) -> pd.DataFrame:
    """Evaluate 7-day shortfall alerts and attach top 3 plain-English reasons from explain.py.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Forecast DataFrame containing horizons 1..7.
    feature_df : pd.DataFrame
        Corresponding feature DataFrame for reason extraction.
    long_df : pd.DataFrame
        Processed historical donations DataFrame.
    shortfall_ratio : float
        Ratio threshold for MEDIUM severity alert (default 0.8).
    shortfall_high_ratio : float
        Ratio threshold for HIGH severity alert (default 0.7).
    typical_lookup : HistoricalTypicalLookup | None
        Precomputed lookup object from evaluate.py.
    only_flagged : bool
        If True (default), output only flagged shortfall alerts.
        If False, return all 88 facility-group series.

    Returns
    -------
    pd.DataFrame
        Alerts DataFrame sorted by severity (HIGH then MEDIUM) and deficit percentage.
    """
    if typical_lookup is None:
        typical_lookup = HistoricalTypicalLookup(long_df)

    # Restrict to horizons 1..7
    h7_forecasts = forecasts_df[forecasts_df["horizon"].isin(range(1, 8))].copy()
    if len(h7_forecasts) == 0:
        raise ValueError("Forecast DataFrame has no rows in horizon 1..7.")

    origin_date = str(h7_forecasts["origin_date"].iloc[0])
    origin_dt = datetime.date.fromisoformat(origin_date)
    target_start_date = (origin_dt + datetime.timedelta(days=1)).isoformat()
    target_end_date = (origin_dt + datetime.timedelta(days=7)).isoformat()

    # Aggregate 7-day cumulative sums
    grouped = (
        h7_forecasts.groupby(["facility", "group"], observed=False)
        .agg(
            forecast_7d=("prediction", "sum"),
            forecast_p10_7d=("pred_p10", "sum"),
            forecast_p90_7d=("pred_p90", "sum"),
            actual_7d=("target", "sum"),
            n_horizons=("horizon", "count"),
        )
        .reset_index()
    )

    # Compute typical 7-day volume for each facility x group
    typical_vals = []
    for _, row in grouped.iterrows():
        typ = typical_lookup.get_typical_7d(
            facility=str(row["facility"]),
            group=str(row["group"]),
            origin_date_str=origin_date,
        )
        typical_vals.append(typ)
    grouped["typical_7d"] = typical_vals

    # Compute deficit percentage and shortfall severity
    def _classify_severity(row: pd.Series) -> str:
        typ = row["typical_7d"]
        pred = row["forecast_7d"]
        if typ > 0:
            if pred < shortfall_high_ratio * typ:
                return "HIGH"
            elif pred < shortfall_ratio * typ:
                return "MEDIUM"
        return "NONE"

    grouped["severity"] = grouped.apply(_classify_severity, axis=1)
    grouped["is_shortfall"] = grouped["severity"].isin(["HIGH", "MEDIUM"])

    # Deficit percentage: (forecast - typical) / typical
    grouped["deficit_pct"] = np.where(
        grouped["typical_7d"] > 0,
        (grouped["forecast_7d"] - grouped["typical_7d"]) / grouped["typical_7d"],
        0.0,
    )
    grouped["shortfall_ratio"] = np.where(
        grouped["typical_7d"] > 0,
        grouped["forecast_7d"] / grouped["typical_7d"],
        np.nan,
    )
    grouped["unit_deficit"] = np.maximum(
        0.0, grouped["typical_7d"] - grouped["forecast_7d"]
    ).round(1)

    # Actual shortfall if historical actual data exists
    has_actuals = not h7_forecasts["target"].isna().all()
    if has_actuals:

        def _classify_actual(row: pd.Series) -> tuple[bool, str]:
            typ = row["typical_7d"]
            act = row["actual_7d"]
            if pd.isna(act) or typ <= 0:
                return False, "NONE"
            if act < shortfall_high_ratio * typ:
                return True, "HIGH"
            elif act < shortfall_ratio * typ:
                return True, "MEDIUM"
            return False, "NONE"

        act_results = [_classify_actual(r) for _, r in grouped.iterrows()]
        grouped["actual_shortfall"] = [r[0] for r in act_results]
        grouped["actual_severity"] = [r[1] for r in act_results]
    else:
        grouped["actual_7d"] = np.nan
        grouped["actual_shortfall"] = None
        grouped["actual_severity"] = None

    grouped["origin_date"] = origin_date
    grouped["target_start_date"] = target_start_date
    grouped["target_end_date"] = target_end_date

    # Filter to flagged alerts if requested
    if only_flagged:
        alerts_subset = grouped[grouped["is_shortfall"]].copy()
    else:
        alerts_subset = grouped.copy()

    # Pre-index h=1 feature rows for explanation (keeping facility and group in columns)
    h1_features = feature_df[feature_df["horizon"] == 1].set_index(
        ["facility", "group"], drop=False
    )

    # Compute top 3 reasons and suggested action prescriptions
    reasons_1_list = []
    reasons_2_list = []
    reasons_3_list = []
    reasons_all_list = []
    reasons_str_list = []
    actions_list = []

    for _, row in alerts_subset.iterrows():
        key = (row["facility"], row["group"])
        if key in h1_features.index:
            feat_row = h1_features.loc[key]
            r_top = reasons(feat_row, top_k=3)
        else:
            r_top = ["unspecified historical pattern", "baseline variance", "model baseline"]

        r1 = r_top[0] if len(r_top) > 0 else ""
        r2 = r_top[1] if len(r_top) > 1 else ""
        r3 = r_top[2] if len(r_top) > 2 else ""

        action_prescr = generate_action_prescription(
            reasons_list=r_top,
            group=str(row["group"]),
            unit_deficit=float(row.get("unit_deficit", 0.0)),
            facility=str(row["facility"]),
        )

        reasons_1_list.append(r1)
        reasons_2_list.append(r2)
        reasons_3_list.append(r3)
        reasons_all_list.append(r_top)
        reasons_str_list.append(" · ".join(r_top))
        actions_list.append(action_prescr)

    alerts_subset["reason_1"] = reasons_1_list
    alerts_subset["reason_2"] = reasons_2_list
    alerts_subset["reason_3"] = reasons_3_list
    alerts_subset["reasons"] = reasons_all_list
    alerts_subset["reasons_str"] = reasons_str_list
    alerts_subset["suggested_action"] = actions_list

    # Sort: HIGH severity first, then MEDIUM, then most severe negative deficit
    severity_order = {"HIGH": 0, "MEDIUM": 1, "NONE": 2}
    alerts_subset["_sev_rank"] = alerts_subset["severity"].map(severity_order)
    alerts_subset = alerts_subset.sort_values(
        by=["_sev_rank", "deficit_pct", "typical_7d"], ascending=[True, True, False]
    ).drop(columns=["_sev_rank", "n_horizons"])

    # Desired column order
    final_cols = [
        "origin_date",
        "target_start_date",
        "target_end_date",
        "facility",
        "group",
        "severity",
        "is_shortfall",
        "forecast_7d",
        "forecast_p10_7d",
        "forecast_p90_7d",
        "typical_7d",
        "unit_deficit",
        "deficit_pct",
        "shortfall_ratio",
        "reason_1",
        "reason_2",
        "reason_3",
        "reasons",
        "reasons_str",
        "suggested_action",
        "actual_7d",
        "actual_shortfall",
        "actual_severity",
    ]
    return alerts_subset[final_cols].reset_index(drop=True)


def generate_and_save_alerts(
    origin_date: str = DATA_CUTOFF,
    outputs_dir: Path = OUTPUTS_DIR,
    precomputer: FeaturePrecomputer | None = None,
    typical_lookup: HistoricalTypicalLookup | None = None,
    models: tuple[lgb.Booster, lgb.Booster, lgb.Booster, list[str]] | None = None,
    calendar_df: pd.DataFrame | None = None,
    print_summary: bool = True,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    """Generate 14-day forecasts and 7-day shortfall alerts, persisting them to parquet.

    Parameters
    ----------
    origin_date : str
        Origin date in 'YYYY-MM-DD' format (default: DATA_CUTOFF).
    outputs_dir : Path
        Target directory to save parquet files (default: outputs/).
    precomputer : FeaturePrecomputer | None
        Cached feature precomputer.
    typical_lookup : HistoricalTypicalLookup | None
        Cached typical historical lookup.
    models : tuple | None
        Cached LightGBM models.
    calendar_df : pd.DataFrame | None
        Preloaded calendar DataFrame.
    print_summary : bool
        If True, prints a summary table to stdout.

    Returns
    -------
    tuple[Path, Path, pd.DataFrame, pd.DataFrame]
        (forecasts_path, alerts_path, forecasts_df, alerts_df)
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    long_path = DATA_PROCESSED_DIR / "long.parquet"
    if not long_path.exists():
        from donorcast.clean import clean_data

        clean_data()
    long_df = pd.read_parquet(long_path)

    if calendar_df is None:
        cal_path = CALENDAR_PROCESSED_FILE
        if not cal_path.exists():
            from donorcast.calendar import build_calendar_dataframe

            calendar_df = build_calendar_dataframe()
            calendar_df.to_parquet(cal_path, index=False)
        else:
            calendar_df = pd.read_parquet(cal_path)

    fac_state_df = load_facility_state_mapping(FACILITY_STATE_FILE)

    if precomputer is None:
        precomputer = FeaturePrecomputer(long_df, fac_state_df)

    if typical_lookup is None:
        typical_lookup = HistoricalTypicalLookup(long_df)

    if models is None:
        models = load_final_models()

    # 2. Multi-horizon forecasting
    forecasts_df, feature_df = forecast_facility_groups(
        origin_date=origin_date,
        long_df=long_df,
        calendar_df=calendar_df,
        fac_state_df=fac_state_df,
        precomputer=precomputer,
        models=models,
        horizon=HORIZON,
    )

    # 3. Compute 7-day shortfall alerts
    alerts_df = compute_alerts_table(
        forecasts_df=forecasts_df,
        feature_df=feature_df,
        long_df=long_df,
        shortfall_ratio=SHORTFALL_MEDIUM_RATIO,
        shortfall_high_ratio=SHORTFALL_HIGH_RATIO,
        typical_lookup=typical_lookup,
        only_flagged=True,
    )

    # 4. Save to Parquet
    forecasts_path = outputs_dir / f"forecasts_{origin_date}.parquet"
    alerts_path = outputs_dir / f"alerts_{origin_date}.parquet"

    forecasts_df.to_parquet(forecasts_path, index=False)
    alerts_df.to_parquet(alerts_path, index=False)

    if print_summary:
        print("=" * 80)
        print(f"       DONORCAST SHORTFALL ALERTS: ORIGIN {origin_date}")
        print("=" * 80)
        print(
            f"Forecast Horizons:       1 to 14 days (up to {(datetime.date.fromisoformat(origin_date) + datetime.timedelta(days=14)).isoformat()})"
        )
        print(
            f"Alert Window (7-day):    {(datetime.date.fromisoformat(origin_date) + datetime.timedelta(days=1)).isoformat()} to {(datetime.date.fromisoformat(origin_date) + datetime.timedelta(days=7)).isoformat()}"
        )
        print(f"Total Series Forecasted: {len(forecasts_df) // 14} (22 facilities x 4 groups)")
        print(
            f"Total Alerts Flagged:    {len(alerts_df)} (HIGH: {(alerts_df['severity'] == 'HIGH').sum()}, MEDIUM: {(alerts_df['severity'] == 'MEDIUM').sum()})"
        )
        print("-" * 80)

        if len(alerts_df) == 0:
            print("No shortfall alerts flagged for this origin period.")
        else:
            print(
                f"{'Severity':<8} | {'Facility':<30} | {'Grp':<3} | {'Fcst 7D':<7} | {'Typ 7D':<7} | {'Deficit':<8} | Top Reasons"
            )
            print("-" * 80)
            for _, r in alerts_df.iterrows():
                print(
                    f"{r['severity']:<8} | {r['facility'][:30]:<30} | {r['group']:<3} | "
                    f"{r['forecast_7d']:<7.1f} | {r['typical_7d']:<7.1f} | {r['deficit_pct']:<+8.1%} | {r['reasons_str']}"
                )

        print("-" * 80)
        print(f"Saved forecasts to: {forecasts_path} ({len(forecasts_df):,} rows)")
        print(f"Saved alerts to:    {alerts_path} ({len(alerts_df):,} alerts)")
        print("=" * 80 + "\n")

    return forecasts_path, alerts_path, forecasts_df, alerts_df


def precompute_replay_origins(
    origins: list[str] = REPLAY_ORIGINS,
    outputs_dir: Path = OUTPUTS_DIR,
) -> list[tuple[Path, Path]]:
    """Precompute outputs for the specified historical replay origins.

    Parameters
    ----------
    origins : list[str]
        List of origin dates to precompute.
    outputs_dir : Path
        Output directory.

    Returns
    -------
    list[tuple[Path, Path]]
        List of (forecasts_path, alerts_path) tuples.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("      PRECOMPUTING HISTORICAL REPLAY ORIGINS FOR DASHBOARD       ")
    print("=================================================================")
    print(f"Replay Origins ({len(origins)}): {', '.join(origins)}")

    # Load data once to reuse across origins
    long_path = DATA_PROCESSED_DIR / "long.parquet"
    if not long_path.exists():
        from donorcast.clean import clean_data

        clean_data()
    long_df = pd.read_parquet(long_path)
    calendar_df = pd.read_parquet(CALENDAR_PROCESSED_FILE)
    fac_state_df = load_facility_state_mapping(FACILITY_STATE_FILE)

    precomputer = FeaturePrecomputer(long_df, fac_state_df)
    typical_lookup = HistoricalTypicalLookup(long_df)
    models = load_final_models()

    results = []
    for idx, origin in enumerate(origins, start=1):
        print(
            f"\n[{idx}/{len(origins)}] Generating forecasts and alerts for replay origin {origin}..."
        )
        fcst_p, alert_p, _, alert_df = generate_and_save_alerts(
            origin_date=origin,
            outputs_dir=outputs_dir,
            precomputer=precomputer,
            typical_lookup=typical_lookup,
            models=models,
            calendar_df=calendar_df,
            print_summary=False,
        )
        high_cnt = (alert_df["severity"] == "HIGH").sum()
        med_cnt = (alert_df["severity"] == "MEDIUM").sum()
        print(f" -> Done. Alerts flagged: {len(alert_df)} (HIGH: {high_cnt}, MEDIUM: {med_cnt})")
        print(f"    Saved: {fcst_p.name} and {alert_p.name}")
        results.append((fcst_p, alert_p))

    print("\nAll replay origins precomputed successfully.")
    return results


if __name__ == "__main__":
    generate_and_save_alerts()
