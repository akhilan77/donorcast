"""Unit tests for the forecasting and shortfall alert module (Task 4.3)."""

from pathlib import Path

import numpy as np
import pandas as pd

from donorcast.cli import create_parser
from donorcast.config import DATA_CUTOFF
from donorcast.shortfall import (
    compute_alerts_table,
    generate_and_save_alerts,
    load_final_models,
)


def test_load_final_models():
    """Verify loading of final LightGBM point and quantile boosters."""
    b_point, b_p10, b_p90, feature_names = load_final_models()

    assert b_point is not None
    assert b_p10 is not None
    assert b_p90 is not None
    assert isinstance(feature_names, list)
    assert len(feature_names) > 20
    assert "facility" in feature_names
    assert "group" in feature_names
    assert "lag_0" in feature_names


def test_shortfall_rules_and_severity():
    """Verify exact shortfall classification rules, severity levels, and threshold edge cases."""
    # Synthetic forecasts dataframe for 1 series across 7 horizons
    df_fcst = pd.DataFrame(
        {
            "facility": ["Hospital Melaka"] * 7,
            "group": ["O"] * 7,
            "origin_date": ["2026-09-22"] * 7,
            "horizon": list(range(1, 8)),
            "target_date": [f"2026-09-{23 + i}" for i in range(7)],
            "target": [np.nan] * 7,
            "prediction": [10.0] * 7,  # total 7d = 70.0
            "pred_p50": [10.0] * 7,
            "pred_p10": [5.0] * 7,
            "pred_p90": [15.0] * 7,
        }
    )

    df_feat = pd.DataFrame(
        {
            "facility": ["Hospital Melaka"],
            "group": ["O"],
            "origin_date": ["2026-09-22"],
            "horizon": [1],
            "days_to_hari_raya": [3],
            "share_student_28": [0.05],
            "is_school_holiday": [1],
            "lag_0": [10.0],
            "lag_1": [10.0],
            "lag_2": [10.0],
            "lag_6": [10.0],
            "lag_13": [10.0],
            "lag_20": [10.0],
            "lag_27": [10.0],
            "rolling_mean_7": [10.0],
            "rolling_std_7": [2.0],
            "rolling_mean_28": [10.0],
            "rolling_std_28": [2.0],
            "same_weekday_mean_4": [10.0],
            "share_mobile_28": [0.1],
            "share_regular_28": [0.5],
            "share_new_donor_28": [0.2],
            "share_apheresis_28": [0.01],
            "share_newdonor_17_24_28": [0.2],
            "day_of_week": [2],
            "week_of_year": [39],
            "month": [9],
            "is_weekend": [0],
            "is_public_holiday": [0],
            "is_ramadan": [0],
            "days_since_hari_raya": [50],
            "days_to_aidiladha": [50],
            "days_since_aidiladha": [50],
            "days_to_cny": [50],
            "days_since_cny": [50],
            "days_to_deepavali": [50],
            "days_since_deepavali": [50],
            "is_mco": [0],
            "is_election_day": [0],
        }
    )

    long_dummy = pd.DataFrame({"facility": [], "group": [], "date": [], "donations": []})

    class MockLookup:
        def __init__(self, typ_val):
            self.typ_val = typ_val

        def get_typical_7d(self, facility, group, origin_date_str):
            return self.typ_val

    # Case 1: forecast_7d = 70.0, typical = 100.0 (ratio = 0.70) -> forecast < 0.8 * typical -> MEDIUM
    lookup_med = MockLookup(typ_val=100.0)
    alerts_med = compute_alerts_table(
        df_fcst, df_feat, long_dummy, typical_lookup=lookup_med, only_flagged=False
    )
    assert len(alerts_med) == 1
    assert alerts_med.iloc[0]["severity"] == "MEDIUM"
    assert alerts_med.iloc[0]["is_shortfall"] is True or alerts_med.iloc[0]["is_shortfall"] == 1

    # Case 2: forecast_7d = 70.0, typical = 110.0 (forecast = 70 < 0.7 * 110 = 77.0) -> HIGH
    lookup_high = MockLookup(typ_val=110.0)
    alerts_high = compute_alerts_table(
        df_fcst, df_feat, long_dummy, typical_lookup=lookup_high, only_flagged=False
    )
    assert len(alerts_high) == 1
    assert alerts_high.iloc[0]["severity"] == "HIGH"
    assert alerts_high.iloc[0]["is_shortfall"] is True or alerts_high.iloc[0]["is_shortfall"] == 1

    # Case 3: forecast_7d = 70.0, typical = 80.0 (ratio = 0.875 >= 0.8) -> NONE
    lookup_none = MockLookup(typ_val=80.0)
    alerts_none = compute_alerts_table(
        df_fcst, df_feat, long_dummy, typical_lookup=lookup_none, only_flagged=False
    )
    assert len(alerts_none) == 1
    assert alerts_none.iloc[0]["severity"] == "NONE"
    assert alerts_none.iloc[0]["is_shortfall"] is False or alerts_none.iloc[0]["is_shortfall"] == 0

    # Case 4: typical = 0.0 -> NONE
    lookup_zero = MockLookup(typ_val=0.0)
    alerts_zero = compute_alerts_table(
        df_fcst, df_feat, long_dummy, typical_lookup=lookup_zero, only_flagged=False
    )
    assert len(alerts_zero) == 1
    assert alerts_zero.iloc[0]["severity"] == "NONE"
    assert alerts_zero.iloc[0]["is_shortfall"] is False or alerts_zero.iloc[0]["is_shortfall"] == 0


def test_generate_and_save_alerts_live(tmp_path: Path):
    """Verify live generation of forecasts and alerts for default DATA_CUTOFF origin."""
    fcst_path, alert_path, df_fcst, df_alert = generate_and_save_alerts(
        origin_date=DATA_CUTOFF,
        outputs_dir=tmp_path,
        print_summary=False,
    )

    # 1. Output files exist
    assert fcst_path.exists()
    assert alert_path.exists()

    # 2. Forecasts schema and dimensions: 22 facilities x 4 groups x 14 horizons = 1232 rows
    assert len(df_fcst) == 1232
    assert "prediction" in df_fcst.columns
    assert "pred_p10" in df_fcst.columns
    assert "pred_p90" in df_fcst.columns

    # 3. Quantile constraints
    assert (df_fcst["prediction"] >= 0.0).all()
    assert (df_fcst["pred_p10"] >= 0.0).all()
    assert (df_fcst["pred_p90"] >= df_fcst["pred_p10"]).all()

    # 4. Alerts schema
    assert len(df_alert) > 0
    assert "severity" in df_alert.columns
    assert "forecast_7d" in df_alert.columns
    assert "typical_7d" in df_alert.columns
    assert "reasons" in df_alert.columns
    assert "reasons_str" in df_alert.columns

    # All alerts are either HIGH or MEDIUM
    assert df_alert["severity"].isin(["HIGH", "MEDIUM"]).all()

    # Verify each alert has top 3 non-empty reasons
    for _, r in df_alert.iterrows():
        assert isinstance(r["reasons"], (list, np.ndarray))
        assert len(r["reasons"]) == 3
        assert all(isinstance(s, str) and len(s) > 0 for s in r["reasons"])
        assert " · " in r["reasons_str"]

    # Verify reloading from parquet
    loaded_fcst = pd.read_parquet(fcst_path)
    loaded_alert = pd.read_parquet(alert_path)
    assert len(loaded_fcst) == 1232
    assert len(loaded_alert) == len(df_alert)


def test_cli_alerts_parser():
    """Verify CLI parser options for donorcast alerts."""
    parser = create_parser()

    # Default invocation
    args = parser.parse_args(["alerts"])
    assert args.command == "alerts"
    assert args.origin == DATA_CUTOFF
    assert args.precompute_replay is False

    # Explicit origin and flag
    args2 = parser.parse_args(["alerts", "--origin", "2025-03-24", "--precompute-replay"])
    assert args2.command == "alerts"
    assert args2.origin == "2025-03-24"
    assert args2.precompute_replay is True
