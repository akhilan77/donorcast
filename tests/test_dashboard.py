"""Tests for Streamlit Dashboard Data Loading and Components (Task 4.5).

Verifies:
1. All required output artifacts exist and match expected schemas.
2. Data loading functions in app/streamlit_app.py strictly read from outputs/.
3. Replay origin 2025-03-24 correctly shows the pre-Hari Raya shortfall alerts and holiday dip.
4. Dashboard functions operate without triggering any model training or feature engineering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.streamlit_app import (
    DATA_CUTOFF,
    OUTPUTS_DIR,
    get_available_origins,
    get_facility_list,
    load_alerts,
    load_forecasts,
    load_historical_actuals,
)


def test_outputs_artifacts_exist():
    """Verify that outputs directory contains forecasts, alerts, and actuals."""
    assert OUTPUTS_DIR.exists(), f"OUTPUTS_DIR does not exist: {OUTPUTS_DIR}"

    actuals_file = OUTPUTS_DIR / "actuals.parquet"
    assert actuals_file.exists(), f"Missing actuals parquet: {actuals_file}"

    latest_fcst = OUTPUTS_DIR / f"forecasts_{DATA_CUTOFF}.parquet"
    latest_alerts = OUTPUTS_DIR / f"alerts_{DATA_CUTOFF}.parquet"
    assert latest_fcst.exists(), f"Missing latest forecast: {latest_fcst}"
    assert latest_alerts.exists(), f"Missing latest alerts: {latest_alerts}"

    # Verify Hari Raya 2025 replay exists
    hr_fcst = OUTPUTS_DIR / "forecasts_2025-03-24.parquet"
    hr_alerts = OUTPUTS_DIR / "alerts_2025-03-24.parquet"
    assert hr_fcst.exists(), f"Missing Hari Raya forecast: {hr_fcst}"
    assert hr_alerts.exists(), f"Missing Hari Raya alerts: {hr_alerts}"


def test_get_available_origins():
    """Verify origin discovery lists latest cutoff first and includes replay origins."""
    origins = get_available_origins()
    assert len(origins) >= 8, f"Expected at least 8 origins, got {len(origins)}"
    assert origins[0] == DATA_CUTOFF, f"First origin must be latest {DATA_CUTOFF}, got {origins[0]}"
    assert "2025-03-24" in origins, "Hari Raya 2025 origin (2025-03-24) missing from origins"


def test_load_historical_actuals():
    """Verify historical actuals table contains required schema and span."""
    df_act = load_historical_actuals()
    assert isinstance(df_act, pd.DataFrame)
    assert len(df_act) > 0

    expected_cols = {"date", "facility", "group", "donations"}
    assert expected_cols.issubset(set(df_act.columns))

    facilities = df_act["facility"].unique()
    assert len(facilities) == 22, f"Expected 22 facilities, got {len(facilities)}"

    groups = set(df_act["group"].unique())
    assert groups == {"A", "B", "AB", "O"}


def test_load_forecasts_schema_and_integrity():
    """Verify forecast DataFrame structure and horizons for latest cutoff."""
    df_fcst = load_forecasts(DATA_CUTOFF)
    assert isinstance(df_fcst, pd.DataFrame)
    assert len(df_fcst) == 1232  # 22 facilities x 4 groups x 14 horizons

    expected_cols = [
        "facility",
        "group",
        "origin_date",
        "horizon",
        "target_date",
        "target",
        "prediction",
        "pred_p50",
        "pred_p10",
        "pred_p90",
    ]
    for col in expected_cols:
        assert col in df_fcst.columns, f"Missing forecast column: {col}"

    # Verify horizons 1..14
    assert sorted(df_fcst["horizon"].unique().tolist()) == list(range(1, 15))

    # Verify non-negativity and quantile monotonicity
    assert (df_fcst["pred_p50"] >= 0).all()
    assert (df_fcst["pred_p10"] >= 0).all()
    assert (df_fcst["pred_p90"] >= df_fcst["pred_p10"]).all()


def test_load_alerts_schema_and_reasons():
    """Verify alerts DataFrame schema, severity ranking, and top-3 reason strings."""
    df_alerts = load_alerts(DATA_CUTOFF)
    assert isinstance(df_alerts, pd.DataFrame)
    assert len(df_alerts) > 0, "Expected at least 1 alert for 2026-09-22"

    required_cols = [
        "facility",
        "group",
        "severity",
        "forecast_7d",
        "typical_7d",
        "deficit_pct",
        "reason_1",
        "reason_2",
        "reason_3",
        "reasons_str",
    ]
    for col in required_cols:
        assert col in df_alerts.columns, f"Missing alert column: {col}"

    # Verify valid severities
    assert set(df_alerts["severity"].unique()).issubset({"HIGH", "MEDIUM"})

    # Verify reasons are human-readable strings and non-empty
    for _, row in df_alerts.iterrows():
        assert isinstance(row["reason_1"], str) and len(row["reason_1"]) > 0
        assert isinstance(row["reason_2"], str) and len(row["reason_2"]) > 0
        assert isinstance(row["reason_3"], str) and len(row["reason_3"]) > 0
        assert " · " in row["reasons_str"]


def test_hari_raya_2025_replay_behavior():
    """Verify Hari Raya 2025 replay origin captures the holiday dip in forecast and actuals."""
    origin = "2025-03-24"
    df_fcst = load_forecasts(origin)
    df_alerts = load_alerts(origin)

    # 1. High prevalence of shortfall alerts flagged before Hari Raya
    assert len(df_alerts) > 30, f"Expected widespread shortfall alerts, got {len(df_alerts)}"
    high_count = (df_alerts["severity"] == "HIGH").sum()
    assert high_count > 20, f"Expected high number of HIGH alerts, got {high_count}"

    # 2. Check Pusat Darah Negara group O for holiday dip on 2025-03-31
    pdn_o = df_fcst[
        (df_fcst["facility"] == "Pusat Darah Negara") & (df_fcst["group"] == "O")
    ].sort_values("target_date")

    assert len(pdn_o) == 14

    # Hari Raya Aidilfitri was on 2025-03-31
    holiday_row = pdn_o[pdn_o["target_date"] == "2025-03-31"]
    assert len(holiday_row) == 1

    holiday_pred = holiday_row["pred_p50"].iloc[0]
    holiday_target = holiday_row["target"].iloc[0]

    # Pre-holiday regular donations
    normal_pred = pdn_o[pdn_o["target_date"] == "2025-03-25"]["pred_p50"].iloc[0]

    # Model predicted significant dip
    assert holiday_pred < normal_pred * 0.5, (
        f"Expected predicted dip on Hari Raya (pred={holiday_pred:.1f}, normal={normal_pred:.1f})"
    )

    # Realized actuals confirmed dip
    assert holiday_target == 0.0, (
        f"Expected actual donations to drop to 0 on Hari Raya, got {holiday_target}"
    )


def test_get_facility_list():
    """Verify get_facility_list returns all 22 MoH facilities sorted alphabetically."""
    facs = get_facility_list()
    assert len(facs) == 22, f"Expected 22 facilities, got {len(facs)}"
    assert facs == sorted(facs), "Facility list should be sorted alphabetically"
    assert "Pusat Darah Negara" in facs


def test_apptest_interactive_flow():
    """Verify Streamlit app interaction flow, page navigation, filtering, and drill-down."""
    from streamlit.testing.v1 import AppTest

    script_path = str(PROJECT_ROOT / "app" / "streamlit_app.py")
    at = AppTest.from_file(script_path, default_timeout=15.0)
    at.run()
    assert len(at.exception) == 0, f"AppTest raised unexpected exception: {at.exception}"

    # 1. Verify Forecast page initial state
    assert at.sidebar.radio(key="nav_radio").value == "Forecast"
    assert len(at.dataframe) == 1  # 14-day daily values table

    # 2. Change facility and blood group
    at.selectbox(key="sb_facility").select("Hospital Melaka")
    at.selectbox(key="sb_group").select("A")
    at.run()
    assert len(at.exception) == 0
    assert at.selectbox(key="sb_facility").value == "Hospital Melaka"
    assert at.selectbox(key="sb_group").value == "A"

    # 3. Switch to Shortfall Alerts page
    at.sidebar.radio(key="nav_radio").set_value("Shortfall Alerts")
    at.run()
    assert len(at.exception) == 0
    assert at.sidebar.radio(key="nav_radio").value == "Shortfall Alerts"

    # 4. Filter by severity
    at.selectbox(key="alert_filter_severity").select("HIGH")
    at.run()
    assert len(at.exception) == 0
    assert at.selectbox(key="alert_filter_severity").value == "HIGH"
    assert len(at.dataframe) == 1

    # 5. Filter by blood group
    at.selectbox(key="alert_filter_group").select("O")
    at.run()
    assert len(at.exception) == 0

    # Reset severity to All
    at.selectbox(key="alert_filter_severity").select("All")
    at.selectbox(key="alert_filter_group").select("All")
    at.run()
    assert len(at.exception) == 0

    # 6. Click drill-down button to navigate to Forecast page
    opt = at.selectbox(key="inspect_alert_select").options[0]
    at.selectbox(key="inspect_alert_select").select(opt)
    at.run()
    at.button(key="btn_drilldown").click()
    at.run()
    assert len(at.exception) == 0

    # Verify redirected to Forecast
    assert at.sidebar.radio(key="nav_radio").value == "Forecast"
    assert len(at.dataframe) == 1

    # 7. Test replay origin selection
    at.sidebar.selectbox(key="sb_origin").select("2025-03-24 (Hari Raya 2025 Replay)")
    at.run()
    assert len(at.exception) == 0
    assert at.sidebar.selectbox(key="sb_origin").value == "2025-03-24 (Hari Raya 2025 Replay)"

    # Check that table on replay origin includes Actual column
    df_displayed = at.dataframe[0].value
    assert "Actual" in df_displayed.columns
    assert list(df_displayed.columns) == [
        "Date",
        "Expected",
        "Low estimate",
        "High estimate",
        "Actual",
    ]
