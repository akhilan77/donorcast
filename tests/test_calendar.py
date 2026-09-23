"""Tests for calendar feature generation and facility-state mapping."""

import pandas as pd
import pytest

from donorcast.calendar import (
    STATE_TO_HOLIDAYS_SUBDIV,
    build_calendar_dataframe,
    load_facility_state_mapping,
)
from donorcast.config import (
    CALENDAR_END,
    CALENDAR_START,
    DATA_PROCESSED_DIR,
    FACILITY_STATE_FILE,
)


@pytest.fixture(scope="module")
def calendar_df():
    """Build calendar dataframe once for testing module."""
    return build_calendar_dataframe(start_date=CALENDAR_START, end_date=CALENDAR_END)


def test_facility_state_mapping():
    """Verify facility to state mapping file has 22 valid facilities matching long.parquet."""
    mapping = load_facility_state_mapping(FACILITY_STATE_FILE)
    assert len(mapping) == 22
    assert list(mapping.columns) == ["facility", "state"]
    assert mapping["facility"].nunique() == 22
    assert not mapping.isna().any().any()

    # Verify every facility in long.parquet is in mapping
    long_file = DATA_PROCESSED_DIR / "long.parquet"
    if long_file.exists():
        df_long = pd.read_parquet(long_file)
        facilities_long = sorted(df_long["facility"].unique())
        facilities_mapped = sorted(mapping["facility"].unique())
        assert facilities_mapped == facilities_long

    # Verify all mapped states are known calendar states
    valid_states = set(STATE_TO_HOLIDAYS_SUBDIV.keys())
    assert set(mapping["state"]).issubset(valid_states)


def test_calendar_row_completeness_and_no_nulls(calendar_df):
    """Every (state, date) row exists between 2006-01-01 and 2026-12-31 with zero null values."""
    num_states = len(STATE_TO_HOLIDAYS_SUBDIV)
    # Total days: 7,670
    num_days = (pd.to_datetime(CALENDAR_END) - pd.to_datetime(CALENDAR_START)).days + 1
    assert num_days == 7_670
    assert len(calendar_df) == num_states * num_days

    # Zero nulls
    assert calendar_df.isna().sum().sum() == 0

    # Primary key uniqueness
    assert not calendar_df.duplicated(subset=["state", "date"]).any()

    # Column list validation
    expected_cols = [
        "state",
        "date",
        "year",
        "month",
        "day_of_week",
        "week_of_year",
        "is_weekend",
        "is_public_holiday",
        "is_ramadan",
        "days_to_hari_raya",
        "days_since_hari_raya",
        "days_to_aidiladha",
        "days_since_aidiladha",
        "days_to_cny",
        "days_since_cny",
        "days_to_deepavali",
        "days_since_deepavali",
        "is_school_holiday",
        "is_mco",
        "is_election_day",
    ]
    assert list(calendar_df.columns) == expected_cols


def test_hari_raya_known_dates(calendar_df):
    """Hari Raya Aidilfitri 2024 (2024-04-10) should have days_to = 0, days_since = 0."""
    row = calendar_df[
        (calendar_df["state"] == "W.P. Kuala Lumpur") & (calendar_df["date"] == "2024-04-10")
    ].iloc[0]
    assert row["days_to_hari_raya"] == 0
    assert row["days_since_hari_raya"] == 0
    assert row["is_public_holiday"] == 1

    # Check 5 days before Hari Raya 2024 (2024-04-05)
    row_before = calendar_df[
        (calendar_df["state"] == "W.P. Kuala Lumpur") & (calendar_df["date"] == "2024-04-05")
    ].iloc[0]
    assert row_before["days_to_hari_raya"] == 5

    # Check 4 days after Hari Raya 2024 (2024-04-14)
    row_after = calendar_df[
        (calendar_df["state"] == "W.P. Kuala Lumpur") & (calendar_df["date"] == "2024-04-14")
    ].iloc[0]
    assert row_after["days_since_hari_raya"] == 4


def test_cny_known_dates(calendar_df):
    """CNY 2023 (2023-01-22) should have days_to = 0, days_since = 0, is_public_holiday = 1."""
    row = calendar_df[
        (calendar_df["state"] == "Selangor") & (calendar_df["date"] == "2023-01-22")
    ].iloc[0]
    assert row["days_to_cny"] == 0
    assert row["days_since_cny"] == 0
    assert row["is_public_holiday"] == 1

    # Check capping at 30
    row_far = calendar_df[
        (calendar_df["state"] == "Selangor") & (calendar_df["date"] == "2023-06-15")
    ].iloc[0]
    assert row_far["days_to_cny"] == 30
    assert row_far["days_since_cny"] == 30


def test_deepavali_and_aidiladha_known_dates(calendar_df):
    """Check Deepavali 2024 (2024-10-31) and Aidiladha 2024 (2024-06-16)."""
    row_deep = calendar_df[
        (calendar_df["state"] == "W.P. Kuala Lumpur") & (calendar_df["date"] == "2024-10-31")
    ].iloc[0]
    assert row_deep["days_to_deepavali"] == 0
    assert row_deep["days_since_deepavali"] == 0

    row_adha = calendar_df[
        (calendar_df["state"] == "Perak") & (calendar_df["date"] == "2024-06-16")
    ].iloc[0]
    assert row_adha["days_to_aidiladha"] == 0
    assert row_adha["days_since_aidiladha"] == 0


def test_ramadan_dates(calendar_df):
    """Ramadan 2024 (~2024-03-11 to 2024-04-09) has is_ramadan == 1, but 2024-01-15 is 0."""
    row_ramadan = calendar_df[
        (calendar_df["state"] == "Kelantan") & (calendar_df["date"] == "2024-03-20")
    ].iloc[0]
    assert row_ramadan["is_ramadan"] == 1

    row_non_ramadan = calendar_df[
        (calendar_df["state"] == "Kelantan") & (calendar_df["date"] == "2024-01-15")
    ].iloc[0]
    assert row_non_ramadan["is_ramadan"] == 0


def test_state_weekends_and_policy_changes(calendar_df):
    """Test state weekend configurations and Johor policy change in 2025."""
    # 1. Kelantan is FRI_SAT throughout (Friday 2023-06-09 is weekend, Sunday 2023-06-11 is not)
    kel_fri = calendar_df[
        (calendar_df["state"] == "Kelantan") & (calendar_df["date"] == "2023-06-09")
    ].iloc[0]
    kel_sun = calendar_df[
        (calendar_df["state"] == "Kelantan") & (calendar_df["date"] == "2023-06-11")
    ].iloc[0]
    assert kel_fri["is_weekend"] == 1
    assert kel_sun["is_weekend"] == 0

    # 2. Selangor is SAT_SUN (Friday 2023-06-09 is not weekend, Sunday 2023-06-11 is weekend)
    sel_fri = calendar_df[
        (calendar_df["state"] == "Selangor") & (calendar_df["date"] == "2023-06-09")
    ].iloc[0]
    sel_sun = calendar_df[
        (calendar_df["state"] == "Selangor") & (calendar_df["date"] == "2023-06-11")
    ].iloc[0]
    assert sel_fri["is_weekend"] == 0
    assert sel_sun["is_weekend"] == 1

    # 3. Johor weekend policy shift:
    # In 2023 (under FRI_SAT): Friday 2023-06-09 is weekend
    jhr_2023_fri = calendar_df[
        (calendar_df["state"] == "Johor") & (calendar_df["date"] == "2023-06-09")
    ].iloc[0]
    assert jhr_2023_fri["is_weekend"] == 1

    # In 2025 (reverted to SAT_SUN from 1 Jan 2025): Friday 2025-01-10 is NOT weekend, Sunday 2025-01-12 is weekend
    jhr_2025_fri = calendar_df[
        (calendar_df["state"] == "Johor") & (calendar_df["date"] == "2025-01-10")
    ].iloc[0]
    jhr_2025_sun = calendar_df[
        (calendar_df["state"] == "Johor") & (calendar_df["date"] == "2025-01-12")
    ].iloc[0]
    assert jhr_2025_fri["is_weekend"] == 0
    assert jhr_2025_sun["is_weekend"] == 1


def test_mco_elections_school_holidays(calendar_df):
    """Test MCO, general elections, and school holiday flags."""
    # MCO Phase 1 start: 2020-03-18
    mco_row = calendar_df[
        (calendar_df["state"] == "W.P. Kuala Lumpur") & (calendar_df["date"] == "2020-03-18")
    ].iloc[0]
    assert mco_row["is_mco"] == 1

    # Non-MCO date: 2019-01-01
    non_mco_row = calendar_df[
        (calendar_df["state"] == "W.P. Kuala Lumpur") & (calendar_df["date"] == "2019-01-01")
    ].iloc[0]
    assert non_mco_row["is_mco"] == 0

    # GE15 Election day: 2022-11-19
    ge15_row = calendar_df[
        (calendar_df["state"] == "Perak") & (calendar_df["date"] == "2022-11-19")
    ].iloc[0]
    assert ge15_row["is_election_day"] == 1

    # School holiday: 2024-05-27 (Term 1 Break 2024-05-25 to 2024-06-02)
    school_row = calendar_df[
        (calendar_df["state"] == "Johor") & (calendar_df["date"] == "2024-05-27")
    ].iloc[0]
    assert school_row["is_school_holiday"] == 1


def test_calendar_proximity_helpers(calendar_df):
    """Test bridge days and festival proximity computations."""
    from donorcast.calendar import compute_bridge_days, compute_min_days_to_festival

    bridge_series = compute_bridge_days(calendar_df)
    assert len(bridge_series) == len(calendar_df)
    assert set(bridge_series.unique()).issubset({0, 1})

    min_fest = compute_min_days_to_festival(calendar_df)
    assert len(min_fest) == len(calendar_df)
    assert (min_fest >= 0).all()
    assert (min_fest <= 30).all()
