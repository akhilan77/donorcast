"""Tests for data cleaning pipeline and reconciliation invariants."""

import pandas as pd
import pytest

from donorcast.clean import clean_data
from donorcast.config import (
    DATA_CUTOFF,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    REPORTS_DIR,
)


@pytest.fixture(scope="module")
def cleaned_dataset():
    """Run data cleaning once for the module tests."""
    df, summary = clean_data(
        raw_dir=DATA_RAW_DIR,
        processed_dir=DATA_PROCESSED_DIR,
        reports_dir=REPORTS_DIR,
        cutoff_date=DATA_CUTOFF,
    )
    return df, summary


def test_processed_table_dimensions(cleaned_dataset):
    """Output table must have exactly 22 facilities x 4 groups x 7,570 days = 666,160 rows."""
    df, summary = cleaned_dataset
    assert len(df) == 666_160
    assert summary["output_rows"] == 666_160
    assert summary["num_facilities"] == 22
    assert summary["num_days"] == 7_570

    expected_cols = [
        "facility",
        "date",
        "group",
        "donations",
        "raw_location_mobile",
        "raw_location_centre",
        "raw_social_student",
        "raw_donations_regular",
        "raw_donations_new",
        "raw_type_apheresis_platelet",
        "raw_type_apheresis_plasma",
        "raw_daily",
        "raw_newdonor_17_24",
        "raw_newdonor_total",
    ]
    assert list(df.columns) == expected_cols


def test_no_facility_date_gaps(cleaned_dataset):
    """Verify that no facility has missing dates between 2006-01-01 and DATA_CUTOFF."""
    _, summary = cleaned_dataset
    assert summary["facility_gaps"] == {}


def test_national_total_reconciliation(cleaned_dataset):
    """The 22 facilities sum to the national total in donations_state.csv on every day."""
    df, summary = cleaned_dataset
    assert summary["national_mismatches"] == 0

    # Explicit check against donations_state.csv
    df_state = pd.read_csv(DATA_RAW_DIR / "donations_state.csv")
    df_malaysia = df_state[(df_state["state"] == "Malaysia") & (df_state["date"] <= DATA_CUTOFF)]
    state_daily_map = df_malaysia.set_index("date")["daily"]

    # In our long df, raw_daily is repeated 4 times per facility-day
    facility_day_df = df.drop_duplicates(subset=["facility", "date"])
    fac_daily_sum = facility_day_df.groupby("date")["raw_daily"].sum()

    diff = (fac_daily_sum - state_daily_map).abs().sum()
    assert diff == 0


def test_breakdown_sums_reconcile():
    """Location, type, donor status, and social breakdowns sum to daily with 0 errors."""
    df_fac = pd.read_csv(DATA_RAW_DIR / "donations_facility.csv")
    df_fac = df_fac[df_fac["date"] <= DATA_CUTOFF]
    valid_fac = df_fac[df_fac["hospital"].notna() & (df_fac["hospital"].str.strip() != "")]

    # Location
    loc_sum = valid_fac["location_centre"] + valid_fac["location_mobile"]
    assert (loc_sum != valid_fac["daily"]).sum() == 0

    # Type
    type_sum = (
        valid_fac["type_wholeblood"]
        + valid_fac["type_apheresis_platelet"]
        + valid_fac["type_apheresis_plasma"]
        + valid_fac["type_other"]
    )
    assert (type_sum != valid_fac["daily"]).sum() == 0

    # Donor Status
    donor_sum = (
        valid_fac["donations_new"]
        + valid_fac["donations_regular"]
        + valid_fac["donations_irregular"]
    )
    assert (donor_sum != valid_fac["daily"]).sum() == 0

    # Social Group
    social_sum = (
        valid_fac["social_civilian"]
        + valid_fac["social_student"]
        + valid_fac["social_policearmy"]
    )
    assert (social_sum != valid_fac["daily"]).sum() == 0


def test_group_sum_mismatches_tracked(cleaned_dataset):
    """Group sum mismatches are counted and reported (~283 rows off by 1-3), not fatal."""
    _, summary = cleaned_dataset
    # Expected ~283 mismatches (exact count is 283 in frozen dataset)
    assert summary["group_mismatches"] == 283
