"""Tests for feature engineering pipeline and critical invariants:

1. Invariant NO_SAME_DAY_BREAKDOWNS:
   No feature uses a same-day breakdown column. All raw breakdown signals
   must only appear through lagged aggregates ending at t.
2. Invariant FEATURES_AS_OF_ORIGIN:
   Features for origin t use only rows dated <= t. Mutating all rows after t
   leaves the computed features for origin t completely unchanged.
3. Feature schema, types (float32 / categorical), horizon 1..14 coverage,
   and calculation logic for same_weekday_mean_4.
"""

import numpy as np
import pandas as pd
import pytest

from donorcast.calendar import build_calendar_dataframe, load_facility_state_mapping
from donorcast.clean import clean_data
from donorcast.config import (
    CALENDAR_PROCESSED_FILE,
    DATA_PROCESSED_DIR,
    FACILITY_STATE_FILE,
    HORIZON,
)
from donorcast.features import (
    CALENDAR_FEATURE_COLS,
    FeaturePrecomputer,
    build_features_for_origins,
)


@pytest.fixture(scope="module")
def shared_data():
    """Load or build the underlying clean data, calendar and facility-state mapping."""
    long_file = DATA_PROCESSED_DIR / "long.parquet"
    if not long_file.exists():
        long_df, _ = clean_data()
    else:
        long_df = pd.read_parquet(long_file)

    cal_file = CALENDAR_PROCESSED_FILE
    if not cal_file.exists():
        cal_df = build_calendar_dataframe()
    else:
        cal_df = pd.read_parquet(cal_file)

    fac_state_df = load_facility_state_mapping(FACILITY_STATE_FILE)
    return long_df, cal_df, fac_state_df


def test_no_same_day_breakdowns(shared_data):
    """Invariant: NO_SAME_DAY_BREAKDOWNS.

    Assert that no raw_ columns appear in the feature set.
    All breakdown information enters only through 28-day lagged share features.
    """
    long_df, cal_df, fac_state_df = shared_data
    sample_origins = ["2020-06-01", "2021-03-15"]

    df_feat = build_features_for_origins(
        origin_dates=sample_origins,
        long_df=long_df,
        calendar_df=cal_df,
        facility_state_df=fac_state_df,
        horizon=HORIZON,
    )

    # Check that no raw_ prefix columns exist in df_feat
    raw_cols = [c for c in df_feat.columns if c.startswith("raw_")]
    assert len(raw_cols) == 0, f"Violated NO_SAME_DAY_BREAKDOWNS: Found raw columns: {raw_cols}"

    # Verify expected feature columns are present
    expected_lag_cols = [
        "lag_0",
        "lag_1",
        "lag_2",
        "lag_6",
        "lag_13",
        "lag_20",
        "lag_27",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_28",
        "rolling_std_28",
        "same_weekday_mean_4",
        "share_mobile_28",
        "share_student_28",
        "share_regular_28",
        "share_new_donor_28",
        "share_apheresis_28",
        "share_newdonor_17_24_28",
    ]
    for col in expected_lag_cols:
        assert col in df_feat.columns, f"Missing expected lag/share column: {col}"

    for col in CALENDAR_FEATURE_COLS:
        assert col in df_feat.columns, f"Missing expected calendar column: {col}"


def test_features_as_of_origin(shared_data):
    """Invariant: FEATURES_AS_OF_ORIGIN.

    Features computed for origin t must use only data dated <= t.
    If we change every value in long_df after date t and rebuild features for origin t,
    all feature columns must remain exactly identical.
    """
    long_df, cal_df, fac_state_df = shared_data
    origin_t = "2019-06-17"

    # Build features on clean original data
    df_feat_original = build_features_for_origins(
        origin_dates=[origin_t],
        long_df=long_df,
        calendar_df=cal_df,
        facility_state_df=fac_state_df,
        horizon=HORIZON,
    )

    # Create corrupted long_df where all rows with date > origin_t are severely corrupted
    long_df_corrupted = long_df.copy()
    future_mask = long_df_corrupted["date"] > origin_t

    # Corrupt all numeric values in future rows
    numeric_cols = [
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
    for col in numeric_cols:
        long_df_corrupted.loc[future_mask, col] = 999999

    # Rebuild features for origin_t using corrupted dataframe
    df_feat_corrupted = build_features_for_origins(
        origin_dates=[origin_t],
        long_df=long_df_corrupted,
        calendar_df=cal_df,
        facility_state_df=fac_state_df,
        horizon=HORIZON,
    )

    # Exclude 'target' because target is on date t+h which intentionally changed
    feature_cols = [
        c
        for c in df_feat_original.columns
        if c not in ["target", "facility", "group", "origin_date", "target_date"]
    ]

    for col in feature_cols:
        orig_vals = df_feat_original[col].values
        corr_vals = df_feat_corrupted[col].values
        np.testing.assert_allclose(
            orig_vals,
            corr_vals,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Feature '{col}' changed when future data (date > {origin_t}) was mutated! Leakage detected.",
        )


def test_feature_matrix_shapes_and_types(shared_data):
    """Verify feature row count, dtypes (float32/int16/category), and no NaNs in features."""
    long_df, cal_df, fac_state_df = shared_data
    origins = ["2018-01-01", "2018-01-04"]

    df_feat = build_features_for_origins(
        origin_dates=origins,
        long_df=long_df,
        calendar_df=cal_df,
        facility_state_df=fac_state_df,
        horizon=HORIZON,
    )

    # 2 origins * 14 horizons * 22 facilities * 4 blood groups = 2464 rows
    n_facilities = len(long_df["facility"].unique())
    n_groups = len(long_df["group"].unique())
    expected_rows = len(origins) * HORIZON * n_facilities * n_groups
    assert len(df_feat) == expected_rows

    # Verify dtypes
    assert df_feat["facility"].dtype.name == "category"
    assert df_feat["group"].dtype.name == "category"
    assert df_feat["lag_0"].dtype == np.float32
    assert df_feat["rolling_mean_7"].dtype == np.float32
    assert df_feat["share_mobile_28"].dtype == np.float32
    assert df_feat["horizon"].dtype == np.int16
    assert df_feat["day_of_week"].dtype == np.int16

    # Verify no NaN in features
    feature_cols = [c for c in df_feat.columns if c != "target"]
    assert not df_feat[feature_cols].isna().any().any(), "Found unexpected NaN in features"


def test_same_weekday_mean_logic(shared_data):
    """Verify same_weekday_mean_4 matches the manual calculation for a target weekday."""
    long_df, cal_df, fac_state_df = shared_data
    precomputer = FeaturePrecomputer(long_df, fac_state_df)

    # Pick an origin date: 2020-05-18 is a Monday (weekday 0)
    origin_t = "2020-05-18"
    df_feat = build_features_for_origins(
        origin_dates=[origin_t],
        long_df=long_df,
        calendar_df=cal_df,
        facility_state_df=fac_state_df,
        horizon=HORIZON,
        precomputer=precomputer,
    )

    # For horizon h=1, target_date is 2020-05-19 (Tuesday, weekday 1)
    # Target weekday = Tuesday (1)
    # The 4 most recent Tuesdays <= 2020-05-18 are:
    # 2020-05-12, 2020-05-05, 2020-04-28, 2020-04-21
    tuesdays = ["2020-05-12", "2020-05-05", "2020-04-28", "2020-04-21"]

    fac = "Pusat Darah Negara"
    grp = "O"

    # Compute manual mean of donations on those 4 tuesdays
    manual_vals = [
        long_df.loc[
            (long_df["date"] == d) & (long_df["facility"] == fac) & (long_df["group"] == grp),
            "donations",
        ].values[0]
        for d in tuesdays
    ]
    manual_mean = float(np.mean(manual_vals))

    row_h1 = df_feat[
        (df_feat["facility"] == fac) & (df_feat["group"] == grp) & (df_feat["horizon"] == 1)
    ].iloc[0]

    assert pytest.approx(row_h1["same_weekday_mean_4"], 1e-4) == manual_mean
