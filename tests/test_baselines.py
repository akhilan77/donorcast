"""Tests for baseline models in src/donorcast/models/baselines.py:

- M0 Seasonal Naive correctness
- M0b Weekday Moving Average correctness
- Prevention of future leakage
- Integration with evaluate harness and output schema validation
"""

import numpy as np
import pandas as pd
import pytest

from donorcast.models.baselines import (
    predict_seasonal_naive,
    predict_weekday_moving_average,
)


@pytest.fixture
def synthetic_long_df():
    """Create 6 weeks of daily data for 2 facilities and 2 groups with known patterns."""
    dates = pd.date_range("2023-01-01", "2023-02-15", freq="D").strftime("%Y-%m-%d")
    rows = []
    for d_str in dates:
        d = pd.Timestamp(d_str)
        w = d.weekday()
        for fac in ["PDN", "Hospital Penang"]:
            fac_mult = 2.0 if fac == "PDN" else 1.0
            for grp in ["A", "O"]:
                grp_mult = 1.5 if grp == "O" else 1.0
                # Base volume: weekday-dependent (e.g. Sunday=w=6 is 50, Monday=w=0 is 10, etc.)
                # plus weekly increment
                week_num = d.isocalendar().week
                val = (10.0 + 5.0 * w + 2.0 * week_num) * fac_mult * grp_mult
                rows.append({"facility": fac, "group": grp, "date": d_str, "donations": float(val)})

    return pd.DataFrame(rows)


def test_m0_seasonal_naive_exact_values(synthetic_long_df):
    """Test that M0 selects the most recent same weekday <= t for both h in 1..7 and h in 8..14."""
    # Origin: Monday 2023-01-16 (t)
    origin = "2023-01-16"
    origin_dt = pd.Timestamp(origin)
    assert origin_dt.weekday() == 0  # Monday

    preds = predict_seasonal_naive(synthetic_long_df, origin_dates=[origin], horizon=14)
    assert len(preds) == 14 * 4  # 14 horizons x 2 facilities x 2 groups

    # Expected columns
    expected_cols = {
        "facility",
        "group",
        "origin_date",
        "horizon",
        "target_date",
        "target",
        "prediction",
    }
    assert expected_cols.issubset(preds.columns)

    # Check Horizon 1: Target = 2023-01-17 (Tuesday). Most recent Tuesday <= 2023-01-16 is 2023-01-10.
    h1 = preds[
        (preds["horizon"] == 1) & (preds["facility"] == "PDN") & (preds["group"] == "O")
    ].iloc[0]
    assert h1["target_date"] == "2023-01-17"
    actual_prev_tue = synthetic_long_df[
        (synthetic_long_df["facility"] == "PDN")
        & (synthetic_long_df["group"] == "O")
        & (synthetic_long_df["date"] == "2023-01-10")
    ]["donations"].iloc[0]
    assert h1["prediction"] == actual_prev_tue

    # Check Horizon 7: Target = 2023-01-23 (Monday). Most recent Monday <= 2023-01-16 is 2023-01-16 itself!
    h7 = preds[
        (preds["horizon"] == 7) & (preds["facility"] == "PDN") & (preds["group"] == "O")
    ].iloc[0]
    assert h7["target_date"] == "2023-01-23"
    actual_origin_mon = synthetic_long_df[
        (synthetic_long_df["facility"] == "PDN")
        & (synthetic_long_df["group"] == "O")
        & (synthetic_long_df["date"] == "2023-01-16")
    ]["donations"].iloc[0]
    assert h7["prediction"] == actual_origin_mon

    # Check Horizon 8: Target = 2023-01-24 (Tuesday week 2). Most recent Tuesday <= 2023-01-16 is still 2023-01-10.
    h8 = preds[
        (preds["horizon"] == 8) & (preds["facility"] == "PDN") & (preds["group"] == "O")
    ].iloc[0]
    assert h8["target_date"] == "2023-01-24"
    assert h8["prediction"] == actual_prev_tue


def test_m0b_weekday_moving_average_exact_values(synthetic_long_df):
    """Test that M0b averages the last 4 same weekdays <= t."""
    # Origin: Monday 2023-01-30 (5th Monday in series)
    origin = "2023-01-30"
    preds = predict_weekday_moving_average(
        synthetic_long_df, origin_dates=[origin], horizon=14, window=4
    )

    # Check Horizon 1: Target = 2023-01-31 (Tuesday). Last 4 Tuesdays <= 2023-01-30 are: 2023-01-24, 2023-01-17, 2023-01-10, 2023-01-03
    h1 = preds[
        (preds["horizon"] == 1) & (preds["facility"] == "PDN") & (preds["group"] == "O")
    ].iloc[0]
    tue_dates = ["2023-01-03", "2023-01-10", "2023-01-17", "2023-01-24"]
    expected_mean = synthetic_long_df[
        (synthetic_long_df["facility"] == "PDN")
        & (synthetic_long_df["group"] == "O")
        & (synthetic_long_df["date"].isin(tue_dates))
    ]["donations"].mean()

    assert pytest.approx(h1["prediction"], 1e-4) == expected_mean

    # Check Horizon 7: Target = 2023-02-06 (Monday). Last 4 Mondays <= 2023-01-30 are: 2023-01-30, 2023-01-23, 2023-01-16, 2023-01-09
    h7 = preds[
        (preds["horizon"] == 7) & (preds["facility"] == "PDN") & (preds["group"] == "O")
    ].iloc[0]
    mon_dates = ["2023-01-09", "2023-01-16", "2023-01-23", "2023-01-30"]
    expected_mon_mean = synthetic_long_df[
        (synthetic_long_df["facility"] == "PDN")
        & (synthetic_long_df["group"] == "O")
        & (synthetic_long_df["date"].isin(mon_dates))
    ]["donations"].mean()

    assert pytest.approx(h7["prediction"], 1e-4) == expected_mon_mean


def test_no_future_leakage_in_baselines(synthetic_long_df):
    """Mutating target rows > origin date t should NOT change M0 or M0b predictions."""
    origin = "2023-01-16"

    m0_orig = predict_seasonal_naive(synthetic_long_df, origin_dates=[origin], horizon=14)
    m0b_orig = predict_weekday_moving_average(synthetic_long_df, origin_dates=[origin], horizon=14)

    # Mutate data > 2023-01-16
    corrupted_df = synthetic_long_df.copy()
    corrupted_df.loc[corrupted_df["date"] > origin, "donations"] = 999999.0

    m0_corrupted = predict_seasonal_naive(corrupted_df, origin_dates=[origin], horizon=14)
    m0b_corrupted = predict_weekday_moving_average(corrupted_df, origin_dates=[origin], horizon=14)

    np.testing.assert_array_almost_equal(
        m0_orig["prediction"].values, m0_corrupted["prediction"].values
    )
    np.testing.assert_array_almost_equal(
        m0b_orig["prediction"].values, m0b_corrupted["prediction"].values
    )
