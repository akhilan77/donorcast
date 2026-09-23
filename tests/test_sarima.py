"""Tests for SARIMAX model in src/donorcast/models/sarima.py:

- Single series fitting and 14-step forecasting with exogenous variables
- Sequential state updating via extend(refit=False)
- No future leakage (predictions at origin t use only data <= t)
- evaluate harness integration and output schema verification
"""

import numpy as np
import pandas as pd
import pytest

from donorcast.models.sarima import (
    EXOG_COLS,
    fit_and_forecast_single_series,
    prepare_data_and_exog,
)


@pytest.fixture
def synthetic_sarima_data():
    """Create 4 years of daily data for 2 facilities with known calendar exogenous patterns."""
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D").strftime("%Y-%m-%d")
    rows = []
    for d_str in dates:
        d = pd.Timestamp(d_str)
        w = d.weekday()
        for fac in ["Pusat Darah Negara", "Hospital Melaka"]:
            fac_mult = 2.0 if "Pusat" in fac else 1.0
            for grp in ["A", "O"]:
                grp_mult = 1.5 if grp == "O" else 1.0
                # Base volume with weekday seasonality
                base = (20.0 + 8.0 * np.sin(2 * np.pi * w / 7)) * fac_mult * grp_mult
                rows.append({"facility": fac, "group": grp, "date": d_str, "donations": float(base)})

    long_df = pd.DataFrame(rows)

    # Synthetic calendar
    cal_rows = []
    for st in ["W.P. Kuala Lumpur", "Melaka"]:
        for d_str in dates:
            d = pd.Timestamp(d_str)
            cal_rows.append(
                {
                    "date": d_str,
                    "state": st,
                    "is_public_holiday": 1 if d.month == 1 and d.day == 1 else 0,
                    "is_school_holiday": 1 if d.month == 12 else 0,
                    "is_ramadan": 0,
                    "is_mco": 0,
                }
            )
    calendar_df = pd.DataFrame(cal_rows)

    fac_state_df = pd.DataFrame(
        [
            {"hospital": "Pusat Darah Negara", "state": "W.P. Kuala Lumpur"},
            {"hospital": "Hospital Melaka", "state": "Melaka"},
        ]
    )

    return long_df, calendar_df, fac_state_df


def test_prepare_data_and_exog(synthetic_sarima_data):
    long_df, calendar_df, fac_state_df = synthetic_sarima_data
    series_dict, facility_exog_dict, facility_to_state, all_dates = prepare_data_and_exog(
        long_df, calendar_df=calendar_df, facility_state_df=fac_state_df
    )

    assert ("Pusat Darah Negara", "O") in series_dict
    assert ("Hospital Melaka", "A") in series_dict
    assert len(series_dict[("Pusat Darah Negara", "O")]) == len(all_dates)

    assert "Pusat Darah Negara" in facility_exog_dict
    assert list(facility_exog_dict["Pusat Darah Negara"].columns) == EXOG_COLS
    assert len(facility_exog_dict["Pusat Darah Negara"]) == len(all_dates)


def test_fit_and_forecast_single_series(synthetic_sarima_data):
    long_df, calendar_df, fac_state_df = synthetic_sarima_data
    series_dict, facility_exog_dict, _, _ = prepare_data_and_exog(
        long_df, calendar_df=calendar_df, facility_state_df=fac_state_df
    )

    fac = "Pusat Darah Negara"
    grp = "O"
    series = series_dict[(fac, grp)]
    exog = facility_exog_dict[fac]

    # Test 3 weekly validation origins in 2023
    val_origins = ["2023-01-02", "2023-01-09", "2023-01-16"]

    preds_df = fit_and_forecast_single_series(
        facility=fac,
        group=grp,
        series=series,
        exog_df=exog,
        origin_dates=val_origins,
        train_start="2020-01-01",
        train_end="2022-12-31",
        horizon=14,
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 7),
    )

    # Expect 3 origins x 14 horizons = 42 rows
    assert len(preds_df) == 42

    expected_cols = [
        "facility",
        "group",
        "origin_date",
        "horizon",
        "target_date",
        "target",
        "prediction",
    ]
    assert list(preds_df.columns) == expected_cols

    # Forecasts should be non-negative and valid numbers
    assert (preds_df["prediction"] >= 0).all()
    assert not preds_df["prediction"].isna().any()
    assert not preds_df["target"].isna().any()


def test_no_future_leakage_in_sarimax(synthetic_sarima_data):
    """Mutating data after origin date t should NOT change forecasts for origin t."""
    long_df, calendar_df, fac_state_df = synthetic_sarima_data
    series_dict, facility_exog_dict, _, _ = prepare_data_and_exog(
        long_df, calendar_df=calendar_df, facility_state_df=fac_state_df
    )

    fac = "Hospital Melaka"
    grp = "A"
    series = series_dict[(fac, grp)].copy()
    exog = facility_exog_dict[fac].copy()

    origin = "2023-01-09"

    preds_orig = fit_and_forecast_single_series(
        facility=fac,
        group=grp,
        series=series,
        exog_df=exog,
        origin_dates=[origin],
        train_start="2020-01-01",
        train_end="2022-12-31",
        horizon=14,
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 7),
    )

    # Mutate series target after origin date
    corrupted_series = series.copy()
    corrupted_series.loc[corrupted_series.index > origin] = 999999.0

    preds_corrupted = fit_and_forecast_single_series(
        facility=fac,
        group=grp,
        series=corrupted_series,
        exog_df=exog,
        origin_dates=[origin],
        train_start="2020-01-01",
        train_end="2022-12-31",
        horizon=14,
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 7),
    )

    np.testing.assert_array_almost_equal(
        preds_orig["prediction"].values, preds_corrupted["prediction"].values, decimal=4
    )
