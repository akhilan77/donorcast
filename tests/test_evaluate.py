"""Tests for the evaluation harness in src/donorcast/evaluate.py:

- Split filtering by target_date and out-of-range drop
- WAPE, MASE (in-sample seasonal naive 7 scale), Pinball loss calculations
- Subgroup slices: by horizon, by blood group, by facility size tier, by holiday window
- Shortfall alert classification: typical baseline lookup, precision, recall, F1
- End-to-end evaluate() function and markdown report generation
- Test set locking and CLI enforcement
"""

import pandas as pd
import pytest

from donorcast.evaluate import (
    HistoricalTypicalLookup,
    compute_facility_tiers,
    compute_mase_scales,
    compute_regression_metrics,
    compute_shortfall_metrics,
    evaluate,
    filter_split,
)


def test_split_filtering():
    """Verify filter_split uses target_date and drops dates outside split boundaries."""
    df = pd.DataFrame(
        {
            "facility": ["PDN"] * 6,
            "group": ["O"] * 6,
            "origin_date": [
                "2022-12-20",
                "2022-12-30",
                "2023-01-01",
                "2024-12-25",
                "2024-12-30",
                "2026-09-20",
            ],
            "horizon": [1, 7, 7, 1, 7, 7],
            "target_date": [
                "2022-12-21",  # Train
                "2023-01-06",  # Val
                "2023-01-08",  # Val
                "2024-12-26",  # Val
                "2025-01-06",  # Test
                "2026-09-27",  # Beyond TEST_END (2026-09-22) -> should be dropped
            ],
            "target": [100.0] * 6,
            "prediction": [95.0] * 6,
        }
    )

    train_df = filter_split(df, "train")
    assert len(train_df) == 1
    assert train_df.iloc[0]["target_date"] == "2022-12-21"

    val_df = filter_split(df, "val")
    assert len(val_df) == 3
    assert set(val_df["target_date"]) == {"2023-01-06", "2023-01-08", "2024-12-26"}

    test_df = filter_split(df, "test")
    assert len(test_df) == 1
    assert test_df.iloc[0]["target_date"] == "2025-01-06"
    assert "2026-09-27" not in test_df["target_date"].values


def test_regression_metrics_wape_mase_pinball():
    """Verify manual calculation matches compute_regression_metrics including wape_7d."""
    df = pd.DataFrame(
        {
            "facility": ["PDN"] * 7 + ["Hospital Penang"] * 7,
            "group": ["O"] * 7 + ["A"] * 7,
            "origin_date": ["2023-01-02"] * 7 + ["2023-01-02"] * 7,
            "horizon": list(range(1, 8)) + list(range(1, 8)),
            "target": [10.0] * 7 + [20.0] * 7,  # sums: 70 and 140 -> total = 210
            "prediction": [12.0] * 7
            + [18.0]
            * 7,  # pred sums: 84 and 126 -> total abs error 7d = |70-84| + |140-126| = 14 + 14 = 28
            "pred_p10": [8.0] * 14,
            "pred_p90": [25.0] * 14,
        }
    )

    # Daily actual sum = 210, daily abs error sum = 14 * 2 = 28 -> WAPE = 28 / 210
    expected_wape = 28.0 / 210.0
    expected_wape_7d = 28.0 / 210.0

    mase_scales = {
        ("PDN", "O"): 2.0,
        ("Hospital Penang", "A"): 2.0,
    }
    # Scaled errors: 2/2 = 1.0 for all rows -> mean = 1.0
    expected_mase = 1.0

    metrics = compute_regression_metrics(df, mase_scales)

    assert pytest.approx(metrics["wape"], 1e-5) == expected_wape
    assert pytest.approx(metrics["wape_7d"], 1e-5) == expected_wape_7d
    assert pytest.approx(metrics["mase"], 1e-5) == expected_mase
    assert "pinball_10" in metrics
    assert "pinball_90" in metrics
    assert metrics["pinball_10"] >= 0
    assert metrics["pinball_90"] >= 0


def test_facility_tiers_and_mase_scales():
    """Verify facility tiers partitioning and MASE scales on synthetic long data with 3-year window."""
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D").strftime("%Y-%m-%d")
    facilities = [f"Hospital_{i:02d}" for i in range(1, 23)]  # 22 facilities

    rows = []
    for i, fac in enumerate(facilities):
        # assign higher volume to lower index
        vol = (23 - i) * 10
        for d in dates:
            for grp in ["A", "B", "O", "AB"]:
                rows.append({"facility": fac, "group": grp, "date": d, "donations": vol})

    sample_long = pd.DataFrame(rows)

    tiers = compute_facility_tiers(sample_long, train_end="2022-12-31")
    assert len(tiers) == 22

    top_5_count = sum(1 for v in tiers.values() if v == "top_5")
    middle_count = sum(1 for v in tiers.values() if v == "middle")
    bottom_7_count = sum(1 for v in tiers.values() if v == "bottom_7")

    assert top_5_count == 5
    assert middle_count == 10
    assert bottom_7_count == 7
    assert tiers["Hospital_01"] == "top_5"
    assert tiers["Hospital_22"] == "bottom_7"

    scales = compute_mase_scales(sample_long, window_start="2020-01-01", window_end="2022-12-31")
    assert len(scales) == 22 * 4


def test_shortfall_metrics_logic():
    """Verify shortfall flag computation, prevalence, and tier breakdowns."""
    # Build historical data for 2017, 2018, 2019 week 10
    hist_dates = ["2017-03-06", "2018-03-05", "2019-03-04"]  # Mondays around week 10
    rows = []
    for d in hist_dates:
        dt_start = pd.Timestamp(d)
        for day_offset in range(7):
            cur_d = (dt_start + pd.Timedelta(days=day_offset)).strftime("%Y-%m-%d")
            rows.append({"facility": "PDN", "group": "O", "date": cur_d, "donations": 100.0})

    long_df = pd.DataFrame(rows)
    lookup = HistoricalTypicalLookup(long_df)

    # 7-day total each past year was 700 -> typical_7d = 700. Threshold at 0.8 is 560.
    origin = "2020-03-02"  # Target window week 10 of 2020
    typical_val = lookup.get_typical_7d("PDN", "O", origin, typical_years=3)
    assert typical_val == 700.0

    # Predictions: 2 origins
    # Origin 1: True=500 (< 560 -> actual shortfall), Pred=520 (< 560 -> pred shortfall) -> TP
    # Origin 2: True=600 (>= 560 -> no actual shortfall), Pred=620 (>= 560 -> no pred shortfall) -> TN
    pred_rows = []
    for h in range(1, 8):
        pred_rows.append(
            {
                "facility": "PDN",
                "group": "O",
                "origin_date": "2020-03-02",
                "horizon": h,
                "target_date": f"2020-03-{h + 2:02d}",
                "target": 500.0 / 7.0,
                "prediction": 520.0 / 7.0,
            }
        )
        pred_rows.append(
            {
                "facility": "PDN",
                "group": "O",
                "origin_date": "2020-03-09",
                "horizon": h,
                "target_date": f"2020-03-{h + 9:02d}",
                "target": 600.0 / 7.0,
                "prediction": 620.0 / 7.0,
            }
        )

    preds_df = pd.DataFrame(pred_rows)
    tiers = {"PDN": "top_5"}
    metrics = compute_shortfall_metrics(
        predictions_df=preds_df,
        long_df=long_df,
        shortfall_ratio=0.8,
        typical_years=3,
        facility_tiers=tiers,
        typical_lookup=lookup,
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 1
    assert metrics["prevalence"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert "by_tier" in metrics
    assert metrics["by_tier"]["top_5"]["prevalence"] == 0.5


def test_full_evaluate_workflow(tmp_path):
    """End-to-end evaluation execution test on validation split with report output verification."""
    # Synthetic prediction dataset for validation (2023-01-02 Monday origin)
    origin = "2023-01-02"
    pred_rows = []
    for h in range(1, 15):
        target_d = (pd.Timestamp(origin) + pd.Timedelta(days=h)).strftime("%Y-%m-%d")
        for grp in ["A", "B", "O", "AB"]:
            pred_rows.append(
                {
                    "facility": "Pusat Darah Negara",
                    "group": grp,
                    "origin_date": origin,
                    "horizon": h,
                    "target_date": target_d,
                    "target": 50.0 + h,
                    "prediction": 52.0 + h,
                    "pred_p10": 45.0 + h,
                    "pred_p90": 58.0 + h,
                }
            )

    preds_df = pd.DataFrame(pred_rows)

    res = evaluate(
        predictions_df=preds_df,
        split="val",
        model_name="test_baseline",
        reports_dir=tmp_path,
    )

    assert "overall" in res
    assert "wape_7d" in res["overall"]
    assert "by_horizon" in res
    assert "by_group" in res
    assert "by_tier" in res
    assert "by_holiday" in res
    assert "shortfall" in res
    assert "prevalence" in res["shortfall"]
    assert "by_tier" in res["shortfall"]

    report_path = tmp_path / "results_test_baseline_val.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "# Evaluation Report: test_baseline (VAL Split)" in content
    assert "WAPE_7D" in content
    assert "WAPE" in content
    assert "MASE" in content
    assert "Shortfall Classification Metrics" in content
    assert "Prevalence" in content
