"""Tests for PyTorch LSTM model in src/donorcast/models/lstm.py:

- SeriesScaler transformation and inverse transformation
- DonorLSTM network architecture and forward pass shapes
- Dataset sample building and tensor indexing
- Training and early stopping on synthetic time series dataset
- No future leakage (predictions at origin t are invariant to data changes after origin date t)
- Versioned model saving (creates v001, config.json, model.pt without overwriting)
- Output schema verification matching evaluate.py requirements
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from donorcast.models.lstm import (
    CAL_DIM,
    LOOKBACK,
    DonorLSTM,
    SeriesScaler,
    build_samples_for_origins,
    predict_lstm,
    prepare_lstm_data,
    save_lstm_model_version,
    train_lstm_model,
)


@pytest.fixture
def synthetic_lstm_data():
    """Create a synthetic 4-year long dataset and calendar structure."""
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D").strftime("%Y-%m-%d")
    rows = []

    facilities = ["Pusat Darah Negara", "Hospital Melaka"]
    groups = ["A", "O"]

    for d_str in dates:
        d_dt = pd.Timestamp(d_str)
        w = d_dt.weekday()
        for fac in facilities:
            for grp in groups:
                base = 20.0 + (10.0 if fac == "Pusat Darah Negara" else 5.0)
                val = max(0.0, base + 5.0 * np.sin(2 * np.pi * w / 7))
                rows.append({"facility": fac, "group": grp, "date": d_str, "donations": float(val)})

    long_df = pd.DataFrame(rows)

    cal_rows = []
    for st in ["W.P. Kuala Lumpur", "Melaka"]:
        for d_str in dates:
            d_dt = pd.Timestamp(d_str)
            w = d_dt.weekday()
            cal_rows.append(
                {
                    "date": d_str,
                    "state": st,
                    "day_of_week": w,
                    "week_of_year": int(d_dt.isocalendar().week),
                    "month": d_dt.month,
                    "is_weekend": 1 if w >= 5 else 0,
                    "is_public_holiday": 0,
                    "is_ramadan": 0,
                    "days_to_hari_raya": 99,
                    "days_since_hari_raya": 99,
                    "days_to_aidiladha": 99,
                    "days_since_aidiladha": 99,
                    "days_to_cny": 99,
                    "days_since_cny": 99,
                    "days_to_deepavali": 99,
                    "days_since_deepavali": 99,
                    "is_school_holiday": 0,
                    "is_mco": 0,
                    "is_election_day": 0,
                }
            )
    calendar_df = pd.DataFrame(cal_rows)

    facility_state_df = pd.DataFrame(
        [
            {"facility": "Pusat Darah Negara", "state": "W.P. Kuala Lumpur"},
            {"facility": "Hospital Melaka", "state": "Melaka"},
        ]
    )

    return long_df, calendar_df, facility_state_df


def test_series_scaler(synthetic_lstm_data):
    long_df, _, _ = synthetic_lstm_data
    scaler = SeriesScaler()
    scaler.fit(long_df, train_end="2022-12-31")

    assert ("Pusat Darah Negara", "A") in scaler.means
    assert ("Hospital Melaka", "O") in scaler.stds

    raw = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    transformed = scaler.transform(raw, "Hospital Melaka", "O")
    reconstructed = scaler.inverse_transform(transformed, "Hospital Melaka", "O")

    np.testing.assert_array_almost_equal(raw, reconstructed, decimal=4)


def test_donor_lstm_forward_shape():
    model = DonorLSTM(
        num_facilities=2,
        num_groups=2,
        hidden_size=32,
        horizon=14,
        cal_dim=CAL_DIM,
    )

    batch_size = 4
    x_seq = torch.randn(batch_size, LOOKBACK, 1)
    fac_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    grp_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    cal_feats = torch.randn(batch_size, 14 * CAL_DIM)

    out = model(x_seq, fac_idx, grp_idx, cal_feats)

    assert out.shape == (batch_size, 14)


def test_train_and_predict_lstm(synthetic_lstm_data):
    long_df, calendar_df, facility_state_df = synthetic_lstm_data
    (
        _series_matrices,
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
        scaler,
        _cal_scaler,
    ) = prepare_lstm_data(long_df, calendar_df, facility_state_df)

    train_origins = ["2021-06-07", "2021-06-14", "2021-06-21"]
    es_origins = ["2022-08-01", "2022-08-08"]
    val_origins = ["2023-01-09", "2023-01-16"]

    train_samples = build_samples_for_origins(
        train_origins,
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
    )
    es_samples = build_samples_for_origins(
        es_origins,
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
    )
    val_samples = build_samples_for_origins(
        val_origins,
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
    )

    assert len(train_samples) == len(train_origins) * 4
    assert len(val_samples) == len(val_origins) * 4

    model, best_epoch, _best_es_loss = train_lstm_model(
        train_samples,
        es_samples,
        num_facilities=2,
        num_groups=2,
        hidden_size=16,
        batch_size=8,
        max_epochs=2,
        lr=0.01,
        patience=2,
        seed=42,
    )

    assert model is not None
    assert best_epoch >= 1

    preds_df = predict_lstm(model, val_samples, scaler, batch_size=8)

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
    assert len(preds_df) == len(val_origins) * 4 * 14
    assert not preds_df["prediction"].isna().any()
    assert (preds_df["prediction"] >= 0.0).all()


def test_no_future_leakage_lstm(synthetic_lstm_data):
    """Mutating target data after origin date t should NOT change predictions for origin t."""
    long_df, calendar_df, facility_state_df = synthetic_lstm_data
    (
        _,
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
        scaler,
        _cal_scaler,
    ) = prepare_lstm_data(long_df, calendar_df, facility_state_df)

    train_origins = ["2021-06-07"]
    es_origins = ["2022-08-01"]
    origin_val = "2023-01-09"

    train_samples = build_samples_for_origins(
        train_origins,
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
    )
    es_samples = build_samples_for_origins(
        es_origins, scaled_series_matrices, fac_to_idx, grp_to_idx, all_dates, facility_cal_matrices
    )
    val_samples = build_samples_for_origins(
        [origin_val],
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
    )

    model, _, _ = train_lstm_model(
        train_samples,
        es_samples,
        num_facilities=2,
        num_groups=2,
        hidden_size=16,
        batch_size=4,
        max_epochs=2,
        patience=2,
        seed=42,
    )

    preds_orig = predict_lstm(model, val_samples, scaler)

    # Mutate data in series matrix after origin_val date in future
    corrupted_matrices = {k: v.copy() for k, v in scaled_series_matrices.items()}
    val_idx = all_dates.index(origin_val)
    for matrix in corrupted_matrices.values():
        matrix[val_idx + 1 :] = 999999.0

    val_samples_corrupted = build_samples_for_origins(
        [origin_val], corrupted_matrices, fac_to_idx, grp_to_idx, all_dates, facility_cal_matrices
    )

    preds_corrupted = predict_lstm(model, val_samples_corrupted, scaler)

    np.testing.assert_array_almost_equal(
        preds_orig["prediction"].values, preds_corrupted["prediction"].values, decimal=4
    )


def test_versioned_lstm_saving(synthetic_lstm_data, tmp_path):
    long_df, calendar_df, facility_state_df = synthetic_lstm_data
    (
        _,
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
        scaler,
        _cal_scaler,
    ) = prepare_lstm_data(long_df, calendar_df, facility_state_df)

    train_samples = build_samples_for_origins(
        ["2021-06-07"],
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
    )
    es_samples = build_samples_for_origins(
        ["2022-08-01"],
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
    )

    model, best_epoch, best_es_loss = train_lstm_model(
        train_samples,
        es_samples,
        num_facilities=2,
        num_groups=2,
        hidden_size=16,
        batch_size=4,
        max_epochs=2,
        patience=2,
        seed=42,
    )

    v1_dir = save_lstm_model_version(
        model, scaler, best_epoch, best_es_loss, val_wape=0.32, models_dir=tmp_path
    )

    assert v1_dir.name == "v001"
    assert (v1_dir / "model.pt").exists()
    assert (v1_dir / "config.json").exists()

    with open(v1_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["model_name"] == "lstm"
    assert cfg["version"] == "v001"
    assert cfg["validation_wape"] == 0.32

    # Saving again creates v002
    v2_dir = save_lstm_model_version(
        model, scaler, best_epoch, best_es_loss, val_wape=0.31, models_dir=tmp_path
    )
    assert v2_dir.name == "v002"
    assert (v2_dir / "model.pt").exists()
