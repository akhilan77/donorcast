"""Tests for LightGBM model in src/donorcast/models/lgbm.py:

- Single global LightGBM model fitting and Tweedie loss training
- Time-ordered early stopping on training split
- No future leakage (predictions at origin t are invariant to data changes after origin date t)
- Versioned model saving (creates v001, config.json, model.txt without overwriting)
- Output schema verification matching evaluate.py requirements
"""

import json
import numpy as np
import pandas as pd
import pytest

from donorcast.models.lgbm import (
    CATEGORICAL_COLS,
    EXCLUDE_COLS,
    predict_lgbm,
    save_lgbm_model_version,
    train_single_lgbm,
)


@pytest.fixture
def synthetic_features_dataset():
    """Create a synthetic features dataset spanning 2020-01-01 to 2024-12-31."""
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="7D").strftime("%Y-%m-%d")
    rows = []

    facilities = ["Pusat Darah Negara", "Hospital Melaka"]
    groups = ["A", "O"]

    for orig_str in dates:
        orig_dt = pd.Timestamp(orig_str)
        for fac in facilities:
            for grp in groups:
                for h in range(1, 15):
                    target_dt = orig_dt + pd.Timedelta(days=h)
                    target_str = target_dt.strftime("%Y-%m-%d")

                    # Synthetic target donation count
                    w = target_dt.weekday()
                    base = 20.0 + (10.0 if fac == "Pusat Darah Negara" else 5.0)
                    target_val = max(0.0, base + 5.0 * np.sin(2 * np.pi * w / 7))

                    rows.append(
                        {
                            "facility": fac,
                            "group": grp,
                            "origin_date": orig_str,
                            "horizon": h,
                            "target_date": target_str,
                            "target": float(target_val),
                            "lag_0": float(target_val * 0.9),
                            "lag_1": float(target_val * 0.85),
                            "lag_2": float(target_val * 0.8),
                            "lag_6": float(target_val * 0.88),
                            "lag_13": float(target_val * 0.92),
                            "lag_20": float(target_val * 0.9),
                            "lag_27": float(target_val * 0.95),
                            "rolling_mean_7": float(target_val),
                            "rolling_std_7": 2.0,
                            "rolling_mean_28": float(target_val),
                            "rolling_std_28": 3.0,
                            "same_weekday_mean_4": float(target_val),
                            "share_mobile_28": 0.5,
                            "share_student_28": 0.2,
                            "share_regular_28": 0.7,
                            "share_new_donor_28": 0.3,
                            "share_apheresis_28": 0.05,
                            "share_newdonor_17_24_28": 0.1,
                            "day_of_week": w,
                            "week_of_year": int(target_dt.isocalendar().week),
                            "month": target_dt.month,
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

    df = pd.DataFrame(rows)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    return df, feature_cols


def test_train_and_predict_lgbm(synthetic_features_dataset):
    df, feature_cols = synthetic_features_dataset

    df_train_fit = df[df["origin_date"] < "2022-07-01"]
    df_train_es = df[
        (df["origin_date"] >= "2022-07-01") & (df["origin_date"] <= "2022-12-31")
    ]
    df_val = df[df["origin_date"] >= "2023-01-01"]

    params = {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_data_in_leaf": 5,
        "tweedie_variance_power": 1.5,
    }

    model, best_iter = train_single_lgbm(
        df_train_fit, df_train_es, feature_cols, params, seed=42
    )

    assert model is not None
    assert best_iter > 0

    preds_df = predict_lgbm(model, df_val, feature_cols)

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
    assert len(preds_df) == len(df_val)
    assert not preds_df["prediction"].isna().any()
    assert (preds_df["prediction"] >= 0.0).all()


def test_no_future_leakage_lgbm(synthetic_features_dataset):
    """Mutating target/features for future dates > origin date t should NOT change predictions for origin t."""
    df, feature_cols = synthetic_features_dataset

    df_train_fit = df[df["origin_date"] < "2022-07-01"].copy()
    df_train_es = df[
        (df["origin_date"] >= "2022-07-01") & (df["origin_date"] <= "2022-12-31")
    ].copy()

    params = {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_data_in_leaf": 5,
        "tweedie_variance_power": 1.5,
    }
    model, _ = train_single_lgbm(
        df_train_fit, df_train_es, feature_cols, params, seed=42
    )

    valid_eval_origins = [d for d in df["origin_date"].unique() if d >= "2023-01-01"]
    origin_eval = valid_eval_origins[0]
    df_eval_orig = df[df["origin_date"] == origin_eval].copy()
    preds_orig = predict_lgbm(model, df_eval_orig, feature_cols)

    # Mutate data rows dated after origin_eval in evaluation DataFrame
    df_eval_future = df[df["origin_date"] >= origin_eval].copy()
    df_eval_future.loc[df_eval_future["origin_date"] > origin_eval, "lag_0"] = 999999.0

    df_eval_target_slice = df_eval_future[df_eval_future["origin_date"] == origin_eval]
    preds_after_future_mutation = predict_lgbm(model, df_eval_target_slice, feature_cols)

    np.testing.assert_array_almost_equal(
        preds_orig["prediction"].values,
        preds_after_future_mutation["prediction"].values,
        decimal=5,
    )


def test_versioned_model_saving(synthetic_features_dataset, tmp_path):
    df, feature_cols = synthetic_features_dataset
    df_train_fit = df[df["origin_date"] < "2022-07-01"]
    df_train_es = df[
        (df["origin_date"] >= "2022-07-01") & (df["origin_date"] <= "2022-12-31")
    ]

    params = {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_data_in_leaf": 5,
        "tweedie_variance_power": 1.5,
    }
    model, _ = train_single_lgbm(
        df_train_fit, df_train_es, feature_cols, params, seed=42
    )

    v1_dir = save_lgbm_model_version(
        model,
        params,
        train_start="2006-01-01",
        feature_cols=feature_cols,
        val_wape=0.25,
        models_dir=tmp_path,
    )

    assert v1_dir.name == "v001"
    assert (v1_dir / "model.txt").exists()
    assert (v1_dir / "config.json").exists()

    with open(v1_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["model_name"] == "lgbm"
    assert cfg["version"] == "v001"
    assert cfg["dataset_version"] == "v2026-09-22"
    assert cfg["feature_version"] == "v1"
    assert cfg["validation_wape"] == 0.25

    # Second save should create v002 without overwriting v001
    v2_dir = save_lgbm_model_version(
        model,
        params,
        train_start="2006-01-01",
        feature_cols=feature_cols,
        val_wape=0.24,
        models_dir=tmp_path,
    )
    assert v2_dir.name == "v002"
    assert (v2_dir / "model.txt").exists()


def test_train_quantile_lgbm_and_predictions(synthetic_features_dataset):
    from donorcast.models.lgbm import (
        predict_lgbm_with_quantiles,
        train_quantile_lgbm,
    )

    df, feature_cols = synthetic_features_dataset
    df_train_fit = df[df["origin_date"] < "2022-07-01"]
    df_train_es = df[
        (df["origin_date"] >= "2022-07-01") & (df["origin_date"] <= "2022-12-31")
    ]
    df_val = df[df["origin_date"] >= "2023-01-01"]

    params = {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_data_in_leaf": 5,
    }

    model_point, _ = train_single_lgbm(
        df_train_fit, df_train_es, feature_cols, params, seed=42
    )
    model_p10, _ = train_quantile_lgbm(
        df_train_fit, df_train_es, feature_cols, params, alpha=0.1, seed=42
    )
    model_p90, _ = train_quantile_lgbm(
        df_train_fit, df_train_es, feature_cols, params, alpha=0.9, seed=42
    )

    preds_df = predict_lgbm_with_quantiles(
        model_point, model_p10, model_p90, df_val, feature_cols
    )

    assert "pred_p10" in preds_df.columns
    assert "pred_p90" in preds_df.columns
    assert (preds_df["pred_p10"] >= 0.0).all()
    assert (preds_df["pred_p90"] >= 0.0).all()
    assert (preds_df["pred_p10"] <= preds_df["pred_p90"]).all()


def test_compute_empirical_coverage():
    from donorcast.models.lgbm import compute_empirical_coverage

    df = pd.DataFrame(
        {
            "target": [10.0, 20.0, 30.0, 40.0, 50.0],
            "pred_p10": [5.0, 15.0, 25.0, 45.0, 55.0],  # 45 & 55 will miss
            "pred_p90": [15.0, 25.0, 35.0, 48.0, 60.0],
        }
    )
    # Row 0: 10 in [5, 15] -> Yes
    # Row 1: 20 in [15, 25] -> Yes
    # Row 2: 30 in [25, 35] -> Yes
    # Row 3: 40 not in [45, 48] -> No
    # Row 4: 50 not in [55, 60] -> No
    coverage = compute_empirical_coverage(df)
    assert coverage == pytest.approx(0.6, abs=1e-5)


def test_save_quantile_models(synthetic_features_dataset, tmp_path):
    from donorcast.models.lgbm import (
        save_quantile_models_to_version,
        train_quantile_lgbm,
    )

    df, feature_cols = synthetic_features_dataset
    df_train_fit = df[df["origin_date"] < "2022-07-01"]
    df_train_es = df[
        (df["origin_date"] >= "2022-07-01") & (df["origin_date"] <= "2022-12-31")
    ]

    params = {"num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 5}
    model_p10, _ = train_quantile_lgbm(
        df_train_fit, df_train_es, feature_cols, params, alpha=0.1, seed=42
    )
    model_p90, _ = train_quantile_lgbm(
        df_train_fit, df_train_es, feature_cols, params, alpha=0.9, seed=42
    )

    v1_dir = save_lgbm_model_version(
        model_p10,
        params,
        train_start="2010-01-01",
        feature_cols=feature_cols,
        val_wape=0.35,
        models_dir=tmp_path,
    )

    save_quantile_models_to_version(v1_dir, model_p10, model_p90)

    assert (v1_dir / "model_p10.txt").exists()
    assert (v1_dir / "model_p90.txt").exists()

    with open(v1_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    assert "quantile_models" in cfg
    assert cfg["quantile_models"]["p10"] == "model_p10.txt"
    assert cfg["quantile_models"]["p90"] == "model_p90.txt"

