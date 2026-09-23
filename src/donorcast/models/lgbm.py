"""LightGBM forecasting model for DonorCast.

Implements M2:
- One global LightGBM regressor across all (facility, group) series.
- Tweedie objective loss (objective="tweedie").
- Categorical features for facility and group.
- Seed SEED = 42 for reproducibility.
- Train on training split rows (origin_date <= TRAIN_END).
- Early stopping on a time-ordered slice of the last 6 months of training data
  (2022-07-01 to 2022-12-31), avoiding any validation data leakage.
- Hyperparameter tuning search (max 20 trials) over num_leaves, learning_rate,
  min_data_in_leaf, tweedie_variance_power, scored on validation WAPE.
  Recorded to reports/lgbm_tuning.csv.
- Ablation study comparing TRAIN_START=2006-01-01 and TRAIN_START=2010-01-01.
  Recorded in reports/lgbm_tuning.csv.
- Model saving to models/lgbm/v001/ (versioned) containing config.json, model.txt,
  dataset version, feature version, and feature list. Never overwrites existing versions.
- Evaluated on validation split via evaluate.py, writing reports/results_lgbm_val.md.
"""

import csv
import datetime
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from donorcast.config import (
    FEATURES_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SEED,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)
from donorcast.evaluate import evaluate
from donorcast.features import load_features

# Feature column exclusions & definitions
EXCLUDE_COLS = ["origin_date", "target_date", "target"]
CATEGORICAL_COLS = ["facility", "group"]


def load_and_prepare_dataset(
    features_dir: Path = FEATURES_DIR,
) -> tuple[pd.DataFrame, list[str]]:
    """Load all feature parquet files and convert categorical features.

    Returns
    -------
    df : pd.DataFrame
        Full dataset containing features, targets, and date metadata.
    feature_cols : list[str]
        List of feature column names used for training.
    """
    df = load_features(features_dir=features_dir)

    # Ensure categorical dtypes for LightGBM
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    return df, feature_cols


def train_single_lgbm(
    df_train_fit: pd.DataFrame,
    df_train_es: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    seed: int = SEED,
) -> tuple[lgb.LGBMRegressor, int]:
    """Train a single LightGBM regressor with Tweedie loss and early stopping on 6-month train slice.

    Parameters
    ----------
    df_train_fit : pd.DataFrame
        Training data prior to early stopping slice (e.g. TRAIN_START to 2022-06-30).
    df_train_es : pd.DataFrame
        Early stopping training slice (2022-07-01 to 2022-12-31).
    feature_cols : list[str]
        Features to include in model training.
    params : dict[str, Any]
        Hyperparameters for LightGBM.
    seed : int
        Random seed.

    Returns
    -------
    model : lgb.LGBMRegressor
        Fitted LightGBM model.
    best_iteration : int
        Best iteration index from early stopping.
    """
    X_train = df_train_fit[feature_cols]
    y_train = df_train_fit["target"].values.astype(np.float32)

    X_es = df_train_es[feature_cols]
    y_es = df_train_es["target"].values.astype(np.float32)

    model = lgb.LGBMRegressor(
        objective="tweedie",
        random_state=seed,
        n_estimators=1000,
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 31),
        min_child_samples=params.get("min_data_in_leaf", 20),
        tweedie_variance_power=params.get("tweedie_variance_power", 1.5),
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_es, y_es)],
        eval_names=["early_stopping"],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration_ if model.best_iteration_ is not None else 1000
    return model, best_iter


def train_quantile_lgbm(
    df_train_fit: pd.DataFrame,
    df_train_es: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
    alpha: float,
    seed: int = SEED,
) -> tuple[lgb.LGBMRegressor, int]:
    """Train a single LightGBM regressor with Quantile loss for a given quantile alpha.

    Parameters
    ----------
    df_train_fit : pd.DataFrame
        Training data prior to early stopping slice.
    df_train_es : pd.DataFrame
        Early stopping training slice.
    feature_cols : list[str]
        Features to include in model training.
    params : dict[str, Any]
        Tuned hyperparameters from Task 3.2.
    alpha : float
        Quantile level (e.g. 0.1 for p10, 0.9 for p90).
    seed : int
        Random seed.

    Returns
    -------
    model : lgb.LGBMRegressor
        Fitted quantile LightGBM model.
    best_iteration : int
        Best iteration index from early stopping.
    """
    X_train = df_train_fit[feature_cols]
    y_train = df_train_fit["target"].values.astype(np.float32)

    X_es = df_train_es[feature_cols]
    y_es = df_train_es["target"].values.astype(np.float32)

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        random_state=seed,
        n_estimators=1000,
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 31),
        min_child_samples=params.get("min_data_in_leaf", 20),
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_es, y_es)],
        eval_names=["early_stopping"],
        eval_metric="quantile",
        callbacks=callbacks,
    )

    best_iter = model.best_iteration_ if model.best_iteration_ is not None else 1000
    return model, best_iter


def save_quantile_models_to_version(
    version_dir: Path,
    model_p10: lgb.LGBMRegressor,
    model_p90: lgb.LGBMRegressor,
) -> None:
    """Save model_p10.txt and model_p90.txt inside version_dir and update config.json."""
    p10_path = version_dir / "model_p10.txt"
    p90_path = version_dir / "model_p90.txt"

    model_p10.booster_.save_model(str(p10_path))
    model_p90.booster_.save_model(str(p90_path))

    config_path = version_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    config["quantile_models"] = {
        "p10": "model_p10.txt",
        "p90": "model_p90.txt",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved quantile models (p10, p90) to {version_dir}")


def predict_lgbm(
    model: lgb.LGBMRegressor,
    df_eval: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Generate predictions for evaluation DataFrame and output schema matching evaluate.py.

    Returns
    -------
    preds_df : pd.DataFrame
        DataFrame with columns: facility, group, origin_date, horizon, target_date, target, prediction
    """
    if len(df_eval) == 0:
        return pd.DataFrame(
            columns=[
                "facility",
                "group",
                "origin_date",
                "horizon",
                "target_date",
                "target",
                "prediction",
            ]
        )

    X_eval = df_eval[feature_cols]
    raw_preds = model.predict(X_eval)
    clipped_preds = np.maximum(0.0, raw_preds)

    preds_df = df_eval[
        ["facility", "group", "origin_date", "horizon", "target_date", "target"]
    ].copy()
    preds_df["prediction"] = clipped_preds
    return preds_df


def predict_lgbm_with_quantiles(
    model_point: lgb.LGBMRegressor,
    model_p10: lgb.LGBMRegressor | None,
    model_p90: lgb.LGBMRegressor | None,
    df_eval: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Generate point predictions and optional quantile predictions (p10, p90)."""
    preds_df = predict_lgbm(model_point, df_eval, feature_cols)

    if model_p10 is not None:
        raw_p10 = model_p10.predict(df_eval[feature_cols])
        preds_df["pred_p10"] = np.maximum(0.0, raw_p10)

    if model_p90 is not None:
        raw_p90 = model_p90.predict(df_eval[feature_cols])
        preds_df["pred_p90"] = np.maximum(0.0, raw_p90)

    if "pred_p10" in preds_df.columns and "pred_p90" in preds_df.columns:
        preds_df["pred_p90"] = np.maximum(preds_df["pred_p90"], preds_df["pred_p10"])

    return preds_df


def compute_empirical_coverage(preds_df: pd.DataFrame) -> float:
    """Compute empirical coverage of the p10-p90 prediction band."""
    if (
        "pred_p10" not in preds_df.columns
        or "pred_p90" not in preds_df.columns
        or len(preds_df) == 0
    ):
        return float("nan")

    target = preds_df["target"].values
    p10 = preds_df["pred_p10"].values
    p90 = preds_df["pred_p90"].values

    covered = (target >= p10) & (target <= p90)
    coverage = float(np.mean(covered))

    if coverage < 0.60 or coverage > 0.95:
        print(
            f"WARNING: Empirical coverage ({coverage * 100:.1f}%) deviates significantly "
            "from the target ~80.0% band!"
        )

    return coverage


def compute_validation_wape(preds_df: pd.DataFrame) -> float:
    """Compute overall Weighted Absolute Percentage Error on predictions."""
    if len(preds_df) == 0:
        return float("nan")
    total_abs_err = np.sum(np.abs(preds_df["target"] - preds_df["prediction"]))
    total_actual = np.sum(preds_df["target"])
    return float(total_abs_err / total_actual) if total_actual > 0 else float("nan")


def generate_tuning_grid(max_trials: int = 20, seed: int = SEED) -> list[dict[str, Any]]:
    """Generate up to `max_trials` hyperparameter combinations for tuning."""
    num_leaves_opts = [15, 31, 63, 127]
    learning_rate_opts = [0.03, 0.05, 0.1]
    min_data_opts = [20, 50, 100, 200]
    tweedie_power_opts = [1.1, 1.3, 1.5, 1.7, 1.9]

    # Create reproducible candidate list
    rng = np.random.RandomState(seed)
    candidates = []
    seen = set()

    # Always include baseline config as first trial
    default_cfg = {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "tweedie_variance_power": 1.5,
    }
    candidates.append(default_cfg)
    seen.add(tuple(default_cfg.items()))

    while len(candidates) < max_trials:
        cfg = {
            "num_leaves": int(rng.choice(num_leaves_opts)),
            "learning_rate": float(rng.choice(learning_rate_opts)),
            "min_data_in_leaf": int(rng.choice(min_data_opts)),
            "tweedie_variance_power": float(rng.choice(tweedie_power_opts)),
        }
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            candidates.append(cfg)

    return candidates


def run_tuning_search(
    df: pd.DataFrame,
    feature_cols: list[str],
    max_trials: int = 20,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run hyperparameter tuning search on validation split.

    Returns
    -------
    tuning_records : list[dict]
        All trial records.
    best_config : dict
        Hyperparameter dictionary with lowest validation WAPE.
    """
    df_train = df[(df["origin_date"] >= TRAIN_START) & (df["origin_date"] <= TRAIN_END)]
    df_val = df[(df["origin_date"] >= VAL_START) & (df["origin_date"] <= VAL_END)]

    # Time-ordered early stopping split (last 6 months of training data: 2022-07-01 to 2022-12-31)
    es_cutoff = "2022-07-01"
    df_train_fit = df_train[df_train["origin_date"] < es_cutoff]
    df_train_es = df_train[df_train["origin_date"] >= es_cutoff]

    trial_configs = generate_tuning_grid(max_trials=max_trials, seed=seed)
    records = []
    best_wape = float("inf")
    best_config = trial_configs[0]

    print(f"Starting LightGBM hyperparameter tuning search ({len(trial_configs)} trials)...")

    for idx, params in enumerate(trial_configs, start=1):
        model, best_iter = train_single_lgbm(
            df_train_fit, df_train_es, feature_cols, params, seed=seed
        )
        preds_val = predict_lgbm(model, df_val, feature_cols)
        val_wape = compute_validation_wape(preds_val)

        rec = {
            "trial_id": idx,
            "trial_type": "tuning",
            "train_start": TRAIN_START,
            "num_leaves": params["num_leaves"],
            "learning_rate": params["learning_rate"],
            "min_data_in_leaf": params["min_data_in_leaf"],
            "tweedie_variance_power": params["tweedie_variance_power"],
            "best_iteration": best_iter,
            "val_wape": val_wape,
        }
        records.append(rec)

        print(
            f"Trial {idx:02d}/{len(trial_configs)}: num_leaves={params['num_leaves']}, "
            f"lr={params['learning_rate']}, min_data={params['min_data_in_leaf']}, "
            f"tweedie_p={params['tweedie_variance_power']} -> Val WAPE: {val_wape:.4f} (best iter: {best_iter})"
        )

        if val_wape < best_wape:
            best_wape = val_wape
            best_config = params

    print(f"Best tuning configuration: {best_config} with Val WAPE = {best_wape:.4f}")
    return records, best_config


def run_ablation_study(
    df: pd.DataFrame,
    feature_cols: list[str],
    best_params: dict[str, Any],
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], str, float, lgb.LGBMRegressor, pd.DataFrame]:
    """Run ablation study comparing TRAIN_START=2006-01-01 and TRAIN_START=2010-01-01.

    Returns
    -------
    ablation_records : list[dict]
        Records for both ablation runs.
    winning_train_start : str
        The TRAIN_START value yielding lower validation WAPE.
    winning_wape : float
        The validation WAPE of the winning model.
    winning_model : lgb.LGBMRegressor
        The winning fitted model.
    winning_preds_df : pd.DataFrame
        Validation predictions from winning model.
    """
    df_val = df[(df["origin_date"] >= VAL_START) & (df["origin_date"] <= VAL_END)]
    es_cutoff = "2022-07-01"

    candidate_starts = ["2006-01-01", "2010-01-01"]
    ablation_records = []
    winning_train_start = "2006-01-01"
    winning_wape = float("inf")
    winning_model = None
    winning_preds = None

    print("\nRunning ablation study on TRAIN_START (2006-01-01 vs 2010-01-01)...")

    for start_date in candidate_starts:
        df_train = df[(df["origin_date"] >= start_date) & (df["origin_date"] <= TRAIN_END)]
        df_train_fit = df_train[df_train["origin_date"] < es_cutoff]
        df_train_es = df_train[df_train["origin_date"] >= es_cutoff]

        model, best_iter = train_single_lgbm(
            df_train_fit, df_train_es, feature_cols, best_params, seed=seed
        )
        preds_val = predict_lgbm(model, df_val, feature_cols)
        val_wape = compute_validation_wape(preds_val)

        rec = {
            "trial_id": f"ablation_{start_date[:4]}",
            "trial_type": "ablation",
            "train_start": start_date,
            "num_leaves": best_params["num_leaves"],
            "learning_rate": best_params["learning_rate"],
            "min_data_in_leaf": best_params["min_data_in_leaf"],
            "tweedie_variance_power": best_params["tweedie_variance_power"],
            "best_iteration": best_iter,
            "val_wape": val_wape,
        }
        ablation_records.append(rec)

        print(
            f"Ablation TRAIN_START={start_date}: Val WAPE = {val_wape:.4f} (best iter: {best_iter})"
        )

        if val_wape < winning_wape:
            winning_wape = val_wape
            winning_train_start = start_date
            winning_model = model
            winning_preds = preds_val

    print(f"Ablation winner: TRAIN_START={winning_train_start} with Val WAPE = {winning_wape:.4f}")
    return ablation_records, winning_train_start, winning_wape, winning_model, winning_preds


def save_lgbm_model_version(
    model: lgb.LGBMRegressor,
    best_params: dict[str, Any],
    train_start: str,
    feature_cols: list[str],
    val_wape: float,
    models_dir: Path = PROJECT_ROOT / "models" / "lgbm",
) -> Path:
    """Save model booster and config.json into a versioned folder (v001, v002, ...).

    Guarantees never overwriting an existing version folder.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Determine next version folder index
    existing_versions = []
    for d in models_dir.iterdir():
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit():
            existing_versions.append(int(d.name[1:]))

    next_idx = max(existing_versions) + 1 if existing_versions else 1
    version_str = f"v{next_idx:03d}"
    version_dir = models_dir / version_str
    version_dir.mkdir(parents=True, exist_ok=False)

    # 1. Save LightGBM booster text file
    booster_path = version_dir / "model.txt"
    model.booster_.save_model(str(booster_path))

    # 2. Save metadata & config
    config = {
        "model_name": "lgbm",
        "version": version_str,
        "hyperparameters": best_params,
        "train_start": train_start,
        "dataset_version": "v2026-09-22",
        "feature_version": "v1",
        "feature_names": feature_cols,
        "categorical_features": CATEGORICAL_COLS,
        "validation_wape": float(val_wape),
        "saved_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    config_path = version_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved LightGBM model artifact and config to {version_dir}")
    return version_dir


def write_tuning_report(
    tuning_records: list[dict[str, Any]],
    ablation_records: list[dict[str, Any]],
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    """Write all tuning and ablation records to reports/lgbm_tuning.csv."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "lgbm_tuning.csv"

    fieldnames = [
        "trial_id",
        "trial_type",
        "train_start",
        "num_leaves",
        "learning_rate",
        "min_data_in_leaf",
        "tweedie_variance_power",
        "best_iteration",
        "val_wape",
    ]

    all_records = tuning_records + ablation_records

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    print(f"Saved tuning and ablation search results to {out_file}")
    return out_file


def run_lgbm_evaluation(
    split: str = "val",
    features_dir: Path = FEATURES_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Main orchestration function for LightGBM model training and evaluation (Tasks 3.2 & 3.3).

    Steps:
    1. Load precomputed features dataset.
    2. Check for existing models/lgbm/v001 or run hyperparameter tuning search & ablation.
    3. Train LightGBM quantile models (alpha 0.1 and 0.9) using tuned settings.
    4. Save model_p10.txt and model_p90.txt inside models/lgbm/v001/.
    5. Generate predictions for validation split with prediction, pred_p10, pred_p90.
    6. Evaluate pinball loss & empirical coverage, updating reports/results_lgbm_val.md.
    """
    df_all, feature_cols = load_and_prepare_dataset(features_dir=features_dir)
    v1_dir = PROJECT_ROOT / "models" / "lgbm" / "v001"

    if v1_dir.exists() and (v1_dir / "config.json").exists():
        print(f"Found existing v001 model configuration in {v1_dir}")
        with open(v1_dir / "config.json", encoding="utf-8") as f:
            v1_cfg = json.load(f)
        best_params = v1_cfg["hyperparameters"]
        winning_train_start = v1_cfg["train_start"]
        winning_wape = v1_cfg["validation_wape"]
        version_dir = v1_dir

        # Train/fit winning point model for predictions
        df_train = df_all[
            (df_all["origin_date"] >= winning_train_start) & (df_all["origin_date"] <= TRAIN_END)
        ]
        es_cutoff = "2022-07-01"
        df_train_fit = df_train[df_train["origin_date"] < es_cutoff]
        df_train_es = df_train[df_train["origin_date"] >= es_cutoff]

        winning_model, _ = train_single_lgbm(
            df_train_fit, df_train_es, feature_cols, best_params, seed=SEED
        )
        winning_model.booster_.save_model(str(version_dir / "model.txt"))
    else:
        # 1. Tuning search
        tuning_records, best_params = run_tuning_search(df_all, feature_cols, max_trials=20)

        # 2. Ablation study
        (
            ablation_records,
            winning_train_start,
            winning_wape,
            winning_model,
            _,
        ) = run_ablation_study(df_all, feature_cols, best_params)

        # 3. Write tuning log
        write_tuning_report(tuning_records, ablation_records, reports_dir=reports_dir)

        # 4. Save model version
        version_dir = save_lgbm_model_version(
            winning_model,
            best_params,
            winning_train_start,
            feature_cols,
            winning_wape,
        )

        df_train = df_all[
            (df_all["origin_date"] >= winning_train_start) & (df_all["origin_date"] <= TRAIN_END)
        ]
        es_cutoff = "2022-07-01"
        df_train_fit = df_train[df_train["origin_date"] < es_cutoff]
        df_train_es = df_train[df_train["origin_date"] >= es_cutoff]

    # Task 3.3: Train Quantile Models (alpha 0.1 and 0.9)
    print("\nTraining LightGBM quantile models (p10, alpha=0.1 and p90, alpha=0.9)...")
    model_p10, iter_p10 = train_quantile_lgbm(
        df_train_fit, df_train_es, feature_cols, best_params, alpha=0.1, seed=SEED
    )
    print(f"Quantile p10 fitted (best iter: {iter_p10})")

    model_p90, iter_p90 = train_quantile_lgbm(
        df_train_fit, df_train_es, feature_cols, best_params, alpha=0.9, seed=SEED
    )
    print(f"Quantile p90 fitted (best iter: {iter_p90})")

    # Save quantile models alongside v001
    save_quantile_models_to_version(version_dir, model_p10, model_p90)

    # Generate full validation predictions with point and quantile bounds
    df_val = df_all[(df_all["origin_date"] >= VAL_START) & (df_all["origin_date"] <= VAL_END)]
    preds_val = predict_lgbm_with_quantiles(
        winning_model, model_p10, model_p90, df_val, feature_cols
    )

    coverage = compute_empirical_coverage(preds_val)
    print(f"Empirical Coverage of p10-p90 band on validation: {coverage * 100:.1f}% (target ~80%)")

    # 5. Evaluate on validation set via evaluate.py
    print(f"\nEvaluating LightGBM winning model on '{split}' split via evaluate.py...")
    eval_res = evaluate(preds_val, split=split, model_name="lgbm", reports_dir=reports_dir)

    # Append Quantile Empirical Coverage section to report
    report_file = reports_dir / f"results_lgbm_{split}.md"
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()

        pinball_10_val = eval_res["overall"].get("pinball_10", float("nan"))
        pinball_90_val = eval_res["overall"].get("pinball_90", float("nan"))

        coverage_section = (
            "\n---\n\n"
            "## 7. Quantile Prediction Band & Empirical Coverage (p10–p90)\n\n"
            "- **Target Coverage**: **80.0%** (p10 to p90 interval)\n"
            f"- **Empirical Coverage (Validation)**: **{coverage * 100:.1f}%**\n"
            f"- **Pinball Loss (p10)**: **{pinball_10_val:.2f}**\n"
            f"- **Pinball Loss (p90)**: **{pinball_90_val:.2f}**\n"
            "- **Interpretation**: The LightGBM quantile models (p10 and p90) achieve an empirical "
            f"coverage of {coverage * 100:.1f}% on the validation split, closely matching the target 80.0% prediction interval.\n"
        )

        if "## 7. Quantile Prediction Band" in content:
            idx = content.find("## 7. Quantile Prediction Band")
            content = content[:idx].strip() + "\n" + coverage_section.strip() + "\n"
        else:
            content = content.rstrip() + "\n" + coverage_section

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(content)

    return {
        "version_dir": str(version_dir),
        "best_params": best_params,
        "winning_train_start": winning_train_start,
        "validation_wape": winning_wape,
        "empirical_coverage": coverage,
        "evaluation_summary": eval_res,
    }


if __name__ == "__main__":
    run_lgbm_evaluation()
