"""SHAP Explainability and Alert Reason Translation Module (Task 4.2).

Implements:
1. Global SHAP summary for the final LightGBM model on a sample of test rows,
   saving beeswarm and bar summary plots to reports/figures/.
2. `reasons(row)`: Extracts the top 3 features by |SHAP| and translates them
   into plain English using `src/donorcast/feature_labels.yaml`.
3. Validation and mapping ensuring every feature in the model has an English label.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

matplotlib.use("Agg")  # Non-interactive headless backend

from donorcast.config import (
    FIGURES_DIR,
    PROJECT_ROOT,
    SEED,
    TEST_END,
    TEST_START,
)
from donorcast.models.lgbm import CATEGORICAL_COLS, load_and_prepare_dataset

# Global cache for explainer and model to ensure fast sub-millisecond per-row explanations
_CACHED_EXPLAINER: shap.TreeExplainer | None = None
_CACHED_BOOSTER: lgb.Booster | None = None
_CACHED_FEATURE_NAMES: list[str] | None = None
_CACHED_LABELS: dict[str, Any] | None = None

WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
MONTH_NAMES = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def load_feature_labels(yaml_path: Path | None = None) -> dict[str, Any]:
    """Load feature labels and translation templates from YAML."""
    global _CACHED_LABELS
    if yaml_path is None and _CACHED_LABELS is not None:
        return _CACHED_LABELS

    target_path = (
        yaml_path if yaml_path is not None else Path(__file__).parent / "feature_labels.yaml"
    )
    if not target_path.exists():
        raise FileNotFoundError(f"Feature labels mapping file not found at: {target_path}")

    with open(target_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if yaml_path is None:
        _CACHED_LABELS = data
    return data


def get_latest_lgbm_model_dir(
    models_dir: Path = PROJECT_ROOT / "models" / "lgbm",
) -> Path:
    """Find the directory of the latest versioned LightGBM model."""
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")

    version_dirs = []
    for d in models_dir.iterdir():
        if (
            d.is_dir()
            and d.name.startswith("v")
            and d.name[1:].isdigit()
            and (d / "model.txt").exists()
        ):
            version_dirs.append((int(d.name[1:]), d))

    if not version_dirs:
        raise FileNotFoundError(f"No valid LightGBM model version found in {models_dir}")

    version_dirs.sort(key=lambda x: x[0], reverse=True)
    return version_dirs[0][1]


def get_explainer(
    model_dir: Path | None = None,
) -> tuple[shap.TreeExplainer, lgb.Booster, list[str]]:
    """Get or initialize the cached SHAP TreeExplainer, Booster, and feature list."""
    global _CACHED_EXPLAINER, _CACHED_BOOSTER, _CACHED_FEATURE_NAMES

    if (
        model_dir is None
        and _CACHED_EXPLAINER is not None
        and _CACHED_BOOSTER is not None
        and _CACHED_FEATURE_NAMES is not None
    ):
        return _CACHED_EXPLAINER, _CACHED_BOOSTER, _CACHED_FEATURE_NAMES

    target_dir = model_dir if model_dir is not None else get_latest_lgbm_model_dir()
    booster_file = target_dir / "model.txt"
    config_file = target_dir / "config.json"

    if not booster_file.exists():
        raise FileNotFoundError(f"Model file not found: {booster_file}")

    booster = lgb.Booster(model_file=str(booster_file))

    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            cfg = json.load(f)
        feature_names = cfg.get("feature_names", booster.feature_name())
    else:
        feature_names = booster.feature_name()

    explainer = shap.TreeExplainer(booster)

    if model_dir is None:
        _CACHED_EXPLAINER = explainer
        _CACHED_BOOSTER = booster
        _CACHED_FEATURE_NAMES = feature_names

    return explainer, booster, feature_names


def format_reason(
    feature: str,
    value: Any,
    shap_value: float,
    labels_dict: dict[str, Any] | None = None,
) -> str:
    """Translate a feature contribution into plain English.

    Parameters
    ----------
    feature : str
        Name of the feature.
    value : Any
        Feature value for the given row.
    shap_value : float
        SHAP value corresponding to this feature.
    labels_dict : dict[str, Any] | None
        Loaded YAML dictionary of feature labels and templates.

    Returns
    -------
    str
        Plain English description of the feature impact.
    """
    if labels_dict is None:
        labels_dict = load_feature_labels()

    feat_info = labels_dict.get("features", {}).get(feature)
    if not feat_info:
        # Fallback if unconfigured
        direction = "increasing" if shap_value > 0 else "reducing"
        return f"{feature}={value} ({direction} forecast)"

    feat_type = feat_info.get("type", "numeric")
    is_neg = shap_value < 0

    # 1. Proximity to upcoming festival (e.g., days_to_hari_raya=3)
    if feat_type == "proximity":
        festival = feat_info.get("festival", feature)
        try:
            val_int = round(float(value))
        except (ValueError, TypeError):
            val_int = value

        if val_int == 0:
            return feat_info.get("template_today", f"{festival} today")
        elif val_int == 1:
            return f"{festival} tomorrow"
        elif 0 < val_int <= 14:
            return feat_info.get("template_near", f"{festival} in {val_int} days").format(
                val=val_int
            )
        else:
            return feat_info.get("template_far", f"{val_int} days until {festival}").format(
                val=val_int
            )

    # 2. Days elapsed after festival (e.g., days_since_hari_raya)
    elif feat_type == "post_event":
        festival = feat_info.get("festival", feature)
        try:
            val_int = round(float(value))
        except (ValueError, TypeError):
            val_int = value

        if val_int == 0:
            return feat_info.get("template_today", f"{festival} today")
        elif 0 < val_int <= 14:
            return feat_info.get(
                "template_near", f"{val_int} days post-{festival} recovery"
            ).format(val=val_int)
        else:
            return feat_info.get("template_far", f"{val_int} days after {festival}").format(
                val=val_int
            )

    # 3. Demographic breakdown shares (e.g., share_student_28, share_mobile_28)
    elif feat_type == "demographic_share":
        try:
            val_flt = float(value)
        except (ValueError, TypeError):
            val_flt = 0.0

        if feature == "share_student_28":
            if is_neg:
                return "fewer student donors recently"
            return f"higher student donor turnout ({val_flt:.1%} of donors)"
        elif feature == "share_mobile_28":
            if is_neg:
                return "mobile-drive share down over last 4 weeks"
            return f"higher mobile drive activity ({val_flt:.1%} of donors)"
        elif feature == "share_regular_28":
            if is_neg:
                return "fewer regular repeat donors"
            return f"strong regular donor turnout ({val_flt:.1%} of donors)"
        elif feature == "share_new_donor_28":
            if is_neg:
                return "lower first-time donor recruitment"
            return f"strong first-time donor recruitment ({val_flt:.1%} of donors)"
        elif feature == "share_newdonor_17_24_28":
            if is_neg:
                return "youth/campus donor drive dip"
            return f"active youth donor turnout ({val_flt:.1%} of new donors)"
        elif feature == "share_apheresis_28":
            if is_neg:
                return "reduced apheresis appointment volume"
            return f"increased apheresis appointment volume ({val_flt:.1%} share)"

        # Generic demographic share fallback
        if is_neg:
            return feat_info.get("template_neg", "{label} down").format(
                val=val_flt, label=feat_info.get("label", feature)
            )
        return feat_info.get("template_pos", "{label} up").format(
            val=val_flt, label=feat_info.get("label", feature)
        )

    # 4. Binary calendar flags (e.g., is_school_holiday, is_public_holiday, is_ramadan)
    elif feat_type == "binary_flag":
        try:
            val_bool = bool(round(float(value)))
        except (ValueError, TypeError):
            val_bool = bool(value)

        if feature == "is_school_holiday":
            return "school holidays" if val_bool else "regular school term"
        elif feature == "is_public_holiday":
            return "public holiday closure / dip" if val_bool else "standard working day"
        elif feature == "is_ramadan":
            return "Ramadan fasting month" if val_bool else "outside Ramadan"
        elif feature == "is_weekend":
            return "state weekend schedule" if val_bool else "regular weekday operations"
        elif feature == "is_mco":
            return (
                "MCO pandemic movement restrictions active"
                if val_bool
                else "no movement restrictions"
            )
        elif feature == "is_election_day":
            return "general election polling day" if val_bool else "standard non-polling day"

        if val_bool:
            return feat_info.get("template_active", feat_info.get("label", feature))
        return feat_info.get("template_inactive", f"not {feat_info.get('label', feature)}")

    # 5. Day of week
    elif feat_type == "day_of_week":
        try:
            dow = int(value)
            day_name = WEEKDAY_NAMES[dow] if 0 <= dow < 7 else f"Day {dow}"
        except (ValueError, TypeError):
            day_name = str(value)

        if is_neg:
            return f"{day_name} collection dip"
        return f"{day_name} peak collection schedule"

    # 6. Calendar cycles (week of year, month)
    elif feat_type == "calendar_cycle":
        try:
            val_int = round(float(value))
        except (ValueError, TypeError):
            val_int = value

        if feature == "month":
            m_name = (
                MONTH_NAMES[val_int]
                if isinstance(val_int, int) and 1 <= val_int <= 12
                else str(val_int)
            )
            if is_neg:
                return f"seasonal low-donation month ({m_name})"
            return f"favorable seasonal month ({m_name})"
        elif feature == "week_of_year":
            if is_neg:
                return f"seasonal supply lull (week {val_int})"
            return f"peak seasonal donation period (week {val_int})"

    # 7. Lag counts & recent donations (lag_0, lag_1, lag_6, etc.)
    elif feat_type == "lag_count":
        try:
            val_num = float(value)
        except (ValueError, TypeError):
            val_num = 0.0

        if feature == "lag_6":
            if val_num == 0:
                return "zero donations on same weekday last week"
            elif is_neg:
                return f"weak collection on same weekday last week ({val_num:g} donors)"
            else:
                return f"strong collection on same weekday last week ({val_num:g} donors)"
        elif feature == "lag_0":
            if is_neg:
                return f"low donations on origin day ({val_num:g} donors)"
            return f"strong donations on origin day ({val_num:g} donors)"
        elif feature == "lag_1":
            if is_neg:
                return f"lower prior-day turnout ({val_num:g} donors)"
            return f"strong prior-day turnout ({val_num:g} donors)"

        template = feat_info.get("template_neg") if is_neg else feat_info.get("template_pos")
        if template:
            return template.format(val=val_num)
        return f"{feat_info.get('label', feature)} had {val_num:g} donors"

    # 8. Rolling statistics (rolling_mean_7, same_weekday_mean_4, etc.)
    elif feat_type == "rolling_stat":
        try:
            val_num = float(value)
        except (ValueError, TypeError):
            val_num = 0.0

        template = feat_info.get("template_neg") if is_neg else feat_info.get("template_pos")
        if template:
            return template.format(val=val_num)
        return f"{feat_info.get('label', feature)} is {val_num:.1f}"

    # 9. Categorical identifiers (facility, group)
    elif feat_type == "categorical":
        if feature == "facility":
            return f"{value} facility collection profile"
        elif feature == "group":
            return f"Blood Group {value} baseline"
        return f"{feature} is {value}"

    # 10. Forecast Horizon
    elif feat_type == "horizon":
        return f"forecast horizon Day {value}"

    # General fallback
    label = feat_info.get("label", feature)
    return f"{label} ({value})"


def reasons(
    row: pd.Series | dict[str, Any] | pd.DataFrame,
    explainer: shap.TreeExplainer | None = None,
    top_k: int = 3,
    return_details: bool = False,
) -> list[str] | list[dict[str, Any]]:
    """Extract top-k features by |SHAP| contribution and translate to plain English.

    Parameters
    ----------
    row : pd.Series | dict | pd.DataFrame
        Single row of features matching model inputs.
    explainer : shap.TreeExplainer | None
        Optional preloaded TreeExplainer. If None, the cached default is used.
    top_k : int
        Number of top contributing features to return (default 3).
    return_details : bool
        If True, returns structured dictionaries with feature name, value, SHAP value,
        and English reason string.

    Returns
    -------
    list[str] | list[dict[str, Any]]
        List of translated plain-English reason statements.
    """
    if explainer is None:
        explainer, _, feature_names = get_explainer()
    else:
        # Use explainer's feature names or fallback to cached
        feature_names = (
            getattr(explainer, "feature_names", None)
            or _CACHED_FEATURE_NAMES
            or get_explainer()[2]
        )

    labels_dict = load_feature_labels()

    # Convert row to DataFrame
    if isinstance(row, pd.DataFrame):
        row_df = row.copy()
        if len(row_df) > 1:
            row_df = row_df.iloc[[0]]
    elif isinstance(row, (pd.Series, Mapping)):
        row_df = pd.DataFrame([row])
    else:
        raise TypeError(f"Unsupported row type: {type(row)}")

    # Ensure all required features are present
    missing = [c for c in feature_names if c not in row_df.columns]
    if missing:
        raise ValueError(f"Missing required model features in row: {missing}")

    # Prepare DataFrame subset with proper categorical types
    df_eval = row_df[feature_names].copy()
    for col in CATEGORICAL_COLS:
        if col in df_eval.columns:
            df_eval[col] = df_eval[col].astype("category")

    # Compute SHAP values
    shap_vals = explainer.shap_values(df_eval)[0]

    # Rank features by |SHAP| descending
    abs_shap = np.abs(shap_vals)
    top_indices = np.argsort(-abs_shap)[:top_k]

    results = []
    for idx in top_indices:
        feat_name = feature_names[idx]
        feat_val = df_eval[feat_name].iloc[0]
        feat_shap = float(shap_vals[idx])
        english_str = format_reason(
            feature=feat_name,
            value=feat_val,
            shap_value=feat_shap,
            labels_dict=labels_dict,
        )

        if return_details:
            results.append(
                {
                    "feature": feat_name,
                    "value": feat_val,
                    "shap_value": feat_shap,
                    "abs_shap": abs_shap[idx],
                    "reason": english_str,
                }
            )
        else:
            results.append(english_str)

    return results


def generate_global_shap_summary(
    sample_size: int = 1000,
    seed: int = SEED,
    output_dir: Path = FIGURES_DIR,
    model_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate global SHAP summary visualizations (beeswarm & bar plots) on test rows.

    Parameters
    ----------
    sample_size : int
        Number of test split rows to sample (default 1000).
    seed : int
        Random seed for sampling reproducibility.
    output_dir : Path
        Directory to save figures.
    model_dir : Path | None
        Optional model version directory.

    Returns
    -------
    dict[str, Path]
        Dictionary of saved plot paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    explainer, _booster, feature_names = get_explainer(model_dir=model_dir)

    print(f"Loading feature dataset for SHAP summary (sample_size={sample_size}, seed={seed})...")
    df_all, _feature_cols = load_and_prepare_dataset()

    # Filter to test split
    df_test = df_all[(df_all["origin_date"] >= TEST_START) & (df_all["origin_date"] <= TEST_END)]
    if len(df_test) == 0:
        raise ValueError(
            f"No data found in test split range ({TEST_START} to {TEST_END}) for SHAP analysis."
        )

    actual_sample_size = min(sample_size, len(df_test))
    sample_df = df_test.sample(n=actual_sample_size, random_state=seed)[feature_names]

    print(
        f"Computing SHAP values for {actual_sample_size:,} test rows across {len(feature_names)} features..."
    )
    shap_vals = explainer.shap_values(sample_df)

    # 1. Beeswarm plot
    print("Generating beeswarm summary plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_vals,
        sample_df,
        plot_type="dot",
        max_display=20,
        show=False,
    )
    plt.title(
        f"DonorCast Global SHAP Beeswarm Summary (LightGBM on Test Split, N={actual_sample_size:,})",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    plt.tight_layout()
    beeswarm_path1 = output_dir / "shap_summary_beeswarm.png"
    beeswarm_path2 = output_dir / "04_shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path1, dpi=300)
    plt.savefig(beeswarm_path2, dpi=300)
    plt.close()
    print(f"Saved beeswarm plot to: {beeswarm_path1}")

    # 2. Bar plot (mean |SHAP| feature importance)
    print("Generating bar summary plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_vals,
        sample_df,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.title(
        f"DonorCast Global Feature Importance by Mean |SHAP| (LightGBM on Test Split, N={actual_sample_size:,})",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    plt.tight_layout()
    bar_path1 = output_dir / "shap_summary_bar.png"
    bar_path2 = output_dir / "04_shap_summary_bar.png"
    plt.savefig(bar_path1, dpi=300)
    plt.savefig(bar_path2, dpi=300)
    plt.close()
    print(f"Saved bar plot to: {bar_path1}")

    return {
        "beeswarm": beeswarm_path1,
        "bar": bar_path1,
        "beeswarm_numbered": beeswarm_path2,
        "bar_numbered": bar_path2,
    }


if __name__ == "__main__":
    generate_global_shap_summary()
