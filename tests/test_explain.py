"""Unit tests for the explainability and reason translation module (Task 4.2)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from donorcast.config import PROJECT_ROOT
from donorcast.explain import (
    format_reason,
    generate_global_shap_summary,
    get_explainer,
    load_feature_labels,
    reasons,
)


def test_all_model_features_have_labels():
    """Verify that every feature in the final LightGBM model has a configured label and template."""
    v3_config = PROJECT_ROOT / "models" / "lgbm" / "v003" / "config.json"
    assert v3_config.exists(), f"Final model config not found at {v3_config}"

    with open(v3_config, encoding="utf-8") as f:
        cfg = json.load(f)

    model_features = cfg["feature_names"]
    labels_dict = load_feature_labels()
    configured_features = labels_dict.get("features", {})

    missing = [f for f in model_features if f not in configured_features]
    assert not missing, f"The following features are missing from feature_labels.yaml: {missing}"

    # Verify format_reason can format every feature without error
    for feat in model_features:
        # Test positive impact
        res_pos = format_reason(feat, value=3.0, shap_value=1.5, labels_dict=labels_dict)
        assert isinstance(res_pos, str) and len(res_pos) > 0

        # Test negative impact
        res_neg = format_reason(feat, value=0.0, shap_value=-1.5, labels_dict=labels_dict)
        assert isinstance(res_neg, str) and len(res_neg) > 0


def test_format_reason_specific_cases():
    """Verify exact plain-English phrasings for holiday proximity, shares, and calendar flags."""
    labels = load_feature_labels()

    # Hari Raya proximity
    r1 = format_reason("days_to_hari_raya", value=3, shap_value=-2.5, labels_dict=labels)
    assert "Hari Raya in 3 days" in r1

    r2 = format_reason("days_to_hari_raya", value=0, shap_value=-5.0, labels_dict=labels)
    assert "Hari Raya today" in r2

    r_post = format_reason("days_since_hari_raya", value=4, shap_value=1.2, labels_dict=labels)
    assert "4 days post-Hari Raya recovery" in r_post

    # Demographic shares
    r_student = format_reason("share_student_28", value=0.02, shap_value=-1.8, labels_dict=labels)
    assert "fewer student donors recently" in r_student

    r_mobile = format_reason("share_mobile_28", value=0.10, shap_value=-3.0, labels_dict=labels)
    assert "mobile-drive share down over last 4 weeks" in r_mobile

    # Calendar flags
    r_school = format_reason("is_school_holiday", value=1, shap_value=-2.0, labels_dict=labels)
    assert "school holidays" in r_school

    r_holiday = format_reason("is_public_holiday", value=1, shap_value=-4.0, labels_dict=labels)
    assert "public holiday closure / dip" in r_holiday

    r_ramadan = format_reason("is_ramadan", value=1, shap_value=-3.5, labels_dict=labels)
    assert "Ramadan fasting month" in r_ramadan

    # Lags
    r_lag6 = format_reason("lag_6", value=0, shap_value=-3.0, labels_dict=labels)
    assert "zero donations on same weekday last week" in r_lag6


def test_reasons_returns_top_3():
    """Verify that reasons(row) extracts exactly top 3 features by |SHAP| and returns English strings."""
    _, _, feature_names = get_explainer()

    # Create dummy row containing all feature names
    dummy_row = {}
    for col in feature_names:
        if col == "facility":
            dummy_row[col] = "Hospital Melaka"
        elif col == "group":
            dummy_row[col] = "O"
        elif col == "days_to_hari_raya":
            dummy_row[col] = 3
        elif col == "share_student_28":
            dummy_row[col] = 0.05
        elif col == "is_school_holiday":
            dummy_row[col] = 1
        elif "share_" in col:
            dummy_row[col] = 0.20
        elif "is_" in col:
            dummy_row[col] = 0
        elif "days_" in col:
            dummy_row[col] = 50
        else:
            dummy_row[col] = 10.0

    res = reasons(dummy_row, top_k=3)
    assert isinstance(res, list)
    assert len(res) == 3
    for s in res:
        assert isinstance(s, str)
        assert len(s) > 0

    # Also test with return_details=True
    details = reasons(dummy_row, top_k=3, return_details=True)
    assert isinstance(details, list)
    assert len(details) == 3
    for d in details:
        assert "feature" in d
        assert "shap_value" in d
        assert "abs_shap" in d
        assert "reason" in d


def test_reasons_ranking_with_mock_explainer():
    """Verify that features are ranked strictly by absolute SHAP magnitude."""
    mock_explainer = MagicMock()
    feature_names = ["days_to_hari_raya", "share_student_28", "is_school_holiday", "lag_0"]
    mock_explainer.feature_names = feature_names

    # Mock shap values: days_to_hari_raya = -5.0 (|5.0|), lag_0 = +3.0 (|3.0|), share_student_28 = -2.0 (|2.0|), is_school_holiday = 0.5 (|0.5|)
    mock_shap = np.array([[-5.0, -2.0, 0.5, 3.0]])
    mock_explainer.shap_values.return_value = mock_shap

    row_data = {
        "days_to_hari_raya": 3,
        "share_student_28": 0.05,
        "is_school_holiday": 1,
        "lag_0": 15.0,
    }

    results = reasons(row_data, explainer=mock_explainer, top_k=3, return_details=True)

    ranked_features = [r["feature"] for r in results]
    assert ranked_features == ["days_to_hari_raya", "lag_0", "share_student_28"]
    assert "Hari Raya in 3 days" in results[0]["reason"]
    assert "fewer student donors recently" in results[2]["reason"]


def test_generate_global_shap_summary_small(tmp_path):
    """Verify that global SHAP summary runs and writes both beeswarm and bar plots."""
    out = generate_global_shap_summary(sample_size=15, seed=42, output_dir=tmp_path)

    assert Path(out["beeswarm"]).exists()
    assert Path(out["bar"]).exists()
    assert Path(out["beeswarm_numbered"]).exists()
    assert Path(out["bar_numbered"]).exists()

    assert Path(out["beeswarm"]).stat().st_size > 1000
    assert Path(out["bar"]).stat().st_size > 1000


def test_generate_action_prescription():
    """Verify operational action prescription generator output and logic."""
    from donorcast.explain import generate_action_prescription

    # Holiday reason
    p1 = generate_action_prescription(
        ["Hari Raya in 4 days", "school holidays", "mobile-drive share down"],
        group="O",
        unit_deficit=95.0,
        facility="Hospital Melaka",
    )
    assert "priority recall SMS" in p1
    assert "regular group O donors" in p1
    assert "buffer transfer" in p1

    # Student reason
    p2 = generate_action_prescription(
        ["fewer student donors recently", "school holidays"],
        group="A",
        unit_deficit=30.0,
        facility="Hospital Pulau Pinang",
    )
    assert "Redirect mobile blood collection teams" in p2
