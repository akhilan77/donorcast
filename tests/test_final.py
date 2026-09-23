"""Tests for the final evaluation pipeline in src/donorcast/final.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from donorcast.final import (
    format_final_results_markdown,
    get_test_origins,
    save_final_lgbm_model,
)


def test_get_test_origins():
    """Verify test origin generation."""
    origins = get_test_origins(test_start="2025-01-01", test_end="2026-09-22", weekday=0)
    assert len(origins) == 90
    assert origins[0] == "2025-01-06"
    assert origins[-1] == "2026-09-21"


def test_format_final_results_markdown():
    """Verify final results markdown formatting with dummy evaluation outputs."""
    dummy_results = {
        "overall": {
            "wape": 0.38,
            "wape_7d": 0.18,
            "mase": 0.75,
            "pinball_10": 1.35,
            "pinball_90": 2.05,
            "count": 1000,
        },
        "shortfall": {
            "total_windows": 100,
            "actual_shortfalls": 20,
            "predicted_shortfalls": 18,
            "prevalence": 0.20,
            "precision": 0.65,
            "recall": 0.50,
            "f1": 0.56,
            "tp": 10,
            "fp": 8,
            "fn": 10,
            "tn": 72,
            "by_tier": {
                "top_5": {
                    "prevalence": 0.15,
                    "precision": 0.60,
                    "recall": 0.40,
                    "f1": 0.48,
                    "total_windows": 25,
                },
                "middle": {
                    "prevalence": 0.20,
                    "precision": 0.65,
                    "recall": 0.50,
                    "f1": 0.56,
                    "total_windows": 45,
                },
                "bottom_7": {
                    "prevalence": 0.25,
                    "precision": 0.70,
                    "recall": 0.60,
                    "f1": 0.64,
                    "total_windows": 30,
                },
            },
        },
        "by_horizon": {
            1: {"wape": 0.35, "mase": 0.70, "count": 100},
            2: {"wape": 0.36, "mase": 0.72, "count": 100},
        },
        "by_group": {
            "A": {"wape_7d": 0.18, "wape": 0.38, "mase": 0.75, "count": 250},
            "B": {"wape_7d": 0.18, "wape": 0.38, "mase": 0.75, "count": 250},
            "O": {"wape_7d": 0.17, "wape": 0.36, "mase": 0.73, "count": 250},
            "AB": {"wape_7d": 0.26, "wape": 0.55, "mase": 0.80, "count": 250},
        },
        "by_tier": {
            "top_5": {"wape_7d": 0.15, "wape": 0.30, "mase": 0.78, "count": 250},
            "middle": {"wape_7d": 0.20, "wape": 0.42, "mase": 0.74, "count": 450},
            "bottom_7": {"wape_7d": 0.29, "wape": 0.65, "mase": 0.77, "count": 300},
        },
        "by_holiday": {
            "holiday": {"wape_7d": 0.19, "wape": 0.48, "mase": 0.95, "count": 100},
            "non_holiday": {"wape_7d": 0.18, "wape": 0.37, "mase": 0.74, "count": 900},
        },
    }

    dummy_m0 = {
        "overall": {"wape": 0.55, "wape_7d": 0.26, "mase": 1.10, "count": 1000},
        "shortfall": dummy_results["shortfall"],
        "by_horizon": dummy_results["by_horizon"],
        "by_group": dummy_results["by_group"],
        "by_tier": dummy_results["by_tier"],
        "by_holiday": dummy_results["by_holiday"],
    }
    dummy_m0b = {
        "overall": {"wape": 0.46, "wape_7d": 0.21, "mase": 0.90, "count": 1000},
        "shortfall": dummy_results["shortfall"],
        "by_horizon": dummy_results["by_horizon"],
        "by_group": dummy_results["by_group"],
        "by_tier": dummy_results["by_tier"],
        "by_holiday": dummy_results["by_holiday"],
    }

    md = format_final_results_markdown(
        results_m0=dummy_m0,
        results_m0b=dummy_m0b,
        results_lgbm=dummy_results,
        coverage=0.812,
        final_version_str="v003",
    )

    assert "# Final Evaluation Report: Test Split" in md
    assert "BASELINE_MUST_BE_BEATEN" in md
    assert "M0b" in md
    assert "M2 (LightGBM)" in md
    assert "Empirical Coverage (Test Split)" in md
    assert "81.2%" in md


def test_save_final_lgbm_model(tmp_path: Path):
    """Verify that save_final_lgbm_model creates expected directory and files."""
    # Create mock LightGBM models
    mock_booster = MagicMock()
    mock_booster.save_model = MagicMock(
        side_effect=lambda path: Path(path).write_text("dummy model")
    )

    mock_model = MagicMock()
    mock_model.booster_ = mock_booster

    mock_p10 = MagicMock()
    mock_p10.booster_ = mock_booster

    mock_p90 = MagicMock()
    mock_p90.booster_ = mock_booster

    # Pre-create v001 in tmp_path
    (tmp_path / "v001").mkdir()

    v_dir = save_final_lgbm_model(
        model=mock_model,
        model_p10=mock_p10,
        model_p90=mock_p90,
        best_params={"num_leaves": 127},
        train_start="2010-01-01",
        train_end="2024-12-31",
        feature_cols=["facility", "group"],
        test_wape_7d=0.185,
        coverage=0.81,
        models_dir=tmp_path,
    )

    assert v_dir.name == "v002"
    assert (v_dir / "model.txt").exists()
    assert (v_dir / "model_p10.txt").exists()
    assert (v_dir / "model_p90.txt").exists()
    assert (v_dir / "config.json").exists()

    with open(v_dir / "config.json") as f:
        cfg = json.load(f)
    assert cfg["version"] == "v002"
    assert cfg["is_final"] is True
    assert cfg["test_wape_7d"] == 0.185
    assert cfg["train_end"] == "2024-12-31"


def test_final_run_guard(tmp_path: Path, monkeypatch):
    """Verify that run_final_evaluation aborts if final_run.json already exists without force."""
    import donorcast.final as final_mod

    fake_final_file = tmp_path / "final_run.json"
    fake_final_file.write_text('{"status": "already_done"}')

    monkeypatch.setattr(final_mod, "FINAL_RUN_FILE", fake_final_file)

    with pytest.raises(RuntimeError, match="TEST_SET_TOUCHED_ONCE"):
        final_mod.run_final_evaluation(force=False, reports_dir=tmp_path)
