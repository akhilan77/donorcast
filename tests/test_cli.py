"""Tests for DonorCast command-line interface (CLI)."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from donorcast.cli import create_parser, main
from donorcast.config import DATA_CUTOFF


def test_create_parser_subcommands():
    """Verify that all expected subcommands are registered in the CLI parser."""
    parser = create_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    # Subparsers action
    subparsers_action = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers_action = action
            break

    assert subparsers_action is not None, "Missing subparsers in CLI parser"
    expected_cmds = {
        "clean",
        "features",
        "baselines",
        "train",
        "final",
        "explain",
        "alerts",
        "all",
    }
    assert expected_cmds.issubset(subparsers_action.choices.keys())


def test_all_subcommand_arguments():
    """Verify that 'donorcast all' accepts default and custom arguments."""
    parser = create_parser()

    # Default invocation
    args = parser.parse_args(["all"])
    assert args.command == "all"
    assert args.origin == DATA_CUTOFF
    assert args.precompute_replay is False

    # Custom invocation
    custom_args = parser.parse_args(
        ["all", "--origin", "2026-06-15", "--precompute-replay"]
    )
    assert custom_args.command == "all"
    assert custom_args.origin == "2026-06-15"
    assert custom_args.precompute_replay is True


def test_handle_all_execution_flow(capsys):
    """Verify that handle_all executes stages in order and prints summary."""
    parser = create_parser()
    args = parser.parse_args(["all", "--origin", "2026-09-22"])

    mock_clean = MagicMock(return_value=(None, {"output_rows": 666160}))
    mock_features = MagicMock(return_value=["features_2026.parquet"])
    mock_baselines = MagicMock(return_value={"summary_file": "reports/results_baselines.md"})
    mock_train = MagicMock(return_value={"version_dir": "models/lgbm/v001"})
    mock_alerts = MagicMock(return_value=[{"alert_id": 1}])

    with patch("donorcast.clean.clean_data", mock_clean), \
         patch("donorcast.features.generate_all_feature_datasets", mock_features), \
         patch("donorcast.models.baselines.run_baselines_evaluation", mock_baselines), \
         patch("donorcast.models.lgbm.run_lgbm_evaluation", mock_train), \
         patch("donorcast.shortfall.generate_and_save_alerts", mock_alerts):

        args.func(args)

    # Check that each mocked stage was called once
    mock_clean.assert_called_once()
    mock_features.assert_called_once()
    mock_baselines.assert_called_once_with(split="val")
    mock_train.assert_called_once_with(split="val")
    mock_alerts.assert_called_once_with(origin_date="2026-09-22")

    captured = capsys.readouterr().out
    assert "DONORCAST END-TO-END PIPELINE ORCHESTRATOR" in captured
    assert "DONORCAST PIPELINE EXECUTION SUMMARY" in captured
    assert "1. Data Cleaning & Integrity" in captured
    assert "2. Multi-Horizon Features" in captured
    assert "3. Baseline Models (M0, M0b)" in captured
    assert "4. Train LightGBM + Quantiles" in captured
    assert "5. Shortfall Alerts & Forecasts" in captured
    assert "TOTAL RUNTIME" in captured


def test_main_without_arguments(capsys):
    """Verify that invoking donorcast with no arguments prints help and exits with 0."""
    with patch("sys.argv", ["donorcast"]), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr().out
    assert "usage: donorcast" in captured
