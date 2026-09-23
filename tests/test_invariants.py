import hashlib
import json
from pathlib import Path

from donorcast.config import DATA_RAW_DIR, HASHES_FILE, RAW_DATA_URLS


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def test_raw_data_frozen():
    """Invariant: RAW_DATA_FROZEN.

    Files in data/raw/ must exist and match their recorded SHA-256 hashes.
    """
    assert HASHES_FILE.exists(), f"Missing hashes file at {HASHES_FILE}"

    with open(HASHES_FILE) as f:
        recorded_hashes = json.load(f)

    for filename in RAW_DATA_URLS:
        file_path = DATA_RAW_DIR / filename
        assert file_path.exists(), f"Raw file {filename} does not exist at {file_path}"
        assert filename in recorded_hashes, f"No recorded hash for {filename} in {HASHES_FILE}"

        current_hash = compute_sha256(file_path)
        assert current_hash == recorded_hashes[filename], (
            f"Hash mismatch for {filename}! Invariant RAW_DATA_FROZEN violated.\n"
            f"Expected: {recorded_hashes[filename]}\n"
            f"Actual:   {current_hash}"
        )


def test_time_ordered_splits():
    """Invariant: TIME_ORDERED_SPLITS.

    Splits are strictly chronological without overlap or random splits.
    Train <= 2022-12-31 < Val <= 2024-12-31 < Test <= 2026-09-22.
    """
    import pandas as pd

    from donorcast.config import (
        TEST_END,
        TEST_START,
        TRAIN_END,
        TRAIN_START,
        VAL_END,
        VAL_START,
    )
    from donorcast.evaluate import filter_split

    t_start = pd.Timestamp(TRAIN_START)
    t_end = pd.Timestamp(TRAIN_END)
    v_start = pd.Timestamp(VAL_START)
    v_end = pd.Timestamp(VAL_END)
    te_start = pd.Timestamp(TEST_START)
    te_end = pd.Timestamp(TEST_END)

    assert t_start < t_end
    assert t_end < v_start <= v_end
    assert v_end < te_start <= te_end

    # Test filtering behavior by target_date
    dummy_df = pd.DataFrame(
        {
            "facility": ["PDN"] * 5,
            "group": ["A"] * 5,
            "origin_date": ["2022-12-25", "2022-12-31", "2023-01-01", "2024-12-31", "2025-01-01"],
            "horizon": [1, 7, 7, 7, 7],
            "target_date": ["2022-12-26", "2023-01-07", "2023-01-08", "2025-01-07", "2025-01-08"],
            "target": [10.0, 15.0, 20.0, 25.0, 30.0],
            "prediction": [10.0, 15.0, 20.0, 25.0, 30.0],
        }
    )

    train_filtered = filter_split(dummy_df, "train")
    assert list(train_filtered["target_date"]) == ["2022-12-26"]

    val_filtered = filter_split(dummy_df, "validation")
    assert list(val_filtered["target_date"]) == ["2023-01-07", "2023-01-08"]

    test_filtered = filter_split(dummy_df, "test")
    assert list(test_filtered["target_date"]) == ["2025-01-07", "2025-01-08"]


def test_test_set_touched_once(tmp_path):
    """Invariant: TEST_SET_TOUCHED_ONCE.

    The test-period evaluation runs only when allow_test=True (via donorcast final),
    which records a timestamp in final_run.json and prevents re-running without --force.
    """
    import pandas as pd
    import pytest

    from donorcast.evaluate import evaluate

    dummy_pred = pd.DataFrame(
        {
            "facility": ["PDN"] * 7,
            "group": ["A"] * 7,
            "origin_date": ["2025-01-06"] * 7,
            "horizon": list(range(1, 8)),
            "target_date": [
                "2025-01-07",
                "2025-01-08",
                "2025-01-09",
                "2025-01-10",
                "2025-01-11",
                "2025-01-12",
                "2025-01-13",
            ],
            "target": [50.0] * 7,
            "prediction": [48.0] * 7,
        }
    )

    # 1. Calling evaluate with split='test' and allow_test=False MUST raise PermissionError
    with pytest.raises(PermissionError, match="TEST_SET_TOUCHED_ONCE"):
        evaluate(dummy_pred, split="test", allow_test=False, reports_dir=tmp_path)

    # 2. First call with allow_test=True succeeds and writes final_run.json
    final_json = tmp_path / "final_run.json"
    import donorcast.evaluate as eval_module

    orig_final_file = eval_module.FINAL_RUN_FILE
    eval_module.FINAL_RUN_FILE = final_json
    try:
        if final_json.exists():
            final_json.unlink()

        res = evaluate(dummy_pred, split="test", allow_test=True, reports_dir=tmp_path)
        assert res is not None
        assert final_json.exists()

        with open(final_json) as f:
            meta = json.load(f)
        assert meta["split"] == "test"
        assert "evaluated_at" in meta

        # 3. Second call with allow_test=True but force=False MUST raise RuntimeError
        with pytest.raises(RuntimeError, match="already evaluated"):
            evaluate(dummy_pred, split="test", allow_test=True, force=False, reports_dir=tmp_path)

        # 4. Calling with force=True succeeds again
        res_forced = evaluate(
            dummy_pred, split="test", allow_test=True, force=True, reports_dir=tmp_path
        )
        assert res_forced is not None
    finally:
        eval_module.FINAL_RUN_FILE = orig_final_file
        if final_json.exists():
            final_json.unlink()
