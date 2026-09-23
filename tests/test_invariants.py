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
