import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# Add src to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from donorcast.config import (
    DATA_CUTOFF,
    DATA_RAW_DIR,
    HASHES_FILE,
    RAW_DATA_URLS,
)


def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_and_freeze_data():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    hashes = {}

    print(f"Downloading raw data from MoH Malaysia (cutoff: {DATA_CUTOFF})...")
    for filename, url in RAW_DATA_URLS.items():
        dest_path = DATA_RAW_DIR / filename
        print(f"Fetching {url} -> {dest_path}")
        urllib.request.urlretrieve(url, dest_path)

        file_hash = compute_sha256(dest_path)
        hashes[filename] = file_hash
        print(f"Saved {filename} (SHA-256: {file_hash})")

    # Record hashes
    with open(HASHES_FILE, "w") as f:
        json.dump(hashes, f, indent=2)
    print(f"Wrote hashes to {HASHES_FILE}")


if __name__ == "__main__":
    download_and_freeze_data()
