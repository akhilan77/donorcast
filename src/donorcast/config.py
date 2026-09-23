from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_EXTERNAL_DIR = DATA_DIR / "external"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw Data Sources & Snapshot
DATA_CUTOFF = "2026-09-22"
RAW_DATA_URLS = {
    "donations_facility.csv": "https://raw.githubusercontent.com/MoH-Malaysia/data-darah-public/main/donations_facility.csv",
    "donations_state.csv": "https://raw.githubusercontent.com/MoH-Malaysia/data-darah-public/main/donations_state.csv",
    "newdonors_facility.csv": "https://raw.githubusercontent.com/MoH-Malaysia/data-darah-public/main/newdonors_facility.csv",
}
HASHES_FILE = DATA_RAW_DIR / "hashes.json"

# Date Splits & Model Config
TRAIN_START = "2006-01-01"
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2026-09-22"

HORIZON = 14
TRAIN_ORIGIN_STEP = 3
EVAL_ORIGIN_WEEKDAY = 0  # Monday
SHORTFALL_RATIO = 0.8
TYPICAL_YEARS = 3
SEED = 42
