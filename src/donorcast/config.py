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
MASE_WINDOW_START = "2020-01-01"
MASE_WINDOW_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2026-09-22"

# Calendar Configuration
CALENDAR_START = "2006-01-01"
CALENDAR_END = "2026-12-31"
FACILITY_STATE_FILE = DATA_EXTERNAL_DIR / "facility_state.csv"
STATE_WEEKENDS_FILE = DATA_EXTERNAL_DIR / "state_weekends.csv"
SCHOOL_HOLIDAYS_FILE = DATA_EXTERNAL_DIR / "school_holidays.csv"
MCO_PERIODS_FILE = DATA_EXTERNAL_DIR / "mco_periods.csv"
GENERAL_ELECTIONS_FILE = DATA_EXTERNAL_DIR / "general_elections.csv"
CALENDAR_PROCESSED_FILE = DATA_PROCESSED_DIR / "calendar.parquet"

HORIZON = 14
TRAIN_ORIGIN_STEP = 3
EVAL_ORIGIN_WEEKDAY = 0  # Monday
SHORTFALL_RATIO = 0.8
TYPICAL_YEARS = 3
SEED = 42

FEATURES_DIR = DATA_PROCESSED_DIR / "features"
DONATION_LAGS = [0, 1, 2, 6, 13, 20, 27]
FINAL_RUN_FILE = REPORTS_DIR / "final_run.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Shortfall Severity Ratios
SHORTFALL_HIGH_RATIO = 0.7
SHORTFALL_MEDIUM_RATIO = 0.8

# Historical Replay Origins for Dashboard Comparison
REPLAY_ORIGINS = [
    "2025-01-27",  # Monday before Chinese New Year 2025 (2025-01-29)
    "2025-03-24",  # Monday before Hari Raya Aidilfitri 2025 (2025-03-30)
    "2025-10-13",  # Monday before Deepavali 2025 (2025-10-20)
    "2025-05-12",  # Test origin 4: Post-festival / regular term Monday
    "2025-08-11",  # Test origin 5: Pre-National Day / school holiday window
    "2025-12-15",  # Test origin 6: Year-end school holiday season
    "2026-02-09",  # Test origin 7: Early 2026 Monday
    "2026-06-15",  # Test origin 8: Mid 2026 Monday
]
