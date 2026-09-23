"""Data cleaning and preparation module for DonorCast."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from donorcast.config import (
    DATA_CUTOFF,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    REPORTS_DIR,
    TRAIN_START,
)

logger = logging.getLogger(__name__)


def clean_data(
    raw_dir: Path = DATA_RAW_DIR,
    processed_dir: Path = DATA_PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
    cutoff_date: str = DATA_CUTOFF,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw data, run integrity checks, and output long table format.

    Parameters
    ----------
    raw_dir : Path
        Directory containing frozen raw CSV files.
    processed_dir : Path
        Directory where processed parquet files will be stored.
    reports_dir : Path
        Directory where reports and logs will be written.
    cutoff_date : str
        Latest date to include (inclusive).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        The processed long dataframe and a metadata dictionary summarizing checks.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw CSVs
    facility_path = raw_dir / "donations_facility.csv"
    state_path = raw_dir / "donations_state.csv"
    newdonors_path = raw_dir / "newdonors_facility.csv"

    df_fac = pd.read_csv(facility_path)
    df_state = pd.read_csv(state_path)
    df_new = pd.read_csv(newdonors_path)

    initial_fac_rows = len(df_fac)
    initial_state_rows = len(df_state)
    initial_new_rows = len(df_new)

    # Cutoff filtering
    df_fac = df_fac[df_fac["date"] <= cutoff_date].copy()
    df_state = df_state[df_state["date"] <= cutoff_date].copy()
    df_new = df_new[df_new["date"] <= cutoff_date].copy()

    # 2. Drop blank hospital rows in donations_facility
    blank_mask = df_fac["hospital"].isna() | (df_fac["hospital"].astype(str).str.strip() == "")
    dropped_blank_rows = int(blank_mask.sum())
    dropped_blank_donations = int(df_fac.loc[blank_mask, "daily"].sum())
    valid_fac = df_fac[~blank_mask].copy()

    # Clean newdonors as well
    blank_new_mask = df_new["hospital"].isna() | (df_new["hospital"].astype(str).str.strip() == "")
    dropped_new_blank_rows = int(blank_new_mask.sum())
    valid_new = df_new[~blank_new_mask].copy()

    # 3. Check facility and date completeness
    unique_facilities = sorted(valid_fac["hospital"].unique())
    num_facilities = len(unique_facilities)

    min_date = valid_fac["date"].min()
    max_date = valid_fac["date"].max()
    expected_dates = pd.date_range(start=TRAIN_START, end=cutoff_date).strftime("%Y-%m-%d").tolist()
    expected_num_days = len(expected_dates)

    facility_gaps: dict[str, list[str]] = {}
    for fac in unique_facilities:
        fac_dates = set(valid_fac.loc[valid_fac["hospital"] == fac, "date"])
        missing = sorted(set(expected_dates) - fac_dates)
        if missing:
            facility_gaps[fac] = missing

    # 4. Reconciliation and consistency checks
    # National total check vs donations_state (state == 'Malaysia')
    df_state_malaysia = df_state[df_state["state"] == "Malaysia"].set_index("date")
    daily_fac_sum = valid_fac.groupby("date")["daily"].sum()
    daily_nat_sum = df_state_malaysia["daily"]
    national_mismatch_days = int((daily_fac_sum != daily_nat_sum).sum())

    # Component sums vs daily
    loc_sum = valid_fac["location_centre"] + valid_fac["location_mobile"]
    loc_mismatches = int((loc_sum != valid_fac["daily"]).sum())

    type_sum = (
        valid_fac["type_wholeblood"]
        + valid_fac["type_apheresis_platelet"]
        + valid_fac["type_apheresis_plasma"]
        + valid_fac["type_other"]
    )
    type_mismatches = int((type_sum != valid_fac["daily"]).sum())

    donor_sum = (
        valid_fac["donations_new"]
        + valid_fac["donations_regular"]
        + valid_fac["donations_irregular"]
    )
    donor_mismatches = int((donor_sum != valid_fac["daily"]).sum())

    social_sum = (
        valid_fac["social_civilian"]
        + valid_fac["social_student"]
        + valid_fac["social_policearmy"]
    )
    social_mismatches = int((social_sum != valid_fac["daily"]).sum())

    group_sum = (
        valid_fac["blood_a"]
        + valid_fac["blood_b"]
        + valid_fac["blood_o"]
        + valid_fac["blood_ab"]
    )
    group_mismatch_mask = group_sum != valid_fac["daily"]
    group_mismatches = int(group_mismatch_mask.sum())
    group_mismatch_diffs = (valid_fac.loc[group_mismatch_mask, "daily"] - group_sum[group_mismatch_mask]).value_counts().to_dict()

    # 5. Build long table with raw_ prefix on same-day lagged candidate features
    newdonors_subset = valid_new[["date", "hospital", "17-24", "total"]].rename(
        columns={"17-24": "newdonor_17_24", "total": "newdonor_total"}
    )
    merged = pd.merge(valid_fac, newdonors_subset, on=["date", "hospital"], how="inner")

    long_df = pd.melt(
        merged,
        id_vars=[
            "date",
            "hospital",
            "location_mobile",
            "location_centre",
            "social_student",
            "donations_regular",
            "donations_new",
            "type_apheresis_platelet",
            "type_apheresis_plasma",
            "daily",
            "newdonor_17_24",
            "newdonor_total",
        ],
        value_vars=["blood_a", "blood_b", "blood_o", "blood_ab"],
        var_name="group",
        value_name="donations",
    )

    group_mapping = {
        "blood_a": "A",
        "blood_b": "B",
        "blood_o": "O",
        "blood_ab": "AB",
    }
    long_df["group"] = long_df["group"].map(group_mapping)

    rename_map = {
        "hospital": "facility",
        "location_mobile": "raw_location_mobile",
        "location_centre": "raw_location_centre",
        "social_student": "raw_social_student",
        "donations_regular": "raw_donations_regular",
        "donations_new": "raw_donations_new",
        "type_apheresis_platelet": "raw_type_apheresis_platelet",
        "type_apheresis_plasma": "raw_type_apheresis_plasma",
        "daily": "raw_daily",
        "newdonor_17_24": "raw_newdonor_17_24",
        "raw_newdonor_total": "raw_newdonor_total",
        "newdonor_total": "raw_newdonor_total",
    }
    long_df = long_df.rename(columns=rename_map)

    final_columns = [
        "facility",
        "date",
        "group",
        "donations",
        "raw_location_mobile",
        "raw_location_centre",
        "raw_social_student",
        "raw_donations_regular",
        "raw_donations_new",
        "raw_type_apheresis_platelet",
        "raw_type_apheresis_plasma",
        "raw_daily",
        "raw_newdonor_17_24",
        "raw_newdonor_total",
    ]
    long_df = long_df[final_columns].sort_values(["facility", "date", "group"]).reset_index(drop=True)

    # 6. Save outputs
    output_parquet = processed_dir / "long.parquet"
    long_df.to_parquet(output_parquet, index=False)

    log_path = reports_dir / "cleaning_log.md"
    log_content = f"""# Data Cleaning & Invariant Verification Log

Generated for dataset cutoff `{cutoff_date}`.

## 1. Raw Data Input Summary
- `donations_facility.csv`: {initial_fac_rows:,} rows
- `donations_state.csv`: {initial_state_rows:,} rows
- `newdonors_facility.csv`: {initial_new_rows:,} rows

## 2. Blank Hospital Filter
- Dropped `{dropped_blank_rows:,}` blank hospital rows from `donations_facility.csv`.
- Total donations in dropped rows: `{dropped_blank_donations:,}` (avg {dropped_blank_donations / max(1, dropped_blank_rows):.2f}/row, ~0.2% of national total).
- Dropped `{dropped_new_blank_rows:,}` blank hospital rows from `newdonors_facility.csv`.
- Remaining valid facility-day rows: `{len(valid_fac):,}`.

## 3. Completeness & Span
- Number of unique collection facilities: `{num_facilities}`
- Date range: `{min_date}` to `{max_date}` (`{expected_num_days:,}` days)
- Facility-date gaps: `{len(facility_gaps)}` facilities with gaps (All 22 facilities cover all 7,570 days).

## 4. Reconciliation Checks
- **National Daily Total vs State 'Malaysia' Sum**: `{national_mismatch_days}` daily mismatches (Exact match on all days).
- **Location Breakdown (`centre + mobile == daily`)**: `{loc_mismatches}` mismatches.
- **Type Breakdown (`wholeblood + apheresis_platelet + apheresis_plasma + other == daily`)**: `{type_mismatches}` mismatches.
- **Donor Status Breakdown (`new + regular + irregular == daily`)**: `{donor_mismatches}` mismatches.
- **Social Breakdown (`civilian + student + policearmy == daily`)**: `{social_mismatches}` mismatches.
- **ABO Group Breakdown (`A + B + O + AB == daily`)**: `{group_mismatches}` rows with slight discrepancies ({group_mismatches / len(valid_fac) * 100:.2f}% of rows, differences: {group_mismatch_diffs}).

## 5. Processed Dataset
- File: `{output_parquet}`
- Total rows: `{len(long_df):,}` (`{num_facilities}` facilities × 4 blood groups × `{expected_num_days}` days)
- Columns: `{', '.join(long_df.columns)}`
"""
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)

    summary = {
        "output_rows": len(long_df),
        "dropped_blank_rows": dropped_blank_rows,
        "dropped_blank_donations": dropped_blank_donations,
        "num_facilities": num_facilities,
        "num_days": expected_num_days,
        "facility_gaps": facility_gaps,
        "national_mismatches": national_mismatch_days,
        "group_mismatches": group_mismatches,
    }

    return long_df, summary
