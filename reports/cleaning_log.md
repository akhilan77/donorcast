# Data Cleaning & Invariant Verification Log

Generated for dataset cutoff `2026-09-22`.

## 1. Raw Data Input Summary
- `donations_facility.csv`: 219,530 rows
- `donations_state.csv`: 105,980 rows
- `newdonors_facility.csv`: 211,960 rows

## 2. Blank Hospital Filter
- Dropped `52,990` blank hospital rows from `donations_facility.csv`.
- Total donations in dropped rows: `28,467` (avg 0.54/row, ~0.2% of national total).
- Dropped `45,420` blank hospital rows from `newdonors_facility.csv`.
- Remaining valid facility-day rows: `166,540`.

## 3. Completeness & Span
- Number of unique collection facilities: `22`
- Date range: `2006-01-01` to `2026-09-22` (`7,570` days)
- Facility-date gaps: `0` facilities with gaps (All 22 facilities cover all 7,570 days).

## 4. Reconciliation Checks
- **National Daily Total vs State 'Malaysia' Sum**: `0` daily mismatches (Exact match on all days).
- **Location Breakdown (`centre + mobile == daily`)**: `0` mismatches.
- **Type Breakdown (`wholeblood + apheresis_platelet + apheresis_plasma + other == daily`)**: `0` mismatches.
- **Donor Status Breakdown (`new + regular + irregular == daily`)**: `0` mismatches.
- **Social Breakdown (`civilian + student + policearmy == daily`)**: `0` mismatches.
- **ABO Group Breakdown (`A + B + O + AB == daily`)**: `283` rows with slight discrepancies (0.17% of rows, differences: {1: 263, 2: 19, 3: 1}).

## 5. Processed Dataset
- File: `C:\Users\akhil\projects\donorcast\data\processed\long.parquet`
- Total rows: `666,160` (`22` facilities × 4 blood groups × `7570` days)
- Columns: `facility, date, group, donations, raw_location_mobile, raw_location_centre, raw_social_student, raw_donations_regular, raw_donations_new, raw_type_apheresis_platelet, raw_type_apheresis_plasma, raw_daily, raw_newdonor_17_24, raw_newdonor_total`
