"""Feature engineering pipeline for direct multi-horizon forecasting in DonorCast.

Each row represents: (facility, group, origin_date t, horizon h in 1..HORIZON)
Target = donations on target_date (t + h).

Features are strictly computed from data dated <= t:
- Lags of donations at t, t-1, t-2, t-6, t-13, t-20, t-27
- Rolling mean and std over 7 and 28 days ending at t
- Same weekday as target: mean of the last 4 occurrences up to t
- Lagged shares over 28 days ending at t:
  mobile, student, regular, new-donor, apheresis, new donors aged 17-24
- Calendar features for the target date t+h (known in advance)
- Horizon h, facility and group categoricals.

Memory optimization:
- Uses float32 for continuous features.
- Saves partitioned Parquet files by year in data/processed/features/.
"""

import datetime
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from donorcast.calendar import build_calendar_dataframe, load_facility_state_mapping
from donorcast.config import (
    CALENDAR_PROCESSED_FILE,
    DATA_PROCESSED_DIR,
    EVAL_ORIGIN_WEEKDAY,
    FACILITY_STATE_FILE,
    FEATURES_DIR,
    HORIZON,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_ORIGIN_STEP,
    TRAIN_START,
    VAL_END,
    VAL_START,
)

CALENDAR_FEATURE_COLS = [
    "day_of_week",
    "week_of_year",
    "month",
    "is_weekend",
    "is_public_holiday",
    "is_ramadan",
    "days_to_hari_raya",
    "days_since_hari_raya",
    "days_to_aidiladha",
    "days_since_aidiladha",
    "days_to_cny",
    "days_since_cny",
    "days_to_deepavali",
    "days_since_deepavali",
    "is_school_holiday",
    "is_mco",
    "is_election_day",
]


def get_all_train_origins(
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    step: int = TRAIN_ORIGIN_STEP,
) -> list[str]:
    """Generate training origin dates spaced by `step` days."""
    dates = pd.date_range(start=train_start, end=train_end, freq=f"{step}D")
    return dates.strftime("%Y-%m-%d").tolist()


def get_all_eval_origins(
    start_date: str = VAL_START,
    end_date: str = TEST_END,
    weekday: int = EVAL_ORIGIN_WEEKDAY,
) -> list[str]:
    """Generate evaluation origin dates (e.g. every Monday) between start and end date."""
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    eval_dates = [d.strftime("%Y-%m-%d") for d in all_dates if d.weekday() == weekday]
    return eval_dates


class FeaturePrecomputer:
    """Fast precomputed historical lookup structures for time-series and breakdown features."""

    def __init__(self, long_df: pd.DataFrame, facility_state_df: pd.DataFrame):
        self.long_df = long_df.copy()
        self.facility_state_df = facility_state_df.copy()

        # Map facility to state
        self.facility_to_state = dict(
            zip(self.facility_state_df["facility"], self.facility_state_df["state"])
        )

        # Unique facilities and groups
        self.facilities = sorted(long_df["facility"].unique())
        self.groups = sorted(long_df["group"].unique())
        self.facility_group_pairs = [(fac, grp) for fac in self.facilities for grp in self.groups]

        # Ensure sorted by date
        self.all_dates = sorted(long_df["date"].unique())
        self.date_to_idx = {d: i for i, d in enumerate(self.all_dates)}
        self.n_dates = len(self.all_dates)

        # Build pivot tables: date x (facility, group) for donations
        piv_donations = (
            self.long_df.pivot(index="date", columns=["facility", "group"], values="donations")
            .reindex(self.all_dates)
            .fillna(0)
        )
        self.donations_matrix = piv_donations.values.astype(np.float32)  # shape (n_dates, n_series)
        self.series_cols = list(piv_donations.columns)  # list of (facility, group)
        self.series_to_col_idx = {s: i for i, s in enumerate(self.series_cols)}
        self.facility_arr = np.array([s[0] for s in self.series_cols])
        self.group_arr = np.array([s[1] for s in self.series_cols])
        self.state_arr = np.array([self.facility_to_state[s[0]] for s in self.series_cols])

        # Build facility-level 28-day rolling share matrices
        fac_day = (
            self.long_df.groupby(["facility", "date"], as_index=False)
            .first()
            .sort_values(["facility", "date"])
        )

        self.fac_to_idx = {f: i for i, f in enumerate(self.facilities)}
        self.series_fac_idx = np.array([self.fac_to_idx[s[0]] for s in self.series_cols])

        def _make_pivot(col: str) -> np.ndarray:
            piv = (
                fac_day.pivot(index="date", columns="facility", values=col)
                .reindex(self.all_dates)
                .fillna(0)
            )
            piv = piv[self.facilities]
            return piv.values.astype(np.float32)

        raw_daily = _make_pivot("raw_daily")
        raw_mobile = _make_pivot("raw_location_mobile")
        raw_student = _make_pivot("raw_social_student")
        raw_regular = _make_pivot("raw_donations_regular")
        raw_new = _make_pivot("raw_donations_new")
        raw_apheresis = _make_pivot("raw_type_apheresis_platelet") + _make_pivot(
            "raw_type_apheresis_plasma"
        )
        raw_newdonor_17_24 = _make_pivot("raw_newdonor_17_24")
        raw_newdonor_total = _make_pivot("raw_newdonor_total")

        def _rolling_sum_28(mat: np.ndarray) -> np.ndarray:
            cumsum = np.vstack(
                [np.zeros((1, mat.shape[1]), dtype=np.float32), np.cumsum(mat, axis=0)]
            )
            roll = np.zeros_like(mat)
            for i in range(len(mat)):
                start_idx = max(0, i - 27)
                roll[i] = cumsum[i + 1] - cumsum[start_idx]
            return roll

        daily_28 = _rolling_sum_28(raw_daily)
        mobile_28 = _rolling_sum_28(raw_mobile)
        student_28 = _rolling_sum_28(raw_student)
        regular_28 = _rolling_sum_28(raw_regular)
        new_28 = _rolling_sum_28(raw_new)
        apheresis_28 = _rolling_sum_28(raw_apheresis)
        new17_28 = _rolling_sum_28(raw_newdonor_17_24)
        newtot_28 = _rolling_sum_28(raw_newdonor_total)

        with np.errstate(divide="ignore", invalid="ignore"):
            self.share_mobile_28 = np.where(daily_28 > 0, mobile_28 / daily_28, 0.0).astype(
                np.float32
            )
            self.share_student_28 = np.where(daily_28 > 0, student_28 / daily_28, 0.0).astype(
                np.float32
            )
            self.share_regular_28 = np.where(daily_28 > 0, regular_28 / daily_28, 0.0).astype(
                np.float32
            )
            self.share_new_donor_28 = np.where(daily_28 > 0, new_28 / daily_28, 0.0).astype(
                np.float32
            )
            self.share_apheresis_28 = np.where(daily_28 > 0, apheresis_28 / daily_28, 0.0).astype(
                np.float32
            )
            self.share_newdonor_17_24_28 = np.where(
                newtot_28 > 0, new17_28 / newtot_28, 0.0
            ).astype(np.float32)

        self.rolling_mean_7 = np.zeros_like(self.donations_matrix)
        self.rolling_std_7 = np.zeros_like(self.donations_matrix)
        self.rolling_mean_28 = np.zeros_like(self.donations_matrix)
        self.rolling_std_28 = np.zeros_like(self.donations_matrix)

        for i in range(self.n_dates):
            w7 = self.donations_matrix[max(0, i - 6) : i + 1]
            self.rolling_mean_7[i] = np.mean(w7, axis=0)
            self.rolling_std_7[i] = np.std(w7, axis=0)

            w28 = self.donations_matrix[max(0, i - 27) : i + 1]
            self.rolling_mean_28[i] = np.mean(w28, axis=0)
            self.rolling_std_28[i] = np.std(w28, axis=0)


def build_features_for_origins(
    origin_dates: Sequence[str],
    long_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    facility_state_df: pd.DataFrame,
    horizon: int = HORIZON,
    precomputer: FeaturePrecomputer | None = None,
) -> pd.DataFrame:
    """Build feature table for the specified origin dates up to horizon h in 1..HORIZON."""
    if precomputer is None:
        precomputer = FeaturePrecomputer(long_df, facility_state_df)

    # Pre-index calendar as dictionary of arrays for high performance: (state, date) -> dict
    cal_dict = {}
    for row in calendar_df.itertuples():
        key = (row.state, row.date)
        cal_dict[key] = {col: getattr(row, col) for col in CALENDAR_FEATURE_COLS}

    valid_origins = [o for o in origin_dates if o in precomputer.date_to_idx]
    if not valid_origins:
        return pd.DataFrame()

    n_series = len(precomputer.series_cols)
    total_blocks = len(valid_origins) * horizon
    total_rows = total_blocks * n_series

    # Pre-allocate output arrays
    out_facility = np.tile(precomputer.facility_arr, total_blocks)
    out_group = np.tile(precomputer.group_arr, total_blocks)
    series_fac_idx = precomputer.series_fac_idx

    out_origin_date = np.empty(total_rows, dtype=object)
    out_target_date = np.empty(total_rows, dtype=object)
    out_horizon = np.empty(total_rows, dtype=np.int16)
    out_target = np.empty(total_rows, dtype=np.float32)

    lag_0 = np.empty(total_rows, dtype=np.float32)
    lag_1 = np.empty(total_rows, dtype=np.float32)
    lag_2 = np.empty(total_rows, dtype=np.float32)
    lag_6 = np.empty(total_rows, dtype=np.float32)
    lag_13 = np.empty(total_rows, dtype=np.float32)
    lag_20 = np.empty(total_rows, dtype=np.float32)
    lag_27 = np.empty(total_rows, dtype=np.float32)

    r_mean_7 = np.empty(total_rows, dtype=np.float32)
    r_std_7 = np.empty(total_rows, dtype=np.float32)
    r_mean_28 = np.empty(total_rows, dtype=np.float32)
    r_std_28 = np.empty(total_rows, dtype=np.float32)
    sw_mean_4 = np.empty(total_rows, dtype=np.float32)

    sh_mobile = np.empty(total_rows, dtype=np.float32)
    sh_student = np.empty(total_rows, dtype=np.float32)
    sh_regular = np.empty(total_rows, dtype=np.float32)
    sh_new = np.empty(total_rows, dtype=np.float32)
    sh_apheresis = np.empty(total_rows, dtype=np.float32)
    sh_new17 = np.empty(total_rows, dtype=np.float32)

    cal_arrays = {col: np.empty(total_rows, dtype=np.int16) for col in CALENDAR_FEATURE_COLS}

    block_idx = 0
    for origin in valid_origins:
        t_idx = precomputer.date_to_idx[origin]
        origin_dt = datetime.date.fromisoformat(origin)

        # Lags
        l0 = precomputer.donations_matrix[t_idx] if t_idx >= 0 else np.zeros(n_series, np.float32)
        l1 = (
            precomputer.donations_matrix[t_idx - 1]
            if t_idx >= 1
            else np.zeros(n_series, np.float32)
        )
        l2 = (
            precomputer.donations_matrix[t_idx - 2]
            if t_idx >= 2
            else np.zeros(n_series, np.float32)
        )
        l6 = (
            precomputer.donations_matrix[t_idx - 6]
            if t_idx >= 6
            else np.zeros(n_series, np.float32)
        )
        l13 = (
            precomputer.donations_matrix[t_idx - 13]
            if t_idx >= 13
            else np.zeros(n_series, np.float32)
        )
        l20 = (
            precomputer.donations_matrix[t_idx - 20]
            if t_idx >= 20
            else np.zeros(n_series, np.float32)
        )
        l27 = (
            precomputer.donations_matrix[t_idx - 27]
            if t_idx >= 27
            else np.zeros(n_series, np.float32)
        )

        rm7 = precomputer.rolling_mean_7[t_idx]
        rs7 = precomputer.rolling_std_7[t_idx]
        rm28 = precomputer.rolling_mean_28[t_idx]
        rs28 = precomputer.rolling_std_28[t_idx]

        smob = precomputer.share_mobile_28[t_idx, series_fac_idx]
        sstud = precomputer.share_student_28[t_idx, series_fac_idx]
        sreg = precomputer.share_regular_28[t_idx, series_fac_idx]
        snew = precomputer.share_new_donor_28[t_idx, series_fac_idx]
        saph = precomputer.share_apheresis_28[t_idx, series_fac_idx]
        snew17 = precomputer.share_newdonor_17_24_28[t_idx, series_fac_idx]

        same_weekday_mean = {}
        for w in range(7):
            offset = (origin_dt.weekday() - w) % 7
            idxs = [t_idx - offset - 7 * k for k in range(4)]
            valid_idxs = [idx for idx in idxs if idx >= 0]
            if valid_idxs:
                same_weekday_mean[w] = np.mean(
                    precomputer.donations_matrix[valid_idxs], axis=0
                ).astype(np.float32)
            else:
                same_weekday_mean[w] = np.zeros(n_series, dtype=np.float32)

        for h in range(1, horizon + 1):
            target_dt = origin_dt + datetime.timedelta(days=h)
            target_date_str = target_dt.isoformat()
            target_w = target_dt.weekday()
            sw_m = same_weekday_mean[target_w]

            target_idx = precomputer.date_to_idx.get(target_date_str, None)
            if target_idx is not None:
                targets = precomputer.donations_matrix[target_idx]
            else:
                targets = np.full(n_series, np.nan, dtype=np.float32)

            start = block_idx * n_series
            end = start + n_series

            out_origin_date[start:end] = origin
            out_target_date[start:end] = target_date_str
            out_horizon[start:end] = h
            out_target[start:end] = targets

            lag_0[start:end] = l0
            lag_1[start:end] = l1
            lag_2[start:end] = l2
            lag_6[start:end] = l6
            lag_13[start:end] = l13
            lag_20[start:end] = l20
            lag_27[start:end] = l27

            r_mean_7[start:end] = rm7
            r_std_7[start:end] = rs7
            r_mean_28[start:end] = rm28
            r_std_28[start:end] = rs28
            sw_mean_4[start:end] = sw_m

            sh_mobile[start:end] = smob
            sh_student[start:end] = sstud
            sh_regular[start:end] = sreg
            sh_new[start:end] = snew
            sh_apheresis[start:end] = saph
            sh_new17[start:end] = snew17

            # Calendar lookup
            for s_i, st in enumerate(precomputer.state_arr):
                cal_row = cal_dict.get((st, target_date_str))
                idx = start + s_i
                if cal_row is not None:
                    for col in CALENDAR_FEATURE_COLS:
                        cal_arrays[col][idx] = cal_row[col]
                else:
                    for col in CALENDAR_FEATURE_COLS:
                        cal_arrays[col][idx] = -1

            block_idx += 1

    df_dict = {
        "facility": pd.Categorical(out_facility),
        "group": pd.Categorical(out_group),
        "origin_date": out_origin_date,
        "horizon": out_horizon,
        "target_date": out_target_date,
        "target": out_target,
        "lag_0": lag_0,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_6": lag_6,
        "lag_13": lag_13,
        "lag_20": lag_20,
        "lag_27": lag_27,
        "rolling_mean_7": r_mean_7,
        "rolling_std_7": r_std_7,
        "rolling_mean_28": r_mean_28,
        "rolling_std_28": r_std_28,
        "same_weekday_mean_4": sw_mean_4,
        "share_mobile_28": sh_mobile,
        "share_student_28": sh_student,
        "share_regular_28": sh_regular,
        "share_new_donor_28": sh_new,
        "share_apheresis_28": sh_apheresis,
        "share_newdonor_17_24_28": sh_new17,
    }
    for col in CALENDAR_FEATURE_COLS:
        df_dict[col] = cal_arrays[col]

    return pd.DataFrame(df_dict)


def generate_all_feature_datasets(
    processed_dir: Path = DATA_PROCESSED_DIR,
    features_dir: Path = FEATURES_DIR,
) -> list[Path]:
    """Generate and save feature datasets partitioned by year for train, validation, and test."""
    features_dir.mkdir(parents=True, exist_ok=True)

    long_path = processed_dir / "long.parquet"
    if not long_path.exists():
        from donorcast.clean import clean_data

        clean_data()

    long_df = pd.read_parquet(long_path)

    cal_path = CALENDAR_PROCESSED_FILE
    if not cal_path.exists():
        calendar_df = build_calendar_dataframe()
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        calendar_df.to_parquet(cal_path, index=False)
    else:
        calendar_df = pd.read_parquet(cal_path)

    fac_state_df = load_facility_state_mapping(FACILITY_STATE_FILE)

    precomputer = FeaturePrecomputer(long_df, fac_state_df)

    # 1. Collect all train origins and eval origins
    train_origins = get_all_train_origins(TRAIN_START, TRAIN_END, TRAIN_ORIGIN_STEP)
    eval_val_origins = get_all_eval_origins(VAL_START, VAL_END, EVAL_ORIGIN_WEEKDAY)
    eval_test_origins = get_all_eval_origins(TEST_START, TEST_END, EVAL_ORIGIN_WEEKDAY)

    all_origins = sorted(set(train_origins + eval_val_origins + eval_test_origins))

    # Group origins by year of the origin date to write parquet per year
    origins_by_year: dict[int, list[str]] = {}
    for orig in all_origins:
        yr = int(orig[:4])
        origins_by_year.setdefault(yr, []).append(orig)

    saved_files = []
    print(
        f"Generating direct multi-horizon features for {len(all_origins)} origins across {len(origins_by_year)} years..."
    )

    for year in sorted(origins_by_year.keys()):
        yr_origins = origins_by_year[year]
        df_yr = build_features_for_origins(
            origin_dates=yr_origins,
            long_df=long_df,
            calendar_df=calendar_df,
            facility_state_df=fac_state_df,
            horizon=HORIZON,
            precomputer=precomputer,
        )

        out_file = features_dir / f"features_{year}.parquet"
        df_yr.to_parquet(out_file, index=False)
        saved_files.append(out_file)
        print(f"Saved {out_file.name}: {len(df_yr):,} rows ({len(yr_origins)} origins).")

    print(f"All features saved successfully to {features_dir}")
    return saved_files


def load_features(
    years: Sequence[int] | None = None,
    features_dir: Path = FEATURES_DIR,
) -> pd.DataFrame:
    """Load feature parquet files for the specified years (or all if None)."""
    if years is not None:
        files = [features_dir / f"features_{yr}.parquet" for yr in years]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(features_dir.glob("features_*.parquet"))

    if not files:
        raise FileNotFoundError(f"No feature parquet files found in {features_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    generate_all_feature_datasets()
