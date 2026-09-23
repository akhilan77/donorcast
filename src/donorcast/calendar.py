"""Calendar features pipeline for DonorCast.

Generates one row per (state, date) from 2006-01-01 to 2026-12-31 with all external
calendar features:
- is_public_holiday (national + state subdivisions from `holidays`)
- is_ramadan, days_to_hari_raya, days_since_hari_raya, days_to_aidiladha, days_since_aidiladha (via `hijri-converter`)
- days_to_cny, days_since_cny, days_to_deepavali, days_since_deepavali (capped at ±30)
- is_school_holiday, is_weekend (per state), is_mco, is_election_day
- day_of_week, week_of_year, month, year
"""

import datetime
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from hijri_converter import Gregorian

from donorcast.config import (
    CALENDAR_END,
    CALENDAR_PROCESSED_FILE,
    CALENDAR_START,
    FACILITY_STATE_FILE,
    GENERAL_ELECTIONS_FILE,
    MCO_PERIODS_FILE,
    SCHOOL_HOLIDAYS_FILE,
    STATE_WEEKENDS_FILE,
)

# Mapping state names used in DonorCast to `holidays` subdivision codes for Malaysia
STATE_TO_HOLIDAYS_SUBDIV: dict[str, str] = {
    "Johor": "01",
    "Kedah": "02",
    "Kelantan": "03",
    "Melaka": "04",
    "Negeri Sembilan": "05",
    "Pahang": "06",
    "Pulau Pinang": "07",
    "Perak": "08",
    "Perlis": "09",
    "Selangor": "10",
    "Terengganu": "11",
    "Sabah": "12",
    "Sarawak": "13",
    "W.P. Kuala Lumpur": "14",
    "W.P. Labuan": "15",
    "W.P. Putrajaya": "16",
}


def load_facility_state_mapping(
    filepath: Path | str = FACILITY_STATE_FILE,
) -> pd.DataFrame:
    """Load the facility -> state mapping table."""
    return pd.read_csv(filepath)


def _compute_days_to_since(
    target_dates: list[datetime.date],
    all_dates: list[datetime.date],
    cap: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute days_to and days_since for a sorted list of event dates, capped at `cap`.

    Parameters
    ----------
    target_dates : list[datetime.date]
        Dates when the event occurred/occurs.
    all_dates : list[datetime.date]
        All sequential calendar dates.
    cap : int
        Maximum distance cutoff in days.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (days_to, days_since) arrays aligned with all_dates.
        - days_to: min(next_event - date, cap) if upcoming event exists, else cap
        - days_since: min(date - prev_event, cap) if previous event exists, else cap
    """
    sorted_targets = np.array(sorted(target_dates), dtype="datetime64[D]")
    all_arr = np.array(all_dates, dtype="datetime64[D]")

    days_to = np.full(len(all_arr), cap, dtype=int)
    days_since = np.full(len(all_arr), cap, dtype=int)

    # For each date, find nearest preceding and succeeding target
    # searchsorted returns index where all_arr[i] can be inserted to maintain order
    idx_right = np.searchsorted(sorted_targets, all_arr, side="left")
    idx_left = np.searchsorted(sorted_targets, all_arr, side="right") - 1

    # Upcoming event: idx_right < len(sorted_targets)
    has_upcoming = idx_right < len(sorted_targets)
    diff_upcoming = np.zeros(len(all_arr), dtype=int)
    diff_upcoming[has_upcoming] = (
        sorted_targets[idx_right[has_upcoming]] - all_arr[has_upcoming]
    ).astype(int)
    days_to[has_upcoming] = np.clip(diff_upcoming[has_upcoming], 0, cap)

    # Past event: idx_left >= 0
    has_past = idx_left >= 0
    diff_past = np.zeros(len(all_arr), dtype=int)
    diff_past[has_past] = (all_arr[has_past] - sorted_targets[idx_left[has_past]]).astype(int)
    days_since[has_past] = np.clip(diff_past[has_past], 0, cap)

    return days_to, days_since


def _extract_festival_dates(
    start_year: int = 2005,
    end_year: int = 2027,
) -> tuple[list[datetime.date], list[datetime.date]]:
    """Extract CNY (Day 1) and Deepavali dates from `holidays` package across all MY subdivisions."""
    years = list(range(start_year, end_year + 1))
    cny_dates = set()
    deepavali_dates = set()

    for subdiv in holidays.Malaysia.subdivisions:
        sub_holidays = holidays.country_holidays("MY", subdiv=subdiv, years=years)
        for d, name in sub_holidays.items():
            if (
                ("Tahun Baharu Cina" in name or "Chinese New Year" in name)
                and "Kedua" not in name
                and "Second" not in name
                and "pergantian" not in name
                and "Cuti" not in name
            ):
                cny_dates.add(d)
            if (
                ("Deepavali" in name or "Diwali" in name)
                and "Cuti" not in name
                and "pergantian" not in name
            ):
                deepavali_dates.add(d)

    return sorted(cny_dates), sorted(deepavali_dates)


def _extract_islamic_dates(
    dates: list[datetime.date],
) -> tuple[set[datetime.date], list[datetime.date], list[datetime.date]]:
    """Compute Ramadan dates, Hari Raya Aidilfitri dates, and Aidiladha dates using hijri-converter."""
    ramadan_dates: set[datetime.date] = set()
    hari_raya_dates: list[datetime.date] = []
    aidiladha_dates: list[datetime.date] = []

    # Buffer dates slightly before start and after end to compute distance edge cases accurately
    extended_start = dates[0] - datetime.timedelta(days=120)
    extended_end = dates[-1] + datetime.timedelta(days=120)
    num_days = (extended_end - extended_start).days + 1

    for i in range(num_days):
        d = extended_start + datetime.timedelta(days=i)
        h = Gregorian(d.year, d.month, d.day).to_hijri()
        if h.month == 9 and dates[0] <= d <= dates[-1]:
            ramadan_dates.add(d)
        if h.month == 10 and h.day == 1:
            hari_raya_dates.append(d)
        if h.month == 12 and h.day == 10:
            aidiladha_dates.append(d)

    return ramadan_dates, hari_raya_dates, aidiladha_dates


def build_calendar_dataframe(
    start_date: str = CALENDAR_START,
    end_date: str = CALENDAR_END,
    state_weekends_file: Path | str = STATE_WEEKENDS_FILE,
    school_holidays_file: Path | str = SCHOOL_HOLIDAYS_FILE,
    mco_periods_file: Path | str = MCO_PERIODS_FILE,
    general_elections_file: Path | str = GENERAL_ELECTIONS_FILE,
) -> pd.DataFrame:
    """Build the complete calendar DataFrame for all states and dates from start_date to end_date.

    Produces one row per (state, date).
    """
    dt_start = datetime.date.fromisoformat(start_date)
    dt_end = datetime.date.fromisoformat(end_date)
    num_days = (dt_end - dt_start).days + 1
    dates = [dt_start + datetime.timedelta(days=i) for i in range(num_days)]
    date_strs = [d.isoformat() for d in dates]

    # Load external auxiliary files
    df_weekends = pd.read_csv(state_weekends_file)
    df_school = pd.read_csv(school_holidays_file)
    df_mco = pd.read_csv(mco_periods_file)
    df_elections = pd.read_csv(general_elections_file)

    # 1. School Holidays mask
    school_holiday_dates: set[str] = set()
    for _, row in df_school.iterrows():
        s_cur = datetime.date.fromisoformat(row["start_date"])
        s_end = datetime.date.fromisoformat(row["end_date"])
        n_days = (s_end - s_cur).days + 1
        for i in range(n_days):
            cur = s_cur + datetime.timedelta(days=i)
            school_holiday_dates.add(cur.isoformat())

    # 2. MCO mask
    mco_dates: set[str] = set()
    for _, row in df_mco.iterrows():
        m_cur = datetime.date.fromisoformat(row["start_date"])
        m_end = datetime.date.fromisoformat(row["end_date"])
        n_days = (m_end - m_cur).days + 1
        for i in range(n_days):
            cur = m_cur + datetime.timedelta(days=i)
            mco_dates.add(cur.isoformat())

    # 3. Election day mask
    election_dates = set(df_elections["date"].tolist())

    # 4. Islamic dates and festival dates (shared across all states)
    ramadan_dates, hari_raya_dates, aidiladha_dates = _extract_islamic_dates(dates)
    cny_dates, deepavali_dates = _extract_festival_dates(
        start_year=dt_start.year - 1, end_year=dt_end.year + 1
    )

    days_to_raya, days_since_raya = _compute_days_to_since(hari_raya_dates, dates)
    days_to_adha, days_since_adha = _compute_days_to_since(aidiladha_dates, dates)
    days_to_cny, days_since_cny = _compute_days_to_since(cny_dates, dates)
    days_to_deep, days_since_deep = _compute_days_to_since(deepavali_dates, dates)

    # Precalculate date features
    day_of_week = np.array([d.weekday() for d in dates], dtype=int)  # 0=Mon, 6=Sun
    # ISO week of year (1..53)
    week_of_year = np.array([d.isocalendar()[1] for d in dates], dtype=int)
    month = np.array([d.month for d in dates], dtype=int)
    year = np.array([d.year for d in dates], dtype=int)
    is_ramadan = np.array([1 if d in ramadan_dates else 0 for d in dates], dtype=int)
    is_school = np.array([1 if ds in school_holiday_dates else 0 for ds in date_strs], dtype=int)
    is_mco = np.array([1 if ds in mco_dates else 0 for ds in date_strs], dtype=int)
    is_election = np.array([1 if ds in election_dates else 0 for ds in date_strs], dtype=int)

    # Generate state by state
    state_dfs = []
    all_states = sorted(STATE_TO_HOLIDAYS_SUBDIV.keys())
    years_range = list(range(dt_start.year, dt_end.year + 1))

    for state in all_states:
        subdiv = STATE_TO_HOLIDAYS_SUBDIV[state]
        state_holidays = holidays.country_holidays("MY", subdiv=subdiv, years=years_range)

        is_public_holiday = np.array([1 if d in state_holidays else 0 for d in dates], dtype=int)

        # Weekend calculation per state with date transitions
        state_weekend_rules = df_weekends[df_weekends["state"] == state]
        is_weekend = np.zeros(len(dates), dtype=int)

        for _, rule in state_weekend_rules.iterrows():
            r_start = rule["valid_from"]
            r_end = rule["valid_to"]
            w_type = rule["weekend_type"]

            mask = [(r_start <= ds <= r_end) for ds in date_strs]
            mask_arr = np.array(mask, dtype=bool)

            if w_type == "FRI_SAT":
                # Friday (4) and Saturday (5)
                is_weekend[mask_arr] = np.isin(day_of_week[mask_arr], [4, 5]).astype(int)
            elif w_type == "SAT_SUN":
                # Saturday (5) and Sunday (6)
                is_weekend[mask_arr] = np.isin(day_of_week[mask_arr], [5, 6]).astype(int)
            elif w_type == "THU_FRI":
                # Thursday (3) and Friday (4)
                is_weekend[mask_arr] = np.isin(day_of_week[mask_arr], [3, 4]).astype(int)

        df_state = pd.DataFrame(
            {
                "state": state,
                "date": date_strs,
                "year": year,
                "month": month,
                "day_of_week": day_of_week,
                "week_of_year": week_of_year,
                "is_weekend": is_weekend,
                "is_public_holiday": is_public_holiday,
                "is_ramadan": is_ramadan,
                "days_to_hari_raya": days_to_raya,
                "days_since_hari_raya": days_since_raya,
                "days_to_aidiladha": days_to_adha,
                "days_since_aidiladha": days_since_adha,
                "days_to_cny": days_to_cny,
                "days_since_cny": days_since_cny,
                "days_to_deepavali": days_to_deep,
                "days_since_deepavali": days_since_deep,
                "is_school_holiday": is_school,
                "is_mco": is_mco,
                "is_election_day": is_election,
            }
        )
        state_dfs.append(df_state)

    calendar_df = pd.concat(state_dfs, ignore_index=True)
    return calendar_df


def compute_bridge_days(calendar_df: pd.DataFrame) -> pd.Series:
    """Identify bridge workdays (non-holiday workdays adjacent to both a weekend and public holiday).

    A bridge day typically occurs on a Monday when Tuesday is a holiday, or on a Friday
    when Thursday is a holiday, leading to high interstate holiday travel.
    """
    df = calendar_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex) and "date" in df.columns:
        df["dt"] = pd.to_datetime(df["date"])
        df = df.sort_values(["state", "dt"])

    # Non-weekend and non-holiday day
    is_workday = (df["is_weekend"] == 0) & (df["is_public_holiday"] == 0)

    # Check previous and next days per state
    prev_is_off = (df.groupby("state")["is_weekend"].shift(1) == 1) | (
        df.groupby("state")["is_public_holiday"].shift(1) == 1
    )
    next_is_off = (df.groupby("state")["is_weekend"].shift(-1) == 1) | (
        df.groupby("state")["is_public_holiday"].shift(-1) == 1
    )

    is_bridge = is_workday & prev_is_off & next_is_off
    return is_bridge.astype(int)


def compute_min_days_to_festival(calendar_df: pd.DataFrame) -> pd.Series:
    """Compute the minimum distance in days to the nearest upcoming major Malaysian festival.

    Considers Hari Raya Aidilfitri, Chinese New Year, Deepavali, and Hari Raya Aidiladha.
    """
    fest_cols = [
        col
        for col in [
            "days_to_hari_raya",
            "days_to_cny",
            "days_to_deepavali",
            "days_to_aidiladha",
        ]
        if col in calendar_df.columns
    ]
    if not fest_cols:
        return pd.Series(30, index=calendar_df.index)
    return calendar_df[fest_cols].min(axis=1).astype(int)


def save_calendar_parquet(
    output_path: Path | str = CALENDAR_PROCESSED_FILE,
) -> pd.DataFrame:
    """Build and save calendar DataFrame to parquet."""
    df = build_calendar_dataframe()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    df_cal = save_calendar_parquet()
    print(f"Calendar successfully built: {len(df_cal):,} rows, {len(df_cal.columns)} columns.")
