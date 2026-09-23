"""DonorCast Blood Donation Operations Dashboard.

A professional operations decision-support dashboard for hospital and blood bank planners.
Answers two questions immediately:
1. "What donations are expected?"
2. "Is there anything I need to pay attention to?"

Reads strictly from precomputed outputs in outputs/ and reports/. Zero model training,
fitting, feature engineering, or cleaning inside the application.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration and Path Constants
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATA_CUTOFF = "2026-09-22"

REPLAY_LABELS: dict[str, str] = {
    "2026-09-22": "Latest (22 Sep 2026)",
    "2025-03-24": "2025-03-24 (Hari Raya 2025 Replay)",
    "2025-01-27": "2025-01-27 (Chinese New Year Replay)",
    "2025-10-13": "2025-10-13 (Deepavali Replay)",
    "2025-05-12": "2025-05-12 (Regular Term Replay)",
    "2025-08-11": "2025-08-11 (Pre-National Day Replay)",
    "2025-12-15": "2025-12-15 (Year-End Replay)",
    "2026-02-09": "2026-02-09 (Early 2026 Replay)",
    "2026-06-15": "2026-06-15 (Mid 2026 Replay)",
}

BLOOD_GROUPS = ["A", "B", "AB", "O"]


# -----------------------------------------------------------------------------
# Data Loading (Strictly Read-Only from outputs/)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_available_origins() -> list[str]:
    """Scan outputs directory for available forecast parquet files."""
    if not OUTPUTS_DIR.exists():
        return [DATA_CUTOFF]
    forecast_files = sorted(OUTPUTS_DIR.glob("forecasts_*.parquet"), reverse=True)
    origins = []
    for f in forecast_files:
        origin = f.stem.replace("forecasts_", "")
        origins.append(origin)
    if DATA_CUTOFF in origins:
        origins.remove(DATA_CUTOFF)
        origins.insert(0, DATA_CUTOFF)
    return origins if origins else [DATA_CUTOFF]


@st.cache_data(show_spinner=False)
def load_historical_actuals() -> pd.DataFrame:
    """Load historical actual donations strictly from outputs/actuals.parquet."""
    actuals_file = OUTPUTS_DIR / "actuals.parquet"
    if actuals_file.exists():
        df = pd.read_parquet(actuals_file)
    else:
        fallback_file = PROJECT_ROOT / "data" / "processed" / "long.parquet"
        if fallback_file.exists():
            df = pd.read_parquet(fallback_file, columns=["date", "facility", "group", "donations"])
        else:
            df = pd.DataFrame(columns=["date", "facility", "group", "donations"])
    df["date"] = df["date"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_forecasts(origin_date: str) -> pd.DataFrame:
    """Load 14-day forecasts for all facility-group pairs from outputs/."""
    fcst_file = OUTPUTS_DIR / f"forecasts_{origin_date}.parquet"
    if not fcst_file.exists():
        return pd.DataFrame(
            columns=[
                "facility",
                "group",
                "origin_date",
                "horizon",
                "target_date",
                "target",
                "prediction",
                "pred_p50",
                "pred_p10",
                "pred_p90",
            ]
        )
    df = pd.read_parquet(fcst_file)
    df["target_date"] = df["target_date"].astype(str)
    df["origin_date"] = df["origin_date"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_alerts(origin_date: str) -> pd.DataFrame:
    """Load 7-day shortfall alerts for all series from outputs/."""
    alert_file = OUTPUTS_DIR / f"alerts_{origin_date}.parquet"
    if not alert_file.exists():
        return pd.DataFrame(
            columns=[
                "origin_date",
                "target_start_date",
                "target_end_date",
                "facility",
                "group",
                "severity",
                "is_shortfall",
                "forecast_7d",
                "forecast_p10_7d",
                "forecast_p90_7d",
                "typical_7d",
                "deficit_pct",
                "shortfall_ratio",
                "reason_1",
                "reason_2",
                "reason_3",
                "reasons_str",
                "actual_7d",
                "actual_shortfall",
                "actual_severity",
            ]
        )
    df = pd.read_parquet(alert_file)
    df["origin_date"] = df["origin_date"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def get_facility_list() -> list[str]:
    """Return alphabetical list of 22 collection facilities."""
    df_fcst = load_forecasts(DATA_CUTOFF)
    if len(df_fcst) > 0:
        return sorted(df_fcst["facility"].unique().tolist())
    df_act = load_historical_actuals()
    if len(df_act) > 0:
        return sorted(df_act["facility"].unique().tolist())
    return ["Pusat Darah Negara"]


# -----------------------------------------------------------------------------
# Professional Dark Operations UI Styles
# -----------------------------------------------------------------------------
def apply_custom_styles() -> None:
    """Inject polished dark theme CSS tokens and component styles."""
    st.markdown(
        """
        <style>
        /* Base typography & color tokens */
        html, body, [data-testid="stAppViewContainer"], .main {
            background-color: #0E1117 !important;
            color: #E5E7EB !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }

        /* Container width & top alignment */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px !important;
        }

        /* Headings hierarchy */
        h1, h2, h3 {
            font-weight: 700 !important;
            color: #F9FAFB !important;
            letter-spacing: -0.02em !important;
        }
        h1 {
            font-size: 28px !important;
            margin-bottom: 2px !important;
        }
        .page-subtitle {
            font-size: 14px !important;
            font-weight: 400 !important;
            color: #9CA3AF !important;
            margin-bottom: 18px !important;
        }
        .section-title {
            font-size: 18px !important;
            font-weight: 700 !important;
            color: #F3F4F6 !important;
            margin-top: 24px !important;
            margin-bottom: 10px !important;
        }

        /* Persistent Top Info Bar */
        .top-meta-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            background-color: #1A1F2E;
            border: 1px solid #2D3748;
            border-radius: 8px;
            padding: 10px 18px;
            margin-bottom: 20px;
            font-size: 13px;
            color: #9CA3AF;
        }
        .top-meta-item strong {
            color: #F3F4F6;
            font-weight: 600;
        }
        .top-meta-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background-color: #242E44;
            color: #3B82F6;
            padding: 3px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
            border: 1px solid #3B82F640;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #121722 !important;
            border-right: 1px solid #1F2937 !important;
        }
        [data-testid="stSidebar"] * {
            color: #E5E7EB !important;
        }
        [data-testid="stSidebar"] hr {
            margin: 1.25rem 0 !important;
            border-color: #1F2937 !important;
        }

        /* Sidebar Navigation Controls & High Contrast Active State */
        .nav-section-title {
            font-size: 11px !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            color: #9CA3AF !important;
            margin-bottom: 8px !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            display: flex !important;
            align-items: center !important;
            padding: 11px 16px !important;
            margin-bottom: 6px !important;
            border-radius: 6px !important;
            border: 1px solid transparent !important;
            border-left: 4px solid transparent !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            color: #D1D5DB !important;
            background-color: transparent !important;
            transition: all 0.15s ease !important;
            cursor: pointer !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background-color: #1A2234 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background-color: #1E293B !important;
            color: #FFFFFF !important;
            border: 1px solid #374151 !important;
            border-left: 4px solid #3B82F6 !important;
            font-weight: 700 !important;
        }

        /* Unified Dark Card System */
        .op-card {
            background-color: #1A1F2E;
            border: 1px solid #2D3748;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 12px;
            transition: border-color 0.15s ease;
        }
        .op-card-label {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #9CA3AF;
            margin-bottom: 6px;
        }
        .op-card-value {
            font-size: 26px;
            font-weight: 700;
            color: #F9FAFB;
            line-height: 1.15;
        }
        .op-card-sub {
            font-size: 12px;
            font-weight: 400;
            color: #9CA3AF;
            margin-top: 4px;
        }

        /* Status Accents (Left Border 4px) */
        .card-rose {
            border-left: 4px solid #EF4444 !important;
        }
        .card-indigo {
            border-left: 4px solid #6366F1 !important;
        }
        .card-emerald {
            border-left: 4px solid #10B981 !important;
        }
        .card-blue {
            border-left: 4px solid #3B82F6 !important;
        }
        .card-neutral {
            border-left: 4px solid #4B5563 !important;
        }

        /* Status Badges */
        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .pill-high {
            background-color: #3B1822;
            color: #F87171;
            border: 1px solid #EF444460;
        }
        .pill-med {
            background-color: #222547;
            color: #818CF8;
            border: 1px solid #6366F160;
        }
        .pill-norm {
            background-color: #132D27;
            color: #34D399;
            border: 1px solid #10B98160;
        }

        /* Buttons */
        div.stButton > button:first-child, div.stDownloadButton > button:first-child {
            background-color: #3B82F6 !important;
            color: #FFFFFF !important;
            border-radius: 6px !important;
            border: 1px solid #2563EB !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            padding: 8px 18px !important;
            transition: all 0.15s ease !important;
        }
        div.stButton > button:first-child:hover, div.stDownloadButton > button:first-child:hover {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
        }

        /* Select boxes & Inputs */
        div[data-baseweb="select"] > div {
            background-color: #1A1F2E !important;
            border-color: #2D3748 !important;
            border-radius: 6px !important;
            color: #F3F4F6 !important;
        }

        /* Dataframes */
        [data-testid="stDataFrame"] {
            border: 1px solid #2D3748 !important;
            border-radius: 8px !important;
            background-color: #1A1F2E !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #1A1F2E !important;
            border: 1px solid #2D3748 !important;
            border-radius: 6px !important;
            color: #9CA3AF !important;
            font-size: 13px !important;
        }

        /* Global Footer */
        .sidebar-footer {
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid #1F2937;
            color: #6B7280;
            font-size: 11px;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_bar(selected_origin: str) -> None:
    """Render a persistent horizontal metadata bar across all pages."""
    origin_formatted = (
        pd.to_datetime(selected_origin).strftime("%d %b %Y")
        if selected_origin != DATA_CUTOFF
        else "22 Sep 2026 (Latest)"
    )
    is_replay = selected_origin != DATA_CUTOFF
    badge_html = (
        '<span class="top-meta-badge" style="color:#34D399; border-color:#10B98140; background:#132D27;">● Operational</span>'
        if not is_replay
        else '<span class="top-meta-badge" style="color:#FBBF24; border-color:#F59E0B40; background:#362612;">↺ Replay Mode</span>'
    )

    st.markdown(
        f"""
        <div class="top-meta-bar">
            <div class="top-meta-item">
                <strong>Data Cutoff:</strong> 22 Sep 2026 &nbsp;·&nbsp;
                <strong>Source:</strong> MoH Malaysia / National Blood Centre
            </div>
            <div class="top-meta-item">
                <strong>Forecast Origin:</strong> {origin_formatted} &nbsp; {badge_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Page 1: Forecast Page
# -----------------------------------------------------------------------------
def render_forecast_page(selected_origin: str) -> None:
    """Render the operational 14-day blood donation forecast page."""
    render_top_bar(selected_origin)

    st.markdown("<h1>Forecast</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        "Expected blood donations for the selected facility and blood group."
        "</div>",
        unsafe_allow_html=True,
    )

    # 1. Clean Top Filter Bar
    facility_list = get_facility_list()
    if "sb_facility" not in st.session_state:
        st.session_state["sb_facility"] = (
            "Pusat Darah Negara" if "Pusat Darah Negara" in facility_list else facility_list[0]
        )
    if "sb_group" not in st.session_state:
        st.session_state["sb_group"] = "O"

    filt_col1, filt_col2 = st.columns([3, 2])
    with filt_col1:
        selected_facility = st.selectbox(
            "Collection Facility",
            facility_list,
            key="sb_facility",
            help="Select one of the 22 MoH blood collection sites in Malaysia.",
        )
        st.session_state["selected_facility"] = selected_facility

    with filt_col2:
        selected_group = st.selectbox(
            "Blood Group",
            BLOOD_GROUPS,
            key="sb_group",
            help="Select ABO blood group.",
        )
        st.session_state["selected_group"] = selected_group

    # Load precomputed datasets
    df_fcst_all = load_forecasts(selected_origin)
    df_act_all = load_historical_actuals()
    df_alerts_all = load_alerts(selected_origin)

    # Filter forecast series
    fcst_sub = (
        df_fcst_all[
            (df_fcst_all["facility"] == selected_facility)
            & (df_fcst_all["group"] == selected_group)
        ]
        .sort_values("horizon")
        .copy()
    )

    if len(fcst_sub) == 0:
        st.warning(
            f"No forecast data available for {selected_facility} (Group {selected_group}) at origin {selected_origin}."
        )
        return

    # Filter 8 weeks (56 days) of actual donations prior to origin
    origin_dt = datetime.date.fromisoformat(selected_origin)
    history_start_dt = origin_dt - datetime.timedelta(days=56)
    history_start_str = history_start_dt.isoformat()

    act_sub = (
        df_act_all[
            (df_act_all["facility"] == selected_facility)
            & (df_act_all["group"] == selected_group)
            & (df_act_all["date"] >= history_start_str)
            & (df_act_all["date"] <= selected_origin)
        ]
        .sort_values("date")
        .copy()
    )

    # Calculate Operational Summary Metrics
    # TODAY (Horizon 1 prediction)
    h1_val = fcst_sub[fcst_sub["horizon"] == 1]["pred_p50"].iloc[0] if len(fcst_sub) > 0 else 0.0

    # NEXT 7 DAYS (Sum of horizons 1 to 7)
    next_7d_val = fcst_sub[fcst_sub["horizon"].isin(range(1, 8))]["pred_p50"].sum()

    # STATUS: Check if this facility x group is in alerts for this origin
    alert_match = df_alerts_all[
        (df_alerts_all["facility"] == selected_facility)
        & (df_alerts_all["group"] == selected_group)
    ]
    if len(alert_match) > 0:
        sev = str(alert_match["severity"].iloc[0])
        if sev == "HIGH":
            status_text = "High Shortfall"
            status_color = "#F87171"
            status_card_class = "card-rose"
            status_desc = "Immediate planning attention required"
        else:
            status_text = "Moderate Shortfall"
            status_color = "#818CF8"
            status_card_class = "card-indigo"
            status_desc = "Monitor supply closely"
    else:
        status_text = "Stable Supply"
        status_color = "#34D399"
        status_card_class = "card-emerald"
        status_desc = "Within expected seasonal volume"

    # 2. Operational Summary Cards (Unified Dark Styling)
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(
            f"""
            <div class="op-card card-blue">
                <div class="op-card-label">TODAY</div>
                <div class="op-card-value">{h1_val:.0f}</div>
                <div class="op-card-sub">Expected donations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="op-card card-blue">
                <div class="op-card-label">NEXT 7 DAYS</div>
                <div class="op-card-value">{next_7d_val:.0f}</div>
                <div class="op-card-sub">Expected total</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown(
            """
            <div class="op-card card-neutral">
                <div class="op-card-label">HORIZON</div>
                <div class="op-card-value" style="font-size: 22px; padding-top: 4px;">14 Days</div>
                <div class="op-card-sub">Daily forward outlook</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi4:
        st.markdown(
            f"""
            <div class="op-card {status_card_class}">
                <div class="op-card-label">SUPPLY STATUS</div>
                <div class="op-card-value" style="color: {status_color};">{status_text}</div>
                <div class="op-card-sub">{status_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Main Forecast Chart with Daily / Weekly Toggle
    chart_header_col, toggle_col = st.columns([5, 3])
    with chart_header_col:
        st.markdown('<div class="section-title">Expected Donations</div>', unsafe_allow_html=True)
    with toggle_col:
        time_view = st.radio(
            "View Aggregation",
            ["Daily View", "Weekly View"],
            horizontal=True,
            key="time_view_toggle",
            label_visibility="collapsed",
        )

    # Prepare datasets for Altair
    # 3a. Historical Actuals
    act_df = pd.DataFrame(
        {
            "date": pd.to_datetime(act_sub["date"]),
            "donations": act_sub["donations"].astype(float),
        }
    )

    # 3b. Forecast Horizon Data
    fcst_dates = pd.to_datetime(fcst_sub["target_date"])
    pred_df = pd.DataFrame(
        {
            "date": fcst_dates,
            "pred_p50": fcst_sub["pred_p50"].astype(float),
            "pred_p10": fcst_sub["pred_p10"].astype(float),
            "pred_p90": fcst_sub["pred_p90"].astype(float),
        }
    )

    has_subsequent_actuals = not fcst_sub["target"].isna().all()
    subsequent_df = None
    if has_subsequent_actuals:
        subsequent_df = pd.DataFrame(
            {
                "date": fcst_dates,
                "donations": fcst_sub["target"].astype(float),
            }
        )

    # If Weekly View is selected, aggregate data by 7-day intervals
    if time_view == "Weekly View":
        # Group historical data by 7-day periods ending at forecast origin
        act_df["days_ago"] = (pd.to_datetime(selected_origin) - act_df["date"]).dt.days
        act_df["week_bin"] = act_df["days_ago"] // 7
        act_weekly = (
            act_df.groupby("week_bin")
            .agg(
                date=("date", "max"),
                donations=("donations", "sum"),
                days_count=("donations", "count"),
            )
            .reset_index()
        )
        act_weekly = act_weekly[act_weekly["days_count"] == 7].sort_values("date")

        # Forecast weekly sums: Week 1 (H1-H7) and Week 2 (H8-H14)
        w1_p50 = pred_df.iloc[0:7]["pred_p50"].sum()
        w1_p10 = pred_df.iloc[0:7]["pred_p10"].sum()
        w1_p90 = pred_df.iloc[0:7]["pred_p90"].sum()
        w1_date = pred_df.iloc[3]["date"]

        w2_p50 = pred_df.iloc[7:14]["pred_p50"].sum()
        w2_p10 = pred_df.iloc[7:14]["pred_p10"].sum()
        w2_p90 = pred_df.iloc[7:14]["pred_p90"].sum()
        w2_date = pred_df.iloc[10]["date"]

        pred_chart_df = pd.DataFrame(
            {
                "date": [w1_date, w2_date],
                "donations": [w1_p50, w2_p50],
                "p10": [round(w1_p10, 1), round(w2_p10, 1)],
                "p50": [round(w1_p50, 1), round(w2_p50, 1)],
                "p90": [round(w1_p90, 1), round(w2_p90, 1)],
            }
        )
        band_chart_df = pd.DataFrame(
            {
                "date": [w1_date, w2_date],
                "pred_p10": [w1_p10, w2_p10],
                "pred_p90": [w1_p90, w2_p90],
            }
        )

        act_chart_df = act_weekly[["date", "donations"]]
        y_axis_title = "Weekly Total Donations"
    else:
        act_chart_df = act_df[["date", "donations"]]
        pred_chart_df = pd.DataFrame(
            {
                "date": fcst_dates,
                "donations": pred_df["pred_p50"],
                "p10": pred_df["pred_p10"].round(1),
                "p50": pred_df["pred_p50"].round(1),
                "p90": pred_df["pred_p90"].round(1),
            }
        )
        band_chart_df = pd.DataFrame(
            {
                "date": fcst_dates,
                "pred_p10": pred_df["pred_p10"],
                "pred_p90": pred_df["pred_p90"],
            }
        )
        y_axis_title = "Daily Donations"

    # Softened gridline styling for dark UI
    axis_config = alt.Axis(
        format="%d %b",
        labelAngle=0,
        labelFontSize=11,
        titleFontSize=12,
        labelColor="#9CA3AF",
        titleColor="#E5E7EB",
        gridColor="#1F2937",
        gridDash=[3, 3],
        gridOpacity=0.7,
        domainColor="#374151",
    )

    # Layer 1: Expected range (soft blue/cyan translucent band)
    band_chart = (
        alt.Chart(band_chart_df)
        .mark_area(
            opacity=0.18,
            color="#3B82F6",
        )
        .encode(
            x=alt.X("date:T", title="Date", axis=axis_config),
            y=alt.Y(
                "pred_p10:Q",
                title=y_axis_title,
                axis=alt.Axis(
                    labelFontSize=11,
                    titleFontSize=12,
                    labelColor="#9CA3AF",
                    titleColor="#E5E7EB",
                    gridColor="#1F2937",
                    gridDash=[3, 3],
                    gridOpacity=0.7,
                ),
            ),
            y2="pred_p90:Q",
        )
    )

    # Layer 2: Actual donations (solid slate line with distinct points)
    line_actual = (
        alt.Chart(act_chart_df)
        .mark_line(color="#94A3B8", strokeWidth=2.5)
        .encode(
            x=alt.X("date:T"),
            y=alt.Y("donations:Q"),
        )
    )
    points_actual = (
        alt.Chart(act_chart_df)
        .mark_circle(color="#94A3B8", size=32, opacity=0.8)
        .encode(
            x=alt.X("date:T"),
            y=alt.Y("donations:Q"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b %Y (%a)"),
                alt.Tooltip("donations:Q", title="Actual donations", format=".0f"),
            ],
        )
    )

    # Layer 3: Expected donations line & points (bold blue)
    line_forecast = (
        alt.Chart(pred_chart_df)
        .mark_line(color="#3B82F6", strokeWidth=3.2)
        .encode(
            x=alt.X("date:T"),
            y=alt.Y("donations:Q"),
        )
    )
    points_forecast = (
        alt.Chart(pred_chart_df)
        .mark_circle(color="#3B82F6", size=55)
        .encode(
            x=alt.X("date:T"),
            y=alt.Y("donations:Q"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b %Y (%a)"),
                alt.Tooltip("p50:Q", title="Expected", format=".1f"),
                alt.Tooltip("p10:Q", title="Low estimate", format=".1f"),
                alt.Tooltip("p90:Q", title="High estimate", format=".1f"),
            ],
        )
    )

    # Layer 4: Vertical Forecast Boundary Line
    origin_line_df = pd.DataFrame({"date": [pd.to_datetime(selected_origin)]})
    boundary_rule = (
        alt.Chart(origin_line_df)
        .mark_rule(
            color="#4B5563",
            strokeDash=[4, 4],
            strokeWidth=1.5,
        )
        .encode(x=alt.X("date:T"))
    )

    chart_layers = [
        band_chart,
        line_actual,
        points_actual,
        boundary_rule,
        line_forecast,
        points_forecast,
    ]

    # Layer 5 (Replay): Subsequent Actual Donations Overlay
    if has_subsequent_actuals and subsequent_df is not None and time_view == "Daily View":
        line_subsequent = (
            alt.Chart(subsequent_df)
            .mark_line(color="#10B981", strokeWidth=2.2, strokeDash=[5, 3])
            .encode(
                x=alt.X("date:T"),
                y=alt.Y("donations:Q"),
            )
        )
        points_subsequent = (
            alt.Chart(subsequent_df)
            .mark_square(color="#10B981", size=36)
            .encode(
                x=alt.X("date:T"),
                y=alt.Y("donations:Q"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%d %b %Y (%a)"),
                    alt.Tooltip("donations:Q", title="Realized Actual", format=".0f"),
                ],
            )
        )
        chart_layers.extend([line_subsequent, points_subsequent])

    # Assemble complete chart
    combined_chart = (
        alt.layer(*chart_layers)
        .properties(height=380)
        .configure_axis(
            labelFont="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            titleFont="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        )
        .configure_view(strokeOpacity=0)
    )

    st.altair_chart(combined_chart, use_container_width=True)

    # Clean Operational Legend Below Chart
    leg1, leg2, leg3, leg4 = st.columns(4)
    with leg1:
        st.markdown(
            '<div style="font-size: 13px; color: #D1D5DB;"><span style="color: #94A3B8; font-weight: bold;">― ●</span> <strong>Actual donations</strong></div>',
            unsafe_allow_html=True,
        )
    with leg2:
        st.markdown(
            '<div style="font-size: 13px; color: #3B82F6;"><span style="font-weight: bold;">― ●</span> <strong>Expected</strong></div>',
            unsafe_allow_html=True,
        )
    with leg3:
        st.markdown(
            '<div style="font-size: 13px; color: #3B82F6;"><span style="display:inline-block; width:13px; height:10px; background:#3B82F6; opacity:0.3; border-radius:2px; margin-right:4px;"></span><strong>Low – High estimate</strong></div>',
            unsafe_allow_html=True,
        )
    with leg4:
        if has_subsequent_actuals and time_view == "Daily View":
            st.markdown(
                '<div style="font-size: 13px; color: #10B981;"><span style="font-weight: bold;">┄ ■</span> <strong>Realized Actual</strong> (Replay)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size: 13px; color: #6B7280;">┊ <strong>Forecast starts</strong></div>',
                unsafe_allow_html=True,
            )

    # Technical details in collapsible expander
    with st.expander("ℹ About this forecast"):
        st.markdown(
            """
            - **Expected**: Forecast midpoint estimate (p50) generated from global regression models trained on 20+ years of collection patterns.
            - **Low / High estimate**: 80% statistical coverage band (p10 to p90 quantile estimates), capturing day-of-week seasonality, festive holidays, and recent donor trends.
            - **Historical comparison**: In replay mode, realized donations are plotted against predictions to inspect model performance during historical events.
            """
        )

    # 4. 14-Day Outlook Table
    st.markdown('<div class="section-title">14-Day Outlook</div>', unsafe_allow_html=True)

    table_data = []
    for _, row in fcst_sub.iterrows():
        dt = pd.to_datetime(row["target_date"])
        entry = {
            "Date": dt.strftime("%Y-%m-%d (%a)"),
            "Expected": round(float(row["pred_p50"]), 1),
            "Low estimate": round(float(row["pred_p10"]), 1),
            "High estimate": round(float(row["pred_p90"]), 1),
        }
        if has_subsequent_actuals:
            act_val = row["target"]
            entry["Actual"] = round(float(act_val), 0) if pd.notna(act_val) else None
        table_data.append(entry)

    df_table = pd.DataFrame(table_data)

    column_cfg = {
        "Date": st.column_config.TextColumn("Date", width="medium"),
        "Expected": st.column_config.NumberColumn(
            "Expected", help="Forecast midpoint", format="%.1f", width="small"
        ),
        "Low estimate": st.column_config.NumberColumn(
            "Low estimate", help="p10 lower bound", format="%.1f", width="small"
        ),
        "High estimate": st.column_config.NumberColumn(
            "High estimate", help="p90 upper bound", format="%.1f", width="small"
        ),
    }
    if has_subsequent_actuals:
        column_cfg["Actual"] = st.column_config.NumberColumn(
            "Actual", help="Realized actual donations", format="%d", width="small"
        )

    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        column_config=column_cfg,
    )


# -----------------------------------------------------------------------------
# Page 2: Shortfall Alerts Page
# -----------------------------------------------------------------------------
def render_alerts_page(selected_origin: str) -> None:
    """Render the operational shortfall alert detection and drill-down page."""
    render_top_bar(selected_origin)

    st.markdown("<h1>Shortfall Alerts</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        "Facilities and blood groups where expected donations may fall below normal levels."
        "</div>",
        unsafe_allow_html=True,
    )

    # 1. Load alerts for current origin
    df_alerts = load_alerts(selected_origin)

    if len(df_alerts) == 0:
        st.info(f"No shortfall alerts flagged for forecast origin {selected_origin}.")
        return

    # 2. Top Summary Cards (Unified Dark Styling)
    total_alerts = len(df_alerts)
    high_alerts = int((df_alerts["severity"] == "HIGH").sum())
    med_alerts = int((df_alerts["severity"] == "MEDIUM").sum())

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown(
            f"""
            <div class="op-card card-neutral">
                <div class="op-card-label">TOTAL ALERTS</div>
                <div class="op-card-value">{total_alerts}</div>
                <div class="op-card-sub">Flagged collection series</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="op-card card-rose">
                <div class="op-card-label">HIGH PRIORITY</div>
                <div class="op-card-value" style="color: #F87171;">{high_alerts}</div>
                <div class="op-card-sub">Expected &lt; 70% of typical volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown(
            f"""
            <div class="op-card card-indigo">
                <div class="op-card-label">MEDIUM PRIORITY</div>
                <div class="op-card-value" style="color: #818CF8;">{med_alerts}</div>
                <div class="op-card-sub">Expected 70%–80% of typical volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Clean Filter Controls & Export Action
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    filt_col1, filt_col2, action_col = st.columns([2, 2, 4])

    with filt_col1:
        group_filter = st.selectbox(
            "Blood Group",
            ["All", "A", "B", "AB", "O"],
            index=0,
            key="alert_filter_group",
        )

    with filt_col2:
        priority_filter = st.selectbox(
            "Priority",
            ["All", "HIGH", "MEDIUM"],
            index=0,
            key="alert_filter_severity",
        )

    # Apply both filters together
    filtered_df = df_alerts.copy()
    if group_filter != "All":
        filtered_df = filtered_df[filtered_df["group"] == group_filter]
    if priority_filter != "All":
        filtered_df = filtered_df[filtered_df["severity"] == priority_filter]

    # Sort alerts by severity (HIGH then MEDIUM, then deficit)
    severity_order = {"HIGH": 0, "MEDIUM": 1, "NONE": 2}
    filtered_df["_sev_sort"] = filtered_df["severity"].map(severity_order)
    filtered_df = filtered_df.sort_values(
        by=["_sev_sort", "deficit_pct", "typical_7d"], ascending=[True, True, False]
    ).drop(columns=["_sev_sort"])

    with action_col:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if len(filtered_df) > 0:
            csv_export = filtered_df[
                [
                    "facility",
                    "group",
                    "severity",
                    "forecast_7d",
                    "typical_7d",
                    "deficit_pct",
                    "reason_1",
                    "reason_2",
                    "reason_3",
                ]
            ].to_csv(index=False)
            st.download_button(
                label="📥 Export Alerts (CSV)",
                data=csv_export,
                file_name=f"donorcast_shortfall_alerts_{selected_origin}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # 4. Clean Operational Alert Table
    st.markdown(
        f'<div class="section-title">Flagged Collection Sites ({len(filtered_df)})</div>',
        unsafe_allow_html=True,
    )

    if len(filtered_df) == 0:
        st.info("No shortfall alerts match the current filter selection.")
    else:
        display_df = pd.DataFrame(
            {
                "Facility": filtered_df["facility"],
                "Group": filtered_df["group"],
                "Expected (7d)": filtered_df["forecast_7d"].map(lambda x: f"{x:.0f}"),
                "Typical (7d)": filtered_df["typical_7d"].map(lambda x: f"{x:.0f}"),
                "Shortfall": filtered_df["deficit_pct"].map(
                    lambda x: f"{abs(x):.0%} below typical"
                ),
                "Priority": filtered_df["severity"],
            }
        )

        column_config = {
            "Facility": st.column_config.TextColumn("Facility", width="large"),
            "Group": st.column_config.TextColumn("Group", width="small"),
            "Expected (7d)": st.column_config.TextColumn("Expected (7d)", width="small"),
            "Typical (7d)": st.column_config.TextColumn("Typical (7d)", width="small"),
            "Shortfall": st.column_config.TextColumn("Shortfall", width="medium"),
            "Priority": st.column_config.TextColumn("Priority", width="small"),
        }

        event = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config=column_config,
            key="alert_table_selection",
        )

        st.markdown(
            """
            <div style="font-size: 12px; color: #9CA3AF; margin-top: 4px; margin-bottom: 20px;">
                Select any row above to inspect associated statistical factors and drill down into the forecast.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 5. Alert Details & Drill-Down Card
        st.markdown('<div class="section-title">Alert Details</div>', unsafe_allow_html=True)

        selected_row_idx = None
        if event and event.selection and event.selection.rows:
            selected_row_idx = event.selection.rows[0]

        alert_options = [
            f"{r['facility']} (Group {r['group']}) — {r['severity']} [{abs(r['deficit_pct']):.0%} below typical]"
            for _, r in filtered_df.iterrows()
        ]

        default_select_idx = selected_row_idx if selected_row_idx is not None else 0
        default_select_idx = min(default_select_idx, len(alert_options) - 1)

        picked_alert_str = st.selectbox(
            "Selected Alert",
            options=alert_options,
            index=default_select_idx,
            key="inspect_alert_select",
            label_visibility="collapsed",
        )

        picked_idx = alert_options.index(picked_alert_str)
        target_row = filtered_df.iloc[picked_idx]

        target_fac = target_row["facility"]
        target_grp = target_row["group"]
        target_sev = target_row["severity"]
        target_def = abs(target_row["deficit_pct"])
        target_fcst = target_row["forecast_7d"]
        target_typ = target_row["typical_7d"]
        r1 = target_row["reason_1"]
        r2 = target_row["reason_2"]
        r3 = target_row["reason_3"]

        badge_class = "pill-high" if target_sev == "HIGH" else "pill-med"
        border_class = "card-rose" if target_sev == "HIGH" else "card-indigo"
        accent_color = "#F87171" if target_sev == "HIGH" else "#818CF8"

        st.markdown(
            f"""
            <div class="op-card {border_class}" style="margin-top: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-size: 18px; font-weight: 700; color: #F9FAFB;">
                        {target_fac} &nbsp;·&nbsp; Blood Group <span style="color: {accent_color};">{target_grp}</span>
                    </div>
                    <div>
                        <span class="status-pill {badge_class}">{target_sev} PRIORITY</span>
                    </div>
                </div>
                <div style="font-size: 14px; color: #D1D5DB; line-height: 1.6; margin-bottom: 14px;">
                    Expected over next 7 days: <strong>{target_fcst:.0f} donations</strong><br/>
                    Typical over next 7 days: <strong>{target_typ:.0f} donations</strong><br/>
                    Shortfall: <strong style="color: {accent_color};">{target_def:.0%} below typical</strong>
                </div>
                <div style="background-color: #121722; border-radius: 6px; padding: 14px 18px; border: 1px solid #2D3748; font-size: 13px; color: #E5E7EB;">
                    <div style="font-weight: 700; color: #F3F4F6; margin-bottom: 8px;">Key Associated Factors</div>
                    <div style="margin-bottom: 4px; color: #D1D5DB;">• {r1}</div>
                    <div style="margin-bottom: 4px; color: #D1D5DB;">• {r2}</div>
                    <div style="margin-bottom: 4px; color: #D1D5DB;">• {r3}</div>
                    <div style="margin-top: 10px; font-size: 11px; color: #9CA3AF; font-style: italic;">
                        These factors represent primary statistical contributors to the model prediction.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        btn_col, _ = st.columns([3, 5])
        with btn_col:
            if st.button("View Forecast →", key="btn_drilldown"):
                st.session_state["target_facility"] = target_fac
                st.session_state["target_group"] = target_grp
                st.session_state["nav_target"] = "Forecast"
                st.rerun()


# -----------------------------------------------------------------------------
# Main Application Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    """Main dashboard layout and navigation coordinator."""
    st.set_page_config(
        page_title="DonorCast — Blood Donation Forecasting",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()

    # Handle cross-page navigation state before widgets instantiate
    if "nav_target" in st.session_state:
        target_page = st.session_state.pop("nav_target")
        st.session_state["nav_radio"] = target_page
        st.session_state["nav_page"] = target_page
    if "target_facility" in st.session_state:
        fac = st.session_state.pop("target_facility")
        st.session_state["sb_facility"] = fac
        st.session_state["selected_facility"] = fac
    if "target_group" in st.session_state:
        grp = st.session_state.pop("target_group")
        st.session_state["sb_group"] = grp
        st.session_state["selected_group"] = grp

    # Sidebar: Clean Wordmark and High-Contrast Navigation
    with st.sidebar:
        # SVG Brand Logo and Wordmark
        st.markdown(
            """
            <div style="margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z" fill="#3B82F6"/>
                        <path d="M12 11v6M9 14h6" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                    <span style="font-size: 20px; font-weight: 800; color: #F9FAFB; letter-spacing: -0.02em;">DONOR<span style="color: #3B82F6;">CAST</span></span>
                </div>
                <div style="color: #9CA3AF; font-size: 13px; font-weight: 400; margin-top: 4px;">
                    Blood Donation Forecasting
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nav-section-title">NAVIGATION</div>', unsafe_allow_html=True)
        nav_options = ["Forecast", "Shortfall Alerts"]
        if "nav_radio" not in st.session_state:
            st.session_state["nav_radio"] = "Forecast"

        selected_page = st.radio(
            "Navigation",
            nav_options,
            key="nav_radio",
            label_visibility="collapsed",
        )
        st.session_state["nav_page"] = selected_page

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown('<div class="nav-section-title">FORECAST ORIGIN</div>', unsafe_allow_html=True)
        available_origins = get_available_origins()
        origin_labels = [REPLAY_LABELS.get(orig, f"{orig} (Replay)") for orig in available_origins]

        if "sb_origin" not in st.session_state:
            st.session_state["sb_origin"] = origin_labels[0]

        chosen_label = st.selectbox(
            "Forecast Origin Date",
            options=origin_labels,
            key="sb_origin",
            label_visibility="collapsed",
            help="Choose latest cutoff (22 Sep 2026) or any historical test-period replay origin.",
        )

        chosen_origin = available_origins[origin_labels.index(chosen_label)]
        st.session_state["selected_origin"] = chosen_origin

        # Clean single-instance operational disclaimer in sidebar footer
        st.markdown(
            """
            <div class="sidebar-footer">
                Decision support system for operational blood drive planning.<br/>
                Not a clinical tool.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main page content routing
    active_origin = st.session_state.get("selected_origin", DATA_CUTOFF)
    if st.session_state["nav_page"] == "Forecast":
        render_forecast_page(selected_origin=active_origin)
    elif st.session_state["nav_page"] == "Shortfall Alerts":
        render_alerts_page(selected_origin=active_origin)


if __name__ == "__main__":
    main()
