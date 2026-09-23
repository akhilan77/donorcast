"""DonorCast Streamlit Dashboard (Task 4.5).

Interactive decision support dashboard for daily blood donation forecasting and shortfall detection.
Displays Forecast and Shortfall Alerts pages with a modern SaaS aesthetic inspired by the n8n design language.
Strictly reads from precomputed artifacts in outputs/ and reports/. No model training, cleaning, or feature
generation inside the application.
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
    "2026-09-22": "Latest Cutoff (2026-09-22)",
    "2025-03-24": "2025-03-24 (Hari Raya 2025 Replay)",
    "2025-01-27": "2025-01-27 (Chinese New Year 2025 Replay)",
    "2025-10-13": "2025-10-13 (Deepavali 2025 Replay)",
    "2025-05-12": "2025-05-12 (May 2025 Regular Term)",
    "2025-08-11": "2025-08-11 (August 2025 Pre-National Day)",
    "2025-12-15": "2025-12-15 (December 2025 Year-End)",
    "2026-02-09": "2026-02-09 (February 2026)",
    "2026-06-15": "2026-06-15 (June 2026)",
}

BLOOD_GROUPS = ["A", "B", "AB", "O"]

# -----------------------------------------------------------------------------
# Cached Data Loading (Strictly Read-Only from outputs/ and reports/)
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
    # Ensure latest cutoff is first
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
        # Fallback to processed long if outputs/actuals.parquet has not yet been exported
        fallback_file = PROJECT_ROOT / "data" / "processed" / "long.parquet"
        if fallback_file.exists():
            df = pd.read_parquet(
                fallback_file, columns=["date", "facility", "group", "donations"]
            )
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
# Global Page Styling (SaaS Design Tokens Inspired by n8n Visual Language)
# -----------------------------------------------------------------------------
def apply_custom_styles() -> None:
    """Inject CSS tokens matching the project design specification."""
    st.markdown(
        """
        <style>
        /* Base typography & colors */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #2d3748;
        }

        /* Top padding reduction */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e2e8f0 !important;
        }
        [data-testid="stSidebar"] hr {
            margin: 1rem 0 !important;
            border-color: #e2e8f0 !important;
        }

        /* Branding Gradient Accent */
        .brand-gradient-text {
            background: linear-gradient(135deg, #ea4b71 0%, #6b73ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            letter-spacing: -0.5px;
            display: inline-block;
        }

        /* Metric Cards */
        .dc-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            margin-bottom: 12px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .dc-card:hover {
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07);
        }
        .dc-card-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #718096;
            margin-bottom: 4px;
        }
        .dc-card-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #2d3748;
            line-height: 1.2;
        }
        .dc-card-sub {
            font-size: 0.8rem;
            color: #718096;
            margin-top: 4px;
        }

        /* High & Medium Severity Card Borders */
        .card-high {
            border-left: 4px solid #ea4b71 !important;
        }
        .card-medium {
            border-left: 4px solid #6b73ff !important;
        }
        .card-neutral {
            border-left: 4px solid #e2e8f0 !important;
        }
        .card-success {
            border-left: 4px solid #00d4aa !important;
        }

        /* Badges & Pills */
        .pill-badge {
            display: inline-flex;
            align-items: center;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .badge-high {
            background-color: #fff0f3;
            color: #ea4b71;
            border: 1px solid #ffccd5;
        }
        .badge-medium {
            background-color: #f0f2ff;
            color: #6b73ff;
            border: 1px solid #d4d8ff;
        }
        .badge-success {
            background-color: #e6faf5;
            color: #00a887;
            border: 1px solid #b3f2e4;
        }

        /* Subtle Section Header */
        .section-header {
            font-size: 1.15rem;
            font-weight: 600;
            color: #2d3748;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }

        /* Global Footer */
        .dc-footer {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            color: #718096;
            font-size: 0.82rem;
            line-height: 1.6;
        }
        .dc-footer a {
            color: #6b73ff;
            text-decoration: none;
        }

        /* Primary Action Buttons */
        div.stButton > button:first-child {
            background-color: #ea4b71;
            color: #ffffff;
            border-radius: 6px;
            border: 1px solid #d63859;
            font-weight: 600;
            padding: 0.45rem 1rem;
            transition: all 0.15s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #d63859;
            border-color: #b82d49;
            color: #ffffff;
        }

        /* Radio Buttons Pill Appearance */
        div[role="radiogroup"] > label {
            background-color: #edf2f7;
            padding: 6px 14px;
            border-radius: 6px;
            margin-right: 8px;
            border: 1px solid #e2e8f0;
        }
        div[role="radiogroup"] > label:hover {
            background-color: #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render standard footer across all dashboard views."""
    st.markdown(
        """
        <div class="dc-footer">
            <div><strong>Ministry of Health Malaysia / National Blood Centre</strong> (Pusat Darah Negara) open data</div>
            <div>Data Cutoff: <strong>2026-09-22</strong> &nbsp;·&nbsp; Model: <strong>Global LightGBM Multi-Horizon Tweedie</strong> &nbsp;·&nbsp; Prediction Interval: <strong>p10–p90</strong></div>
            <div style="margin-top: 4px; color: #a0aec0; font-style: italic;">
                Decision support system for operational blood drive planning and recall campaigns. Not a clinical diagnostic tool.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Page 1: Forecast Page
# -----------------------------------------------------------------------------
def render_forecast_page(selected_origin: str) -> None:
    """Render the 14-day multi-horizon forecast exploration page."""
    st.markdown('<h1 style="margin-bottom: 2px;">Forecast</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #718096; font-size: 1.05rem; margin-bottom: 1.5rem;">'
        "14-day blood donation forecast by facility and blood group."
        "</p>",
        unsafe_allow_html=True,
    )

    # 1. Facility & Blood Group Controls
    facility_list = get_facility_list()
    if "sb_facility" not in st.session_state:
        st.session_state["sb_facility"] = (
            "Pusat Darah Negara" if "Pusat Darah Negara" in facility_list else facility_list[0]
        )
    if "sb_group" not in st.session_state:
        st.session_state["sb_group"] = "O"

    ctrl_col1, ctrl_col2 = st.columns([3, 2])
    with ctrl_col1:
        selected_facility = st.selectbox(
            "Collection Facility",
            facility_list,
            key="sb_facility",
            help="Select one of the 22 MoH blood collection sites in Malaysia.",
        )
        st.session_state["selected_facility"] = selected_facility

    with ctrl_col2:
        selected_group = st.selectbox(
            "Blood Group",
            BLOOD_GROUPS,
            key="sb_group",
            help="Select ABO blood group.",
        )
        st.session_state["selected_group"] = selected_group

    # 2. Compact KPI / Header Cards
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    is_replay = selected_origin != DATA_CUTOFF
    origin_display_label = (
        f"{selected_origin} (Replay)" if is_replay else f"{selected_origin} (Latest)"
    )

    with kpi_col1:
        st.markdown(
            f"""
            <div class="dc-card card-neutral">
                <div class="dc-card-label">Selected Facility</div>
                <div class="dc-card-value" style="font-size: 1.15rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{selected_facility}">{selected_facility}</div>
                <div class="dc-card-sub">MoH Collection Site</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="dc-card card-high">
                <div class="dc-card-label">Blood Group</div>
                <div class="dc-card-value" style="color: #ea4b71;">Group {selected_group}</div>
                <div class="dc-card-sub">ABO Blood Typing</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="dc-card card-medium">
                <div class="dc-card-label">Forecast Origin</div>
                <div class="dc-card-value" style="font-size: 1.35rem; color: #6b73ff;">{origin_display_label}</div>
                <div class="dc-card-sub">Forecast generated as of T</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            """
            <div class="dc-card card-neutral">
                <div class="dc-card-label">Forecast Horizon</div>
                <div class="dc-card-value">14 Days</div>
                <div class="dc-card-sub">Multi-horizon point + interval</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Load Data for Facility and Group
    df_fcst_all = load_forecasts(selected_origin)
    df_act_all = load_historical_actuals()

    # Filter forecast
    fcst_sub = df_fcst_all[
        (df_fcst_all["facility"] == selected_facility)
        & (df_fcst_all["group"] == selected_group)
    ].sort_values("horizon").copy()

    if len(fcst_sub) == 0:
        st.warning(
            f"No precomputed forecast data available for {selected_facility} ({selected_group}) at origin {selected_origin}."
        )
        render_footer()
        return

    # Filter 8 weeks (56 days) of actual donations prior to origin
    origin_dt = datetime.date.fromisoformat(selected_origin)
    history_start_dt = origin_dt - datetime.timedelta(days=56)
    history_start_str = history_start_dt.isoformat()

    act_sub = df_act_all[
        (df_act_all["facility"] == selected_facility)
        & (df_act_all["group"] == selected_group)
        & (df_act_all["date"] >= history_start_str)
        & (df_act_all["date"] <= selected_origin)
    ].sort_values("date").copy()

    # 4. Main Forecast Chart (Altair)
    st.markdown('<div class="section-header">Donation Forecast & Historical Trajectory</div>', unsafe_allow_html=True)

    # Prepare datasets for Altair
    # 4a. Historical Actuals (Past 8 Weeks)
    act_chart_df = pd.DataFrame(
        {
            "date": pd.to_datetime(act_sub["date"]),
            "donations": act_sub["donations"].astype(float),
            "series": "Historical Actual (Past 8 Weeks)",
        }
    )

    # 4b. Forecast Horizon Data (14 Days)
    fcst_dates = pd.to_datetime(fcst_sub["target_date"])
    band_chart_df = pd.DataFrame(
        {
            "date": fcst_dates,
            "pred_p10": fcst_sub["pred_p10"].astype(float),
            "pred_p90": fcst_sub["pred_p90"].astype(float),
            "series": "80% Prediction Interval (p10–p90)",
        }
    )

    pred_line_df = pd.DataFrame(
        {
            "date": fcst_dates,
            "donations": fcst_sub["pred_p50"].astype(float),
            "horizon": fcst_sub["horizon"].astype(int),
            "p10": fcst_sub["pred_p10"].round(1),
            "p50": fcst_sub["pred_p50"].round(1),
            "p90": fcst_sub["pred_p90"].round(1),
            "series": "Forecast (p50)",
        }
    )

    # Check if subsequent actuals exist (for replay origins)
    has_subsequent_actuals = not fcst_sub["target"].isna().all()
    subsequent_chart_df = None
    if has_subsequent_actuals:
        subsequent_chart_df = pd.DataFrame(
            {
                "date": fcst_dates,
                "donations": fcst_sub["target"].astype(float),
                "horizon": fcst_sub["horizon"].astype(int),
                "series": "Subsequent Actual (Replay Realized)",
            }
        )

    # Construct Altair layers
    # Layer 1: p10-p90 shaded area
    band_chart = (
        alt.Chart(band_chart_df)
        .mark_area(
            opacity=0.18,
            color="#ea4b71",
        )
        .encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%d %b", labelAngle=-30)),
            y=alt.Y("pred_p10:Q", title="Daily Donations"),
            y2="pred_p90:Q",
        )
    )

    # Layer 2: Historical Actuals Line & Points (Slate #2d3748)
    line_actual = (
        alt.Chart(act_chart_df)
        .mark_line(color="#2d3748", strokeWidth=2)
        .encode(
            x=alt.X("date:T"),
            y=alt.Y("donations:Q"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d (%a)"),
                alt.Tooltip("donations:Q", title="Historical Donations", format=".0f"),
                alt.Tooltip("series:N", title="Series"),
            ],
        )
    )
    points_actual = (
        alt.Chart(act_chart_df)
        .mark_circle(color="#2d3748", size=24, opacity=0.7)
        .encode(
            x=alt.X("date:T"),
            y=alt.Y("donations:Q"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d (%a)"),
                alt.Tooltip("donations:Q", title="Historical Donations", format=".0f"),
                alt.Tooltip("series:N", title="Series"),
            ],
        )
    )

    # Layer 3: Forecast Line & Points (Rose #ea4b71)
    line_forecast = (
        alt.Chart(pred_line_df)
        .mark_line(color="#ea4b71", strokeWidth=3)
        .encode(
            x=alt.X("date:T"),
            y=alt.Y("donations:Q"),
        )
    )
    points_forecast = (
        alt.Chart(pred_line_df)
        .mark_circle(color="#ea4b71", size=55)
        .encode(
            x=alt.X("date:T"),
            y=alt.Y("donations:Q"),
            tooltip=[
                alt.Tooltip("date:T", title="Forecast Date", format="%Y-%m-%d (%a)"),
                alt.Tooltip("horizon:O", title="Horizon (Day)"),
                alt.Tooltip("p50:Q", title="Forecast (p50)", format=".1f"),
                alt.Tooltip("p10:Q", title="p10 Bound", format=".1f"),
                alt.Tooltip("p90:Q", title="p90 Bound", format=".1f"),
            ],
        )
    )

    # Layer 4: Vertical Forecast Boundary Line at Origin Date (Slate #718096 dashed)
    origin_line_df = pd.DataFrame({"date": [pd.to_datetime(selected_origin)]})
    boundary_rule = (
        alt.Chart(origin_line_df)
        .mark_rule(
            color="#718096",
            strokeDash=[4, 4],
            strokeWidth=1.5,
        )
        .encode(x=alt.X("date:T"))
    )
    boundary_text = (
        alt.Chart(origin_line_df)
        .mark_text(
            text="Forecast Boundary (Origin T)",
            align="right",
            dx=-8,
            dy=-140,
            fontSize=11,
            color="#4a5568",
            fontWeight=600,
        )
        .encode(x=alt.X("date:T"), y=alt.value(20))
    )

    # Assemble layers
    chart_layers = [
        band_chart,
        line_actual,
        points_actual,
        boundary_rule,
        boundary_text,
        line_forecast,
        points_forecast,
    ]

    # Layer 5 (Optional): Subsequent Actuals for Replay Origins (Emerald #00d4aa dashed)
    if has_subsequent_actuals and subsequent_chart_df is not None:
        line_subsequent = (
            alt.Chart(subsequent_chart_df)
            .mark_line(color="#00d4aa", strokeWidth=2.2, strokeDash=[5, 3])
            .encode(
                x=alt.X("date:T"),
                y=alt.Y("donations:Q"),
            )
        )
        points_subsequent = (
            alt.Chart(subsequent_chart_df)
            .mark_square(color="#00a887", size=40)
            .encode(
                x=alt.X("date:T"),
                y=alt.Y("donations:Q"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%Y-%m-%d (%a)"),
                    alt.Tooltip("horizon:O", title="Horizon"),
                    alt.Tooltip("donations:Q", title="Realized Actual", format=".0f"),
                    alt.Tooltip("series:N", title="Series"),
                ],
            )
        )
        chart_layers.extend([line_subsequent, points_subsequent])

    combined_chart = (
        alt.layer(*chart_layers)
        .properties(height=420)
        .configure_axis(
            gridColor="#edf2f7",
            gridDash=[2, 2],
            domainColor="#e2e8f0",
            labelColor="#718096",
            titleColor="#4a5568",
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure_view(strokeOpacity=0)
    )

    st.altair_chart(combined_chart, use_container_width=True)

    # Visual legend explanation below chart
    leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)
    with leg_col1:
        st.markdown(
            '<div style="font-size: 0.82rem; color: #2d3748;"><span style="color: #2d3748; font-size: 1.1rem;">― ●</span> <strong>Historical Actual</strong> (Past 8 Wks)</div>',
            unsafe_allow_html=True,
        )
    with leg_col2:
        st.markdown(
            '<div style="font-size: 0.82rem; color: #ea4b71;"><span style="color: #ea4b71; font-size: 1.1rem;">― ●</span> <strong>Forecast (p50)</strong> (14 Days)</div>',
            unsafe_allow_html=True,
        )
    with leg_col3:
        st.markdown(
            '<div style="font-size: 0.82rem; color: #ea4b71;"><span style="display:inline-block; width:14px; height:10px; background:#ea4b71; opacity:0.3; border-radius:2px; margin-right:4px;"></span><strong>p10–p90 Band</strong> (80% Interval)</div>',
            unsafe_allow_html=True,
        )
    with leg_col4:
        if has_subsequent_actuals:
            st.markdown(
                '<div style="font-size: 0.82rem; color: #00a887;"><span style="color: #00d4aa; font-size: 1.1rem;">┄ ■</span> <strong>Subsequent Actual</strong> (Replay)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size: 0.82rem; color: #718096;">┊ <strong>Origin Boundary</strong> (T)</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <p style="font-size: 0.78rem; color: #a0aec0; margin-top: 10px; margin-bottom: 24px;">
            <em>Statistical Note: The shaded p10–p90 prediction interval reflects calibrated 80% statistical coverage from LightGBM quantile regression. It does not represent guaranteed clinical bounds or deterministic collection limits.</em>
        </p>
        """,
        unsafe_allow_html=True,
    )

    # 5. Compact Forecast Table Below Chart
    st.markdown('<div class="section-header">14-Day Daily Forecast Values</div>', unsafe_allow_html=True)

    table_data = []
    for _, row in fcst_sub.iterrows():
        entry = {
            "Horizon": f"Day +{int(row['horizon'])}",
            "Date": str(row["target_date"]),
            "Day of Week": pd.to_datetime(row["target_date"]).strftime("%A"),
            "p10 (Lower)": f"{row['pred_p10']:.1f}",
            "p50 (Median)": f"{row['pred_p50']:.1f}",
            "p90 (Upper)": f"{row['pred_p90']:.1f}",
        }
        if has_subsequent_actuals:
            act_val = row["target"]
            entry["Actual Realized"] = f"{act_val:.0f}" if pd.notna(act_val) else "—"
        table_data.append(entry)

    df_table = pd.DataFrame(table_data)
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
    )

    render_footer()


# -----------------------------------------------------------------------------
# Page 2: Shortfall Alerts Page
# -----------------------------------------------------------------------------
def render_alerts_page(selected_origin: str) -> None:
    """Render the 7-day shortfall alerts and SHAP reason explanation page."""
    st.markdown('<h1 style="margin-bottom: 2px;">Shortfall Alerts</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #718096; font-size: 1.05rem; margin-bottom: 1.5rem;">'
        "Facilities and blood groups where the forecast indicates a potential 7-day donation shortfall."
        "</p>",
        unsafe_allow_html=True,
    )

    # 1. Load alerts for current origin
    df_alerts = load_alerts(selected_origin)

    if len(df_alerts) == 0:
        st.info(f"No shortfall alerts flagged for forecast origin {selected_origin}.")
        render_footer()
        return

    # 2. KPI Summary Cards
    total_alerts = len(df_alerts)
    high_alerts = int((df_alerts["severity"] == "HIGH").sum())
    med_alerts = int((df_alerts["severity"] == "MEDIUM").sum())

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="dc-card card-neutral">
                <div class="dc-card-label">Total Shortfall Alerts</div>
                <div class="dc-card-value">{total_alerts}</div>
                <div class="dc-card-sub">Flagged 7-day collection series</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="dc-card card-high">
                <div class="dc-card-label">High Severity Alerts</div>
                <div class="dc-card-value" style="color: #ea4b71;">{high_alerts}</div>
                <div class="dc-card-sub">Forecast &lt; 70% of 3-yr typical volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="dc-card card-medium">
                <div class="dc-card-label">Medium Severity Alerts</div>
                <div class="dc-card-value" style="color: #6b73ff;">{med_alerts}</div>
                <div class="dc-card-sub">Forecast 70%–80% of 3-yr typical volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Filter Controls
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    filt_col1, filt_col2, _filt_col3 = st.columns([2, 2, 3])

    with filt_col1:
        group_filter = st.selectbox(
            "Filter by Blood Group",
            ["All", "A", "B", "AB", "O"],
            index=0,
            key="alert_filter_group",
        )

    with filt_col2:
        severity_filter = st.selectbox(
            "Filter by Severity",
            ["All", "HIGH", "MEDIUM"],
            index=0,
            key="alert_filter_severity",
        )

    # Apply filters
    filtered_df = df_alerts.copy()
    if group_filter != "All":
        filtered_df = filtered_df[filtered_df["group"] == group_filter]
    if severity_filter != "All":
        filtered_df = filtered_df[filtered_df["severity"] == severity_filter]

    # Ensure sorting: HIGH first, then MEDIUM, then most severe negative deficit
    severity_order = {"HIGH": 0, "MEDIUM": 1, "NONE": 2}
    filtered_df["_sev_sort"] = filtered_df["severity"].map(severity_order)
    filtered_df = filtered_df.sort_values(
        by=["_sev_sort", "deficit_pct", "typical_7d"], ascending=[True, True, False]
    ).drop(columns=["_sev_sort"])

    # 4. Alert Table Display
    st.markdown(
        f'<div class="section-header">Flagged Shortfalls ({len(filtered_df)} Series Matching Filters)</div>',
        unsafe_allow_html=True,
    )

    if len(filtered_df) == 0:
        st.info("No shortfall alerts match the current filter selection.")
    else:
        # Prepare display dataframe
        display_df = pd.DataFrame(
            {
                "Facility": filtered_df["facility"],
                "Blood Group": filtered_df["group"],
                "Severity": filtered_df["severity"],
                "Forecast 7-day": filtered_df["forecast_7d"].map(lambda x: f"{x:.1f}"),
                "Typical 7-day": filtered_df["typical_7d"].map(lambda x: f"{x:.1f}"),
                "% Below Typical": filtered_df["deficit_pct"].map(lambda x: f"{x:.1%}"),
                "Top 3 Reasons": filtered_df["reasons_str"],
            }
        )

        # Interactive dataframe with row selection
        event = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="alert_table_selection",
        )

        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #a0aec0; margin-top: 6px; margin-bottom: 20px;">
                <em>Interpretability Note: Top 3 reasons are derived from TreeSHAP feature contributions for the h=1 forecast. They indicate observational statistical associations and should not be construed as clinical or causal mechanisms.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 5. Alert Detail & Seamless Navigation to Forecast Page
        st.markdown('<div class="section-header">Alert Inspection & Drill-Down</div>', unsafe_allow_html=True)

        selected_row_idx = None
        if event and event.selection and event.selection.rows:
            selected_row_idx = event.selection.rows[0]

        # Allow user to either click a table row or pick from the filtered dropdown
        alert_options = [
            f"{r['facility']} (Group {r['group']}) — {r['severity']} [{r['deficit_pct']:.1%}]"
            for _, r in filtered_df.iterrows()
        ]

        default_select_idx = selected_row_idx if selected_row_idx is not None else 0
        default_select_idx = min(default_select_idx, len(alert_options) - 1)

        picked_alert_str = st.selectbox(
            "Select Alert to Inspect in Forecast View",
            options=alert_options,
            index=default_select_idx,
            key="inspect_alert_select",
        )

        # Retrieve selected row
        picked_idx = alert_options.index(picked_alert_str)
        target_row = filtered_df.iloc[picked_idx]

        target_fac = target_row["facility"]
        target_grp = target_row["group"]
        target_sev = target_row["severity"]
        target_def = target_row["deficit_pct"]
        target_fcst = target_row["forecast_7d"]
        target_typ = target_row["typical_7d"]
        r1 = target_row["reason_1"]
        r2 = target_row["reason_2"]
        r3 = target_row["reason_3"]

        badge_class = "badge-high" if target_sev == "HIGH" else "badge-medium"
        card_border_class = "card-high" if target_sev == "HIGH" else "card-medium"

        # Action card
        st.markdown(
            f"""
            <div class="dc-card {card_border_class}" style="margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 1.15rem; font-weight: 700; color: #2d3748;">
                        {target_fac} &nbsp;·&nbsp; Blood Group <span style="color: #ea4b71;">{target_grp}</span>
                    </div>
                    <div>
                        <span class="pill-badge {badge_class}">{target_sev} SHORTFALL</span>
                    </div>
                </div>
                <div style="color: #4a5568; font-size: 0.92rem; margin-bottom: 10px;">
                    Forecast 7-day: <strong>{target_fcst:.1f} donations</strong> vs typical <strong>{target_typ:.1f}</strong> (<strong style="color: #ea4b71;">{target_def:.1%}</strong>)
                </div>
                <div style="background-color: #f8fafc; border-radius: 6px; padding: 10px 14px; border: 1px solid #edf2f7; font-size: 0.85rem; color: #4a5568;">
                    <div style="font-weight: 600; color: #718096; margin-bottom: 4px; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em;">Key Forecast Drivers (SHAP)</div>
                    <div>1. {r1}</div>
                    <div>2. {r2}</div>
                    <div>3. {r3}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_col1, _nav_col2 = st.columns([2, 5])
        with nav_col1:
            if st.button("📈 Open in Forecast Chart", key="btn_drilldown"):
                st.session_state["target_facility"] = target_fac
                st.session_state["target_group"] = target_grp
                st.session_state["nav_target"] = "Forecast"
                st.rerun()

    render_footer()


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

    # 1. Sidebar Brand & Navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 1.25rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.7rem;">🩸</span>
                    <span class="brand-gradient-text" style="font-size: 1.45rem;">DONORCAST</span>
                </div>
                <div style="color: #718096; font-size: 0.85rem; font-weight: 500; margin-top: 2px;">
                    Blood Donation Forecasting
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Page Switcher
        nav_options = ["Forecast", "Shortfall Alerts"]
        current_nav = st.session_state.get("nav_page", "Forecast")
        if current_nav not in nav_options:
            current_nav = "Forecast"

        selected_page = st.radio(
            "Navigation",
            nav_options,
            index=nav_options.index(current_nav),
            key="nav_radio",
            label_visibility="collapsed",
        )
        st.session_state["nav_page"] = selected_page

        st.markdown("<hr/>", unsafe_allow_html=True)

        # Global Origin Date Selector
        st.markdown(
            '<div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #718096; letter-spacing: 0.05em; margin-bottom: 4px;">Forecast Origin</div>',
            unsafe_allow_html=True,
        )

        available_origins = get_available_origins()
        origin_labels = [REPLAY_LABELS.get(orig, f"{orig} (Replay)") for orig in available_origins]

        default_origin = st.session_state.get("selected_origin", DATA_CUTOFF)
        if default_origin not in available_origins:
            default_origin = available_origins[0]
        origin_idx = available_origins.index(default_origin)

        chosen_label = st.selectbox(
            "Forecast Origin Date",
            options=origin_labels,
            index=origin_idx,
            key="sb_origin",
            label_visibility="collapsed",
            help="Choose latest cutoff (2026-09-22) or any historical test-period replay origin.",
        )

        chosen_origin = available_origins[origin_labels.index(chosen_label)]
        st.session_state["selected_origin"] = chosen_origin

        st.markdown("<hr/>", unsafe_allow_html=True)

        # Sidebar Footer & Status
        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #718096; line-height: 1.5;">
                <div style="margin-bottom: 6px;">
                    <span style="font-weight: 600; color: #4a5568;">Data Cutoff:</span><br/>
                    <code>2026-09-22</code>
                </div>
                <div style="margin-bottom: 6px;">
                    <span style="font-weight: 600; color: #4a5568;">Data Source:</span><br/>
                    MoH Malaysia / National Blood Centre
                </div>
                <div style="margin-top: 10px; color: #a0aec0; font-size: 0.74rem;">
                    Decision support system.<br/>Not a clinical tool.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Main Page Content Routing
    active_origin = st.session_state.get("selected_origin", DATA_CUTOFF)
    if st.session_state["nav_page"] == "Forecast":
        render_forecast_page(selected_origin=active_origin)
    elif st.session_state["nav_page"] == "Shortfall Alerts":
        render_alerts_page(selected_origin=active_origin)


if __name__ == "__main__":
    main()
