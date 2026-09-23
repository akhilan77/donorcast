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

NAV_LABELS: dict[str, str] = {
    "Overview": "🏠  Executive Overview",
    "Forecast": "📈  14-Day Forecast",
    "Shortfall Alerts": "⚠️  Shortfall Alerts",
}

# Geographical coordinates (Latitude, Longitude, State, Region) for all 22 MoH collection centers
FACILITY_METADATA: dict[str, dict[str, str | float]] = {
    "Hospital Duchess Of Kent": {"lat": 5.8458, "lon": 118.1068, "state": "Sabah", "region": "East Malaysia"},
    "Hospital Melaka": {"lat": 2.2198, "lon": 102.2612, "state": "Melaka", "region": "Peninsular"},
    "Hospital Miri": {"lat": 4.3804, "lon": 113.9859, "state": "Sarawak", "region": "East Malaysia"},
    "Hospital Pulau Pinang": {"lat": 5.4172, "lon": 100.3110, "state": "Pulau Pinang", "region": "Peninsular"},
    "Hospital Queen Elizabeth II": {"lat": 5.9525, "lon": 116.0848, "state": "Sabah", "region": "East Malaysia"},
    "Hospital Raja Perempuan Zainab II": {"lat": 6.1264, "lon": 102.2458, "state": "Kelantan", "region": "Peninsular"},
    "Hospital Raja Permaisuri Bainun": {"lat": 4.6042, "lon": 101.0906, "state": "Perak", "region": "Peninsular"},
    "Hospital Seberang Jaya": {"lat": 5.3948, "lon": 100.4082, "state": "Pulau Pinang", "region": "Peninsular"},
    "Hospital Seri Manjung": {"lat": 4.1950, "lon": 100.6663, "state": "Perak", "region": "Peninsular"},
    "Hospital Sibu": {"lat": 2.3087, "lon": 111.8906, "state": "Sarawak", "region": "East Malaysia"},
    "Hospital Sultan Haji Ahmad Shah": {"lat": 3.4478, "lon": 102.4208, "state": "Pahang", "region": "Peninsular"},
    "Hospital Sultanah Aminah": {"lat": 1.4583, "lon": 103.7461, "state": "Johor", "region": "Peninsular"},
    "Hospital Sultanah Bahiyah": {"lat": 6.1495, "lon": 100.4069, "state": "Kedah", "region": "Peninsular"},
    "Hospital Sultanah Nora Ismail": {"lat": 1.8496, "lon": 102.9469, "state": "Johor", "region": "Peninsular"},
    "Hospital Sultanah Nur Zahirah": {"lat": 5.3283, "lon": 103.1517, "state": "Terengganu", "region": "Peninsular"},
    "Hospital Taiping": {"lat": 4.8569, "lon": 100.7411, "state": "Perak", "region": "Peninsular"},
    "Hospital Tawau": {"lat": 4.2541, "lon": 117.8931, "state": "Sabah", "region": "East Malaysia"},
    "Hospital Tengku Ampuan Afzan": {"lat": 3.8038, "lon": 103.3235, "state": "Pahang", "region": "Peninsular"},
    "Hospital Tengku Ampuan Rahimah": {"lat": 3.0219, "lon": 101.4397, "state": "Selangor", "region": "Peninsular"},
    "Hospital Tuanku Jaafar": {"lat": 2.7097, "lon": 101.9472, "state": "Negeri Sembilan", "region": "Peninsular"},
    "Hospital Umum Sarawak": {"lat": 1.5433, "lon": 110.3424, "state": "Sarawak", "region": "East Malaysia"},
    "Pusat Darah Negara": {"lat": 3.1706, "lon": 101.7032, "state": "W.P. Kuala Lumpur", "region": "Peninsular"},
}


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
                "unit_deficit",
                "deficit_pct",
                "shortfall_ratio",
                "reason_1",
                "reason_2",
                "reason_3",
                "reasons_str",
                "suggested_action",
                "actual_7d",
                "actual_shortfall",
                "actual_severity",
            ]
        )
    df = pd.read_parquet(alert_file)
    df["origin_date"] = df["origin_date"].astype(str)
    if "unit_deficit" not in df.columns and "typical_7d" in df.columns and "forecast_7d" in df.columns:
        df["unit_deficit"] = (df["typical_7d"] - df["forecast_7d"]).clip(lower=0.0).round(1)
    if "suggested_action" not in df.columns:
        df["suggested_action"] = df.apply(
            lambda r: f"Schedule an extra mobile drive and message regular group {r.get('group', 'all')} donors this week.",
            axis=1,
        )
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
    return sorted(FACILITY_METADATA.keys())


# -----------------------------------------------------------------------------
# Professional Dark Operations UI Styles
# -----------------------------------------------------------------------------
def apply_custom_styles() -> None:
    """Inject polished dark theme CSS tokens and component styles."""
    st.markdown(
        """
        <style>
        /* 1. Header & Sidebar Controls (Keep collapse/expand button visible & accessible) */
        header[data-testid="stHeader"] {
            background: transparent !important;
            color: #94A3B8 !important;
        }

        [data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            position: fixed !important;
            top: 12px !important;
            left: 12px !important;
            z-index: 999999 !important;
            background: #1E293B !important;
            border: 1px solid #3B82F6 !important;
            border-radius: 8px !important;
            padding: 4px 8px !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
            cursor: pointer !important;
        }

        [data-testid="stSidebarCollapsedControl"]:hover {
            background: #2563EB !important;
            border-color: #60A5FA !important;
        }

        /* 2. Base dark typography & theme palette */
        html, body, [data-testid="stAppViewContainer"], .main {
            background-color: #0A0E1A !important;
            color: #E2E8F0 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }

        /* 3. Main container width & top alignment */
        .block-container {
            padding-top: 1.25rem !important;
            padding-bottom: 3rem !important;
            max-width: 1260px !important;
        }

        /* 4. Headings styling */
        h1, h2, h3 {
            font-weight: 700 !important;
            color: #F8FAFC !important;
            letter-spacing: -0.025em !important;
        }
        h1 {
            font-size: 28px !important;
            margin-bottom: 4px !important;
        }
        .page-subtitle {
            font-size: 14px !important;
            font-weight: 400 !important;
            color: #94A3B8 !important;
            margin-bottom: 22px !important;
            line-height: 1.5 !important;
        }
        .section-title {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #F1F5F9 !important;
            margin-top: 24px !important;
            margin-bottom: 12px !important;
            letter-spacing: -0.01em !important;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* 5. Persistent Top Info Bar */
        .top-meta-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            background: linear-gradient(180deg, #131B2E 0%, #0F172A 100%);
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 12px 20px;
            margin-bottom: 22px;
            font-size: 13px;
            color: #94A3B8;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }
        .top-meta-item strong {
            color: #F8FAFC;
            font-weight: 600;
        }
        .top-meta-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 12px;
            letter-spacing: 0.02em;
        }

        /* 6. Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0D1322 !important;
            border-right: 1px solid #1E293B !important;
        }
        [data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }
        [data-testid="stSidebar"] hr {
            margin: 1.25rem 0 !important;
            border-color: #1E293B !important;
        }

        /* 7. Sidebar Navigation Cards & Radios (Complete bullet removal) */
        .nav-section-title {
            font-size: 11px !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            color: #94A3B8 !important;
            margin-bottom: 8px !important;
        }

        /* Hide all radio circle elements and indicators across BaseWeb and standard DOM */
        [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child,
        [data-testid="stRadio"] [data-baseweb="radio"] > span,
        [data-testid="stRadio"] input[type="radio"],
        [data-testid="stRadio"] svg {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        [data-testid="stRadio"] div[role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 8px !important;
        }

        [data-testid="stRadio"] div[role="radiogroup"] > label,
        [data-testid="stRadio"] [data-baseweb="radio"] {
            display: flex !important;
            align-items: center !important;
            width: 100% !important;
            padding: 12px 14px !important;
            background-color: #162032 !important;
            border: 1px solid #27354E !important;
            border-left: 4px solid #475569 !important;
            border-radius: 8px !important;
            font-size: 13.5px !important;
            font-weight: 600 !important;
            color: #CBD5E1 !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
            margin: 0 !important;
        }

        [data-testid="stRadio"] div[role="radiogroup"] > label:hover,
        [data-testid="stRadio"] [data-baseweb="radio"]:hover {
            background-color: #1E2E48 !important;
            color: #FFFFFF !important;
            border-color: #3B82F680 !important;
            border-left-color: #3B82F6 !important;
            transform: translateX(3px) !important;
        }

        [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
        [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
        [data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) {
            background: linear-gradient(90deg, #1E3A8A 0%, #162238 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #3B82F6 !important;
            border-left: 4px solid #60A5FA !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
        }

        [data-testid="stRadio"] div[role="radiogroup"] > label p,
        [data-testid="stRadio"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p,
        [data-testid="stRadio"] [data-baseweb="radio"] div[data-testid="stMarkdownContainer"] p {
            font-size: 13.5px !important;
            font-weight: 600 !important;
            color: inherit !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        /* 8. Unified Dark Card System */
        .op-card {
            background-color: #111827;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            transition: all 0.2s ease;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        }
        .op-card:hover {
            border-color: #3B82F660;
            box-shadow: 0 6px 16px rgba(0,0,0,0.35);
        }
        .op-card-label {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94A3B8;
            margin-bottom: 6px;
        }
        .op-card-value {
            font-size: 26px;
            font-weight: 800;
            color: #F8FAFC;
            line-height: 1.15;
        }
        .op-card-sub {
            font-size: 12px;
            font-weight: 400;
            color: #94A3B8;
            margin-top: 4px;
        }

        /* Status Accents (Left Border 4px) */
        .card-rose {
            border-left: 4px solid #EF4444 !important;
        }
        .card-indigo {
            border-left: 4px solid #818CF8 !important;
        }
        .card-emerald {
            border-left: 4px solid #10B981 !important;
        }
        .card-blue {
            border-left: 4px solid #3B82F6 !important;
        }
        .card-neutral {
            border-left: 4px solid #475569 !important;
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
            background-color: #45121E;
            color: #F87171;
            border: 1px solid #EF444460;
        }
        .pill-med {
            background-color: #23224B;
            color: #A5B4FC;
            border: 1px solid #6366F160;
        }
        .pill-norm {
            background-color: #0E3024;
            color: #34D399;
            border: 1px solid #10B98160;
        }

        /* Buttons & Actions */
        div.stButton > button:first-child, div.stDownloadButton > button:first-child {
            background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 100%) !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            border: 1px solid #3B82F6 !important;
            font-size: 13.5px !important;
            font-weight: 600 !important;
            padding: 8px 18px !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
        }
        div.stButton > button:first-child:hover, div.stDownloadButton > button:first-child:hover {
            background: linear-gradient(180deg, #1D4ED8 0%, #1E40AF 100%) !important;
            border-color: #60A5FA !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.45) !important;
            transform: translateY(-1px) !important;
        }

        /* Select boxes & Inputs */
        div[data-baseweb="select"] > div {
            background-color: #111827 !important;
            border-color: #1E293B !important;
            border-radius: 8px !important;
            color: #F8FAFC !important;
        }

        /* Dataframes */
        [data-testid="stDataFrame"] {
            border: 1px solid #1E293B !important;
            border-radius: 8px !important;
            background-color: #111827 !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #111827 !important;
            border: 1px solid #1E293B !important;
            border-radius: 6px !important;
            color: #94A3B8 !important;
            font-size: 13px !important;
        }

        /* Global Footer */
        .sidebar-footer {
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid #1E293B;
            color: #64748B;
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
        '<span class="top-meta-badge" style="color:#34D399; border:1px solid #10B98140; background:#0E3024;">● Operational</span>'
        if not is_replay
        else '<span class="top-meta-badge" style="color:#FBBF24; border:1px solid #F59E0B40; background:#362612;">↺ Replay Mode</span>'
    )

    st.markdown(
        f"""
        <div class="top-meta-bar">
            <div class="top-meta-item">
                <strong>Data Cutoff:</strong> 22 Sep 2026 &nbsp;·&nbsp;
                <strong>Source:</strong> Ministry of Health Malaysia / National Blood Centre
            </div>
            <div class="top-meta-item">
                <strong>Forecast Origin:</strong> {origin_formatted} &nbsp; {badge_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Component: Reliable Interactive National Facility Status Map
# -----------------------------------------------------------------------------
def render_facility_map(df_alerts: pd.DataFrame) -> None:
    """Render an interactive geographic scatter map of all 22 facilities across Malaysia."""
    all_facs = get_facility_list()
    map_rows = []

    for fac in all_facs:
        meta = FACILITY_METADATA.get(
            fac, {"lat": 3.1706, "lon": 101.7032, "state": "Unknown", "region": "Peninsular"}
        )
        fac_alerts = df_alerts[df_alerts["facility"] == fac]

        if (fac_alerts["severity"] == "HIGH").any():
            status_desc = "High Shortfall"
            status_color = "#EF4444"
            point_size = 280
            priority_rank = 1
        elif (fac_alerts["severity"] == "MEDIUM").any():
            status_desc = "Moderate Shortfall"
            status_color = "#818CF8"
            point_size = 220
            priority_rank = 2
        else:
            status_desc = "Stable Operations"
            status_color = "#10B981"
            point_size = 140
            priority_rank = 3

        fac_def = (
            float(fac_alerts["unit_deficit"].sum())
            if len(fac_alerts) > 0 and "unit_deficit" in fac_alerts.columns
            else 0.0
        )
        flagged_groups = ", ".join(fac_alerts["group"].tolist()) if len(fac_alerts) > 0 else "None"

        map_rows.append(
            {
                "facility": fac,
                "lat": float(meta["lat"]),
                "lon": float(meta["lon"]),
                "state": str(meta["state"]),
                "region": str(meta["region"]),
                "status": status_desc,
                "status_color": status_color,
                "point_size": point_size,
                "priority_rank": priority_rank,
                "deficit_units": round(fac_def, 0),
                "flagged_groups": flagged_groups,
            }
        )

    df_map = pd.DataFrame(map_rows)

    color_scale = alt.Scale(
        domain=["High Shortfall", "Moderate Shortfall", "Stable Operations"],
        range=["#EF4444", "#818CF8", "#10B981"],
    )

    points = (
        alt.Chart(df_map)
        .mark_circle(opacity=0.9, stroke="#FFFFFF", strokeWidth=1.2)
        .encode(
            x=alt.X(
                "lon:Q",
                scale=alt.Scale(domain=[99.0, 119.5]),
                axis=alt.Axis(
                    title="Longitude (°E)",
                    gridColor="#1E293B",
                    labelColor="#94A3B8",
                    titleColor="#CBD5E1",
                    domainColor="#334155",
                ),
            ),
            y=alt.Y(
                "lat:Q",
                scale=alt.Scale(domain=[1.0, 7.5]),
                axis=alt.Axis(
                    title="Latitude (°N)",
                    gridColor="#1E293B",
                    labelColor="#94A3B8",
                    titleColor="#CBD5E1",
                    domainColor="#334155",
                ),
            ),
            color=alt.Color(
                "status:N",
                scale=color_scale,
                legend=alt.Legend(
                    title="Operations Status",
                    orient="top-right",
                    labelColor="#E2E8F0",
                    titleColor="#F8FAFC",
                ),
            ),
            size=alt.Size("point_size:Q", legend=None),
            tooltip=[
                alt.Tooltip("facility:N", title="Facility"),
                alt.Tooltip("state:N", title="State"),
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("status:N", title="Status"),
                alt.Tooltip("flagged_groups:N", title="Flagged ABO Groups"),
                alt.Tooltip("deficit_units:Q", title="Projected Deficit (units)", format=".0f"),
            ],
        )
    )

    labels = (
        alt.Chart(df_map)
        .mark_text(
            align="left",
            baseline="middle",
            dx=10,
            fontSize=10,
            fontWeight="bold",
            color="#E2E8F0",
        )
        .encode(
            x=alt.X("lon:Q"),
            y=alt.Y("lat:Q"),
            text=alt.Text("facility:N"),
        )
    )

    chart = (
        alt.layer(points, labels)
        .properties(height=380)
        .configure_view(strokeOpacity=0)
    )

    st.altair_chart(chart, use_container_width=True)


# -----------------------------------------------------------------------------
# Page 0: Executive Command Center Landing Page
# -----------------------------------------------------------------------------
def render_landing_page(selected_origin: str) -> None:
    """Render the executive summary overview and operational situation command landing page."""
    render_top_bar(selected_origin)

    st.markdown("<h1>Executive Operations Overview</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        "National blood supply pulse, geospatial risk tracking, and 14-day early warning across 22 Ministry of Health collection centers."
        "</div>",
        unsafe_allow_html=True,
    )

    df_fcst = load_forecasts(selected_origin)
    df_alerts = load_alerts(selected_origin)

    # 1. High-Level National Forecast Totals
    h1_national = float(df_fcst[df_fcst["horizon"] == 1]["pred_p50"].sum()) if len(df_fcst) > 0 else 0.0
    next_7d_national = (
        float(df_fcst[df_fcst["horizon"].isin(range(1, 8))]["pred_p50"].sum())
        if len(df_fcst) > 0
        else 0.0
    )
    total_alerts_count = len(df_alerts)
    high_alerts_count = int((df_alerts["severity"] == "HIGH").sum())
    total_deficit_units = (
        float(df_alerts["unit_deficit"].sum()) if "unit_deficit" in df_alerts.columns else 0.0
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class="op-card card-blue">
                <div class="op-card-label">NATIONAL TODAY (H=1)</div>
                <div class="op-card-value">{h1_national:,.0f}</div>
                <div class="op-card-sub">Expected daily collections</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="op-card card-blue">
                <div class="op-card-label">7-DAY OUTLOOK</div>
                <div class="op-card-value">{next_7d_national:,.0f}</div>
                <div class="op-card-sub">Projected cumulative volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        status_color = "#EF4444" if high_alerts_count > 0 else "#818CF8"
        status_card = "card-rose" if high_alerts_count > 0 else "card-indigo"
        st.markdown(
            f"""
            <div class="op-card {status_card}">
                <div class="op-card-label">FLAGGED SITES</div>
                <div class="op-card-value" style="color: {status_color};">{total_alerts_count} <span style="font-size: 14px; font-weight: 500;">({high_alerts_count} High)</span></div>
                <div class="op-card-sub">Collection series with shortfall</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="op-card card-rose">
                <div class="op-card-label">NET DEFICIT GAP</div>
                <div class="op-card-value" style="color: #F87171;">−{total_deficit_units:,.0f} <span style="font-size: 14px; font-weight: 500;">units</span></div>
                <div class="op-card-sub">Below 28-day baseline demand</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. National Geographic Situation Map & Regional Pulse
    st.markdown('<div class="section-title">📍 National Geographic Facility Status</div>', unsafe_allow_html=True)
    render_facility_map(df_alerts)

    # 3. National ABO Group Shortfall Breakdown
    st.markdown('<div class="section-title">🩸 National Shortfall by Blood Group</div>', unsafe_allow_html=True)
    abo_cols = st.columns(4)
    for i, grp in enumerate(BLOOD_GROUPS):
        grp_df = df_alerts[df_alerts["group"] == grp]
        grp_def = float(grp_df["unit_deficit"].sum()) if "unit_deficit" in grp_df.columns else 0.0
        grp_count = len(grp_df)
        with abo_cols[i]:
            st.markdown(
                f"""
                <div class="op-card card-neutral" style="padding: 14px 18px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 17px; font-weight: 800; color: #F1F5F9;">Group {grp}</span>
                        <span style="font-size: 12px; color: #94A3B8;">{grp_count} sites flagged</span>
                    </div>
                    <div style="font-size: 20px; font-weight: 700; color: {'#EF4444' if grp_def > 0 else '#10B981'}; margin-top: 6px;">
                        {'−' + f'{grp_def:,.0f} units short' if grp_def > 0 else 'Balanced Supply'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 4. Urgent Action Priority Center
    if high_alerts_count > 0:
        st.markdown('<div class="section-title">🚨 Top Operational Priorities (High Shortfall)</div>', unsafe_allow_html=True)
        top_high = df_alerts[df_alerts["severity"] == "HIGH"].head(3)
        for _, row in top_high.iterrows():
            fac = row["facility"]
            grp = row["group"]
            def_units = float(row.get("unit_deficit", 0.0))
            def_pct = abs(float(row.get("deficit_pct", 0.0)))
            r1 = row.get("reason_1", "Low seasonal trend")
            action = row.get("suggested_action", f"Schedule mobile blood drive and message group {grp} donors.")
            st.markdown(
                f"""
                <div class="op-card card-rose" style="margin-bottom: 10px; padding: 14px 18px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <div style="font-size: 15px; font-weight: 700; color: #F8FAFC;">
                            {fac} &nbsp;·&nbsp; Blood Group <span style="color: #F87171;">{grp}</span>
                        </div>
                        <span class="status-pill pill-high">CRITICAL −{def_units:.0f} UNITS ({def_pct:.0%})</span>
                    </div>
                    <div style="font-size: 12.5px; color: #94A3B8; margin-bottom: 8px;">
                        <strong>Key Factor:</strong> {r1}
                    </div>
                    <div style="background-color: #0C281E; border-radius: 6px; padding: 10px 14px; border: 1px solid #10B98140; font-size: 12.5px; color: #E2E8F0;">
                        <strong style="color: #34D399;">Action:</strong> {action}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 5. Command Action Cards
    st.markdown('<div class="section-title">⚡ Operational Navigation</div>', unsafe_allow_html=True)
    nav_col1, nav_col2 = st.columns(2)

    with nav_col1:
        st.markdown(
            f"""
            <div class="op-card card-rose" style="min-height: 130px;">
                <div style="font-size: 16px; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
                    ⚠️ Shortfall Alerts & Prescriptions ({total_alerts_count} Flagged)
                </div>
                <div style="font-size: 13px; color: #94A3B8; line-height: 1.5; margin-bottom: 12px;">
                    Review collection series falling below 70%–80% of typical volume and access automated operational recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Shortfall Alerts →", key="btn_goto_alerts"):
            st.session_state["nav_target"] = "Shortfall Alerts"
            st.rerun()

    with nav_col2:
        st.markdown(
            """
            <div class="op-card card-blue" style="min-height: 130px;">
                <div style="font-size: 16px; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
                    📈 14-Day Collection Forecasts
                </div>
                <div style="font-size: 13px; color: #94A3B8; line-height: 1.5; margin-bottom: 12px;">
                    Inspect daily donor predictions, 80% statistical coverage intervals (p10–p90), and historical actual comparisons.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open 14-Day Forecast →", key="btn_goto_forecast"):
            st.session_state["nav_target"] = "Forecast"
            st.rerun()


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
    h1_val = fcst_sub[fcst_sub["horizon"] == 1]["pred_p50"].iloc[0] if len(fcst_sub) > 0 else 0.0
    next_7d_val = fcst_sub[fcst_sub["horizon"].isin(range(1, 8))]["pred_p50"].sum()

    # STATUS: Check if this facility x group is in alerts for this origin
    alert_match = df_alerts_all[
        (df_alerts_all["facility"] == selected_facility)
        & (df_alerts_all["group"] == selected_group)
    ]
    if len(alert_match) > 0:
        sev = str(alert_match["severity"].iloc[0])
        unit_gap = float(alert_match["unit_deficit"].iloc[0]) if "unit_deficit" in alert_match.columns else 0.0
        if sev == "HIGH":
            status_text = "High Shortfall"
            status_color = "#F87171"
            status_card_class = "card-rose"
            status_desc = f"Deficit: ~{unit_gap:.0f} units below typical" if unit_gap > 0 else "Immediate planning attention required"
        else:
            status_text = "Moderate Shortfall"
            status_color = "#818CF8"
            status_card_class = "card-indigo"
            status_desc = f"Deficit: ~{unit_gap:.0f} units below typical" if unit_gap > 0 else "Monitor supply closely"
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

    # 3. Forecast Trajectory Chart
    st.markdown('<div class="section-title">Forecast Trajectory</div>', unsafe_allow_html=True)

    time_view = st.radio(
        "Aggregation",
        ["Daily View", "Weekly Cumulative"],
        horizontal=True,
        label_visibility="collapsed",
        key="fcst_time_view",
    )

    act_df = act_sub.copy()
    act_df["date"] = pd.to_datetime(act_df["date"])

    pred_df = fcst_sub.copy()
    pred_df["date"] = pd.to_datetime(pred_df["target_date"])

    has_subsequent_actuals = selected_origin != DATA_CUTOFF and not pred_df["target"].isna().all()
    subsequent_df = None
    if has_subsequent_actuals:
        subsequent_df = pd.DataFrame(
            {
                "date": pred_df["date"],
                "donations": pred_df["target"],
            }
        ).dropna()

    fcst_dates = pred_df["date"]

    if time_view == "Weekly Cumulative":
        act_df["week_end"] = act_df["date"].dt.to_period("W-SUN").dt.end_time.dt.date
        act_weekly = (
            act_df.groupby("week_end")["donations"]
            .sum()
            .reset_index()
            .rename(columns={"week_end": "date"})
        )
        act_weekly["date"] = pd.to_datetime(act_weekly["date"])

        w1_p50 = pred_df[pred_df["horizon"].isin(range(1, 8))]["pred_p50"].sum()
        w1_p10 = pred_df[pred_df["horizon"].isin(range(1, 8))]["pred_p10"].sum()
        w1_p90 = pred_df[pred_df["horizon"].isin(range(1, 8))]["pred_p90"].sum()
        w1_date = fcst_dates.iloc[6]

        w2_p50 = pred_df[pred_df["horizon"].isin(range(8, 15))]["pred_p50"].sum()
        w2_p10 = pred_df[pred_df["horizon"].isin(range(8, 15))]["pred_p10"].sum()
        w2_p90 = pred_df[pred_df["horizon"].isin(range(8, 15))]["pred_p90"].sum()
        w2_date = fcst_dates.iloc[13]

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

    axis_config = alt.Axis(
        format="%d %b",
        labelAngle=0,
        labelFontSize=11,
        titleFontSize=12,
        labelColor="#94A3B8",
        titleColor="#E2E8F0",
        gridColor="#1E293B",
        gridDash=[3, 3],
        gridOpacity=0.7,
        domainColor="#334155",
    )

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
                    labelColor="#94A3B8",
                    titleColor="#E2E8F0",
                    gridColor="#1E293B",
                    gridDash=[3, 3],
                    gridOpacity=0.7,
                ),
            ),
            y2="pred_p90:Q",
        )
    )

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

    origin_line_df = pd.DataFrame({"date": [pd.to_datetime(selected_origin)]})
    boundary_rule = (
        alt.Chart(origin_line_df)
        .mark_rule(
            color="#475569",
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

    leg1, leg2, leg3, leg4 = st.columns(4)
    with leg1:
        st.markdown(
            '<div style="font-size: 13px; color: #CBD5E1;"><span style="color: #94A3B8; font-weight: bold;">― ●</span> <strong>Actual donations</strong></div>',
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
                '<div style="font-size: 13px; color: #64748B;">┊ <strong>Forecast starts</strong></div>',
                unsafe_allow_html=True,
            )

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

    exp_col, _ = st.columns([3, 5])
    with exp_col:
        csv_forecast = df_table.to_csv(index=False)
        st.download_button(
            label="📥 Export 14-Day Outlook (CSV)",
            data=csv_forecast,
            file_name=f"donorcast_forecast_{selected_facility}_{selected_group}_{selected_origin}.csv",
            mime="text/csv",
            use_container_width=True,
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

    df_alerts = load_alerts(selected_origin)

    if len(df_alerts) == 0:
        st.info(f"No shortfall alerts flagged for forecast origin {selected_origin}.")
        return

    # 2. Top Summary Cards (Unified Dark Styling)
    total_alerts = len(df_alerts)
    high_alerts = int((df_alerts["severity"] == "HIGH").sum())
    med_alerts = int((df_alerts["severity"] == "MEDIUM").sum())
    total_deficit_units = (
        float(df_alerts["unit_deficit"].sum()) if "unit_deficit" in df_alerts.columns else 0.0
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

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

    with kpi4:
        st.markdown(
            f"""
            <div class="op-card card-rose">
                <div class="op-card-label">PROJECTED DEFICIT</div>
                <div class="op-card-value" style="color: #F87171;">−{total_deficit_units:,.0f} <span style="font-size: 15px; font-weight: 500;">units</span></div>
                <div class="op-card-sub">Total 7-day volume gap</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. National Facility Geospatial Risk Map
    st.markdown('<div class="section-title">National Facility Risk Map</div>', unsafe_allow_html=True)
    render_facility_map(df_alerts)

    # 4. National ABO Group Shortfall Breakdown
    st.markdown('<div class="section-title">National ABO Shortfall Breakdown</div>', unsafe_allow_html=True)
    abo_cols = st.columns(4)
    for i, grp in enumerate(BLOOD_GROUPS):
        grp_df = df_alerts[df_alerts["group"] == grp]
        grp_def = float(grp_df["unit_deficit"].sum()) if "unit_deficit" in grp_df.columns else 0.0
        grp_count = len(grp_df)
        with abo_cols[i]:
            st.markdown(
                f"""
                <div class="op-card card-neutral" style="padding: 12px 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 16px; font-weight: 800; color: #F8FAFC;">Group {grp}</span>
                        <span style="font-size: 12px; color: #94A3B8;">{grp_count} sites flagged</span>
                    </div>
                    <div style="font-size: 18px; font-weight: 700; color: #EF4444; margin-top: 4px;">
                        −{grp_def:,.0f} <span style="font-size: 13px; font-weight: 400; color: #CBD5E1;">units</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 5. Clean Filter Controls & Export Action
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

    filtered_df = df_alerts.copy()
    if group_filter != "All":
        filtered_df = filtered_df[filtered_df["group"] == group_filter]
    if priority_filter != "All":
        filtered_df = filtered_df[filtered_df["severity"] == priority_filter]

    severity_order = {"HIGH": 0, "MEDIUM": 1, "NONE": 2}
    filtered_df["_sev_sort"] = filtered_df["severity"].map(severity_order)
    filtered_df = filtered_df.sort_values(
        by=["_sev_sort", "deficit_pct", "typical_7d"], ascending=[True, True, False]
    ).drop(columns=["_sev_sort"])

    with action_col:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if len(filtered_df) > 0:
            export_cols = [
                col
                for col in [
                    "facility",
                    "group",
                    "severity",
                    "forecast_7d",
                    "typical_7d",
                    "unit_deficit",
                    "deficit_pct",
                    "reason_1",
                    "reason_2",
                    "reason_3",
                    "suggested_action",
                ]
                if col in filtered_df.columns
            ]
            csv_export = filtered_df[export_cols].to_csv(index=False)
            st.download_button(
                label="📥 Export Alerts Briefing (CSV)",
                data=csv_export,
                file_name=f"donorcast_shortfall_alerts_{selected_origin}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # 6. Clean Operational Alert Table
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
                "Deficit (Units)": filtered_df["unit_deficit"].map(lambda x: f"~{x:.0f} units short") if "unit_deficit" in filtered_df.columns else "N/A",
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
            "Deficit (Units)": st.column_config.TextColumn("Deficit (Units)", width="medium"),
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
            <div style="font-size: 12px; color: #94A3B8; margin-top: 4px; margin-bottom: 20px;">
                Select any row above to inspect associated statistical factors and drill down into the forecast.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 7. Alert Details & Drill-Down Card
        st.markdown('<div class="section-title">Alert Details & Prescriptions</div>', unsafe_allow_html=True)

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
        target_gap = float(target_row.get("unit_deficit", 0.0))
        target_fcst = target_row["forecast_7d"]
        target_typ = target_row["typical_7d"]
        r1 = target_row["reason_1"]
        r2 = target_row["reason_2"]
        r3 = target_row["reason_3"]
        suggested_action = target_row.get(
            "suggested_action",
            f"Schedule an extra mobile drive and message regular group {target_grp} donors this week.",
        )

        badge_class = "pill-high" if target_sev == "HIGH" else "pill-med"
        border_class = "card-rose" if target_sev == "HIGH" else "card-indigo"
        accent_color = "#F87171" if target_sev == "HIGH" else "#818CF8"

        st.markdown(
            f"""
            <div class="op-card {border_class}" style="margin-top: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-size: 18px; font-weight: 700; color: #F8FAFC;">
                        {target_fac} &nbsp;·&nbsp; Blood Group <span style="color: {accent_color};">{target_grp}</span>
                    </div>
                    <div>
                        <span class="status-pill {badge_class}">{target_sev} PRIORITY</span>
                    </div>
                </div>
                <div style="font-size: 14px; color: #CBD5E1; line-height: 1.6; margin-bottom: 14px;">
                    Expected over next 7 days: <strong>{target_fcst:.0f} donations</strong><br/>
                    Typical over next 7 days: <strong>{target_typ:.0f} donations</strong><br/>
                    Projected Shortfall: <strong style="color: {accent_color};">{target_gap:.0f} units short ({target_def:.0%} below typical)</strong>
                </div>
                <div style="background-color: #0F172A; border-radius: 6px; padding: 14px 18px; border: 1px solid #1E293B; font-size: 13px; color: #E2E8F0; margin-bottom: 12px;">
                    <div style="font-weight: 700; color: #F1F5F9; margin-bottom: 8px;">Key Associated Factors</div>
                    <div style="margin-bottom: 4px; color: #CBD5E1;">• {r1}</div>
                    <div style="margin-bottom: 4px; color: #CBD5E1;">• {r2}</div>
                    <div style="margin-bottom: 4px; color: #CBD5E1;">• {r3}</div>
                    <div style="margin-top: 8px; font-size: 11px; color: #94A3B8; font-style: italic;">
                        Primary statistical contributors driving model expectation.
                    </div>
                </div>
                <div style="background-color: #0C281E; border-radius: 6px; padding: 14px 18px; border: 1px solid #10B98140; font-size: 13px; color: #E2E8F0;">
                    <div style="font-weight: 700; color: #34D399; margin-bottom: 6px;">📋 Recommended Operational Action</div>
                    <div style="color: #F8FAFC; line-height: 1.5;">{suggested_action}</div>
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

    # Sidebar: Brand Logo and Navigation
    with st.sidebar:
        # SVG Brand Logo and Wordmark
        st.markdown(
            """
            <div style="margin-bottom: 1.6rem; padding-bottom: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%); padding: 8px; border-radius: 10px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(37, 99, 235, 0.35);">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z" fill="#FFFFFF"/>
                            <path d="M12 10.5v6M9 13.5h6" stroke="#2563EB" stroke-width="2.2" stroke-linecap="round"/>
                        </svg>
                    </div>
                    <div>
                        <div style="font-size: 20px; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em; line-height: 1.1;">
                            DONOR<span style="color: #60A5FA;">CAST</span>
                        </div>
                        <div style="color: #94A3B8; font-size: 12px; font-weight: 500; margin-top: 2px;">
                            Operations Forecasting
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nav-section-title">NAVIGATION</div>', unsafe_allow_html=True)
        nav_options = ["Forecast", "Shortfall Alerts", "Overview"]
        if "nav_radio" not in st.session_state:
            st.session_state["nav_radio"] = "Forecast"

        selected_page = st.radio(
            "Navigation",
            nav_options,
            key="nav_radio",
            format_func=lambda x: NAV_LABELS.get(x, x),
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

        # Sidebar Live System Capsule
        st.markdown(
            """
            <div style="background-color: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 12px 14px; margin-top: 14px; font-size: 11.5px; color: #94A3B8; line-height: 1.6;">
                <div style="font-weight: 700; color: #E2E8F0; margin-bottom: 4px; font-size: 12px; display: flex; align-items: center; gap: 6px;">
                    <span style="color: #10B981;">●</span> SYSTEM STATUS
                </div>
                <div>Coverage: <strong>22 Facilities (88 Series)</strong></div>
                <div>Horizon: <strong>14 Days Forward</strong></div>
                <div>Granularity: <strong>Daily by ABO Group</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clean single-instance operational disclaimer in sidebar footer
        st.markdown(
            """
            <div class="sidebar-footer">
                Decision support system for operational blood drive planning.<br/>
                Ministry of Health Malaysia Open Data.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main page content routing
    active_origin = st.session_state.get("selected_origin", DATA_CUTOFF)
    if st.session_state["nav_page"] == "Overview":
        render_landing_page(selected_origin=active_origin)
    elif st.session_state["nav_page"] == "Forecast":
        render_forecast_page(selected_origin=active_origin)
    elif st.session_state["nav_page"] == "Shortfall Alerts":
        render_alerts_page(selected_origin=active_origin)


if __name__ == "__main__":
    main()
