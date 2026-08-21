"""
Complete Streets Active-Mode Crash Dashboard
Just & Green Transportation Lab | University of Florida

Run with:
    streamlit run dashboard.py

Expects `power_bi_export.csv` (as produced by the combined
eda_analysis_combined.py pipeline) in the same folder as this script,
or upload it via the sidebar.
"""

import glob
import os
import re
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# geopandas/shapely are optional -- only needed for the census-tract spatial
# join maps in the When & Where tab. Everything else in the dashboard works
# fine without them.
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# ============================================================================
# PAGE CONFIG + THEME
# ============================================================================
st.set_page_config(
    page_title="Active-Mode Crash Dashboard",
    page_icon="\U0001F6B2",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODE_COLORS = {"Bicycle": "#2196F3", "E-Bike": "#4CAF50", "E-Scooter": "#FF9800"}
SEVERITY_ORDER = ["No Injury", "Injury", "Serious Injury", "Fatality"]
SEVERITY_COLORS = {
    "No Injury": "#A5D6A7", "Injury": "#FFF176",
    "Serious Injury": "#FFB74D", "Fatality": "#B71C1C",
}
MODES = ["Bicycle", "E-Bike", "E-Scooter"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GENDER_COLORS = {"Male": "#5C6BC0", "Female": "#EC407A", "Unknown": "#B0BEC5"}
AGE_BAND_ORDER = ["0-14", "15-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Unknown"]

# Candidate column names we'll search for -- see find_col() below.
# Confirmed from eda_analysis_combined.py Section 20/20b/20c real headers:
#   power_bi_export.csv:  REPORT_NUMBER, MODE, YEAR, HOUR, DOW, MONTH,
#     COUNTY_NAME, LIGHT_CONDITION, WEATHER_CONDITION, S4_CRASH_SEVERITY,
#     LOC_TYPE, DAY_NIGHT, mv_involved, S4_IS_*, CITED, AVG_AADT,
#     MEDIAN_TYPE, SHOULDER_WIDTH, NUM_THRU_LANES, CONTEXT_CLASS,
#     INTERSECTION_CONTROL, FARS_LANDUSE, CRASH_TYPE, ROAD_TYPE, POSTED_SPEED,
#     MICROMOBILITY_SPEED_MPH
#   power_bi_export_demographics.csv: REPORT_NUMBER, MODE, SEX, AGE, YEAR,
#     COUNTY_NAME, S4_CRASH_SEVERITY, DAY_NIGHT, LOC_TYPE
#   dashboard_meta.csv: metric, value, note
AGE_COL_CANDIDATES = ["AGE", "S4_AGE_AT_TIME_OF_CRASH", "S4_AGE", "NM_AGE", "PERSON_AGE"]
GENDER_COL_CANDIDATES = ["SEX", "GENDER", "S4_SEX", "NM_SEX", "PERSON_SEX"]
CRASH_ID_CANDIDATES = ["REPORT_NUMBER", "CRASH_ID", "S4_CRASH_ID", "OFFICIAL_CRASH_ID", "CASE_NUMBER"]
CRASH_TYPE_CANDIDATES = ["CRASH_TYPE", "S4_CRASH_TYPE", "Crash_Type"]
ROAD_TYPE_CANDIDATES = ["ROAD_TYPE", "S4_ROAD_TYPE", "Road_Type", "ROADWAY_TYPE"]
SPEED_COL_CANDIDATES = ["POSTED_SPEED", "Posted_Speed", "SPEED_LIMIT"]
MICRO_SPEED_COL_CANDIDATES = ["MICROMOBILITY_SPEED_MPH", "Micromobility_Speed_Mph"]
# S4_LATITUDE/S4_LONGITUDE preferred -- eda_analysis_combined.py notes these
# are ~98.5% complete vs. the raw LATITUDE/LONGITUDE columns.
LAT_COL_CANDIDATES = ["S4_LATITUDE", "LATITUDE"]
LON_COL_CANDIDATES = ["S4_LONGITUDE", "LONGITUDE"]
INFRA_COL_LABELS = {
    "MEDIAN_TYPE": "Median Type",
    "SHOULDER_WIDTH": "Shoulder Width",
    "NUM_THRU_LANES": "Number of Through Lanes",
    "CONTEXT_CLASS": "Context Class",
}
META_METRIC_LABELS = {
    "query_params_report_numbers": "Signal4Data report numbers originally queried",
    "crash_event_total_crashes": "Total crashes in combined crash_event (S4_Crash_bicycle + Signal4Data)",
    "bicycle_population_s4_crash_bicycle": "Bicycle crashes in source population (S4_Crash_bicycle)",
    "active_mode_crashes": "Active-mode crashes (this dashboard's scope)",
    "bicycle_crashes": "Bicycle crashes",
    "ebike_crashes": "E-Bike crashes",
    "escooter_crashes": "E-Scooter crashes",
    "narrative_labeled_crashes": "Signal4Data crashes with Qwen narrative label",
}
META_FUNNEL_ORDER = [
    "crash_event_total_crashes", "active_mode_crashes",
]

# ----------------------------------------------------------------------------
# Crash Causation (LLM narrative classification) -- see cause_analysis_export.csv,
# built from multilabel_RegBike_cause.xlsx + multilabel_ebike_cause.xlsx by
# tagging each row's ID as REPORT_NUMBER and its `prediction` field as MODE.
# ----------------------------------------------------------------------------
CAUSE_DRIVER = [
    "driver_failed_to_yield_turning", "driver_ran_stop_sign_or_red_light",
    "rear_end_following_too_close", "distraction_inattention",
    "speeding_reckless_driving", "impairment", "dooring",
]
CAUSE_NON_MOTORIST = [
    "non_motorist_failed_to_yield_entering_roadway",
    "non_motorist_ran_stop_sign_or_signal", "wrong_way_riding",
    "sidewalk_driveway_conflict",
]
CAUSE_AMBIGUOUS = [
    "obstructed_sightline", "low_visibility_no_lights", "hit_and_run",
    "insufficient_information", "mechanical_failure", "other",
]

def cause_attribution(cause):
    if cause in CAUSE_DRIVER:
        return "Driver-attributable"
    if cause in CAUSE_NON_MOTORIST:
        return "Non-motorist-attributable"
    return "Ambiguous / environmental"

CAUSE_LABELS = {c: c.replace("_", " ").capitalize() for c in
                CAUSE_DRIVER + CAUSE_NON_MOTORIST + CAUSE_AMBIGUOUS}
INFRA_TYPE_LABELS = {
    "travel_lane": "Travel lane", "sidewalk": "Sidewalk", "crosswalk": "Crosswalk",
    "bike_lane": "Bike lane", "driveway_or_parking_lot": "Driveway/parking lot",
    "shoulder": "Shoulder", "multi_use_path": "Multi-use path", "unknown": "Unknown",
}
ATTRIBUTION_COLORS = {
    "Driver-attributable": "#5C6BC0",
    "Non-motorist-attributable": "#EF5350",
    "Ambiguous / environmental": "#BDBDBD",
}


def find_col(df, candidates):
    """Return the first matching column name (case-insensitive), or None."""
    if df is None:
        return None
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def age_to_band(age):
    try:
        age = float(age)
    except (TypeError, ValueError):
        return "Unknown"
    if pd.isna(age):
        return "Unknown"
    if age < 15:
        return "0-14"
    if age < 18:
        return "15-17"
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


def normalize_gender(val):
    if pd.isna(val):
        return "Unknown"
    v = str(val).strip().lower()
    if v in ("m", "male", "1"):
        return "Male"
    if v in ("f", "female", "2"):
        return "Female"
    return "Unknown"

PLOT_FONT = dict(family="Segoe UI, -apple-system, sans-serif", size=13, color="#1a1a2e")
PLOT_BG = "#ffffff"

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.main { background-color: #f5f7fa; }

section[data-testid="stSidebar"] {
    background-color: #12172b;
}
section[data-testid="stSidebar"] * { color: #e8eaf2 !important; }
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stRadio label { color: #cfd3e6 !important; }

/* Buttons (Browse files, Download, Reset) sit on light backgrounds even
   inside the dark sidebar -- override the blanket light-text rule above.
   IMPORTANT: also target every nested span/icon inside these controls
   (the "* {…}" rule above still wins for children unless we re-assert it
   here), or the button box goes dark while its label text stays the old
   light color -- that's what caused the light-on-light "Upload" /
   "Download filtered data" text. */
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] button *,
section[data-testid="stSidebar"] .stDownloadButton button,
section[data-testid="stSidebar"] .stDownloadButton button *,
section[data-testid="stSidebar"] .stFileUploader button,
section[data-testid="stSidebar"] .stFileUploader button *,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] * {
    color: #12172b !important;
    fill: #12172b !important;
    background-color: transparent !important;
    font-weight: 600 !important;
}
/* Border goes ONLY on the actual <button> element, never on its
   descendants (the label wrapper div/span/p inside it) -- putting it on
   "button *" as well draws a second nested box hugging the text, which is
   the "double outline" artifact around the All/KSI/Fatal preset buttons. */
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] .stDownloadButton button,
section[data-testid="stSidebar"] .stFileUploader button,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background-color: #ffffff !important;
    border: 1px solid #4a5490 !important;
}
section[data-testid="stSidebar"] button:hover,
section[data-testid="stSidebar"] button:hover *,
section[data-testid="stSidebar"] .stDownloadButton button:hover,
section[data-testid="stSidebar"] .stDownloadButton button:hover *,
section[data-testid="stSidebar"] .stFileUploader button:hover,
section[data-testid="stSidebar"] .stFileUploader button:hover * {
    background-color: #eef1fb !important;
    color: #12172b !important;
    border-color: #2b3f8c !important;
}

/* Streamlit/BaseWeb gives every button a colored focus ring (box-shadow)
   that lingers after a click -- most visible as a stray red/orange outline
   around the file-uploader's "Browse files" button. Kill it everywhere in
   the sidebar; the existing border/background rules above already give
   buttons enough visual definition without it. */
section[data-testid="stSidebar"] button:focus,
section[data-testid="stSidebar"] button:focus-visible,
section[data-testid="stSidebar"] button:focus:not(:active),
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:focus,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:focus:not(:active),
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:focus,
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:focus:not(:active),
section[data-testid="stSidebar"] div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="stButton"] * {
    box-shadow: none !important;
    outline: none !important;
}

/* File-uploader dropzone card (the box with the upload icon, "Upload"
   button, and "200MB per file" helper text) -- force the card itself
   plus every descendant to dark text on its light background. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    border: 1px solid #4a5490 !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #12172b !important;
    fill: #12172b !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #5a6088 !important;
}

/* Primary-styled reset button gets a stronger accent fill (including its
   nested label span, so the fix above doesn't force it back to dark).
   Border again only on the button itself, not its descendants -- see the
   comment above the secondary-button rule for why. */
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] * {
    color: #ffffff !important;
    fill: #ffffff !important;
    background-color: #c0392b !important;
}
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    border: 1px solid #c0392b !important;
}
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover,
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover * {
    background-color: #a5281c !important;
}

/* Multiselect / selectbox control boxes (Crash Type, Road Type, etc.)
   render on a white background even inside the dark sidebar -- force their
   placeholder text ("Choose options"), chosen values, and dropdown arrow
   to a dark, readable color instead of inheriting the near-white sidebar
   default.

   IMPORTANT: this must never reach inside a tag chip (the colored
   Mode/County pills). The previous version applied color/fill/opacity to
   *every* descendant, including the chip's delete-"x" button -- forcing
   opacity:1 on it made a hover-only highlight box permanently visible
   behind the icon, and forcing its fill overrode the icon's own white-on-
   color styling. Net effect: a stray dark/white box instead of a clean X.
   The :not() clause below excludes the whole tag subtree up front so
   Streamlit's own (already-correct) tag styling is left alone entirely,
   rather than trying to patch it back afterward. */
section[data-testid="stSidebar"] [data-baseweb="select"],
section[data-testid="stSidebar"] [data-baseweb="select"] *:not([data-baseweb="tag"], [data-baseweb="tag"] *) {
    color: #33395c !important;
    fill: #33395c !important;
    opacity: 1 !important;
}


.dash-header {
    padding: 1.1rem 1.6rem;
    background: linear-gradient(120deg, #12172b 0%, #1e2a5e 60%, #2b3f8c 100%);
    border-radius: 14px;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 18px rgba(20, 30, 70, 0.18);
}
.dash-header h1 {
    color: #ffffff; font-size: 1.7rem; font-weight: 800; margin: 0;
    letter-spacing: -0.02em;
}
.dash-header p {
    color: #b9c2e8; font-size: 0.92rem; margin: 0.25rem 0 0 0;
}

.kpi-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    box-shadow: 0 2px 10px rgba(20, 30, 70, 0.07);
    border-left: 4px solid #2b3f8c;
    height: 100%;
}
.kpi-label {
    font-size: 0.78rem; font-weight: 600; color: #7a819c;
    text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.25rem;
}
.kpi-value { font-size: 1.65rem; font-weight: 800; color: #12172b; line-height: 1.1; }
.kpi-sub { font-size: 0.78rem; color: #9aa0b8; margin-top: 0.2rem; }

.section-note {
    background: #eef1fb; border-left: 4px solid #7a86d4;
    border-radius: 8px; padding: 0.7rem 1rem; font-size: 0.85rem;
    color: #33395c; margin: 0.6rem 0 1.1rem 0;
}

div[data-testid="stPlotlyChart"] {
    background: #ffffff; border-radius: 12px; padding: 0.6rem;
    box-shadow: 0 2px 10px rgba(20, 30, 70, 0.10);
    border: 1px solid #e6e9f5;
    margin-bottom: 1.1rem;
}
div[data-testid="column"] { padding: 0 0.5rem; }
div[data-testid="stHorizontalBlock"] { gap: 1rem; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background-color: #ffffff; border-radius: 8px 8px 0 0; padding: 10px 18px;
    font-weight: 600; color: #7a819c;
}
.stTabs [aria-selected="true"] { color: #2b3f8c !important; background-color: #eef1fb; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================
DEFAULT_PATH = "power_bi_export.csv"
DEFAULT_DEMO_PATH = "power_bi_export_demographics.csv"
DEFAULT_META_PATH = "dashboard_meta.csv"
DEFAULT_NARRATIVE_PATH = "narrative_text_export.csv"
DEFAULT_HOTSPOT_PATH = "spatiotemporal_hotspots_by_mode.csv"
DEFAULT_CAUSE_PATH = "cause_analysis_export.csv"


def _mtime_key(path_or_buffer):
    """Cache-busting key: on-disk files are keyed by mtime so an edited/
    replaced CSV invalidates the cache even if the process never restarts.
    Uploaded file objects don't need this -- Streamlit already gives each
    upload a distinct identity."""
    if isinstance(path_or_buffer, str) and os.path.exists(path_or_buffer):
        return os.path.getmtime(path_or_buffer)
    return None


@st.cache_data
def load_data(path_or_buffer, _mtime=None):
    df = pd.read_csv(path_or_buffer)
    df["MODE"] = pd.Categorical(df["MODE"], categories=MODES, ordered=True)
    if "S4_CRASH_SEVERITY" in df.columns:
        df["S4_CRASH_SEVERITY"] = pd.Categorical(
            df["S4_CRASH_SEVERITY"], categories=SEVERITY_ORDER, ordered=True
        )
    if "DOW" in df.columns:
        df["DOW"] = pd.Categorical(df["DOW"], categories=DOW_ORDER, ordered=True)
    return df


@st.cache_data
def load_demographics(path_or_buffer, _mtime=None):
    """Person-level demographics (age/gender), one row per active-mode
    person involved in a crash. Schema is flexible -- see find_col()."""
    ddf = pd.read_csv(path_or_buffer)
    age_col = find_col(ddf, AGE_COL_CANDIDATES)
    gender_col = find_col(ddf, GENDER_COL_CANDIDATES)
    if age_col:
        ddf["_AGE"] = pd.to_numeric(ddf[age_col], errors="coerce")
        ddf["_AGE_BAND"] = pd.Categorical(
            ddf["_AGE"].apply(age_to_band), categories=AGE_BAND_ORDER, ordered=True
        )
    if gender_col:
        ddf["_GENDER"] = ddf[gender_col].apply(normalize_gender)
    return ddf


@st.cache_data
def load_meta(path_or_buffer, _mtime=None):
    """Pipeline funnel counts (e.g. raw records -> geocoded -> matched ->
    final export) used on the About tab. Expected as a simple two-column
    CSV: a stage/step label column and a count column, but we degrade
    gracefully to a raw table if the shape is unrecognized."""
    return pd.read_csv(path_or_buffer)


@st.cache_data
def load_narratives(path_or_buffer, _mtime=None):
    """Per-crash narrative text (Signal4Data crashes with a Qwen-classified
    narrative only -- the S4_Crash_bicycle-only population has no narrative
    text). Powers the interactive keyword/text-mining tab."""
    ndf = pd.read_csv(path_or_buffer)
    if "REPORT_NUMBER" in ndf.columns:
        ndf["REPORT_NUMBER"] = ndf["REPORT_NUMBER"].astype(str)
    if "NARRATIVE_TEXT" in ndf.columns:
        ndf["NARRATIVE_TEXT"] = ndf["NARRATIVE_TEXT"].fillna("").astype(str)
    return ndf


@st.cache_data
def load_hotspots(path_or_buffer, _mtime=None):
    """Precomputed DBSCAN spatiotemporal cluster table (one row per
    cluster), from eda_analysis_combined.py section 09d."""
    return pd.read_csv(path_or_buffer)


@st.cache_data
def load_tracts(path_or_buffer, _mtime=None):
    """Census tract boundaries (+ population) for the spatial-join maps in
    the When & Where tab. Returns a GeoDataFrame in EPSG:4326, or None if
    geopandas isn't installed / the file can't be read."""
    if not GEOPANDAS_AVAILABLE:
        return None
    try:
        tracts = gpd.read_file(path_or_buffer)
        if tracts.crs is None:
            tracts = tracts.set_crs(4326)
        else:
            tracts = tracts.to_crs(4326)
        return tracts
    except Exception:
        return None


@st.cache_data
def load_cause_data(path_or_buffer, _mtime=None):
    """LLM narrative-classified crash causation (one row per crash) --
    primary_cause / infrastructure_type / speed_contributing, built from
    multilabel_RegBike_cause.xlsx + multilabel_ebike_cause.xlsx. See
    cause_analysis_export.csv for the combined REPORT_NUMBER/MODE schema."""
    cdf = pd.read_csv(path_or_buffer)
    cdf["REPORT_NUMBER"] = cdf["REPORT_NUMBER"].astype(str)
    if "primary_cause" in cdf.columns:
        cdf["ATTRIBUTION"] = cdf["primary_cause"].apply(cause_attribution)
        cdf["CAUSE_LABEL"] = cdf["primary_cause"].map(CAUSE_LABELS).fillna(cdf["primary_cause"])
    if "infrastructure_type" in cdf.columns:
        cdf["INFRA_LABEL"] = cdf["infrastructure_type"].map(INFRA_TYPE_LABELS).fillna(cdf["infrastructure_type"])
    return cdf


def file_input(label, default_path, key):
    """Sidebar uploader with a friendly fallback: use the file sitting next
    to the script if present, else let the user upload it, else skip
    silently for optional files."""
    up = st.file_uploader(label, type="csv", key=key)
    if up is not None:
        return up
    if os.path.exists(default_path):
        return default_path
    return None


with st.sidebar:
    st.markdown("### \U0001F4C2 Data Source")
    main_src = file_input(f"Upload {DEFAULT_PATH}", DEFAULT_PATH, "main_upload")
    with st.expander("Optional: demographics & pipeline info"):
        demo_src = file_input(f"Upload {DEFAULT_DEMO_PATH}", DEFAULT_DEMO_PATH, "demo_upload")
        meta_src = file_input(f"Upload {DEFAULT_META_PATH}", DEFAULT_META_PATH, "meta_upload")
        narrative_src = file_input(f"Upload {DEFAULT_NARRATIVE_PATH}", DEFAULT_NARRATIVE_PATH, "narrative_upload")
        hotspot_src = file_input(f"Upload {DEFAULT_HOTSPOT_PATH}", DEFAULT_HOTSPOT_PATH, "hotspot_upload")
        cause_src = file_input(f"Upload {DEFAULT_CAUSE_PATH}", DEFAULT_CAUSE_PATH, "cause_upload")
    with st.expander("Optional: census tract boundaries (for tract-level maps)"):
        st.caption(
            "GeoJSON with one row per Florida census tract: a `GEOID` column, "
            "a total-population column, and `geometry`. Build it once with "
            "TIGER/Line tract shapefiles (`tl_2023_12_tract`) joined to ACS "
            "table B01003 (total population) by GEOID, e.g. via `tidycensus` "
            "or the Census API, then `to_crs(4326)` and export as GeoJSON."
        )
        tract_up = st.file_uploader("Upload census_tracts.geojson", type=["geojson", "json"], key="tract_upload")
        tract_src = tract_up if tract_up is not None else (
            "census_tracts.geojson" if os.path.exists("census_tracts.geojson") else None
        )
        tract_pop_col = st.text_input(
            "Population column name in that file", value="POPULATION", key="tract_pop_col"
        )

if main_src is not None:
    df_raw = load_data(main_src, _mtime=_mtime_key(main_src))
else:
    st.info(
        f"\U0001F4C2 **Waiting on crash data.** This dashboard needs "
        f"`{DEFAULT_PATH}` to run -- either place it in the same folder as "
        f"`dashboard.py` before launching `streamlit run dashboard.py`, or "
        f"upload it using the **Data Source** panel in the sidebar. "
        f"This isn't an error, just Streamlit waiting on a file."
    )
    st.stop()

demo_raw = load_demographics(demo_src, _mtime=_mtime_key(demo_src)) if demo_src is not None else None
meta_raw = load_meta(meta_src, _mtime=_mtime_key(meta_src)) if meta_src is not None else None
narrative_raw = load_narratives(narrative_src, _mtime=_mtime_key(narrative_src)) if narrative_src is not None else None
hotspot_raw = load_hotspots(hotspot_src, _mtime=_mtime_key(hotspot_src)) if hotspot_src is not None else None
cause_raw = load_cause_data(cause_src, _mtime=_mtime_key(cause_src)) if cause_src is not None else None
tracts_raw = load_tracts(tract_src, _mtime=_mtime_key(tract_src)) if tract_src is not None else None

MAIN_CRASH_ID_COL = find_col(df_raw, CRASH_ID_CANDIDATES)
DEMO_CRASH_ID_COL = find_col(demo_raw, CRASH_ID_CANDIDATES) if demo_raw is not None else None
DEMO_AGE_AVAILABLE = demo_raw is not None and "_AGE" in demo_raw.columns
DEMO_GENDER_AVAILABLE = demo_raw is not None and "_GENDER" in demo_raw.columns
CRASH_TYPE_COL = find_col(df_raw, CRASH_TYPE_CANDIDATES)
ROAD_TYPE_COL = find_col(df_raw, ROAD_TYPE_CANDIDATES)
SPEED_COL = find_col(df_raw, SPEED_COL_CANDIDATES)
MICRO_SPEED_COL = find_col(df_raw, MICRO_SPEED_COL_CANDIDATES)
LAT_COL = find_col(df_raw, LAT_COL_CANDIDATES)
LON_COL = find_col(df_raw, LON_COL_CANDIDATES)

DRIVER_FLAGS = [c for c in df_raw.columns if c.startswith("S4_IS_")]
DRIVER_FLAG_LABELS = {
    "S4_IS_AGGRESSIVE_DRIVING": "Aggressive Driving",
    "S4_IS_ALCOHOL_RELATED": "Alcohol Related",
    "S4_IS_DRUG_RELATED": "Drug Related",
    "S4_IS_DISTRACTED": "Distracted Driving",
    "S4_IS_SPEEDING_RELATED": "Speeding Related",
    "S4_IS_AGING_DRIVER": "Aging Driver (65+)",
    "S4_IS_TEENAGER_DRIVER": "Teenage Driver",
    "S4_IS_UNRESTRAINED": "Unrestrained Occupant",
}

# ============================================================================
# PIPELINE FIGURES -- auto-discovered, no manual path/upload step.
# eda_analysis_combined.py writes PNGs to results/figures/<NN_section>/*.png.
# We look for that folder in the handful of places it's likely to sit
# relative to this script and, if found, fold the matching subfolder's
# figures directly into the relevant tab below (see PIPELINE_FIGURE_MAP).
# If the folder isn't present (e.g. running this dashboard somewhere the
# results/ output wasn't copied), we just skip it quietly -- no error, no
# text box asking the user to point at it.
#
# NOTE: this is NOT purely a legacy fallback -- some pipeline figures (e.g.
# 06b/06c/06d in 06_crash_typing) are built from columns that never get
# merged into power_bi_export.csv, so they have no interactive equivalent
# in this dashboard and can only be shown as the static PNG the pipeline
# already produced. Don't remove this block on the assumption "everything
# already has an interactive chart" without checking column-by-column first.
# ============================================================================
def _find_figures_dir():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    for cand in (
        "results/figures", "figures",
        os.path.join(here, "results", "figures"), os.path.join(here, "figures"),
    ):
        if os.path.isdir(cand):
            return cand
    return None


def _label_from_filename(fname):
    stem = os.path.splitext(fname)[0]
    stem = re.sub(r"^\d+[a-z]?_", "", stem)
    return stem.replace("_", " ").title()


@st.cache_data
def _load_pipeline_figures(fig_dir):
    """subfolder name -> sorted list of PNG paths.

    Filenames in PIPELINE_FIGURES_WITH_INTERACTIVE_EQUIVALENT are skipped
    here -- those specific PNGs now have a live Plotly chart built from
    power_bi_export.csv elsewhere in this file, so showing the static PNG
    too would just be a duplicate. Everything else in these folders still
    has no interactive equivalent and is shown as-is.
    """
    if not fig_dir:
        return {}
    by_folder = {}
    for p in sorted(glob.glob(os.path.join(fig_dir, "**", "*.png"), recursive=True)):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in PIPELINE_FIGURES_WITH_INTERACTIVE_EQUIVALENT:
            continue
        by_folder.setdefault(os.path.basename(os.path.dirname(p)), []).append(p)
    return by_folder


# Filenames (no extension) that now have a matching interactive chart built
# from power_bi_export.csv, so the static PNG fallback should skip them.
PIPELINE_FIGURES_WITH_INTERACTIVE_EQUIVALENT = {
    "06a_crash_group_distribution",   # Crash Typing chart (CRASH_GROUP), tab7
    "06b_crash_type_descriptions",    # Crash Type Description chart (CRASH_TYPE_DESC), tab7
    "06d_contributing_factors",       # Contributing Factors chart (ROAD_/ENVIRONMENT_CIRCUMSTANCE), tab7
    "10b_distraction_type_by_mode",   # Driver Distraction Type chart (DISTRACTION_TYPE), tab4
}

PIPELINE_FIGURES = _load_pipeline_figures(_find_figures_dir())

# Which results/figures/<subfolder> feeds which tab -- every subfolder
# eda_analysis_combined.py writes (01 through 13) is mapped somewhere so
# nothing in the results folder gets silently dropped. 12_pedestrian_context
# is explicitly "reference only" in the pipeline (pedestrian crashes stay
# classified as "Other", outside this dashboard's active-mode scope), so it
# lives on the About tab alongside the classification-methodology figures
# rather than implying it's part of the Bicycle/E-Bike/E-Scooter totals.
PIPELINE_FIGURE_MAP = {
    "tab0": ["12_pedestrian_context"],
    "tab1": ["01_overview"],
    "tab2": ["05_severity"],
    "tab3": ["03_when", "04_where", "09_latlon"],
    "tab4": ["10_driver_behavior", "11_violations"],
    "tab5": ["13_roadway_infrastructure"],
    "tab6": ["02_who"],
    "tab7": ["06_crash_typing", "07_text_mining", "08_qwen"],
}


def render_pipeline_figures(tab_key):
    """Render an expander of the pipeline's own static figures for whatever
    subfolder(s) map to this tab -- only appears if matching PNGs were
    actually found on disk."""
    folders = PIPELINE_FIGURE_MAP.get(tab_key, [])
    imgs = [p for f in folders for p in PIPELINE_FIGURES.get(f, [])]
    if not imgs:
        return
    with st.expander(f"\U0001F5BC\uFE0F Pipeline figures from `eda_analysis_combined.py` ({len(imgs)})", expanded=False):
        for i in range(0, len(imgs), 3):
            row_cols = st.columns(3)
            for p, slot in zip(imgs[i:i + 3], row_cols):
                with slot:
                    st.image(p, caption=_label_from_filename(os.path.basename(p)), use_container_width=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown(
    """
    <div class="dash-header">
        <h1>\U0001F6B2 Active-Mode Crash Dashboard</h1>
        <p>Bicycle &bull; E-Bike &bull; E-Scooter crashes in Florida &mdash;
        Just &amp; Green Transportation Lab, University of Florida</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================
# RS (reset sequence) is suffixed onto every filter widget's key. Reset All
# Filters bumps this counter instead of deleting session_state entries --
# deleting a key can still leave stale-looking widgets on the frontend for
# some multiselect/slider components, whereas a new suffix forces Streamlit
# to mount genuinely new widgets that can only start from their defaults.
if "reset_seq" not in st.session_state:
    st.session_state.reset_seq = 0
RS = st.session_state.reset_seq

with st.sidebar:
    st.markdown("### \U0001F39B Filters")

    year_min, year_max = int(df_raw["YEAR"].min()), int(df_raw["YEAR"].max())
    year_range = st.slider("Year Range", year_min, year_max, (year_min, year_max), key=f"filter_year_{RS}")

    hour_range = st.slider("Hour of Day", 0, 23, (0, 23), key=f"filter_hour_{RS}")

    sel_modes = st.multiselect("Mode", MODES, default=MODES, key=f"filter_modes_{RS}")

    county_options = sorted(df_raw["COUNTY_NAME"].dropna().unique().tolist())
    sel_counties = st.multiselect("County (leave empty = all)", county_options, default=[], key=f"filter_counties_{RS}")

    sel_daynight = st.radio("Day / Night", ["All", "Day", "Night"], horizontal=True, key=f"filter_daynight_{RS}")
    sel_loctype = st.radio("Location Type", ["All", "Intersection", "Segment"], horizontal=True, key=f"filter_loctype_{RS}")

    sev_options = [s for s in SEVERITY_ORDER if s in df_raw["S4_CRASH_SEVERITY"].unique()]
    st.caption(
        "Quick severity tiers (per the analysis brief's a/b/c pattern -- "
        "all crashes / KSI / fatal-only). Selecting a preset overwrites the "
        "checkboxes below; you can still fine-tune them afterward."
    )
    ksi_labels = [s for s in ["Fatality", "Serious Injury"] if s in sev_options]
    fatal_labels = [s for s in ["Fatality"] if s in sev_options]
    preset_cols = st.columns(3)
    if preset_cols[0].button("All", use_container_width=True, key=f"sev_preset_all_{RS}"):
        st.session_state[f"filter_severity_{RS}"] = sev_options
        st.rerun()
    if preset_cols[1].button("KSI", use_container_width=True, key=f"sev_preset_ksi_{RS}"):
        st.session_state[f"filter_severity_{RS}"] = ksi_labels
        st.rerun()
    if preset_cols[2].button("Fatal", use_container_width=True, key=f"sev_preset_fatal_{RS}"):
        st.session_state[f"filter_severity_{RS}"] = fatal_labels
        st.rerun()
    sel_severity = st.multiselect("Injury Severity", sev_options, default=sev_options, key=f"filter_severity_{RS}")

    sel_crash_types = []
    if CRASH_TYPE_COL:
        ct_options = sorted(df_raw[CRASH_TYPE_COL].dropna().unique().tolist())
        sel_crash_types = st.multiselect("Crash Type (leave empty = all)", ct_options, default=[], key=f"filter_crash_type_{RS}")

    sel_road_types = []
    if ROAD_TYPE_COL:
        rt_options = sorted(df_raw[ROAD_TYPE_COL].dropna().unique().tolist())
        sel_road_types = st.multiselect("Road Type (leave empty = all)", rt_options, default=[], key=f"filter_road_type_{RS}")

    speed_range = None
    if SPEED_COL:
        spd_series = pd.to_numeric(df_raw[SPEED_COL], errors="coerce").dropna()
        if len(spd_series):
            spd_min, spd_max = int(spd_series.min()), int(spd_series.max())
            if spd_min < spd_max:
                speed_range = st.slider("Posted Speed Limit (mph)", spd_min, spd_max, (spd_min, spd_max), key=f"filter_speed_{RS}")

    age_range = None
    age_default = None
    sel_genders = []
    gender_options = []
    if DEMO_AGE_AVAILABLE or DEMO_GENDER_AVAILABLE:
        st.markdown("### \U0001F464 Demographics")
        st.caption(
            "Only affects crash totals if you narrow these below their "
            "full default range -- otherwise all crashes stay included, "
            "even ones with no matched person-level record."
        )
        if DEMO_AGE_AVAILABLE:
            age_series = demo_raw["_AGE"].dropna()
            if len(age_series):
                a_min, a_max = int(age_series.min()), int(age_series.max())
                age_default = (a_min, a_max)
                if a_min < a_max:
                    age_range = st.slider("Age", a_min, a_max, age_default, key=f"filter_age_{RS}")
        if DEMO_GENDER_AVAILABLE:
            gender_options = [g for g in ["Male", "Female", "Unknown"] if g in demo_raw["_GENDER"].unique()]
            sel_genders = st.multiselect("Gender", gender_options, default=gender_options, key=f"filter_gender_{RS}")
    elif demo_src is None:
        st.caption(
            "Age/gender filters need `power_bi_export_demographics.csv` "
            "(person-level export) -- add it under the Data Source panel above."
        )

    st.markdown("---")
    reset_clicked = st.button("\U0001F504 Reset All Filters", use_container_width=True, type="primary")
    st.caption(f"Loaded **{len(df_raw):,}** total crash records.")

if reset_clicked:
    # Bump the generation counter so every filter_* widget above re-mounts
    # under a fresh key next run -- also sweep any old-generation filter_*
    # entries out of session_state so they don't just sit there unused.
    st.session_state.reset_seq += 1
    for k in list(st.session_state.keys()):
        if k.startswith("filter_"):
            del st.session_state[k]
    st.rerun()

# ---- apply filters ----
df = df_raw[
    df_raw["YEAR"].between(year_range[0], year_range[1])
    & df_raw["HOUR"].between(hour_range[0], hour_range[1])
    & df_raw["MODE"].isin(sel_modes)
    & df_raw["S4_CRASH_SEVERITY"].isin(sel_severity)
].copy()

if sel_counties:
    df = df[df["COUNTY_NAME"].isin(sel_counties)]
if sel_daynight != "All":
    df = df[df["DAY_NIGHT"] == sel_daynight]
if sel_loctype != "All":
    df = df[df["LOC_TYPE"] == sel_loctype]
if sel_crash_types:
    df = df[df[CRASH_TYPE_COL].isin(sel_crash_types)]
if sel_road_types:
    df = df[df[ROAD_TYPE_COL].isin(sel_road_types)]
if speed_range is not None:
    spd = pd.to_numeric(df[SPEED_COL], errors="coerce")
    df = df[spd.between(speed_range[0], speed_range[1]) | spd.isna()]

# ---- demographics filter (joins back to crash-level via crash id) ----
# IMPORTANT: only crashes that have a *matching* person-level demographics
# row can be affected by age/gender -- a crash isn't dropped from the main
# table just because it lacks a demographics record. The main df is only
# narrowed when the user actively moves the age slider or deselects a
# gender away from its "everything selected" default.
age_is_active_filter = age_range is not None and age_default is not None and age_range != age_default
gender_is_active_filter = bool(gender_options) and set(sel_genders) != set(gender_options)

demo = None
if demo_raw is not None:
    demo = demo_raw.copy()
    if age_range is not None:
        demo = demo[demo["_AGE"].between(age_range[0], age_range[1]) | demo["_AGE"].isna()]
    if sel_genders:
        demo = demo[demo["_GENDER"].isin(sel_genders)]
    if (age_is_active_filter or gender_is_active_filter) and MAIN_CRASH_ID_COL and DEMO_CRASH_ID_COL:
        matching_ids = set(demo[DEMO_CRASH_ID_COL].dropna().unique())
        df = df[df[MAIN_CRASH_ID_COL].isin(matching_ids)]
    # keep demo aligned to whatever crash-level filtering happened too
    if MAIN_CRASH_ID_COL and DEMO_CRASH_ID_COL:
        demo = demo[demo[DEMO_CRASH_ID_COL].isin(set(df[MAIN_CRASH_ID_COL].unique()))]

if df.empty:
    st.warning("No crashes match the current filter combination. Try widening a filter.")
    st.stop()

with st.sidebar:
    st.download_button(
        "\u2B07 Download filtered data (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_crashes.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================================
# KPI ROW -- mode-focused (Bicycle / E-Bike / E-Scooter each get their own
# card) rather than one generic aggregate row, since comparing modes is the
# point of this dashboard.
# ============================================================================
total = len(df)
fatalities = int((df["S4_CRASH_SEVERITY"] == "Fatality").sum())
fatal_rate = fatalities / total * 100 if total else 0
serious_rate = (df["S4_CRASH_SEVERITY"] == "Serious Injury").mean() * 100 if total else 0
mv_rate = df["mv_involved"].mean() * 100 if total else 0
cite_rate = df["CITED"].mean() * 100 if total else 0

mode_cols = st.columns(3)
for col, m in zip(mode_cols, MODES):
    sub = df[df["MODE"] == m]
    n = len(sub)
    m_fatal = (sub["S4_CRASH_SEVERITY"] == "Fatality").mean() * 100 if n else 0
    m_serious = (sub["S4_CRASH_SEVERITY"] == "Serious Injury").mean() * 100 if n else 0
    share = n / total * 100 if total else 0
    col.markdown(
        f"""<div class="kpi-card" style="border-left-color:{MODE_COLORS[m]};">
                <div class="kpi-label">{m}</div>
                <div class="kpi-value">{n:,}</div>
                <div class="kpi-sub">{share:.0f}% of filtered crashes &middot;
                {m_fatal:.1f}% fatal &middot; {m_serious:.1f}% serious injury</div>
            </div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    f"""<div style="display:flex; gap:1.5rem; flex-wrap:wrap; padding:0.5rem 0.2rem 0.2rem 0.2rem;
    font-size:0.82rem; color:#5a6088;">
        <span><b>{total:,}</b> total filtered crashes ({year_range[0]}\u2013{year_range[1]})</span>
        <span><b>{fatalities:,}</b> fatalities ({fatal_rate:.1f}%)</span>
        <span><b>{mv_rate:.1f}%</b> involved a motor vehicle</span>
        <span><b>{cite_rate:.1f}%</b> resulted in a citation</span>
    </div>""",
    unsafe_allow_html=True,
)

st.write("")

# ============================================================================
# HELPERS
# ============================================================================
def fmt_n(n):
    """Consistent 'n = 1,234' formatting for figure subtitles/captions."""
    try:
        return f"n = {int(n):,}"
    except (TypeError, ValueError):
        return f"n = {n}"


def style_fig(fig, height=380, title=None, n=None):
    """n: optional sample size (int, or dict of {group_label: count}) shown
    as a light subtitle under the chart title, e.g. 'n = 1,234' or
    'Bicycle n=812 | E-Bike n=201 | E-Scooter n=190' -- so a reader never has
    to guess how many records a %/rate chart is built on."""
    full_title = title
    if title and n is not None:
        if isinstance(n, dict):
            n_txt = " &nbsp;|&nbsp; ".join(f"{k} n={v:,}" for k, v in n.items())
        else:
            n_txt = fmt_n(n)
        full_title = (
            f"{title}<br><span style='font-size:11px;font-weight:400;color:#6b7280'>{n_txt}</span>"
        )
    fig.update_layout(
        font=PLOT_FONT,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        height=height + (16 if (title and n is not None) else 0),
        margin=dict(l=10, r=10, t=(70 if (title and n is not None) else 56) if title else 26, b=10),
        # Centered (not right-anchored) so a full "Bicycle / E-Bike /
        # E-Scooter" legend has room on both sides instead of overflowing
        # past the chart's right edge and getting clipped mid-word. Legend
        # title dropped too ("MODE" / "S4_CRASH_SEVERITY" etc. just ate
        # width without adding information the chart doesn't already show).
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            title=dict(text=""), font=dict(size=11),
        ),
        title=dict(text=full_title, font=dict(size=15, family=PLOT_FONT["family"], color="#12172b")) if full_title else None,
    )
    return fig


def add_count_labels(fig, counts, pct_values=None, fmt="{:,}"):
    """Overlay raw-count text onto a bar trace that's otherwise plotted as a
    percent/rate, so the reader sees both at once. `counts` must be aligned
    (same order) with the trace's x/y categories. If pct_values is given,
    labels read '37.2% (n=118)'; otherwise just the count."""
    counts = list(counts)
    if pct_values is not None:
        labels = [f"{p:.1f}% (n={c:,.0f})" for p, c in zip(pct_values, counts)]
    else:
        labels = [fmt.format(c) for c in counts]
    fig.update_traces(text=labels, textposition="outside")
    return fig


# ============================================================================
# TABS
# ============================================================================
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "\u2139\uFE0F About This Dashboard",
    "\U0001F4C8 Overview & Trends",
    "\U0001F6A8 Severity & Outcomes",
    "\U0001F55B When & Where",
    "\U0001F464 Driver Behavior & Citations",
    "\U0001F6E3 Roadway Infrastructure",
    "\U0001F9D1 Demographics",
    "\U0001F4DD Narrative, Typing & Hotspots",
    "\U0001F50D Crash Causation",
    "\U0001F4CC Insights",
])

# ---------------------------------------------------------------------------
# TAB 0 -- ABOUT / DATA PIPELINE
# ---------------------------------------------------------------------------
with tab0:
    st.markdown("## What this dashboard shows")
    st.markdown(
        """
This dashboard tracks **crashes involving people on bicycles, e-bikes, and
e-scooters** ("active-mode" or "micromobility" road users) in Florida,
built from **the S4_Crash_bicycle population** (Bicycle) plus **Signal4
crash reports** (E-Bike, E-Scooter, Other) matched against **FDOT roadway
tables**. It's produced by the Just & Green Transportation Lab at the
University of Florida as part of a Complete Streets road-safety analysis.

**Scope:** every chart here is scoped to the three active modes -- Bicycle,
E-Bike, E-Scooter. Everything else (pedestrian-only, motor-vehicle-only,
single-vehicle, animal, etc. -- labeled "Other" in the source pipeline) is
excluded from the dashboard entirely, not just hidden by a default filter.

**Why e-bikes and e-scooters get special attention:** these are the two
fastest-growing and least-studied categories of active-mode travel.
Traditional crash datasets and roadway-safety literature were built around
conventional bicycles and pedestrians; e-bikes and e-scooters ride at
higher speeds, on different infrastructure (sidewalks, bike lanes,
shared-use paths), and are involved in crash patterns that don't map
cleanly onto older bicycle-safety research. Understanding how their crash
profile differs from conventional bicycles is central to this project.
        """
    )

    st.markdown("## How each mode is classified")
    st.markdown(
        """
- **Bicycle** -- sourced from **S4_Crash_bicycle**, a larger, dedicated
  bicycle-crash population (bigger N than relying on Signal4Data alone).
  Every row in that source is already a bicycle crash, so no structural
  crash-form code is needed to identify it. The only exception: if Qwen's
  narrative classifier says a specific crash is actually **E-Bike** or
  **E-Scooter**, that overrides the Bicycle label (see below).
- **E-Bike / E-Scooter** -- identified **exclusively** through Qwen LLM
  narrative classification of Signal4Data crash narratives. Florida crash
  forms have no structural code for either, so if the narrative wasn't
  classified, the crash can't be identified as E-Bike/E-Scooter. This Qwen
  narrative call is authoritative even when a crash also shows up in the
  S4_Crash_bicycle population.
        """
    )

    st.markdown("## Why the numbers don't match the raw crash count")
    st.markdown(
        """
This dashboard's active-mode export is built from **two source systems**,
not one flat pull, so a raw report-number count from either source alone
won't match what you see here:
        """
    )
    st.markdown(
        """
- **S4_Crash_bicycle** -- a dedicated, larger bicycle-crash population.
  Every record here is a bicycle crash; the only thing that can move a
  record out of Bicycle is a Qwen narrative override to E-Bike/E-Scooter.
- **Signal4Data** -- a separate crash-report pull covering *all* crash
  types (bicycle, pedestrian, single-vehicle, MV-only, animal, etc). It
  supplies E-Bike, E-Scooter, and everything classified **Other**.
- **Overlap** -- the same physical crash can legitimately appear in both
  source systems. When a REPORT_NUMBER shows up in both, it's counted
  **once**: Qwen's E-Bike/E-Scooter call wins if present, otherwise it's
  counted as Bicycle from S4_Crash_bicycle. Nothing is double-counted.
- **Final export** -- whatever's left after mode classification is what's
  loaded into this dashboard right now.
        """
    )

    st.markdown("### What actually happens at the classification step")
    st.markdown(
        """
Every crash in the combined crash_event table runs through one
classification rule, in this order:

1. **Qwen narrative says E-Bike or E-Scooter** (from `multilabel_ebike.xlsx`
   + `multilabel_RegBike.xlsx`, Signal4Data narratives) -- authoritative,
   even if that REPORT_NUMBER is also present in S4_Crash_bicycle.
2. **REPORT_NUMBER is in S4_Crash_bicycle** (and wasn't overridden above)
   &rarr; classified as **Bicycle**.
3. **Everything else &rarr; "Other"**, including pedestrian-only,
   single-vehicle, and motor-vehicle-only crashes from Signal4Data, plus
   non-motorist records coded `'Other Cyclist'` -- these are deliberately
   **not** guessed into E-Bike or E-Scooter, since that code also covers
   unicycles, tricycles, cargo bikes, and para-cycles, and there isn't
   enough evidence to know which.

So the gap between the combined **crash_event** total and this dashboard's
**active-mode export** isn't missing or lost data -- it's every crash that
landed in "Other" at this step, mostly pedestrian-only and MV-only crashes
from Signal4Data that were never bicycle/e-bike/e-scooter to begin with.
E-Bike and E-Scooter can **only** come from an explicit Qwen narrative
match -- there's no structural fallback for those two modes, since Florida
crash forms have no dedicated code for either.
        """
    )

    if meta_raw is not None:
        metric_col = find_col(meta_raw, ["metric"]) or meta_raw.columns[0]
        value_col = find_col(meta_raw, ["value"]) or (meta_raw.columns[1] if len(meta_raw.columns) > 1 else meta_raw.columns[0])
        note_col = find_col(meta_raw, ["note"])

        m = meta_raw.set_index(metric_col)[value_col]

        # Two source inputs merge into crash_event now (S4_Crash_bicycle +
        # Signal4Data), so they're shown as parallel KPI cards rather than
        # forced into a single linear funnel -- a funnel implies each step
        # narrows the one before it, which no longer holds once two separate
        # source populations combine.
        source_metrics = [s for s in ["bicycle_population_s4_crash_bicycle", "query_params_report_numbers"] if s in m.index]
        if source_metrics:
            st.markdown("### Two source populations")
            sc = st.columns(len(source_metrics))
            for col, s in zip(sc, source_metrics):
                col.markdown(
                    f"""<div class="kpi-card">
                        <div class="kpi-label">{META_METRIC_LABELS.get(s, s)}</div>
                        <div class="kpi-value">{int(m[s]):,}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.caption(
                "S4_Crash_bicycle is the larger, dedicated bicycle population. "
                "Signal4Data supplies E-Bike/E-Scooter (via Qwen narrative) and "
                "everything else. They merge into one combined crash_event table below."
            )

        st.markdown("### Pipeline funnel (from `dashboard_meta.csv`)")
        funnel_steps = [s for s in META_FUNNEL_ORDER if s in m.index]
        # Final export count isn't a row in dashboard_meta.csv itself --
        # it's however many rows made it into the currently loaded
        # power_bi_export.csv, so we append it as the last funnel stage.
        if funnel_steps:
            labels = [META_METRIC_LABELS.get(s, s) for s in funnel_steps] + ["Final dashboard export"]
            values = [m[s] for s in funnel_steps] + [len(df_raw)]
            fig = go.Figure(go.Funnel(
                y=labels, x=values, marker=dict(color="#2b3f8c"),
            ))
            st.plotly_chart(style_fig(fig, title="Records Retained at Each Pipeline Step", height=340), use_container_width=True)
        else:
            st.dataframe(meta_raw, use_container_width=True, hide_index=True)

        mode_metrics = [s for s in ["bicycle_crashes", "ebike_crashes", "escooter_crashes"] if s in m.index]
        if mode_metrics:
            st.markdown("**Active-mode breakdown**")
            bc1, bc2, bc3 = st.columns(len(mode_metrics))
            for col, s in zip([bc1, bc2, bc3][:len(mode_metrics)], mode_metrics):
                col.markdown(
                    f"""<div class="kpi-card">
                        <div class="kpi-label">{META_METRIC_LABELS.get(s, s)}</div>
                        <div class="kpi-value">{int(m[s]):,}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        if note_col:
            with st.expander("What each metric means"):
                for _, row in meta_raw.iterrows():
                    label = META_METRIC_LABELS.get(row[metric_col], row[metric_col])
                    st.markdown(f"**{label}** ({int(row[value_col]):,}) -- {row[note_col]}")
    else:
        st.markdown(
            f"""<div class="section-note">
            No <code>{DEFAULT_META_PATH}</code> loaded yet, so the exact
            step-by-step funnel isn't shown here -- add it under the
            <b>Data Source</b> panel in the sidebar to see records retained
            at each pipeline stage (S4_Crash_bicycle + Signal4Data query pulled
            &rarr; combined crash_event &rarr; active-mode classified &rarr;
            final export). In the meantime, the current loaded export contains
            <b>{len(df_raw):,}</b> crash records after all pipeline steps.
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("## Data sources")
    st.markdown(
        """
- **S4_Crash_bicycle** -- the larger, dedicated bicycle-crash population.
  Source of every Bicycle-mode record, minus any Qwen E-Bike/E-Scooter
  override.
- **Signal4** crash report tables (crash-level fields: date/time, location,
  severity, contributing factors, citations) -- source of E-Bike, E-Scooter,
  and Other.
- **FDOT** roadway tables (AADT, intersection control, functional
  classification -- matched by location where available), pulled from
  whichever source system (S4_Crash_bicycle or Signal4Data) each crash
  came from.
- **Non-motorist person-level records** (age, gender, and other
  demographics of the person on the bicycle/e-bike/e-scooter, when a
  demographics export is loaded)
        """
    )


    render_pipeline_figures("tab0")

# ---------------------------------------------------------------------------
# TAB 1 -- OVERVIEW & TRENDS
# ---------------------------------------------------------------------------

with tab1:
    c1, c2 = st.columns([1, 1.4])

    with c1:
        mode_counts = df["MODE"].value_counts().reindex(MODES).fillna(0)
        fig = go.Figure(go.Pie(
            labels=mode_counts.index, values=mode_counts.values, hole=0.55,
            marker=dict(colors=[MODE_COLORS[m] for m in mode_counts.index]),
            textinfo="label+percent", textfont=dict(size=13),
        ))
        fig.add_annotation(text=f"{total:,}<br>crashes", showarrow=False, font=dict(size=15, color="#12172b"))
        st.plotly_chart(style_fig(fig, title="Crash Share by Mode"), use_container_width=True)

    with c2:
        annual = df.groupby(["YEAR", "MODE"], observed=True).size().reset_index(name="count")
        fig = px.line(
            annual, x="YEAR", y="count", color="MODE", markers=True,
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        st.plotly_chart(style_fig(fig, title="Annual Crash Trend by Mode"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        monthly = df.groupby(["MONTH", "MODE"], observed=True).size().reset_index(name="count")
        monthly["Month"] = monthly["MONTH"].apply(lambda m: MONTH_NAMES[m - 1])
        fig = px.line(
            monthly, x="Month", y="count", color="MODE", markers=True,
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES, "Month": MONTH_NAMES},
        )
        st.plotly_chart(style_fig(fig, title="Seasonal Pattern by Month"), use_container_width=True)

    with c4:
        hour_mode = (
            df.groupby(["MODE", "HOUR"], observed=True).size()
            .reset_index(name="count")
            .pivot(index="MODE", columns="HOUR", values="count")
            .reindex(MODES).fillna(0)
        )
        fig = go.Figure(go.Heatmap(
            z=hour_mode.values, x=hour_mode.columns, y=hour_mode.index,
            colorscale="YlOrRd", colorbar=dict(title="Crashes"),
        ))
        st.plotly_chart(style_fig(fig, title="Crashes by Hour of Day"), use_container_width=True)


    render_pipeline_figures("tab1")

# ---------------------------------------------------------------------------
# TAB 2 -- SEVERITY & OUTCOMES
# ---------------------------------------------------------------------------

with tab2:
    c1, c2 = st.columns([1.3, 1])

    with c1:
        sev_mode = (
            df.groupby(["MODE", "S4_CRASH_SEVERITY"], observed=True).size()
            .reset_index(name="count")
        )
        sev_mode["pct"] = sev_mode["count"] / sev_mode.groupby("MODE")["count"].transform("sum") * 100
        fig = px.bar(
            sev_mode, x="MODE", y="pct", color="S4_CRASH_SEVERITY",
            color_discrete_map=SEVERITY_COLORS,
            category_orders={"MODE": MODES, "S4_CRASH_SEVERITY": SEVERITY_ORDER},
            custom_data=["count"],
        )
        fig.update_traces(
            texttemplate="%{customdata[0]:,}", textposition="inside", textfont=dict(size=10, color="white"),
        )
        fig.update_layout(yaxis_title="% of crashes", xaxis_title=None, barmode="stack")
        mode_n = df.groupby("MODE", observed=True).size().reindex(MODES).fillna(0).astype(int).to_dict()
        st.plotly_chart(
            style_fig(fig, title="Injury Severity Mix by Mode", n=mode_n), use_container_width=True
        )
        st.caption("Segment labels are raw crash counts; bar height is % of that mode's crashes.")

    with c2:
        mode_sizes = {m: int((df["MODE"] == m).sum()) for m in MODES}
        fatal_n = {m: int(((df["MODE"] == m) & (df["S4_CRASH_SEVERITY"] == "Fatality")).sum()) for m in MODES}
        serious_n = {m: int(((df["MODE"] == m) & (df["S4_CRASH_SEVERITY"] == "Serious Injury")).sum()) for m in MODES}
        rate_df = pd.DataFrame({
            "MODE": MODES,
            "Fatal %": [fatal_n[m] / mode_sizes[m] * 100 if mode_sizes[m] else 0 for m in MODES],
            "Serious Injury %": [serious_n[m] / mode_sizes[m] * 100 if mode_sizes[m] else 0 for m in MODES],
        })
        fig = go.Figure()
        fig.add_bar(
            name="Fatal %", x=rate_df["MODE"], y=rate_df["Fatal %"], marker_color="#B71C1C",
            text=[f"{v:.2f}% (n={fatal_n[m]:,})" for m, v in zip(rate_df["MODE"], rate_df["Fatal %"])],
            textposition="outside",
        )
        fig.add_bar(
            name="Serious Injury %", x=rate_df["MODE"], y=rate_df["Serious Injury %"], marker_color="#EF9A9A",
            text=[f"{v:.1f}% (n={serious_n[m]:,})" for m, v in zip(rate_df["MODE"], rate_df["Serious Injury %"])],
            textposition="outside",
        )
        fig.update_layout(barmode="group", yaxis_title="%")
        st.plotly_chart(
            style_fig(fig, title="Fatal & Serious Injury Rate", n=mode_sizes), use_container_width=True
        )
        st.caption(
            "Denominator for each mode's % is that mode's total crash count in the current "
            "filter (shown as n= in the subtitle); bar labels give the numerator count too."
        )

    c3, c4 = st.columns(2)
    with c3:
        sev_yr = (
            df.groupby(["YEAR", "S4_CRASH_SEVERITY"], observed=True).size()
            .reset_index(name="count")
        )
        # % of that year's crashes, not raw counts -- raw counts conflate the
        # changing severity MIX with the separate fact that overall crash
        # volume also rose/fell year to year.
        sev_yr["pct"] = sev_yr["count"] / sev_yr.groupby("YEAR")["count"].transform("sum") * 100
        fig = px.area(
            sev_yr, x="YEAR", y="pct", color="S4_CRASH_SEVERITY",
            color_discrete_map=SEVERITY_COLORS,
            category_orders={"S4_CRASH_SEVERITY": SEVERITY_ORDER},
            custom_data=["count"],
        )
        fig.update_traces(hovertemplate="%{x}: %{y:.1f}%% (n=%{customdata[0]:,})<extra>%{fullData.name}</extra>")
        fig.update_layout(yaxis_title="% of that year's crashes")
        st.plotly_chart(
            style_fig(fig, title="Severity Mix Over Time (% of Crashes)", n=total),
            use_container_width=True,
        )
        st.caption("Hover a band for the raw crash count behind that year/severity slice.")

    with c4:
        mode_sizes4 = {m: int((df["MODE"] == m).sum()) for m in MODES}
        mv_n = {m: int(((df["MODE"] == m) & (df["mv_involved"] == True)).sum()) for m in MODES}  # noqa: E712
        mv_df = df.groupby("MODE", observed=True)["mv_involved"].mean().reindex(MODES).fillna(0) * 100
        fig = go.Figure(go.Bar(
            x=mv_df.index, y=mv_df.values,
            marker_color=[MODE_COLORS[m] for m in mv_df.index],
            text=[f"{v:.0f}% (n={mv_n.get(m, 0):,})" for m, v in mv_df.items()], textposition="outside",
        ))
        fig.update_layout(yaxis_title="% crashes with MV involved", yaxis_range=[0, 110])
        st.plotly_chart(
            style_fig(fig, title="Motor Vehicle Involvement by Mode", n=mode_sizes4), use_container_width=True
        )

    fars = df[df["FARS_LANDUSE"].notna()]
    if len(fars):
        urban = (fars["FARS_LANDUSE"] == "Urban").sum()
        rural = (fars["FARS_LANDUSE"] == "Rural").sum()
        st.markdown(
            f"""<div class="section-note">
            <b>FARS federal fatal-crash coding</b> (applies only to the
            {len(fars):,} fatal crashes in the current filter with a FARS
            match): {urban:,} occurred on <b>Urban</b> roads,
            {rural:,} on <b>Rural</b> roads.
            </div>""",
            unsafe_allow_html=True,
        )


    render_pipeline_figures("tab2")

# ---------------------------------------------------------------------------
# TAB 3 -- WHEN & WHERE
# ---------------------------------------------------------------------------

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        dow_mode = df.groupby(["DOW", "MODE"], observed=True).size().reset_index(name="count")
        fig = px.bar(
            dow_mode, x="DOW", y="count", color="MODE",
            color_discrete_map=MODE_COLORS,
            category_orders={"DOW": DOW_ORDER, "MODE": MODES},
        )
        fig.update_layout(xaxis_title=None, yaxis_title="Crashes")
        st.plotly_chart(style_fig(fig, title="Crashes by Day of Week", n=total), use_container_width=True)

    with c2:
        dn_mode = df.groupby(["MODE", "DAY_NIGHT"], observed=True).size().reset_index(name="count")
        dn_mode["pct"] = dn_mode["count"] / dn_mode.groupby("MODE")["count"].transform("sum") * 100
        fig = px.bar(
            dn_mode, x="MODE", y="pct", color="DAY_NIGHT",
            color_discrete_map={"Day": "#FDD835", "Night": "#283593"},
            category_orders={"MODE": MODES}, custom_data=["count"],
        )
        fig.update_traces(texttemplate="%{y:.0f}%<br>(n=%{customdata[0]:,})", textposition="inside",
                           textfont=dict(size=10, color="#12172b"))
        fig.update_layout(yaxis_title="% of crashes", xaxis_title=None, barmode="stack")
        dn_mode_n = dn_mode.groupby("MODE")["count"].sum().reindex(MODES).fillna(0).astype(int).to_dict()
        st.plotly_chart(
            style_fig(fig, title="Day vs. Night Share by Mode", n=dn_mode_n), use_container_width=True
        )

    c3, c4 = st.columns(2)
    with c3:
        loc_mode = df.groupby(["MODE", "LOC_TYPE"], observed=True).size().reset_index(name="count")
        loc_mode["pct"] = loc_mode["count"] / loc_mode.groupby("MODE")["count"].transform("sum") * 100
        fig = px.bar(
            loc_mode, x="MODE", y="pct", color="LOC_TYPE",
            color_discrete_map={"Intersection": "#5C6BC0", "Segment": "#26A69A"},
            category_orders={"MODE": MODES}, custom_data=["count"],
        )
        fig.update_traces(texttemplate="%{y:.0f}%<br>(n=%{customdata[0]:,})", textposition="inside",
                           textfont=dict(size=10, color="white"))
        fig.update_layout(yaxis_title="% of crashes", xaxis_title=None, barmode="stack")
        loc_mode_n = loc_mode.groupby("MODE")["count"].sum().reindex(MODES).fillna(0).astype(int).to_dict()
        st.plotly_chart(
            style_fig(fig, title="Intersection vs. Segment by Mode", n=loc_mode_n), use_container_width=True
        )

    with c4:
        light_top = df["LIGHT_CONDITION"].value_counts().nlargest(6).index
        light_df = df[df["LIGHT_CONDITION"].isin(light_top)]
        lm = light_df.groupby(["LIGHT_CONDITION", "MODE"], observed=True).size().reset_index(name="count")
        # % within mode: raw counts make Bicycle (the largest group) dominate
        # every bar, which hides whether E-Bike/E-Scooter have a genuinely
        # different SHAPE of light-condition distribution.
        mode_totals = df.groupby("MODE", observed=True).size()
        lm["pct"] = lm.apply(lambda r: r["count"] / mode_totals.get(r["MODE"], 1) * 100, axis=1)
        fig = px.bar(
            lm, y="LIGHT_CONDITION", x="pct", color="MODE", orientation="h", barmode="group",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES}, custom_data=["count"],
        )
        fig.update_traces(texttemplate="%{x:.0f}% (n=%{customdata[0]:,})", textposition="outside",
                           textfont=dict(size=9))
        fig.update_layout(yaxis_title=None, xaxis_title="% of that mode's crashes",
                           yaxis={"categoryorder": "total ascending"})
        light_n = {m: int(mode_totals.get(m, 0)) for m in MODES}
        st.plotly_chart(
            style_fig(fig, title="Light Conditions (% Within Mode)", n=light_n, height=420),
            use_container_width=True,
        )

    c5, c6 = st.columns(2)
    with c5:
        wthr_top = df["WEATHER_CONDITION"].value_counts().nlargest(5).index
        wthr_df = df[df["WEATHER_CONDITION"].isin(wthr_top)]
        wm = wthr_df.groupby(["WEATHER_CONDITION", "MODE"], observed=True).size().reset_index(name="count")
        wm["pct"] = wm.apply(lambda r: r["count"] / mode_totals.get(r["MODE"], 1) * 100, axis=1)
        fig = px.bar(
            wm, y="WEATHER_CONDITION", x="pct", color="MODE", orientation="h", barmode="group",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES}, custom_data=["count"],
        )
        fig.update_traces(texttemplate="%{x:.0f}% (n=%{customdata[0]:,})", textposition="outside",
                           textfont=dict(size=9))
        fig.update_layout(yaxis_title=None, xaxis_title="% of that mode's crashes",
                           yaxis={"categoryorder": "total ascending"})
        wthr_n = {m: int(mode_totals.get(m, 0)) for m in MODES}
        st.plotly_chart(
            style_fig(fig, title="Weather Conditions (% Within Mode)", n=wthr_n, height=400),
            use_container_width=True,
        )

    with c6:
        top_counties = df["COUNTY_NAME"].value_counts().nlargest(15).index
        co_df = df[df["COUNTY_NAME"].isin(top_counties)]
        cm = co_df.groupby(["COUNTY_NAME", "MODE"], observed=True).size().reset_index(name="count")
        fig = px.bar(
            cm, y="COUNTY_NAME", x="count", color="MODE", orientation="h",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        fig.update_layout(yaxis_title=None, xaxis_title="Crashes",
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(
            style_fig(fig, title="Top 15 Counties", height=430, n=int(co_df.shape[0])),
            use_container_width=True,
        )

    st.markdown("#### Crash Locations")
    if LAT_COL and LON_COL:
        geo = df[[LAT_COL, LON_COL, "MODE"]].copy()
        geo[LAT_COL] = pd.to_numeric(geo[LAT_COL], errors="coerce")
        geo[LON_COL] = pd.to_numeric(geo[LON_COL], errors="coerce")
        # Same Florida bounding-box sanity filter the pipeline's own 09c
        # scatter uses, to drop bad/placeholder geocodes.
        geo = geo[
            geo[LAT_COL].between(24, 31) & geo[LON_COL].between(-88, -79)
        ]
        if len(geo):
            fig = px.scatter_mapbox(
                geo, lat=LAT_COL, lon=LON_COL, color="MODE",
                color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
                opacity=0.55, zoom=5.4, height=560,
            )
            fig = style_fig(
                fig, height=560,
                title=f"Crash Locations by Mode ({len(geo):,} of {total:,} filtered crashes geocoded)",
            )
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=56, b=0))
            fig.update_traces(marker=dict(size=6))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No crashes with valid Florida coordinates in the current filter selection.")
    else:
        st.markdown(
            """<div class="section-note">
            No latitude/longitude columns found in the loaded export, so the
            map can't be drawn. The pipeline computes these as
            <code>S4_LATITUDE</code> / <code>S4_LONGITUDE</code> (preferred
            -- ~98.5% complete) with <code>LATITUDE</code> / <code>LONGITUDE</code>
            as a fallback; include one of those pairs in
            <code>power_bi_export.csv</code> to enable this map. In the
            meantime, see the static Florida scatter plot from
            <code>09_latlon</code> in the pipeline-figures expander below.
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Crashes by Census Tract")
    st.markdown(
        """<div class="section-note">
        <b>Method:</b> each geocoded crash's point (lat/lon) is
        <b>spatially joined</b> to the Florida census tract polygon it falls
        inside of (a point-in-polygon join against 2023 TIGER/Line tract
        boundaries, <code>geopandas.sjoin(..., predicate="within")</code>,
        both layers in EPSG:4326 / WGS84). Every crash lands in exactly one
        tract (or none, if its coordinates fall outside all tract polygons
        -- e.g. bad geocodes). Crashes are then aggregated to
        <code>GEOID</code> and joined to ACS total-population estimates to
        build the three maps below. This gives a <i>rate</i> per tract
        (crashes relative to who lives there), which raw point density on
        the scatter map above can't show.
        </div>""",
        unsafe_allow_html=True,
    )

    if not GEOPANDAS_AVAILABLE:
        st.warning(
            "`geopandas` isn't installed in this environment, so the census-tract maps can't "
            "render. Install it (`pip install geopandas shapely`) and re-run the dashboard."
        )
    elif tracts_raw is None:
        st.info(
            "No census tract boundary file loaded yet. Upload a `census_tracts.geojson` "
            "(GEOID + population + geometry, see the **Optional: census tract boundaries** "
            "panel in the sidebar) to enable these three maps."
        )
    elif not (LAT_COL and LON_COL):
        st.info("No latitude/longitude columns in the loaded export, so points can't be joined to tracts.")
    elif "GEOID" not in tracts_raw.columns:
        st.warning("The uploaded tract file has no `GEOID` column -- can't aggregate to it.")
    else:
        geo_pts = df[[LAT_COL, LON_COL, "MODE"]].copy()
        geo_pts[LAT_COL] = pd.to_numeric(geo_pts[LAT_COL], errors="coerce")
        geo_pts[LON_COL] = pd.to_numeric(geo_pts[LON_COL], errors="coerce")
        geo_pts = geo_pts[geo_pts[LAT_COL].between(24, 31) & geo_pts[LON_COL].between(-88, -79)]

        if len(geo_pts) == 0:
            st.info("No crashes with valid Florida coordinates in the current filter selection.")
        else:
            pts_gdf = gpd.GeoDataFrame(
                geo_pts,
                geometry=gpd.points_from_xy(geo_pts[LON_COL], geo_pts[LAT_COL]),
                crs=4326,
            )
            joined = gpd.sjoin(pts_gdf, tracts_raw[["GEOID", "geometry"]], how="left", predicate="within")
            n_matched = joined["GEOID"].notna().sum()
            n_unmatched = len(joined) - n_matched
            st.caption(
                f"**{n_matched:,}** of **{len(joined):,}** geocoded, filtered crashes "
                f"(n = {len(joined):,}) matched to a tract; **{n_unmatched:,}** fell outside "
                f"every tract polygon (bad/edge-of-state geocodes) and are excluded from the maps below."
            )
            joined = joined.dropna(subset=["GEOID"])

            if len(joined) == 0:
                st.info("No crashes matched a census tract in the current filter selection.")
            else:
                tract_counts = (
                    joined.groupby(["GEOID", "MODE"], observed=True).size()
                    .unstack(fill_value=0).reindex(columns=MODES, fill_value=0)
                )
                tract_counts["TOTAL_MICRO"] = tract_counts[MODES].sum(axis=1)
                tract_geo = tracts_raw.merge(tract_counts.reset_index(), on="GEOID", how="left")
                for c in list(MODES) + ["TOTAL_MICRO"]:
                    tract_geo[c] = tract_geo[c].fillna(0)

                has_pop = tract_pop_col in tract_geo.columns
                if has_pop:
                    tract_geo[tract_pop_col] = pd.to_numeric(tract_geo[tract_pop_col], errors="coerce")
                    tract_geo["EBIKE_PER_10K_POP"] = np.where(
                        tract_geo[tract_pop_col] > 0,
                        tract_geo["E-Bike"] / tract_geo[tract_pop_col] * 10_000, np.nan,
                    )
                tract_geo["EBIKE_SHARE_OF_MICRO"] = np.where(
                    tract_geo["TOTAL_MICRO"] > 0,
                    tract_geo["E-Bike"] / tract_geo["TOTAL_MICRO"] * 100, np.nan,
                )

                def choropleth(gdf_col, value_col, title, colorbar_title, subtitle_n, colorscale="YlOrRd"):
                    fig = px.choropleth_mapbox(
                        tract_geo, geojson=tract_geo.geometry.__geo_interface__,
                        locations=tract_geo.index, color=gdf_col,
                        color_continuous_scale=colorscale,
                        mapbox_style="open-street-map", zoom=5.4,
                        center={"lat": 27.8, "lon": -81.7}, opacity=0.7,
                        labels={gdf_col: colorbar_title},
                    )
                    fig = style_fig(fig, height=520, title=title, n=subtitle_n)
                    fig.update_layout(margin=dict(l=0, r=0, t=70, b=0))
                    return fig

                m1, m2 = st.columns(2)
                with m1:
                    st.plotly_chart(
                        choropleth(
                            "TOTAL_MICRO", "TOTAL_MICRO",
                            "1. Micromobility Crashes per Tract (all modes)",
                            "Crashes", n_matched,
                        ),
                        use_container_width=True,
                    )
                    st.caption("Raw crash count per tract, all three modes combined.")

                with m2:
                    if has_pop:
                        st.plotly_chart(
                            choropleth(
                                "EBIKE_PER_10K_POP", "EBIKE_PER_10K_POP",
                                "2. E-Bike Crashes per 10,000 Residents",
                                "E-bike crashes / 10k pop", n_matched, colorscale="Reds",
                            ),
                            use_container_width=True,
                        )
                        st.caption(
                            f"E-bike crash count in the tract \u00f7 tract population "
                            f"(`{tract_pop_col}`) \u00d7 10,000 -- normalizes for the fact that "
                            f"densely populated tracts will rack up more crashes by exposure alone."
                        )
                    else:
                        st.info(
                            f"Population column '{tract_pop_col}' not found in the tract file -- "
                            f"can't compute crashes-per-capita. Check the column name in the sidebar."
                        )

                st.plotly_chart(
                    choropleth(
                        "EBIKE_SHARE_OF_MICRO", "EBIKE_SHARE_OF_MICRO",
                        "3. E-Bike Share of All Micromobility Crashes per Tract (%)",
                        "% e-bike", n_matched, colorscale="Purples",
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "E-bike crashes \u00f7 (bicycle + e-bike + e-scooter crashes) in that tract, as a "
                    "%. Only meaningful where TOTAL_MICRO is non-trivial -- a tract with 1 total "
                    "crash that happens to be an e-bike crash shows 100% here, so read this "
                    "alongside Map 1's raw count, not in isolation."
                )

    render_pipeline_figures("tab3")

# ---------------------------------------------------------------------------
# TAB 4 -- DRIVER BEHAVIOR & CITATIONS
# ---------------------------------------------------------------------------

with tab4:
    c1, c2 = st.columns([1.4, 1])

    with c1:
        rows = []
        for m in MODES:
            sub = df[df["MODE"] == m]
            n = len(sub)
            for f in DRIVER_FLAGS:
                rows.append({
                    "Mode": m,
                    "Flag": DRIVER_FLAG_LABELS.get(f, f),
                    "Pct": sub[f].mean() * 100 if n else 0,
                })
        flag_df = pd.DataFrame(rows)
        flag_n = {m: int((df["MODE"] == m).sum()) for m in MODES}
        flag_df["N"] = flag_df["Mode"].map(flag_n)
        flag_df["count"] = (flag_df["Pct"] / 100 * flag_df["N"]).round().astype(int)
        fig = px.bar(
            flag_df, y="Flag", x="Pct", color="Mode", orientation="h", barmode="group",
            color_discrete_map=MODE_COLORS, category_orders={"Mode": MODES}, custom_data=["count"],
        )
        fig.update_traces(texttemplate="%{x:.0f}% (n=%{customdata[0]:,})", textposition="outside", textfont=dict(size=9))
        fig.update_layout(xaxis_title="% of crashes for that mode", yaxis_title=None)
        st.plotly_chart(
            style_fig(fig, title="Driver Behavior Flags by Mode", height=440, n=flag_n), use_container_width=True
        )

    with c2:
        cite_mode_n = {m: int((df["MODE"] == m).sum()) for m in MODES}
        cite_n = {m: int(((df["MODE"] == m) & (df["CITED"] == True)).sum()) for m in MODES}  # noqa: E712
        cite_df = df.groupby("MODE", observed=True)["CITED"].mean().reindex(MODES).fillna(0) * 100
        fig = go.Figure(go.Bar(
            x=cite_df.index, y=cite_df.values,
            marker_color=[MODE_COLORS[m] for m in cite_df.index],
            text=[f"{v:.0f}% (n={cite_n.get(m, 0):,})" for m, v in cite_df.items()], textposition="outside",
        ))
        fig.update_layout(yaxis_title="% crashes with citation", yaxis_range=[0, 110])
        st.plotly_chart(
            style_fig(fig, title="Citation Rate by Mode", n=cite_mode_n), use_container_width=True
        )

        cite_yr = df.groupby(["YEAR", "MODE"], observed=True)["CITED"].mean().reset_index()
        cite_yr["CITED"] *= 100
        fig2 = px.line(
            cite_yr, x="YEAR", y="CITED", color="MODE", markers=True,
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        fig2.update_layout(yaxis_title="% cited", xaxis_title=None)
        st.plotly_chart(style_fig(fig2, title="Citation Rate Over Time, by Mode", height=280), use_container_width=True)
        st.caption(
            "A declining rate here can reflect changing enforcement/charging practice, more "
            "crashes being self-reported without an officer response, or a reporting-lag "
            "artifact in the most recent year(s) (citations can be filed after the initial "
            "report) -- check whether the drop is concentrated in the latest year(s) before "
            "treating it as a real behavioral trend."
        )

    st.markdown("---")
    st.markdown("## Driver Distraction Type")
    if "DISTRACTION_TYPE" in df.columns and df["DISTRACTION_TYPE"].notna().any():
        dist_df = df[df["DISTRACTION_TYPE"].notna()].copy()
        dist_df = dist_df[~dist_df["DISTRACTION_TYPE"].astype(str).str.lower().str.contains("not distracted")]
        if len(dist_df):
            dist_top_n = st.slider("Number of distraction types to show", 5, 15, 8, key="distraction_topn")
            dist_totals = dist_df["DISTRACTION_TYPE"].value_counts().nlargest(dist_top_n).index
            dsub2 = dist_df[dist_df["DISTRACTION_TYPE"].isin(dist_totals)]
            dm3 = dsub2.groupby(["DISTRACTION_TYPE", "MODE"], observed=True).size().reset_index(name="count")
            fig = px.bar(
                dm3, y="DISTRACTION_TYPE", x="count", color="MODE", orientation="h",
                color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
            )
            fig.update_layout(
                yaxis_title=None, xaxis_title="Crashes",
                yaxis={"categoryorder": "total ascending"}, barmode="stack",
            )
            st.plotly_chart(
                style_fig(fig, title=f"Driver Distraction Type (top {dist_top_n}, by mode, excl. 'Not Distracted')", height=440),
                use_container_width=True,
            )
            st.caption(
                "Source: `driver.csv` (`DRIVER_DISTRACTION_CODE`) -- the specific distraction "
                "behind the Distracted Driving flag above. 'Not Distracted' is excluded so the "
                "chart focuses on actual distraction types. When a crash has more than one "
                "driver record, the non-'Not Distracted' code is preferred."
            )
        else:
            st.info("No distraction-type records (excl. 'Not Distracted') in the current filter selection.")
    else:
        st.info(
            "No `DISTRACTION_TYPE` column in the loaded export. Re-run "
            "`eda_analysis_combined.py` to pick it up in `power_bi_export.csv`."
        )


    render_pipeline_figures("tab4")

# ---------------------------------------------------------------------------
# TAB 5 -- ROADWAY INFRASTRUCTURE
# ---------------------------------------------------------------------------

with tab5:
    aadt_df = df[df["AVG_AADT"].notna()]
    ictrl_df = df[df["INTERSECTION_CONTROL"].notna()]

    infra_cols_present = {c: lbl for c, lbl in INFRA_COL_LABELS.items() if c in df.columns and df[c].notna().any()}
    missing_infra = [lbl for c, lbl in INFRA_COL_LABELS.items() if c not in infra_cols_present]

    note_text = (
        f"Roadway-infrastructure fields only match a subset of crashes via the "
        f"FDOT segment/intersection tables: AADT is available for "
        f"{len(aadt_df):,} of {total:,} filtered crashes "
        f"({len(aadt_df) / total * 100:.0f}%), intersection control type for "
        f"{len(ictrl_df):,} ({len(ictrl_df) / total * 100:.0f}%)."
    )
    if missing_infra:
        note_text += f" {', '.join(missing_infra)} {'is' if len(missing_infra) == 1 else 'are'} not populated in the current extract, so those charts are omitted."
    st.markdown(f"""<div class="section-note">{note_text}</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if len(aadt_df):
            fig = px.violin(
                aadt_df, x="MODE", y="AVG_AADT", color="MODE", box=True, points=False,
                color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
            )
            fig.update_layout(yaxis_title="Average Annual Daily Traffic", xaxis_title=None, showlegend=False)
            st.plotly_chart(style_fig(fig, title="Traffic Volume (AADT) by Mode"), use_container_width=True)
        else:
            st.info("No AADT data in the current filter selection.")

    with c2:
        if len(ictrl_df):
            ic = ictrl_df.groupby(["INTERSECTION_CONTROL", "MODE"], observed=True).size().reset_index(name="count")
            ictrl_mode_totals = ictrl_df.groupby("MODE", observed=True).size()
            ic["pct"] = ic.apply(lambda r: r["count"] / ictrl_mode_totals.get(r["MODE"], 1) * 100, axis=1)
            fig = px.bar(
                ic, y="INTERSECTION_CONTROL", x="pct", color="MODE", orientation="h", barmode="group",
                color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
            )
            fig.update_layout(yaxis_title=None, xaxis_title="% of that mode's crashes (with intersection-control data)",
                               yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_fig(fig, title="Intersection Control Type by Mode (% Within Mode)"), use_container_width=True)
        else:
            st.info("No intersection-control data in the current filter selection.")

    if SPEED_COL:
        spd_df = df[pd.to_numeric(df[SPEED_COL], errors="coerce").notna()].copy()
        spd_df[SPEED_COL] = pd.to_numeric(spd_df[SPEED_COL], errors="coerce")
        if len(spd_df):
            fig = px.violin(
                spd_df, x="MODE", y=SPEED_COL, color="MODE", box=True, points=False,
                color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
            )
            fig.update_layout(yaxis_title="Posted Speed Limit (mph)", xaxis_title=None, showlegend=False)
            st.plotly_chart(style_fig(fig, title="Posted Speed Limit Distribution by Mode"), use_container_width=True)
        else:
            st.info("No Posted Speed Limit data in the current filter selection.")

    if MICRO_SPEED_COL:
        mspd_df = df[pd.to_numeric(df[MICRO_SPEED_COL], errors="coerce").notna()].copy()
        mspd_df[MICRO_SPEED_COL] = pd.to_numeric(mspd_df[MICRO_SPEED_COL], errors="coerce")
        if len(mspd_df):
            fig = px.violin(
                mspd_df, x="MODE", y=MICRO_SPEED_COL, color="MODE", box=True, points="all",
                color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
            )
            fig.update_layout(yaxis_title="Self-Reported Speed (mph)", xaxis_title=None, showlegend=False)
            st.plotly_chart(
                style_fig(fig, title=f"Micromobility Speed From Crash Narratives by Mode (n={len(mspd_df):,})"),
                use_container_width=True,
            )
            st.caption(
                "Extracted from the `micromobility_speed` narrative field. Crashes where the "
                "narrative gave no numeric speed (or only '0mph' placeholders) are excluded."
            )
        else:
            total_with_speed = pd.to_numeric(df_raw[MICRO_SPEED_COL], errors="coerce").notna().sum()
            st.info(
                f"No Micromobility Speed data in the current filter selection "
                f"({total_with_speed:,} crash(es) in the full dataset have a value -- "
                f"try widening the sidebar filters, e.g. reset Injury Severity to 'All' or widen the date range)."
            )
    elif SPEED_COL:
        # Only mention the missing column once, right next to the sibling chart
        # that DID find its column -- avoids a confusing silent gap here.
        st.caption(
            "Micromobility Speed (self-reported speed from crash narratives) isn't present in "
            "the currently loaded `power_bi_export.csv` -- re-run `eda_analysis_combined.py` and "
            "reload the CSV to pick it up."
        )

    st.markdown("---")
    st.markdown("## Speed \u00d7 Infrastructure: Is Speed Higher When the Narrative Mentions Sidewalk Riding?")
    if MICRO_SPEED_COL and narrative_raw is not None and "NARRATIVE_TEXT" in narrative_raw.columns and MAIN_CRASH_ID_COL:
        INFRA_SPEED_THEMES = {
            "Sidewalk riding": r"\bsidewalk\b",
            "Crosswalk": r"\bcrosswalk\b",
            "Bike lane": r"\bbike ?lane\b",
            "Wrong-way / against traffic": r"\bwrong.?way\b|against traffic|facing traffic",
        }
        nmerge = narrative_raw[[c for c in ["REPORT_NUMBER", "NARRATIVE_TEXT"] if c in narrative_raw.columns]].copy()
        nmerge["REPORT_NUMBER"] = nmerge["REPORT_NUMBER"].astype(str)
        nmerge["NARRATIVE_TEXT"] = nmerge["NARRATIVE_TEXT"].fillna("").str.lower()

        spd_infra = df[[MAIN_CRASH_ID_COL, "MODE", MICRO_SPEED_COL]].copy()
        spd_infra[MAIN_CRASH_ID_COL] = spd_infra[MAIN_CRASH_ID_COL].astype(str)
        spd_infra[MICRO_SPEED_COL] = pd.to_numeric(spd_infra[MICRO_SPEED_COL], errors="coerce")
        spd_infra = spd_infra.merge(nmerge, left_on=MAIN_CRASH_ID_COL, right_on="REPORT_NUMBER", how="inner")
        spd_infra = spd_infra[spd_infra[MICRO_SPEED_COL].notna()]

        if len(spd_infra) < 10:
            st.info(
                f"Only {len(spd_infra):,} filtered crashes have both a narrative and a "
                f"self-reported speed -- too few to compare. Try widening the sidebar filters."
            )
        else:
            theme_pick = st.selectbox(
                "Infrastructure/behavior theme to compare against speed",
                list(INFRA_SPEED_THEMES.keys()), key="speed_infra_theme",
            )
            pattern = INFRA_SPEED_THEMES[theme_pick]
            spd_infra["FLAG"] = spd_infra["NARRATIVE_TEXT"].str.contains(pattern, regex=True, na=False)
            spd_infra["Group"] = np.where(spd_infra["FLAG"], f"{theme_pick} mentioned", "Not mentioned")

            n_flag = int(spd_infra["FLAG"].sum())
            n_noflag = int((~spd_infra["FLAG"]).sum())
            group_n = {f"{theme_pick} mentioned": n_flag, "Not mentioned": n_noflag}

            gcol1, gcol2 = st.columns([1.3, 1])
            with gcol1:
                fig = px.violin(
                    spd_infra, x="Group", y=MICRO_SPEED_COL, color="Group", box=True, points="outliers",
                    category_orders={"Group": [f"{theme_pick} mentioned", "Not mentioned"]},
                    color_discrete_sequence=["#B71C1C", "#90A4AE"],
                )
                fig.update_layout(yaxis_title="Self-reported speed (mph)", xaxis_title=None, showlegend=False)
                st.plotly_chart(
                    style_fig(fig, title=f"Speed by Narrative Mention: {theme_pick}", n=group_n),
                    use_container_width=True,
                )

            with gcol2:
                med_flag = spd_infra.loc[spd_infra["FLAG"], MICRO_SPEED_COL].median() if n_flag else np.nan
                med_noflag = spd_infra.loc[~spd_infra["FLAG"], MICRO_SPEED_COL].median() if n_noflag else np.nan
                try:
                    from scipy import stats as _stats
                    if n_flag >= 5 and n_noflag >= 5:
                        u_stat, p_val = _stats.mannwhitneyu(
                            spd_infra.loc[spd_infra["FLAG"], MICRO_SPEED_COL],
                            spd_infra.loc[~spd_infra["FLAG"], MICRO_SPEED_COL],
                            alternative="two-sided",
                        )
                        p_txt = f"Mann-Whitney U p = {p_val:.3f}"
                    else:
                        p_txt = "Too few flagged narratives for a significance test"
                except ImportError:
                    p_txt = "scipy not installed -- no significance test computed"

                st.markdown(
                    f"""<div class="section-note">
                    <b>Median speed, {theme_pick.lower()}:</b> {med_flag:.1f} mph (n={n_flag:,})<br>
                    <b>Median speed, not mentioned:</b> {med_noflag:.1f} mph (n={n_noflag:,})<br>
                    <b>{p_txt}</b><br><br>
                    This is a blunt keyword match on free-text narratives, not a validated
                    infrastructure classification -- treat it as a lead, not a final number.
                    Small n for the "mentioned" group is common since most narratives don't
                    explicitly call out sidewalk/crosswalk/bike-lane use.
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        missing_bits = []
        if not MICRO_SPEED_COL:
            missing_bits.append("`micromobility_speed` in `power_bi_export.csv`")
        if narrative_raw is None:
            missing_bits.append(f"`{DEFAULT_NARRATIVE_PATH}` (upload it in the sidebar)")
        st.info(
            "Speed \u00d7 infrastructure comparison needs both a self-reported speed column and "
            "narrative text: missing " + " and ".join(missing_bits) + "."
        )

    if infra_cols_present:
        infra_items = list(infra_cols_present.items())
        for i in range(0, len(infra_items), 2):
            pair = infra_items[i:i + 2]
            cols = st.columns(2)
            for (col_name, col_label), slot in zip(pair, cols):
                with slot:
                    sub = df[df[col_name].notna()]
                    if not len(sub):
                        continue
                    if pd.api.types.is_numeric_dtype(sub[col_name]):
                        fig = px.violin(
                            sub, x="MODE", y=col_name, color="MODE", box=True, points=False,
                            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
                        )
                        fig.update_layout(yaxis_title=col_label, xaxis_title=None, showlegend=False)
                    else:
                        top_vals = sub[col_name].value_counts().nlargest(8).index
                        vt = sub[sub[col_name].isin(top_vals)].groupby([col_name, "MODE"], observed=True).size().reset_index(name="count")
                        fig = px.bar(
                            vt, y=col_name, x="count", color="MODE", orientation="h",
                            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
                        )
                        fig.update_layout(yaxis_title=None, xaxis_title="Crashes",
                                           yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(style_fig(fig, title=f"{col_label} by Mode"), use_container_width=True)

    road_type_df = df[df[ROAD_TYPE_COL].notna()] if ROAD_TYPE_COL else pd.DataFrame()
    if len(road_type_df):
        st.caption(
            "Road Type reflects the raw FDOT `TRAFFICWAY_CODE` -- the "
            "pipeline doesn't currently map these codes to readable labels."
        )
        rt = road_type_df.groupby([ROAD_TYPE_COL, "MODE"], observed=True).size().reset_index(name="count")
        rt_mode_totals = road_type_df.groupby("MODE", observed=True).size()
        rt["pct"] = rt.apply(lambda r: r["count"] / rt_mode_totals.get(r["MODE"], 1) * 100, axis=1)
        fig = px.bar(
            rt, y=ROAD_TYPE_COL, x="pct", color="MODE", orientation="h", barmode="group",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        fig.update_layout(yaxis_title="Trafficway Code", xaxis_title="% of that mode's crashes",
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_fig(fig, title="Road Type (Trafficway Code) by Mode (% Within Mode)"), use_container_width=True)


    render_pipeline_figures("tab5")

# ---------------------------------------------------------------------------
# TAB 6 -- DEMOGRAPHICS
# ---------------------------------------------------------------------------

with tab6:
    if demo is None or demo.empty:
        st.markdown(
            f"""<div class="section-note">
            No demographics data loaded. Add <code>{DEFAULT_DEMO_PATH}</code>
            (person-level age/gender export) under the <b>Data Source</b>
            panel in the sidebar to see age and gender breakdowns for the
            people involved in these crashes.
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        demo_mode_col = find_col(demo, ["MODE"])
        has_mode = demo_mode_col is not None

        if DEMO_AGE_AVAILABLE and has_mode:
            age_valid = demo[demo["_AGE"].notna()]
            if len(age_valid):
                fig = px.violin(
                    age_valid, x=demo_mode_col, y="_AGE", color=demo_mode_col, box=True, points="outliers",
                    color_discrete_map=MODE_COLORS, category_orders={demo_mode_col: MODES},
                )
                fig.update_layout(yaxis_title="Age", xaxis_title=None, showlegend=False)
                st.plotly_chart(style_fig(fig, title="Age Distribution by Mode (Violin Plot)", height=420), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if DEMO_AGE_AVAILABLE:
                if has_mode:
                    age_band_df = demo.groupby([demo_mode_col, "_AGE_BAND"], observed=True).size().reset_index(name="count")
                    fig = px.bar(
                        age_band_df, x="_AGE_BAND", y="count", color=demo_mode_col,
                        color_discrete_map=MODE_COLORS,
                        category_orders={"_AGE_BAND": AGE_BAND_ORDER},
                    )
                else:
                    age_band_df = demo.groupby("_AGE_BAND", observed=True).size().reset_index(name="count")
                    fig = px.bar(age_band_df, x="_AGE_BAND", y="count", category_orders={"_AGE_BAND": AGE_BAND_ORDER})
                fig.update_layout(xaxis_title=None, yaxis_title="People involved")
                st.plotly_chart(style_fig(fig, title="Age Distribution by Band"), use_container_width=True)
            else:
                st.info("No recognizable age column found in the demographics file.")

        with c2:
            if DEMO_GENDER_AVAILABLE:
                gender_counts = demo["_GENDER"].value_counts()
                fig = go.Figure(go.Pie(
                    labels=gender_counts.index, values=gender_counts.values, hole=0.55,
                    marker=dict(colors=[GENDER_COLORS.get(g, "#B0BEC5") for g in gender_counts.index]),
                    textinfo="label+percent",
                ))
                st.plotly_chart(style_fig(fig, title="Gender Breakdown"), use_container_width=True)
            else:
                st.info("No recognizable gender column found in the demographics file.")

        if DEMO_AGE_AVAILABLE and has_mode:
            sev_col = find_col(demo, ["S4_CRASH_SEVERITY", "SEVERITY", "INJURY_SEVERITY"])
            if sev_col:
                st.markdown("##### Age Distribution by Injury Severity, by Mode")
                st.caption(
                    "Shown as three separate charts (one per mode) rather than one combined "
                    "chart, since patterns like 'people in fatal crashes skew older' are easier "
                    "to see within a single mode than averaged across all three."
                )
                age_sev_cols = st.columns(3)
                for slot, mode in zip(age_sev_cols, MODES):
                    with slot:
                        sub = demo[(demo[demo_mode_col] == mode) & demo["_AGE"].notna() & demo[sev_col].notna()]
                        if len(sub) < 5:
                            st.info(f"Not enough {mode} records with age + severity in this filter.")
                            continue
                        fig = px.violin(
                            sub, x=sev_col, y="_AGE", color=sev_col, box=True, points=False,
                            color_discrete_map=SEVERITY_COLORS, category_orders={sev_col: SEVERITY_ORDER},
                        )
                        fig.update_layout(yaxis_title="Age", xaxis_title=None, showlegend=False)
                        st.plotly_chart(
                            style_fig(fig, title=f"{mode} (n={len(sub):,})", height=340),
                            use_container_width=True,
                        )

            if sev_col and DEMO_GENDER_AVAILABLE:
                gm = demo.groupby([demo_mode_col, "_GENDER"], observed=True).size().reset_index(name="count")
                gm["pct"] = gm["count"] / gm.groupby(demo_mode_col)["count"].transform("sum") * 100
                fig = px.bar(
                    gm, x=demo_mode_col, y="pct", color="_GENDER",
                    color_discrete_map=GENDER_COLORS, category_orders={demo_mode_col: MODES},
                )
                fig.update_layout(yaxis_title="% of people", xaxis_title=None, barmode="stack")
                st.plotly_chart(style_fig(fig, title="Gender Mix by Mode"), use_container_width=True)

        st.caption(
            f"Demographics reflect **{len(demo):,}** person-level records "
            f"matched to the currently filtered crashes."
        )

    render_pipeline_figures("tab6")

# ---------------------------------------------------------------------------
# TAB 7 -- NARRATIVE TEXT MINING, CRASH TYPING, QWEN CLASSIFICATION, HOTSPOTS
#   Built live from the currently filtered `df` (+ narrative_text_export.csv
#   / spatiotemporal_hotspots_by_mode.csv when loaded) so every chart here
#   responds to the sidebar filters and mode selection, instead of being a
#   fixed PNG pulled from results/figures/.
# ---------------------------------------------------------------------------
with tab7:
    st.markdown("## Crash Typing")
    if "CRASH_GROUP" in df.columns and df["CRASH_GROUP"].notna().any():
        top_n = st.slider("Number of crash-type groups to show", 5, 20, 12, key="typing_topn")
        grp_totals = df["CRASH_GROUP"].value_counts().nlargest(top_n).index
        gsub = df[df["CRASH_GROUP"].isin(grp_totals)]
        gm = gsub.groupby(["CRASH_GROUP", "MODE"], observed=True).size().reset_index(name="count")
        fig = px.bar(
            gm, y="CRASH_GROUP", x="count", color="MODE", orientation="h",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        fig.update_layout(
            yaxis_title=None, xaxis_title="Crashes",
            yaxis={"categoryorder": "total ascending"}, barmode="stack",
        )
        st.plotly_chart(
            style_fig(fig, title=f"Crash Scenario / Type (top {top_n}, by mode)", height=480),
            use_container_width=True,
        )
        st.caption(
            "Source: `bicycle_typing_20.csv` (`S4_CRASH_GROUP_DESCRIPTION`) -- describes "
            "what happened (e.g. who failed to yield), not road/environment conditions. "
            "Respects every sidebar filter, including the Mode selector."
        )
    else:
        st.info(
            "No `CRASH_GROUP` column in the loaded export. Re-run "
            "`eda_analysis_combined.py` to pick it up in `power_bi_export.csv`."
        )

    if "CRASH_TYPE_DESC" in df.columns and df["CRASH_TYPE_DESC"].notna().any():
        desc_top_n = st.slider("Number of crash-type descriptions to show", 5, 20, 12, key="typedesc_topn")
        desc_totals = df["CRASH_TYPE_DESC"].value_counts().nlargest(desc_top_n).index
        dsub = df[df["CRASH_TYPE_DESC"].isin(desc_totals)]
        dm2 = dsub.groupby(["CRASH_TYPE_DESC", "MODE"], observed=True).size().reset_index(name="count")
        fig = px.bar(
            dm2, y="CRASH_TYPE_DESC", x="count", color="MODE", orientation="h",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        fig.update_layout(
            yaxis_title=None, xaxis_title="Crashes",
            yaxis={"categoryorder": "total ascending"}, barmode="stack",
        )
        st.plotly_chart(
            style_fig(fig, title=f"Crash Type Description (top {desc_top_n}, by mode)", height=480),
            use_container_width=True,
        )
        st.caption(
            "Source: `bicycle_typing_20.csv` (`S4_CRASH_TYPE_DESCRIPTION`) -- the specific "
            "collision mechanics (e.g. right-hook, dooring, overtaking), narrower than the "
            "Crash Scenario / Type grouping above. Bicycle-typed crashes only."
        )

    st.markdown("---")
    st.markdown("## Contributing Factors")
    cf_cols_present = [c for c in ("ROAD_CIRCUMSTANCE", "ENVIRONMENT_CIRCUMSTANCE") if c in df.columns]
    if cf_cols_present and df[cf_cols_present].notna().any().any():
        cf_rows = []
        for c in cf_cols_present:
            src_label = "Road" if c == "ROAD_CIRCUMSTANCE" else "Environment"
            vc = df[c].dropna().value_counts()
            for factor, count in vc.items():
                cf_rows.append({"Factor": factor, "Count": count, "Source": src_label})
        cf_df = pd.DataFrame(cf_rows)
        cf_top_n = st.slider("Number of contributing factors to show", 5, 20, 12, key="contrib_topn")
        top_factors = cf_df.groupby("Factor")["Count"].sum().nlargest(cf_top_n).index
        cf_sub = cf_df[cf_df["Factor"].isin(top_factors)]
        fig = px.bar(
            cf_sub, y="Factor", x="Count", color="Source", orientation="h",
            color_discrete_sequence=["#42A5F5", "#FFA726"],
        )
        fig.update_layout(
            yaxis_title=None, xaxis_title="Crashes",
            yaxis={"categoryorder": "total ascending"}, barmode="stack",
        )
        st.plotly_chart(
            style_fig(fig, title=f"Top {cf_top_n} Contributing Factors -- All Active Modes", height=480),
            use_container_width=True,
        )
        st.caption(
            "Source: `crash_event.csv` (`ROAD_CIRCUMSTANCES_1` + `ENVIRONMENT_CIRCUMSTANCES_1`) -- "
            "road/environment CONDITIONS present at the crash, covering Bicycle + E-Bike + "
            "E-Scooter. Different from the crash-typing charts above, which show what happened "
            "rather than the conditions it happened under. Respects every sidebar filter."
        )
    else:
        st.info(
            "No `ROAD_CIRCUMSTANCE`/`ENVIRONMENT_CIRCUMSTANCE` columns in the loaded export. "
            "Re-run `eda_analysis_combined.py` to pick them up in `power_bi_export.csv`."
        )

    st.markdown("---")
    st.markdown("## Qwen Narrative Classification")
    if "IN_QWEN_NARRATIVES" in df.columns:
        qdf = df[df["IN_QWEN_NARRATIVES"] == True].copy()  # noqa: E712
        st.caption(
            f"**{len(qdf):,}** of the **{total:,}** currently filtered crashes have a "
            f"Signal4Data narrative that was run through the Qwen classifier "
            f"({len(qdf) / total * 100:.1f}%). The rest come from the S4_Crash_bicycle "
            f"population directly, or from Signal4Data crashes with no matched narrative."
        )
        if len(qdf) and "QWEN_CLASS" in qdf.columns:
            qc1, qc2 = st.columns(2)
            with qc1:
                qcounts = qdf["QWEN_CLASS"].value_counts()
                fig = go.Figure(go.Pie(
                    labels=qcounts.index, values=qcounts.values, hole=0.5,
                    textinfo="label+percent",
                ))
                st.plotly_chart(style_fig(fig, title="Qwen Raw Classification"), use_container_width=True)
            with qc2:
                cross = qdf.groupby(["QWEN_CLASS", "MODE"], observed=True).size().reset_index(name="count")
                pivot = cross.pivot(index="QWEN_CLASS", columns="MODE", values="count").fillna(0)
                fig = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns, y=pivot.index,
                    colorscale="Blues", colorbar=dict(title="Crashes"),
                ))
                st.plotly_chart(
                    style_fig(fig, title="Qwen Raw Class vs. Final Mode"), use_container_width=True
                )
                st.caption(
                    "Final Mode can differ from the raw Qwen label -- e.g. a Qwen "
                    "'Bicyclist' call gets overridden to E-Bike/E-Scooter if that "
                    "REPORT_NUMBER's S4_Crash_bicycle row was overridden (see About tab)."
                )
        elif len(qdf) == 0:
            st.info("No narrative-classified crashes in the current filter selection.")
    else:
        st.info(
            "No `QWEN_CLASS`/`IN_QWEN_NARRATIVES` columns in the loaded export. "
            "Re-run `eda_analysis_combined.py` to pick them up."
        )

    st.markdown("---")
    st.markdown("## Narrative Text Mining")
    if narrative_raw is not None and "NARRATIVE_TEXT" in narrative_raw.columns:
        filtered_rns = set(df[MAIN_CRASH_ID_COL].astype(str)) if MAIN_CRASH_ID_COL else None
        ntext = narrative_raw.copy()
        if filtered_rns is not None:
            ntext = ntext[ntext["REPORT_NUMBER"].isin(filtered_rns)]
        text_mode_col = "MODE" if "MODE" in ntext.columns else ("QWEN_MODE" if "QWEN_MODE" in ntext.columns else None)
        if text_mode_col:
            ntext = ntext[ntext[text_mode_col].isin(sel_modes)]

        st.caption(
            f"**{len(ntext):,}** narratives match the current sidebar filters "
            f"(Mode, Year, Severity, etc. all apply here too)."
        )

        if len(ntext):
            default_keywords = "phone,texting,helmet,alcohol,dark,sidewalk,crosswalk,wrong way,speeding,failed to yield,intoxicated,fled"
            kw_input = st.text_input(
                "Keywords to search (comma-separated) -- edit freely and the chart updates live",
                value=default_keywords, key="keyword_search_input",
            )
            keywords = [k.strip().lower() for k in kw_input.split(",") if k.strip()]

            if keywords and text_mode_col:
                rows = []
                for m in [mm for mm in MODES if mm in ntext[text_mode_col].unique()]:
                    sub_txt = ntext[ntext[text_mode_col] == m]["NARRATIVE_TEXT"]
                    n = len(sub_txt)
                    for kw in keywords:
                        pct = sub_txt.str.contains(re.escape(kw), case=False, na=False).mean() * 100 if n else 0
                        rows.append({"Mode": m, "Keyword": kw, "Pct": pct, "N": n})
                kdf = pd.DataFrame(rows)
                if len(kdf):
                    pivot = kdf.pivot(index="Keyword", columns="Mode", values="Pct").reindex(
                        columns=[m for m in MODES if m in kdf["Mode"].unique()]
                    )
                    fig = go.Figure(go.Heatmap(
                        z=pivot.values, x=pivot.columns, y=pivot.index,
                        colorscale="YlOrRd", colorbar=dict(title="% of narratives"),
                        text=np.round(pivot.values, 1), texttemplate="%{text}",
                    ))
                    st.plotly_chart(
                        style_fig(fig, title="Keyword Mentions (% of narratives) by Mode", height=max(320, 34 * len(keywords))),
                        use_container_width=True,
                    )

            st.markdown("#### Top words by mode")
            st.caption("Reflects the Mode filter in the sidebar.")
            STOPWORDS = set((
                "the a an and or of to in on at for with was were is are be been being this that "
                "it its he she they them his her their who was driver vehicle crash report "
                "not no did do does had have has as by from into out up down "
                "1 2 3 4 5 6 7 8 9 0"
            ).split())
            word_modes = [m for m in MODES if text_mode_col and m in ntext[text_mode_col].unique()]
            if word_modes:
                word_cols = st.columns(len(word_modes))
                for wmode, wcol in zip(word_modes, word_cols):
                    sub_txt = ntext[ntext[text_mode_col] == wmode]["NARRATIVE_TEXT"]
                    words = re.findall(r"[a-z']{3,}", " ".join(sub_txt.tolist()))
                    words = [w for w in words if w not in STOPWORDS]
                    top_words = Counter(words).most_common(20)
                    with wcol:
                        if top_words:
                            wdf = pd.DataFrame(top_words, columns=["word", "count"])
                            fig = px.bar(wdf.sort_values("count"), x="count", y="word", orientation="h")
                            fig.update_layout(yaxis_title=None, xaxis_title="Mentions")
                            st.plotly_chart(
                                style_fig(fig, title=f"Top 20 Words -- {wmode} (n={len(sub_txt):,})", height=460),
                                use_container_width=True,
                            )
                        else:
                            st.info(f"No words for {wmode}.")
        else:
            st.info("No narratives match the current filter selection.")
    else:
        st.markdown(
            f"""<div class="section-note">
            No <code>{DEFAULT_NARRATIVE_PATH}</code> loaded, so the interactive keyword
            tool isn't available -- add it under the <b>Data Source</b> panel in the
            sidebar. It's produced by <code>eda_analysis_combined.py</code> alongside
            <code>power_bi_export.csv</code>.
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("## Spatiotemporal Hotspot Clusters")
    if hotspot_raw is not None and "MODE" in hotspot_raw.columns:
        hs = hotspot_raw[hotspot_raw["MODE"].isin(sel_modes)].copy()
        emerging_only = st.checkbox("Emerging clusters only (late-period growth)", value=False, key="hotspot_emerging_only")
        if emerging_only and "EMERGING" in hs.columns:
            hs = hs[hs["EMERGING"] == True]  # noqa: E712
        st.caption(
            f"**{len(hs):,}** clusters match the Mode filter in the sidebar "
            f"(hotspot table isn't affected by Year/Severity/etc. filters -- it's "
            f"precomputed per mode over the full time range)."
        )
        if len(hs) and {"CENTER_LAT", "CENTER_LON"}.issubset(hs.columns):
            fig = px.scatter_mapbox(
                hs, lat="CENTER_LAT", lon="CENTER_LON", color="MODE",
                size="N_CRASHES", size_max=28,
                color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
                hover_data=["CLUSTER_ID", "N_CRASHES", "N_EARLY_PERIOD", "N_LATE_PERIOD", "GROWTH_RATIO"],
                zoom=5.4, height=560,
            )
            fig = style_fig(fig, height=560, title="Cluster Centers (bubble size = crashes in cluster)")
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=56, b=0))
            st.plotly_chart(fig, use_container_width=True)

            sort_col = "GROWTH_RATIO" if "GROWTH_RATIO" in hs.columns else hs.columns[0]
            st.dataframe(
                hs.sort_values(sort_col, ascending=False),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No clusters match the current Mode selection.")
        st.caption(
            "Exploratory DBSCAN clustering (see `eda_analysis_combined.py` section 09d) -- "
            "not a validated hotspot-detection pipeline. 'Emerging' = late-period count "
            "> 1.5x early-period count, with at least 5 late-period crashes."
        )
    else:
        st.markdown(
            f"""<div class="section-note">
            No <code>{DEFAULT_HOTSPOT_PATH}</code> loaded, so the interactive hotspot
            explorer isn't available -- add it under the <b>Data Source</b> panel in the
            sidebar.
            </div>""",
            unsafe_allow_html=True,
        )

    render_pipeline_figures("tab7")


# ---------------------------------------------------------------------------
# TAB 8 -- CRASH CAUSATION (LLM narrative classification)
# ---------------------------------------------------------------------------
with tab8:
    st.markdown("## Crash Causation (LLM Narrative Classification)")
    st.markdown(
        """<div class="section-note">
        Every crash here was read by an LLM classifier and assigned a
        <code>primary_cause</code>, an <code>infrastructure_type</code> (where the
        rider was at the moment of impact), and whether <code>speed_contributing</code>
        played a role -- the closest thing in this dataset to "why did this crash
        actually happen," as opposed to what conditions surrounded it. Source:
        <code>multilabel_RegBike_cause.xlsx</code> + <code>multilabel_ebike_cause.xlsx</code>.
        </div>""",
        unsafe_allow_html=True,
    )

    if cause_raw is None or "primary_cause" not in cause_raw.columns:
        st.markdown(
            f"""<div class="section-note">
            No <code>{DEFAULT_CAUSE_PATH}</code> loaded, so the causation
            breakdown isn't available -- add it under the <b>Data Source</b> panel
            in the sidebar.
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        cdf = cause_raw[cause_raw["MODE"].isin(sel_modes)].copy()
        if MAIN_CRASH_ID_COL:
            filtered_rns = set(df[MAIN_CRASH_ID_COL].astype(str))
            cdf = cdf[cdf["REPORT_NUMBER"].isin(filtered_rns)]

        st.caption(
            f"**{len(cdf):,}** narrative-classified crashes match the current sidebar "
            f"filters (Mode, Year, Severity, County, etc. all apply here too)."
        )

        if len(cdf) == 0:
            st.info("No causation-classified crashes in the current filter selection.")
        else:
            present_modes = [m for m in MODES if m in cdf["MODE"].unique()]

            def insight(text):
                st.markdown(
                    f"""<div class="section-note">\U0001F4A1 <b>Key insight:</b> {text}</div>""",
                    unsafe_allow_html=True,
                )

            # ================================================================
            # 1. Fault attribution -- driver vs. non-motorist vs. ambiguous
            # ================================================================
            st.markdown("### 1. Fault Attribution")
            att = cdf.groupby(["MODE", "ATTRIBUTION"], observed=True).size().reset_index(name="count")
            att["pct"] = att.groupby("MODE")["count"].transform(lambda s: s / s.sum() * 100)
            fig = px.bar(
                att, x="MODE", y="pct", color="ATTRIBUTION", barmode="stack",
                category_orders={"MODE": present_modes, "ATTRIBUTION": list(ATTRIBUTION_COLORS.keys())},
                color_discrete_map=ATTRIBUTION_COLORS,
                text=att["pct"].round(1).astype(str) + "%",
            )
            fig.update_layout(yaxis_title="% of narrative-classified crashes", xaxis_title=None)
            st.plotly_chart(style_fig(fig, title="Fault Attribution by Mode"), use_container_width=True)
            st.caption(
                "Driver-attributable: failed to yield turning, ran stop/red light, following "
                "too close, distracted, speeding/reckless, impaired, dooring. Non-motorist-"
                "attributable: failed to yield entering roadway, ran stop/signal, wrong-way "
                "riding, sidewalk-driveway conflict."
            )
            overall_att = cdf["ATTRIBUTION"].value_counts(normalize=True) * 100
            top2 = cdf["CAUSE_LABEL"].value_counts(normalize=True).nlargest(2) * 100
            insight(
                f"Across the current selection, fault splits nearly evenly: "
                f"{overall_att.get('Driver-attributable', 0):.1f}% driver-attributable vs. "
                f"{overall_att.get('Non-motorist-attributable', 0):.1f}% non-motorist-attributable. "
                f"Just two categories -- <b>{top2.index[0]}</b> ({top2.iloc[0]:.1f}%) and "
                f"<b>{top2.index[1] if len(top2) > 1 else ''}</b> "
                f"({top2.iloc[1] if len(top2) > 1 else 0:.1f}%) -- account for "
                f"{top2.sum():.1f}% of all classified crashes. This is overwhelmingly a "
                f"yielding problem at points where paths cross, not a diverse mix of failure modes."
            )

            # ================================================================
            # 2. Where crashes happen
            # ================================================================
            st.markdown("### 2. Where the Rider Was at Impact")
            im = cdf.groupby(["INFRA_LABEL", "MODE"], observed=True).size().reset_index(name="count")
            fig = px.bar(
                im, y="INFRA_LABEL", x="count", color="MODE", orientation="h",
                color_discrete_map=MODE_COLORS, category_orders={"MODE": present_modes},
            )
            fig.update_layout(
                yaxis_title=None, xaxis_title="Crashes",
                yaxis={"categoryorder": "total ascending"}, barmode="stack",
            )
            st.plotly_chart(
                style_fig(fig, title="Infrastructure Type at Impact, by Mode", height=440),
                use_container_width=True,
            )
            infra_pct = cdf["INFRA_LABEL"].value_counts(normalize=True) * 100
            off_road = infra_pct.get("Sidewalk", 0) + infra_pct.get("Crosswalk", 0)
            bike_lane_pct = infra_pct.get("Bike lane", 0)
            insight(
                f"Sidewalk + crosswalk together account for <b>{off_road:.1f}%</b> of "
                f"narrative-classified crashes in the current selection -- roughly half of all "
                f"crashes happen where the rider wasn't in the road at all. Only "
                f"<b>{bike_lane_pct:.1f}%</b> happened in a dedicated bike lane, despite how "
                f"much infrastructure conversation centers on bike lanes specifically."
            )

            # ================================================================
            # 3. Cause x location interaction (the "right hook" pattern)
            # ================================================================
            st.markdown("### 3. Primary Cause by Location -- Infrastructure Doesn't Remove the Yielding Problem, It Relocates It")
            mode_label = " + ".join(present_modes)
            cross = cdf.groupby(["INFRA_LABEL", "CAUSE_LABEL"], observed=True).size().reset_index(name="count")
            pivot = cross.pivot(index="INFRA_LABEL", columns="CAUSE_LABEL", values="count").fillna(0)
            top_cause_cols = cdf["CAUSE_LABEL"].value_counts().nlargest(8).index
            pivot_cols = [c for c in top_cause_cols if c in pivot.columns]
            pivot_pct = pivot[pivot_cols].div(pivot[pivot_cols].sum(axis=1).replace(0, 1), axis=0) * 100
            fig = go.Figure(go.Heatmap(
                z=pivot_pct.values, x=pivot_pct.columns, y=pivot_pct.index,
                colorscale="Reds", colorbar=dict(title="% of that location's crashes"),
                text=np.round(pivot_pct.values, 1), texttemplate="%{text}",
            ))
            st.plotly_chart(
                style_fig(fig, title=f"Primary Cause x Location -- {mode_label} (row %)", height=420),
                use_container_width=True,
            )
            bike_lane_row = pivot_pct.loc["Bike lane"] if "Bike lane" in pivot_pct.index else None
            if bike_lane_row is not None and len(bike_lane_row):
                top_bl_cause = bike_lane_row.idxmax()
                insight(
                    f"For the current mode selection ({mode_label}), bike-lane crashes are "
                    f"dominated by <b>{top_bl_cause}</b> ({bike_lane_row.max():.1f}% of bike-lane "
                    f"crashes) -- a dedicated bike lane doesn't remove the 'driver turns across "
                    f"the rider's path' problem, it's often where that pattern is most "
                    f"concentrated (a 'right-hook' at intersections, since the lane puts the "
                    f"rider exactly where a right-turning driver is most likely to miss them)."
                )

            # ================================================================
            # 4. Sidewalk riding is a driveway problem
            # ================================================================
            st.markdown("### 4. Sidewalk Crashes: a Driveway Problem More Than a Road-Crossing Problem")
            sw = cdf[cdf["INFRA_LABEL"] == "Sidewalk"]
            if len(sw):
                sw_causes = sw["CAUSE_LABEL"].value_counts(normalize=True).nlargest(6) * 100
                fig = px.bar(
                    x=sw_causes.values, y=sw_causes.index, orientation="h",
                    color_discrete_sequence=["#5C6BC0"],
                )
                fig.update_layout(yaxis_title=None, xaxis_title="% of sidewalk crashes",
                                   yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(
                    style_fig(fig, title=f"Top Causes of Sidewalk Crashes ({', '.join(present_modes)})", height=340),
                    use_container_width=True,
                )
                dwc = sw["CAUSE_LABEL"].value_counts(normalize=True).get("Sidewalk driveway conflict", 0) * 100
                dft = sw["CAUSE_LABEL"].value_counts(normalize=True).get("Driver failed to yield turning", 0) * 100
                insight(
                    f"Sidewalk crashes split mainly between driveway conflicts "
                    f"({dwc:.1f}%) and drivers failing to yield while turning ({dft:.1f}%) -- "
                    f"together {dwc + dft:.1f}% of sidewalk crashes. The risk of sidewalk "
                    f"riding looks concentrated at the many driveway crossings a sidewalk route "
                    f"passes, not from riding along the sidewalk itself -- pointing toward "
                    f"driveway-crossing treatments as a more targeted fix than sidewalk-riding bans."
                )
            else:
                st.info("No sidewalk crashes in the current filter selection.")

            # ================================================================
            # 5. Speed as a documented factor
            # ================================================================
            st.markdown("### 5. Speed as a Documented Factor (Either Party)")
            sc = cdf.groupby(["MODE", "speed_contributing"], observed=True).size().reset_index(name="count")
            sc["pct"] = sc.groupby("MODE")["count"].transform(lambda s: s / s.sum() * 100)
            fig = px.bar(
                sc, x="MODE", y="pct", color="speed_contributing", barmode="stack",
                category_orders={"MODE": present_modes, "speed_contributing": ["yes", "unclear", "no"]},
                color_discrete_map={"yes": "#C0392B", "unclear": "#BDBDBD", "no": "#81C784"},
            )
            fig.update_layout(yaxis_title="% of narrative-classified crashes", xaxis_title=None)
            st.plotly_chart(style_fig(fig, title="Speed Flagged as Contributing (Driver or Rider), by Mode"), use_container_width=True)
            st.caption(
                "**Not micromobility-speed-specific.** This flags whether the narrative says "
                "*anyone's* speed -- the rider's or the driver's -- contributed to the crash; it "
                "isn't restricted to the micromobility rider. In a sample check, ~65% of 'yes' "
                "narratives referenced the rider's speed and ~38% referenced the driver's/"
                "vehicle's (some flag both). For rider-speed-only numbers, see the "
                "`MICROMOBILITY_SPEED_MPH` chart on the Insights tab (Section 4)."
            )
            yes_by_mode = cdf[cdf["speed_contributing"] == "yes"].groupby("MODE", observed=True).size() \
                / cdf.groupby("MODE", observed=True).size() * 100
            yes_text = "; ".join(f"{m}: {yes_by_mode.get(m, 0):.1f}%" for m in present_modes)
            insight(
                f"'Yes' rates by mode -- {yes_text}. Don't read the low overall rate as "
                f"'speed isn't a factor': officer narratives often just don't discuss speed "
                f"unless it was extreme or measured, so <b>unclear</b> (the largest bucket for "
                f"every mode) reflects a reporting gap, not evidence that speed was low."
            )

            # ================================================================
            # 6. Hit-and-run & wrong-way riding
            # ================================================================
            st.markdown("### 6. Hit-and-Run & Wrong-Way Riding")
            hr = cdf[cdf["primary_cause"] == "hit_and_run"]
            wwr = cdf[cdf["primary_cause"] == "wrong_way_riding"]
            hc1, hc2 = st.columns(2)
            with hc1:
                if len(hr):
                    hr_infra = hr["INFRA_LABEL"].value_counts(normalize=True) * 100
                    fig = px.bar(
                        x=hr_infra.values, y=hr_infra.index, orientation="h",
                        color_discrete_sequence=["#C0392B"],
                    )
                    fig.update_layout(yaxis_title=None, xaxis_title="% of hit-and-run crashes",
                                       yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(
                        style_fig(fig, title=f"Hit-and-Run Location (n={len(hr):,})", height=320),
                        use_container_width=True,
                    )
                else:
                    st.info("No hit-and-run crashes in the current filter selection.")
            with hc2:
                if len(wwr):
                    wwr_infra = wwr["INFRA_LABEL"].value_counts(normalize=True) * 100
                    fig = px.bar(
                        x=wwr_infra.values, y=wwr_infra.index, orientation="h",
                        color_discrete_sequence=["#FF9800"],
                    )
                    fig.update_layout(yaxis_title=None, xaxis_title="% of wrong-way-riding crashes",
                                       yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(
                        style_fig(fig, title=f"Wrong-Way Riding Location (n={len(wwr):,})", height=320),
                        use_container_width=True,
                    )
                else:
                    st.info("No wrong-way-riding crashes in the current filter selection.")
            insight(
                f"Hit-and-run ({len(hr):,} crashes, {len(hr) / len(cdf) * 100:.1f}% of the "
                f"current selection) is spread fairly evenly across location types -- a driver-"
                f"behavior issue independent of where the rider was, not a location-specific risk. "
                f"Wrong-way riding ({len(wwr):,} crashes, {len(wwr) / len(cdf) * 100:.1f}%) "
                f"concentrates in travel lanes and bike lanes rather than sidewalks -- mostly a "
                f"'riding against traffic in a facility meant for one-way travel' problem."
            )

            # ================================================================
            # Data quality notes
            # ================================================================
            st.markdown("### Data Quality Notes")
            noise = cause_raw[cause_raw["MODE"] == "Other"]
            cross_class = cause_raw.groupby("SOURCE_FILE")["MODE"].value_counts() if "SOURCE_FILE" in cause_raw.columns else None
            notes = [
                f"**Confidence is effectively capped** in the source classifier's output -- "
                f"not a strong signal for filtering 'high-quality' rows beyond excluding a small low tail.",
                f"**{len(noise):,} rows across both source files** landed as mode `Other` rather "
                f"than Bicycle/E-Bike/E-Scooter and are excluded from every chart on this tab.",
                "`cause_flag` is almost entirely empty in the source files and isn't a usable QA filter as-is.",
            ]
            for n in notes:
                st.markdown(f"- {n}")
            if cross_class is not None:
                with st.expander("Raw prediction counts by source file (mode-label cross-contamination)"):
                    st.dataframe(cross_class.unstack(fill_value=0), use_container_width=True)

    render_pipeline_figures("tab8")


# ---------------------------------------------------------------------------
# TAB 9 -- INSIGHTS (curated summary of mode_comparison_findings.md +
# mode_comparison_findings_supplement.md). These two reports were built from
# the pre-computed `tables/` pipeline and `mode_comparison_report.xlsx`, not
# from power_bi_export.csv directly -- so unlike every other tab, the charts
# here use the literal numbers from those reports rather than recomputing
# live off the sidebar filters. Treat this tab as a fixed summary, not an
# interactive query.
# ---------------------------------------------------------------------------
with tab9:
    st.markdown("## Key Insights: Bicycle vs. E-Bike vs. E-Scooter")
    st.markdown(
        """<div class="section-note">
        Curated from three analysis passes -- <code>mode_comparison_findings.md</code>
        (chi-square/Kruskal-Wallis over <code>power_bi_export.csv</code>, ranked by
        effect size), <code>mode_comparison_findings_supplement.md</code> (mined
        from the 40 pre-computed <code>tables/</code>), and the LLM narrative
        causation classification behind the <b>Crash Causation</b> tab. Sections 1-9
        below are <b>static</b> -- built from numbers already computed in the two
        reports, not re-filtered by the sidebar, since the underlying granular
        tables aren't part of the CSVs this dashboard loads. Section 10 is <b>live</b>
        and does respect the sidebar filters -- see the note above that section.
        </div>""",
        unsafe_allow_html=True,
    )

    def stat_card(col, label, value, sub, color):
        col.markdown(
            f"""<div class="kpi-card" style="border-left-color:{color};">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    def insight(text):
        st.markdown(f"""<div class="section-note">\U0001F4A1 <b>Why it matters:</b> {text}</div>""",
                     unsafe_allow_html=True)

    # ================================================================
    # Headline stat cards
    # ================================================================
    _mode_n_hdr = {m: int((df["MODE"] == m).sum()) for m in MODES}
    _fatal_n_hdr = {m: int(((df["MODE"] == m) & (df["S4_CRASH_SEVERITY"] == "Fatality")).sum()) for m in MODES}
    _fatal_rate_hdr = {m: (_fatal_n_hdr[m] / _mode_n_hdr[m] if _mode_n_hdr[m] else np.nan) for m in MODES}
    _bike_rate_hdr = _fatal_rate_hdr.get("Bicycle", np.nan)
    _hdr_ratio = max(
        (_fatal_rate_hdr.get("E-Bike", np.nan) / _bike_rate_hdr) if _bike_rate_hdr else np.nan,
        (_fatal_rate_hdr.get("E-Scooter", np.nan) / _bike_rate_hdr) if _bike_rate_hdr else np.nan,
    ) if _bike_rate_hdr else np.nan

    s1, s2, s3, s4 = st.columns(4)
    stat_card(
        s1, "FATALITY RISK PER CRASH",
        f"~{_hdr_ratio:.1f}x higher" if pd.notna(_hdr_ratio) else "~7x higher (static)",
        "E-bike/e-scooter vs. bicycle, live for current filters -- see Section 1 below", "#B71C1C",
    )
    stat_card(s2, "E-SCOOTER CRASHES", "50.2% pedestrian-involved",
              "vs. 13.9% bicycle -- ~3.6x higher", "#FF9800")
    stat_card(s3, "CRASH GROWTH, 2014-2025", "~200x / ~59x",
              "E-bike / e-scooter, while bicycle was +15%", "#4CAF50")
    stat_card(s4, "MEDIAN CRASH SPEED", "18.0 mph e-bike",
              "vs. 12.5 mph bicycle (+44%)", "#2196F3")

    st.write("")
    st.markdown("---")

    # ================================================================
    # 1. Fatality risk -- LIVE (recomputed from the currently loaded/
    # filtered df, using the exact same S4_CRASH_SEVERITY == "Fatality"
    # definition as the Fatal & Serious Injury Rate chart on the Severity &
    # Outcomes tab). This used to be a hardcoded [1.99, 14.17, 13.27]
    # snapshot from the static mode_comparison_findings.md report, computed
    # once off a full/unfiltered power_bi_export.csv at a different time.
    # That's why it could show a starker ratio than the Severity & Outcomes
    # tab: two different denominators (the static report's snapshot of the
    # full dataset vs. whatever the sidebar filters happen to be) computed
    # months apart, not two different underlying facts. Recomputing it live
    # off the same df as tab2 removes that mismatch entirely.
    # ================================================================
    st.markdown("### 1. Fatality Risk Per Crash")
    st.markdown(
        """<div class="section-note">
        \u26A0\uFE0F <b>This card was previously static</b> (hardcoded
        1.99 / 14.17 / 13.27 fatalities-per-1,000 from an earlier offline
        report) and could disagree with the <b>Severity & Outcomes</b> tab's
        "Fatal & Serious Injury Rate" chart, which has always computed live
        off the currently loaded/filtered data. That was the source of the
        "why is it 7x here but not there" gap -- two snapshots of the data
        taken at different times/filters, not two different findings. This
        card now recomputes live from the same <code>S4_CRASH_SEVERITY</code>
        field and the current sidebar filters, so it will always match tab 2.
        </div>""",
        unsafe_allow_html=True,
    )
    mode_n_s1 = {m: int((df["MODE"] == m).sum()) for m in MODES}
    fatal_n_s1 = {m: int(((df["MODE"] == m) & (df["S4_CRASH_SEVERITY"] == "Fatality")).sum()) for m in MODES}
    fatal_per_1k = {m: (fatal_n_s1[m] / mode_n_s1[m] * 1000 if mode_n_s1[m] else np.nan) for m in MODES}
    fatal_df = pd.DataFrame({
        "MODE": MODES,
        "Fatalities per 1,000 crashes": [fatal_per_1k[m] for m in MODES],
    })
    fig = px.bar(
        fatal_df, x="MODE", y="Fatalities per 1,000 crashes", color="MODE",
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
    )
    fig.update_traces(
        text=[f"{fatal_per_1k[m]:.2f} (n={fatal_n_s1[m]:,}/{mode_n_s1[m]:,})" for m in MODES],
        textposition="outside",
    )
    fig.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(
        style_fig(fig, title="Fatalities per 1,000 Crashes, by Mode (live)", n=mode_n_s1),
        use_container_width=True,
    )
    bike_rate = fatal_per_1k.get("Bicycle", np.nan)
    ebike_ratio = fatal_per_1k.get("E-Bike", np.nan) / bike_rate if bike_rate else np.nan
    escoot_ratio = fatal_per_1k.get("E-Scooter", np.nan) / bike_rate if bike_rate else np.nan
    insight(
        f"With the current sidebar filters, a reported e-bike crash is "
        f"~{ebike_ratio:.1f}x as likely to be fatal as a bicycle crash, and e-scooter is "
        f"~{escoot_ratio:.1f}x. In the earlier static full-dataset snapshot these ratios were "
        f"~7.1x and ~6.7x -- the two won't always match exactly since this now respects Mode/"
        f"Year/Severity filters, but they should tell the same directional story as the "
        f"Severity & Outcomes tab's Fatal & Serious Injury Rate chart. Note the small "
        f"fatality counts per mode above (n=) -- these are rates over rare events, so treat "
        f"single-digit or low-double-digit numerators as noisy, especially once you narrow "
        f"the filters."
    )

    # ================================================================
    # 2. Crash type -- who e-scooters actually collide with
    # ================================================================
    st.markdown("### 2. Crash Type: E-Scooters Collide With Pedestrians, Not Vehicles")
    ct_df = pd.DataFrame({
        "MODE": MODES * 3,
        "Crash Type": ["Pedestrian-involved"] * 3 + ["Single Vehicle"] * 3 + ["Bicycle-type collision"] * 3,
        "Pct": [13.9, 17.2, 50.2, 22.9, 19.7, 28.0, 61.2, 60.6, 20.1],
    })
    fig = px.bar(
        ct_df, x="MODE", y="Pct", color="Crash Type", barmode="group",
        category_orders={"MODE": MODES}, color_discrete_sequence=["#EF5350", "#FFA726", "#5C6BC0"],
    )
    fig.update_layout(yaxis_title="% of that mode's crashes", xaxis_title=None)
    st.plotly_chart(style_fig(fig, title="Crash Type by Mode"), use_container_width=True)
    insight(
        "E-scooter crashes are pedestrian-involved half the time -- roughly 3.6x the bicycle "
        "rate. This is the strongest real signal in the original mode-comparison analysis and "
        "points to riders operating on sidewalks/shared paths rather than the roadway, colliding "
        "with pedestrians instead of vehicles. A bicycle-oriented lane network doesn't fix this "
        "if scooter riders aren't using the roadway in the first place -- protected micromobility "
        "lanes separated from both traffic <i>and</i> pedestrians, plus sidewalk-riding "
        "enforcement, are the levers this points to specifically for e-scooters."
    )

    # ================================================================
    # 3. Growth trajectory
    # ================================================================
    st.markdown("### 3. Growth Trajectory: The Clearest \"Why Now\" Argument")
    growth_df = pd.DataFrame({
        "Year": [2014, 2019, 2022, 2025] * 3,
        "MODE": ["Bicycle"] * 4 + ["E-Bike"] * 4 + ["E-Scooter"] * 4,
        "Crashes": [8121, 7847, 8181, 9329, 11, 60, 351, 2207, 27, 100, 376, 1589],
    })
    fig = px.line(
        growth_df, x="Year", y="Crashes", color="MODE", markers=True,
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES}, log_y=True,
    )
    fig.update_layout(yaxis_title="Crashes per year (log scale)", xaxis_title=None)
    st.plotly_chart(style_fig(fig, title="Crashes by Year, by Mode"), use_container_width=True)
    insight(
        "E-bike crashes grew ~200x from 2014 to 2025 (11 &rarr; 2,207); e-scooter grew ~59x "
        "(27 &rarr; 1,589). Bicycle crashes were essentially flat (+15% over 11 years). The "
        "population of at-risk riders exploded in a decade where bicycle crash counts barely "
        "moved -- the strongest \"policy/infrastructure hasn't caught up\" argument in the data. "
        "Caveat: this tracks device adoption, not a change in relative riskiness -- a true rate "
        "comparison (crashes per rider/trip/registered device) would need an exposure "
        "denominator this dataset doesn't have."
    )

    # ================================================================
    # 4. Speed
    # ================================================================
    st.markdown("### 4. Speed: E-Bikes and E-Scooters Run Faster Than Pedal Bikes")
    sp1, sp2 = st.columns(2)
    with sp1:
        speed_a = pd.DataFrame({"MODE": MODES, "Median mph": [12.5, 18.0, 15.0], "n": [51, 540, 184]})
        fig = px.bar(speed_a, x="MODE", y="Median mph", color="MODE", color_discrete_map=MODE_COLORS,
                     category_orders={"MODE": MODES}, text="Median mph")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(
            style_fig(fig, title="Median Crash Speed (extraction run 1)", height=340), use_container_width=True
        )
        st.caption("n=51 / 540 / 184 -- from `mode_comparison_findings.md`.")
    with sp2:
        speed_b = pd.DataFrame({"MODE": MODES, "Mean mph": [14.4, 17.9, 16.3], "n": [101, 582, 210]})
        fig = px.bar(speed_b, x="MODE", y="Mean mph", color="MODE", color_discrete_map=MODE_COLORS,
                     category_orders={"MODE": MODES}, text="Mean mph")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(
            style_fig(fig, title="Mean Crash Speed (extraction run 2)", height=340), use_container_width=True
        )
        st.caption("n=101 / 582 / 210 -- from `mode_comparison_findings_supplement.md`.")
    insight(
        "Both independent narrative-text speed extractions agree on direction and rough "
        "magnitude: e-bikes crash at meaningfully higher speed than pedal bikes (+44% median in "
        "run 1), e-scooters land in between. Speed is only recorded/extractable in a small "
        "fraction of narratives (<1% of all crashes have a populated <code>MICROMOBILITY_SPEED_MPH</code> "
        "field) -- treat both charts as directional, not precise population estimates. Still, "
        "this is corroborated independently by the narrative-keyword <code>speed_related</code> "
        "mention rate (4.7% bicycle vs. <b>11.3% e-bike</b> vs. 7.2% e-scooter) and supports "
        "class-based e-bike speed regulation (the existing Class 1/2/3 framework)."
    )

    # ================================================================
    # 5. Age & gender
    # ================================================================
    st.markdown("### 5. Age & Gender Skew Younger and More Female for E-Scooter")
    ag1, ag2 = st.columns(2)
    with ag1:
        age_df = pd.DataFrame({"MODE": MODES, "Median age": [38, 33, 26]})
        fig = px.bar(age_df, x="MODE", y="Median age", color="MODE", color_discrete_map=MODE_COLORS,
                     category_orders={"MODE": MODES}, text="Median age")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(style_fig(fig, title="Median Rider Age", height=340), use_container_width=True)
    with ag2:
        gender_df = pd.DataFrame({"MODE": MODES, "Female share (%)": [19.5, 18.2, 31.1]})
        fig = px.bar(gender_df, x="MODE", y="Female share (%)", color="MODE", color_discrete_map=MODE_COLORS,
                     category_orders={"MODE": MODES}, text="Female share (%)")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(style_fig(fig, title="Female Rider Share", height=340), use_container_width=True)
    insight(
        "Median age drops from 38 (bicycle) to 33 (e-bike) to 26 (e-scooter), while female "
        "share rises from ~19% (bicycle/e-bike) to 31.1% (e-scooter). A bicycle-crash-history "
        "safety campaign will miss the actual e-scooter population, which skews younger with "
        "meaningfully more women riders -- age-targeted, campus/downtown-adjacent education is "
        "a better fit than a generic \"share the road\" campaign. Separately, riders in "
        "<b>incapacitating</b>-severity crashes specifically average just 30.7 years old for "
        "e-scooter vs. 43.2 (bicycle) and 41.4 (e-bike) -- e-scooter's most severe non-fatal "
        "crashes are disproportionately happening to young riders."
    )

    # ================================================================
    # 6. Citations & driver behavior
    # ================================================================
    st.markdown("### 6. Citation Rates Diverge Sharply at the Fatal Tier")
    cite_df = pd.DataFrame({
        "MODE": MODES * 3,
        "Tier": ["All crashes"] * 3 + ["KSI only"] * 3 + ["Fatal only"] * 3,
        "Pct": [33.7, 29.3, 31.0, 38.8, 32.4, 33.9, 22.6, 14.4, 22.9],
    })
    fig = px.bar(
        cite_df, x="Tier", y="Pct", color="MODE", barmode="group",
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES, "Tier": ["All crashes", "KSI only", "Fatal only"]},
    )
    fig.update_layout(yaxis_title="Driver citation rate (%)", xaxis_title=None)
    st.plotly_chart(style_fig(fig, title="Citation Rate by Severity Tier, by Mode"), use_container_width=True)

    behav_df = pd.DataFrame({
        "MODE": MODES * 2,
        "Flag": ["Alcohol-related driver"] * 3 + ["Drug-related driver"] * 3,
        "Pct": [7.7, 1.1, 9.6, 5.9, 2.1, 5.5],
    })
    fig = px.bar(
        behav_df, x="Flag", y="Pct", color="MODE", barmode="group",
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
    )
    fig.update_layout(yaxis_title="% of fatal crashes", xaxis_title=None)
    st.plotly_chart(
        style_fig(fig, title="Driver Impairment Flags in Fatal Crashes, by Mode", height=340),
        use_container_width=True,
    )
    insight(
        "Citation rates look similar across modes for all-crashes (29-34%) but diverge sharply "
        "at the fatal tier: in fatal e-bike crashes, the at-fault driver is cited only "
        "<b>14.4%</b> of the time -- roughly 8 points below both bicycle and e-scooter. N is "
        "small here (95 drivers in fatal e-bike crashes, 73 for e-scooter, 2,142 for bicycle), "
        "so treat as suggestive. Alcohol-related driver involvement in fatal crashes is highest "
        "for e-scooter (9.6%) and lowest for e-bike (1.1%) -- distraction, speeding, and "
        "aggressive-driving flags don't meaningfully differentiate modes at any severity tier."
    )

    # ================================================================
    # 7. Geographic concentration
    # ================================================================
    st.markdown("### 7. Geographic Concentration: Statewide vs. a Handful of Metro Corridors")
    geo1, geo2 = st.columns([3, 2])
    with geo1:
        cluster_df = pd.DataFrame({"MODE": MODES, "DBSCAN clusters (statewide)": [708, 4, 11]})
        fig = px.bar(cluster_df, x="MODE", y="DBSCAN clusters (statewide)", color="MODE",
                     color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
                     text="DBSCAN clusters (statewide)")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(style_fig(fig, title="Number of Spatiotemporal Crash Clusters, by Mode", height=360),
                         use_container_width=True)
    with geo2:
        st.markdown("**Largest known clusters**")
        st.dataframe(
            pd.DataFrame({
                "Mode": ["E-Scooter", "E-Scooter", "E-Bike", "E-Bike"],
                "Location": ["Miami Beach", "Downtown Miami / Brickell", "Key West", "Tampa / Clearwater"],
                "Crashes": [227, 160, 58, 34],
            }),
            hide_index=True, use_container_width=True,
        )
        st.caption("Miami-Dade + Broward = 43% of all e-scooter crashes statewide (vs. 23% for bicycle in those two counties).")
    insight(
        "Bicycle crashes spread across 708 spatiotemporal clusters statewide; e-scooter has "
        "only 11 and e-bike only 4 -- consistent with e-bike/e-scooter crashes concentrating in "
        "a handful of dense urban/tourist corridors (Miami Beach, downtown Miami, Key West, "
        "Tampa) rather than spreading statewide the way bicycle crashes do. Practically: "
        "e-scooter/e-bike infrastructure investment doesn't need to be a statewide program -- it "
        "can target a small number of identifiable metro corridors where shared-device fleets "
        "actually operate, a cheaper and more targeted ask than a statewide mandate. None of "
        "these clusters are flagged \"emerging\" -- likely too little early-period baseline data "
        "to compute growth off of yet, not evidence the hotspots have stabilized."
    )

    # ================================================================
    # 8. Narrative keyword themes
    # ================================================================
    st.markdown("### 8. Narrative Text Themes")
    kw_df = pd.DataFrame({
        "Keyword": ["speed_related", "sidewalk", "helmet", "failed_to_yield", "crosswalk", "hit_and_run"],
        "Bicycle": [4.7, 37.3, 2.8, 11.8, 25.9, 3.6],
        "E-Bike": [11.3, 45.3, 5.6, 8.7, 25.5, 4.0],
        "E-Scooter": [7.2, 40.0, 3.1, 10.4, 29.0, 5.1],
    })
    kw_melt = kw_df.melt(id_vars="Keyword", var_name="MODE", value_name="Pct")
    fig = go.Figure(go.Heatmap(
        z=kw_df[MODES].values, x=MODES, y=kw_df["Keyword"],
        colorscale="YlOrRd", colorbar=dict(title="% of narratives"),
        text=kw_df[MODES].values, texttemplate="%{text}",
    ))
    st.plotly_chart(style_fig(fig, title="Keyword Mention Rate (% of narratives), by Mode", height=360),
                     use_container_width=True)
    insight(
        "<code>sidewalk</code> is mentioned often across <i>all three</i> modes (37-45%), not "
        "just e-scooter -- worth reading against Finding 2 above: the crash-<i>type</i> "
        "classification shows a sharp e-scooter-specific skew toward pedestrian collisions, but "
        "sidewalk <i>mentions</i> are common everywhere. Bicycles may ride on sidewalks about as "
        "often but collide with vehicles/other bikes rather than pedestrians. "
        "<code>helmet</code> mentions run ~2x higher for e-bike than bicycle/e-scooter -- unclear "
        "if that's a real usage difference or just reporting emphasis. <code>hit_and_run</code> is "
        "highest for e-scooter (5.1%). Caveat: this is regex keyword matching, not NLP "
        "classification -- a lead generator for which narratives to read, not a precise rate."
    )

    # ================================================================
    # 9. Road infrastructure context
    # ================================================================
    st.markdown("### 9. Road Infrastructure Context")
    ri1, ri2, ri3 = st.columns(3)
    with ri1:
        ow_df = pd.DataFrame({"MODE": MODES, "One-way street crashes (%)": [4.1, 3.9, 6.1]})
        fig = px.bar(ow_df, x="MODE", y="One-way street crashes (%)", color="MODE",
                     color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
                     text="One-way street crashes (%)")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(style_fig(fig, title="One-Way Street Crashes", height=320), use_container_width=True)
    with ri2:
        ic_df = pd.DataFrame({
            "MODE": MODES * 2,
            "Control": ["Stop-controlled"] * 3 + ["Signalized"] * 3,
            "Pct": [38.2, 41.7, 41.8, 60.9, 57.5, 57.9],
        })
        fig = px.bar(ic_df, x="MODE", y="Pct", color="Control", barmode="stack",
                     category_orders={"MODE": MODES}, color_discrete_sequence=["#FFA726", "#5C6BC0"])
        fig.update_layout(yaxis_title="%", xaxis_title=None)
        st.plotly_chart(style_fig(fig, title="Intersection Control Type", height=320), use_container_width=True)
    with ri3:
        lc_df = pd.DataFrame({"MODE": MODES, "Dark, unlit (%)": [5.4, 3.6, 3.1]})
        fig = px.bar(lc_df, x="MODE", y="Dark, unlit (%)", color="MODE",
                     color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
                     text="Dark, unlit (%)")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(style_fig(fig, title="Dark, Unlit-Road Crashes", height=320), use_container_width=True)
    insight(
        "E-scooter crashes happen on one-way streets at 6.1% vs. 4.1% (bicycle) / 3.9% (e-bike) "
        "-- about 50% higher, consistent with scooter-share concentration in dense downtown "
        "grids. E-bike/e-scooter crashes also skew slightly more toward stop-controlled "
        "intersections (41.7-41.8%) vs. bicycle (38.2%), with the difference mirrored in "
        "signalized-intersection share. Bicycle has the highest share of crashes in dark, unlit "
        "conditions (5.4% vs. 3.1-3.6%) -- likely reflects usage patterns (utility/commuting "
        "bicycling at night vs. daytime e-bike/scooter trips) more than a road-design effect. "
        "Weather shows no meaningful mode differences (all three ~85-87% clear-weather)."
    )

    # ================================================================
    # 10. Crash causation highlights (live, from the Crash Causation tab)
    # ================================================================
    st.markdown("### 10. Crash Causation Highlights")
    st.caption(
        "Unlike the sections above, this one is **live** -- computed from the same narrative-"
        "classified causation data as the Crash Causation tab, filtered to the modes currently "
        "selected in the sidebar. See that tab for the full breakdown."
    )
    if cause_raw is not None and "primary_cause" in cause_raw.columns:
        chdf = cause_raw[cause_raw["MODE"].isin(sel_modes)].copy()
        if MAIN_CRASH_ID_COL:
            chdf = chdf[chdf["REPORT_NUMBER"].isin(set(df[MAIN_CRASH_ID_COL].astype(str)))]
        if len(chdf):
            ch1, ch2 = st.columns(2)
            with ch1:
                att2 = chdf["ATTRIBUTION"].value_counts(normalize=True).reindex(
                    list(ATTRIBUTION_COLORS.keys())
                ).fillna(0) * 100
                fig = go.Figure(go.Pie(
                    labels=att2.index, values=att2.values, hole=0.5,
                    marker=dict(colors=[ATTRIBUTION_COLORS[k] for k in att2.index]),
                    textinfo="label+percent",
                ))
                st.plotly_chart(style_fig(fig, title="Fault Attribution (Current Selection)", height=340),
                                 use_container_width=True)
            with ch2:
                sc2 = chdf.groupby(["MODE", "speed_contributing"], observed=True).size().reset_index(name="count")
                sc2["pct"] = sc2.groupby("MODE")["count"].transform(lambda s: s / s.sum() * 100)
                fig = px.bar(
                    sc2[sc2["speed_contributing"] == "yes"], x="MODE", y="pct", color="MODE",
                    color_discrete_map=MODE_COLORS, text="pct",
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(showlegend=False, yaxis_title="% flagged speed-contributing", xaxis_title=None)
                st.plotly_chart(style_fig(fig, title="Speed Flagged as Contributing (Driver or Rider), by Mode", height=340),
                                 use_container_width=True)
                st.caption(
                    "Either party's speed, not rider-only -- see the caveat on the Crash "
                    "Causation tab. For rider-speed-specific numbers, see Section 4 above."
                )
            infra_pct2 = chdf["INFRA_LABEL"].value_counts(normalize=True) * 100
            bl = chdf[chdf["INFRA_LABEL"] == "Bike lane"]
            bl_top = bl["CAUSE_LABEL"].value_counts(normalize=True).nlargest(1) * 100 if len(bl) else None
            sw2 = chdf[chdf["INFRA_LABEL"] == "Sidewalk"]
            sw_combo = sw2["CAUSE_LABEL"].value_counts(normalize=True).reindex(
                ["Sidewalk driveway conflict", "Driver failed to yield turning"]
            ).sum() * 100 if len(sw2) else 0
            bl_text = (
                f"In bike lanes specifically, <b>{bl_top.index[0]}</b> accounts for "
                f"{bl_top.iloc[0]:.1f}% of bike-lane crashes -- infrastructure alone doesn't "
                f"remove the yielding problem. "
                if bl_top is not None and len(bl_top) else ""
            )
            insight(
                f"Fault splits close to evenly between driver- and non-motorist-attributable "
                f"causes in the current selection. Sidewalk + crosswalk account for "
                f"{infra_pct2.get('Sidewalk', 0) + infra_pct2.get('Crosswalk', 0):.1f}% of "
                f"narrative-classified crashes -- about half happen where the rider wasn't in "
                f"the road at all. {bl_text}"
                f"On sidewalks, driveway conflicts plus drivers failing to yield while turning "
                f"together explain {sw_combo:.1f}% of sidewalk crashes -- sidewalk risk looks "
                f"like a driveway-crossing problem more than a general road-crossing one."
            )
        else:
            st.info("No causation-classified crashes in the current filter selection.")
    else:
        st.info(f"No `{DEFAULT_CAUSE_PATH}` loaded -- add it under the Data Source panel to see this section.")

    # ================================================================
    # 11. Severity trend over time (aggregate, all modes pooled)
    # ================================================================
    st.markdown("### 11. Severity Trend Over Time (All Active Modes, Pooled)")
    st.caption(
        "This source table isn't split by mode -- 2025's total (13,125) matches "
        "bicycle + e-bike + e-scooter combined for that year exactly."
    )
    sev_trend = pd.DataFrame({
        "Year": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        "Fatality %": [1.68, 1.86, 1.84, 1.67, 2.05, 1.99, 2.48, 2.63, 2.51, 2.34, 1.88, 1.80, 1.27],
        "Serious Injury %": [11.66, 11.29, 11.25, 10.42, 10.76, 9.79, 10.82, 9.88, 9.37, 8.33, 8.53, 8.27, 8.08],
    })
    st1, st2 = st.columns(2)
    with st1:
        fig = px.line(sev_trend, x="Year", y="Fatality %", markers=True,
                      color_discrete_sequence=["#C0392B"])
        st.plotly_chart(style_fig(fig, title="Fatality Share by Year", height=320), use_container_width=True)
    with st2:
        fig = px.line(sev_trend, x="Year", y="Serious Injury %", markers=True,
                      color_discrete_sequence=["#FF9800"])
        st.plotly_chart(style_fig(fig, title="Serious-Injury Share by Year", height=320), use_container_width=True)
    insight(
        "Both fatality share and serious-injury share peaked around 2020-2022 and have "
        "declined since, even as total crash volume nearly doubled (driven by e-bike/e-scooter "
        "growth). This is consistent with two different stories this pooled table can't "
        "separate on its own: genuine per-crash safety improvement, or e-scooter's crash mix "
        "(which already skews toward the lowest incapacitating-injury share of the three modes, "
        "per Section 1 above) increasingly diluting the aggregate as it becomes a growing share "
        "of total crashes. A by-mode version of this table would settle which story is right -- "
        "worth requesting before citing this as a safety-improvement trend."
    )

    # ================================================================
    # 12. Pedestrian crash sub-types
    # ================================================================
    st.markdown("### 12. Pedestrian Crash Sub-Types")
    ped_df = pd.DataFrame({
        "Circumstance": ["Unusual Circumstances", "Crossing Roadway (Vehicle Not Turning)",
                          "Crossing Roadway (Vehicle Turning)", "Crossing Driveway or Alley",
                          "Backing Vehicle", "Off Roadway", "Walking Along Roadway",
                          "Dash/Dart-Out", "Other categories (5)"],
        "Count": [281, 77, 50, 34, 28, 28, 13, 13, 27],
    })
    fig = px.bar(ped_df.sort_values("Count"), x="Count", y="Circumstance", orientation="h",
                 color_discrete_sequence=["#5C6BC0"])
    fig.update_layout(yaxis_title=None, xaxis_title="Crashes (n=551)")
    st.plotly_chart(style_fig(fig, title="Pedestrian-Involved Crash Circumstances", height=380),
                     use_container_width=True)
    insight(
        "\"Unusual Circumstances\" dominates (51%) but is a catch-all, not very actionable on "
        "its own. The two \"crossing roadway\" categories combined are only 23.1% of this "
        "population -- smaller than you might expect given how much infrastructure discussion "
        "centers on crossings specifically. Scope caveat: this source table has no `MODE` "
        "column and n=551 is smaller than any single mode's full pedestrian-involved count, so "
        "it's likely a filtered subset rather than the complete population behind the 50.2% "
        "e-scooter pedestrian-involved rate -- worth confirming the exact filter before citing "
        "precise shares from this table."
    )

    # ================================================================
    # 13. Driver behavior at the full-population level
    # ================================================================
    st.markdown("### 13. Driver Behavior Flags, Full Population (Not Just Fatal Crashes)")
    st.caption(
        "Same flags as Section 6, but across all crashes by mode (Bicycle N=101,377 / "
        "E-Bike N=6,353 / E-Scooter N=5,274) -- confirms the fatal-tier alcohol/distraction "
        "findings weren't a small-N fluke."
    )
    fullbehav_df = pd.DataFrame({
        "MODE": MODES * 2,
        "Flag": ["Alcohol-related driver"] * 3 + ["Distracted driver"] * 3,
        "Pct": [0.61, 0.22, 0.36, 8.37, 6.94, 7.18],
    })
    fig = px.bar(
        fullbehav_df, x="Flag", y="Pct", color="MODE", barmode="group",
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
    )
    fig.update_layout(yaxis_title="% of all crashes (that mode)", xaxis_title=None)
    st.plotly_chart(style_fig(fig, title="Driver Behavior Flags, All Severities, by Mode", height=340),
                     use_container_width=True)
    insight(
        "E-bike drivers show the lowest alcohol involvement (0.22% vs. 0.61% bicycle) and the "
        "lowest distraction flag rate (6.94% vs. 8.37% bicycle) at full population, matching the "
        "direction of the small-N fatal-crash findings in Section 6. `driver_distraction_by_mode.csv` "
        "corroborates this: 'Not Distracted' is recorded for 83.2% of e-bike-crash drivers vs. "
        "79.8-79.9% for bicycle/e-scooter. Combined with e-bike's low fatal-crash citation rate "
        "(Section 6), a consistent picture emerges: e-bike crashes more often involve an "
        "attentive, sober driver who still failed to see or yield to the rider -- pointing more "
        "toward a visibility/conspicuity and infrastructure-geometry problem (see the bike-lane "
        "'right-hook' pattern in Section 3/10) than a driver-impairment problem, for this mode "
        "specifically."
    )

    # ================================================================
    # Data-quality / caveats
    # ================================================================
    st.markdown("---")
    st.markdown("### Data-Quality Notes Carried Over From Both Reports")
    st.markdown(
        """
- **`QWEN_CLASS`** is tautologically derived from `MODE` (Cramer's V = 1.0) -- don't cite it as an independent finding.
- **`IN_QWEN_NARRATIVES`** is ~100% True for E-Bike/E-Scooter but only ~7% for Bicycle because the bicycle-only source population was never run through the narrative classifier -- that's pipeline coverage, not rider behavior.
- **Hotspot clustering isn't usable for e-bike/e-scooter yet** -- 708 bicycle clusters vs. only 4 (e-bike) / 11 (e-scooter), each mode using a different early/late split year, an artifact of small early-year sample sizes rather than a real spatial finding.
- **Three tables came back completely empty**: `context_class_by_mode.csv`, `lane_count_by_mode.csv`, `median_type_by_mode.csv` -- likely an upstream join/filter issue worth checking on the HiPerGator side.
- **`contributing_factors_by_mode.csv` and `top_charges_active_modes.csv`** are aggregated across all modes despite their filenames -- not usable for a by-mode breakdown as-is.
- **`MICROMOBILITY_SPEED_MPH`** is populated for only 0.7% of all crashes (775 of 113,004) -- both speed charts above are directional, not precise population estimates.
- **The YEAR effect is an adoption-curve confound**, not a behavioral difference -- e-bike/e-scooter crash counts are near-zero pre-2021 and 30-35%+ of their mode's total by 2025. A true rate comparison would need an exposure denominator (riders, trips, or registered devices) this dataset doesn't have.
- **Section 10 (Crash Causation Highlights) is live**, unlike the rest of this tab -- it recomputes from `cause_analysis_export.csv` filtered to the sidebar's current Mode/Year/Severity/etc. selection, so its numbers will move as you change filters while every other section on this tab stays fixed.
- **One raw speed-extraction record shows 1,520 mph** for an e-bike crash (clearly a misparsed figure) -- already excluded from the Section 4 summary stats above (n=582 not 583), so nothing here needed correcting, but don't pull directly from the raw extraction table without filtering it out first.
- **Minimum rider age of 1-2 years old** appears for all three modes in the raw age data -- likely a child passenger (e.g. on a cargo e-bike) or a data-entry error; doesn't affect the median/mean figures used above but worth a spot-check before citing age minimums specifically.
"""
    )

    render_pipeline_figures("tab9")


st.markdown(
    """<div style="text-align:center; color:#9aa0b8; font-size:0.78rem; padding: 1.2rem 0 0.4rem 0;">
    Data: Signal4 crash tables + FDOT roadway tables, processed via eda_analysis_combined.py &middot;
    Just & Green Transportation Lab, University of Florida
    </div>""",
    unsafe_allow_html=True,
)