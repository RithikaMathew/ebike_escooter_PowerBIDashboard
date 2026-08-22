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
            "Drop a `census_tracts.geojson` file (GEOID + a total-population column + geometry) "
            "in the same folder you run this dashboard from and it loads automatically -- no "
            "upload needed. `build_census_tracts.py` (run once, locally) generates that file "
            "for Florida from TIGER/Line tract boundaries + ACS table B01003. The uploader below "
            "is just a fallback if you'd rather not put the file next to the script."
        )
        tract_up = st.file_uploader("Or upload census_tracts.geojson", type=["geojson", "json"], key="tract_upload")
        tract_src = tract_up if tract_up is not None else (
            "census_tracts.geojson" if os.path.exists("census_tracts.geojson") else None
        )
        if tract_src == "census_tracts.geojson":
            st.caption("\u2713 Found `census_tracts.geojson` in the working directory -- using it automatically.")
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
    """n: optional overall sample size for this chart, shown as a small
    'n = 1,234' subtitle under the title. Per-category counts belong in
    hover text (see add_hover_counts), not stacked into the title, so they
    don't collide with the legend."""
    has_subtitle = title and n is not None
    full_title = title
    if has_subtitle:
        n_txt = fmt_n(n) if not isinstance(n, dict) else fmt_n(sum(n.values()))
        full_title = (
            f"{title}<br><span style='font-size:11px;font-weight:400;color:#6b7280'>{n_txt}</span>"
        )
    fig.update_layout(
        font=PLOT_FONT,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        height=height + (26 if has_subtitle else 0),
        margin=dict(l=10, r=10, t=(96 if has_subtitle else 56) if title else 26, b=10),
        # Centered (not right-anchored) so a full "Bicycle / E-Bike /
        # E-Scooter" legend has room on both sides instead of overflowing
        # past the chart's right edge and getting clipped mid-word. Legend
        # title dropped too ("MODE" / "S4_CRASH_SEVERITY" etc. just ate
        # width without adding information the chart doesn't already show).
        # When there's a two-line title+subtitle, the legend needs to sit
        # further above the plot (higher y) or it overlaps the subtitle text.
        legend=dict(
            orientation="h", yanchor="bottom", y=1.2 if has_subtitle else 1.02, xanchor="center", x=0.5,
            title=dict(text=""), font=dict(size=11),
        ),
        title=dict(
            text=full_title, y=0.97 if has_subtitle else 0.9, yanchor="top",
            font=dict(size=15, family=PLOT_FONT["family"], color="#12172b"),
        ) if full_title else None,
    )
    return fig


def insight(text, label="Key insight"):
    """Shared 'callout' box used across tabs to highlight a takeaway.
    Tab 5's Speed x Infrastructure section calls this -- it used to rely on
    a version of this function defined ~500 lines later inside Tab 8, which
    meant it referenced an undefined name and would raise a NameError the
    first time that branch of Tab 5 ran (moved here so it's defined before
    any tab needs it). Tab 9 still defines its own local `insight()` with a
    different label ("Why it matters") on purpose -- that one's fine as-is."""
    st.markdown(
        f"""<div class="section-note">\U0001F4A1 <b>{label}:</b> {text}</div>""",
        unsafe_allow_html=True,
    )


def add_hover_counts(fig, counts, pct_mode=True):
    """Attach per-category raw counts to a trace's HOVER text only (not
    visible on the chart itself), so a % or rate chart still lets the reader
    see the underlying n for that bar/segment without cluttering the chart.
    `counts` must align (same order) with the trace's x/y categories."""
    counts = list(counts)
    if pct_mode:
        fig.update_traces(
            customdata=np.array(counts).reshape(-1, 1),
            hovertemplate="%{x}: %{y:.1f}% (n=%{customdata[0]:,})<extra>%{fullData.name}</extra>",
        )
    else:
        fig.update_traces(
            customdata=np.array(counts).reshape(-1, 1),
            hovertemplate="%{x}: %{y:,} (n=%{customdata[0]:,})<extra>%{fullData.name}</extra>",
        )
    return fig

