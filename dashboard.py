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
   default. */
section[data-testid="stSidebar"] [data-baseweb="select"],
section[data-testid="stSidebar"] [data-baseweb="select"] * {
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


@st.cache_data
def load_data(path_or_buffer):
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
def load_demographics(path_or_buffer):
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
def load_meta(path_or_buffer):
    """Pipeline funnel counts (e.g. raw records -> geocoded -> matched ->
    final export) used on the About tab. Expected as a simple two-column
    CSV: a stage/step label column and a count column, but we degrade
    gracefully to a raw table if the shape is unrecognized."""
    return pd.read_csv(path_or_buffer)


@st.cache_data
def load_narratives(path_or_buffer):
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
def load_hotspots(path_or_buffer):
    """Precomputed DBSCAN spatiotemporal cluster table (one row per
    cluster), from eda_analysis_combined.py section 09d."""
    return pd.read_csv(path_or_buffer)


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

if main_src is not None:
    df_raw = load_data(main_src)
else:
    st.info(
        f"\U0001F4C2 **Waiting on crash data.** This dashboard needs "
        f"`{DEFAULT_PATH}` to run -- either place it in the same folder as "
        f"`dashboard.py` before launching `streamlit run dashboard.py`, or "
        f"upload it using the **Data Source** panel in the sidebar. "
        f"This isn't an error, just Streamlit waiting on a file."
    )
    st.stop()

demo_raw = load_demographics(demo_src) if demo_src is not None else None
meta_raw = load_meta(meta_src) if meta_src is not None else None
narrative_raw = load_narratives(narrative_src) if narrative_src is not None else None
hotspot_raw = load_hotspots(hotspot_src) if hotspot_src is not None else None

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
def style_fig(fig, height=380, title=None):
    fig.update_layout(
        font=PLOT_FONT,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        height=height,
        margin=dict(l=10, r=10, t=56 if title else 26, b=10),
        # Centered (not right-anchored) so a full "Bicycle / E-Bike /
        # E-Scooter" legend has room on both sides instead of overflowing
        # past the chart's right edge and getting clipped mid-word. Legend
        # title dropped too ("MODE" / "S4_CRASH_SEVERITY" etc. just ate
        # width without adding information the chart doesn't already show).
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            title=dict(text=""), font=dict(size=11),
        ),
        title=dict(text=title, font=dict(size=15, family=PLOT_FONT["family"], color="#12172b")) if title else None,
    )
    return fig


# ============================================================================
# TABS
# ============================================================================
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "\u2139\uFE0F About This Dashboard",
    "\U0001F4C8 Overview & Trends",
    "\U0001F6A8 Severity & Outcomes",
    "\U0001F55B When & Where",
    "\U0001F464 Driver Behavior & Citations",
    "\U0001F6E3 Roadway Infrastructure",
    "\U0001F9D1 Demographics",
    "\U0001F4DD Narrative, Typing & Hotspots",
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
        )
        fig.update_layout(yaxis_title="% of crashes", xaxis_title=None, barmode="stack")
        st.plotly_chart(style_fig(fig, title="Injury Severity Mix by Mode"), use_container_width=True)

    with c2:
        rate_df = pd.DataFrame({
            "MODE": MODES,
            "Fatal %": [
                (df[df["MODE"] == m]["S4_CRASH_SEVERITY"] == "Fatality").mean() * 100
                if (df["MODE"] == m).any() else 0 for m in MODES
            ],
            "Serious Injury %": [
                (df[df["MODE"] == m]["S4_CRASH_SEVERITY"] == "Serious Injury").mean() * 100
                if (df["MODE"] == m).any() else 0 for m in MODES
            ],
        })
        fig = go.Figure()
        fig.add_bar(name="Fatal %", x=rate_df["MODE"], y=rate_df["Fatal %"], marker_color="#B71C1C")
        fig.add_bar(name="Serious Injury %", x=rate_df["MODE"], y=rate_df["Serious Injury %"], marker_color="#EF9A9A")
        fig.update_layout(barmode="group", yaxis_title="%")
        st.plotly_chart(style_fig(fig, title="Fatal & Serious Injury Rate"), use_container_width=True)

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
        )
        fig.update_layout(yaxis_title="% of that year's crashes")
        st.plotly_chart(style_fig(fig, title="Severity Mix Over Time (% of Crashes)"), use_container_width=True)

    with c4:
        mv_df = df.groupby("MODE", observed=True)["mv_involved"].mean().reindex(MODES).fillna(0) * 100
        fig = go.Figure(go.Bar(
            x=mv_df.index, y=mv_df.values,
            marker_color=[MODE_COLORS[m] for m in mv_df.index],
            text=[f"{v:.0f}%" for v in mv_df.values], textposition="outside",
        ))
        fig.update_layout(yaxis_title="% crashes with MV involved", yaxis_range=[0, 110])
        st.plotly_chart(style_fig(fig, title="Motor Vehicle Involvement by Mode"), use_container_width=True)

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
        st.plotly_chart(style_fig(fig, title="Crashes by Day of Week"), use_container_width=True)

    with c2:
        dn_mode = df.groupby(["MODE", "DAY_NIGHT"], observed=True).size().reset_index(name="count")
        dn_mode["pct"] = dn_mode["count"] / dn_mode.groupby("MODE")["count"].transform("sum") * 100
        fig = px.bar(
            dn_mode, x="MODE", y="pct", color="DAY_NIGHT",
            color_discrete_map={"Day": "#FDD835", "Night": "#283593"},
            category_orders={"MODE": MODES},
        )
        fig.update_layout(yaxis_title="% of crashes", xaxis_title=None, barmode="stack")
        st.plotly_chart(style_fig(fig, title="Day vs. Night Share by Mode"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        loc_mode = df.groupby(["MODE", "LOC_TYPE"], observed=True).size().reset_index(name="count")
        loc_mode["pct"] = loc_mode["count"] / loc_mode.groupby("MODE")["count"].transform("sum") * 100
        fig = px.bar(
            loc_mode, x="MODE", y="pct", color="LOC_TYPE",
            color_discrete_map={"Intersection": "#5C6BC0", "Segment": "#26A69A"},
            category_orders={"MODE": MODES},
        )
        fig.update_layout(yaxis_title="% of crashes", xaxis_title=None, barmode="stack")
        st.plotly_chart(style_fig(fig, title="Intersection vs. Segment by Mode"), use_container_width=True)

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
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        fig.update_layout(yaxis_title=None, xaxis_title="% of that mode's crashes",
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_fig(fig, title="Light Conditions (% Within Mode)"), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        wthr_top = df["WEATHER_CONDITION"].value_counts().nlargest(5).index
        wthr_df = df[df["WEATHER_CONDITION"].isin(wthr_top)]
        wm = wthr_df.groupby(["WEATHER_CONDITION", "MODE"], observed=True).size().reset_index(name="count")
        wm["pct"] = wm.apply(lambda r: r["count"] / mode_totals.get(r["MODE"], 1) * 100, axis=1)
        fig = px.bar(
            wm, y="WEATHER_CONDITION", x="pct", color="MODE", orientation="h", barmode="group",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        fig.update_layout(yaxis_title=None, xaxis_title="% of that mode's crashes",
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_fig(fig, title="Weather Conditions (% Within Mode)"), use_container_width=True)

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
        st.plotly_chart(style_fig(fig, title="Top 15 Counties", height=430), use_container_width=True)

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
        fig = px.bar(
            flag_df, y="Flag", x="Pct", color="Mode", orientation="h", barmode="group",
            color_discrete_map=MODE_COLORS, category_orders={"Mode": MODES},
        )
        fig.update_layout(xaxis_title="% of crashes for that mode", yaxis_title=None)
        st.plotly_chart(style_fig(fig, title="Driver Behavior Flags by Mode", height=440), use_container_width=True)

    with c2:
        cite_df = df.groupby("MODE", observed=True)["CITED"].mean().reindex(MODES).fillna(0) * 100
        fig = go.Figure(go.Bar(
            x=cite_df.index, y=cite_df.values,
            marker_color=[MODE_COLORS[m] for m in cite_df.index],
            text=[f"{v:.0f}%" for v in cite_df.values], textposition="outside",
        ))
        fig.update_layout(yaxis_title="% crashes with citation", yaxis_range=[0, 110])
        st.plotly_chart(style_fig(fig, title="Citation Rate by Mode"), use_container_width=True)

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


st.markdown(
    """<div style="text-align:center; color:#9aa0b8; font-size:0.78rem; padding: 1.2rem 0 0.4rem 0;">
    Data: Signal4 crash tables + FDOT roadway tables, processed via eda_analysis_combined.py &middot;
    Just & Green Transportation Lab, University of Florida
    </div>""",
    unsafe_allow_html=True,
)