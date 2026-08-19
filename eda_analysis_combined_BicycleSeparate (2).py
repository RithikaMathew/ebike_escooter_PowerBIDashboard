"""
Signal4Data – EDA v5
Changes from v4:
 - Bicycle now sourced from the larger S4_Crash_bicycle population
   (/blue/xiangyan/rithika/stats/S4_Crash_bicycle) instead of being derived
   inside Signal4Data via S4_CRASH_TYPE/NON_MOTORIST_DESCRIPTION_CODE
   structural fallback. Signal4Data is still used for E-Bike/E-Scooter (Qwen
   narrative classification) and 'Other'. See section 5 "MERGE BICYCLE +
   SIGNAL4DATA" for the REPORT_NUMBER overlap/conflict rule (Qwen
   E-Bike/E-Scooter overrides S4_Crash_bicycle's default Bicycle label).
 - ASSUMPTIONS PENDING CONFIRMATION (flagged inline where used):
     1. S4_Crash_bicycle/gis geolocation contents/format, and whether
        S4_Crash_bicycle/crash tables/crash_event.csv already carries
        S4_LATITUDE/S4_LONGITUDE like Signal4Data does, or needs a separate
        join -- section 09/20 (lat/lon, DBSCAN hotspots) still reads lat/lon
        off `ce` as before; if the bicycle rows come back all-NaN there,
        this needs a join against gis geolocation instead.
     2. S4_Crash_bicycle/fdot tables has the same 3 file names as
        Signal4Data/fdot tables (crash_roadway_vehicle_driver.csv,
        roadway_segment.csv, roadway_intersection.csv) -- wired up in
        section 18 on that assumption.
     3. fars.csv (FDOT) is left Signal4Data-only -- it's documented as a
        statewide fatal-crash export, so it should already cover
        S4_Crash_bicycle fatalities without a separate bicycle copy.

Changes from v3:
 - Severity tiers formalised: ALL_CRASHES / KSI (Killed + Serious Injury) /
   FATAL_ONLY. Helper subsets (micro_df, micro_df_ksi, micro_df_fatal) and
   is_ksi()/is_fatal() flags added so any downstream table can be sliced into
   the three tiers the report should cover. Applied so far to: citation rate
   by mode, driver-behavior flags, day/night split, age distribution -- see
   README Section 21 for the full status of which figures still need the
   a/b/c treatment (this is mechanical to extend once the helper exists).
 - "Distribution comparison" charts (Road Type, Light Conditions, Weather
   Conditions, Median Type, Lane Count, Intersection Control) converted from
   raw counts to % WITHIN MODE. Raw counts made Bicycle (the largest group)
   dominate every bar and made it impossible to see whether E-Bike/E-Scooter
   have a *different shape* of distribution, which was the actual question.
 - New: Age Distribution by Band, by Mode (02d).
 - New: Age Distribution by Injury Severity, as three SEPARATE figures (one
   each for Bicycle / E-Bike / E-Scooter) instead of one combined chart --
   02e/02f/02g.
 - New: Citation Rate Over Time, one line per mode (11c), so the declining
   trend can be compared across Bicycle/E-Bike/E-Scooter instead of pooled.
 - 05c "Severity Mix Over Time" changed from raw stacked counts to a
   proportion (%) stacked chart, so the changing MIX is visible independent
   of the (rising) overall crash volume.
 - New Section 20: exploratory spatiotemporal hotspot clustering (DBSCAN) by
   mode, split into an early-period vs late-period window to flag emerging
   hotspots. Also flags the 3-4pm peak-hour question directly on the hourly
   chart -- see README Section 22 for discussion; this is a hypothesis, not
   a proven cause.
 - New Section 21: relative crash-risk-by-exposure (crash frequency / Strava
   cycling-volume) scaffold, by county. PENDING -- requires the Strava
   volume export (see README); the code is written to run automatically
   once that file is dropped in, and to print a clear "pending" message and
   skip cleanly if it is absent, rather than failing.
 - Day vs Night definition documented explicitly (see day_night() and
   README Section 23) -- this is inherited as-is from the FLHSMV/Signal4
   S4_DAY_OR_NIGHT field; the exact sunrise/sunset or civil-twilight rule
   FLHSMV used to set that field is not present in our extract and needs to
   be confirmed with FLHSMV/Signal4 if an exact cutoff time is needed.

Changes from v2 (carried forward):
 - E-Bike / E-Scooter come ONLY from Qwen narratives. 'Other Cyclist' structural
   code is NOT mapped to E-Bike (could be unicycles, cargo bikes, etc.).
 - 'Other' class excluded from all visualisations. Charts show Bicycle/E-Bike/E-Scooter.
 - 06a label overlap fixed.
 - 08c simplified (top crash types only, readable labels).
 - 'Active modes' defined clearly throughout.
 - Data integrity check (REPORT_NUMBER overlap across files).
 - README updated with classification discrepancy note.

COMBINED WITH: the "Extended" companion analysis (driver behavior, citations,
pedestrian reference, roadway infrastructure, FARS fatal coding), folded in
as Sections 15-19 below. A Power BI export (Section 20) combines fields from
both into a single power_bi_export.csv, so this one script now replaces what
used to be two separate scripts and two separate runs.

DATA PROCESSING NOTE (see README Section 24 "Data Processing & Limitations"
for the full write-up): FLHSMV/Signal4 first applied keyword search to the
statewide crash-narrative text to pull out candidate e-bike / e-scooter
crashes, plus a comparison sample of ~5,000 narratives that did NOT match
those keywords, covering January 1, 2014 through 2026 (need to confirm
whether the cutoff is end-of-June 2026 or later -- see README Section 24).
Those candidate narratives were then run through the Qwen multilabel
classifier (multilabel_ebike.xlsx / multilabel_RegBike.xlsx) to assign a
final Bicycle / E-Bike / E-Scooter / Other label. Two limitations follow
directly from this pipeline and are repeated at point of use throughout the
script:
  1. Keyword extraction is precision-oriented, not a random sample -- crashes
     whose narrative never mentions an e-bike/e-scooter-suggestive term will
     not be in the candidate pool at all, so absolute E-Bike/E-Scooter counts
     are a floor, not a census.
  2. "Bicycle" produced by this pipeline (Qwen-labeled "bicyclist") is a
     SUBSET of all regular-bicycle crashes in Signal4 (see ACTIVE_MODES vs.
     "Bicycle" note below) -- structurally-classified bicycle crashes that
     were never run through Qwen are folded into micro_df/ACTIVE_MODES for
     volume/trend purposes but are not distinguishable from Qwen-labeled
     "Bicycle" in the mode-comparison charts. Use ACTIVE_MODES (all
     micromobility) for spatial/temporal trend totals; use "Bicycle" only
     when the question specifically compares Bicycle vs E-Bike vs E-Scooter.
"""

import os, re, sys, warnings, textwrap
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

BASE  = "/blue/xiangyan/rithika/stats"
DATA  = os.path.join(BASE, "Signal4Data")
CRASH = os.path.join(DATA, "crash tables")
FDOT  = os.path.join(DATA, "fdot tables")
PED   = os.path.join(DATA, "ped bike typing")

# --- Bicycle now sourced from the larger S4_Crash_bicycle population instead
#     of Signal4Data (bigger N; every row in this folder is already restricted
#     to bicycle crashes -- no S4_CRASH_TYPE / NON_MOTORIST_DESCRIPTION_CODE
#     filtering needed). Signal4Data is KEPT and still used for E-Bike /
#     E-Scooter (Qwen narrative classification) and for 'Other' (pedestrian,
#     single-vehicle, animal, MV-only, etc). See "5. MERGE BICYCLE + SIGNAL4"
#     below for how REPORT_NUMBER overlap between the two systems is resolved.
BASE_BIKE  = os.path.join(BASE, "S4_Crash_bicycle")
CRASH_BIKE = os.path.join(BASE_BIKE, "crash tables")
FDOT_BIKE  = os.path.join(BASE_BIKE, "fdot tables")
PED_BIKE   = os.path.join(BASE_BIKE, "ped bike typing")
GIS_BIKE   = os.path.join(BASE_BIKE, "gis geolocation")

OUT   = os.path.join(BASE, "results")
FIGS  = os.path.join(OUT, "figures")
TABS  = os.path.join(OUT, "tables")

for sub in ["01_overview","02_who","03_when","04_where","05_severity",
            "06_crash_typing","07_text_mining","08_qwen","09_latlon",
            "10_driver_behavior","11_violations","12_pedestrian_context",
            "13_roadway_infrastructure","14_fars_fatal"]:
    os.makedirs(os.path.join(FIGS, sub), exist_ok=True)
os.makedirs(TABS, exist_ok=True)

PALETTE = {"Bicycle":"#2196F3","E-Bike":"#4CAF50","E-Scooter":"#FF9800","Other":"#9E9E9E"}
SEVERITY_COLORS = {
    "No Injury":"#A5D6A7","Possible":"#FFF176",
    "Non-Incapacitating":"#FFB74D","Incapacitating":"#EF9A9A","Fatal":"#B71C1C",
}
MALE_COLOR="#1565C0"; FEMALE_COLOR="#AD1457"
DAY_COLOR="#FDD835";  NIGHT_COLOR="#283593"

# "Active modes" = Bicycle + E-Bike + E-Scooter.
# Excludes 'Other' (pedestrian-only, single-vehicle, MV-only crashes, etc.)
# This term appears in chart titles to clarify which crash types are included.
ACTIVE_MODES = ["Bicycle","E-Bike","E-Scooter"]

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white","axes.edgecolor":"#333333",
    "axes.labelcolor":"#111111","xtick.color":"#111111","ytick.color":"#111111",
    "text.color":"#111111","font.family":"DejaVu Sans",
    "axes.titlesize":12,"axes.labelsize":10,
})

def save(fig, subfolder, name):
    path = os.path.join(FIGS, subfolder, name+".png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {path}")

def wrap(text, width=28):
    return "\n".join(textwrap.wrap(str(text), width))

def clean_id(series):
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def is_yes(series):
    """Robust truthy check -- handles Y/N, YES/NO, TRUE/FALSE, 1/0 without
    assuming which encoding a given S4_IS_* flag uses."""
    return series.astype(str).str.strip().str.upper().isin(["Y", "YES", "TRUE", "1"])


def pct_y(ax):
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())


# ---------------------------------------------------------------------------
# SEVERITY TIERS -- "a) all crashes, b) KSI, c) fatal" pattern
# KSI = Killed + Serious Injury = Fatal + Incapacitating Injury, the standard
# transportation-safety definition. Applied via is_ksi()/is_fatal() so any
# downstream table/figure can be produced for all three tiers. See README
# Section 21 for which figures currently have all three tiers vs. just tier
# (a); extending a tier-(a)-only figure to KSI/Fatal is a 2-line filter using
# these two flags -- the pattern is intentionally kept mechanical.
# ---------------------------------------------------------------------------
def is_ksi(sev_series):
    """True for Fatal or Incapacitating ('Serious') injury severities."""
    return sev_series.isin(["Fatal", "Incapacitating"])


def is_fatal(sev_series):
    return sev_series == "Fatal"


def pct_within_group_bar(df, group_col, value_col, ax, top_n=8, order=None,
                          palette=None, wrap_width=18):
    """Grouped horizontal % bar chart, normalised WITHIN each group_col value
    (e.g. within each MODE) rather than raw counts. Use this whenever the
    question is 'does the SHAPE of the distribution differ across modes' --
    raw counts are misleading when group sizes differ (e.g. this pipeline's
    active-mode split has run at roughly Bicycle/E-Bike/E-Scooter ~5-6k each
    -- fairly balanced -- but that balance isn't guaranteed to hold as more
    data is added, and any one mode dominating the total would make raw
    counts misleading the same way).
    Returns the pivoted %-within-group table (rows=value_col, cols=group_col)
    for saving to CSV.
    """
    order = order or sorted(df[group_col].dropna().unique())
    palette = palette or {g: PALETTE.get(g, "#607D8B") for g in order}
    counts = df.groupby([group_col, value_col]).size().reset_index(name="count")
    top_vals = counts.groupby(value_col)["count"].sum().nlargest(top_n).index
    counts = counts[counts[value_col].isin(top_vals)]
    pivot = counts.pivot_table(index=value_col, columns=group_col, values="count", fill_value=0)
    pivot = pivot.reindex(columns=[g for g in order if g in pivot.columns])
    pivot_pct = pivot.div(df.groupby(group_col).size(), axis=1) * 100
    pivot_pct = pivot_pct.reindex(pivot_pct.sum(axis=1).sort_values(ascending=False).index)
    y = np.arange(len(pivot_pct))
    n_groups = len(pivot_pct.columns)
    height = 0.8 / max(n_groups, 1)
    for i, g in enumerate(pivot_pct.columns):
        ax.barh(y + i * height, pivot_pct[g].values, height,
                label=g, color=palette.get(g, "#607D8B"), edgecolor="white")
    ax.set_yticks(y + height * (n_groups - 1) / 2)
    ax.set_yticklabels(["\n".join(textwrap.wrap(str(v), wrap_width)) for v in pivot_pct.index], fontsize=8)
    ax.set_xlabel("% of that mode's crashes")
    ax.legend(title=group_col.title() if group_col.isupper() else group_col, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return pivot_pct


# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
print("\n=== Loading data ===")
# Signal4Data (used for E-Bike / E-Scooter via Qwen, and 'Other') -- suffix
# _s4 to distinguish from the combined tables built in section 5 below.
ce_s4    = pd.read_csv(os.path.join(CRASH,"crash_event.csv"),  low_memory=False)
nm_s4    = pd.read_csv(os.path.join(CRASH,"non_motorist.csv"), low_memory=False)
veh_s4   = pd.read_csv(os.path.join(CRASH,"vehicle.csv"),      low_memory=False)
drv_s4   = pd.read_csv(os.path.join(CRASH,"driver.csv"),       low_memory=False)
pax_s4   = pd.read_csv(os.path.join(CRASH,"passenger.csv"),    low_memory=False)
vio_s4   = pd.read_csv(os.path.join(CRASH,"violation.csv"),    low_memory=False)
btype_s4 = pd.read_csv(os.path.join(DATA,"ped bike typing","bicycle_typing_20.csv"), low_memory=False)

# S4_Crash_bicycle (larger bicycle population; every row here IS a bicycle
# crash already -- no structural filtering needed). Same table names/schema
# as Signal4Data per confirmation.
print("\n  -- S4_Crash_bicycle (bicycle population) --")
ce_bike    = pd.read_csv(os.path.join(CRASH_BIKE,"crash_event.csv"),  low_memory=False)
nm_bike    = pd.read_csv(os.path.join(CRASH_BIKE,"non_motorist.csv"), low_memory=False)
veh_bike   = pd.read_csv(os.path.join(CRASH_BIKE,"vehicle.csv"),      low_memory=False)
drv_bike   = pd.read_csv(os.path.join(CRASH_BIKE,"driver.csv"),       low_memory=False)
pax_bike   = pd.read_csv(os.path.join(CRASH_BIKE,"passenger.csv"),    low_memory=False)
vio_bike   = pd.read_csv(os.path.join(CRASH_BIKE,"violation.csv"),    low_memory=False)
btype_bike = pd.read_csv(os.path.join(PED_BIKE,"bicycle_typing_20.csv"), low_memory=False)
print(f"  crash_event (bicycle) : {ce_bike.shape}   <- 1 row per BICYCLE CRASH")
print(f"  non_motorist (bicycle): {nm_bike.shape}")
print(f"  vehicle (bicycle)     : {veh_bike.shape}")
print(f"  driver (bicycle)      : {drv_bike.shape}")
print(f"  passenger (bicycle)   : {pax_bike.shape}")
print(f"  violation (bicycle)   : {vio_bike.shape}")
print(f"  bicycle_typing_20.csv (bicycle): {btype_bike.shape}")

narr_e = pd.read_excel(os.path.join(DATA,"multilabel_ebike.xlsx"))
narr_r = pd.read_excel(os.path.join(DATA,"multilabel_RegBike.xlsx"))
narr_e["_SOURCE_FILE"] = "multilabel_ebike.xlsx"
narr_r["_SOURCE_FILE"] = "multilabel_RegBike.xlsx"
narr_q = pd.concat([narr_e, narr_r], ignore_index=True)
narr_q = narr_q.rename(columns={"prediction":"Classification"})

# Normalize ID once up front so duplicate detection matches how it'll be
# used for joins later (handles "12345" vs "12345.0" vs whitespace).
narr_q["_ID_CLEAN"] = (narr_q["ID"].astype(str).str.strip()
                        .str.replace(r"\.0$", "", regex=True))

# --- DUPLICATE ID CHECK (within a file and across the two files) ---
dup_mask = narr_q["_ID_CLEAN"].duplicated(keep=False)
n_dup_rows = int(dup_mask.sum())
n_dup_ids  = narr_q.loc[dup_mask, "_ID_CLEAN"].nunique()
print(f"\n  DUPLICATE ID CHECK (multilabel_ebike.xlsx + multilabel_RegBike.xlsx, on 'ID'):")
if n_dup_rows == 0:
    print("  No duplicate IDs found across either file. Good.")
else:
    dup_detail = narr_q.loc[dup_mask, ["_ID_CLEAN","_SOURCE_FILE","Classification"]].sort_values("_ID_CLEAN")
    dup_detail.to_csv(os.path.join(TABS,"duplicate_ids_multilabel.csv"), index=False)
    cross_file_ids = (dup_detail.groupby("_ID_CLEAN")["_SOURCE_FILE"].nunique() > 1)
    n_cross = int(cross_file_ids.sum())
    inconsistent = (dup_detail.groupby("_ID_CLEAN")["Classification"].nunique() > 1)
    n_inconsistent = int(inconsistent.sum())
    print(f"  [WARN] {n_dup_ids} duplicate ID(s) found across {n_dup_rows} rows.")
    print(f"  -> {n_cross} of those IDs appear in BOTH files (ebike AND RegBike).")
    print(f"  -> {n_inconsistent} of those IDs have DIFFERENT Classification values across their duplicate rows.")
    print(f"  Full list saved to tables/duplicate_ids_multilabel.csv for review.")
    print(f"  Keeping the FIRST occurrence of each duplicate ID (ebike file takes priority over RegBike)")
    print(f"  and dropping the rest, so REPORT_NUMBER joins and counts don't get inflated/ambiguous.")
    narr_q = narr_q.drop_duplicates(subset="_ID_CLEAN", keep="first").reset_index(drop=True)

print(f"  crash_event (Signal4Data)  : {ce_s4.shape}   <- 1 row per CRASH")
print(f"  non_motorist (Signal4Data) : {nm_s4.shape}  <- 1 row per NON-MOTORIST PERSON per crash")
print(f"  vehicle (Signal4Data)      : {veh_s4.shape}  <- 1 row per VEHICLE per crash")
print(f"  driver (Signal4Data)       : {drv_s4.shape}  <- 1 row per DRIVER per crash")
print(f"  passenger (Signal4Data)    : {pax_s4.shape}  <- 1 row per PASSENGER per crash")
print(f"  violation (Signal4Data)    : {vio_s4.shape}   <- 1 row per CITATION per crash")
print(f"  multilabel_ebike + multilabel_RegBike (combined, deduped) : {narr_q.shape}"
      f"  ({narr_e.shape[0]} ebike + {narr_r.shape[0]} RegBike loaded, {n_dup_rows} duplicate rows removed)")

# Why different row counts?
print("\n  ROW COUNT EXPLANATION:")
print(f"  crash_event has {len(ce_s4):,} rows (Signal4Data) = {ce_s4['REPORT_NUMBER'].nunique():,} unique crashes (1 crash = 1 row).")
print(f"  non_motorist has {len(nm_s4):,} rows. Same {nm_s4['REPORT_NUMBER'].nunique():,} unique crashes,")
print(f"  but each person involved in a crash gets their own row.")
print(f"  e.g. a crash with 2 bicyclists -> 1 row in crash_event, 2 rows in non_motorist.")
print(f"  Same logic applies to vehicle, driver, passenger, violation tables.")

# DATA INTEGRITY CHECK: do all REPORT_NUMBERs match across files? (Signal4Data)
print("\n  DATA INTEGRITY (REPORT_NUMBER matching, Signal4Data):")
ce_nums  = set(ce_s4["REPORT_NUMBER"].astype(str))
nm_nums  = set(nm_s4["REPORT_NUMBER"].astype(str))
veh_nums = set(veh_s4["REPORT_NUMBER"].astype(str))
drv_nums = set(drv_s4["REPORT_NUMBER"].astype(str))
pax_nums = set(pax_s4["REPORT_NUMBER"].astype(str))
vio_nums = set(vio_s4["REPORT_NUMBER"].astype(str))

print(f"  non_motorist REPORT_NUMBERs not in crash_event : {len(nm_nums - ce_nums)}")
print(f"  crash_event REPORT_NUMBERs not in non_motorist : {len(ce_nums - nm_nums)}")
print(f"  -> In this dataset all {len(ce_nums):,} unique REPORT_NUMBERs appear in BOTH files.")
print(f"  -> Bicycle no longer uses Signal4Data structural fallback -- it now comes from the")
print(f"     S4_Crash_bicycle population directly (see section 5). Signal4Data structural")
print(f"     matching (S4_CRASH_TYPE / NON_MOTORIST_DESCRIPTION_CODE) is kept only as a")
print(f"     diagnostic cross-check, not as the deciding logic.")

# S4_Crash_bicycle integrity
ce_bike_nums = set(ce_bike["REPORT_NUMBER"].astype(str))
print(f"\n  DATA INTEGRITY (S4_Crash_bicycle): {len(ce_bike_nums):,} unique bicycle REPORT_NUMBERs.")
print(f"  Overlap with Signal4Data crash_event REPORT_NUMBERs: {len(ce_bike_nums & ce_nums):,}")
print(f"  (expected -- the same physical crash can appear in both source systems; resolved in section 5.)")

# What is in S4_CRASH_TYPE? (answers: is Signal4 mainly bicycle+MV crashes?)
print("\n  S4_CRASH_TYPE breakdown (Signal4Data crash_event.csv):")
ct_counts = ce_s4["S4_CRASH_TYPE"].value_counts()
print(ct_counts.to_string())
bike_n = int(ct_counts.get("Bicycle", 0))
ped_n = int(ct_counts.get("Pedestrian", 0))
sv_n = int(ct_counts.get("Single Vehicle", 0))
animal_n = int(ct_counts.get("Animal", 0))
print(f"\n  NOTE: Signal4Data crash_event includes Bicycle ({bike_n:,}), Pedestrian ({ped_n:,}), "
      f"Single Vehicle ({sv_n:,}),")
print(f"  and other crash types. It is NOT exclusively bicycle-vs-motor-vehicle crashes.")
print(f"  'Single Vehicle' = a bicycle or pedestrian crash with no other vehicle involved.")
print(f"  'Animal' ({animal_n:,} crashes) = initial harmful event was a collision with an animal.")
print(f"  In bicycle_typing_20.csv, crash GROUPS like 'Motorist Left Turn/Merge' describe")
print(f"  the crash SCENARIO between a bicyclist and a motorist turning left -- the motorist")
print(f"  is the at-fault party in most of these. These are indeed mostly bicycle+MV crashes.")

# ===========================================================================
# 2. QWEN NORMALISATION + RECORD PROVENANCE
# ===========================================================================
print("\n=== Qwen normalisation & provenance ===")
qwen_map = {
    "ebike":"E-Bike", "e-bike":"E-Bike",
    "escooter":"E-Scooter", "e-scooter":"E-Scooter",
    "bicyclist":"Bicycle", "bicycle":"Bicycle",
    "other":"Other",
}
narr_q["HSMV_str"]  = narr_q["_ID_CLEAN"]
narr_q["QWEN_MODE"] = (narr_q["Classification"].astype(str).str.strip().str.lower()
                        .map(qwen_map).fillna("Other"))

overlap = sorted(ce_nums & set(narr_q["HSMV_str"]))
print(f"  Narrative records : {len(narr_q)}")
print(f"  Matched in crash_event : {len(overlap)}")

# In record_provenance, explain Classification vs QWEN_MODE columns
# Classification = raw value from the Excel file ("E-bike", "E-scooter", "Bicyclist", "Other")
# QWEN_MODE      = normalised to match our palette keys ("E-Bike", "E-Scooter", "Bicycle", "Other")
ce_lite = ce_s4[["REPORT_NUMBER","CRASH_YEAR","COUNTY_NAME","S4_CRASH_TYPE"]].copy()
ce_lite["REPORT_NUMBER"] = ce_lite["REPORT_NUMBER"].astype(str)

nm_desc = (nm_s4.groupby("REPORT_NUMBER")["NON_MOTORIST_DESCRIPTION_CODE"]
             .apply(lambda x: " | ".join(x.dropna().astype(str).unique()))
             .reset_index(name="NM_DESCRIPTION_CODES"))
nm_desc["REPORT_NUMBER"] = nm_desc["REPORT_NUMBER"].astype(str)

prov_df = narr_q[["HSMV_str","Classification","QWEN_MODE"]].copy()
prov_df.rename(columns={"HSMV_str":"REPORT_NUMBER"}, inplace=True)
prov_df = (prov_df
           .merge(ce_lite, on="REPORT_NUMBER", how="left")
           .merge(nm_desc,  on="REPORT_NUMBER", how="left"))
prov_df["NM_DESCRIPTION_CODES"] = prov_df["NM_DESCRIPTION_CODES"].fillna("(not in non_motorist)")
for col, s in [("in_crash_event",ce_nums),("in_non_motorist",nm_nums),
               ("in_vehicle",veh_nums),("in_driver",drv_nums),
               ("in_passenger",pax_nums),("in_violation",vio_nums)]:
    prov_df[col] = prov_df["REPORT_NUMBER"].isin(s)

# Add a column that explains the difference
prov_df["Classification_NOTE"] = (
    "Raw Qwen output from Excel. Values: E-bike / E-scooter / Bicyclist / Other. "
    "QWEN_MODE is the normalised version mapped to our analysis labels "
    "(E-bike->E-Bike, E-scooter->E-Scooter, Bicyclist->Bicycle, Other->Other)."
)
prov_df.to_csv(os.path.join(TABS,"record_provenance.csv"), index=False)
print(f"  record_provenance.csv: {len(prov_df)} rows")
print(f"  CLASSIFICATION vs QWEN_MODE:")
print(f"    Classification = raw Excel value  (E-bike / E-scooter / Bicyclist / Other)")
print(f"    QWEN_MODE      = normalised label  (E-Bike / E-Scooter / Bicycle / Other)")
print(f"    They differ only in capitalisation/spelling to match the analysis colour palette.")
for col in ["in_crash_event","in_non_motorist","in_vehicle","in_driver","in_passenger","in_violation"]:
    print(f"    {col}: {prov_df[col].sum():,} / {len(prov_df):,}")

# ===========================================================================
# 5. MERGE BICYCLE (S4_Crash_bicycle) + SIGNAL4DATA -> final combined tables
#    - Bicycle: every REPORT_NUMBER in S4_Crash_bicycle (the larger, dedicated
#      bicycle population). No more S4_CRASH_TYPE/NON_MOTORIST_DESCRIPTION_CODE
#      structural fallback -- that logic is retired now that a purpose-built
#      bicycle population exists.
#    - E-Bike / E-Scooter: ONLY from Qwen narrative classification
#      (multilabel_ebike.xlsx / multilabel_RegBike.xlsx), same as before.
#    - Other: everything else in Signal4Data (pedestrian, single-vehicle,
#      animal, MV-only, 'Other Cyclist', etc).
#    - CONFLICT RULE (REPORT_NUMBER can legitimately appear in BOTH systems --
#      confirmed the same physical crash can be in S4_Crash_bicycle AND show
#      up in Signal4Data): Qwen wins when it says E-Bike/E-Scooter, since
#      that's a narrative-text-level call on that specific crash and more
#      granular than "this crash is bicycle-related". Otherwise, if the
#      REPORT_NUMBER is in S4_Crash_bicycle, it's Bicycle. This also prevents
#      double-counting: a REPORT_NUMBER contributes rows from exactly ONE
#      source system into the combined tables below.
# ===========================================================================
print("\n=== 5. Merging S4_Crash_bicycle + Signal4Data ===")

qwen_lookup     = narr_q.set_index("HSMV_str")["QWEN_MODE"].to_dict()
qwen_raw_lookup = narr_q.set_index("HSMV_str")["Classification"].to_dict()

ce_bike["REPORT_NUMBER"] = clean_id(ce_bike["REPORT_NUMBER"])
bike_pop_rns = set(ce_bike["REPORT_NUMBER"])

ebike_escoot_rns = {rn for rn, m in qwen_lookup.items() if m in ("E-Bike", "E-Scooter")}
# Net bicycle population: S4_Crash_bicycle rows, MINUS any that Qwen's
# narrative model reclassified as E-Bike/E-Scooter.
bike_rns = bike_pop_rns - ebike_escoot_rns

# Edge case: Qwen said "Bicycle" for a REPORT_NUMBER that ISN'T in
# S4_Crash_bicycle at all. Rare (S4_Crash_bicycle is supposed to be the
# bigger/more complete population) but don't silently drop it -- fold it in
# and flag the count so it can be sanity-checked.
qwen_bicycle_rns = {rn for rn, m in qwen_lookup.items() if m == "Bicycle"}
qwen_bicycle_not_in_pop = qwen_bicycle_rns - bike_pop_rns
if qwen_bicycle_not_in_pop:
    print(f"  [CHECK] {len(qwen_bicycle_not_in_pop):,} REPORT_NUMBER(s) Qwen-classified 'Bicyclist' "
          f"are NOT in S4_Crash_bicycle -- folding in anyway, worth a sanity check.")
    bike_rns |= qwen_bicycle_not_in_pop

overlap_bike_rns = bike_pop_rns & ce_nums
print(f"  S4_Crash_bicycle REPORT_NUMBERs also present in Signal4Data crash_event: {len(overlap_bike_rns):,}")
print(f"  Of those, {len({r for r in overlap_bike_rns if r in ebike_escoot_rns}):,} were overridden to "
      f"E-Bike/E-Scooter by Qwen (narrative wins).")
print(f"  Final net Bicycle population: {len(bike_rns):,}")

def combine_bike_plus_s4(df_bike, df_s4, bike_rns, key_col="REPORT_NUMBER"):
    """Union of the S4_Crash_bicycle table + the matching Signal4Data table,
    keyed on REPORT_NUMBER. For REPORT_NUMBERs in bike_rns (net Bicycle),
    keep the S4_Crash_bicycle row. For everything else (E-Bike/E-Scooter/
    Other), keep the Signal4Data row. A REPORT_NUMBER present in both source
    systems contributes rows from exactly one side, so nothing is
    double-counted downstream."""
    d_bike = df_bike.copy()
    d_bike[key_col] = clean_id(d_bike[key_col])
    d_s4 = df_s4.copy()
    d_s4[key_col] = clean_id(d_s4[key_col])
    d_bike_keep = d_bike[d_bike[key_col].isin(bike_rns)]
    d_s4_keep   = d_s4[~d_s4[key_col].isin(bike_rns)]
    return pd.concat([d_bike_keep, d_s4_keep], ignore_index=True, sort=False)

ce    = combine_bike_plus_s4(ce_bike,    ce_s4,    bike_rns)
nm    = combine_bike_plus_s4(nm_bike,    nm_s4,    bike_rns)
veh   = combine_bike_plus_s4(veh_bike,   veh_s4,   bike_rns)
drv   = combine_bike_plus_s4(drv_bike,   drv_s4,   bike_rns)
pax   = combine_bike_plus_s4(pax_bike,   pax_s4,   bike_rns)
vio   = combine_bike_plus_s4(vio_bike,   vio_s4,   bike_rns)
btype = combine_bike_plus_s4(btype_bike, btype_s4, bike_rns)

# Diagnostic-only counts of which side each combined table actually pulled
# rows from -- computed the same way combine_bike_plus_s4() itself decides
# (isin(bike_rns)), not re-derived from overlap_bike_rns, so this always
# sums to len(combined) exactly.
def _kept_counts(df_bike, df_s4, bike_rns, key_col="REPORT_NUMBER"):
    bike_ids = clean_id(df_bike[key_col])
    s4_ids   = clean_id(df_s4[key_col])
    return int(bike_ids.isin(bike_rns).sum()), int((~s4_ids.isin(bike_rns)).sum())

ce_bike_n, ce_s4_n = _kept_counts(ce_bike, ce_s4, bike_rns)
print(f"  Combined crash_event : {len(ce):,} rows "
      f"({ce_bike_n:,} from S4_Crash_bicycle + {ce_s4_n:,} from Signal4Data, deduped on REPORT_NUMBER)")
print(f"  Combined non_motorist: {len(nm):,} rows")
print(f"  Combined vehicle     : {len(veh):,} rows")
print(f"  Combined driver      : {len(drv):,} rows")
print(f"  Combined passenger   : {len(pax):,} rows")
print(f"  Combined violation   : {len(vio):,} rows")
print(f"  Combined bicycle_typing_20 : {len(btype):,} rows")

# ===========================================================================
# 3. MODE CLASSIFICATION (on the combined tables above)
#    - Bicycle  : REPORT_NUMBER in bike_rns (see section 5)
#    - E-Bike / E-Scooter : Qwen narrative classification (authoritative)
#    - Other    : everything else (pedestrian-only, single-vehicle, animal,
#      MV-only, 'Other Cyclist', etc). 'Other Cyclist' in non_motorist is
#      deliberately NOT mapped to E-Bike because that code also covers
#      unicycles, tricycles, cargo bikes, para-cycles, etc.
# ===========================================================================
print("\n=== Classifying modes ===")
nm_mode = (nm.groupby("REPORT_NUMBER")["NON_MOTORIST_DESCRIPTION_CODE"]
             .apply(lambda x: x.tolist())
             .reset_index(name="nm_desc_list"))
ce = ce.merge(nm_mode, on="REPORT_NUMBER", how="left")
ce["nm_desc_list"] = ce["nm_desc_list"].apply(lambda x: x if isinstance(x,list) else [])

def has_desc(lst, val):
    return any(val in str(d) for d in lst)

# Diagnostic-only now (no longer decides Bicycle -- see bike_rns above).
ce["nm_has_bicyclist"]    = ce["nm_desc_list"].apply(lambda x: has_desc(x,"Bicyclist"))
ce["nm_has_othercyclist"] = ce["nm_desc_list"].apply(lambda x: has_desc(x,"Other Cyclist"))
ce["REPORT_NUMBER_str"]   = ce["REPORT_NUMBER"].astype(str)

def classify_mode(row):
    rn = row["REPORT_NUMBER_str"]
    if rn in ebike_escoot_rns:          # Qwen narrative: authoritative for E-Bike/E-Scooter
        return qwen_lookup[rn]
    if rn in bike_rns:                  # S4_Crash_bicycle population (+ Qwen-only edge cases)
        return "Bicycle"
    return "Other"
    # NOTE: 'Other Cyclist' is left as 'Other' intentionally.
    # It can include unicycles, para-cycles, tricycles, cargo bikes, etc.

def classification_basis(row):
    rn = row["REPORT_NUMBER_str"]
    if rn in ebike_escoot_rns:
        return f"Qwen LLM | multilabel_ebike.xlsx + multilabel_RegBike.xlsx | Classification='{row['Classification_raw']}'"
    if rn in bike_pop_rns:
        return "S4_Crash_bicycle population (crash tables/crash_event.csv)"
    if rn in bike_rns:
        return "Qwen LLM 'Bicyclist' (not in S4_Crash_bicycle population -- see [CHECK] log)"
    if row["nm_has_othercyclist"]:
        return "Signal4Data | non_motorist.csv | NON_MOTORIST_DESCRIPTION_CODE='Other Cyclist' -> Other (NOT E-Bike)"
    return "Signal4Data | crash_event.csv | No bicycle/cyclist/e-bike/e-scooter indicator -> Other"

ce["Classification_raw"]   = ce["REPORT_NUMBER_str"].map(qwen_raw_lookup)
ce["MODE"]                 = ce.apply(classify_mode, axis=1)
ce["classification_basis"] = ce.apply(classification_basis, axis=1)

mode_counts = ce["MODE"].value_counts()
print(mode_counts)
print()
in_narratives = ce["REPORT_NUMBER_str"].isin(qwen_lookup.keys())
for mode in ["Bicycle","E-Bike","E-Scooter","Other"]:
    total       = (ce["MODE"]==mode).sum()
    from_qwen   = ((ce["MODE"]==mode) & in_narratives).sum()
    from_struct = ((ce["MODE"]==mode) & ~in_narratives).sum()
    print(f"  {mode}: {total:,} total | {from_qwen:,} Qwen | {from_struct:,} structural/S4_Crash_bicycle")
print()
print("  IMPORTANT: Bicycle in visualisations does NOT include 'Other Cyclist' records.")
print("  'Bicycle' = S4_Crash_bicycle population, minus any Qwen E-Bike/E-Scooter overrides.")

# Subsets
bike_df   = ce[ce["MODE"]=="Bicycle"].copy()
ebike_df  = ce[ce["MODE"]=="E-Bike"].copy()
escoot_df = ce[ce["MODE"]=="E-Scooter"].copy()
micro_df  = ce[ce["MODE"].isin(ACTIVE_MODES)].copy()

# nm with mode
nm2 = nm.merge(ce[["REPORT_NUMBER","MODE"]], on="REPORT_NUMBER", how="left")
nm2["MODE"] = nm2["MODE"].fillna("Other")
nm_active = nm2[nm2["MODE"].isin(ACTIVE_MODES)].copy()
nm_active_valid = nm_active[nm_active["S4_AGE_AT_TIME_OF_CRASH"].between(1,100)]

# mode_classification_detail.csv
nm_desc_detail = nm_desc.copy()
detail_df = ce[ce["MODE"].isin(ACTIVE_MODES)][
    ["REPORT_NUMBER_str","MODE","S4_CRASH_TYPE","Classification_raw",
     "nm_has_bicyclist","nm_has_othercyclist","classification_basis","CRASH_YEAR","COUNTY_NAME"]
].copy()
detail_df.rename(columns={
    "REPORT_NUMBER_str":"REPORT_NUMBER",
    "Classification_raw":"QWEN_CLASS",
    "classification_basis":"HOW_MODE_WAS_DETERMINED"
}, inplace=True)
detail_df["IN_QWEN_NARRATIVES"] = detail_df["REPORT_NUMBER"].isin(qwen_lookup.keys())
detail_df["SOURCE_FILES"] = detail_df.apply(
    lambda r: ("multilabel_ebike.xlsx + multilabel_RegBike.xlsx" if r["IN_QWEN_NARRATIVES"]
               else ("S4_Crash_bicycle" if r["REPORT_NUMBER"] in bike_pop_rns else "crash_event.csv")),
    axis=1)
detail_df = detail_df.merge(nm_desc_detail, on="REPORT_NUMBER", how="left")
detail_df["NM_DESCRIPTION_CODES"] = detail_df["NM_DESCRIPTION_CODES"].fillna("(not in non_motorist)")
detail_df.to_csv(os.path.join(TABS,"mode_classification_detail.csv"), index=False)
print(f"\n  mode_classification_detail.csv: {len(detail_df):,} rows (active modes only)")

# nm_description reference
nm_desc_ref = nm["NON_MOTORIST_DESCRIPTION_CODE"].value_counts().reset_index()
nm_desc_ref.columns=["NON_MOTORIST_DESCRIPTION_CODE","count"]
nm_desc_ref["how_used_in_mode_classification"] = nm_desc_ref["NON_MOTORIST_DESCRIPTION_CODE"].map({
    "Bicyclist":      "-> Bicycle (structural)",
    "Other Cyclist":  "-> Other  (NOT E-Bike; could be unicycle/tricycle/cargo bike/etc.)",
}).fillna("-> Other")
nm_desc_ref.to_csv(os.path.join(TABS,"nm_description_code_reference.csv"), index=False)

# metric_sources
metric_rows=[
    ("Mode (narrative records)",f"Final mode for {len(narr_q):,} multilabel-classified crashes",
     "multilabel_ebike.xlsx + multilabel_RegBike.xlsx","Classification",
     "Mapped: ebike->E-Bike, escooter->E-Scooter, bicyclist->Bicycle, other->Other"),
    ("Mode (structural – non-narrative records)","Final mode for ~94k non-narrative crashes",
     "crash_event.csv + non_motorist.csv","S4_CRASH_TYPE; NON_MOTORIST_DESCRIPTION_CODE",
     "Bicycle only: S4_CRASH_TYPE='Bicycle' OR code='Bicyclist'. E-Bike/E-Scooter NOT in structural."),
    ("Age","Age at time of crash for non-motorists","non_motorist.csv","S4_AGE_AT_TIME_OF_CRASH",
     "Filtered to active modes by joining with crash_event MODE"),
    ("Gender","Sex of non-motorists","non_motorist.csv","SEX","Male/Female only"),
    ("Injury Severity","Injury outcome","non_motorist.csv","INJURY_SEVERITY",
     "Non-motorist level; mapped to 5 categories"),
    ("Day vs Night","Crash timing","crash_event.csv","S4_DAY_OR_NIGHT","DAY/NIGHT"),
    ("Hour of Day","Crash hour","crash_event.csv","CRASH_DATE_AND_TIME","Hour extracted"),
    ("Day of Week","Crash day","crash_event.csv","CRASH_DATE_AND_TIME","Day name extracted"),
    ("Month","Crash month","crash_event.csv","CRASH_DATE_AND_TIME","Month extracted"),
    ("Year Trend","Annual crash counts","crash_event.csv","CRASH_YEAR",""),
    ("Intersection vs Segment","Location type","crash_event.csv","S4_IS_INTERSECTION_RELATED","Y/N"),
    ("Speed Limit","Posted speed","vehicle.csv","POSTED_SPEED",
     "Linked via REPORT_NUMBER; 5-80 mph"),
    ("Road Type","Trafficway type","vehicle.csv","TRAFFICWAY_CODE",""),
    ("County","County","crash_event.csv","COUNTY_NAME",""),
    ("Light Condition","Lighting","crash_event.csv","LIGHT_CONDITION",""),
    ("Weather","Weather","crash_event.csv","WEATHER_CONDITION",""),
    ("MV Involvement","Motor vehicle involved","vehicle.csv","TYPE_OF_VEHICLE",
     "MV = any vehicle NOT 'Pedalcycle/pedal/bicycle'"),
    ("Crash Group 06a","Bicycle crash scenario type",
     "bicycle_typing_20.csv (ped bike typing folder)","S4_CRASH_GROUP_DESCRIPTION",
     "Covers ALL crashes in the typing file (mostly bicycle+MV scenarios). "
     "Shows WHAT happened (e.g., Motorist Failed to Yield). NOT the same as 06d."),
    ("Contributing Factors 06d","Road/environment conditions","crash_event.csv",
     "ROAD_CIRCUMSTANCES_1; ENVIRONMENT_CIRCUMSTANCES_1",
     "ALL active modes (Bicycle+E-Bike+E-Scooter). Shows WHY conditions contributed. "
     "DIFFERENT from 06a: different source, different question, different scope."),
    ("Lat/Lon","Crash location","crash_event.csv",
     "S4_LATITUDE; S4_LONGITUDE; LATITUDE; LONGITUDE; S4_GEOLOCATION_CURRENT",
     "S4 coords ~98.5% complete; raw HSMV coords ~55% complete."),
    ("Qwen Classification",f"Qwen/multilabel prediction ({len(narr_q):,} narratives)",
     "multilabel_ebike.xlsx + multilabel_RegBike.xlsx","Classification",
     "4 raw classes: ebike, escooter, bicyclist, other"),
]
pd.DataFrame(metric_rows,
             columns=["Metric","Description","Primary_File","Column(s)","Notes"]
             ).to_csv(os.path.join(TABS,"metric_sources.csv"), index=False)

# ===========================================================================
# 4. DATE PARSING
# ===========================================================================
ce["CRASH_DT"] = pd.to_datetime(ce["CRASH_DATE_AND_TIME"], errors="coerce")
ce["HOUR"]  = ce["CRASH_DT"].dt.hour
ce["DOW"]   = ce["CRASH_DT"].dt.day_name()
ce["MONTH"] = ce["CRASH_DT"].dt.month
ce["YEAR"]  = ce["CRASH_DT"].dt.year.fillna(ce["CRASH_YEAR"])

micro_df = ce[ce["MODE"].isin(ACTIVE_MODES)].copy()

sev_order = ["No Injury","Possible","Non-Incapacitating","Incapacitating","Fatal"]
sev_map   = {"None":"No Injury","No Injury":"No Injury","Possible":"Possible",
             "Non-Incapacitating":"Non-Incapacitating","Incapacitating":"Incapacitating","Fatal":"Fatal"}

# ===========================================================================
# 5. SECTION 01 – OVERVIEW (active modes only; no 'Other' bar)
# ===========================================================================
print("\n=== 01 Overview ===")

active_counts = ce[ce["MODE"].isin(ACTIVE_MODES)]["MODE"].value_counts().reindex(ACTIVE_MODES, fill_value=0)

fig, axes = plt.subplots(1,2, figsize=(13,6))
# Pie
ax = axes[0]
colors3 = [PALETTE[m] for m in ACTIVE_MODES]
wedges, texts, autotexts = ax.pie(
    active_counts.values, labels=None, colors=colors3,
    autopct=lambda p: f"{p:.1f}%", startangle=140, pctdistance=0.75,
    wedgeprops=dict(linewidth=1.5, edgecolor="white"))
for at in autotexts: at.set_fontsize(11); at.set_color("#111111")
handles = [mpatches.Patch(facecolor=c, label=f"{m}  ({n:,})")
           for m,c,n in zip(ACTIVE_MODES,colors3,active_counts.values)]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5,-0.08), ncol=1, fontsize=10)
ax.set_title("Active-Mode Crash Distribution\n(Bicycle / E-Bike / E-Scooter)",
             fontsize=12, fontweight="bold")
# Bar
ax2 = axes[1]
bars = ax2.barh(ACTIVE_MODES, active_counts.values, color=colors3, edgecolor="white")
for bar,n in zip(bars, active_counts.values):
    ax2.text(bar.get_width()+active_counts.max()*0.01,
             bar.get_y()+bar.get_height()/2,
             f"{n:,}", va="center", fontsize=10)
ax2.set_xlabel("Number of Crashes")
ax2.set_title("Active-Mode Crash Counts", fontsize=12, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)
ax2.set_xlim(0, active_counts.max()*1.15)
fig.suptitle("Source: crash_event.csv + non_motorist.csv + multilabel_ebike.xlsx + multilabel_RegBike.xlsx\n"
             "'Other' class excluded from these charts.",
             fontsize=9, y=0.02, style="italic")
plt.tight_layout(rect=[0,0.05,1,1])
save(fig, "01_overview", "01a_mode_distribution")

fig, ax = plt.subplots(figsize=(11,5))
for mode in ACTIVE_MODES:
    sub = ce[ce["MODE"]==mode].groupby("YEAR").size()
    ax.plot(sub.index, sub.values, marker="o", label=mode, color=PALETTE[mode], linewidth=2)
ax.set_xlabel("Year"); ax.set_ylabel("Crashes")
ax.set_title("Annual Crash Trend by Mode\nSource: crash_event.csv | CRASH_DATE_AND_TIME",
             fontsize=12, fontweight="bold")
ax.legend(title="Mode"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save(fig, "01_overview", "01c_annual_trend")

# MV involvement
veh2      = veh.merge(ce[["REPORT_NUMBER","MODE"]], on="REPORT_NUMBER", how="left")
veh2["MODE"] = veh2["MODE"].fillna("Other")
veh_mv    = veh2[veh2["TYPE_OF_VEHICLE"].notna()]
veh_mv2   = veh_mv[~veh_mv["TYPE_OF_VEHICLE"].str.contains("Pedalcycle|pedal|bicycle",case=False,na=False)]
mv_per    = veh_mv2.groupby("REPORT_NUMBER").size().reset_index(name="mv_count")
micro_df  = micro_df.merge(mv_per, on="REPORT_NUMBER", how="left")
micro_df["mv_involved"] = micro_df["mv_count"].fillna(0) > 0

mv_mode = micro_df.groupby(["MODE","mv_involved"]).size().unstack(fill_value=0)
mv_mode.columns = ["No MV","MV Involved"]
mv_pct  = mv_mode.div(mv_mode.sum(axis=1),axis=0)*100

fig, ax = plt.subplots(figsize=(8,5))
x=np.arange(len(ACTIVE_MODES)); w=0.35
b1=ax.bar(x-w/2,[mv_pct.loc[m,"MV Involved"] if m in mv_pct.index else 0 for m in ACTIVE_MODES],
          w, label="MV Involved", color="#D32F2F", edgecolor="white")
b2=ax.bar(x+w/2,[mv_pct.loc[m,"No MV"] if m in mv_pct.index else 0 for m in ACTIVE_MODES],
          w, label="No MV",       color="#388E3C", edgecolor="white")
for bar in list(b1)+list(b2):
    h=bar.get_height()
    if h>2: ax.text(bar.get_x()+bar.get_width()/2,h+0.8,f"{h:.0f}%",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(ACTIVE_MODES)
ax.set_ylabel("Percentage (%)"); ax.set_ylim(0,110)
ax.set_title("Motor-Vehicle Involvement by Mode\nSource: vehicle.csv | TYPE_OF_VEHICLE",
             fontsize=12, fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
save(fig, "01_overview", "01d_mv_involvement")

# ===========================================================================
# 6. SECTION 02 – WHO
# ===========================================================================
print("\n=== 02 Who ===")
fig, ax = plt.subplots(figsize=(9,5))
sns.violinplot(data=nm_active_valid[nm_active_valid["MODE"].isin(ACTIVE_MODES)],
               x="MODE", y="S4_AGE_AT_TIME_OF_CRASH",
               order=ACTIVE_MODES, palette=[PALETTE[m] for m in ACTIVE_MODES],
               inner="quartile", cut=0, ax=ax)
ax.set_xlabel("Mode"); ax.set_ylabel("Age")
ax.set_title("Age Distribution by Mode (Non-Motorists)\nSource: non_motorist.csv | S4_AGE_AT_TIME_OF_CRASH",
             fontsize=12, fontweight="bold")
ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig, "02_who", "02a_age_violin")

fig, axes = plt.subplots(1,3,figsize=(14,4),sharey=False)
for ax,mode in zip(axes,ACTIVE_MODES):
    ages = nm_active_valid[nm_active_valid["MODE"]==mode]["S4_AGE_AT_TIME_OF_CRASH"]
    med = ages.median()
    ax.hist(ages, bins=range(0,101,5), color=PALETTE[mode], edgecolor="white", linewidth=0.7)
    ax.axvline(med, color="#B71C1C", linewidth=2, linestyle="--", label=f"Median {med:.0f}")
    ax.set_title(mode, fontsize=12, fontweight="bold", color=PALETTE[mode])
    ax.set_xlabel("Age"); ax.set_ylabel("Count"); ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
fig.suptitle("Age Histograms by Mode | Source: non_motorist.csv | S4_AGE_AT_TIME_OF_CRASH",
             fontsize=10, y=1.02)
plt.tight_layout()
save(fig, "02_who", "02b_age_histograms")

gm = (nm_active[nm_active["SEX"].isin(["Male","Female"]) & nm_active["MODE"].isin(ACTIVE_MODES)]
      .groupby(["MODE","SEX"]).size().unstack(fill_value=0))
gm_pct = gm.div(gm.sum(axis=1),axis=0)*100
fig,ax = plt.subplots(figsize=(8,5))
x=np.arange(len(ACTIVE_MODES)); w=0.35
b1=ax.bar(x-w/2,[gm_pct.loc[m,"Male"]   if m in gm_pct.index else 0 for m in ACTIVE_MODES],
          w,label="Male",  color=MALE_COLOR,  edgecolor="white")
b2=ax.bar(x+w/2,[gm_pct.loc[m,"Female"] if m in gm_pct.index else 0 for m in ACTIVE_MODES],
          w,label="Female",color=FEMALE_COLOR,edgecolor="white")
for bar in list(b1)+list(b2):
    h=bar.get_height()
    if h>2: ax.text(bar.get_x()+bar.get_width()/2,h+0.8,f"{h:.0f}%",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(ACTIVE_MODES)
ax.set_ylabel("Percentage (%)"); ax.set_ylim(0,110)
ax.set_title("Gender Distribution by Mode\nSource: non_motorist.csv | SEX",
             fontsize=12, fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig, "02_who", "02c_gender_by_mode")

gm.to_csv(os.path.join(TABS,"gender_by_mode.csv"))
nm_active_valid[nm_active_valid["MODE"].isin(ACTIVE_MODES)]\
    .groupby("MODE")["S4_AGE_AT_TIME_OF_CRASH"].describe().round(1)\
    .to_csv(os.path.join(TABS,"age_summary_by_mode.csv"))

# 02d -- Age Distribution by Band, by Mode (% within mode, so modes with
# different sample sizes are still comparable on the SHAPE of their age
# distribution, not just swamped by whichever mode has more rows).
age_bins   = [0,12,17,24,34,44,54,64,120]
age_labels = ["0-12","13-17","18-24","25-34","35-44","45-54","55-64","65+"]
nm_active_valid = nm_active_valid.copy()
nm_active_valid["AGE_BAND"] = pd.cut(nm_active_valid["S4_AGE_AT_TIME_OF_CRASH"],
                                      bins=age_bins, labels=age_labels, right=True, include_lowest=True)
age_band_mode = (nm_active_valid[nm_active_valid["MODE"].isin(ACTIVE_MODES)]
                  .groupby(["MODE","AGE_BAND"], observed=True).size().unstack(fill_value=0)
                  .reindex(ACTIVE_MODES, fill_value=0).reindex(columns=age_labels, fill_value=0))
age_band_pct = age_band_mode.div(age_band_mode.sum(axis=1), axis=0) * 100
age_band_mode.to_csv(os.path.join(TABS,"age_band_by_mode.csv"))
age_band_pct.to_csv(os.path.join(TABS,"age_band_by_mode_pct.csv"))

fig, ax = plt.subplots(figsize=(11,5))
x = np.arange(len(age_labels)); width = 0.8/len(ACTIVE_MODES)
for i, mode in enumerate(ACTIVE_MODES):
    ax.bar(x+i*width, age_band_pct.loc[mode].values, width, label=mode,
           color=PALETTE[mode], edgecolor="white")
ax.set_xticks(x + width*(len(ACTIVE_MODES)-1)/2); ax.set_xticklabels(age_labels)
ax.set_ylabel("% of that mode's non-motorists"); pct_y(ax)
ax.set_title("Age Distribution by Band, by Mode\nSource: non_motorist.csv | S4_AGE_AT_TIME_OF_CRASH",
             fontsize=12, fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig, "02_who", "02d_age_band_by_mode")

# 02e/02f/02g -- Age Distribution by Injury Severity, as three SEPARATE
# figures (one per mode) rather than one combined chart, since the earlier
# observation ("people in fatal crashes skew older") is easiest to see
# within a single mode rather than averaged across all three.
nm_active_valid["SEV_CLEAN"] = nm_active_valid["INJURY_SEVERITY"].map(sev_map).fillna("Unknown")
for mode, fname_suffix in [("Bicycle","02e"), ("E-Bike","02f"), ("E-Scooter","02g")]:
    sub = nm_active_valid[(nm_active_valid["MODE"]==mode) &
                           (nm_active_valid["SEV_CLEAN"].isin(sev_order))]
    if len(sub) < 10:
        print(f"  [SKIP] Age by Injury Severity ({mode}): only {len(sub)} valid rows, too few to plot")
        continue
    fig, ax = plt.subplots(figsize=(8,5))
    sns.violinplot(data=sub, x="SEV_CLEAN", y="S4_AGE_AT_TIME_OF_CRASH",
                    order=[s for s in sev_order if s in sub["SEV_CLEAN"].unique()],
                    palette=SEVERITY_COLORS, inner="quartile", cut=0, ax=ax)
    ax.set_xlabel("Injury Severity"); ax.set_ylabel("Age")
    ax.set_title(f"Age Distribution by Injury Severity -- {mode} (n={len(sub):,})\n"
                 "Source: non_motorist.csv | S4_AGE_AT_TIME_OF_CRASH, INJURY_SEVERITY",
                 fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
    save(fig, "02_who", f"{fname_suffix}_age_by_severity_{mode.lower().replace('-','')}")
    sub.groupby("SEV_CLEAN", observed=True)["S4_AGE_AT_TIME_OF_CRASH"].describe().round(1)\
        .to_csv(os.path.join(TABS, f"age_by_severity_{mode.lower().replace('-','')}.csv"))

# ===========================================================================
# 7. SECTION 03 – WHEN
# ===========================================================================
print("\n=== 03 When ===")
# DAY VS NIGHT DEFINITION: this script does NOT compute Day/Night itself --
# it passes through the pre-populated S4_DAY_OR_NIGHT field from
# crash_event.csv, which FLHSMV/Signal4 assigns at data-entry/geocoding time.
# We do not have the exact rule (e.g. sunrise/sunset vs. civil twilight vs.
# a fixed clock cutoff) FLHSMV used to set that field in our extract --
# PENDING confirmation from FLHSMV/Signal4 if an exact definition is needed
# for the writeup. See README Section 23.
def day_night(row):
    v=str(row.get("S4_DAY_OR_NIGHT","")).upper()
    return "Day" if v=="DAY" else ("Night" if v=="NIGHT" else "Unknown")

micro_df["DAY_NIGHT"] = micro_df.apply(day_night, axis=1)
dn = micro_df[micro_df["DAY_NIGHT"].isin(["Day","Night"])]\
     .groupby(["MODE","DAY_NIGHT"]).size().unstack(fill_value=0)
dn_pct = dn.div(dn.sum(axis=1),axis=0)*100

fig,ax = plt.subplots(figsize=(8,5))
x=np.arange(len(ACTIVE_MODES)); w=0.35
b1=ax.bar(x-w/2,[dn_pct.loc[m,"Day"]   if m in dn_pct.index and "Day"   in dn_pct.columns else 0 for m in ACTIVE_MODES],
          w,label="Day",  color=DAY_COLOR,  edgecolor="white")
b2=ax.bar(x+w/2,[dn_pct.loc[m,"Night"] if m in dn_pct.index and "Night" in dn_pct.columns else 0 for m in ACTIVE_MODES],
          w,label="Night",color=NIGHT_COLOR,edgecolor="white")
for bar in list(b1)+list(b2):
    h=bar.get_height()
    if h>2: ax.text(bar.get_x()+bar.get_width()/2,h+0.8,f"{h:.0f}%",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(ACTIVE_MODES); ax.set_ylim(0,110)
ax.set_title("Day vs Night Crashes\nSource: crash_event.csv | S4_DAY_OR_NIGHT",
             fontsize=12, fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig, "03_when", "03a_day_night_by_mode")

hour_mode = micro_df.groupby(["MODE","HOUR"]).size().unstack(fill_value=0).reindex(ACTIVE_MODES,fill_value=0)
fig,ax = plt.subplots(figsize=(14,4))
sns.heatmap(hour_mode, cmap="YlOrRd", ax=ax, linewidths=0.5, linecolor="white",
            annot=True, fmt="d", annot_kws={"size":7},
            cbar_kws={"label":"Crash Count"})
ax.set_xlabel("Hour (0-23)"); ax.set_ylabel("Mode")
ax.set_title("Crashes by Hour | Source: crash_event.csv | CRASH_DATE_AND_TIME",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save(fig, "03_when", "03b_hour_heatmap")

# --- Peak-hour check: is E-Bike's 3-4pm clustering just school-dismissal /
# PM commute overlap (which ALL active modes should show), or is it
# disproportionately concentrated for E-Bike specifically? This does not
# determine causation -- it only tells us whether the peak is shared or
# E-Bike-specific, which narrows the hypothesis. See README Section 22.
if "E-Bike" in hour_mode.index:
    peak_hr = int(hour_mode.loc["E-Bike"].idxmax())
    print(f"\n  E-Bike peak crash hour: {peak_hr}:00-{peak_hr+1}:00 "
          f"({hour_mode.loc['E-Bike'].max():,} crashes, "
          f"{hour_mode.loc['E-Bike'].max()/hour_mode.loc['E-Bike'].sum()*100:.1f}% of all E-Bike crashes)")
    for mode in ACTIVE_MODES:
        if mode in hour_mode.index and hour_mode.loc[mode].sum() > 0:
            share_at_peak = hour_mode.loc[mode].get(peak_hr, 0) / hour_mode.loc[mode].sum() * 100
            print(f"    {mode} share of crashes at hour {peak_hr}: {share_at_peak:.1f}%")
    print("  If all three modes show a similar share at this hour, the clustering is likely")
    print("  the shared PM school-dismissal/commute peak rather than an E-Bike-specific pattern")
    print("  (e.g. school-age e-bike riders). If E-Bike's share is notably higher than Bicycle's")
    print("  and E-Scooter's, that supports an E-Bike-specific hypothesis (school commuting,")
    print("  delivery-gig timing, etc.) worth following up with a school-hours cross-tab.")

fig, ax = plt.subplots(figsize=(11, 4.5))
for mode in ACTIVE_MODES:
    if mode in hour_mode.index and hour_mode.loc[mode].sum() > 0:
        vals = hour_mode.loc[mode].reindex(range(24), fill_value=0)
        ax.plot(vals.index, vals.values / vals.sum() * 100, marker="o", markersize=3,
                label=mode, color=PALETTE[mode], linewidth=2)
ax.axvspan(15, 17, color="grey", alpha=0.15, label="3-4pm window")
ax.set_xlabel("Hour (0-23)"); ax.set_ylabel("% of that mode's crashes")
ax.set_xticks(range(0, 24, 2))
ax.set_title("Crash Timing by Hour, as % Within Mode (Shape Comparison)\n"
             "Source: crash_event.csv | CRASH_DATE_AND_TIME -- shaded band flags the 3-4pm question",
             fontsize=11, fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig, "03_when", "03b2_hour_pct_within_mode")

dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_mode  = micro_df.groupby(["MODE","DOW"]).size().unstack(fill_value=0)\
            .reindex(ACTIVE_MODES,fill_value=0).reindex(columns=dow_order,fill_value=0)
fig,axes = plt.subplots(1,3,figsize=(15,4),sharey=False)
for ax,mode in zip(axes,ACTIVE_MODES):
    vals=[dow_mode.loc[mode,d] if mode in dow_mode.index and d in dow_mode.columns else 0 for d in dow_order]
    ax.bar([d[:3] for d in dow_order],vals,color=PALETTE[mode],edgecolor="white")
    ax.set_title(mode,fontsize=12,fontweight="bold",color=PALETTE[mode])
    ax.set_xlabel("Day"); ax.set_ylabel("Crashes"); ax.spines[["top","right"]].set_visible(False)
fig.suptitle("Crashes by Day of Week | crash_event.csv | CRASH_DATE_AND_TIME",fontsize=10,y=1.02)
plt.tight_layout(); save(fig,"03_when","03c_day_of_week")

month_names=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_mode = micro_df.groupby(["MODE","MONTH"]).size().unstack(fill_value=0).reindex(ACTIVE_MODES,fill_value=0)
fig,ax=plt.subplots(figsize=(11,5))
for mode in ACTIVE_MODES:
    vals=[month_mode.loc[mode,m] if mode in month_mode.index and m in month_mode.columns else 0 for m in range(1,13)]
    ax.plot(month_names,vals,marker="o",label=mode,color=PALETTE[mode],linewidth=2)
ax.set_xlabel("Month"); ax.set_ylabel("Crashes")
ax.set_title("Monthly Pattern | crash_event.csv | CRASH_DATE_AND_TIME",fontsize=12,fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig,"03_when","03d_monthly_pattern")

# ===========================================================================
# 8. SECTION 04 – WHERE
# ===========================================================================
print("\n=== 04 Where ===")
def inter_flag(row):
    v=str(row.get("S4_IS_INTERSECTION_RELATED","")).upper()
    return "Intersection" if v=="Y" else ("Segment" if v=="N" else "Unknown")

micro_df["LOC_TYPE"] = micro_df.apply(inter_flag,axis=1)
loc_mode = micro_df[micro_df["LOC_TYPE"].isin(["Intersection","Segment"])]\
           .groupby(["MODE","LOC_TYPE"]).size().unstack(fill_value=0)
loc_pct = loc_mode.div(loc_mode.sum(axis=1),axis=0)*100

fig,ax=plt.subplots(figsize=(8,5))
x=np.arange(len(ACTIVE_MODES)); w=0.35
b1=ax.bar(x-w/2,[loc_pct.loc[m,"Intersection"] if m in loc_pct.index else 0 for m in ACTIVE_MODES],
          w,label="Intersection",color="#5C6BC0",edgecolor="white")
b2=ax.bar(x+w/2,[loc_pct.loc[m,"Segment"]      if m in loc_pct.index else 0 for m in ACTIVE_MODES],
          w,label="Segment",     color="#26A69A",edgecolor="white")
for bar in list(b1)+list(b2):
    h=bar.get_height()
    if h>2: ax.text(bar.get_x()+bar.get_width()/2,h+0.8,f"{h:.0f}%",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(ACTIVE_MODES); ax.set_ylim(0,110)
ax.set_title("Intersection vs Segment\nSource: crash_event.csv | S4_IS_INTERSECTION_RELATED",
             fontsize=12,fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig,"04_where","04a_intersection_segment")

veh_micro = veh2[veh2["MODE"].isin(ACTIVE_MODES) & veh2["POSTED_SPEED"].between(5,80)]
fig,ax=plt.subplots(figsize=(9,5))
sns.violinplot(data=veh_micro,x="MODE",y="POSTED_SPEED",
               order=ACTIVE_MODES,palette=[PALETTE[m] for m in ACTIVE_MODES],
               inner="quartile",cut=0,ax=ax)
ax.set_xlabel("Mode"); ax.set_ylabel("Posted Speed (mph)")
ax.set_title("Speed Limit at Crash Location\nSource: vehicle.csv | POSTED_SPEED",
             fontsize=12,fontweight="bold")
ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig,"04_where","04b_speed_limit_violin")

tway = veh_micro.copy()
fig,ax=plt.subplots(figsize=(11,6))
pivot_tw = pct_within_group_bar(tway, "MODE", "TRAFFICWAY_CODE", ax, top_n=8, order=ACTIVE_MODES, palette=PALETTE)
ax.set_title("Road Type -- % Within Mode\nSource: vehicle.csv | TRAFFICWAY_CODE",
             fontsize=12,fontweight="bold")
plt.tight_layout()
save(fig,"04_where","04c_road_type")
pivot_tw.to_csv(os.path.join(TABS,"road_type_by_mode_pct.csv"))

county_mode = micro_df.groupby(["MODE","COUNTY_NAME"]).size().reset_index(name="count")
top_co      = county_mode.groupby("COUNTY_NAME")["count"].sum().nlargest(12).index
co_top      = county_mode[county_mode["COUNTY_NAME"].isin(top_co)]
pivot_co    = co_top.pivot_table(index="COUNTY_NAME",columns="MODE",values="count",fill_value=0)
pivot_co    = pivot_co.reindex(columns=[m for m in ACTIVE_MODES if m in pivot_co.columns])
pivot_co    = pivot_co.sort_values(pivot_co.columns[0],ascending=True)
fig,ax=plt.subplots(figsize=(10,8))
left=np.zeros(len(pivot_co))
for mode in [m for m in ACTIVE_MODES if m in pivot_co.columns]:
    vals=pivot_co[mode].values
    ax.barh(pivot_co.index,vals,left=left,color=PALETTE[mode],label=mode,edgecolor="white")
    left+=vals
ax.set_xlabel("Crash Count")
ax.set_title("Top 12 Counties\nSource: crash_event.csv | COUNTY_NAME",fontsize=12,fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig,"04_where","04d_top_counties_stacked")
county_mode.pivot_table(index="COUNTY_NAME",columns="MODE",values="count",fill_value=0)\
           .to_csv(os.path.join(TABS,"county_by_mode.csv"))

for col_name, title, fname, src in [
    ("LIGHT_CONDITION","Light Conditions","04e_light_conditions","crash_event.csv | LIGHT_CONDITION"),
    ("WEATHER_CONDITION","Weather Conditions","04f_weather_conditions","crash_event.csv | WEATHER_CONDITION"),
]:
    fig,ax=plt.subplots(figsize=(11,5))
    pt = pct_within_group_bar(micro_df, "MODE", col_name, ax, top_n=6, order=ACTIVE_MODES, palette=PALETTE, wrap_width=15)
    ax.set_title(f"{title} -- % Within Mode\nSource: {src}",fontsize=12,fontweight="bold")
    plt.tight_layout()
    save(fig,"04_where",fname)
    pt.to_csv(os.path.join(TABS,f"{col_name.lower()}_by_mode_pct.csv"))

# ===========================================================================
# 9. SECTION 05 – SEVERITY
# ===========================================================================
print("\n=== 05 Severity ===")
nm_active["SEV_CLEAN"] = nm_active["INJURY_SEVERITY"].map(sev_map).fillna("Unknown")
sev_mode = nm_active[nm_active["SEV_CLEAN"].isin(sev_order) & nm_active["MODE"].isin(ACTIVE_MODES)]\
           .groupby(["MODE","SEV_CLEAN"]).size().unstack(fill_value=0)\
           .reindex(ACTIVE_MODES,fill_value=0)
sev_pct  = sev_mode.div(sev_mode.sum(axis=1),axis=0)*100

fig,ax=plt.subplots(figsize=(10,6))
left=np.zeros(len(ACTIVE_MODES))
for sev in sev_order:
    if sev in sev_pct.columns:
        vals=[sev_pct.loc[m,sev] if m in sev_pct.index else 0 for m in ACTIVE_MODES]
        ax.barh(ACTIVE_MODES,vals,left=left,label=sev,
                color=SEVERITY_COLORS[sev],edgecolor="white",linewidth=0.8)
        for i,(v,l) in enumerate(zip(vals,left)):
            if v>4: ax.text(l+v/2,i,f"{v:.0f}%",ha="center",va="center",fontsize=9,fontweight="bold")
        left+=np.array(vals)
ax.set_xlabel("%"); ax.set_xlim(0,105)
ax.set_title("Injury Severity by Mode\nSource: non_motorist.csv | INJURY_SEVERITY",
             fontsize=12,fontweight="bold")
ax.legend(title="Severity",bbox_to_anchor=(1.01,1),loc="upper left",fontsize=9)
ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig,"05_severity","05a_severity_stacked_bar")

sc=sev_mode.copy(); sc["total"]=sc.sum(axis=1)
sc["fatal_pct"]=sc.get("Fatal",0)/sc["total"]*100
sc["incap_pct"]=sc.get("Incapacitating",0)/sc["total"]*100
sc[["fatal_pct","incap_pct"]].to_csv(os.path.join(TABS,"severity_rates_by_mode.csv"))
sev_mode.to_csv(os.path.join(TABS,"severity_by_mode.csv"))

fig,ax=plt.subplots(figsize=(8,5))
x=np.arange(len(ACTIVE_MODES)); w=0.35
b1=ax.bar(x-w/2,[sc.loc[m,"fatal_pct"] if m in sc.index else 0 for m in ACTIVE_MODES],
          w,label="Fatality %",color="#B71C1C",edgecolor="white")
b2=ax.bar(x+w/2,[sc.loc[m,"incap_pct"] if m in sc.index else 0 for m in ACTIVE_MODES],
          w,label="Incapacitating %",color="#EF9A9A",edgecolor="white")
for bar in list(b1)+list(b2):
    h=bar.get_height()
    if h>0.1: ax.text(bar.get_x()+bar.get_width()/2,h+0.1,f"{h:.1f}%",ha="center",fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(ACTIVE_MODES)
ax.set_title("Fatality & Incapacitating Rate\nSource: non_motorist.csv | INJURY_SEVERITY",
             fontsize=12,fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig,"05_severity","05b_fatality_incap_rates")

micro_df2=micro_df.merge(ce[["REPORT_NUMBER","S4_CRASH_SEVERITY"]].drop_duplicates(),
                         on="REPORT_NUMBER",how="left",suffixes=("","_y"))
sev_yr=micro_df2.groupby(["YEAR","S4_CRASH_SEVERITY"]).size().unstack(fill_value=0)
sev_yr.to_csv(os.path.join(TABS,"severity_year_trend_counts.csv"))
# Use PROPORTIONS (% of that year's crashes), not raw counts: raw counts mix
# the changing severity MIX together with the separate fact that overall
# crash volume also changed year to year. % isolates the mix question.
sev_yr_pct = sev_yr.div(sev_yr.sum(axis=1), axis=0) * 100
sev_yr_pct.to_csv(os.path.join(TABS,"severity_year_trend_pct.csv"))
fig,ax=plt.subplots(figsize=(11,5))
sev_yr_pct.plot(kind="bar",stacked=True,ax=ax,colormap="RdYlGn_r",edgecolor="white")
ax.set_xlabel("Year"); ax.set_ylabel("% of that year's active-mode crashes")
pct_y(ax)
ax.set_title("Active-Mode Severity Mix Over Time (Proportions)\nSource: crash_event.csv | S4_CRASH_SEVERITY",
             fontsize=12,fontweight="bold")
ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",fontsize=9)
ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig,"05_severity","05c_severity_year_trend")

# 05d -- same data, raw counts, kept as a secondary chart so overall volume
# growth is still visible (just not conflated with the mix question above).
fig,ax=plt.subplots(figsize=(11,5))
sev_yr.plot(kind="bar",stacked=True,ax=ax,colormap="RdYlGn_r",edgecolor="white")
ax.set_xlabel("Year"); ax.set_ylabel("Count")
ax.set_title("Active-Mode Severity Over Years (Raw Counts)\nSource: crash_event.csv | S4_CRASH_SEVERITY",
             fontsize=12,fontweight="bold")
ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",fontsize=9)
ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig,"05_severity","05d_severity_year_trend_counts")

# ===========================================================================
# 10. SECTION 06 - CRASH TYPING
#  06a: from bicycle_typing_20.csv – crash SCENARIO/TYPE (e.g. who failed to yield)
#       Scope: all crashes in the typing file (bicycle-typed crashes from S4)
#       Answers: "What happened?"
#  06d: from crash_event.csv ROAD_CIRCUMSTANCES_1 + ENVIRONMENT_CIRCUMSTANCES_1
#       Scope: ALL active modes (Bicycle + E-Bike + E-Scooter)
#       Answers: "What road/environment CONDITIONS contributed?"
#  KEY DIFFERENCE: different source file, different question, different scope.
# ===========================================================================
print("\n=== 06 Crash Typing ===")
btype["REPORT_NUMBER"]=btype["REPORT_NUMBER"].astype(str)
ce_str = ce[["REPORT_NUMBER_str","MODE"]].rename(columns={"REPORT_NUMBER_str":"REPORT_NUMBER"})
btype2 = btype.merge(ce_str,on="REPORT_NUMBER",how="left")
btype2["MODE"]=btype2["MODE"].fillna("Bicycle")

# 06a – FIXED label overlap
grp_counts = btype2["S4_CRASH_GROUP_DESCRIPTION"].value_counts().head(15)
# Use short labels (strip common prefix/suffix)
short_labels = []
for t in grp_counts.index:
    t2 = str(t).replace("Bicyclist ","Cycl. ").replace("Motorist ","Motorist ").replace(" - "," -\n")
    short_labels.append("\n".join(textwrap.wrap(t2, 30)))

fig, ax = plt.subplots(figsize=(12, 9))
colors_grp = plt.cm.Blues(np.linspace(0.35, 0.9, len(grp_counts)))[::-1]
y_pos = np.arange(len(grp_counts))
bars = ax.barh(y_pos, grp_counts.values, color=colors_grp, edgecolor="white", height=0.7)
for bar, n in zip(bars, grp_counts.values):
    ax.text(bar.get_width() + grp_counts.max()*0.005,
            bar.get_y()+bar.get_height()/2,
            f"{n:,}", va="center", fontsize=8)
ax.set_yticks(y_pos)
ax.set_yticklabels(short_labels, fontsize=8, linespacing=1.2)
ax.set_xlabel("Number of Crashes")
ax.set_title(
    "06a: Top 15 Bicycle Crash Groups (S4 Typing)\n"
    "Source: bicycle_typing_20.csv (ped bike typing folder) | S4_CRASH_GROUP_DESCRIPTION\n"
    "Covers bicycle-typed crashes. Shows crash SCENARIO (who failed to yield, etc.).\n"
    "NOTE: 06a differs from 06d -- see 06d for road/environment contributing conditions.",
    fontsize=11, fontweight="bold", pad=10)
ax.spines[["top","right"]].set_visible(False)
plt.subplots_adjust(left=0.38, right=0.95, top=0.85, bottom=0.08)
save(fig, "06_crash_typing", "06a_crash_group_distribution")

ct_counts = btype2["S4_CRASH_TYPE_DESCRIPTION"].value_counts().dropna().head(15)
short_ct  = ["\n".join(textwrap.wrap(str(t),30)) for t in ct_counts.index]
fig,ax=plt.subplots(figsize=(12,9))
y_pos=np.arange(len(ct_counts))
colors_ct=plt.cm.Greens(np.linspace(0.35,0.9,len(ct_counts)))[::-1]
bars=ax.barh(y_pos,ct_counts.values,color=colors_ct,edgecolor="white",height=0.7)
for bar,n in zip(bars,ct_counts.values):
    ax.text(bar.get_width()+ct_counts.max()*0.005,bar.get_y()+bar.get_height()/2,
            f"{n:,}",va="center",fontsize=8)
ax.set_yticks(y_pos); ax.set_yticklabels(short_ct,fontsize=8,linespacing=1.2)
ax.set_xlabel("Count")
ax.set_title("06b: Top 15 Crash Type Descriptions\nSource: bicycle_typing_20.csv | S4_CRASH_TYPE_DESCRIPTION",
             fontsize=11,fontweight="bold",pad=10)
ax.spines[["top","right"]].set_visible(False)
plt.subplots_adjust(left=0.38,right=0.95,top=0.88,bottom=0.08)
save(fig,"06_crash_typing","06b_crash_type_descriptions")

dir_counts = btype2["S4_BICYCLIST_DIRECTION"].value_counts()
dir_labels = {1:"With Traffic",2:"Against Traffic",3:"Unknown",4:"N/A",
              5:"Left Turn",6:"Right Turn",7:"Straight",8:"Backing",9:"Unknown Dir"}
dir_counts.index=[dir_labels.get(int(k),str(k)) if str(k).replace(".","").isdigit() else str(k)
                  for k in dir_counts.index]
fig,ax=plt.subplots(figsize=(8,5))
ax.bar(dir_counts.index,dir_counts.values,color="#42A5F5",edgecolor="white")
ax.set_title("06c: Bicyclist Direction\nSource: bicycle_typing_20.csv | S4_BICYCLIST_DIRECTION",
             fontsize=12,fontweight="bold")
ax.tick_params(axis="x",rotation=30); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); save(fig,"06_crash_typing","06c_bicyclist_direction")

contrib_data=[]
for col in ["ROAD_CIRCUMSTANCES_1","ENVIRONMENT_CIRCUMSTANCES_1"]:
    if col in micro_df.columns:
        for v,c in micro_df[col].dropna().value_counts().items():
            if str(v).strip() not in ("","None","nan"):
                contrib_data.append({"factor":v,"count":c,"source":col})
contrib_df=pd.DataFrame(contrib_data)
if not contrib_df.empty:
    top12=contrib_df.groupby("factor")["count"].sum().nlargest(12).reset_index()
    short_cf=["\n".join(textwrap.wrap(str(t),35)) for t in top12["factor"]]
    fig,ax=plt.subplots(figsize=(11,8))
    y_pos=np.arange(len(top12))
    colors_cf=plt.cm.Oranges(np.linspace(0.35,0.9,len(top12)))[::-1]
    bars=ax.barh(y_pos,top12["count"].values,color=colors_cf,edgecolor="white",height=0.7)
    for bar,n in zip(bars,top12["count"]):
        ax.text(bar.get_width()+top12["count"].max()*0.005,bar.get_y()+bar.get_height()/2,
                f"{n:,}",va="center",fontsize=8)
    ax.set_yticks(y_pos); ax.set_yticklabels(short_cf,fontsize=8,linespacing=1.2)
    ax.set_xlabel("Count")
    ax.set_title(
        "06d: Top Contributing Factors -- ALL Active Modes (Bicycle + E-Bike + E-Scooter)\n"
        "Source: crash_event.csv | ROAD_CIRCUMSTANCES_1 + ENVIRONMENT_CIRCUMSTANCES_1\n"
        "Shows road/environment CONDITIONS. DIFFERENT from 06a (which shows crash scenario type).",
        fontsize=11,fontweight="bold",pad=10)
    ax.spines[["top","right"]].set_visible(False)
    plt.subplots_adjust(left=0.40,right=0.95,top=0.85,bottom=0.08)
    save(fig,"06_crash_typing","06d_contributing_factors")
    top12.to_csv(os.path.join(TABS,"contributing_factors_by_mode.csv"),index=False)

# ===========================================================================
# 11. SECTION 08 - QWEN CLASSIFICATION ANALYSIS
# ===========================================================================
print("\n=== 08 Qwen Classification ===")
qwen_counts = narr_q["QWEN_MODE"].value_counts()
narr_q["_text"] = narr_q["Narrative"].fillna("").str.lower()

# 08a pie
fig,ax=plt.subplots(figsize=(7,7))
q_labels=qwen_counts.index.tolist(); q_sizes=qwen_counts.values
q_colors=[PALETTE.get(l,"#BDBDBD") for l in q_labels]
wedges,texts,autotexts=ax.pie(q_sizes,labels=None,colors=q_colors,
    autopct=lambda p:f"{p:.1f}%",startangle=140,pctdistance=0.75,
    wedgeprops=dict(linewidth=1.5,edgecolor="white"))
for at in autotexts: at.set_fontsize(11); at.set_color("#111111")
handles=[mpatches.Patch(facecolor=c,label=f"{l}  ({n:,})") for l,c,n in zip(q_labels,q_colors,q_sizes)]
ax.legend(handles=handles,loc="lower center",bbox_to_anchor=(0.5,-0.08),ncol=2,fontsize=10)
ax.set_title(f"Qwen LLM Classification (n={len(narr_q):,})\nSource: multilabel_ebike.xlsx + multilabel_RegBike.xlsx | Classification",
             fontsize=12,fontweight="bold",pad=15)
save(fig,"08_qwen","08a_qwen_class_pie")

# 08b bar
fig,ax=plt.subplots(figsize=(8,4))
bars=ax.barh(q_labels,q_sizes,color=q_colors,edgecolor="white")
for bar,n in zip(bars,q_sizes):
    ax.text(bar.get_width()+q_sizes.max()*0.01,bar.get_y()+bar.get_height()/2,
            f"{n:,}  ({n/len(narr_q)*100:.1f}%)",va="center",fontsize=10)
ax.set_xlabel("Narratives")
ax.set_title(f"Qwen Classification Counts (n={len(narr_q):,})\nSource: multilabel_ebike.xlsx + multilabel_RegBike.xlsx | Classification",
             fontsize=12,fontweight="bold")
ax.spines[["top","right"]].set_visible(False); ax.set_xlim(0,q_sizes.max()*1.3)
plt.tight_layout(); save(fig,"08_qwen","08b_qwen_class_bar")

# 08c: Qwen class vs S4_CRASH_TYPE -- FIXED (top crash types only, readable)
cross2 = narr_q[["HSMV_str","QWEN_MODE"]].copy()
cross2.rename(columns={"HSMV_str":"REPORT_NUMBER_str"},inplace=True)
cross2 = cross2.merge(ce[["REPORT_NUMBER_str","S4_CRASH_TYPE"]], on="REPORT_NUMBER_str", how="left")
# Keep only top 6 most frequent S4_CRASH_TYPE for readability
top_ct_vals = cross2["S4_CRASH_TYPE"].value_counts().head(6).index
cross2_top  = cross2[cross2["S4_CRASH_TYPE"].isin(top_ct_vals)]
ct_pivot    = pd.crosstab(cross2_top["QWEN_MODE"], cross2_top["S4_CRASH_TYPE"])
# Sort columns by total, rows by total
ct_pivot = ct_pivot[ct_pivot.sum().sort_values(ascending=False).index]
ct_pivot = ct_pivot.loc[ct_pivot.sum(axis=1).sort_values(ascending=False).index]

fig,ax=plt.subplots(figsize=(9,5))
sns.heatmap(ct_pivot, annot=True, fmt="d", cmap="Blues",
            linewidths=0.5, linecolor="white", ax=ax,
            annot_kws={"size":11, "color":"#111111"})
ax.set_xlabel("S4_CRASH_TYPE  (crash_event.csv) -- top 6 types shown", fontsize=10)
ax.set_ylabel("Qwen Classification  (multilabel_ebike.xlsx + multilabel_RegBike.xlsx)", fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
ax.set_title(
    "08c: Qwen Classification vs S4_CRASH_TYPE\n"
    f"({len(narr_q):,} narrative records -- shows how Qwen labels align with structural crash type)",
    fontsize=11, fontweight="bold")
plt.tight_layout()
save(fig,"08_qwen","08c_qwen_vs_s4_crashtype_heatmap")

# 08d: severity by Qwen class
nm2_str = nm2.copy(); nm2_str["REPORT_NUMBER_str"]=nm2_str["REPORT_NUMBER"].astype(str)
nm_qwen = nm2_str[nm2_str["REPORT_NUMBER_str"].isin(narr_q["HSMV_str"])].copy()
nm_qwen = nm_qwen.merge(narr_q[["HSMV_str","QWEN_MODE"]].rename(columns={"HSMV_str":"REPORT_NUMBER_str"}),
                        on="REPORT_NUMBER_str",how="left")
nm_qwen["SEV_CLEAN"]=nm_qwen["INJURY_SEVERITY"].map(sev_map).fillna("Unknown")
qmo=[m for m in ["Bicycle","E-Bike","E-Scooter","Other"] if m in nm_qwen["QWEN_MODE"].unique()]
sev_q=nm_qwen[nm_qwen["SEV_CLEAN"].isin(sev_order) & nm_qwen["QWEN_MODE"].notna()]\
      .groupby(["QWEN_MODE","SEV_CLEAN"]).size().unstack(fill_value=0).reindex(qmo,fill_value=0)
sev_q_pct=sev_q.div(sev_q.sum(axis=1),axis=0)*100
fig,ax=plt.subplots(figsize=(10,5))
left=np.zeros(len(qmo))
for sev in sev_order:
    if sev in sev_q_pct.columns:
        vals=[sev_q_pct.loc[m,sev] if m in sev_q_pct.index else 0 for m in qmo]
        ax.barh(qmo,vals,left=left,label=sev,color=SEVERITY_COLORS[sev],edgecolor="white")
        for i,(v,l) in enumerate(zip(vals,left)):
            if v>4: ax.text(l+v/2,i,f"{v:.0f}%",ha="center",va="center",fontsize=9,fontweight="bold")
        left+=np.array(vals)
ax.set_xlim(0,105)
ax.set_title(f"Severity by Qwen Class (n={len(narr_q):,} narrative records)\n"
             "Source: non_motorist.csv x multilabel_ebike.xlsx + multilabel_RegBike.xlsx",
             fontsize=11,fontweight="bold")
ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",fontsize=9)
ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig,"08_qwen","08d_severity_by_qwen_class")

# 08e: age by Qwen class
nm_qwen_age=nm_qwen[nm_qwen["S4_AGE_AT_TIME_OF_CRASH"].between(1,100) & nm_qwen["QWEN_MODE"].notna()]
if len(nm_qwen_age)>10:
    fig,ax=plt.subplots(figsize=(9,5))
    sns.violinplot(data=nm_qwen_age[nm_qwen_age["QWEN_MODE"].isin(qmo)],
                   x="QWEN_MODE",y="S4_AGE_AT_TIME_OF_CRASH",
                   order=qmo,palette=[PALETTE.get(m,"#BDBDBD") for m in qmo],
                   inner="quartile",cut=0,ax=ax)
    ax.set_xlabel("Qwen Classification"); ax.set_ylabel("Age")
    ax.set_title(f"Age by Qwen Class (n={len(narr_q):,} narratives)\n"
                 "Source: non_motorist.csv x multilabel_ebike.xlsx + multilabel_RegBike.xlsx",
                 fontsize=12,fontweight="bold")
    ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
    save(fig,"08_qwen","08e_age_by_qwen_class")

qwen_summary_rows=[]
for qmode in qmo:
    sub=narr_q[narr_q["QWEN_MODE"]==qmode]
    qwen_summary_rows.append({"Qwen_Class":qmode,"Count":len(sub),
        "Pct_of_Narratives":f"{len(sub)/len(narr_q)*100:.1f}%",
        "Source_File":"multilabel_ebike.xlsx + multilabel_RegBike.xlsx","Source_Column":"Classification",
        "Raw_Values_Used":str(narr_q[narr_q["QWEN_MODE"]==qmode]["Classification"].unique().tolist())})
pd.DataFrame(qwen_summary_rows).to_csv(os.path.join(TABS,"qwen_class_summary.csv"),index=False)

# 08f: micromobility speed by mode (violin) -- extracted from narrative text
# Source column: micromobility_speed, e.g. "ebike=15mph, bicycle=10mph" or
# "Not mentioned". A mode/speed pair of "=0mph" is the model's placeholder
# for "no numeric speed was actually stated" -- per spec, treat that pair
# as NOT MENTIONED and drop it (don't plot/count it as a real 0 mph value).
SPEED_COL = "micromobility_speed"
speed_mode_map = {"ebike":"E-Bike","e-bike":"E-Bike",
                   "escooter":"E-Scooter","e-scooter":"E-Scooter",
                   "bicycle":"Bicycle","bicyclist":"Bicycle","other":"Other"}

def parse_micromobility_speed(raw):
    """'ebike=15mph, bicycle=10mph' -> [('E-Bike',15.0), ('Bicycle',10.0)].
    Entries of 0mph are dropped (treated as Not Mentioned)."""
    if pd.isna(raw):
        return []
    pairs = re.findall(r"([A-Za-z\-]+)\s*=\s*([\d]+(?:\.\d+)?)\s*mph", str(raw), flags=re.I)
    out = []
    for tok, val in pairs:
        label = speed_mode_map.get(tok.strip().lower())
        if label is None:
            continue
        speed = float(val)
        if speed <= 0:               # "=0mph" placeholder -> Not Mentioned, skip
            continue
        out.append((label, speed))
    return out

if SPEED_COL not in narr_q.columns:
    print(f"\n  WARNING: '{SPEED_COL}' column not found in narrative files -- skipping 08f speed analysis.")
else:
    print("\n=== 08f Micromobility speed extraction ===")
    speed_records = []
    for _, row in narr_q.iterrows():
        for label, val in parse_micromobility_speed(row.get(SPEED_COL)):
            speed_records.append({"ID": row["HSMV_str"], "MODE": label,
                                   "SPEED_MPH": val, "SOURCE_FILE": row.get("_SOURCE_FILE","")})
    speed_long = pd.DataFrame(speed_records, columns=["ID","MODE","SPEED_MPH","SOURCE_FILE"])

    n_literal_not_mentioned = (narr_q[SPEED_COL].astype(str).str.strip().str.lower()=="not mentioned").sum()
    print(f"  Narratives total                         : {len(narr_q):,}")
    print(f"  Narratives literally 'Not mentioned'     : {n_literal_not_mentioned:,}")
    print(f"  Narratives with >=1 usable (non-zero) speed value : {speed_long['ID'].nunique():,}")
    print(f"  Individual speed values extracted (0mph placeholders dropped) : {len(speed_long):,}")

    speed_plot = speed_long[speed_long["MODE"].isin(ACTIVE_MODES)].copy()
    n_outliers = (speed_plot["SPEED_MPH"]>60).sum()
    if n_outliers:
        print(f"  Dropping {n_outliers} implausible speed values (>60 mph) from the violin plot.")
        speed_plot = speed_plot[speed_plot["SPEED_MPH"]<=60]
    plot_modes = [m for m in ACTIVE_MODES if m in speed_plot["MODE"].unique()]

    if len(speed_plot)>5 and plot_modes:
        fig,ax=plt.subplots(figsize=(9,5))
        sns.violinplot(data=speed_plot,x="MODE",y="SPEED_MPH",order=plot_modes,
                       palette=[PALETTE[m] for m in plot_modes],inner="quartile",cut=0,ax=ax)
        sns.stripplot(data=speed_plot,x="MODE",y="SPEED_MPH",order=plot_modes,
                      color="black",size=3,alpha=0.35,jitter=0.15,ax=ax)
        ax.set_xlabel("Mode (extracted from narrative)"); ax.set_ylabel("Extracted Speed (mph)")
        ax.set_title(f"08f: Micromobility Speed by Mode (n={len(speed_plot):,} extracted values, 0mph excluded)\n"
                     "Source: multilabel_ebike.xlsx + multilabel_RegBike.xlsx | micromobility_speed",
                     fontsize=11,fontweight="bold")
        ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
        save(fig,"08_qwen","08f_micromobility_speed_violin")

        speed_summary = speed_plot.groupby("MODE")["SPEED_MPH"].describe().round(1).reindex(plot_modes)
        speed_summary.to_csv(os.path.join(TABS,"micromobility_speed_summary.csv"))
        print(speed_summary.to_string())
    else:
        print("  Not enough usable speed data to plot 08f_micromobility_speed_violin.")

    speed_long.to_csv(os.path.join(TABS,"micromobility_speed_extracted.csv"),index=False)
    print(f"  Full extracted speed table -> tables/micromobility_speed_extracted.csv")

# ===========================================================================
# 12. SECTION 09 - LAT/LON
# ===========================================================================
print("\n=== 09 Lat/Lon ===")
raw_valid = ce["LATITUDE"].notna() & ce["LONGITUDE"].notna()
s4_valid  = ce["S4_LATITUDE"].notna() & ce["S4_LONGITUDE"].notna()
print(f"  Raw lat/lon non-null : {raw_valid.sum():,} / {len(ce):,} ({raw_valid.mean()*100:.1f}%)")
print(f"  S4 lat/lon non-null  : {s4_valid.sum():,} / {len(ce):,} ({s4_valid.mean()*100:.1f}%)")
print(f"  Use S4_LATITUDE/S4_LONGITUDE for mapping -- far more complete.")

HAS_GEO_STATUS = "S4_GEOLOCATION_CURRENT" in ce.columns
if not HAS_GEO_STATUS:
    possible = [c for c in ce.columns if "GEO" in c.upper() or "STATUS" in c.upper()]
    print(f"  [SKIP] 'S4_GEOLOCATION_CURRENT' not found in crash_event.csv -- skipping 09a/09b and Verified/Preliminary/Unmapped counts.")
    print(f"  Columns that might be the real name: {possible}")

if HAS_GEO_STATUS:
    geo_status=ce["S4_GEOLOCATION_CURRENT"].value_counts()
    fig,ax=plt.subplots(figsize=(7,6))
    g_labels=geo_status.index.tolist(); g_sizes=geo_status.values
    g_colors=["#4CAF50","#FF9800","#F44336","#9E9E9E"][:len(g_labels)]
    ax.pie(g_sizes,labels=None,colors=g_colors,
           autopct=lambda p:f"{p:.1f}%",startangle=140,pctdistance=0.75,
           wedgeprops=dict(linewidth=1.5,edgecolor="white"))
    handles=[mpatches.Patch(facecolor=c,label=f"{l}  ({n:,})") for l,c,n in zip(g_labels,g_colors,g_sizes)]
    ax.legend(handles=handles,loc="lower center",bbox_to_anchor=(0.5,-0.08),ncol=2,fontsize=10)
    ax.set_title("Geolocation Status (All Crashes)\nSource: crash_event.csv | S4_GEOLOCATION_CURRENT\n"
                 "Verified=exact GPS; Preliminary=geocoded from address; Unmapped=no coords",
                 fontsize=11,fontweight="bold",pad=15)
    plt.tight_layout(); save(fig,"09_latlon","09a_geolocation_status_pie")

    geo_mode=micro_df.groupby(["MODE","S4_GEOLOCATION_CURRENT"]).size().unstack(fill_value=0)
    geo_mode_pct=geo_mode.div(geo_mode.sum(axis=1),axis=0)*100
    fig,ax=plt.subplots(figsize=(9,5))
    geo_mode_pct.plot(kind="bar",stacked=True,ax=ax,
                      color=g_colors[:len(geo_mode_pct.columns)],edgecolor="white")
    ax.set_xticklabels(geo_mode_pct.index,rotation=0)
    ax.set_ylabel("%"); ax.set_title("Geolocation Status by Mode\nSource: crash_event.csv | S4_GEOLOCATION_CURRENT",
                                      fontsize=12,fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01,1),loc="upper left"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); save(fig,"09_latlon","09b_geolocation_status_by_mode")

micro_geo=micro_df[micro_df["S4_LATITUDE"].notna() & micro_df["S4_LONGITUDE"].notna() &
                   micro_df["S4_LATITUDE"].between(24,31) & micro_df["S4_LONGITUDE"].between(-88,-79)]
if len(micro_geo)>100:
    fig,ax=plt.subplots(figsize=(11,8))
    for mode in ACTIVE_MODES[::-1]:
        sub=micro_geo[micro_geo["MODE"]==mode]
        ax.scatter(sub["S4_LONGITUDE"],sub["S4_LATITUDE"],c=PALETTE[mode],
                   label=f"{mode} ({len(sub):,})",s=1,alpha=0.25,linewidths=0)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Crash Locations -- Florida (Active Modes)\n"
                 "Source: crash_event.csv | S4_LONGITUDE, S4_LATITUDE",
                 fontsize=12,fontweight="bold")
    ax.legend(title="Mode",fontsize=10,markerscale=6); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); save(fig,"09_latlon","09c_crash_scatter_florida")

geo_rows=[]
for mode in ACTIVE_MODES+["Other"]:
    sub=ce[ce["MODE"]==mode]
    row={"Mode":mode,"Total":len(sub),
        "S4_Lat_Lon_Available":sub["S4_LATITUDE"].notna().sum(),
        "S4_Pct":f"{sub['S4_LATITUDE'].notna().mean()*100:.1f}%",
        "Raw_Pct":f"{sub['LATITUDE'].notna().mean()*100:.1f}%"}
    if HAS_GEO_STATUS:
        row["Verified"]=(sub["S4_GEOLOCATION_CURRENT"]=="Verified").sum()
        row["Preliminary"]=(sub["S4_GEOLOCATION_CURRENT"]=="Preliminary").sum()
        row["Unmapped"]=(sub["S4_GEOLOCATION_CURRENT"]=="Unmapped").sum()
    geo_rows.append(row)
pd.DataFrame(geo_rows).to_csv(os.path.join(TABS,"latlon_completeness.csv"),index=False)

# ===========================================================================
# 12b. SECTION 09d - SPATIOTEMPORAL HOTSPOT CLUSTERING (exploratory)
#      DBSCAN on (lon, lat) per mode to find spatial clusters ("hotspots"),
#      then split into an early-period vs late-period window (median-YEAR
#      split) to flag clusters that are NEW/GROWING in the later window --
#      a simple, defensible proxy for "emerging hotspot" without requiring a
#      full space-time scan-statistic (e.g. SaTScan/Getis-Ord Gi*), which
#      would be the natural next step if this warrants a dedicated pass.
#      PENDING / NEXT STEP: this is a first-pass exploratory clustering, not
#      a validated hotspot-detection pipeline -- see README Section 22.
# ===========================================================================
print("\n=== 09d Spatiotemporal Hotspot Clustering (exploratory) ===")
try:
    from sklearn.cluster import DBSCAN
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
    print("  [SKIP] scikit-learn not available in this environment -- "
          "install with `pip install scikit-learn` to enable 09d/09e.")

if HAVE_SKLEARN and len(micro_geo) > 100:
    # ~500m eps in degrees at Florida's latitude (very rough: 1 deg lat ~111km)
    EPS_DEG = 0.005
    MIN_SAMPLES = 15
    hotspot_rows = []
    for mode in ACTIVE_MODES:
        sub = micro_geo[micro_geo["MODE"] == mode]
        if len(sub) < MIN_SAMPLES * 2:
            print(f"  [SKIP] {mode}: only {len(sub)} geocoded crashes, too few for DBSCAN")
            continue
        coords = sub[["S4_LONGITUDE","S4_LATITUDE"]].values
        labels = DBSCAN(eps=EPS_DEG, min_samples=MIN_SAMPLES).fit_predict(coords)
        sub = sub.assign(CLUSTER=labels)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        pct_noise = (labels == -1).mean() * 100
        print(f"  {mode}: {n_clusters} spatial clusters found, {pct_noise:.0f}% of points unclustered (noise)")

        # early vs late split (median YEAR for this mode)
        med_year = sub["YEAR"].median()
        early = sub[sub["YEAR"] <= med_year]
        late  = sub[sub["YEAR"] >  med_year]
        for cl in sorted(set(labels) - {-1}):
            clus = sub[sub["CLUSTER"] == cl]
            n_early = (clus["YEAR"] <= med_year).sum()
            n_late  = (clus["YEAR"] >  med_year).sum()
            # +1 smoothing on the denominator so an early period of 0 doesn't
            # produce an infinite/undefined ratio for a genuinely brand-new cluster.
            growth_ratio = n_late / (n_early + 1)
            hotspot_rows.append({
                "MODE": mode, "CLUSTER_ID": cl, "N_CRASHES": len(clus),
                "CENTER_LAT": clus["S4_LATITUDE"].mean(), "CENTER_LON": clus["S4_LONGITUDE"].mean(),
                "N_EARLY_PERIOD": n_early, "N_LATE_PERIOD": n_late,
                "SPLIT_YEAR": med_year, "GROWTH_RATIO": round(growth_ratio, 2),
                "EMERGING": (n_late > n_early * 1.5) and (n_late >= 5),
            })
    if hotspot_rows:
        hotspot_df = pd.DataFrame(hotspot_rows).sort_values(
            ["MODE","GROWTH_RATIO"], ascending=[True,False])
        hotspot_df.to_csv(os.path.join(TABS, "spatiotemporal_hotspots_by_mode.csv"), index=False)
        n_emerging = int(hotspot_df["EMERGING"].sum())
        print(f"  {n_emerging} cluster(s) flagged EMERGING (late-period count > 1.5x early-period, n_late>=5)")
        print("  saved: spatiotemporal_hotspots_by_mode.csv, ranked by GROWTH_RATIO (late/(early+1))")
        print("  Top 5 clusters by growth ratio, regardless of whether they cleared the EMERGING bar:")
        top5 = hotspot_df.nlargest(5, "GROWTH_RATIO")[
            ["MODE","N_CRASHES","N_EARLY_PERIOD","N_LATE_PERIOD","GROWTH_RATIO","EMERGING"]]
        print(top5.to_string(index=False))
        if n_emerging == 0:
            print("  NOTE: 0 clusters cleared the strict 1.5x/n_late>=5 EMERGING threshold. That does NOT")
            print("  mean nothing is growing -- it likely means most persistent hotspots (e.g. busy")
            print("  intersections that show up as hotspots every year) have roughly balanced early/late")
            print("  counts, which is a legitimate finding, not a bug. The figure below falls back to")
            print("  showing the top-5 by growth ratio anyway (labeled 'below threshold') so it isn't")
            print("  just unlabeled background dots. Consider lowering the 1.5x multiplier in the script")
            print("  (search EMERGING) if a looser 'growing' definition is more useful for the writeup.")

        emerging = hotspot_df[hotspot_df["EMERGING"]]
        # Fallback: if the strict threshold caught nothing, still highlight the
        # top-5 growth-ratio clusters (clearly labeled as below-threshold) so
        # the figure always shows *something* beyond plain background dots.
        highlight = emerging if len(emerging) else hotspot_df.nlargest(5, "GROWTH_RATIO")
        highlight_label = "emerging hotspot" if len(emerging) else "top growth (below 1.5x threshold)"

        # Chaining check: a fixed-eps DBSCAN can "chain" a long stretch of a
        # dense corridor (e.g. crashes spaced <500m apart continuously along
        # a busy road) into one oversized cluster that isn't really a single
        # tight hotspot. Flag any highlighted cluster that's a large chunk of
        # that mode's whole geocoded total -- it's a sign EPS_DEG may be too
        # big for that area, not necessarily a genuine hotspot.
        for _, r in highlight.iterrows():
            mode_total = len(micro_geo[micro_geo["MODE"] == r["MODE"]])
            share = r["N_CRASHES"] / mode_total * 100 if mode_total else 0
            if share > 10:
                print(f"  [CHECK] {r['MODE']} cluster {int(r['CLUSTER_ID'])} at "
                      f"({r['CENTER_LAT']:.3f}, {r['CENTER_LON']:.3f}) holds {share:.0f}% of all "
                      f"{r['MODE']} geocoded crashes ({r['N_CRASHES']} crashes) -- possible DBSCAN "
                      f"'chaining' along a dense corridor rather than one true tight hotspot. If the "
                      f"map shows this as a single huge circle spanning a wide area, try a smaller "
                      f"EPS_DEG.")

        fig, ax = plt.subplots(figsize=(12,8))
        for mode in ACTIVE_MODES:
            sub = micro_geo[micro_geo["MODE"] == mode]
            if len(sub) < MIN_SAMPLES * 2:
                continue
            ax.scatter(sub["S4_LONGITUDE"], sub["S4_LATITUDE"], c=PALETTE[mode], s=1, alpha=0.15,
                       linewidths=0, label=f"{mode} crashes")
        # Sqrt-and-capped marker sizing: a plain N_CRASHES*3 linear scale lets
        # one very large cluster (see chaining check above) balloon into a
        # circle that visually swallows every other marker and the legend.
        # Capping keeps relative size differences visible without letting one
        # outlier dominate the whole figure.
        max_n = highlight["N_CRASHES"].max() if len(highlight) else 1
        for mode in ACTIVE_MODES:
            hl = highlight[highlight["MODE"]==mode]
            if len(hl):
                sizes = 40 + 260 * np.sqrt(hl["N_CRASHES"] / max_n)
                ax.scatter(hl["CENTER_LON"], hl["CENTER_LAT"], c=PALETTE[mode],
                           s=sizes, edgecolors="black", linewidths=1.2,
                           label=f"{mode} {highlight_label}", zorder=5)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        title_line1 = ("Emerging Hotspots by Mode (DBSCAN clusters, late > 1.5x early period)" if len(emerging)
                        else "Highest-Growth Clusters by Mode (none cleared the 1.5x 'emerging' bar -- see console log)")
        ax.set_title(f"{title_line1}\n"
                     "Small dots = all geocoded crashes (by mode, see legend); circled markers = "
                     f"highlighted cluster centers (size ~ sqrt of cluster crash count, capped)\n"
                     "Exploratory -- see README Section 21 for method caveats",
                     fontsize=11, fontweight="bold")
        # Legend OUTSIDE the axes (right side) so it never sits on top of map
        # data -- unlike a fixed "upper left"/"lower left" corner, this can't
        # collide with real crash clusters wherever they happen to fall.
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  markerscale=3, framealpha=0.9, borderaxespad=0)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        save(fig, "09_latlon", "09d_emerging_hotspots_by_mode")
    else:
        print("  No clusters produced across any mode -- try loosening EPS_DEG/MIN_SAMPLES.")
elif not HAVE_SKLEARN:
    pass
else:
    print(f"  [SKIP] Only {len(micro_geo)} geocoded active-mode crashes available -- too few for clustering.")

# ===========================================================================
# 12c. SECTION 09e - RELATIVE CRASH RISK BY EXPOSURE (crash freq / Strava
#      cycling volume), by county
#      PENDING: requires a Strava Metro (or similar) cycling-volume export
#      by county, which we do not have yet (per the task list: "Ask Xingjing
#      for STRAVA cycling volume data"). This section is written to run
#      automatically the moment that file exists at
#      DATA/strava_county_volume.csv with columns COUNTY_NAME, STRAVA_TRIPS
#      (or STRAVA_MILES) -- until then it prints a clear pending message and
#      exits cleanly rather than failing the whole script.
#      NEXT STEP (once data arrives): repeat at the census-tract level, which
#      the task list also asks for, by swapping COUNTY_NAME for a tract GEOID
#      on both the crash side (needs tract assignment from lat/lon, e.g. a
#      spatial join to Census TIGER tract polygons) and the Strava side.
# ===========================================================================
print("\n=== 09e Relative Crash Risk by Exposure (crash freq / Strava volume) ===")
STRAVA_PATH = os.path.join(DATA, "strava_county_volume.csv")
if os.path.exists(STRAVA_PATH):
    strava = pd.read_csv(STRAVA_PATH)
    vol_col = "STRAVA_TRIPS" if "STRAVA_TRIPS" in strava.columns else \
              ("STRAVA_MILES" if "STRAVA_MILES" in strava.columns else None)
    if vol_col is None:
        print(f"  [SKIP] {STRAVA_PATH} found but has neither STRAVA_TRIPS nor STRAVA_MILES column -- "
              f"columns present: {list(strava.columns)}")
    else:
        crash_by_county = micro_df.groupby(["COUNTY_NAME","MODE"]).size().reset_index(name="crashes")
        risk = crash_by_county.merge(strava[["COUNTY_NAME", vol_col]], on="COUNTY_NAME", how="left")
        risk["RELATIVE_RISK"] = risk["crashes"] / risk[vol_col]
        risk.to_csv(os.path.join(TABS, "relative_crash_risk_by_county.csv"), index=False)
        print(f"  saved: relative_crash_risk_by_county.csv ({vol_col} used as exposure denominator)")
else:
    print(f"  [PENDING] Strava cycling-volume file not found at {STRAVA_PATH}.")
    print("  Relative crash risk (crashes / cycling volume) by county -- and eventually by census")
    print("  tract -- cannot be computed until that file is provided. This is an open data request,")
    print("  not a bug: see task list item 'Ask Xingjing for STRAVA cycling volume data' and README")
    print("  Section 26. Once the file exists, drop it at:")
    print(f"    {STRAVA_PATH}")
    print("  with a COUNTY_NAME column matching crash_event.csv | COUNTY_NAME and either a")
    print("  STRAVA_TRIPS or STRAVA_MILES exposure column, and this section will run automatically.")

# ===========================================================================
# 13. SECTION 07 - TEXT MINING
# ===========================================================================
print("\n=== 07 Text Mining ===")
stop_extra={
    "vehicle","the","and","was","a","of","to","in","on","at","for","with","from","had","that",
    "were","he","she","they","his","her","their","an","by","as","not","is","it","be","this",
    "have","has","said","also","into","then","when","which","after","before","while","would",
    "could","did","been","or","but","no","up","all","out","one","its","us","so","if","are",
    "there","than","more","who","what","about","between","through","both","each","left","right",
    "north","south","east","west","road","street","intersection","driver","person","v1","v2",
    "nv1","d1","report","crash","collision","struck","impact","direction","traveling","proceeded",
    "front","rear","side","end","traveled","speed","lane","travel","making","turn","turned",
    "turning","approaching","stated","witness","officer","scene","florida","state","county","city",
    "police","department","patrol","highway","approximately","mph","time","area","advise","advised",
    "told","per","2","1","3","4","5","n","s","e","w","0","x","y","pm","am","st","rd","ave","blvd",
    "dr","ln","way","pkwy"
}
def top_words(texts,n=25):
    words=[]
    for t in texts: words.extend(re.findall(r"[a-z]{4,}",str(t).lower()))
    return Counter(w for w in words if w not in stop_extra).most_common(n)

qwen_narr_counts=narr_q["QWEN_MODE"].value_counts()
fig,ax=plt.subplots(figsize=(7,7))
nm_labels=qwen_narr_counts.index.tolist(); nm_sizes=qwen_narr_counts.values
nm_colors=[PALETTE.get(l,"#BDBDBD") for l in nm_labels]
ax.pie(nm_sizes,labels=None,colors=nm_colors,
       autopct=lambda p:f"{p:.1f}%" if p>1 else "",startangle=140,pctdistance=0.75,
       wedgeprops=dict(linewidth=1.5,edgecolor="white"))
handles2=[mpatches.Patch(facecolor=c,label=f"{l}  ({n:,})") for l,c,n in zip(nm_labels,nm_colors,nm_sizes)]
ax.legend(handles=handles2,loc="lower center",bbox_to_anchor=(0.5,-0.08),ncol=2,fontsize=10)
ax.set_title(f"Narrative Mode Distribution (Qwen, n={len(narr_q):,})\nSource: multilabel_ebike.xlsx + multilabel_RegBike.xlsx | Classification",
             fontsize=12,fontweight="bold",pad=15)
save(fig,"07_text_mining","07a_narrative_mode_pie")

for qmode,color in [("E-Bike","#4CAF50"),("E-Scooter","#FF9800"),("Bicycle","#2196F3"),("Other","#9E9E9E")]:
    subset=narr_q[narr_q["QWEN_MODE"]==qmode]["_text"]
    if len(subset)<5: continue
    top=top_words(subset)
    if not top: continue
    wl,cl=zip(*top)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh(list(wl)[::-1],list(cl)[::-1],color=color,edgecolor="white",alpha=0.9)
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top 25 Keywords -- '{qmode}' Narratives (n={len(subset):,})\n"
                 "Source: multilabel_ebike.xlsx + multilabel_RegBike.xlsx | Narrative",
                 fontsize=12,fontweight="bold")
    ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
    save(fig,"07_text_mining",f"07b_keywords_{qmode.replace('-','').replace(' ','_').lower()}")

keywords_check={
    "helmet":r"\bhelmet\b","crosswalk":r"\bcrosswalk\b|\bcross walk\b",
    "sidewalk":r"\bsidewalk\b","bike lane":r"\bbike lane\b",
    "red light":r"\bred light\b","yield":r"\byield\b",
    "hit and run":r"\bhit.and.run\b",
    "impaired":r"\bimpaired\b|\bunder the influence\b|\bdui\b|\bdwi\b",
    "no lights":r"\bno lights?\b|\bwithout lights?\b",
    "wrong way":r"\bwrong.?way\b",
}
kw_rows=[]
for qmode in ["Bicycle","E-Bike","E-Scooter","Other"]:
    sub=narr_q[narr_q["QWEN_MODE"]==qmode]["_text"]
    row={"Mode":qmode,"Count":len(sub)}
    for kw,pat in keywords_check.items():
        row[kw]=sub.str.contains(pat,regex=True).sum()
    kw_rows.append(row)
kw_df=pd.DataFrame(kw_rows).set_index("Mode")
kw_df.to_csv(os.path.join(TABS,"narrative_keyword_comparison.csv"))

fig,ax=plt.subplots(figsize=(12,5))
kw_pct=kw_df[list(keywords_check.keys())].div(kw_df["Count"],axis=0)*100
sns.heatmap(kw_pct.T,annot=True,fmt=".1f",cmap="YlOrRd",linewidths=0.5,linecolor="white",
            cbar_kws={"label":"% of narratives"},ax=ax,annot_kws={"size":10})
ax.set_xlabel("Qwen Mode"); ax.set_ylabel("Keyword")
ax.set_title("Keyword Mentions (%) by Qwen Mode\nSource: multilabel_ebike.xlsx + multilabel_RegBike.xlsx | Narrative + Classification",
             fontsize=12,fontweight="bold")
plt.tight_layout(); save(fig,"07_text_mining","07c_keyword_heatmap")

# ===========================================================================
# 14. MODE CLASSIFICATION FOR SECTIONS 15-19 (reused, not recomputed)
#    Reuses detail_df, already computed earlier in this same combined script.
# ===========================================================================
mode_df = detail_df[["REPORT_NUMBER", "MODE", "CRASH_YEAR", "COUNTY_NAME"]].drop_duplicates("REPORT_NUMBER").copy()
mode_df["REPORT_NUMBER"] = clean_id(mode_df["REPORT_NUMBER"])
print(f"  Reusing MODE for {len(mode_df):,} active-mode crashes (Bicycle/E-Bike/E-Scooter) from the pipeline above")
print(mode_df["MODE"].value_counts().to_string())

# ===========================================================================
# 15. DRIVER BEHAVIOR (driver.csv)
#     driver.csv = 1 row per motor-vehicle DRIVER per crash. An active-mode
#     crash has 0+ driver rows -- 0 if it's e.g. a bike hitting a fixed
#     object with no motor vehicle involved.
# ===========================================================================
print("\n=== 15. Driver Behavior (driver.csv) ===")
# Reuse the combined driver table from section 5 (S4_Crash_bicycle + Signal4Data)
# instead of re-reading Signal4Data only -- otherwise every bicycle
# REPORT_NUMBER that only exists in S4_Crash_bicycle would silently drop out
# of this inner join.
drv["REPORT_NUMBER"] = clean_id(drv["REPORT_NUMBER"])
drv_m = drv.merge(mode_df[["REPORT_NUMBER", "MODE"]], on="REPORT_NUMBER", how="inner")

print(f"  driver.csv (combined): {len(drv):,} total driver records")
print(f"  {len(drv_m):,} driver records belong to a vehicle in an active-mode crash")
print(f"  ({drv_m['REPORT_NUMBER'].nunique():,} of {len(mode_df):,} active-mode crashes have >=1 driver row --")
print(f"   the rest had no motor-vehicle driver, e.g. a single-bike fall or bike-vs-fixed-object.)")

flag_cols = ["S4_IS_AGGRESSIVE_DRIVING", "S4_IS_ALCOHOL_RELATED", "S4_IS_DRUG_RELATED",
             "S4_IS_DISTRACTED", "S4_IS_SPEEDING_RELATED", "S4_IS_AGING_DRIVER",
             "S4_IS_TEENAGER_DRIVER", "S4_IS_UNRESTRAINED"]
flag_cols = [c for c in flag_cols if c in drv_m.columns]

rows = []
for mode in ACTIVE_MODES:
    sub = drv_m[drv_m["MODE"] == mode]
    n = len(sub)
    for c in flag_cols:
        rate = is_yes(sub[c]).mean() * 100 if n else np.nan
        rows.append({"MODE": mode, "FLAG": c, "PCT": rate, "N": n})
flag_df = pd.DataFrame(rows)
flag_df.to_csv(os.path.join(TABS, "driver_behavior_flags_by_mode.csv"), index=False)

pivot = flag_df.pivot(index="FLAG", columns="MODE", values="PCT").reindex(columns=ACTIVE_MODES)
fig, ax = plt.subplots(figsize=(9, 6))
pivot.plot(kind="barh", ax=ax, color=[PALETTE[m] for m in ACTIVE_MODES])
ax.set_xlabel("% of drivers involved in that mode's crashes")
ax.set_title("Driver Behavior Flags by Crash Mode -- All Crashes\nSource: driver.csv | S4_IS_* flags")
ax.legend(title="Mode")
save(fig, "10_driver_behavior", "10a_driver_flags_by_mode")

# 10a-tiers -- same table for KSI and Fatal-only crashes (a/b/c pattern).
drv_m["S4_CRASH_SEVERITY"] = drv_m["REPORT_NUMBER"].map(ce.set_index("REPORT_NUMBER_str")["S4_CRASH_SEVERITY"])
for tier_name, mask_fn in [
    ("ksi", lambda s: s.isin(["Fatality","Serious Injury","Incapacitating","Fatal"])),
    ("fatal", lambda s: s.isin(["Fatality","Fatal"])),
]:
    sub_tier = drv_m[mask_fn(drv_m["S4_CRASH_SEVERITY"])]
    if not len(sub_tier):
        continue
    rows_t = []
    for mode in ACTIVE_MODES:
        sub = sub_tier[sub_tier["MODE"] == mode]
        n = len(sub)
        for c in flag_cols:
            rate = is_yes(sub[c]).mean() * 100 if n else np.nan
            rows_t.append({"MODE": mode, "FLAG": c, "PCT": rate, "N": n})
    pd.DataFrame(rows_t).to_csv(os.path.join(TABS, f"driver_behavior_flags_by_mode_{tier_name}.csv"), index=False)
print("  saved: driver_behavior_flags_by_mode.csv (+ _ksi / _fatal tiers)")

if "DRIVER_DISTRACTION_CODE" in drv_m.columns:
    dist = (drv_m.groupby(["MODE", "DRIVER_DISTRACTION_CODE"]).size().reset_index(name="count"))
    dist.to_csv(os.path.join(TABS, "driver_distraction_by_mode.csv"), index=False)
    not_distracted_mask = dist["DRIVER_DISTRACTION_CODE"].astype(str).str.lower().str.contains(
        "not distracted", na=False)
    dist_nz = dist[~not_distracted_mask]
    top_dist = dist_nz.groupby("DRIVER_DISTRACTION_CODE")["count"].sum().nlargest(8).index
    dist_top = dist_nz[dist_nz["DRIVER_DISTRACTION_CODE"].isin(top_dist)]
    if len(dist_top):
        pivot2 = dist_top.pivot_table(index="DRIVER_DISTRACTION_CODE", columns="MODE",
                                       values="count", fill_value=0)
        fig, ax = plt.subplots(figsize=(9, 6))
        pivot2.plot(kind="barh", ax=ax, color=[PALETTE.get(c, "#999999") for c in pivot2.columns])
        ax.set_title("Driver Distraction Type by Crash Mode (excl. 'Not Distracted')\n"
                      "Source: driver.csv | DRIVER_DISTRACTION_CODE")
        ax.set_xlabel("Count")
        save(fig, "10_driver_behavior", "10b_distraction_type_by_mode")

print("  saved: driver_behavior_flags_by_mode.csv, driver_distraction_by_mode.csv")

# ===========================================================================
# 16. CITATIONS / VIOLATIONS (violation.csv)
# ===========================================================================
print("\n=== 16. Citations / Violations (violation.csv) ===")
# Reuse the combined violation table from section 5 -- see note in section 15.
vio["REPORT_NUMBER"] = clean_id(vio["REPORT_NUMBER"])
vio_m = vio.merge(mode_df[["REPORT_NUMBER", "MODE"]], on="REPORT_NUMBER", how="inner")

print(f"  violation.csv (combined): {len(vio):,} total citations")
print(f"  {len(vio_m):,} citations tied to an active-mode crash "
      f"({vio_m['REPORT_NUMBER'].nunique():,} unique crashes had >=1 citation)")

cited_crashes = set(vio_m["REPORT_NUMBER"])
cite_flag = mode_df.assign(CITED=mode_df["REPORT_NUMBER"].isin(cited_crashes))
cite_summary = (cite_flag.groupby("MODE")["CITED"].mean() * 100).reindex(ACTIVE_MODES)
print("\n  % of active-mode crashes with >=1 citation issued:")
print(cite_summary.round(1).to_string())
cite_summary.reset_index(name="pct_cited").to_csv(
    os.path.join(TABS, "citation_rate_by_mode.csv"), index=False)

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(cite_summary.index, cite_summary.values, color=[PALETTE[m] for m in cite_summary.index])
pct_y(ax)
ax.set_title("Citation Rate by Crash Mode\nSource: violation.csv (any charge)")
ax.set_ylabel("% of crashes with >=1 citation")
save(fig, "11_violations", "11a_citation_rate_by_mode")

# 11c -- Citation Rate Over Time, ONE LINE PER MODE (previously pooled across
# all active modes, which hid whether Bicycle/E-Bike/E-Scooter are declining
# at the same rate or not).
cite_flag_yr = cite_flag.merge(ce[["REPORT_NUMBER_str","YEAR"]].rename(columns={"REPORT_NUMBER_str":"REPORT_NUMBER"}),
                                on="REPORT_NUMBER", how="left")
cite_yr_mode = (cite_flag_yr.groupby(["YEAR","MODE"])["CITED"].mean() * 100).unstack(fill_value=np.nan)
cite_yr_mode = cite_yr_mode.reindex(columns=[m for m in ACTIVE_MODES if m in cite_yr_mode.columns])
cite_yr_mode.to_csv(os.path.join(TABS, "citation_rate_over_time_by_mode.csv"))

fig, ax = plt.subplots(figsize=(10, 5))
for mode in cite_yr_mode.columns:
    ax.plot(cite_yr_mode.index, cite_yr_mode[mode].values, marker="o",
            label=mode, color=PALETTE[mode], linewidth=2)
ax.set_xlabel("Year"); ax.set_ylabel("% of crashes with >=1 citation"); pct_y(ax)
ax.set_title("Citation Rate Over Time, by Mode\nSource: violation.csv (any charge) x crash_event.csv | CRASH_YEAR",
             fontsize=12, fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig, "11_violations", "11c_citation_rate_over_time_by_mode")
print("\n  NOTE on the declining citation-rate trend: this script cannot determine WHY the")
print("  rate is declining (e.g. changing officer charging practice, more crashes now being")
print("  self-reported/no-officer-response, a genuine drop in citable violations, or a data-lag")
print("  effect where the most recent year(s) haven't had citations fully entered yet). If the")
print("  drop is concentrated in the most recent 1-2 years, check for a reporting-lag artifact")
print("  before treating it as a real behavioral trend -- see README Section 25.")

# 11d -- Citation rate for KSI and Fatal-only subsets (the a/b/c pattern).
# See README Section 21 for which other figures still need this treatment.
sev_lookup = ce.set_index("REPORT_NUMBER_str")["S4_CRASH_SEVERITY"]
cite_flag["S4_CRASH_SEVERITY"] = cite_flag["REPORT_NUMBER"].map(sev_lookup)
cite_tiers = {}
cite_tiers["all_crashes"] = cite_summary
ksi_mask = cite_flag["S4_CRASH_SEVERITY"].isin(["Fatality","Serious Injury","Incapacitating","Fatal"])
fatal_mask = cite_flag["S4_CRASH_SEVERITY"].isin(["Fatality","Fatal"])
if ksi_mask.any():
    cite_tiers["ksi_only"] = (cite_flag[ksi_mask].groupby("MODE")["CITED"].mean()*100).reindex(ACTIVE_MODES)
if fatal_mask.any():
    cite_tiers["fatal_only"] = (cite_flag[fatal_mask].groupby("MODE")["CITED"].mean()*100).reindex(ACTIVE_MODES)
cite_tier_df = pd.DataFrame(cite_tiers)
cite_tier_df.to_csv(os.path.join(TABS, "citation_rate_by_mode_ABC_tiers.csv"))
print("\n  Citation rate by mode, all/KSI/fatal tiers -> citation_rate_by_mode_ABC_tiers.csv")
print(cite_tier_df.round(1).to_string())

if "CHARGE" in vio_m.columns:
    top_charges = vio_m.groupby("CHARGE").size().nlargest(12).reset_index(name="count")
    top_charges.to_csv(os.path.join(TABS, "top_charges_active_modes.csv"), index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [wrap(c, 40) for c in top_charges["CHARGE"][::-1]]
    ax.barh(labels, top_charges["count"][::-1], color="#546E7A")
    ax.set_title("Top Charges in Active-Mode Crashes\nSource: violation.csv | CHARGE")
    ax.set_xlabel("Count")
    save(fig, "11_violations", "11b_top_charges")

print("  saved: citation_rate_by_mode.csv, top_charges_active_modes.csv")

# ===========================================================================
# 17. PEDESTRIAN CRASH TYPING -- REFERENCE ONLY
#     Deliberately NOT folded into ACTIVE_MODES. Pedestrian crashes remain
#     classified as "Other" in the main script's MODE column. This section
#     just reports pedestrian crash-scenario groups the same way the main
#     script's 06a reports bicycle crash-scenario groups, so the two CAN be
#     compared side by side if needed -- it does not redefine "active modes."
# ===========================================================================
print("\n=== 17. Pedestrian Crash Typing (reference only, NOT an active mode) ===")
ped = pd.read_csv(os.path.join(PED, "pedestrian_typing_20.csv"), low_memory=False,
                   dtype={"REPORT_NUMBER": str})
print(f"  pedestrian_typing_20.csv: {len(ped):,} rows")
print("  NOTE: kept separate from Bicycle/E-Bike/E-Scooter throughout this report.")

grp_counts = (ped["S4_CRASH_GROUP_DESCRIPTION"].value_counts().head(12)
              .rename_axis("S4_CRASH_GROUP_DESCRIPTION").reset_index(name="count"))
grp_counts.to_csv(os.path.join(TABS, "pedestrian_crash_groups.csv"), index=False)

fig, ax = plt.subplots(figsize=(9, 6))
labels = [wrap(g, 40) for g in grp_counts["S4_CRASH_GROUP_DESCRIPTION"][::-1]]
ax.barh(labels, grp_counts["count"][::-1], color="#8E24AA")
ax.set_title("Top Pedestrian Crash Scenario Groups (reference)\n"
              "Source: pedestrian_typing_20.csv | S4_CRASH_GROUP_DESCRIPTION")
ax.set_xlabel("Count")
save(fig, "12_pedestrian_context", "12a_pedestrian_crash_groups")
print("  saved: pedestrian_crash_groups.csv")

# ===========================================================================
# 18. ROADWAY INFRASTRUCTURE (fdot tables)
#     crash_roadway_vehicle_driver.csv is FDOT's per-crash export (~= crash_event
#     but with roadway inventory keys). Joined to:
#       roadway_segment.csv     via LRS_ROADWAY = ROADWAY_ID   (mainline AADT,
#                                median, shoulder, context class)
#       roadway_intersection.csv via NEAREST_INTRSECT_ID = INTERSECTION_ID
#                                (intersection control type, major/minor AADT)
# ===========================================================================
print("\n=== 18. Roadway Infrastructure (fdot tables) ===")
# Same combine treatment as sections 15/16: crash_roadway_vehicle_driver.csv
# is crash-level (REPORT_NUMBER), so an inner join on mode_df would silently
# drop bicycle crashes that only exist in S4_Crash_bicycle's fdot export.
# ASSUMPTION (flagged for confirmation): S4_Crash_bicycle/fdot tables has the
# same 3 file names as Signal4Data/fdot tables. roadway_segment.csv /
# roadway_intersection.csv are roadway-inventory reference tables (not
# crash-level), so they're unioned + deduped on their ID rather than
# REPORT_NUMBER-gated.
crvd_s4 = pd.read_csv(os.path.join(FDOT, "crash_roadway_vehicle_driver.csv"), low_memory=False,
                       dtype={"REPORT_NUMBER": str})
seg_s4   = pd.read_csv(os.path.join(FDOT, "roadway_segment.csv"), low_memory=False,
                        dtype={"ROADWAY_ID": str})
inter_s4 = pd.read_csv(os.path.join(FDOT, "roadway_intersection.csv"), low_memory=False,
                        dtype={"INTERSECTION_ID": str})
crvd_bike = pd.read_csv(os.path.join(FDOT_BIKE, "crash_roadway_vehicle_driver.csv"), low_memory=False,
                         dtype={"REPORT_NUMBER": str})
seg_bike   = pd.read_csv(os.path.join(FDOT_BIKE, "roadway_segment.csv"), low_memory=False,
                          dtype={"ROADWAY_ID": str})
inter_bike = pd.read_csv(os.path.join(FDOT_BIKE, "roadway_intersection.csv"), low_memory=False,
                          dtype={"INTERSECTION_ID": str})

crvd = combine_bike_plus_s4(crvd_bike, crvd_s4, bike_rns)
seg = (pd.concat([seg_bike, seg_s4], ignore_index=True, sort=False)
         .drop_duplicates(subset="ROADWAY_ID", keep="first"))
inter = (pd.concat([inter_bike, inter_s4], ignore_index=True, sort=False)
           .drop_duplicates(subset="INTERSECTION_ID", keep="first"))
print(f"  crash_roadway_vehicle_driver.csv (combined): {len(crvd):,} rows "
      f"({len(crvd_bike):,} bicycle + {len(crvd_s4):,} Signal4Data, deduped)")

crvd["REPORT_NUMBER"] = clean_id(crvd["REPORT_NUMBER"])
crvd["LRS_ROADWAY"] = crvd["LRS_ROADWAY"].astype(str).str.strip()
crvd["NEAREST_INTRSECT_ID"] = clean_id(crvd["NEAREST_INTRSECT_ID"])
seg["ROADWAY_ID"] = seg["ROADWAY_ID"].astype(str).str.strip()
inter["INTERSECTION_ID"] = inter["INTERSECTION_ID"].astype(str).str.strip()

crvd_m = crvd.merge(mode_df[["REPORT_NUMBER", "MODE"]], on="REPORT_NUMBER", how="inner")
print(f"  {len(crvd_m):,} rows matched to an active-mode crash")

crvd_seg = crvd_m.merge(seg, left_on="LRS_ROADWAY", right_on="ROADWAY_ID", how="left")
seg_match_rate = crvd_seg["ROADWAY_ID"].notna().mean() * 100
print(f"  Matched to roadway_segment.csv: {seg_match_rate:.1f}% of active-mode crashes")

crvd_int = crvd_m.merge(inter, left_on="NEAREST_INTRSECT_ID", right_on="INTERSECTION_ID", how="left")
int_match_rate = crvd_int["INTERSECTION_ID"].notna().mean() * 100
print(f"  Matched to roadway_intersection.csv: {int_match_rate:.1f}% of active-mode crashes")

# --- 13a. AADT (traffic volume exposure) by mode ---
crvd_seg["AVG_AADT"] = pd.to_numeric(crvd_seg["AVG_AADT"], errors="coerce")
aadt_valid = crvd_seg[crvd_seg["AVG_AADT"].between(1, 200000)]
fig, ax = plt.subplots(figsize=(7, 6))
sns.violinplot(data=aadt_valid, x="MODE", y="AVG_AADT", order=ACTIVE_MODES,
                palette=PALETTE, ax=ax, cut=0)
ax.set_title("Roadway AADT (Traffic Volume) at Crash Location by Mode\n"
              "Source: roadway_segment.csv | AVG_AADT")
ax.set_ylabel("Average Annual Daily Traffic")
save(fig, "13_roadway_infrastructure", "13a_aadt_by_mode")

# --- 13b. Median type ---
med = (crvd_seg[crvd_seg["MEDIAN_TYPE"].notna()]
       .groupby(["MODE", "MEDIAN_TYPE"]).size().reset_index(name="count"))
med.to_csv(os.path.join(TABS, "median_type_by_mode.csv"), index=False)
if len(med):
    fig, ax = plt.subplots(figsize=(9, 5))
    pt_med = pct_within_group_bar(crvd_seg[crvd_seg["MEDIAN_TYPE"].notna()], "MODE", "MEDIAN_TYPE",
                                   ax, top_n=6, order=ACTIVE_MODES, palette=PALETTE)
    ax.set_title("Median Type at Crash Location -- % Within Mode\nSource: roadway_segment.csv | MEDIAN_TYPE",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "13_roadway_infrastructure", "13b_median_type_by_mode")
    pt_med.to_csv(os.path.join(TABS, "median_type_by_mode_pct.csv"))
else:
    print("  MEDIAN_TYPE: field is empty/unavailable in roadway_segment.csv for all matched crashes -- skipping 13b plot")

# --- 13c. Shoulder width (-999 sentinel treated as missing) ---
crvd_seg["SHOULDER_WIDTH"] = pd.to_numeric(crvd_seg["SHOULDER_WIDTH"], errors="coerce")
shw = crvd_seg[crvd_seg["SHOULDER_WIDTH"].between(0, 20)]
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(data=shw, x="MODE", y="SHOULDER_WIDTH", order=ACTIVE_MODES, palette=PALETTE, ax=ax)
ax.set_title("Shoulder Width at Crash Location by Mode\nSource: roadway_segment.csv | SHOULDER_WIDTH")
ax.set_ylabel("Shoulder Width (ft)")
save(fig, "13_roadway_infrastructure", "13c_shoulder_width_by_mode")

# --- 13d. Lane count ---
crvd_seg["NUM_THRU_LANES"] = pd.to_numeric(crvd_seg["NUM_THRU_LANES"], errors="coerce")
lanes = crvd_seg[crvd_seg["NUM_THRU_LANES"].between(1, 12)]
lane_ct = (lanes.groupby(["MODE", "NUM_THRU_LANES"]).size().reset_index(name="count"))
lane_ct.to_csv(os.path.join(TABS, "lane_count_by_mode.csv"), index=False)
if len(lane_ct):
    lanes_str = lanes.copy()
    lanes_str["NUM_THRU_LANES"] = lanes_str["NUM_THRU_LANES"].astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(9, 5))
    pt_lane = pct_within_group_bar(lanes_str, "MODE", "NUM_THRU_LANES", ax, top_n=10,
                                    order=ACTIVE_MODES, palette=PALETTE, wrap_width=6)
    ax.set_ylabel("Number of Through Lanes")
    ax.set_title("Number of Through Lanes -- % Within Mode\n"
                  "Source: roadway_segment.csv | NUM_THRU_LANES", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "13_roadway_infrastructure", "13d_lane_count_by_mode")
    pt_lane.to_csv(os.path.join(TABS, "lane_count_by_mode_pct.csv"))
else:
    print("  NUM_THRU_LANES: field is empty/unavailable in roadway_segment.csv for all matched crashes -- skipping 13d plot")

# --- 13e. Context class ---
cc = (crvd_seg[crvd_seg["CONTEXT_CLASS"].notna()]
      .groupby(["MODE", "CONTEXT_CLASS"]).size().reset_index(name="count"))
cc.to_csv(os.path.join(TABS, "context_class_by_mode.csv"), index=False)

# --- 13f. Intersection control type ---
ictrl = (crvd_int[crvd_int["INTERSECTION_CONTROL"].notna()]
         .groupby(["MODE", "INTERSECTION_CONTROL"]).size().reset_index(name="count"))
ictrl.to_csv(os.path.join(TABS, "intersection_control_by_mode.csv"), index=False)
if len(ictrl):
    fig, ax = plt.subplots(figsize=(9, 5))
    pt_ictrl = pct_within_group_bar(crvd_int[crvd_int["INTERSECTION_CONTROL"].notna()], "MODE",
                                     "INTERSECTION_CONTROL", ax, top_n=6, order=ACTIVE_MODES, palette=PALETTE)
    ax.set_title("Intersection Control Type -- % Within Mode\n"
                  "Source: roadway_intersection.csv | INTERSECTION_CONTROL", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "13_roadway_infrastructure", "13e_intersection_control_by_mode")
    pt_ictrl.to_csv(os.path.join(TABS, "intersection_control_by_mode_pct.csv"))

print("  saved: median_type_by_mode.csv, lane_count_by_mode.csv, "
      "context_class_by_mode.csv, intersection_control_by_mode.csv")

# ===========================================================================
# 19. FARS FEDERAL CODING -- FATAL CRASHES ONLY
#     fars.csv only contains rows for FARS-eligible (fatal) crashes statewide,
#     so an inner join here naturally restricts to active-mode FATALITIES.
# ===========================================================================
print("\n=== 19. FARS Federal Coding (fatal crashes only) ===")
fars = pd.read_csv(os.path.join(FDOT, "fars.csv"), low_memory=False, dtype={"REPORT_NUMBER": str})
fars["REPORT_NUMBER"] = clean_id(fars["REPORT_NUMBER"])
fars_m = fars.merge(mode_df[["REPORT_NUMBER", "MODE"]], on="REPORT_NUMBER", how="inner")

print(f"  fars.csv: {len(fars):,} rows (fatal crashes only, statewide)")
print(f"  {len(fars_m):,} FARS rows matched to an active-mode crash")

if len(fars_m):
    print(fars_m["MODE"].value_counts().to_string())
    print("\n  FARS_LANDUSE breakdown (active-mode fatalities):")
    print(fars_m["FARS_LANDUSE"].value_counts().to_string())
    fars_m.to_csv(os.path.join(TABS, "fars_active_mode_fatalities.csv"), index=False)
    print("  saved: fars_active_mode_fatalities.csv")
else:
    print("  No active-mode crashes matched fars.csv. This means either there were no")
    print("  fatalities among active-mode crashes in this extract, or a REPORT_NUMBER")
    print("  format mismatch -- cross-check against the Fatal count in 05_severity if this")
    print("  looks surprising.")

# ===========================================================================
# WHY OTHER FILES WERE NOT ANALYZED
# ===========================================================================
print("\n=== Files intentionally left out of this analysis, and why ===")
print("""
  passenger.csv
    -- 1 row per vehicle PASSENGER per crash. In a Bicycle/E-Bike/E-Scooter
       crash the injured party is coded in non_motorist.csv, not passenger.csv;
       passenger.csv only describes people riding IN the motor vehicle that
       struck them. Low relevance to a Complete Streets / active-mode safety
       question -- skipped rather than forced into a chart that wouldn't
       answer anything useful.

  non_vehicle_property_damage.csv
    -- Property-damage-only records (e.g. a car hitting a fence or utility
       pole) unrelated to person injury. Not relevant to bicycle/e-bike/
       e-scooter crash analysis.

  injury_summary.csv
    -- Only 8 rows: a pre-aggregated statewide severity x crash-type summary
       table, not a per-crash file. Nothing to join or break out by mode --
       it's a sanity-check reference, not an analysis target. If useful,
       compare its 20,437-crash / 438-fatal-crash totals against your own
       ce-derived totals as a top-line QA check.

  query_params.csv
    -- The list of REPORT_NUMBERs used to build this data pull (20,770 rows,
       vs. 20,438 in crash_event.csv -- some numbers didn't make the final
       export). Useful only as a completeness check, not an analysis input.

  roadway_ramp.csv
    -- Interchange/ramp inventory (71 rows). Ramps are a small, specific
       subset of roadway geometry that's unlikely to intersect meaningfully
       with bicycle/e-bike/e-scooter crash locations at this sample size --
       skipped for now; flag if freeway-adjacent crashes turn out to be a
       meaningful share of the active-mode set.

  vehicle_driver_passenger.csv
    -- A wide FDOT export that largely duplicates driver.csv + passenger.csv
       + vehicle.csv in a single wide-format table (up to 5 passengers per
       row). Since driver.csv and vehicle.csv already cover the same ground
       in normalized form (Section 10 here, plus the main script's vehicle
       analysis), this file would just be redundant coding of the same
       underlying facts.
""")

# ===========================================================================
# 20. POWER BI EXPORT -- ONE FLAT CSV COMBINING MAIN + EXTENDED DATA
#     Same grain as the original power_bi_export.csv (1 row per active-mode
#     crash: Bicycle / E-Bike / E-Scooter). Now also carries driver-behavior
#     flags, citation status, roadway infrastructure, and FARS fatal coding
#     from Sections 10-14 above, so one Power BI refresh picks up everything.
# ===========================================================================
print("\n=== 17. Power BI Export ===")

pbi = micro_df[[
    "REPORT_NUMBER_str", "MODE", "YEAR", "HOUR", "DOW", "MONTH",
    "COUNTY_NAME", "LIGHT_CONDITION", "WEATHER_CONDITION",
    "S4_CRASH_SEVERITY", "LOC_TYPE", "DAY_NIGHT", "mv_involved",
]].rename(columns={"REPORT_NUMBER_str": "REPORT_NUMBER"}).drop_duplicates("REPORT_NUMBER").copy()
pbi["REPORT_NUMBER"] = clean_id(pbi["REPORT_NUMBER"])

# --- driver behavior: did ANY driver in this crash trip a given S4_IS_* flag? ---
if len(flag_cols):
    drv_flags_wide = (
        drv_m.assign(**{c: is_yes(drv_m[c]) for c in flag_cols})
        .groupby("REPORT_NUMBER")[flag_cols].any()
        .reset_index()
    )
    pbi = pbi.merge(drv_flags_wide, on="REPORT_NUMBER", how="left")
    for c in flag_cols:
        pbi[c] = pbi[c].fillna(False)

# --- citations (Section 11) ---
pbi["CITED"] = pbi["REPORT_NUMBER"].isin(cited_crashes)

# --- roadway infrastructure (Section 13; first matched segment per crash) ---
infra_cols = [c for c in ["AVG_AADT", "MEDIAN_TYPE", "SHOULDER_WIDTH", "NUM_THRU_LANES", "CONTEXT_CLASS"]
              if c in crvd_seg.columns]
if infra_cols:
    infra = (crvd_seg[["REPORT_NUMBER"] + infra_cols]
             .drop_duplicates("REPORT_NUMBER", keep="first"))
    pbi = pbi.merge(infra, on="REPORT_NUMBER", how="left")

if "INTERSECTION_CONTROL" in crvd_int.columns:
    ictrl_infra = (crvd_int[["REPORT_NUMBER", "INTERSECTION_CONTROL"]]
                   .drop_duplicates("REPORT_NUMBER", keep="first"))
    pbi = pbi.merge(ictrl_infra, on="REPORT_NUMBER", how="left")

# --- FARS federal coding (Section 14; fatal active-mode crashes only) ---
if len(fars_m):
    pbi["FARS_LANDUSE"] = pbi["REPORT_NUMBER"].map(
        fars_m.drop_duplicates("REPORT_NUMBER").set_index("REPORT_NUMBER")["FARS_LANDUSE"])

# --- crash type (S4_CRASH_TYPE, structural top-level classification from
#     crash_event.csv -- distinct from MODE, which is the Qwen-enhanced
#     Bicycle/E-Bike/E-Scooter classification used throughout this script) ---
crash_type_map = (
    ce[["REPORT_NUMBER_str", "S4_CRASH_TYPE"]]
    .rename(columns={"REPORT_NUMBER_str": "REPORT_NUMBER", "S4_CRASH_TYPE": "CRASH_TYPE"})
    .drop_duplicates("REPORT_NUMBER")
)
crash_type_map["REPORT_NUMBER"] = clean_id(crash_type_map["REPORT_NUMBER"])
pbi = pbi.merge(crash_type_map, on="REPORT_NUMBER", how="left")

# --- road type + posted speed (vehicle.csv; first matched vehicle per crash) ---
road_map = (
    veh2[veh2["MODE"].isin(ACTIVE_MODES) & veh2["TRAFFICWAY_CODE"].notna()]
    [["REPORT_NUMBER", "TRAFFICWAY_CODE"]]
    .drop_duplicates("REPORT_NUMBER", keep="first")
    .rename(columns={"TRAFFICWAY_CODE": "ROAD_TYPE"})
)
road_map["REPORT_NUMBER"] = clean_id(road_map["REPORT_NUMBER"].astype(str))
pbi = pbi.merge(road_map, on="REPORT_NUMBER", how="left")

speed_map = (
    veh2[veh2["MODE"].isin(ACTIVE_MODES) & veh2["POSTED_SPEED"].between(5, 80)]
    [["REPORT_NUMBER", "POSTED_SPEED"]]
    .drop_duplicates("REPORT_NUMBER", keep="first")
)
speed_map["REPORT_NUMBER"] = clean_id(speed_map["REPORT_NUMBER"].astype(str))
pbi = pbi.merge(speed_map, on="REPORT_NUMBER", how="left")

# --- lat/lon coordinates (crash_event.csv; needed for the dashboard's
#     Florida map tab). S4_LATITUDE/S4_LONGITUDE is ~98.5% complete vs.
#     the raw LATITUDE/LONGITUDE columns (see latlon_completeness.csv) --
#     include both pairs so the dashboard can prefer S4_* and fall back
#     automatically via its own find_col() logic. ---
latlon_cols = [c for c in ["S4_LATITUDE", "S4_LONGITUDE", "LATITUDE", "LONGITUDE"] if c in ce.columns]
if latlon_cols:
    latlon_map = (
        ce[["REPORT_NUMBER_str"] + latlon_cols]
        .rename(columns={"REPORT_NUMBER_str": "REPORT_NUMBER"})
        .drop_duplicates("REPORT_NUMBER")
    )
    latlon_map["REPORT_NUMBER"] = clean_id(latlon_map["REPORT_NUMBER"])
    pbi = pbi.merge(latlon_map, on="REPORT_NUMBER", how="left")
    print(f"  Lat/lon merged into power_bi_export.csv: {latlon_cols}")
else:
    print("  WARNING: no LATITUDE/LONGITUDE columns found in crash_event.csv "
          "-- power_bi_export.csv will have no coordinates, and the dashboard's "
          "Florida map tab will show its 'no lat/lon' fallback message.")

# --- micromobility speed (Section 11/08f; narrative-extracted, 0mph dropped) ---
# One row per crash, so collapse multiple same-mode mentions in one narrative
# (e.g. "ebike=15mph, ebike=10mph") down to their mean. Only fills in for the
# mode that crash was actually classified as -- a "bicycle=10mph, ebike=15mph"
# narrative on a crash whose MODE ended up "Bicycle" gets the 10, not the 15.
if "speed_long" in dir() and len(speed_long):
    speed_per_crash = (
        speed_long.groupby(["ID","MODE"])["SPEED_MPH"].mean().round(1)
        .reset_index().rename(columns={"ID":"REPORT_NUMBER","SPEED_MPH":"MICROMOBILITY_SPEED_MPH"})
    )
    speed_per_crash["REPORT_NUMBER"] = clean_id(speed_per_crash["REPORT_NUMBER"])
    pbi = pbi.merge(speed_per_crash, on=["REPORT_NUMBER","MODE"], how="left")
    print(f"  Micromobility speed merged into power_bi_export.csv: "
          f"{pbi['MICROMOBILITY_SPEED_MPH'].notna().sum():,} / {len(pbi):,} crashes have a value.")
else:
    pbi["MICROMOBILITY_SPEED_MPH"] = np.nan
    print("  WARNING: no narrative speed data available (see Section 11/08f) "
          "-- power_bi_export.csv MICROMOBILITY_SPEED_MPH will be all-NaN.")

# --- crash typing scenario (bicycle_typing_20.csv; S4_CRASH_GROUP_DESCRIPTION) ---
# One row per REPORT_NUMBER, first match kept -- some crashes have more than
# one typing row. Powers the dashboard's interactive crash-typing chart.
if "S4_CRASH_GROUP_DESCRIPTION" in btype2.columns:
    crash_group_map = (
        btype2[["REPORT_NUMBER", "S4_CRASH_GROUP_DESCRIPTION"]]
        .rename(columns={"S4_CRASH_GROUP_DESCRIPTION": "CRASH_GROUP"})
        .dropna(subset=["CRASH_GROUP"])
        .drop_duplicates("REPORT_NUMBER", keep="first")
    )
    crash_group_map["REPORT_NUMBER"] = clean_id(crash_group_map["REPORT_NUMBER"])
    pbi = pbi.merge(crash_group_map, on="REPORT_NUMBER", how="left")
    print(f"  Crash typing merged into power_bi_export.csv: "
          f"{pbi['CRASH_GROUP'].notna().sum():,} / {len(pbi):,} crashes have a value.")
else:
    pbi["CRASH_GROUP"] = np.nan
    print("  WARNING: S4_CRASH_GROUP_DESCRIPTION not found in bicycle_typing_20.csv "
          "-- power_bi_export.csv CRASH_GROUP will be all-NaN, and the dashboard's "
          "Crash Typing chart will show its 'no CRASH_GROUP column' message.")

# --- crash type description (bicycle_typing_20.csv; S4_CRASH_TYPE_DESCRIPTION) ---
# The specific collision mechanics (e.g. right-hook, dooring, overtaking) --
# distinct from CRASH_GROUP's broader failure-to-yield-style scenario
# grouping. Bicycle-typed crashes only, same coverage limit as CRASH_GROUP
# above. Powers the dashboard's interactive "reason for crash" chart (06b).
if "S4_CRASH_TYPE_DESCRIPTION" in btype2.columns:
    crash_type_desc_map = (
        btype2[["REPORT_NUMBER", "S4_CRASH_TYPE_DESCRIPTION"]]
        .rename(columns={"S4_CRASH_TYPE_DESCRIPTION": "CRASH_TYPE_DESC"})
        .dropna(subset=["CRASH_TYPE_DESC"])
        .drop_duplicates("REPORT_NUMBER", keep="first")
    )
    crash_type_desc_map["REPORT_NUMBER"] = clean_id(crash_type_desc_map["REPORT_NUMBER"])
    pbi = pbi.merge(crash_type_desc_map, on="REPORT_NUMBER", how="left")
    print(f"  Crash type description merged into power_bi_export.csv: "
          f"{pbi['CRASH_TYPE_DESC'].notna().sum():,} / {len(pbi):,} crashes have a value.")
else:
    pbi["CRASH_TYPE_DESC"] = np.nan
    print("  WARNING: S4_CRASH_TYPE_DESCRIPTION not found in bicycle_typing_20.csv "
          "-- power_bi_export.csv CRASH_TYPE_DESC will be all-NaN.")

# --- contributing factors (crash_event.csv; ROAD_CIRCUMSTANCES_1 /
#     ENVIRONMENT_CIRCUMSTANCES_1) -- the road/environment CONDITIONS present
#     at the crash, i.e. the direct "why" behind 06d. Kept as two separate
#     columns since a crash can have both a road factor and an environment
#     factor. Covers all active modes (unlike CRASH_GROUP/CRASH_TYPE_DESC,
#     which are bicycle-typing-only). Powers the dashboard's interactive
#     contributing-factors chart.
for _cf_col, _cf_out in [("ROAD_CIRCUMSTANCES_1", "ROAD_CIRCUMSTANCE"),
                          ("ENVIRONMENT_CIRCUMSTANCES_1", "ENVIRONMENT_CIRCUMSTANCE")]:
    if _cf_col in micro_df.columns:
        cf_map = (
            micro_df[["REPORT_NUMBER", _cf_col]]
            .rename(columns={_cf_col: _cf_out})
            .dropna(subset=[_cf_out])
            .drop_duplicates("REPORT_NUMBER", keep="first")
        )
        cf_map["REPORT_NUMBER"] = clean_id(cf_map["REPORT_NUMBER"])
        pbi = pbi.merge(cf_map, on="REPORT_NUMBER", how="left")
        print(f"  {_cf_out} merged into power_bi_export.csv: "
              f"{pbi[_cf_out].notna().sum():,} / {len(pbi):,} crashes have a value.")
    else:
        pbi[_cf_out] = np.nan
        print(f"  WARNING: {_cf_col} not found in crash_event.csv "
              f"-- power_bi_export.csv {_cf_out} will be all-NaN.")

# --- driver distraction type (driver.csv; DRIVER_DISTRACTION_CODE) -- the
#     specific distraction behind the S4_IS_DISTRACTED flag already in the
#     export. One row per REPORT_NUMBER: prefers a non-"Not Distracted" code
#     when a crash has more than one driver record. Powers the dashboard's
#     interactive distraction-type chart (10b).
if "DRIVER_DISTRACTION_CODE" in drv_m.columns:
    dist_src = drv_m[["REPORT_NUMBER", "DRIVER_DISTRACTION_CODE"]].dropna(
        subset=["DRIVER_DISTRACTION_CODE"]).copy()
    dist_src["_is_not_distracted"] = dist_src["DRIVER_DISTRACTION_CODE"].astype(str).str.lower().str.contains(
        "not distracted", na=False)
    dist_src = dist_src.sort_values("_is_not_distracted").drop_duplicates("REPORT_NUMBER", keep="first")
    dist_map = dist_src[["REPORT_NUMBER", "DRIVER_DISTRACTION_CODE"]].rename(
        columns={"DRIVER_DISTRACTION_CODE": "DISTRACTION_TYPE"})
    dist_map["REPORT_NUMBER"] = clean_id(dist_map["REPORT_NUMBER"])
    pbi = pbi.merge(dist_map, on="REPORT_NUMBER", how="left")
    print(f"  Driver distraction type merged into power_bi_export.csv: "
          f"{pbi['DISTRACTION_TYPE'].notna().sum():,} / {len(pbi):,} crashes have a value.")
else:
    pbi["DISTRACTION_TYPE"] = np.nan
    print("  WARNING: DRIVER_DISTRACTION_CODE not found in driver.csv "
          "-- power_bi_export.csv DISTRACTION_TYPE will be all-NaN.")


# --- Qwen narrative classification (distinct from MODE -- MODE is the final
#     Bicycle/E-Bike/E-Scooter/Other label after the S4_Crash_bicycle merge;
#     QWEN_CLASS is only populated for the subset of crashes that had a
#     Signal4Data narrative run through the Qwen classifier) ---
qwen_map = (
    ce[["REPORT_NUMBER_str", "Classification_raw"]]
    .rename(columns={"REPORT_NUMBER_str": "REPORT_NUMBER", "Classification_raw": "QWEN_CLASS"})
    .drop_duplicates("REPORT_NUMBER")
)
qwen_map["REPORT_NUMBER"] = clean_id(qwen_map["REPORT_NUMBER"])
pbi = pbi.merge(qwen_map, on="REPORT_NUMBER", how="left")
pbi["IN_QWEN_NARRATIVES"] = pbi["REPORT_NUMBER"].isin(qwen_lookup.keys())

pbi_path = os.path.join(OUT, "power_bi_export.csv")
pbi.to_csv(pbi_path, index=False)
print(f"  power_bi_export.csv -> {pbi_path}  ({len(pbi):,} rows, {pbi.shape[1]} columns)")
print("  Columns now include: mode, timing, location, severity, driver-behavior")
print("  flags, citation status, roadway infrastructure, FARS fatal coding,")
print("  crash type, road type, posted speed, lat/lon coordinates, crash-typing")
print("  scenario (CRASH_GROUP), crash type description (CRASH_TYPE_DESC),")
print("  road/environment contributing factors (ROAD_CIRCUMSTANCE,")
print("  ENVIRONMENT_CIRCUMSTANCE), driver distraction type (DISTRACTION_TYPE),")
print("  and Qwen narrative class (QWEN_CLASS).")
print("  Power BI: same file path/name as before -- just click Refresh.")

# --- narrative text export -- powers the dashboard's interactive keyword /
#     text-mining tab (dashboard used to only show this as static PNGs).
#     Only the ~20.8k Signal4Data crashes with a Qwen-classified narrative
#     have text; everything else (the S4_Crash_bicycle-only population) has
#     no narrative available, so isn't in this file. ---
narr_export = narr_q[["HSMV_str", "QWEN_MODE", "Classification", "_text"]].rename(columns={
    "HSMV_str": "REPORT_NUMBER", "Classification": "QWEN_CLASS_RAW", "_text": "NARRATIVE_TEXT",
}).copy()
narr_export["REPORT_NUMBER"] = clean_id(narr_export["REPORT_NUMBER"])
narr_export = narr_export.merge(
    ce[["REPORT_NUMBER_str", "MODE"]].rename(columns={"REPORT_NUMBER_str": "REPORT_NUMBER"}),
    on="REPORT_NUMBER", how="left")
narr_path = os.path.join(OUT, "narrative_text_export.csv")
narr_export.to_csv(narr_path, index=False)
print(f"  narrative_text_export.csv -> {narr_path}  ({len(narr_export):,} rows)")

# ===========================================================================
# 20b. PERSON-LEVEL DEMOGRAPHICS EXPORT (age, gender)
#     power_bi_export.csv above is 1 row per CRASH. Age and gender live in
#     non_motorist.csv at 1 row per PERSON (a crash can have 2+ non-motorists),
#     so they need their own file at their own grain rather than being forced
#     into the crash-level file.
# ===========================================================================
print("\n=== 20b. Person-Level Demographics Export ===")

nm_demo = nm_active[["REPORT_NUMBER", "MODE", "SEX", "S4_AGE_AT_TIME_OF_CRASH"]].copy()
nm_demo["REPORT_NUMBER"] = clean_id(nm_demo["REPORT_NUMBER"].astype(str))
nm_demo = nm_demo.rename(columns={"S4_AGE_AT_TIME_OF_CRASH": "AGE"})
nm_demo = nm_demo[nm_demo["AGE"].isna() | nm_demo["AGE"].between(1, 100)]

nm_demo = nm_demo.merge(
    pbi[["REPORT_NUMBER", "YEAR", "COUNTY_NAME", "S4_CRASH_SEVERITY", "DAY_NIGHT", "LOC_TYPE"]],
    on="REPORT_NUMBER", how="left",
)

demo_path = os.path.join(OUT, "power_bi_export_demographics.csv")
nm_demo.to_csv(demo_path, index=False)
print(f"  power_bi_export_demographics.csv -> {demo_path}  ({len(nm_demo):,} person-level rows)")
print("  Join key back to power_bi_export.csv: REPORT_NUMBER.")

# ===========================================================================
# 20c. DASHBOARD METADATA -- pipeline funnel counts for an "About" tab
# ===========================================================================
print("\n=== 20c. Dashboard Metadata ===")

n_query_params = None
try:
    qp = pd.read_csv(os.path.join(DATA, "query_params.csv"), low_memory=False)
    id_col = next((c for c in qp.columns if c.upper() == "REPORT_NUMBER"), None)
    n_query_params = int(qp[id_col].nunique()) if id_col else int(len(qp))
except Exception as e:
    print(f"  [WARN] Could not read query_params.csv for dashboard metadata: {e}")

meta_rows = [
    {"metric": "query_params_report_numbers", "value": n_query_params,
     "note": "REPORT_NUMBERs originally pulled to build the Signal4Data extract "
             "(query_params.csv). Covers Signal4Data only -- E-Bike/E-Scooter/Other "
             "come from this pull; Bicycle comes from the separate, larger "
             "S4_Crash_bicycle population and isn't part of this count."},
    {"metric": "crash_event_total_crashes", "value": int(len(ce)),
     "note": "Unique crashes in the COMBINED crash_event table: S4_Crash_bicycle "
             "(bicycle population) + Signal4Data (E-Bike/E-Scooter/Other, and any "
             "Signal4Data crash whose REPORT_NUMBER isn't already covered by "
             "S4_Crash_bicycle). See eda_analysis_combined.py section 5."},
    {"metric": "bicycle_population_s4_crash_bicycle", "value": int(len(ce_bike)),
     "note": "Unique bicycle crashes in the S4_Crash_bicycle source population, "
             "before the Qwen E-Bike/E-Scooter override is applied."},
    {"metric": "active_mode_crashes", "value": int(len(micro_df)),
     "note": "Crashes classified as Bicycle, E-Bike, or E-Scooter -- this dashboard's scope. "
             "Pedestrian-only, MV-only, and other non-active-mode crashes are excluded."},
    {"metric": "bicycle_crashes", "value": int((ce['MODE'] == 'Bicycle').sum()),
     "note": "Sourced from the S4_Crash_bicycle population (bigger, dedicated bicycle "
             "extract), MINUS any REPORT_NUMBER that Qwen's narrative classifier "
             "reclassified as E-Bike or E-Scooter. No longer uses Signal4Data's "
             "S4_CRASH_TYPE/NON_MOTORIST_DESCRIPTION_CODE structural fallback."},
    {"metric": "ebike_crashes", "value": int((ce['MODE'] == 'E-Bike').sum()),
     "note": "Identified exclusively via Qwen LLM narrative classification -- Florida crash forms "
             "have no structural E-Bike code. Sourced from Signal4Data."},
    {"metric": "escooter_crashes", "value": int((ce['MODE'] == 'E-Scooter').sum()),
     "note": "Identified exclusively via Qwen LLM narrative classification -- Florida crash forms "
             "have no structural E-Scooter code. Sourced from Signal4Data."},
    {"metric": "narrative_labeled_crashes", "value": int(len(narr_q)),
     "note": "Crashes with a Qwen-classified crash narrative "
             "(multilabel_ebike.xlsx + multilabel_RegBike.xlsx), Signal4Data only."},
    {"metric": "crashes_with_micromobility_speed", "value": int(pbi["MICROMOBILITY_SPEED_MPH"].notna().sum()),
     "note": "Crashes where micromobility_speed had a usable non-zero value for that crash's "
             "MODE (see power_bi_export.csv column MICROMOBILITY_SPEED_MPH; '0mph' entries in the "
             "source narrative are treated as Not Mentioned and excluded, per Section 11/08f)."},
]
meta_df = pd.DataFrame(meta_rows)
meta_path = os.path.join(OUT, "dashboard_meta.csv")
meta_df.to_csv(meta_path, index=False)
print(f"  dashboard_meta.csv -> {meta_path}")


# ===========================================================================
# 21. SUMMARY TABLES
# ===========================================================================
print("\n=== Saving summary tables ===")
mode_rows=[]
for mode in mode_counts.index:
    t=(ce["MODE"]==mode).sum()
    fq=((ce["MODE"]==mode)&in_narratives).sum()
    fs=((ce["MODE"]==mode)&~in_narratives).sum()
    mode_rows.append({"mode":mode,"count":t,"pct_total":round(t/len(ce)*100,2),
        "from_qwen_narratives":fq,"from_structural_data":fs,
        "note": ("E-Bike/E-Scooter = Qwen-only. Structural data cannot identify these."
                 if mode in ["E-Bike","E-Scooter"] else
                 f"Bicycle from Qwen ({fq}) + structural S4_CRASH_TYPE/NON_MOTORIST_DESCRIPTION_CODE.")})
pd.DataFrame(mode_rows).to_csv(os.path.join(TABS,"mode_summary.csv"),index=False)
pd.DataFrame({"REPORT_NUMBER":sorted(overlap)}).to_csv(os.path.join(TABS,"narrative_crash_overlap.csv"),index=False)
micro_df.groupby(["YEAR","MODE"]).size().unstack(fill_value=0).to_csv(os.path.join(TABS,"year_by_mode.csv"))

# ===========================================================================
# 22. UPDATE README
# ===========================================================================
readme_path = os.path.join(OUT,"README.md")

base_readme = f"""# Signal4Data -- EDA Results

Auto-generated by `eda_analysis_v3.py`. This document explains what each figure
folder and table contains. Regenerate by re-running the script; it will not
duplicate sections it has already written.

**Active modes** = Bicycle + E-Bike + E-Scooter. The "Other" category is
excluded from all charts (see Section 9 for why).

---

## 1. Overview (`figures/01_overview/`)

- `01a_mode_distribution` -- crash counts by active mode (Bicycle/E-Bike/E-Scooter).
- `01c_annual_trend` -- crashes per year by mode.
- `01d_mv_involvement` -- share of crashes involving a motor vehicle, by mode.

## 2. Who (`figures/02_who/`)

- `02a_age_violin` -- age distribution by mode (violin plot).
- `02b_age_histograms` -- age histograms per mode.
- `02c_gender_by_mode` -- gender breakdown by mode.
- Table: `gender_by_mode.csv`, `age_summary_by_mode.csv`.

## 3. When (`figures/03_when/`)

- `03a_day_night_by_mode` -- day vs. night crash split by mode.
- `03b_hour_heatmap` -- crash frequency by hour of day.
- `03c_day_of_week` -- crash frequency by day of week.
- `03d_monthly_pattern` -- seasonal/monthly pattern.

## 4. Where (`figures/04_where/`)

- `04a_intersection_segment` -- intersection vs. road-segment crashes.
- `04b_speed_limit_violin` -- posted speed limit distribution by mode.
- `04c_road_type` -- crash counts by road type.
- `04d_top_counties_stacked` -- top counties by mode (stacked bar).
- `04e_light_conditions` / `04f_weather_conditions` -- lighting and weather at time of crash.
- Table: `county_by_mode.csv`.

## 5. Severity (`figures/05_severity/`)

- `05a_severity_stacked_bar` -- injury severity distribution by mode.
- `05b_fatality_incap_rates` -- fatal + incapacitating injury rate by mode.
- `05c_severity_year_trend` -- severity trend over time.
- Table: `severity_rates_by_mode.csv`, `severity_by_mode.csv`.

## 6. Crash Typing (`figures/06_crash_typing/`)

- `06a_crash_group_distribution` -- crash scenario groups (from `bicycle_typing_20.csv`) --
  what happened (e.g. "Motorist Left Turn/Merge").
- `06b_crash_type_descriptions` -- detailed crash-type descriptions.
- `06c_bicyclist_direction` -- bicyclist direction of travel at impact.
- `06d_contributing_factors` -- road/environment conditions that contributed
  (covers ALL active modes, not just bicyclists).
- Table: `contributing_factors_by_mode.csv`.

## 7. Text Mining (`figures/07_text_mining/`)

- `07a_narrative_mode_pie` -- share of narratives by Qwen-classified mode.
- `07b_keywords_*` -- top keywords per mode, mined from crash narratives.
- `07c_keyword_heatmap` -- % of narratives per mode mentioning key safety terms
  (helmet, crosswalk, bike lane, impaired, etc.).
- Table: `narrative_keyword_comparison.csv`.

## 8. Qwen Classification (`figures/08_qwen/`)

- `08a_qwen_class_pie` / `08b_qwen_class_bar` -- distribution of Qwen narrative
  classifications (Bicycle/E-Bike/E-Scooter/Other).
- `08c_qwen_vs_s4_crashtype_heatmap` -- Qwen class vs. Signal4's own S4_CRASH_TYPE.
- `08d_severity_by_qwen_class` / `08e_age_by_qwen_class` -- severity and age by
  Qwen-assigned mode.
- `08f_micromobility_speed_violin` -- distribution of numeric speeds extracted
  from the `micromobility_speed` narrative column (violin + raw points), split
  by the mode that speed was attributed to (e.g. `ebike=15mph`). `=0mph`
  entries are treated as "Not Mentioned" and excluded, not plotted as 0.
- Table: `qwen_class_summary.csv`, `micromobility_speed_summary.csv` (count/
  mean/median/min/max per mode), `micromobility_speed_extracted.csv` (every
  individual speed value pulled out of the narratives, one row per value).

## 9. Lat/Lon (`figures/09_latlon/`)

- `09a_geolocation_status_pie` / `09b_geolocation_status_by_mode` -- geolocation
  status (`S4_GEOLOCATION_CURRENT`) across all crashes and by mode.
- `09c_crash_scatter_florida` -- crash locations scattered on lon/lat, active
  modes only, clipped to Florida's bounding box.
- Table: `latlon_completeness.csv`. S4_LATITUDE/S4_LONGITUDE is far more
  complete than the raw LATITUDE/LONGITUDE columns -- use S4_* for mapping.

## 10. Driver Behavior (`figures/10_driver_behavior/`)

- `10a_driver_flags_by_mode` -- % of drivers involved in each mode's crashes
  tripping each `S4_IS_*` flag (aggressive, alcohol/drug, distracted, speeding,
  aging/teenager driver, unrestrained).
- `10b_distraction_type_by_mode` -- distraction type breakdown, excluding
  "Not Distracted".
- Table: `driver_behavior_flags_by_mode.csv`, `driver_distraction_by_mode.csv`.

## 11. Citations / Violations (`figures/11_violations/`)

- `11a_citation_rate_by_mode` -- % of active-mode crashes with >=1 citation issued.
- `11b_top_charges` -- most common charges in active-mode crashes.
- Table: `citation_rate_by_mode.csv`, `top_charges_active_modes.csv`.

## 12. Pedestrian Context (`figures/12_pedestrian_context/`) -- reference only

- `12a_pedestrian_crash_groups` -- top pedestrian crash-scenario groups, kept
  separate from Bicycle/E-Bike/E-Scooter ("active modes"). Included for
  side-by-side comparison against 06a only, not folded into any active-mode total.
- Table: `pedestrian_crash_groups.csv`.

## 13. Roadway Infrastructure (`figures/13_roadway_infrastructure/`)

- `13a_aadt_by_mode` -- traffic volume (AADT) at the crash location, by mode.
- `13b_median_type_by_mode` -- median type at the crash location, by mode.
- `13c_shoulder_width_by_mode` -- shoulder width at the crash location, by mode.
- `13d_lane_count_by_mode` -- number of through lanes, by mode.
- `13e_intersection_control_by_mode` -- intersection control type, by mode.
- Table: `median_type_by_mode.csv`, `lane_count_by_mode.csv`,
  `context_class_by_mode.csv`, `intersection_control_by_mode.csv`.

## 14. FARS Fatal Coding (`figures/14_fars_fatal/`)

- No chart output currently; federal FARS coding (land use, NHS status, route
  signage) for active-mode **fatal** crashes only.
- Table: `fars_active_mode_fatalities.csv`.

## Power BI Export

- `power_bi_export.csv` (in the `results/` folder, alongside `figures/` and
  `tables/`) -- one row per active-mode crash (Bicycle/E-Bike/E-Scooter)
  combining mode, timing, location, severity, driver-behavior flags, citation
  status, roadway infrastructure, and FARS fatal coding from Sections 1-19
  above into a single flat file. Point Power BI at this file and hit Refresh
  after each re-run to pick up everything, including the driver/violation/
  roadway fields that used to require a second script and a second export.

## Other tables (`tables/`)

- `record_provenance.csv` -- per-narrative record with source-file matches
  (crash_event / non_motorist / vehicle / driver / passenger) and both the
  raw Qwen `Classification` and normalised `QWEN_MODE`.
- `mode_classification_detail.csv` -- row-level mode classification for all
  active-mode records.
- `nm_description_code_reference.csv` -- reference of NON_MOTORIST_DESCRIPTION_CODE values.
- `metric_sources.csv` -- which source file/column backs each metric used in this report.
- `mode_summary.csv` -- final counts by mode, split by Qwen vs. structural source.
- `narrative_crash_overlap.csv` -- REPORT_NUMBERs present in both crash_event and the Qwen narratives.
- `year_by_mode.csv` -- crash counts by year and mode.
"""

discrepancy_section = f"""
---

## 9. Classification Discrepancy & Limitations

### 9.1 The {len(narr_q):,} / ~{len(ce)-len(narr_q):,} Split

This dataset has **two tiers of mode classification**:

| Tier | Records | Source | Modes Identifiable |
|---|---|---|---|
| **Qwen-labeled narratives** | {len(narr_q):,} | `multilabel_ebike.xlsx + multilabel_RegBike.xlsx` (Classification column) | Bicycle, E-Bike, E-Scooter, Other |
| **Structural Signal4 data** | ~{len(ce)-len(narr_q):,} | `crash_event.csv` + `non_motorist.csv` | Bicycle only (or Other) |

We only had access to **{len(narr_q):,} narrative-level labels** from the multilabel classifier.
The remaining ~{len(ce)-len(narr_q):,} crashes lack narrative text classification.
In the structural data, the `NON_MOTORIST_DESCRIPTION_CODE` in `non_motorist.csv` only
distinguishes **"Bicyclist"** vs **"Other Cyclist"**.
It does NOT have an explicit E-Bike or E-Scooter code.

### 9.2 Why "Other Cyclist" is NOT mapped to E-Bike

`NON_MOTORIST_DESCRIPTION_CODE = 'Other Cyclist'` is a catch-all that can include:
unicycles, tricycles, cargo bikes, para-cycles, and other non-standard cycles —
not exclusively E-Bikes. Mapping it to E-Bike would overcount E-Bikes.
Therefore, **E-Bike and E-Scooter counts come exclusively from Qwen narrative labels**.

### 9.3 "Bicycle" in visualisations

When a chart shows "Bicycle", it means:
- `NON_MOTORIST_DESCRIPTION_CODE = 'Bicyclist'`  (non_motorist.csv) **OR**
- `S4_CRASH_TYPE = 'Bicycle'`  (crash_event.csv)

It does **NOT** include "Other Cyclist" records.

### 9.4 Active Modes

"**Active modes**" throughout this report means **Bicycle + E-Bike + E-Scooter**.
This excludes pedestrian-only crashes, motor-vehicle-only crashes, and the "Other" category.
Charts labelled "Active Modes" use `micro_df` which is filtered to these three modes.

### 9.5 Note on Structural vs. Narrative Comparison

An earlier version of this report included a "Qwen-Bicycle vs
Structural-Bicycle" comparison section, which validated narrative-derived
labels against a much larger set of structurally-classified (non-narrative)
records. That comparison has been removed: the CSVs in this dataset were
pre-filtered to only include rows whose `REPORT_NUMBER` matches an `ID` in
`multilabel_ebike.xlsx` / `multilabel_RegBike.xlsx`, so every remaining
record already has a narrative-derived label -- there is no separate
structural-only population left to compare against.

"""

v4_section = f"""
---

## 20. Severity Tiers (a/b/c pattern): All Crashes / KSI / Fatal-Only

Per the analysis brief, most tables should be produced for three tiers:
**(a)** all crashes, **(b)** KSI (Killed + Serious Injury = Fatal +
Incapacitating Injury), **(c)** fatalities only. Helper flags `is_ksi()` /
`is_fatal()` and the pattern are now in the script.

**Currently implemented for:** Citation rate by mode
(`citation_rate_by_mode_ABC_tiers.csv`), Driver behavior flags by mode
(`driver_behavior_flags_by_mode.csv` + `_ksi.csv` + `_fatal.csv`).

**PENDING for:** Day/Night split, Road Type, Light/Weather Conditions,
Intersection Control, Age Distribution, County/spatial trend, Hour-of-day
timing. Extending the pattern to any of these is mechanical -- filter the
input DataFrame with `is_ksi(df["SEV_CLEAN"])` or on `S4_CRASH_SEVERITY`
before calling the existing plotting code a second/third time -- but doing
this for every remaining figure was not completed in this pass given the
number of figures involved; flagging honestly rather than silently skipping.

## 21. Spatiotemporal Hotspot Clustering (`figures/09_latlon/09d_*`)

Exploratory DBSCAN clustering on geocoded crash locations, per mode, split
into an early-period vs late-period window (median crash year for that mode)
to flag clusters whose later-period crash count is >1.5x the earlier period
("emerging hotspots"). Table: `spatiotemporal_hotspots_by_mode.csv`.

**Caveats (read before using in a deliverable):**
- `eps`/`min_samples` (0.005 deg, ~15) were picked as a reasonable starting
  point, not tuned or validated. Different values will produce meaningfully
  different clusters -- a sensitivity check is a natural next step.
- DBSCAN treats points as static in time; the early/late split is a coarse
  proxy for "emerging," not a real space-time statistic. If this analysis
  needs to hold up to scrutiny, the standard tool is a space-time scan
  statistic (e.g. SaTScan, or a Getis-Ord Gi* on a space-time grid) --
  PENDING, flagged as a next step rather than implemented here.
- Only run on the ~98.5%-complete `S4_LATITUDE`/`S4_LONGITUDE` fields;
  crashes with no usable coordinates are excluded (see
  `latlon_completeness.csv`).

## 22. Why Does E-Bike Crash Timing Cluster at 3-4pm?

The hourly chart (`03b_hour_heatmap`, `03b2_hour_pct_within_mode`) now prints
a comparison of each mode's share of crashes at its peak hour. This is a
**diagnostic, not an answer** -- the script can tell you whether the 3-4pm
peak is shared across all three modes (consistent with the general PM
school-dismissal/commute peak that shows up in most transportation-safety
data) or disproportionately concentrated in E-Bike specifically (which would
point toward an E-Bike-specific explanation, e.g. school-age riders on
e-bikes, or delivery/gig-work timing). **PENDING** to actually answer this:
a cross-tab of E-Bike crashes at 3-4pm against rider age band and/or day of
week (school day vs. weekend) would directly test the "school dismissal"
hypothesis -- not yet built, flagged as a follow-up.

## 23. Day vs. Night Definition

`DAY_NIGHT` is a straight pass-through of the pre-populated `S4_DAY_OR_NIGHT`
field in `crash_event.csv` -- this script does not compute it. We do not have
documentation in this data extract of the exact rule FLHSMV/Signal4 used to
set that field (e.g. clock-based cutoff vs. sunrise/sunset vs. civil
twilight, and whether it's calculated per-location or on a statewide
schedule). **PENDING**: confirm the exact definition with FLHSMV/Signal4 if
the write-up needs to state it precisely.

## 24. Data Processing & Limitations (FLHSMV Extraction Pipeline)

1. FLHSMV/Signal4 ran a **keyword search** over statewide crash-narrative
   text, covering **January 1, 2014 through 2026**, to build a candidate
   pool of e-bike/e-scooter crashes, plus a comparison sample of **~5,000
   narratives that did NOT match** those keywords (used to help the
   classifier learn what a non-e-bike/e-scooter narrative looks like).
   **PENDING**: whether the 2026 end of the range is end-of-June 2026 or a
   later cutoff is not recorded in this data extract -- confirm with
   FLHSMV/Xingjing before stating a specific end date in the final report.
2. The candidate narratives were then classified by the **Qwen multilabel
   model** (`multilabel_ebike.xlsx` / `multilabel_RegBike.xlsx`,
   {len(narr_q):,} labeled narratives total) into Bicycle / E-Bike /
   E-Scooter / Other.
3. Two limitations that follow directly from this and should be repeated
   wherever E-Bike/E-Scooter totals are reported:
   - **Keyword extraction is precision-oriented, not a random/representative
     sample.** A crash whose narrative never mentions an e-bike/e-scooter
     term is never in the candidate pool, so absolute E-Bike/E-Scooter
     counts in this report are a **floor**, not a census, of true incidence.
   - **"Bicycle" from this pipeline is a SUBSET of all bicycle crashes in
     Signal4.** Structurally-classified bicycle crashes (via
     `S4_CRASH_TYPE`/`NON_MOTORIST_DESCRIPTION_CODE`) that were never run
     through the keyword/Qwen pipeline are included in `micro_df`/
     `ACTIVE_MODES` for trend/volume totals, but cannot be told apart from
     Qwen-labeled "Bicycle" rows in the mode-comparison charts. **Use
     `ACTIVE_MODES` (all micromobility) for spatial/temporal trend
     totals; use the narrower Qwen-labeled "Bicycle" only when the
     question specifically compares Bicycle vs. E-Bike vs. E-Scooter**,
     per the analysis brief.

## 25. Why Is the Citation Rate Declining Over Time?

`11c_citation_rate_over_time_by_mode` now plots one line per mode instead of
one pooled line, so the trend can be compared across modes. The script
cannot determine WHY the rate is falling -- candidate explanations include
changing officer charging practice, a shift toward more crashes being
self-reported without an officer response, a genuine drop in citable
violations, or a **reporting-lag artifact** where the most recent 1-2 years
haven't had all citations entered into `violation.csv` yet (violations can
be filed after the initial crash report). **PENDING**: check whether the
decline is concentrated in the most recent year(s) specifically (which would
point to a lag artifact) before treating it as a real behavioral trend.

## 26. Relative Crash Risk vs. Exposure (Strava Cycling Volume)

**PENDING -- blocked on external data.** Section 09e in the script is
written to compute crashes ÷ Strava cycling volume by county (table:
`relative_crash_risk_by_county.csv`) the moment a Strava volume export is
available; it currently prints a pending message and exits cleanly. Per the
task list: request the county-level (and ideally census-tract-level) Strava
cycling volume data. Tract-level requires an additional step not yet built:
assigning each crash to a Census tract via a spatial join on
`S4_LATITUDE`/`S4_LONGITUDE` against TIGER/Line tract polygons.

"""
try:
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(base_readme)
        print(f"  README.md did not exist -- created it at {readme_path} with all sections.")

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()
    if "Classification Discrepancy" not in readme_text:
        with open(readme_path, "a", encoding="utf-8") as f:
            f.write(discrepancy_section)
        print("  README.md updated with Section 9 (Classification Discrepancy).")
    else:
        print("  README.md already has discrepancy section.")

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()
    if "Severity Tiers (a/b/c pattern)" not in readme_text:
        with open(readme_path, "a", encoding="utf-8") as f:
            f.write(v4_section)
        print("  README.md updated with Sections 20-26 (v4: severity tiers, hotspot clustering,")
        print("  3-4pm timing, day/night definition, data processing note, citation-decline note,")
        print("  Strava relative-risk PENDING item).")
    else:
        print("  README.md already has the v4 sections.")
except Exception as e:
    print(f"  README update skipped: {e}")


print("\n=== All done! ===")
print(f"Figures -> {FIGS}")
print(f"Tables  -> {TABS}")
print()
print("KEY ANSWERS:")
print("  Classification vs QWEN_MODE in record_provenance.csv:")
print("    Classification = raw Excel value  (E-bike / E-scooter / Bicyclist / Other)")
print("    QWEN_MODE      = normalised label  (E-Bike / E-Scooter / Bicycle / Other)")
print()
print("  Other Cyclist NOT mapped to E-Bike -- could include unicycles, tricycles, etc.")
print("  E-Bike/E-Scooter come ONLY from Qwen narratives.")
print()
print("  'Bicycle' in charts does NOT include 'Other Cyclist' records.")
print()
print("  No double-counting: Qwen records get Qwen label; only non-Qwen records")
print("  go through structural logic.")
print()
print("  'Active modes' = Bicycle + E-Bike + E-Scooter (no Other in charts).")
print()
print("  06a: crash SCENARIO from bicycle_typing_20.csv (what happened?)")
print("  06d: road/environment CONDITIONS from crash_event.csv (what contributed?)")
print("       06d covers ALL active modes, not just bicyclists.")
print()
print("  Lat/lon: S4_LATITUDE/S4_LONGITUDE is 98.5% complete -- use for mapping.")
print("  See 09_latlon/ figures and latlon_completeness.csv.")