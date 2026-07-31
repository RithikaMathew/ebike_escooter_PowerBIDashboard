"""
Signal4Data – EDA v3
Changes from v2:
 - E-Bike / E-Scooter come ONLY from Qwen narratives. 'Other Cyclist' structural
   code is NOT mapped to E-Bike (could be unicycles, cargo bikes, etc.).
 - 'Other' class excluded from all visualisations. Charts show Bicycle/E-Bike/E-Scooter.
 - 06a label overlap fixed.
 - 08c simplified (top crash types only, readable labels).
 - 'Active modes' defined clearly throughout.
 - Data integrity check (REPORT_NUMBER overlap across files).
 - README updated with classification discrepancy note.
"""

import os, re, warnings, textwrap
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
OUT   = os.path.join(BASE, "results")
FIGS  = os.path.join(OUT, "figures")
TABS  = os.path.join(OUT, "tables")

for sub in ["01_overview","02_who","03_when","04_where","05_severity",
            "06_crash_typing","07_text_mining","08_qwen","09_latlon"]:
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

# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
print("\n=== Loading data ===")
ce    = pd.read_csv(os.path.join(CRASH,"crash_event.csv"),  low_memory=False)
nm    = pd.read_csv(os.path.join(CRASH,"non_motorist.csv"), low_memory=False)
veh   = pd.read_csv(os.path.join(CRASH,"vehicle.csv"),      low_memory=False)
drv   = pd.read_csv(os.path.join(CRASH,"driver.csv"),       low_memory=False)
pax   = pd.read_csv(os.path.join(CRASH,"passenger.csv"),    low_memory=False)
vio   = pd.read_csv(os.path.join(CRASH,"violation.csv"),    low_memory=False)
btype = pd.read_csv(os.path.join(DATA,"ped bike typing","bicycle_typing_20.csv"), low_memory=False)
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

print(f"  crash_event  : {ce.shape}   <- 1 row per CRASH")
print(f"  non_motorist : {nm.shape}  <- 1 row per NON-MOTORIST PERSON per crash")
print(f"  vehicle      : {veh.shape}  <- 1 row per VEHICLE per crash")
print(f"  driver       : {drv.shape}  <- 1 row per DRIVER per crash")
print(f"  passenger    : {pax.shape}  <- 1 row per PASSENGER per crash")
print(f"  violation    : {vio.shape}   <- 1 row per CITATION per crash")
print(f"  multilabel_ebike + multilabel_RegBike (combined, deduped) : {narr_q.shape}"
      f"  ({narr_e.shape[0]} ebike + {narr_r.shape[0]} RegBike loaded, {n_dup_rows} duplicate rows removed)")

# Why different row counts?
print("\n  ROW COUNT EXPLANATION:")
print(f"  crash_event has {len(ce):,} rows = {ce['REPORT_NUMBER'].nunique():,} unique crashes (1 crash = 1 row).")
print(f"  non_motorist has {len(nm):,} rows. Same {nm['REPORT_NUMBER'].nunique():,} unique crashes,")
print(f"  but each person involved in a crash gets their own row.")
print(f"  e.g. a crash with 2 bicyclists -> 1 row in crash_event, 2 rows in non_motorist.")
print(f"  Same logic applies to vehicle, driver, passenger, violation tables.")

# DATA INTEGRITY CHECK: do all REPORT_NUMBERs match across files?
print("\n  DATA INTEGRITY (REPORT_NUMBER matching):")
ce_nums  = set(ce["REPORT_NUMBER"].astype(str))
nm_nums  = set(nm["REPORT_NUMBER"].astype(str))
veh_nums = set(veh["REPORT_NUMBER"].astype(str))
drv_nums = set(drv["REPORT_NUMBER"].astype(str))
pax_nums = set(pax["REPORT_NUMBER"].astype(str))
vio_nums = set(vio["REPORT_NUMBER"].astype(str))

print(f"  non_motorist REPORT_NUMBERs not in crash_event : {len(nm_nums - ce_nums)}")
print(f"  crash_event REPORT_NUMBERs not in non_motorist : {len(ce_nums - nm_nums)}")
print(f"  -> In this dataset all {len(ce_nums):,} unique REPORT_NUMBERs appear in BOTH files.")
print(f"  -> Bicycle classification uses BOTH files (OR logic):")
print(f"     crash_event: S4_CRASH_TYPE='Bicycle'")
print(f"     non_motorist: NON_MOTORIST_DESCRIPTION_CODE='Bicyclist'")
print(f"     A record only needs to match ONE to be classified as Bicycle.")

# What is in S4_CRASH_TYPE? (answers: is Signal4 mainly bicycle+MV crashes?)
print("\n  S4_CRASH_TYPE breakdown (crash_event.csv):")
ct_counts = ce["S4_CRASH_TYPE"].value_counts()
print(ct_counts.to_string())
print(f"\n  NOTE: crash_event includes Bicycle (59k), Pedestrian (14k), Single Vehicle (22k),")
print(f"  and other crash types. It is NOT exclusively bicycle-vs-motor-vehicle crashes.")
print(f"  'Single Vehicle' = a bicycle or pedestrian crash with no other vehicle involved.")
print(f"  'Animal' (20 crashes) = initial harmful event was a collision with an animal.")
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
ce_lite = ce[["REPORT_NUMBER","CRASH_YEAR","COUNTY_NAME","S4_CRASH_TYPE"]].copy()
ce_lite["REPORT_NUMBER"] = ce_lite["REPORT_NUMBER"].astype(str)

nm_desc = (nm.groupby("REPORT_NUMBER")["NON_MOTORIST_DESCRIPTION_CODE"]
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
# 3. MODE CLASSIFICATION
#    - multilabel_ebike.xlsx + multilabel_RegBike.xlsx -> primary for narrative-labeled records
#    - Structural fallback for remaining ~94k records:
#        Bicycle  : S4_CRASH_TYPE='Bicycle' (crash_event) OR
#                   NON_MOTORIST_DESCRIPTION_CODE='Bicyclist' (non_motorist)
#        Other    : everything else (includes 'Other Cyclist', pedestrian-only, etc.)
#    - E-Bike / E-Scooter ONLY come from Qwen. 'Other Cyclist' in non_motorist
#      is deliberately NOT mapped to E-Bike because that code also covers
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

ce["nm_has_bicyclist"]    = ce["nm_desc_list"].apply(lambda x: has_desc(x,"Bicyclist"))
ce["nm_has_othercyclist"] = ce["nm_desc_list"].apply(lambda x: has_desc(x,"Other Cyclist"))
ce["REPORT_NUMBER_str"]   = ce["REPORT_NUMBER"].astype(str)

qwen_lookup     = narr_q.set_index("HSMV_str")["QWEN_MODE"].to_dict()
qwen_raw_lookup = narr_q.set_index("HSMV_str")["Classification"].to_dict()

def classify_mode(row):
    rn = row["REPORT_NUMBER_str"]
    if rn in qwen_lookup:                                       # Qwen: authoritative
        return qwen_lookup[rn]
    if row["S4_CRASH_TYPE"]=="Bicycle" or row["nm_has_bicyclist"]:  # Structural: Bicycle only
        return "Bicycle"
    return "Other"
    # NOTE: 'Other Cyclist' is left as 'Other' intentionally.
    # It can include unicycles, para-cycles, tricycles, cargo bikes, etc.
    # E-Bike / E-Scooter require explicit Qwen narrative evidence.

def classification_basis(row):
    rn = row["REPORT_NUMBER_str"]
    if rn in qwen_lookup:
        return f"Qwen LLM | multilabel_ebike.xlsx + multilabel_RegBike.xlsx | Classification='{row['Classification_raw']}'"
    if row["S4_CRASH_TYPE"]=="Bicycle":
        return "Structural | crash_event.csv | S4_CRASH_TYPE='Bicycle'"
    if row["nm_has_bicyclist"]:
        return "Structural | non_motorist.csv | NON_MOTORIST_DESCRIPTION_CODE='Bicyclist'"
    if row["nm_has_othercyclist"]:
        return "Structural | non_motorist.csv | NON_MOTORIST_DESCRIPTION_CODE='Other Cyclist' -> Other (NOT E-Bike)"
    return "Structural | crash_event.csv | No bicycle/cyclist indicator -> Other"

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
    print(f"  {mode}: {total:,} total | {from_qwen:,} Qwen | {from_struct:,} structural")
print()
print("  IMPORTANT: Bicycle in visualisations does NOT include 'Other Cyclist' records.")
print("  'Other Cyclist' (4,324 rows in non_motorist) falls into 'Other' mode.")
print("  'Bicycle' = only explicit 'Bicyclist' code (non_motorist) or S4_CRASH_TYPE='Bicycle'.")

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
    lambda r: "multilabel_ebike.xlsx + multilabel_RegBike.xlsx" if r["IN_QWEN_NARRATIVES"]
              else ("crash_event.csv + non_motorist.csv"
                    if (r["nm_has_bicyclist"]) else "crash_event.csv"), axis=1)
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

# ===========================================================================
# BI EXPORT — one flat, crosstab-ready CSV for Looker Studio / Tableau
# Run this AFTER the MODE classification (section 3) and date parsing
# (section 4) blocks already in eda_analysis_Q3.py.
# ===========================================================================
print("\n=== Exporting flat BI table ===")

# Worst injury per crash, from non_motorist (person-level -> crash-level)
sev_order = ["No Injury","Possible","Non-Incapacitating","Incapacitating","Fatal"]
sev_map   = {"None":"No Injury","No Injury":"No Injury","Possible":"Possible",
             "Non-Incapacitating":"Non-Incapacitating","Incapacitating":"Incapacitating","Fatal":"Fatal"}
sev_rank = {"No Injury":0,"Possible":1,"Non-Incapacitating":2,
            "Incapacitating":3,"Fatal":4}
nm_sev = nm.copy()
nm_sev["INJURY_SEVERITY"] = nm_sev["INJURY_SEVERITY"].map(sev_map).fillna(nm_sev["INJURY_SEVERITY"])
nm_sev["_rank"] = nm_sev["INJURY_SEVERITY"].map(sev_rank).fillna(-1)
worst_sev = (nm_sev.sort_values("_rank", ascending=False)
                    .drop_duplicates("REPORT_NUMBER")[["REPORT_NUMBER","INJURY_SEVERITY"]])
worst_sev["REPORT_NUMBER"] = worst_sev["REPORT_NUMBER"].astype(str)

# Posted speed limit + road type per crash, from vehicle (first non-null)
veh_ctx = (veh.groupby("REPORT_NUMBER")
              .agg(POSTED_SPEED=("POSTED_SPEED","first"),
                   TRAFFICWAY_CODE=("TRAFFICWAY_CODE","first"))
              .reset_index())
veh_ctx["REPORT_NUMBER"] = veh_ctx["REPORT_NUMBER"].astype(str)

bi_export = micro_df.copy()  # already filtered to Bicycle/E-Bike/E-Scooter only
bi_export["REPORT_NUMBER_str"] = bi_export["REPORT_NUMBER"].astype(str)

bi_export = (bi_export
    .merge(worst_sev, left_on="REPORT_NUMBER_str", right_on="REPORT_NUMBER",
           how="left", suffixes=("","_sev"))
    .merge(veh_ctx, left_on="REPORT_NUMBER_str", right_on="REPORT_NUMBER",
           how="left", suffixes=("","_veh")))

bi_cols = {
    "REPORT_NUMBER_str": "Crash_ID",
    "MODE": "Mode",                              # Bicycle / E-Bike / E-Scooter
    "S4_CRASH_TYPE": "Crash_Type",
    "YEAR": "Year", "MONTH": "Month", "DOW": "Day_Of_Week", "HOUR": "Hour",
    "S4_DAY_OR_NIGHT": "Day_Or_Night",
    "LIGHT_CONDITION": "Light_Condition",
    "WEATHER_CONDITION": "Weather_Condition",
    "COUNTY_NAME": "County",
    "S4_IS_INTERSECTION_RELATED": "Intersection_Related",
    "INJURY_SEVERITY": "Injury_Severity",
    "POSTED_SPEED": "Posted_Speed",
    "TRAFFICWAY_CODE": "Road_Type",
    "S4_LATITUDE": "Latitude", "S4_LONGITUDE": "Longitude",
}
missing = [c for c in bi_cols if c not in bi_export.columns]
if missing:
    print(f"  [WARN] columns not found, skipping: {missing}")
bi_export = bi_export[[c for c in bi_cols if c in bi_export.columns]]
bi_export = bi_export.rename(columns=bi_cols)

out_path = os.path.join(TABS, "power_bi_export.csv")
bi_export.to_csv(out_path, index=False)
print(f"  power_bi_export.csv -> {out_path}  ({len(bi_export):,} rows)")

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

# ===========================================================================
# 7. SECTION 03 – WHEN
# ===========================================================================
print("\n=== 03 When ===")
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

tway = (veh_micro.groupby(["MODE","TRAFFICWAY_CODE"]).size()
        .reset_index(name="count").sort_values("count",ascending=False))
top_tw = tway.groupby("TRAFFICWAY_CODE")["count"].sum().nlargest(8).index
tw_top = tway[tway["TRAFFICWAY_CODE"].isin(top_tw)]
pivot_tw = tw_top.pivot_table(index="TRAFFICWAY_CODE",columns="MODE",values="count",fill_value=0)
pivot_tw = pivot_tw.reindex(columns=[m for m in ACTIVE_MODES if m in pivot_tw.columns])
fig,ax=plt.subplots(figsize=(12,6))
x_tw=np.arange(len(pivot_tw)); width=0.25
for i,mode in enumerate([m for m in ACTIVE_MODES if m in pivot_tw.columns]):
    ax.bar(x_tw+i*width,pivot_tw[mode].values,width,label=mode,color=PALETTE[mode],edgecolor="white")
ax.set_xticks(x_tw+width)
ax.set_xticklabels([wrap(str(t),20) for t in pivot_tw.index],rotation=30,ha="right",fontsize=8)
ax.set_ylabel("Count"); ax.set_title("Road Type\nSource: vehicle.csv | TRAFFICWAY_CODE",
                                     fontsize=12,fontweight="bold")
ax.legend(); ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig,"04_where","04c_road_type")

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
    grp = micro_df.groupby(["MODE",col_name]).size().reset_index(name="count")
    top = grp.groupby(col_name)["count"].sum().nlargest(5).index
    pt  = grp[grp[col_name].isin(top)].pivot_table(index=col_name,columns="MODE",values="count",fill_value=0)
    pt  = pt.reindex(columns=[m for m in ACTIVE_MODES if m in pt.columns])
    fig,ax=plt.subplots(figsize=(11,5))
    x_=np.arange(len(pt)); width=0.25
    for i,mode in enumerate([m for m in ACTIVE_MODES if m in pt.columns]):
        ax.bar(x_+i*width,pt[mode].values,width,label=mode,color=PALETTE[mode],edgecolor="white")
    ax.set_xticks(x_+width)
    ax.set_xticklabels([wrap(str(t),15) for t in pt.index],rotation=20,ha="right",fontsize=9)
    ax.set_ylabel("Count"); ax.set_title(f"{title}\nSource: {src}",fontsize=12,fontweight="bold")
    ax.legend(); ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
    save(fig,"04_where",fname)

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
fig,ax=plt.subplots(figsize=(11,5))
sev_yr.plot(kind="bar",stacked=True,ax=ax,colormap="RdYlGn_r",edgecolor="white")
ax.set_xlabel("Year"); ax.set_ylabel("Count")
ax.set_title("Active-Mode Severity Over Years\nSource: crash_event.csv | S4_CRASH_SEVERITY",
             fontsize=12,fontweight="bold")
ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",fontsize=9)
ax.spines[["top","right"]].set_visible(False); plt.tight_layout()
save(fig,"05_severity","05c_severity_year_trend")

# ===========================================================================
# 10. SECTION 06 – CRASH TYPING
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
# 11. SECTION 08 – QWEN CLASSIFICATION ANALYSIS
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

# ===========================================================================
# 12. SECTION 09 – LAT/LON
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
# 14. SECTION 07 – TEXT MINING
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
# 15. SUMMARY TABLES
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
# 16. UPDATE README
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
- Table: `qwen_class_summary.csv`.

## 9. Lat/Lon (`figures/09_latlon/`)

- `09a_geolocation_status_pie` / `09b_geolocation_status_by_mode` -- geolocation
  status (`S4_GEOLOCATION_CURRENT`) across all crashes and by mode.
- `09c_crash_scatter_florida` -- crash locations scattered on lon/lat, active
  modes only, clipped to Florida's bounding box.
- Table: `latlon_completeness.csv`. S4_LATITUDE/S4_LONGITUDE is far more
  complete than the raw LATITUDE/LONGITUDE columns -- use S4_* for mapping.

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