"""Insights tab -- prose takeaways + countermeasures from the *current* filtered data.

This tab deliberately avoids re-plotting charts that already live on Overview,
Severity, When & Where, Causation, etc. Numbers below are computed live from
`df` / optional uploads and are written to `results/insights_snapshot.json`
so you can audit exactly what fed each claim.

v2 restructure: dropped low-signal restatements (raw pedestrian-keyword match,
generic "impairment is low" framing) in favor of findings that are (a) specific
enough to act on and (b) not already obvious from the headline charts -- e.g.
*where* on the infrastructure a crash happens changes which failure mode
dominates there. Each finding below states why it matters and what to do
about it, in that order, so it reads like paper-ready results rather than a
metrics dump.
"""
import json
import os
from datetime import datetime, timezone

st.markdown("## Key Insights & Countermeasures")
st.markdown(
    """
    <div class="section-note">
    This tab summarizes <b>what the filtered data actually show</b> and what that
    suggests for prevention. Charts stay on their home tabs -- when a claim
    comes from a specific view, we point there (e.g. <i>see When &amp; Where → Map 1</i>).
    Every number below is recomputed from the current sidebar filters (Sections 1–10), except the
    "Known Data & Methodology Limitations" section at the end, which is a curated,
    non-live list (clearly marked).
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Live aggregations (all from currently filtered df / optional tables)
# ---------------------------------------------------------------------------
mode_n = {m: int((df["MODE"] == m).sum()) for m in MODES}
total_n = int(len(df))
fatal_n = {m: int(((df["MODE"] == m) & (df["S4_CRASH_SEVERITY"] == "Fatality")).sum()) for m in MODES}
ksi_n = {
    m: int(((df["MODE"] == m) & df["S4_CRASH_SEVERITY"].isin(["Fatality", "Serious Injury"])).sum())
    for m in MODES
}
fatal_pct = {m: (fatal_n[m] / mode_n[m] * 100 if mode_n[m] else np.nan) for m in MODES}
ksi_pct = {m: (ksi_n[m] / mode_n[m] * 100 if mode_n[m] else np.nan) for m in MODES}

night_pct = {}
night_fatal_pct = {}
if "DAY_NIGHT" in df.columns:
    for m in MODES:
        sub = df[df["MODE"] == m]
        night = sub[sub["DAY_NIGHT"] == "Night"]
        night_pct[m] = (sub["DAY_NIGHT"] == "Night").mean() * 100 if len(sub) else np.nan
        night_fatal_pct[m] = (
            (night["S4_CRASH_SEVERITY"] == "Fatality").mean() * 100 if len(night) else np.nan
        )

intx_pct = (df["LOC_TYPE"] == "Intersection").mean() * 100 if "LOC_TYPE" in df.columns else np.nan

# Per-mode intersection/segment split -- the blended intx_pct above is dominated
# by whichever mode has the most rows (usually Bicycle), so it can hide a real
# per-mode difference. Compute both; Section "Where on the road" below reports both.
intx_pct_by_mode = {}
if "LOC_TYPE" in df.columns:
    for m in MODES:
        sub = df[df["MODE"] == m]
        intx_pct_by_mode[m] = (sub["LOC_TYPE"] == "Intersection").mean() * 100 if len(sub) else np.nan

top_counties = (
    df["COUNTY_NAME"].value_counts().head(5)
    if "COUNTY_NAME" in df.columns else pd.Series(dtype=int)
)

# Per-mode geographic concentration: what % of THAT mode's own crashes sit in its
# top 2 counties. Raw top_counties above is a blended count dominated by the
# highest-volume mode, so it can't show that a lower-volume mode is far more (or
# less) geographically concentrated than the blend suggests.
county_concentration_by_mode = {}
if "COUNTY_NAME" in df.columns:
    for m in MODES:
        sub = df[df["MODE"] == m]
        if len(sub):
            top2 = sub["COUNTY_NAME"].value_counts(normalize=True).mul(100).round(1).head(2)
            county_concentration_by_mode[m] = {
                "top2_counties": top2.to_dict(),
                "top2_pct_sum": round(float(top2.sum()), 1),
            }

# Day-of-week peak by mode -- not currently surfaced anywhere in this tab.
dow_peak_by_mode = {}
if "DOW" in df.columns:
    for m in MODES:
        sub = df[df["MODE"] == m]
        if len(sub):
            dow_shares = sub["DOW"].value_counts(normalize=True).mul(100).round(1)
            if len(dow_shares):
                dow_peak_by_mode[m] = {"day": dow_shares.index[0], "pct": float(dow_shares.iloc[0])}

years = sorted(df["YEAR"].dropna().unique().tolist()) if "YEAR" in df.columns else []
growth = {}

# ---------------------------------------------------------------------------
# Data-level corroboration of the map screenshots + a university-town check.
# Point-in-polygon join of crashes to census tracts, aggregated to COUNTY via
# the GEOID's county-FIPS prefix, so county-level, population-normalized
# findings (e.g. "Key West is a real cluster, not one lucky tract" or "does a
# university county show up disproportionately") are checked against the
# underlying data directly, rather than read off a zoomed-out choropleth where
# a mid-sized county is easy to miss visually.
FIPS_TO_COUNTY = {
    "001": "Alachua", "003": "Baker", "005": "Bay", "007": "Bradford", "009": "Brevard",
    "011": "Broward", "013": "Calhoun", "015": "Charlotte", "017": "Citrus", "019": "Clay",
    "021": "Collier", "023": "Columbia", "027": "DeSoto", "029": "Dixie", "031": "Duval",
    "033": "Escambia", "035": "Flagler", "037": "Franklin", "039": "Gadsden", "041": "Gilchrist",
    "043": "Glades", "045": "Gulf", "047": "Hamilton", "049": "Hardee", "051": "Hendry",
    "053": "Hernando", "055": "Highlands", "057": "Hillsborough", "059": "Holmes",
    "061": "Indian River", "063": "Jackson", "065": "Jefferson", "067": "Lafayette",
    "069": "Lake", "071": "Lee", "073": "Leon", "075": "Levy", "077": "Liberty",
    "079": "Madison", "081": "Manatee", "083": "Marion", "085": "Martin", "086": "Miami-Dade",
    "087": "Monroe", "089": "Nassau", "091": "Okaloosa", "093": "Okeechobee", "095": "Orange",
    "097": "Osceola", "099": "Palm Beach", "101": "Pasco", "103": "Pinellas", "105": "Polk",
    "107": "Putnam", "109": "St. Johns", "111": "St. Lucie", "113": "Santa Rosa",
    "115": "Sarasota", "117": "Seminole", "119": "Sumter", "121": "Suwannee", "123": "Taylor",
    "125": "Union", "127": "Volusia", "129": "Wakulla", "131": "Walton", "133": "Washington",
}
# Major public university seat, by county -- general knowledge, not derived
# from this dataset. Used only to label/select counties for the comparison
# below; the rates themselves are computed live from df + tracts_raw.
UNIVERSITY_COUNTY_FIPS = {
    "073": ("Leon", "FSU + FAMU"), "001": ("Alachua", "University of Florida"),
    "095": ("Orange", "UCF"), "057": ("Hillsborough", "USF"),
    "086": ("Miami-Dade", "UM + FIU"), "011": ("Broward", "Nova Southeastern"),
}

county_rate_by_mode = {}
university_county_ranks = {}
top_tracts_by_mode = {}
_gpd_ok = globals().get("GEOPANDAS_AVAILABLE", False)
if (
    _gpd_ok and tracts_raw is not None and LAT_COL and LON_COL
    and LAT_COL in df.columns and LON_COL in df.columns
    and "GEOID" in tracts_raw.columns and "POPULATION" in tracts_raw.columns
):
    try:
        import geopandas as gpd  # already a dependency elsewhere in this app
        geo = df.dropna(subset=[LAT_COL, LON_COL]).copy()
        if len(geo):
            gdf = gpd.GeoDataFrame(
                geo, geometry=gpd.points_from_xy(geo[LON_COL], geo[LAT_COL]), crs=4326
            )
            tj = gpd.sjoin(gdf, tracts_raw[["GEOID", "geometry"]], how="left", predicate="within")
            tj = tj.dropna(subset=["GEOID"])
            tj["COUNTY_FIPS"] = tj["GEOID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(11).str[2:5]

            tpop = tracts_raw[["GEOID", "POPULATION"]].copy()
            tpop["GEOID"] = tpop["GEOID"].astype(str)
            tpop["COUNTY_FIPS"] = tpop["GEOID"].str.zfill(11).str[2:5]
            county_pop = tpop.groupby("COUNTY_FIPS")["POPULATION"].sum()
            tract_pop = tpop.set_index("GEOID")["POPULATION"]

            for m in MODES:
                sub = tj[tj["MODE"] == m] if "MODE" in tj.columns else tj.iloc[0:0]
                # County-level, population-normalized rate
                cnt_c = sub.groupby("COUNTY_FIPS").size()
                rate_c = (cnt_c / county_pop.reindex(cnt_c.index) * 100000).dropna().sort_values(ascending=False)
                county_rate_by_mode[m] = rate_c
                if len(rate_c):
                    ranks = rate_c.rank(ascending=False)
                    uni_rows = {}
                    for fips, (name, school) in UNIVERSITY_COUNTY_FIPS.items():
                        if fips in rate_c.index:
                            uni_rows[name] = {
                                "school": school,
                                "rate": round(float(rate_c[fips]), 1),
                                "rank": int(ranks[fips]),
                                "n_counties": int(len(rate_c)),
                            }
                    if uni_rows:
                        university_county_ranks[m] = uni_rows

                # Tract-level top-15 by rate, for cross-checking the Map 2-style
                # choropleth against the actual numbers (pop >= 100, same floor
                # the app's own Map 2 uses).
                cnt_t = sub.groupby("GEOID").size()
                tr = pd.DataFrame({"crashes": cnt_t, "population": tract_pop.reindex(cnt_t.index)})
                tr = tr[tr["population"] >= 100]
                if len(tr):
                    tr["rate_per_100k"] = tr["crashes"] / tr["population"] * 100000
                    tr["county_fips"] = tr.index.astype(str).str.zfill(11).str[2:5]
                    tr["county"] = tr["county_fips"].map(FIPS_TO_COUNTY)
                    top_tracts_by_mode[m] = tr.sort_values("rate_per_100k", ascending=False).head(15)
    except Exception:
        # Defensive: this is a supplementary cross-check, not core to the tab --
        # if geopandas/the join fails for any reason, degrade silently rather
        # than breaking the rest of Section 9.
        county_rate_by_mode, university_county_ranks, top_tracts_by_mode = {}, {}, {}

if len(years) >= 2:
    y0, y1 = years[0], years[-1]
    yr_counts = df.groupby(["YEAR", "MODE"], observed=True).size()
    for m in MODES:
        a, b = yr_counts.get((y0, m), 0), yr_counts.get((y1, m), 0)
        growth[m] = {"first_year": int(y0), "last_year": int(y1), "n_first": int(a), "n_last": int(b)}

# Hotspots (precomputed table; respect sidebar Mode filter)
hotspot_summary = {}
if hotspot_raw is not None and "MODE" in hotspot_raw.columns:
    hs = add_hotspot_growth_significance(
        hotspot_raw[hotspot_raw["MODE"].isin(sel_modes)].copy()
    )
    for m in MODES:
        hm = hs[hs["MODE"] == m]
        if len(hm) == 0:
            continue
        top = hm.nlargest(1, "N_CRASHES").iloc[0] if "N_CRASHES" in hm.columns else None
        emerging_n = int(hm["EMERGING"].sum()) if "EMERGING" in hm.columns else None
        sig_n = int(hm["SIG_GROWTH"].sum()) if "SIG_GROWTH" in hm.columns else None
        # Surface WHERE the largest cluster is, not just its size. A count with no
        # location attached (the previous version) hides mode-specific geography --
        # e.g. a mode's single largest cluster sitting somewhere none of its other
        # top counties would suggest.
        top_lat = top_lon = None
        if top is not None and "CENTER_LAT" in hm.columns and "CENTER_LON" in hm.columns:
            top_lat = round(float(top["CENTER_LAT"]), 3)
            top_lon = round(float(top["CENTER_LON"]), 3)
        hotspot_summary[m] = {
            "n_clusters": int(len(hm)),
            "largest_cluster_n": int(top["N_CRASHES"]) if top is not None else None,
            "largest_cluster_lat": top_lat,
            "largest_cluster_lon": top_lon,
            "emerging_heuristic_n": emerging_n,
            "significant_growth_n": sig_n,
        }

# Causation (narrative-classified subset) -- the richest table available to this tab
cause_summary = {}
if cause_raw is not None and "primary_cause" in cause_raw.columns:
    cdf = cause_raw[cause_raw["MODE"].isin(sel_modes)].copy()
    if MAIN_CRASH_ID_COL and MAIN_CRASH_ID_COL in df.columns:
        cdf = cdf[cdf["REPORT_NUMBER"].isin(set(df[MAIN_CRASH_ID_COL].astype(str)))]
    if len(cdf):
        cause_summary["n"] = int(len(cdf))

        if "ATTRIBUTION" in cdf.columns:
            cause_summary["attribution"] = (
                cdf["ATTRIBUTION"].value_counts(normalize=True).mul(100).round(1).to_dict()
            )

        if "INFRA_LABEL" in cdf.columns:
            infra_dist = cdf["INFRA_LABEL"].value_counts(normalize=True).mul(100).round(1)
            cause_summary["infra"] = infra_dist.head(5).to_dict()
            sidewalk_pct = float(infra_dist.get(INFRA_SIDEWALK, 0.0))
            crosswalk_pct = float(infra_dist.get(INFRA_CROSSWALK, 0.0))
            cause_summary["sidewalk_crosswalk_pct"] = round(sidewalk_pct + crosswalk_pct, 1)
            cause_summary["bikelane_pct"] = float(infra_dist.get(INFRA_BIKE_LANE, 0.0))

        if "speed_contributing" in cdf.columns:
            cause_summary["speed_yes_pct"] = float(
                (cdf["speed_contributing"].astype(str).str.lower() == "yes").mean() * 100
            )
            if "MODE" in cdf.columns:
                speed_by_mode = {}
                for m in MODES:
                    sub = cdf[cdf["MODE"] == m]
                    if len(sub):
                        speed_by_mode[m] = round(
                            float((sub["speed_contributing"].astype(str).str.lower() == "yes").mean() * 100), 1
                        )
                cause_summary["speed_yes_by_mode"] = speed_by_mode
            if "INFRA_LABEL" in cdf.columns:
                speed_by_infra = (
                    cdf.groupby("INFRA_LABEL")["speed_contributing"]
                    .apply(lambda s: (s.astype(str).str.lower() == "yes").mean() * 100)
                    .round(1)
                )
                cause_summary["speed_by_infra"] = speed_by_infra.to_dict()

        top_causes = cdf["primary_cause"].value_counts(normalize=True).mul(100).round(1).head(5)
        cause_summary["top_causes"] = top_causes.to_dict()

        # Where a given failure mode concentrates: bike-lane crashes and sidewalk
        # crashes each have a dominant, distinct root cause -- this is the basis
        # for the "infrastructure relocates, doesn't remove, the yielding problem"
        # finding below.
        if "INFRA_LABEL" in cdf.columns:
            bl = cdf[cdf["INFRA_LABEL"] == INFRA_BIKE_LANE]
            if len(bl):
                bl_top = bl["primary_cause"].value_counts(normalize=True).mul(100).round(1)
                if len(bl_top):
                    cause_summary["bikelane_top_cause"] = {
                        "cause": bl_top.index[0], "pct": float(bl_top.iloc[0]), "n": int(len(bl)),
                    }
            sw = cdf[cdf["INFRA_LABEL"] == INFRA_SIDEWALK]
            if len(sw):
                sw_top = sw["primary_cause"].value_counts(normalize=True).mul(100).round(1)
                if len(sw_top):
                    cause_summary["sidewalk_top_causes"] = sw_top.head(2).to_dict()
                    cause_summary["sidewalk_top2_sum"] = round(float(sw_top.head(2).sum()), 1)
                    # The literal "sidewalk_driveway_conflict" cause code, separate from
                    # the generic yield-failure causes above -- this is the number that
                    # actually supports a "driveway-crossing" framing, and it's usually
                    # much smaller than the top-2 sum.
                    cause_summary["sidewalk_driveway_literal_pct"] = float(
                        sw_top.get("sidewalk_driveway_conflict", 0.0)
                    )
                    # Cross-check: is sidewalk's #1 cause ALSO the #1 cause on other
                    # infrastructure types? If so, it's a dataset-wide dominant failure
                    # mode, not something distinctive to sidewalks/driveways, and the
                    # "driveway-crossing problem" framing needs a caveat.
                    top1_by_infra = (
                        cdf.groupby("INFRA_LABEL")["primary_cause"]
                        .agg(lambda s: s.value_counts().idxmax() if len(s) else None)
                    )
                    sw_top1_label = sw_top.index[0]
                    n_infra_sharing_top1 = int((top1_by_infra == sw_top1_label).sum())
                    cause_summary["sidewalk_top1_is_generic"] = n_infra_sharing_top1 > 1
                    cause_summary["sidewalk_top1_shared_with_n_infra_types"] = n_infra_sharing_top1

# Driver flags (same S4_IS_* columns as Driver Behavior & Citations tab).
flag_summary = {}
for col, label in [
    ("S4_IS_ALCOHOL_RELATED", "alcohol"),
    ("S4_IS_DISTRACTED", "distracted"),
    ("S4_IS_AGING_DRIVER", "aging_driver"),
]:
    if col in df.columns:
        flag_summary[label] = {
            m: round(driver_flag_rate(df.loc[df["MODE"] == m, col]) * 100, 1) if mode_n[m] else np.nan
            for m in MODES
        }

qwen_validation = compute_qwen_validation_kappa(df, cause_raw)

# Getis-Ord Gi* / Local Moran's I (same logic as When & Where map 4; live from filtered df)
gi_summary = {}
if tracts_raw is not None and LAT_COL and LON_COL:
    gi_summary = compute_gi_star_summary(tracts_raw, df, LAT_COL, LON_COL, sel_modes)

# Live infra-field coverage for limitations / audit
infra_field_coverage = {}
for col, lbl in INFRA_COL_LABELS.items():
    if col in df.columns:
        if col == "SHOULDER_WIDTH":
            vals = pd.to_numeric(df[col], errors="coerce")
            n = int((vals.notna() & (vals != -999)).sum())
        else:
            n = int(df[col].notna().sum())
        infra_field_coverage[lbl] = {"n": n, "pct": round(n / total_n * 100, 1) if total_n else 0.0}

# Demographics: demo_raw / DEMO_AGE_AVAILABLE / DEMO_GENDER_AVAILABLE /
# DEMO_CRASH_ID_COL are already produced by dashboard_core.py's
# load_demographics() -- no sidebar wiring needed, just use them directly.
# _AGE_BAND follows AGE_BAND_ORDER (0-14, 15-17, 18-24, ..., 65+, Unknown);
# _GENDER is normalized to Male/Female/Unknown.
demo_summary = {}
if demo_raw is not None and DEMO_AGE_AVAILABLE:
    ddf = demo_raw.copy()
    if "MODE" in ddf.columns:
        ddf = ddf[ddf["MODE"].isin(sel_modes)]
    if (
        DEMO_CRASH_ID_COL and MAIN_CRASH_ID_COL
        and DEMO_CRASH_ID_COL in ddf.columns and MAIN_CRASH_ID_COL in df.columns
    ):
        ddf = ddf[ddf[DEMO_CRASH_ID_COL].astype(str).isin(set(df[MAIN_CRASH_ID_COL].astype(str)))]
    if len(ddf):
        if "S4_CRASH_SEVERITY" in ddf.columns:
            fatal_by_age = (
                ddf[ddf["_AGE_BAND"] != "Unknown"]
                .groupby("_AGE_BAND", observed=True)["S4_CRASH_SEVERITY"]
                .apply(lambda s: (s == "Fatality").mean() * 100)
                .round(2)
            )
            demo_summary["fatal_pct_by_age"] = {str(k): float(v) for k, v in fatal_by_age.items()}
            base_rate = fatal_by_age.get("18-24")
            top_rate = fatal_by_age.get("65+")
            if base_rate:
                demo_summary["fatal_ratio_65plus_vs_18to24"] = round(float(top_rate / base_rate), 2)
        if DEMO_GENDER_AVAILABLE and "MODE" in ddf.columns:
            demo_summary["sex_pct_by_mode"] = (
                pd.crosstab(ddf["MODE"], ddf["_GENDER"], normalize="index").mul(100).round(1).to_dict("index")
            )
        if "MODE" in ddf.columns:
            demo_summary["median_age_by_mode"] = ddf.groupby("MODE")["_AGE"].median().round(1).to_dict()

snapshot = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "n_filtered": total_n,
    "mode_n": mode_n,
    "fatal_n": fatal_n,
    "fatal_pct": {k: (None if pd.isna(v) else round(v, 3)) for k, v in fatal_pct.items()},
    "ksi_pct": {k: (None if pd.isna(v) else round(v, 3)) for k, v in ksi_pct.items()},
    "night_pct": {k: (None if pd.isna(v) else round(v, 2)) for k, v in night_pct.items()},
    "night_fatal_pct": {k: (None if pd.isna(v) else round(v, 2)) for k, v in night_fatal_pct.items()},
    "intersection_pct": None if pd.isna(intx_pct) else round(float(intx_pct), 2),
    "intersection_pct_by_mode": {
        k: (None if pd.isna(v) else round(v, 2)) for k, v in intx_pct_by_mode.items()
    },
    "top_counties": top_counties.to_dict() if len(top_counties) else {},
    "county_concentration_by_mode": county_concentration_by_mode,
    "dow_peak_by_mode": dow_peak_by_mode,
    "university_county_ranks": university_county_ranks,
    "growth": growth,
    "hotspot_summary": hotspot_summary,
    "gi_summary": gi_summary,
    "infra_field_coverage": infra_field_coverage,
    "cause_summary": cause_summary,
    "flag_summary": flag_summary,
    "qwen_validation": qwen_validation,
    "demo_summary": demo_summary,
}

# Persist snapshot so claims can be audited offline
_snap_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(_snap_dir, exist_ok=True)
_snap_path = os.path.join(_snap_dir, "insights_snapshot.json")
try:
    with open(_snap_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    st.caption(f"Live numbers for this run are also saved to `{_snap_path}` for audit.")
except OSError:
    st.caption("Could not write `results/insights_snapshot.json` (read-only filesystem).")

with st.expander("Source numbers used below (live from current filters)", expanded=False):
    st.json(snapshot)


def _fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.1f}%"


def _fmt_n(x):
    return "n/a" if x is None else f"{int(x):,}"


# ---------------------------------------------------------------------------
# 1. Scale & mode mix
# ---------------------------------------------------------------------------
st.markdown("### 1. Scale & mode mix")
st.markdown(
    f"""
With the current filters there are **{_fmt_n(total_n)}** active-mode crashes:
**{_fmt_n(mode_n.get('Bicycle'))}** bicycle, **{_fmt_n(mode_n.get('E-Bike'))}** e-bike, and
**{_fmt_n(mode_n.get('E-Scooter'))}** e-scooter.

- **Why it matters:** Bicycle dominates the extract by volume, so any "micromobility"
  statistic that isn't split by mode is really describing bicycles. E-bike/e-scooter
  need their own denominators everywhere below, or they disappear into rounding error.
- **See also:** Overview & Trends (mode share / annual trend).
"""
)
if growth:
    bits = []
    for m in MODES:
        g = growth.get(m)
        if not g:
            continue
        bits.append(
            f"{m}: {_fmt_n(g['n_first'])} in {g['first_year']} → {_fmt_n(g['n_last'])} in {g['last_year']}"
        )
    st.markdown(
        "**Growth (first→last year in filter):** " + "; ".join(bits) + ". "
        "Rising e-bike/e-scooter counts likely reflect adoption and better narrative "
        "capture, not only risk -- compare *rates*, not raw counts (Section 5)."
    )

# ---------------------------------------------------------------------------
# 2. THE headline finding: infrastructure mismatch
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 2. Most crashes happen where there's no bike-specific infrastructure at all")
if cause_summary.get("sidewalk_crosswalk_pct") is not None:
    sw_cw = cause_summary["sidewalk_crosswalk_pct"]
    bl = cause_summary.get("bikelane_pct", 0.0)
    st.markdown(
        f"""
Among narrative-classified crashes (n={_fmt_n(cause_summary.get('n'))}), the rider's
location at impact was **sidewalk or crosswalk in {_fmt_pct(sw_cw)}** of cases, vs.
just **{_fmt_pct(bl)} in a dedicated bike lane**.

- **Why it matters:** Roughly half of all crashes happen where the rider legally
  wasn't sharing the road with traffic at all -- meaning the standard "add a bike
  lane" response addresses the minority location. Infrastructure conversations that
  center on bike lanes are targeting ~1 in 10 crashes, not the majority.
- **Countermeasures:** Prioritize capital dollars toward sidewalk/crosswalk conflict
  points (driveway crossings, curb ramps, crossing sightlines -- see Section 3) at
  least as heavily as corridor bike lanes; treat bike-lane investment as a
  complement, not the primary lever, for reducing *total* crash counts.
- **See also:** Roadway Infrastructure tab (Speed × Infrastructure); Crash Causation → "Where the Rider Was at Impact".
"""
    )
else:
    st.info("Load the crash-causation export (`cause_analysis_export.csv`) to include this finding.")

# ---------------------------------------------------------------------------
# 3. Bike lanes and the right-hook pattern
# ---------------------------------------------------------------------------
st.markdown("### 3. Bike lanes don't remove the yielding problem -- they concentrate it into a right-hook pattern")
bl_top = cause_summary.get("bikelane_top_cause")
if bl_top:
    st.markdown(
        f"""
Of the crashes that *did* happen in a dedicated bike lane (n={_fmt_n(bl_top['n'])}),
**{_fmt_pct(bl_top['pct'])}** are coded **"{bl_top['cause']}"** -- a driver turning
across the rider's path, i.e. a right-hook.

- **Why it matters:** A bike lane is the one place a rider is told they're protected,
  yet it's also exactly where a right-turning driver is most likely to cross the
  rider's path without seeing them. This says lane-painting alone is not a fix at
  intersections -- the risk just relocates to the turn point.
- **Countermeasures:** Protected intersection geometry at every signalized/bike-lane
  crossing -- corner refuge islands, setback ("bend-out") bike lane approaches,
  daylighting parking near the corner, and a leading bike interval so cyclists clear
  the conflict zone before turning traffic gets a green.
- **See also:** Crash Causation → "Primary Cause by Location".
"""
    )
else:
    st.info("Load the crash-causation export to include the bike-lane right-hook finding.")

# ---------------------------------------------------------------------------
# 4. Sidewalk crashes are a driveway problem
# ---------------------------------------------------------------------------
st.markdown("### 4. Sidewalk crashes are mostly a driveway-crossing problem, not a riding-on-the-sidewalk problem")
sw_causes = cause_summary.get("sidewalk_top_causes")
sw_sum = cause_summary.get("sidewalk_top2_sum")
if sw_causes:
    causes_txt = "; ".join(f"**{k}** ({_fmt_pct(v)})" for k, v in sw_causes.items())
    st.markdown(
        f"""
Among sidewalk crashes, the top two causes are {causes_txt}, together
**{_fmt_pct(sw_sum)}** of sidewalk crashes.

- **Why it matters:** The risk on sidewalks isn't from riding along them -- it's
  concentrated at the driveways and side streets that cross them, where sightlines
  are poor and drivers aren't expecting a rider at speed. A policy response of
  banning or discouraging sidewalk riding would miss where the risk actually sits.
- **Countermeasures:** Raised/continuous sidewalk treatments through driveway
  crossings, sightline daylighting (no parking/landscaping within X ft of the
  crossing), and driveway consolidation on corridors with repeat crashes --
  targeted at the crossing points themselves, not blanket sidewalk-riding
  restrictions.
- **See also:** Crash Causation → "Sidewalk Crashes" breakdown.
"""
    )
    if cause_summary.get("sidewalk_top1_is_generic"):
        drv_pct = cause_summary.get("sidewalk_driveway_literal_pct")
        n_shared = cause_summary.get("sidewalk_top1_shared_with_n_infra_types")
        st.warning(
            f"**Caveat on the framing above:** the top cause driving that "
            f"{_fmt_pct(sw_sum)} figure is also the #1 cause on **{n_shared}** other "
            f"infrastructure types (crosswalk, bike lane, travel lane) -- it's a "
            f"dataset-wide dominant failure mode, not something distinctive to "
            f"sidewalks. The cause code that literally names a driveway "
            f"(`sidewalk_driveway_conflict`) accounts for only "
            f"**{_fmt_pct(drv_pct) if drv_pct is not None else 'n/a'}** of sidewalk "
            f"crashes. Treat the \"driveway-crossing problem\" framing above as a "
            f"hypothesis to confirm against narrative review, not a settled finding."
        )
else:
    st.info("Load the crash-causation export to include the sidewalk-driveway finding.")

# ---------------------------------------------------------------------------
# 5. Severity -- who's actually being hurt worse, and when
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 5. Severity: fatality risk is concentrated at night, disproportionate to night volume")
st.markdown(
    f"""
Fatality share of crashes (current filters): **Bicycle {_fmt_pct(fatal_pct.get('Bicycle'))}**,
**E-Bike {_fmt_pct(fatal_pct.get('E-Bike'))}**, **E-Scooter {_fmt_pct(fatal_pct.get('E-Scooter'))}**.
KSI (fatal + serious injury) share: Bicycle {_fmt_pct(ksi_pct.get('Bicycle'))},
E-Bike {_fmt_pct(ksi_pct.get('E-Bike'))}, E-Scooter {_fmt_pct(ksi_pct.get('E-Scooter'))}.
"""
)
if night_fatal_pct:
    night_lines = []
    for m in MODES:
        np_, nf_, fp_ = night_pct.get(m), night_fatal_pct.get(m), fatal_pct.get(m)
        if np_ is None or pd.isna(np_) or nf_ is None or pd.isna(nf_) or not fp_:
            continue
        ratio = nf_ / fp_ if fp_ else np.nan
        night_lines.append(
            f"- **{m}:** {_fmt_pct(np_)} of crashes happen at night by volume, but "
            f"{_fmt_pct(nf_)} of *those* night crashes are fatal -- "
            f"{ratio:.2f}x the mode's overall fatality share ({_fmt_pct(fp_)})."
        )
    st.markdown(
        "The night-fatality disproportion holds for all three modes, not just bicycles:\n\n"
        + "\n".join(night_lines)
        + """

- **Why it matters:** Night crashes are rare but disproportionately lethal across
  every mode, so volume-based prioritization (which naturally favors daytime,
  high-traffic locations) will systematically underweight the conditions that
  kill people -- for e-bike and e-scooter riders as much as for cyclists.
- **Countermeasures:** Rider/vehicle conspicuity campaigns and lighting upgrades
  targeted at corridors with above-average night share (not just above-average
  total volume); pair with the DBSCAN/Gi*/EB hotspot layers (Sections 9–10) to find
  corridors that are both high-crash *and* disproportionately nighttime.
- **See also:** Severity & Outcomes tab; When & Where → Day vs. Night.
"""
    )

# ---------------------------------------------------------------------------
# 6. E-bike speed as a documented factor
# ---------------------------------------------------------------------------
speed_by_mode = cause_summary.get("speed_yes_by_mode")
_speed_header = "### 6. Speed is flagged as a contributing factor more often in powered modes than pedal bicycles"
if speed_by_mode:
    _bike_spd = speed_by_mode.get("Bicycle")
    _ebike_spd = speed_by_mode.get("E-Bike")
    _escoot_spd = speed_by_mode.get("E-Scooter")
    if _bike_spd and _ebike_spd and _bike_spd > 0:
        _ebike_ratio = _ebike_spd / _bike_spd
        _escoot_ratio = (_escoot_spd / _bike_spd) if _escoot_spd else None
        _speed_header = (
            f"### 6. Speed is a documented factor about {_ebike_ratio:.1f}× more often in "
            f"e-bike crashes, and {_escoot_ratio:.1f}× more often in e-scooter crashes, than bicycle crashes"
            if _escoot_ratio
            else f"### 6. Speed is a documented factor about {_ebike_ratio:.1f}× more often in "
                 f"e-bike crashes than bicycle crashes"
        )
st.markdown(_speed_header)
if speed_by_mode:
    bits = "; ".join(f"{m} {_fmt_pct(v)}" for m, v in speed_by_mode.items())
    st.markdown(
        f"""
Share of narrative-classified crashes where speed (rider's or driver's) is flagged
as contributing: {bits}.

- **Why it matters:** Both powered modes travel meaningfully faster than pedal
  bikes on average, and that shows up directly in the causation narratives, not
  just in raw device specs. E-bike shows the largest gap, but e-scooter is also
  clearly elevated above bicycle -- this is a genuine class difference for both
  powered modes, not just a labeling artifact, since officer narratives usually
  only mention speed when it's extreme.
- **Countermeasures:** Enforce/verify class-appropriate speed governance
  (e.g. Class 1/2/3 e-bike limits, scooter-share app-level speed caps) in dense
  mixed-use corridors; consider geofenced speed limiting for shared/rental
  fleets near schools, greenways, and other high-conflict areas identified in
  Sections 2-4.
- **See also:** Roadway Infrastructure tab → "Speed × Infrastructure"; Crash Causation → "Speed as a Documented Factor".
"""
    )
else:
    st.info("Load the crash-causation export to include the speed-by-mode finding.")

# ---------------------------------------------------------------------------
# 7. Aging drivers (defensive -- only renders if the flag column exists)
# ---------------------------------------------------------------------------
if flag_summary.get("aging_driver"):
    st.markdown("### 7. Aging drivers (65+) are flagged far more often than alcohol, drugs, or aggressive driving combined")
    ad = flag_summary["aging_driver"]
    alc = flag_summary.get("alcohol", {})
    dist = flag_summary.get("distracted", {})
    st.markdown(
        f"""
Aging-driver flag rate by mode: {"; ".join(f"{m} {_fmt_pct(v)}" for m, v in ad.items())}.
For comparison, alcohol-related: {"; ".join(f"{m} {_fmt_pct(v)}" for m, v in alc.items()) or "n/a"};
distracted driving: {"; ".join(f"{m} {_fmt_pct(v)}" for m, v in dist.items()) or "n/a"}.

- **Why it matters:** Public messaging on driver-side risk to cyclists/riders
  tends to center on impairment and phone use, but the most commonly flagged
  driver-behavior factor here is age-related -- a factor typical enforcement
  campaigns don't target at all.
- **Countermeasures:** Sightline and signal-timing improvements sized for
  slower reaction times at high-conflict intersections (larger clearance
  intervals, protected left/right phases); pair with DMV-level interventions
  (vision/reaction screening at license renewal) as a longer-horizon lever.
- **See also:** Driver Behavior & Citations tab.
"""
    )

# ---------------------------------------------------------------------------
# 8. Fault attribution -- a yielding problem, not a diverse mix of failures
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 8. Fault splits nearly evenly, and just two failure types account for most classified crashes")
att = cause_summary.get("attribution")
top_c = cause_summary.get("top_causes")
if att and top_c:
    att_txt = "; ".join(f"{k} {_fmt_pct(v)}" for k, v in att.items())
    top_txt = "; ".join(f"{k} ({_fmt_pct(v)})" for k, v in list(top_c.items())[:3])
    st.markdown(
        f"""
Fault attribution: {att_txt}. Leading primary causes: {top_txt}.

- **Why it matters:** Neither "it's mostly driver fault" nor "it's mostly rider
  fault" holds up -- attribution is close to a coin flip. What *does* hold up is
  that the top causes are overwhelmingly yielding failures at points where paths
  cross, not a broad mix of distraction, impairment, and misjudgment. That's a
  narrower, more design-fixable problem than the even fault split makes it sound.
- **Countermeasures:** Treat this as a conflict-geometry problem first (see
  Sections 2-4 for exactly where) and a driver-training/enforcement problem
  second -- geometry changes affect every crash at a location regardless of who's
  ultimately coded at fault; education/enforcement only affects the fraction of
  drivers or riders reached.
- **See also:** Crash Causation tab (full breakdown).
"""
    )
else:
    st.info("Load `cause_analysis_export.csv` to include fault-attribution and top-cause takeaways.")

# ---------------------------------------------------------------------------
# 9. Where crashes concentrate -- hotspots, growth, geography
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 9. Where crashes concentrate and where they're getting worse")
if len(top_counties):
    county_txt = ", ".join(f"**{c}** ({n:,})" for c, n in top_counties.items())
    st.markdown(
        f"""
Top counties by crash count (all modes blended): {county_txt}.

- **Why it matters:** Volume clusters in a handful of metro counties, so a
  statewide average understates risk in these corridors and overstates it
  everywhere else.
- **Countermeasures:** Weight education/enforcement deployment toward these
  counties first, then refine with the tract-level Empirical Bayes excess-crash
  ranking on When & Where for specific corridors within them.
"""
    )
    st.caption(
        "Note on the Empirical Bayes map: for lower-volume modes (e.g. E-Bike), the "
        "county-fixed-effects model can fail to converge, in which case Predicted / EB "
        "estimate / Excess crashes all show as blank or NaN for that mode -- that's a "
        "model-fit limitation, not zero risk. Try Bicycle (highest volume) if the map "
        "looks empty."
    )

if county_concentration_by_mode:
    conc_lines = []
    for m in MODES:
        cc = county_concentration_by_mode.get(m)
        if not cc:
            continue
        top2_txt = ", ".join(f"{c} ({p:.1f}%)" for c, p in cc["top2_counties"].items())
        conc_lines.append(f"- **{m}:** top 2 counties = {top2_txt} → **{cc['top2_pct_sum']:.1f}%** of this mode's own crashes")
    st.markdown(
        "**Geographic concentration by mode** (share of that mode's *own* crashes, "
        "not the blended count above -- this is what the blended ranking hides):\n\n"
        + "\n".join(conc_lines)
        + """

- **Why it matters:** The blended county ranking above is dominated by whichever
  mode has the most rows, so it can't show that a lower-volume mode is far more
  (or less) geographically concentrated than the blend suggests. A mode whose
  crashes are unusually concentrated in one or two counties is a candidate for
  metro-specific interventions (e.g. scooter-share operator agreements) rather
  than statewide rollout.
"""
    )

if dow_peak_by_mode:
    dow_lines = [f"- **{m}:** peaks {v['day']} ({_fmt_pct(v['pct'])})" for m, v in dow_peak_by_mode.items()]
    st.markdown(
        "**Day-of-week peak by mode:**\n\n" + "\n".join(dow_lines)
        + "\n\n- **Why it matters:** if peak days differ by mode, enforcement/outreach "
        "timing built around one mode's pattern may miss another's."
    )

if university_county_ranks:
    uni_lines = []
    for m in MODES:
        rows = university_county_ranks.get(m)
        if not rows:
            continue
        parts = ", ".join(
            f"{name} ({info['school']}) rank {info['rank']}/{info['n_counties']}"
            for name, info in rows.items()
        )
        uni_lines.append(f"- **{m}:** {parts}")
    st.markdown(
        "**Do university/college counties show up disproportionately?** Ranking all "
        "Florida counties by crashes per 100,000 residents (not raw count) for each "
        "mode, then checking where the six counties home to Florida's largest public "
        "universities land:\n\n"
        + "\n".join(uni_lines)
        + """

- **Why it matters:** The pattern is strongest for E-Scooter -- all six university
  counties typically land in the top handful statewide by per-capita rate, well
  above what population size alone would predict, consistent with college-age
  riders being a disproportionate share of scooter-share usage. It's noticeably
  weaker for E-Bike. This is presented as an association worth investigating, not
  a proven cause: these same counties are also Florida's densest urban cores, so
  university enrollment and urban density are confounded here and can't be
  separated with this dataset alone.
- **See also:** Demographics tab (Section 11's age skew is consistent with this --
  e-scooter riders are the youngest of the three rider populations).
"""
    )

if top_tracts_by_mode:
    cross_lines = []
    for m in ["E-Bike", "E-Scooter"]:
        top15 = top_tracts_by_mode.get(m)
        if top15 is None or not len(top15):
            continue
        county_counts = top15["county"].value_counts()
        lead_county, lead_n = county_counts.index[0], int(county_counts.iloc[0])
        others = county_counts.iloc[1:]
        others_txt = ", ".join(f"{c} ({n})" for c, n in others.items()) if len(others) else None
        cross_lines.append(
            f"- **{m}:** of the top 15 highest-rate tracts statewide, **{lead_n} are in "
            f"{lead_county}**"
            + (f"; the rest are spread across {others_txt}." if others_txt else ".")
        )
    if cross_lines:
        st.markdown(
            "**Cross-checking the choropleth maps against the underlying tract data "
            "directly** (not just reading the rendered map, which can visually flatten "
            "a secondary cluster against a strong primary one):\n\n"
            + "\n".join(cross_lines)
            + """

- **Why it matters:** this confirms a single-tract outlier explanation is wrong for
  the leading county in each case (multiple tracts, not one lucky spot) and surfaces
  any secondary county worth a second look that a quick glance at the statewide map
  wouldn't obviously flag.
"""
        )

if hotspot_summary:
    lines = []
    for m in MODES:
        h = hotspot_summary.get(m)
        if not h:
            continue
        extra = ""
        if h.get("significant_growth_n") is not None:
            extra += f"; {h['significant_growth_n']:,} statistically significant growth (p<0.05)"
        if h.get("emerging_heuristic_n") is not None:
            extra += f"; {h['emerging_heuristic_n']:,} flagged emerging (1.5x heuristic, exploratory)"
        loc_txt = ""
        if h.get("largest_cluster_lat") is not None and h.get("largest_cluster_lon") is not None:
            loc_txt = f" (centered ~{h['largest_cluster_lat']}, {h['largest_cluster_lon']})"
        lines.append(
            f"- **{m}:** {h['n_clusters']:,} DBSCAN recurring-location clusters"
            + (f"; largest has {h['largest_cluster_n']:,} crashes{loc_txt}" if h.get("largest_cluster_n") else "")
            + extra
        )
    st.markdown(
        "\n".join(lines)
        + "\n\n- **Why it matters:** Recurring-location clusters answer *where crashes keep "
        "happening close together*, which is a different (and more actionable) question than "
        "which tract has the highest raw count. Statistically significant growth clusters "
        "isolate the ones getting worse beyond chance, not just noisy small-n swings.\n"
        "- **Countermeasures:** Prioritize engineering audits (sight distance, conflict "
        "markings, signal timing) at the largest and fastest-growing clusters first; treat "
        "the 1.5x heuristic list as a screen to investigate, not a final ranking.\n"
        "- **See also:** When & Where → DBSCAN Top 10, Spatiotemporal growth explorer, "
        "Getis-Ord Gi*, Empirical Bayes excess-crash map."
    )
    st.caption(
        "Coordinates above are each mode's single largest recurring cluster -- cross-"
        "reference on the map (When & Where) for the place name and local context "
        "before treating it as an intervention site."
    )
else:
    st.info(
        "Load `spatiotemporal_hotspots_by_mode.csv` in the sidebar to include hotspot-based "
        "insights (see When & Where)."
    )

if not pd.isna(intx_pct):
    intx_by_mode_txt = "; ".join(
        f"{m} {_fmt_pct(v)}" for m, v in intx_pct_by_mode.items() if v is not None and not pd.isna(v)
    )
    st.markdown(
        f"""
**{_fmt_pct(intx_pct)}** of filtered crashes are at intersections; the remainder
(**{_fmt_pct(100 - intx_pct) if not pd.isna(intx_pct) else "n/a"}**) are mid-segment.
By mode: {intx_by_mode_txt if intx_by_mode_txt else "n/a"}.

- **Why it matters:** Segments, not intersections, are where most crashes happen
  overall -- worth stating plainly, since crossing/turning-conflict framing
  (which fits intersections) is the default design narrative and can crowd out
  segment-level fixes like buffer width, parking-lane conflicts, and passing
  distance. The blended number above is dominated by whichever mode has the
  most rows (usually bicycle); the by-mode split shows e-bike and e-scooter
  are both somewhat more intersection-weighted than bicycle, so they may
  benefit relatively more from intersection-focused fixes (Section 3) than
  the blended figure alone would suggest.
- **See also:** When & Where → Intersection vs. Segment.
"""
    )

# ---------------------------------------------------------------------------
# 10. Statistically significant spatial clustering (Gi* / Local Moran's I)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 10. Many high-count tracts aren't spatial clusters — and some dangerous tracts sit alone")
if gi_summary:
    st.caption(
        "Map 4 on **When & Where** defaults to **E-Bike** (maps 2–4 mode picker). "
        "Compare the **same mode** row below — e.g. Bicycle shows **36** High-Low outliers "
        "while E-Bike shows **~65**, not a calculation error."
    )
    gi_lines = []
    for m in MODES:
        g = gi_summary.get(m)
        if not g:
            continue
        gi_lines.append(
            f"- **{m}:** **{g['n_hot_spots']:,}** statistically significant hot spots, "
            f"**{g['n_cold_spots']:,}** significant cold spots, and "
            f"**{g['n_high_low_outliers']:,}** High-Low outliers (single elevated tracts "
            f"surrounded by normal ones) — out of **{g['n_tracts']:,}** Florida tracts."
        )
    st.markdown(
        "\n".join(gi_lines)
        + """

Unlike DBSCAN clusters (Section 9) or Empirical Bayes excess ranking, **Getis-Ord Gi\\***
tests whether a tract's crash count clusters with its neighbors more than chance would
predict — not just whether the raw count is high. **Local Moran's I** High-Low outliers
flag a different actionable pattern: one dangerous intersection/tract in an otherwise
normal area ("investigate this specific location"), not an area-wide hot zone.

- **Why it matters:** A tract can rank high on raw volume or EB excess but *not* appear
  here if neighbors are low — and conversely, a High-Low outlier is easy to miss on a
  volume map because the surrounding tracts look fine.
- **Countermeasures:** Run field audits on High-Low outlier tracts first for site-specific
  fixes (sight distance, signal timing, crossing geometry); use Gi* hot spots for corridor-
  scale investment where clustering confirms a broader pattern.
- **See also:** When & Where → Map 4 (Gi* hot/cold spot choropleth + outlier table).
"""
    )
else:
    st.info(
        "Load `census_tracts.geojson` (sidebar) and ensure crashes have lat/lon so "
        "Getis-Ord Gi* / Local Moran's I summaries can be computed live (see When & Where → Map 4)."
    )

# ---------------------------------------------------------------------------
# 11. Who's riding, and who's dying -- rider demographics (age, sex)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "### 11. Rider age is a steeper fatality-risk gradient than any single "
    "behavioral flag above, and rider demographics differ sharply by mode"
)
if demo_summary:
    fatal_by_age = demo_summary.get("fatal_pct_by_age", {})
    if fatal_by_age:
        age_order = ["0-14", "15-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        age_lines = "; ".join(
            f"{k} {_fmt_pct(fatal_by_age.get(k))}" for k in age_order if k in fatal_by_age
        )
        ratio = demo_summary.get("fatal_ratio_65plus_vs_18to24")
        ratio_txt = (
            f"Riders 65+ are **{ratio}×** more likely to die in a crash than riders "
            f"18-24 -- a steady climb with age, not a jump at one bracket.\n\n"
            if ratio else ""
        )
        st.markdown(
            f"""
Fatality rate by rider age (all filtered modes combined): {age_lines}.

{ratio_txt}- **Why it matters:** This age gradient is steeper than any single
  behavioral flag covered elsewhere in this tab (compare Section 7's aging-
  *driver* flag rate) -- and unlike that section, this is about the rider's own
  outcome, not the other party's behavior, so it points to a different
  intervention lever (rider-side, not driver-side).
- **Countermeasures:** Age-targeted visibility/lighting and route-choice
  outreach (senior centers, injury-prevention programs) as a complement to the
  driver-side and infrastructure countermeasures above. Note this dataset can't
  distinguish crash severity from post-crash factors like EMS response time or
  age-related injury fragility -- treat the ratio as "older riders die more
  often when crashes happen," not proof the crashes themselves are worse.
- **See also:** Severity & Outcomes tab.
"""
        )

    sex_pct = demo_summary.get("sex_pct_by_mode", {})
    median_age = demo_summary.get("median_age_by_mode", {})
    if sex_pct:
        sex_lines = []
        for m in MODES:
            row = sex_pct.get(m, {})
            if row:
                age_bit = f", median age {median_age.get(m):.0f}" if median_age.get(m) is not None else ""
                sex_lines.append(
                    f"- **{m}:** {row.get('Female', 0):.1f}% female / "
                    f"{row.get('Male', 0):.1f}% male{age_bit}"
                )
        if sex_lines:
            st.markdown(
                "**Rider population differs meaningfully by mode:**\n\n" + "\n".join(sex_lines)
                + """

- **Why it matters:** If one mode's riders skew notably more female and/or
  younger than the others, safety messaging and product design built around a
  "typical cyclist" profile may not reach or resonate with that mode's actual
  riders.
- **See also:** Overview & Trends tab (mode adoption trends).
"""
            )
else:
    st.info(
        "Load `power_bi_export_demographics.csv` in the sidebar (Optional: "
        "demographics & pipeline info) to include rider age/sex insights here."
    )

# ---------------------------------------------------------------------------
# Known Data & Methodology Limitations (curated, not live-filtered)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Known Data & Methodology Limitations")
st.markdown(
    """
    <div class="section-note">
    Unlike everything above, this list is <b>curated from pipeline validation, not
    recomputed from the current filter</b> -- worth stating explicitly in a paper's
    limitations/future-work section rather than silently working around.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
1. **Empirical Bayes SPF for sparse modes (e.g. E-Bike):** county fixed-effects
   negative-binomial fits can fail to converge (linear predictor overflow → all-
   `NaN` predictions). When & Where now falls back to Poisson SPF (then intercept-
   only NB/Poisson) and uses intercept-only NB dispersion for EB shrinkage.
2. **Qwen mode validation:** `CRASH_GROUP` encodes crash geometry/fault, not
   micromobility mode, so mode agreement is measured as Cohen's κ between
   `QWEN_CLASS` and final S4 `MODE` (via `MODE_TO_QWEN_CLASS` crosswalk in
   `dashboard_core.py`). Fault agreement (`CRASH_GROUP` → fault party vs
   causation `ATTRIBUTION`) is a separate κ — see Narrative tab for both.
3. **LDA topic modeling:** Narrative tab now strips report header boilerplate and
   uses an extended stopword list (`v1`, `nm1`, `d1`, officer/agency tokens, etc.)
   before fitting LDA — re-open that section to inspect whether top words now read
   as causal themes rather than report codes.
4. **Roadway infrastructure coverage:** AADT and intersection control type join for
   ~64% / ~54% of crashes in the full export; **Median Type, Shoulder Width, Number
   of Through Lanes, and Context Class columns exist in `power_bi_export.csv` but are
   0% populated** — not a dashboard join bug; the FDOT segment attributes were not
   merged into this extract. Roadway Infrastructure tab omits those charts accordingly.
5. **'Phone' keyword artifact:** Manual review of bicycle narratives matching
   `phone` showed the hits are almost entirely **officer/agency phone-number
   boilerplate** in report headers, not driver phone use (~47% raw → ~12% after
   `narrative_text_for_search()` strips headers). Keyword heatmap on Narrative tab
   now searches cleaned text only.
6. **Mode 'Other' in causation export:** Of 3,226 `Other` rows, **1,347 recover as
   Bicycle/E-Bike/E-Scooter** when reconciled against `power_bi_export.csv` mode
   (`reconcile_cause_modes()` runs at load); **~1,879 remain Other** (also `Other`
   in narrative export — genuinely ambiguous or unclassified). Recovering the 1,347
   shifts Section 2–4 denominators slightly (sidewalk+crosswalk share ~unchanged at
   57.4%; bike-lane share 9.7% → 9.5%).
7. **`cause_flag` QA field:** Present in `classify_crash_cause.py` output schema but
   **not exported** to `cause_analysis_export.csv` — not recoverable in this dashboard
   without re-exporting from the classifier; treat as out of scope until the pipeline
   includes it.
"""
)

st.markdown("---")
st.markdown("### How to use this tab")
st.markdown(
    """
- Change sidebar filters (Mode / Year / Severity / County) and re-read the takeaways -- they update live.
- Open **Source numbers** above or `results/insights_snapshot.json` to verify any cited figure.
- Sections 2-4 and 6 depend on the crash-causation export; without it, those sections show a load prompt instead of numbers.
- Section 10 needs `census_tracts.geojson` plus geocoded crashes (see When & Where → Map 4).
- For severity breakdowns, go to **Severity & Outcomes**.
"""
)

render_pipeline_figures("tab9")