# Active-Mode Crash Dashboard

Interactive Streamlit dashboard for exploring bicycle, e-bike, and e-scooter
("active-mode") crashes in Florida, built on Signal4 crash tables and FDOT
roadway data.

Just & Green Transportation Lab, University of Florida.

## Pipeline

The dashboard reads from CSVs produced by a multi-stage pipeline. Run in order:

1. **`classify_emobility.py`** *(run separately, on HiPerGator)* — classifies
   crash narratives into Bicycle / E-Bike / E-Scooter / Other using
   Qwen2.5-72B served via vLLM. Produces the narrative-label output the next
   step consumes.
2. **`eda_analysis_combined.py`** — the main EDA pipeline. Merges
   crash_event, non_motorist, and vehicle tables with the narrative labels
   and FDOT roadway data, and writes:
   - `power_bi_export.csv` — crash-level export (mode, timing, location,
     severity, driver-behavior flags, citations, roadway infra, road type,
     posted speed, lat/lon)
   - `power_bi_export_demographics.csv` — person-level age/gender export
   - `dashboard_meta.csv` — pipeline funnel counts (raw → geocoded →
     matched → final)
   - `results/figures/` — static PNGs by section, auto-discovered and
     embedded into the dashboard's tabs
3. **`dashboard.py`** — the Streamlit app itself.

**If you add a new figure to `eda_analysis_combined.py` and don't see it in
the dashboard**, it's almost always one of these two things, not a dashboard
bug — see the FAQ item on this below:
- you haven't re-run `eda_analysis_combined.py` since adding the figure, so
  the PNG doesn't exist on disk yet, or
- it does exist, but it's sitting inside a **collapsed** "🖼️ Pipeline
  figures from `eda_analysis_combined.py`" expander at the bottom of the
  relevant tab — click to open it.

## Setup

### On HiPerGator (how this is actually being run right now)

There's no dedicated conda/venv for this project yet — the working setup is
the **`pytorch` module's Python**, which already has pandas / numpy /
matplotlib / seaborn / scikit-learn / openpyxl installed for this user. Every
new login session, before running anything:

```bash
module load pytorch/2.8.0
cd /blue/xiangyan/rithika/stats
```

then run the pipeline / dashboard as normal, e.g.:

```bash
python3 eda_analysis_combined.py
```

Notes:
- `module load` only lasts for the current shell session — you need to run
  it again after logging back in. It is **not** remembered by a `.sbatch`
  batch job either; add `module load pytorch/2.8.0` as a line inside any
  `.sbatch` script before the `python3` call, or the job will fail with
  `ModuleNotFoundError: No module named 'pandas'` even though it works fine
  interactively.
- Verify you're pointed at the right interpreter with `which python3` /
  `python3 --version` — it should resolve to
  `/apps/pytorch/2.8.0/bin/python3`, version `3.13.x`. Plain `python3` with
  no module loaded falls back to the OS's bare `/usr/bin/python3` (3.9),
  which has none of these packages and will fail immediately on `import
  pandas`.
- A cleaner long-term option is a dedicated conda env (`conda create -n
  stats ...`) or venv scoped to this project instead of riding on the
  pytorch module — not set up yet; flagged as a nice-to-have, not blocking
  anything currently.

### Running the dashboard locally (off HiPerGator, e.g. on a laptop)

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip3 install -r requirements.txt
```

## Running locally

Place `power_bi_export.csv`, `power_bi_export_demographics.csv`,
`dashboard_meta.csv`, and the `results/figures/` folder next to
`dashboard.py` (or upload them via the sidebar), then:

```bash
streamlit run app.py
```

## Data notes

- `S4_LATITUDE`/`S4_LONGITUDE` (preferred, ~98.5% complete) or the raw
  `LATITUDE`/`LONGITUDE` are required in `power_bi_export.csv` for the
  Florida crash-location map on the "When & Where" tab. `eda_analysis_combined.py`
  merges these in automatically.
- The demographics and meta files are optional — the dashboard degrades
  gracefully (skips the affected tabs/sections) if they're missing.

## Deployment

https://ebikeescooterpowerbidashboard-dpdpqduho5p45wyrmpenmq.streamlit.app/

---

## FAQ / Analysis Brief — Status of Each Question

This section works through the questions raised on the analysis, in order,
with what's done, what's still open, and why. ✅ = done · 🟡 = partial /
needs input from the team · ⛔ = pending, blocked on something outside the
code.

### For most analyses, break out a) all crashes, b) KSI, c) fatalities

- **Dashboard: ✅ Done.** Sidebar → Injury Severity filter now has three
  quick-preset buttons — **All / KSI / Fatal** — above the checkbox list.
  Clicking one filters *every* tab at once (Severity, When/Where, Driver
  Behavior, Citations, Roadway Infrastructure, Demographics), since nearly
  every chart reads off the same filtered dataframe. This is the
  recommended way to explore the split — it's live, not three static
  copies of every chart.
- **`eda_analysis_combined.py` (static PNGs): 🟡 Partial.** `is_ksi()` /
  `is_fatal()` helpers exist and are applied to Citation Rate by Mode and
  Driver Behavior Flags by Mode (each now also has `_ksi` and `_fatal`
  CSV/print output). Not yet extended to every other static figure (Road
  Type, Light/Weather, Age, Day/Night, spatial/county trend, hour-of-day) —
  tell us which ones matter most for the writeup and we'll extend the
  pattern; it's mechanical once the helper exists.

### Trend analysis (spatial & temporal): use all micromobility, not just
"Bicycle"

**✅ Already correct by design.** Spatial/temporal trend charts in both
files are built from `ACTIVE_MODES`/`micro_df` (all micromobility — Bicycle
+ E-Bike + E-Scooter, where "Bicycle" here includes both Qwen-labeled and
structurally-classified bicycle crashes). The narrower Qwen-only "Bicycle"
label is used only in the mode-vs-mode comparison charts, per the brief.

### Data processing steps & limitations need to be described carefully

**🟡 Partial — drafted, one date still needs confirming.** FLHSMV/Signal4
first applied keyword search to statewide crash-narrative text to pull a
candidate pool of e-bike/e-scooter crashes, **covering January 1, 2014
through 2026**, plus a comparison sample of ~5,000 narratives that did NOT
match those keywords. Candidates were then classified by the Qwen
multilabel model (`multilabel_ebike.xlsx` / `multilabel_RegBike.xlsx`) into
Bicycle / E-Bike / E-Scooter / Other. Two limitations that follow directly
and should be repeated wherever E-Bike/E-Scooter totals are quoted:
1. Keyword extraction is precision-oriented, not a random sample — a crash
   whose narrative never mentions an e-bike/e-scooter term is never in the
   candidate pool, so absolute E-Bike/E-Scooter counts are a **floor**, not
   a census.
2. "Bicycle" from this pipeline is a **subset** of all bicycle crashes in
   Signal4 — see the trend-analysis item above.

**⛔ Still open:** whether the 2026 end of the range is end-of-June 2026 or
a later cutoff — confirm with FLHSMV/Xingjing before stating a specific end
date in a final report.

### Why does E-Bike crash timing cluster at 3–4pm?

**🟡 Diagnostic added, not a definitive answer.** The hourly chart now
prints each mode's share of crashes at its peak hour, so you can tell
whether 3–4pm is a *shared* PM-commute/school-dismissal peak across all
three modes, or E-Bike-specific. `eda_analysis_combined.py` also adds a
`%`-within-mode hourly-shape chart with the 3–4pm window shaded. Actually
answering "why" needs a follow-up cross-tab (E-Bike crashes at 3–4pm ×
rider age band × school-day vs. weekend) — not built yet.

### Higher fatal/serious-injury and citation rates for "Bicycle" — is that
right?

**⛔ Not a code fix — a real analytic observation to sanity-check.** One
confound worth checking before writing this up: Bicycle is a much larger,
more heterogeneous sample than E-Bike/E-Scooter (which come only from the
narrower Qwen-labeled pool), so its rates sit on more stable statistical
footing — E-Bike/E-Scooter rates could still shift meaningfully as more
narratives get classified. No confidence intervals are on the rate charts
yet; can be added on request so "higher" is backed by an interval, not just
a point estimate.

### "Severity mix over time" should use proportions

**✅ Done, both files.** `eda_analysis_combined.py`: `05c` is now a %
stacked chart (old raw-count version kept as `05d` so volume growth is
still visible separately). Dashboard: "Severity Mix Over Time" now plots %
of that year's crashes instead of raw stacked counts.

### What's the Day vs. Night definition?

**⛔ Pending — needs FLHSMV/Signal4 to confirm.** `DAY_NIGHT` is a
pass-through of the pre-populated `S4_DAY_OR_NIGHT` field in
`crash_event.csv`; neither script computes it. This data extract doesn't
document FLHSMV's exact rule (clock cutoff vs. sunrise/sunset vs. civil
twilight, statewide vs. per-location).

### New comparison figures across Bicycle / E-Bike / E-Scooter (Light,
Weather, Driver Behavior, Intersection Control, Road Type, Age Band)

**✅ Done — and converted to % within mode**, since raw counts let whichever
mode has more rows swamp every bar and hide whether the *shape* of the
distribution actually differs. (In the most recent full run the three
active modes came out fairly close in size — Bicycle 5,620 / E-Bike 6,353 /
E-Scooter 5,274 — but %-within-mode is the right default regardless, since
that balance isn't guaranteed to hold as more narratives get classified.)

| Figure | `eda_analysis_combined.py` | Dashboard |
|---|---|---|
| Light Conditions by Mode | `04e_light_conditions` (now %) | now % |
| Weather Conditions by Mode | `04f_weather_conditions` (now %) | now % |
| Driver Behavior by Mode | `10a_driver_flags_by_mode` (already %) | already % |
| Intersection Control by Mode | `13e_intersection_control_by_mode` (now %) | now % |
| Road Type by Mode | `04c_road_type` (now %) | now % |
| Age Distribution by Band, by Mode | new `02d_age_band_by_mode` | already existed |
| Median Type by Mode | `13b_median_type_by_mode` (now %) — **confirmed skipped in the latest run**: `MEDIAN_TYPE` is empty for every crash matched to `roadway_segment.csv` in this data extract, not a code bug (script now prints an explicit skip message saying so) | n/a — no data to show |
| Lane Count by Mode | `13d_lane_count_by_mode` (now %) — **confirmed skipped**: `NUM_THRU_LANES` is likewise empty for all matched crashes in this extract | n/a — no data to show |

### Explore spatiotemporal clustering for emerging hotspots

**✅ First pass done, exploratory.** New section `09d` in
`eda_analysis_combined.py` (DBSCAN per mode) finds spatial clusters, splits
each into early- vs. late-period (median crash year for that mode), and
flags clusters where the later period has >1.5× the earlier period's count
as "emerging." Outputs `spatiotemporal_hotspots_by_mode.csv` and a map
figure (`09d_emerging_hotspots_by_mode.png`) — this shows up under Tab 3
("When & Where") in the "Pipeline figures" expander once the script has
been run with `scikit-learn` installed.

**Caveats, please read before using in a deliverable:** `eps`/`min_samples`
are a reasonable starting guess, not tuned; the early/late split is a
coarse proxy for "emerging," not a validated space-time statistic. The
standard tool if this needs to hold up formally is a space-time scan
statistic (SaTScan / Getis-Ord Gi* on a space-time grid) — not
implemented, flagged as a follow-up.

### Relative crash risk = crash frequency ÷ Strava cycling volume, by
county (and tract)

**⛔ Pending — blocked on external data, exactly as flagged originally.**
`eda_analysis_combined.py` section `09e` is written to compute this
automatically the moment a Strava county-volume file exists at
`{DATA}/strava_county_volume.csv` (columns: `COUNTY_NAME` +
`STRAVA_TRIPS`/`STRAVA_MILES`) — until then it prints a "pending" message
and exits cleanly. Tract-level needs one more step not yet built: a spatial
join of crash lat/lon to Census TIGER tract polygons. **Action item:** get
the Strava export from Xingjing.

### Why is the citation rate declining over time? Add a line per mode.

- **✅ Per-mode lines: done, both files.** `eda_analysis_combined.py`: new
  `11c_citation_rate_over_time_by_mode`. Dashboard: "Citation Rate Over
  Time" now plots one line per mode instead of one pooled line.
- **⛔ "Why": not answered.** Leading hypothesis to rule out first: a
  reporting-lag artifact, where the most recent 1–2 years haven't had all
  citations entered into `violation.csv` yet (citations can be filed after
  the initial crash report) — check whether the decline is concentrated in
  the latest year(s) before treating it as a real behavioral trend.

### Age Distribution by Injury Severity as three separate figures (one per
mode)

**✅ Done, both files.** `eda_analysis_combined.py`: new
`02e_age_by_severity_bicycle`, `02f_age_by_severity_ebike`,
`02g_age_by_severity_escooter` (this comparison didn't exist at all before).
Dashboard Tab 6 now renders three side-by-side violin plots (one per mode)
instead of one combined chart.

---

## Known UI Fixes

- **Double-box outline on the All/KSI/Fatal preset buttons (and the
  Download/Reset buttons):** the sidebar CSS was setting `border` on
  `button *` (every element *inside* the button, not just the button
  itself), which drew a second nested border hugging the label text.
  Fixed by scoping the border rule to the `<button>` element only.

## Other fixes made along the way (not explicitly requested)

- **Bug fix:** the original Lane Count figure (`13d`) had a stray
  unconditional `save(fig, ...)` call *outside* its `if len(lane_ct):`
  guard — when lane data was empty, this saved a stale/blank figure left
  over from whatever plot ran before it. Removed.
- Added CSV exports for every new %-within-mode table (e.g.
  `road_type_by_mode_pct.csv`, `light_condition_by_mode_pct.csv`,
  `age_band_by_mode_pct.csv`) alongside the raw-count versions that already
  existed, so nothing is lost.
- **Bug fix:** the `S4_CRASH_TYPE breakdown` note under "Data Loading" was
  printing hardcoded example numbers ("Bicycle (59k), Pedestrian (14k),
  Single Vehicle (22k)") left over from an earlier/different data pull,
  instead of the actual counts for the current run. Now computed dynamically
  from `ct_counts` every run, so it can't drift out of sync with the real
  data again.
- Added an explicit skip message for `13b_median_type_by_mode` (previously
  it just silently didn't produce a file when `MEDIAN_TYPE` had no data,
  matching the message `13d` already had for `NUM_THRU_LANES`).

## First Full Run — Confirmed Results (worth keeping on record)

Ran clean end-to-end on the full dataset (20,437 crashes; 17,247
active-mode: Bicycle 5,620 / E-Bike 6,353 / E-Scooter 5,274 — the three
modes came out fairly close in size, not lopsided). A few things the run
surfaced that are worth flagging to the team rather than just leaving in a
console log:

- **3–4pm timing (README §22):** at E-Bike's peak hour (16:00–17:00),
  E-Bike is 10.5% of its own crashes and E-Scooter is 10.2% of its own,
  vs. Bicycle at 8.0%. That's a real, if modest, gap — mild support for an
  E-Bike/E-Scooter-specific explanation (school dismissal, delivery/gig
  timing) rather than pure shared PM-commute overlap. Still needs the
  age-band/school-day cross-tab to actually confirm a cause.
- **Citation rate reverses by tier (README §1/§25):** Bicycle has the
  highest citation rate at the *all-crashes* (38.3%) and *KSI* (44.9%)
  tiers, but at the *fatal-only* tier E-Bike drops to 14.4% while Bicycle
  (26.8%) and E-Scooter (22.9%) stay much closer together. Worth a closer
  look before writing up "Bicycle has the highest citation rate" as a
  blanket statement — it depends which severity tier you're looking at.
- **Hotspot chaining check came back clean:** none of the DBSCAN clusters
  in this run held more than 10% of their mode's total geocoded crashes, so
  no `[CHECK]` chaining warning fired. The one large-looking circle in the
  09d figure (E-Scooter, 227 crashes) is a genuine size difference against
  much smaller 15–34-crash clusters, not a DBSCAN artifact from a fixed
  501m radius chaining a road corridor together.
- **13b (Median Type) and 13d (Lane Count) by mode are empty in this data
  extract** — `MEDIAN_TYPE` and `NUM_THRU_LANES` have zero non-null values
  across every crash matched to `roadway_segment.csv`. This is a real gap
  in the FDOT roadway-segment join for this pull, not a script bug; both
  now print an explicit skip message saying so instead of silently
  producing nothing.