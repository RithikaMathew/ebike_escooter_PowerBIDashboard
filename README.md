# Active-Mode Crash Dashboard

Interactive Streamlit dashboard for exploring bicycle, e-bike, and e-scooter ("active-mode") crashes in Florida, built on Signal4 crash tables and FDOT roadway data.

**Just & Green Transportation Lab, University of Florida.**

Live deployment: https://ebikeescooterpowerbidashboard-dpdpqduho5p45wyrmpenmq.streamlit.app/

---

## Repository Structure

```
.
├── app.py                                              # Streamlit entry point
├── dashboard.py                                        # Delegates to app.py
├── dashboard_core.py                                   # Core dashboard logic + data loading
├── classify_crash_cause.py                             # Crash cause classification
├── build_census_tracts.py                              # Builds census tract GeoJSON
├── eda_analysis_combined_BicycleSeparate (2).py        # Main EDA pipeline
├── census_tracts.geojson                               # Florida census tract polygons (optional)
├── requirements.txt
├── tabs/
│   ├── tab0_about.py
│   ├── tab1_overview.py
│   ├── tab2_severity.py
│   ├── tab3_when_where.py
│   ├── tab4_driver_behavior.py
│   ├── tab5_infrastructure.py
│   ├── tab6_demographics.py
│   ├── tab7_narrative.py
│   ├── tab8_causation.py
│   └── tab9_insights.py
└── results/figures/                                    # Static PNGs from EDA pipeline
```

Crash CSVs are **not** in this public repo. They live in the private GitHub repo [`RithikaMathew/powerbi-data`](https://github.com/RithikaMathew/powerbi-data) and are fetched at runtime.

---

## Pipeline

The dashboard reads from CSVs produced by a multi-stage pipeline. Run in order:

1. **`classify_crash_cause.py`** *(run separately, on HiPerGator)* — classifies crash narratives into Bicycle / E-Bike / E-Scooter / Other using Qwen2.5-72B served via vLLM. Produces the narrative-label output consumed by the next step.

2. **`eda_analysis_combined_BicycleSeparate (2).py`** — the main EDA pipeline. Merges crash_event, non_motorist, and vehicle tables with narrative labels and FDOT roadway data, and writes:
   - `power_bi_export.csv` — crash-level export (mode, timing, location, severity, driver-behavior flags, citations, roadway infra, road type, posted speed, lat/lon)
   - `power_bi_export_demographics.csv` — person-level age/gender export
   - `dashboard_meta.csv` — pipeline funnel counts (raw → geocoded → matched → final)
   - `spatiotemporal_hotspots_by_mode.csv` — DBSCAN clusters + early/late growth
   - `narrative_text_export.csv` — raw narrative text
   - `cause_analysis_export.csv` — crash cause analysis output
   - `results/figures/` — static PNGs by section

   Upload the CSV outputs to the private `powerbi-data` repo (keep them out of this public repo).

3. **`app.py`** — Streamlit entry point (`dashboard.py` delegates here).

---

## Setup

### On HiPerGator

The working setup uses the **`pytorch` module's Python**, which has pandas / numpy / matplotlib / seaborn / scikit-learn / openpyxl pre-installed. Every new login session, run:

```bash
module load pytorch/2.8.0
cd /blue/xiangyan/rithika/stats
```

Then run the pipeline as normal:

```bash
python3 "eda_analysis_combined_BicycleSeparate (2).py"
```

> **Note:** `module load` only lasts for the current shell session. Add `module load pytorch/2.8.0` inside any `.sbatch` script before the `python3` call.

### Locally (laptop)

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` includes Streamlit, Plotly, `requests`, and spatial dependencies: `geopandas`, `shapely`, `libpysal`, `esda`, `statsmodels`, `scipy`, `scikit-learn`.

---

## Data Access (private CSVs)

The dashboard loads CSVs from `RithikaMathew/powerbi-data` using a GitHub personal access token stored in Streamlit secrets — never in public code.

### Streamlit Community Cloud

1. Open the app’s settings → **Secrets**.
2. Add (without deleting any existing settings):

```toml
GITHUB_DATA_TOKEN = "github_pat_..."
```

3. Save, then reboot the app.

### Local development

Create `.streamlit/secrets.toml` (already gitignored):

```toml
GITHUB_DATA_TOKEN = "github_pat_..."
```

If a CSV is present next to `app.py`, that local file is used instead of fetching from GitHub (useful for offline work).

### Expected private-repo files
https://github.com/RithikaMathew/powerbi-data

| File | Required |
|---|---|
| `power_bi_export.csv` | yes |
| `power_bi_export_demographics.csv` | optional |
| `dashboard_meta.csv` | optional |
| `narrative_text_export.csv` | optional |
| `spatiotemporal_hotspots_by_mode.csv` | optional |
| `cause_analysis_export.csv` | optional |

Also optional locally: `census_tracts.geojson` (tract maps) and `results/figures/` (static PNG expanders).

---

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard degrades gracefully — tabs and sections that depend on missing optional files are skipped automatically.

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| **About** | Project overview and data source documentation |
| **Overview** | High-level crash counts and mode breakdown |
| **Severity** | Injury severity distributions and trends over time |
| **When & Where** | Temporal patterns, crash maps, hotspot analysis |
| **Driver Behavior** | Driver behavior flags and contributing factor rates |
| **Infrastructure** | Roadway type, intersection control, lighting conditions |
| **Demographics** | Rider age and gender distributions |
| **Narrative** | Crash narrative text mining and contributing factors |
| **Causation** | Crash cause classification and model outputs |
| **Insights** | Integrated findings and cross-tab analysis |

### When & Where tab — map and hotspot layers

- **Map 1** — raw crash counts per census tract, selectable by mode
- **Map 2** — crashes per 100,000 residents (percentile-colored)
- **Map 3** — mode share of micromobility crashes per tract
- **DBSCAN overlay / Top 10** — recurring spatial crash clusters, with county
- **Spatiotemporal growth** — clusters trending worse over time (Poisson rate-ratio primary; 1.5× heuristic exploratory)
- **Getis-Ord Gi\*** — statistically significant hot/cold spots
- **Empirical Bayes** — excess-crash ranking + tract choropleth; SPF is statewide (population exposure) with county fixed effects when the fit is stable

---

## Data Notes

- `S4_LATITUDE`/`S4_LONGITUDE` (~98.5% complete) or fallback `LATITUDE`/`LONGITUDE` are required in `power_bi_export.csv` for crash-location maps.
- Severity risk factors use `S4_CRASH_SEVERITY`.
- The sidebar **All / KSI / Fatal** preset buttons filter every tab simultaneously.
- Crash data covers **January 1, 2014 through 2026** (confirm exact end date with FLHSMV before citing in a report). E-Bike/E-Scooter counts are a **floor** — crashes whose narratives never mention an e-bike/e-scooter keyword were never in the candidate pool.
- Relative crash risk by Strava cycling volume is pending external data (`strava_county_volume.csv`); the pipeline section runs automatically once that file is present.
