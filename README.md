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

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running locally

Place `power_bi_export.csv`, `power_bi_export_demographics.csv`,
`dashboard_meta.csv`, and the `results/figures/` folder next to
`dashboard.py` (or upload them via the sidebar), then:

```bash
streamlit run dashboard.py
```

## Data notes

- `S4_LATITUDE`/`S4_LONGITUDE` (preferred, ~98.5% complete) or the raw
  `LATITUDE`/`LONGITUDE` are required in `power_bi_export.csv` for the
  Florida crash-location map on the "When & Where" tab. `eda_analysis_combined.py`
  merges these in automatically.
- The demographics and meta files are optional — the dashboard degrades
  gracefully (skips the affected tabs/sections) if they're missing.

## Deployment

Deployed via [Streamlit Community Cloud](https://share.streamlit.io):
connect the GitHub repo, point it at `dashboard.py`, and it builds
automatically from `requirements.txt`. See `.gitignore` for what's excluded
from the repo (local venv, node_modules, secrets).
