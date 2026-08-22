"""
build_census_tracts.py

Run this ONCE, locally (needs internet access -- Claude's sandbox doesn't
have it, which is why this has to be a script you run rather than a file
Claude can hand you directly), to build the census_tracts.geojson the
dashboard's "Crashes by Census Tract" maps need.

What it does:
  1. Downloads 2023 TIGER/Line census tract boundaries for Florida (state
     FIPS 12) from the Census Bureau.
  2. Downloads total population per tract (ACS 5-year estimate, table
     B01003) from the Census API.
  3. Joins them on GEOID and writes census_tracts.geojson.

Install requirements:
    pip install geopandas requests

Get a free Census API key (takes ~1 minute, no approval wait):
    https://api.census.gov/data/key_signup.html

Usage:
    python build_census_tracts.py --api-key "PASTE_YOUR_OWN_40_CHAR_KEY_HERE"
    # Get a free key at https://api.census.gov/data/key_signup.html -- it's
    # a real, personal key emailed to you, NOT the string above.
    # Writes ./census_tracts.geojson in the current folder -- the dashboard
    # (app.py) picks it up automatically from the working directory, no
    # upload needed.
"""

import argparse
import io
import zipfile

import geopandas as gpd
import requests

FL_FIPS = "12"
TIGER_URL = f"https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_{FL_FIPS}_tract.zip"
ACS_URL = "https://api.census.gov/data/2022/acs/acs5"


def download_tract_boundaries():
    print(f"Downloading TIGER/Line tract boundaries from {TIGER_URL} ...")
    r = requests.get(TIGER_URL, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall("tl_tract_fl")
    gdf = gpd.read_file("tl_tract_fl")
    gdf = gdf.to_crs(4326)
    gdf = gdf.rename(columns={"GEOID": "GEOID"})[["GEOID", "geometry"]]
    print(f"  -> {len(gdf):,} Florida tracts")
    return gdf


def download_population(api_key):
    print("Downloading ACS 5-year total population (table B01003) ...")
    params = {"get": "B01003_001E", "for": "tract:*", "in": f"state:{FL_FIPS} county:*"}
    if api_key:
        params["key"] = api_key
    r = requests.get(ACS_URL, params=params, timeout=120)
    if r.status_code != 200:
        print(
            f"  -> Census API returned HTTP {r.status_code}, not JSON (likely a bad/unactivated "
            f"key). Skipping population -- writing tract boundaries only for now."
        )
        return None
    try:
        rows = r.json()
    except requests.exceptions.JSONDecodeError:
        print(
            "  -> Census API returned HTTP 200 but non-JSON content (invalid key). Skipping "
            "population -- writing tract boundaries only for now."
        )
        return None
    header, data = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    out = []
    for row in data:
        geoid = row[idx["state"]] + row[idx["county"]] + row[idx["tract"]]
        pop = row[idx["B01003_001E"]]
        out.append({"GEOID": geoid, "POPULATION": int(pop) if pop not in (None, "-666666666") else None})
    print(f"  -> {len(out):,} tract population rows")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=None, help="Census API key (recommended; get one free at api.census.gov)")
    ap.add_argument("--out", default="census_tracts.geojson")
    args = ap.parse_args()

    tracts = download_tract_boundaries()
    pop_rows = download_population(args.api_key)

    if pop_rows is None:
        merged = tracts.copy()
        merged["POPULATION"] = None
    else:
        import pandas as pd
        pop_df = pd.DataFrame(pop_rows)
        merged = tracts.merge(pop_df, on="GEOID", how="left")

    merged.to_file(args.out, driver="GeoJSON")
    if pop_rows is None:
        print(f"Wrote {args.out} -- {len(merged):,} tracts, no population data (see message above). "
              f"The total-crashes and e-bike-share tract maps will work now; the per-10k-residents "
              f"map needs population -- rerun with a working --api-key later to fill it in.")
    else:
        print(f"Wrote {args.out} -- {len(merged):,} tracts, "
              f"{merged['POPULATION'].notna().sum():,} with a population value.")
    print("The dashboard picks this up automatically from the working directory -- no upload needed.")