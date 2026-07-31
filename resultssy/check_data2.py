import pandas as pd

df = pd.read_csv("power_bi_export.csv")

print("SHAPE:", df.shape)
print("\nCOLUMNS:\n", list(df.columns))
print("\nDTYPES:\n", df.dtypes)
print("\nNULL % PER COLUMN:\n", (df.isna().mean()*100).round(1).sort_values(ascending=False))
print("\nHEAD:\n", df.head(3).to_string())

# Categorical-ish columns -- print whatever exists
cat_cols = ["MODE","DOW","LIGHT_CONDITION","WEATHER_CONDITION","S4_CRASH_SEVERITY",
            "LOC_TYPE","DAY_NIGHT","MEDIAN_TYPE","CONTEXT_CLASS",
            "INTERSECTION_CONTROL","FARS_LANDUSE","COUNTY_NAME"]
for c in cat_cols:
    if c in df.columns:
        print(f"\n--- {c} (top 10) ---")
        print(df[c].value_counts(dropna=False).head(10))

# Boolean/flag columns
bool_like = [c for c in df.columns if c.startswith("S4_IS_") or c in ("mv_involved","CITED")]
print("\nBOOLEAN-ISH COLUMNS FOUND:", bool_like)
for c in bool_like:
    print(c, "->", df[c].dtype, df[c].value_counts(dropna=False).to_dict())

# Numeric ranges
for c in ["YEAR","HOUR","MONTH","AVG_AADT","SHOULDER_WIDTH","NUM_THRU_LANES"]:
    if c in df.columns:
        print(f"{c} range:", df[c].min(), "-", df[c].max())
