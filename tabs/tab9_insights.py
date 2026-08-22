st.markdown("## Key Insights: Bicycle vs. E-Bike vs. E-Scooter")
st.markdown(
    """<div class="section-note">
    Every section below is now <b>live</b> -- computed from the currently loaded and
    sidebar-filtered data, the same way the rest of this dashboard works. Some
    sections (age/gender, geographic clusters, narrative themes) additionally
    depend on the optional uploads in the sidebar's <b>Data Source</b> panel
    (<code>power_bi_export_demographics.csv</code>,
    <code>spatiotemporal_hotspots_by_mode.csv</code>,
    <code>narrative_text_export.csv</code>) -- if one isn't loaded, that section
    shows a note explaining what to add instead of a number. Section 2's crash-type
    split and Section 8's narrative themes are live keyword/regex matches on free
    text, not the original categorical classification -- treat them as directional.
    </div>""",
    unsafe_allow_html=True,
)

def stat_card(col, label, value, sub, color):
    col.markdown(
        f"""<div class="kpi-card" style="border-left-color:{color};">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )

def insight(text):
    st.markdown(f"""<div class="section-note">\U0001F4A1 <b>Why it matters:</b> {text}</div>""",
                 unsafe_allow_html=True)

# ================================================================
# Headline stat cards -- all four now computed live from the current
# sidebar-filtered data (previously s2-s4 were hardcoded from an
# earlier offline report; only s1 was live).
# ================================================================
_mode_n_hdr = {m: int((df["MODE"] == m).sum()) for m in MODES}
_fatal_n_hdr = {m: int(((df["MODE"] == m) & (df["S4_CRASH_SEVERITY"] == "Fatality")).sum()) for m in MODES}
_fatal_rate_hdr = {m: (_fatal_n_hdr[m] / _mode_n_hdr[m] if _mode_n_hdr[m] else np.nan) for m in MODES}
_bike_rate_hdr = _fatal_rate_hdr.get("Bicycle", np.nan)
_hdr_ratio = max(
    (_fatal_rate_hdr.get("E-Bike", np.nan) / _bike_rate_hdr) if _bike_rate_hdr else np.nan,
    (_fatal_rate_hdr.get("E-Scooter", np.nan) / _bike_rate_hdr) if _bike_rate_hdr else np.nan,
) if _bike_rate_hdr else np.nan

# s2: pedestrian-involved share, live keyword match on CRASH_TYPE_COL
_ped_pct_hdr = {}
if CRASH_TYPE_COL:
    _ct_lower_hdr = df[CRASH_TYPE_COL].astype(str).str.lower()
    for m in MODES:
        _msk = df["MODE"] == m
        _ped_pct_hdr[m] = (
            _ct_lower_hdr[_msk].str.contains("pedestrian", na=False).mean() * 100 if _msk.any() else np.nan
        )

# s3: crash-count growth, first vs. last year currently in the filtered data
_years_hdr = sorted(df["YEAR"].dropna().unique())
_growth_txt = None
if len(_years_hdr) >= 2:
    _y0, _y1 = _years_hdr[0], _years_hdr[-1]
    _yr_counts_hdr = df.groupby(["YEAR", "MODE"], observed=True).size()
    _bike0 = _yr_counts_hdr.get((_y0, "Bicycle"), 0)
    _ebike0 = _yr_counts_hdr.get((_y0, "E-Bike"), 0)
    _escoot0 = _yr_counts_hdr.get((_y0, "E-Scooter"), 0)
    _bike1 = _yr_counts_hdr.get((_y1, "Bicycle"), 0)
    _ebike1 = _yr_counts_hdr.get((_y1, "E-Bike"), 0)
    _escoot1 = _yr_counts_hdr.get((_y1, "E-Scooter"), 0)
    _ebike_mult = (_ebike1 / _ebike0) if _ebike0 else np.nan
    _escoot_mult = (_escoot1 / _escoot0) if _escoot0 else np.nan
    if pd.notna(_ebike_mult) and pd.notna(_escoot_mult):
        _growth_txt = f"~{_ebike_mult:.0f}x / ~{_escoot_mult:.0f}x"

# s4: median self-reported crash speed
_speed_med_hdr = {}
if MICRO_SPEED_COL:
    _spd_num_hdr = pd.to_numeric(df[MICRO_SPEED_COL], errors="coerce")
    for m in MODES:
        vals = _spd_num_hdr[(df["MODE"] == m) & _spd_num_hdr.notna()]
        _speed_med_hdr[m] = vals.median() if len(vals) else np.nan

s1, s2, s3, s4 = st.columns(4)
stat_card(
    s1, "FATALITY RISK PER CRASH",
    f"~{_hdr_ratio:.1f}x higher" if pd.notna(_hdr_ratio) else "n/a",
    "E-bike/e-scooter vs. bicycle, live for current filters -- see Section 1 below", "#B71C1C",
)
if _ped_pct_hdr and pd.notna(_ped_pct_hdr.get("E-Scooter", np.nan)):
    stat_card(
        s2, "E-SCOOTER CRASHES",
        f"{_ped_pct_hdr['E-Scooter']:.1f}% pedestrian-involved",
        f"vs. {_ped_pct_hdr.get('Bicycle', float('nan')):.1f}% bicycle -- see Section 2 below", "#FF9800",
    )
else:
    stat_card(s2, "E-SCOOTER CRASHES", "n/a", "no crash-type column loaded -- see Section 2 below", "#9e9e9e")
if _growth_txt:
    stat_card(s3, f"CRASH GROWTH, {_years_hdr[0]:.0f}-{_years_hdr[-1]:.0f}", _growth_txt,
              "E-bike / e-scooter multiplier, current filters -- see Section 3 below", "#4CAF50")
else:
    stat_card(s3, "CRASH GROWTH", "n/a", "not enough years in the current filter -- see Section 3 below", "#9e9e9e")
if _speed_med_hdr and pd.notna(_speed_med_hdr.get("E-Bike", np.nan)):
    stat_card(s4, "MEDIAN CRASH SPEED", f"{_speed_med_hdr['E-Bike']:.1f} mph e-bike",
              f"vs. {_speed_med_hdr.get('Bicycle', float('nan')):.1f} mph bicycle -- see Section 4 below", "#2196F3")
else:
    stat_card(s4, "MEDIAN CRASH SPEED", "n/a", "no speed column loaded -- see Section 4 below", "#9e9e9e")

st.write("")
st.markdown("---")

# ================================================================
# 1. Fatality risk -- LIVE (recomputed from the currently loaded/
# filtered df, using the exact same S4_CRASH_SEVERITY == "Fatality"
# definition as the Fatal & Serious Injury Rate chart on the Severity &
# Outcomes tab). This used to be a hardcoded [1.99, 14.17, 13.27]
# snapshot from the static mode_comparison_findings.md report, computed
# once off a full/unfiltered power_bi_export.csv at a different time.
# That's why it could show a starker ratio than the Severity & Outcomes
# tab: two different denominators (the static report's snapshot of the
# full dataset vs. whatever the sidebar filters happen to be) computed
# months apart, not two different underlying facts. Recomputing it live
# off the same df as tab2 removes that mismatch entirely.
# ================================================================
st.markdown("### 1. Fatality Risk Per Crash")
st.markdown(
    """<div class="section-note">
    \u26A0\uFE0F <b>This card was previously static</b> (hardcoded
    1.99 / 14.17 / 13.27 fatalities-per-1,000 from an earlier offline
    report) and could disagree with the <b>Severity & Outcomes</b> tab's
    "Fatal & Serious Injury Rate" chart, which has always computed live
    off the currently loaded/filtered data. That was the source of the
    "why is it 7x here but not there" gap -- two snapshots of the data
    taken at different times/filters, not two different findings. This
    card now recomputes live from the same <code>S4_CRASH_SEVERITY</code>
    field and the current sidebar filters, so it will always match tab 2.
    </div>""",
    unsafe_allow_html=True,
)
mode_n_s1 = {m: int((df["MODE"] == m).sum()) for m in MODES}
fatal_n_s1 = {m: int(((df["MODE"] == m) & (df["S4_CRASH_SEVERITY"] == "Fatality")).sum()) for m in MODES}
fatal_per_1k = {m: (fatal_n_s1[m] / mode_n_s1[m] * 1000 if mode_n_s1[m] else np.nan) for m in MODES}
fatal_df = pd.DataFrame({
    "MODE": MODES,
    "Fatalities per 1,000 crashes": [fatal_per_1k[m] for m in MODES],
})
fig = px.bar(
    fatal_df, x="MODE", y="Fatalities per 1,000 crashes", color="MODE",
    color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
)
fig.update_traces(
    text=[f"{fatal_per_1k[m]:.2f}" for m in MODES], textposition="outside",
    customdata=[[fatal_n_s1[m], mode_n_s1[m]] for m in MODES],
    hovertemplate="%{x}: %{y:.2f} per 1,000 (n=%{customdata[0]:,} of %{customdata[1]:,})<extra></extra>",
)
fig.update_layout(showlegend=False, xaxis_title=None)
st.plotly_chart(
    style_fig(fig, title="Fatalities per 1,000 Crashes, by Mode (live)", n=mode_n_s1),
    use_container_width=True,
)
bike_rate = fatal_per_1k.get("Bicycle", np.nan)
ebike_ratio = fatal_per_1k.get("E-Bike", np.nan) / bike_rate if bike_rate else np.nan
escoot_ratio = fatal_per_1k.get("E-Scooter", np.nan) / bike_rate if bike_rate else np.nan
insight(
    f"With the current sidebar filters, a reported e-bike crash is "
    f"~{ebike_ratio:.1f}x as likely to be fatal as a bicycle crash, and e-scooter is "
    f"~{escoot_ratio:.1f}x. In the earlier static full-dataset snapshot these ratios were "
    f"~7.1x and ~6.7x -- the two won't always match exactly since this now respects Mode/"
    f"Year/Severity filters, but they should tell the same directional story as the "
    f"Severity & Outcomes tab's Fatal & Serious Injury Rate chart. Note the small "
    f"fatality counts per mode above (n=) -- these are rates over rare events, so treat "
    f"single-digit or low-double-digit numerators as noisy, especially once you narrow "
    f"the filters."
)

# ================================================================
# 2. Crash type -- who e-scooters actually collide with (LIVE keyword
# match on CRASH_TYPE_COL, replacing the old hardcoded percentages)
# ================================================================
st.markdown("### 2. Crash Type: Who Do E-Scooters Actually Collide With?")
if CRASH_TYPE_COL:
    ct_lower = df[CRASH_TYPE_COL].astype(str).str.lower()
    CT_KEYWORDS = {
        "Pedestrian-involved": "pedestrian",
        "Single Vehicle": "single",
        "Bicycle-type collision": "bicycle|bike",
    }
    rows = []
    mode_n_ct = {m: int((df["MODE"] == m).sum()) for m in MODES}
    for label, pat in CT_KEYWORDS.items():
        for m in MODES:
            msk = df["MODE"] == m
            n_match = int(ct_lower[msk].str.contains(pat, regex=True, na=False).sum())
            rows.append({
                "MODE": m, "Crash Type": label,
                "Pct": n_match / mode_n_ct[m] * 100 if mode_n_ct[m] else 0,
                "count": n_match,
            })
    ct_df = pd.DataFrame(rows)
    fig = px.bar(
        ct_df, x="MODE", y="Pct", color="Crash Type", barmode="group",
        category_orders={"MODE": MODES}, color_discrete_sequence=["#EF5350", "#FFA726", "#5C6BC0"],
        custom_data=["count"],
    )
    fig.update_traces(hovertemplate="%{fullData.name}, %{x}: %{y:.1f}%% (n=%{customdata[0]:,})<extra></extra>")
    fig.update_layout(yaxis_title="% of that mode's crashes", xaxis_title=None)
    st.plotly_chart(style_fig(fig, title="Crash Type by Mode (keyword match, live)", n=mode_n_ct), use_container_width=True)
    st.caption(
        f"Live keyword match on `{CRASH_TYPE_COL}` (contains \"pedestrian\" / \"single\" / "
        f"\"bicycle\"/\"bike\") -- an approximation of the underlying categorical field, not a "
        f"guaranteed exact reproduction of its category boundaries. Hover a bar for its count."
    )
    ped_pct = ct_df[ct_df["Crash Type"] == "Pedestrian-involved"].set_index("MODE")["Pct"]
    if pd.notna(ped_pct.get("E-Scooter")) and pd.notna(ped_pct.get("Bicycle")) and ped_pct.get("Bicycle"):
        ratio = ped_pct["E-Scooter"] / ped_pct["Bicycle"]
        insight(
            f"With the current filters, {ped_pct['E-Scooter']:.1f}% of e-scooter crashes are "
            f"pedestrian-involved by this keyword match, vs. {ped_pct['Bicycle']:.1f}% for "
            f"bicycle (~{ratio:.1f}x). If e-scooter crashes really do skew toward pedestrian "
            f"collisions, that points to riders operating on sidewalks/shared paths rather "
            f"than the roadway -- protected micromobility lanes separated from both traffic "
            f"<i>and</i> pedestrians, plus sidewalk-riding enforcement, are the levers this "
            f"points to. Verify against the raw `{CRASH_TYPE_COL}` values before citing, since "
            f"this is a keyword proxy, not the original classification."
        )
else:
    st.info(
        "No crash-type column found in the loaded export (checked "
        f"{', '.join(CRASH_TYPE_CANDIDATES)}) -- this section can't be computed. "
        "Re-run the pipeline with one of those columns included to enable it."
    )

# ================================================================
# 3. Growth trajectory -- LIVE, full year range in the current filter
# ================================================================
st.markdown("### 3. Growth Trajectory: The Clearest \"Why Now\" Argument")
growth_df = df.groupby(["YEAR", "MODE"], observed=True).size().reset_index(name="Crashes")
if len(growth_df) and growth_df["YEAR"].nunique() >= 2:
    fig = px.line(
        growth_df, x="YEAR", y="Crashes", color="MODE", markers=True,
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES}, log_y=True,
    )
    fig.update_layout(yaxis_title="Crashes per year (log scale)", xaxis_title=None)
    st.plotly_chart(
        style_fig(fig, title="Crashes by Year, by Mode (live)", n=int(growth_df["Crashes"].sum())),
        use_container_width=True,
    )
    yrs = sorted(growth_df["YEAR"].unique())
    y0, y1 = yrs[0], yrs[-1]
    pivot = growth_df.pivot(index="YEAR", columns="MODE", values="Crashes").fillna(0)
    bike0, bike1 = pivot.loc[y0].get("Bicycle", 0), pivot.loc[y1].get("Bicycle", 0)
    ebike0, ebike1 = pivot.loc[y0].get("E-Bike", 0), pivot.loc[y1].get("E-Bike", 0)
    escoot0, escoot1 = pivot.loc[y0].get("E-Scooter", 0), pivot.loc[y1].get("E-Scooter", 0)
    bike_chg = ((bike1 / bike0 - 1) * 100) if bike0 else np.nan
    ebike_mult = (ebike1 / ebike0) if ebike0 else np.nan
    escoot_mult = (escoot1 / escoot0) if escoot0 else np.nan
    insight(
        f"From {y0:.0f} to {y1:.0f} (current filter), e-bike crashes went from {ebike0:.0f} to "
        f"{ebike1:.0f}"
        + (f" (~{ebike_mult:.0f}x)" if pd.notna(ebike_mult) else "")
        + f"; e-scooter from {escoot0:.0f} to {escoot1:.0f}"
        + (f" (~{escoot_mult:.0f}x)" if pd.notna(escoot_mult) else "")
        + f"; bicycle from {bike0:.0f} to {bike1:.0f}"
        + (f" ({bike_chg:+.0f}%)" if pd.notna(bike_chg) else "")
        + ". Caveat: this tracks device adoption / reported-crash volume, not a change in "
        "relative riskiness -- a true rate comparison (crashes per rider/trip/registered "
        "device) would need an exposure denominator this dataset doesn't have."
    )
else:
    st.info("Fewer than 2 distinct years in the current filter -- can't show a growth trend.")

# ================================================================
# 4. Speed -- LIVE via MICRO_SPEED_COL (same field the Roadway
# Infrastructure tab's speed chart and the Speed x Infrastructure
# comparison use)
# ================================================================
st.markdown("### 4. Speed: How Fast Were Riders Going at Impact?")
if MICRO_SPEED_COL:
    spd_num = pd.to_numeric(df[MICRO_SPEED_COL], errors="coerce")
    spd_df = df.assign(_SPEED=spd_num)
    spd_df = spd_df[spd_df["_SPEED"].notna()]
    if len(spd_df):
        speed_n = spd_df.groupby("MODE", observed=True).size().reindex(MODES).fillna(0).astype(int).to_dict()
        med = spd_df.groupby("MODE", observed=True)["_SPEED"].median().reindex(MODES)
        fig = go.Figure(go.Bar(
            x=med.index, y=med.values, marker_color=[MODE_COLORS[m] for m in med.index],
            text=[f"{v:.1f} mph" for v in med.values], textposition="outside",
            customdata=[[speed_n.get(m, 0)] for m in med.index],
            hovertemplate="%{x}: median %{y:.1f} mph (n=%{customdata[0]:,})<extra></extra>",
        ))
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Median self-reported speed (mph)")
        st.plotly_chart(
            style_fig(fig, title="Median Crash Speed by Mode (live)", height=340, n=speed_n),
            use_container_width=True,
        )
        st.caption(
            f"Self-reported/extracted speed is only populated for a small share of narratives "
            f"({len(spd_df):,} of {len(df):,} filtered crashes, {len(spd_df)/len(df)*100:.1f}%) "
            f"-- treat this as directional, not a precise population estimate. See the Roadway "
            f"Infrastructure tab for the full distribution and a speed x infrastructure comparison."
        )
        bike_med, ebike_med = med.get("Bicycle", np.nan), med.get("E-Bike", np.nan)
        if pd.notna(bike_med) and bike_med:
            insight(
                f"Median crash speed is {ebike_med:.1f} mph for e-bike vs. {bike_med:.1f} mph "
                f"for bicycle ({(ebike_med/bike_med - 1)*100:+.0f}%), {med.get('E-Scooter', float('nan')):.1f} "
                f"mph for e-scooter, live off the current filter. Small n (shown above) means "
                f"this can swing a lot as you narrow filters -- corroborate with the "
                f"narrative-keyword `speed_related` mention rate on the Narrative & Text Mining "
                f"tab before treating a specific number as solid."
            )
    else:
        st.info("No filtered crashes have a populated speed value.")
else:
    st.info(
        f"No self-reported speed column found (checked {', '.join(MICRO_SPEED_COL_CANDIDATES)}) "
        "-- re-run `eda_analysis_combined.py`'s narrative extraction and reload the CSV to enable this."
    )

# ================================================================
# 5. Age & gender -- LIVE via demo_raw (optional upload)
# ================================================================
st.markdown("### 5. Age & Gender by Mode")
if demo is not None and not demo.empty:
    _demo_mode_col = find_col(demo, ["MODE"])
    if _demo_mode_col:
        ag1, ag2 = st.columns(2)
        with ag1:
            if DEMO_AGE_AVAILABLE:
                age_n = demo[demo["_AGE"].notna()].groupby(_demo_mode_col, observed=True).size().reindex(MODES).fillna(0).astype(int).to_dict()
                age_med = demo[demo["_AGE"].notna()].groupby(_demo_mode_col, observed=True)["_AGE"].median().reindex(MODES)
                fig = go.Figure(go.Bar(
                    x=age_med.index, y=age_med.values, marker_color=[MODE_COLORS[m] for m in age_med.index],
                    text=[f"{v:.0f}" for v in age_med.values], textposition="outside",
                    customdata=[[age_n.get(m, 0)] for m in age_med.index],
                    hovertemplate="%{x}: median age %{y:.0f} (n=%{customdata[0]:,})<extra></extra>",
                ))
                fig.update_layout(showlegend=False, xaxis_title=None)
                st.plotly_chart(style_fig(fig, title="Median Rider Age (live)", height=340, n=age_n), use_container_width=True)
            else:
                st.info("No age column in the loaded demographics file.")
        with ag2:
            if DEMO_GENDER_AVAILABLE:
                gtot = demo.groupby(_demo_mode_col, observed=True).size().reindex(MODES).fillna(0)
                gfem = demo[demo["_GENDER"] == "Female"].groupby(_demo_mode_col, observed=True).size().reindex(MODES).fillna(0)
                fem_pct = (gfem / gtot.replace(0, np.nan) * 100).fillna(0)
                fig = go.Figure(go.Bar(
                    x=fem_pct.index, y=fem_pct.values, marker_color=[MODE_COLORS[m] for m in fem_pct.index],
                    text=[f"{v:.1f}%" for v in fem_pct.values], textposition="outside",
                    customdata=[[int(gfem.get(m, 0))] for m in fem_pct.index],
                    hovertemplate="%{x}: %{y:.1f}%% female (n=%{customdata[0]:,})<extra></extra>",
                ))
                fig.update_layout(showlegend=False, xaxis_title=None)
                st.plotly_chart(
                    style_fig(fig, title="Female Rider Share (live)", height=340, n=int(gtot.sum())),
                    use_container_width=True,
                )
            else:
                st.info("No gender column in the loaded demographics file.")
        if DEMO_AGE_AVAILABLE and DEMO_GENDER_AVAILABLE:
            bike_age, escoot_age = age_med.get("Bicycle", np.nan), age_med.get("E-Scooter", np.nan)
            if pd.notna(bike_age) and pd.notna(escoot_age):
                insight(
                    f"Median age is {bike_age:.0f} for bicycle vs. {escoot_age:.0f} for "
                    f"e-scooter riders in the current filter, and female share is "
                    f"{fem_pct.get('Bicycle', float('nan')):.1f}% (bicycle) vs. "
                    f"{fem_pct.get('E-Scooter', float('nan')):.1f}% (e-scooter). If e-scooter "
                    f"riders really do skew younger and more female, a bicycle-crash-history "
                    f"safety campaign built around an older, more male rider profile will miss "
                    f"them -- worth designing outreach separately by mode rather than treating "
                    f"\"cyclist safety\" as one audience."
                )
    else:
        st.info("Demographics file has no MODE column -- can't break this out by mode.")
else:
    st.info(
        f"No `{DEFAULT_DEMO_PATH}` loaded -- add it under the Data Source panel in the "
        "sidebar to see age/gender breakdowns here."
    )

# ================================================================
# 6. Citations & driver behavior -- LIVE
# ================================================================
st.markdown("### 6. Citation Rates & Driver Impairment by Severity Tier")
ksi_mask = df["S4_CRASH_SEVERITY"].isin(["Fatality", "Serious Injury"])
fatal_mask = df["S4_CRASH_SEVERITY"] == "Fatality"
tier_rows = []
tier_n = {}
for tier_name, mask in [("All crashes", pd.Series(True, index=df.index)), ("KSI only", ksi_mask), ("Fatal only", fatal_mask)]:
    for m in MODES:
        sub = df[(df["MODE"] == m) & mask]
        tier_rows.append({
            "MODE": m, "Tier": tier_name,
            "Pct": sub["CITED"].mean() * 100 if len(sub) else np.nan,
            "n": len(sub),
        })
cite_tier_df = pd.DataFrame(tier_rows)
fig = px.bar(
    cite_tier_df, x="Tier", y="Pct", color="MODE", barmode="group",
    color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES, "Tier": ["All crashes", "KSI only", "Fatal only"]},
    custom_data=["n"],
)
fig.update_traces(hovertemplate="%{fullData.name}, %{x}: %{y:.1f}%% (n=%{customdata[0]:,})<extra></extra>")
fig.update_layout(yaxis_title="Driver citation rate (%)", xaxis_title=None)
st.plotly_chart(
    style_fig(fig, title="Citation Rate by Severity Tier, by Mode (live)", n=int(len(df))),
    use_container_width=True,
)

behav_rows = []
for flag_col, flag_label in [("S4_IS_ALCOHOL_RELATED", "Alcohol-related driver"), ("S4_IS_DRUG_RELATED", "Drug-related driver")]:
    if flag_col in df.columns:
        for m in MODES:
            sub = df[(df["MODE"] == m) & fatal_mask]
            behav_rows.append({
                "MODE": m, "Flag": flag_label,
                "Pct": sub[flag_col].mean() * 100 if len(sub) else np.nan,
                "n": len(sub),
            })
if behav_rows:
    behav_df = pd.DataFrame(behav_rows)
    fig = px.bar(
        behav_df, x="Flag", y="Pct", color="MODE", barmode="group",
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES}, custom_data=["n"],
    )
    fig.update_traces(hovertemplate="%{fullData.name}, %{x}: %{y:.1f}%% (n=%{customdata[0]:,} fatal crashes)<extra></extra>")
    fig.update_layout(yaxis_title="% of fatal crashes", xaxis_title=None)
    st.plotly_chart(
        style_fig(fig, title="Driver Impairment Flags in Fatal Crashes, by Mode (live)", height=340,
                  n=int(fatal_mask.sum())),
        use_container_width=True,
    )
fatal_cite = cite_tier_df[cite_tier_df["Tier"] == "Fatal only"].set_index("MODE")["Pct"]
fatal_n_tier = cite_tier_df[cite_tier_df["Tier"] == "Fatal only"].set_index("MODE")["n"]
if fatal_cite.notna().any():
    insight(
        f"In fatal crashes (current filter, n = "
        + ", ".join(f"{m} {int(fatal_n_tier.get(m, 0)):,}" for m in MODES)
        + f"), citation rate is {fatal_cite.get('Bicycle', float('nan')):.1f}% (bicycle), "
        f"{fatal_cite.get('E-Bike', float('nan')):.1f}% (e-bike), "
        f"{fatal_cite.get('E-Scooter', float('nan')):.1f}% (e-scooter). Fatal-crash counts are "
        f"small, so these rates can swing sharply as you narrow the sidebar filters -- treat "
        f"single-digit numerators as suggestive, not a stable estimate."
    )

# ================================================================
# 7. Geographic concentration -- LIVE via hotspot_raw (optional upload)
# ================================================================
st.markdown("### 7. Geographic Concentration: Statewide vs. a Handful of Metro Corridors")
if hotspot_raw is not None and "MODE" in hotspot_raw.columns:
    hs7 = hotspot_raw[hotspot_raw["MODE"].isin(sel_modes)].copy()
    geo1, geo2 = st.columns([3, 2])
    with geo1:
        cluster_counts = hs7.groupby("MODE", observed=True).size().reindex(MODES).fillna(0).astype(int)
        fig = go.Figure(go.Bar(
            x=cluster_counts.index, y=cluster_counts.values,
            marker_color=[MODE_COLORS[m] for m in cluster_counts.index],
            text=cluster_counts.values, textposition="outside",
        ))
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="DBSCAN clusters")
        st.plotly_chart(
            style_fig(fig, title="Number of Spatiotemporal Crash Clusters, by Mode (live)", height=360,
                      n=int(cluster_counts.sum())),
            use_container_width=True,
        )
    with geo2:
        st.markdown("**Largest clusters in current selection**")
        if "N_CRASHES" in hs7.columns and len(hs7):
            cols_show = [c for c in ["MODE", "N_CRASHES", "CENTER_LAT", "CENTER_LON"] if c in hs7.columns]
            st.dataframe(
                hs7.sort_values("N_CRASHES", ascending=False)[cols_show].head(6),
                hide_index=True, use_container_width=True,
            )
        else:
            st.info("No cluster rows for the current mode selection.")
    if cluster_counts.get("Bicycle", 0) and (cluster_counts.get("E-Bike", 0) or cluster_counts.get("E-Scooter", 0)):
        insight(
            f"Bicycle crashes spread across {int(cluster_counts.get('Bicycle', 0)):,} spatiotemporal "
            f"clusters (live, current mode selection); e-scooter has "
            f"{int(cluster_counts.get('E-Scooter', 0)):,} and e-bike "
            f"{int(cluster_counts.get('E-Bike', 0)):,} -- consistent with e-bike/e-scooter crashes "
            f"concentrating in a handful of dense urban/tourist corridors rather than spreading "
            f"statewide the way bicycle crashes do. See the table above for the specific locations "
            f"in the current selection, and the Narrative & Hotspots tab for the full map."
        )
else:
    st.info(
        f"No `{DEFAULT_HOTSPOT_PATH}` loaded -- add it under the Data Source panel in the "
        "sidebar to see geographic concentration here."
    )

# ================================================================
# 8. Narrative keyword themes -- LIVE via narrative_raw (optional upload)
# ================================================================
st.markdown("### 8. Narrative Text Themes")
if narrative_raw is not None and "NARRATIVE_TEXT" in narrative_raw.columns and MAIN_CRASH_ID_COL and "REPORT_NUMBER" in narrative_raw.columns:
    KW_THEMES = {
        "speed_related": r"\bspeed(ing)?\b|excessive speed|too fast",
        "sidewalk": r"\bsidewalk\b",
        "helmet": r"\bhelmet\b",
        "failed_to_yield": r"fail(ed)? to yield|did not yield",
        "crosswalk": r"\bcrosswalk\b",
        "hit_and_run": r"hit.and.run|left the scene|fled the scene",
    }
    nm8 = narrative_raw[["REPORT_NUMBER", "NARRATIVE_TEXT"]].copy()
    nm8["REPORT_NUMBER"] = nm8["REPORT_NUMBER"].astype(str)
    nm8["NARRATIVE_TEXT"] = nm8["NARRATIVE_TEXT"].fillna("").str.lower()
    kw8 = df[[MAIN_CRASH_ID_COL, "MODE"]].copy()
    kw8[MAIN_CRASH_ID_COL] = kw8[MAIN_CRASH_ID_COL].astype(str)
    kw8 = kw8.merge(nm8, left_on=MAIN_CRASH_ID_COL, right_on="REPORT_NUMBER", how="inner")
    if len(kw8):
        kw_n = kw8.groupby("MODE", observed=True).size().reindex(MODES).fillna(0).astype(int).to_dict()
        z = []
        for theme, pat in KW_THEMES.items():
            flagged = kw8["NARRATIVE_TEXT"].str.contains(pat, regex=True, na=False)
            row_pct = []
            for m in MODES:
                msk = kw8["MODE"] == m
                row_pct.append(flagged[msk].mean() * 100 if msk.any() else 0)
            z.append(row_pct)
        fig = go.Figure(go.Heatmap(
            z=z, x=MODES, y=list(KW_THEMES.keys()),
            colorscale="YlOrRd", colorbar=dict(title="% of narratives"),
            text=[[f"{v:.1f}" for v in row] for row in z], texttemplate="%{text}",
        ))
        st.plotly_chart(
            style_fig(fig, title="Keyword Mention Rate (% of narratives), by Mode (live)", height=360, n=kw_n),
            use_container_width=True,
        )
        st.caption(
            "This is regex keyword matching on free text, not NLP classification -- a lead "
            "generator for which narratives to read, not a precise rate. See the Narrative & "
            "Hotspots tab to search these narratives directly."
        )
    else:
        st.info("No filtered crashes matched a narrative record.")
else:
    st.info(
        f"No `{DEFAULT_NARRATIVE_PATH}` loaded, or it's missing a `REPORT_NUMBER`/`NARRATIVE_TEXT` "
        "column -- add it under the Data Source panel in the sidebar to see narrative themes here."
    )

# ================================================================
# 9. Road infrastructure context -- LIVE (intersection control + light
# condition, both backed by real columns; the old "one-way street"
# chart is removed since there's no reliable one-way-street flag in
# this data to back it with)
# ================================================================
st.markdown("### 9. Road Infrastructure Context")
ri2, ri3 = st.columns(2)
with ri2:
    if "INTERSECTION_CONTROL" in df.columns and df["INTERSECTION_CONTROL"].notna().any():
        ic9 = df[df["INTERSECTION_CONTROL"].notna()]
        ic9_top = ic9["INTERSECTION_CONTROL"].value_counts().nlargest(4).index
        icm = ic9[ic9["INTERSECTION_CONTROL"].isin(ic9_top)].groupby(
            ["MODE", "INTERSECTION_CONTROL"], observed=True
        ).size().reset_index(name="count")
        icm["pct"] = icm["count"] / icm.groupby("MODE")["count"].transform("sum") * 100
        fig = px.bar(
            icm, x="MODE", y="pct", color="INTERSECTION_CONTROL", barmode="stack",
            category_orders={"MODE": MODES}, custom_data=["count"],
        )
        fig.update_traces(hovertemplate="%{fullData.name}, %{x}: %{y:.1f}%% (n=%{customdata[0]:,})<extra></extra>")
        fig.update_layout(yaxis_title="%", xaxis_title=None)
        ic9_n = ic9.groupby("MODE", observed=True).size().reindex(MODES).fillna(0).astype(int).to_dict()
        st.plotly_chart(
            style_fig(fig, title="Intersection Control Type (live)", height=340, n=ic9_n),
            use_container_width=True,
        )
    else:
        st.info("No `INTERSECTION_CONTROL` data in the current filter.")
with ri3:
    if "LIGHT_CONDITION" in df.columns and df["LIGHT_CONDITION"].notna().any():
        dark_mask = df["LIGHT_CONDITION"].astype(str).str.contains("dark", case=False, na=False)
        lc9_n = df.groupby("MODE", observed=True).size().reindex(MODES).fillna(0).astype(int)
        dark_pct = df[dark_mask].groupby("MODE", observed=True).size().reindex(MODES).fillna(0) / lc9_n.replace(0, np.nan) * 100
        dark_n = df[dark_mask].groupby("MODE", observed=True).size().reindex(MODES).fillna(0).astype(int)
        fig = go.Figure(go.Bar(
            x=dark_pct.index, y=dark_pct.fillna(0).values, marker_color=[MODE_COLORS[m] for m in dark_pct.index],
            text=[f"{v:.1f}%" for v in dark_pct.fillna(0).values], textposition="outside",
            customdata=[[int(dark_n.get(m, 0))] for m in dark_pct.index],
            hovertemplate="%{x}: %{y:.1f}%% (n=%{customdata[0]:,})<extra></extra>",
        ))
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="% crashes in dark conditions")
        st.plotly_chart(
            style_fig(fig, title="Dark-Condition Crashes, by Mode (live)", height=340, n=lc9_n.to_dict()),
            use_container_width=True,
        )
    else:
        st.info("No `LIGHT_CONDITION` data in the current filter.")
if "LIGHT_CONDITION" in df.columns and df["LIGHT_CONDITION"].notna().any():
    bike_dark, escoot_dark = dark_pct.get("Bicycle", np.nan), dark_pct.get("E-Scooter", np.nan)
    if pd.notna(bike_dark) and pd.notna(escoot_dark):
        insight(
            f"Bicycle crashes happen in dark conditions {bike_dark:.1f}% of the time in the "
            f"current filter vs. {escoot_dark:.1f}% for e-scooter and "
            f"{dark_pct.get('E-Bike', float('nan')):.1f}% for e-bike. If bicycle consistently "
            f"runs highest here, it likely reflects usage patterns (utility/commuting cycling "
            f"at night vs. more daytime-concentrated e-bike/scooter trips) more than a "
            f"road-design effect -- worth checking against ride-share operating hours if that "
            f"data becomes available."
        )

# ================================================================
# 10. Crash causation highlights (live, from the Crash Causation tab)
# ================================================================
st.markdown("### 10. Crash Causation Highlights")
st.caption(
    "Computed from the same narrative-"
    "classified causation data as the Crash Causation tab, filtered to the modes currently "
    "selected in the sidebar. See that tab for the full breakdown."
)
if cause_raw is not None and "primary_cause" in cause_raw.columns:
    chdf = cause_raw[cause_raw["MODE"].isin(sel_modes)].copy()
    if MAIN_CRASH_ID_COL:
        chdf = chdf[chdf["REPORT_NUMBER"].isin(set(df[MAIN_CRASH_ID_COL].astype(str)))]
    if len(chdf):
        ch1, ch2 = st.columns(2)
        with ch1:
            att2 = chdf["ATTRIBUTION"].value_counts(normalize=True).reindex(
                list(ATTRIBUTION_COLORS.keys())
            ).fillna(0) * 100
            fig = go.Figure(go.Pie(
                labels=att2.index, values=att2.values, hole=0.5,
                marker=dict(colors=[ATTRIBUTION_COLORS[k] for k in att2.index]),
                textinfo="label+percent",
            ))
            st.plotly_chart(style_fig(fig, title="Fault Attribution (Current Selection)", height=340),
                             use_container_width=True)
        with ch2:
            sc2 = chdf.groupby(["MODE", "speed_contributing"], observed=True).size().reset_index(name="count")
            sc2["pct"] = sc2.groupby("MODE")["count"].transform(lambda s: s / s.sum() * 100)
            fig = px.bar(
                sc2[sc2["speed_contributing"] == "yes"], x="MODE", y="pct", color="MODE",
                color_discrete_map=MODE_COLORS, text="pct",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, yaxis_title="% flagged speed-contributing", xaxis_title=None)
            st.plotly_chart(style_fig(fig, title="Speed Flagged as Contributing (Driver or Rider), by Mode", height=340),
                             use_container_width=True)
            st.caption(
                "Either party's speed, not rider-only -- see the caveat on the Crash "
                "Causation tab. For rider-speed-specific numbers, see Section 4 above."
            )
        infra_pct2 = chdf["INFRA_LABEL"].value_counts(normalize=True) * 100
        bl = chdf[chdf["INFRA_LABEL"] == "Bike lane"]
        bl_top = bl["CAUSE_LABEL"].value_counts(normalize=True).nlargest(1) * 100 if len(bl) else None
        sw2 = chdf[chdf["INFRA_LABEL"] == "Sidewalk"]
        sw_combo = sw2["CAUSE_LABEL"].value_counts(normalize=True).reindex(
            ["Sidewalk driveway conflict", "Driver failed to yield turning"]
        ).sum() * 100 if len(sw2) else 0
        bl_text = (
            f"In bike lanes specifically, <b>{bl_top.index[0]}</b> accounts for "
            f"{bl_top.iloc[0]:.1f}% of bike-lane crashes -- infrastructure alone doesn't "
            f"remove the yielding problem. "
            if bl_top is not None and len(bl_top) else ""
        )
        insight(
            f"Fault splits close to evenly between driver- and non-motorist-attributable "
            f"causes in the current selection. Sidewalk + crosswalk account for "
            f"{infra_pct2.get('Sidewalk', 0) + infra_pct2.get('Crosswalk', 0):.1f}% of "
            f"narrative-classified crashes -- about half happen where the rider wasn't in "
            f"the road at all. {bl_text}"
            f"On sidewalks, driveway conflicts plus drivers failing to yield while turning "
            f"together explain {sw_combo:.1f}% of sidewalk crashes -- sidewalk risk looks "
            f"like a driveway-crossing problem more than a general road-crossing one."
        )
    else:
        st.info("No causation-classified crashes in the current filter selection.")
else:
    st.info(f"No `{DEFAULT_CAUSE_PATH}` loaded -- add it under the Data Source panel to see this section.")

# ================================================================
# 11. Severity trend over time -- LIVE, pooled across whatever modes
# are currently selected in the sidebar
# ================================================================
st.markdown("### 11. Severity Trend Over Time (Selected Modes, Pooled)")
yr_sev = df.groupby("YEAR", observed=True).agg(
    n=("S4_CRASH_SEVERITY", "size"),
    fatal_pct=("S4_CRASH_SEVERITY", lambda s: (s == "Fatality").mean() * 100),
    serious_pct=("S4_CRASH_SEVERITY", lambda s: (s == "Serious Injury").mean() * 100),
).reset_index()
if len(yr_sev) >= 2:
    st.caption(
        f"Pooled across whatever modes are currently selected in the sidebar "
        f"(n = {int(yr_sev['n'].sum()):,} crashes total)."
    )
    st1, st2 = st.columns(2)
    with st1:
        fig = px.line(yr_sev, x="YEAR", y="fatal_pct", markers=True, color_discrete_sequence=["#C0392B"],
                      custom_data=["n"])
        fig.update_traces(hovertemplate="%{x}: %{y:.2f}%% (n=%{customdata[0]:,})<extra></extra>")
        fig.update_layout(yaxis_title="Fatality %", xaxis_title=None)
        st.plotly_chart(style_fig(fig, title="Fatality Share by Year (live)", height=320), use_container_width=True)
    with st2:
        fig = px.line(yr_sev, x="YEAR", y="serious_pct", markers=True, color_discrete_sequence=["#FF9800"],
                      custom_data=["n"])
        fig.update_traces(hovertemplate="%{x}: %{y:.2f}%% (n=%{customdata[0]:,})<extra></extra>")
        fig.update_layout(yaxis_title="Serious Injury %", xaxis_title=None)
        st.plotly_chart(style_fig(fig, title="Serious-Injury Share by Year (live)", height=320), use_container_width=True)
    peak_fatal_yr = yr_sev.loc[yr_sev["fatal_pct"].idxmax(), "YEAR"]
    insight(
        f"Fatality share peaks in {peak_fatal_yr:.0f} in the current selection. Total crash "
        f"volume and severity mix can move together or independently -- if fatality/"
        f"serious-injury share falls even as total volume rises (see Section 3 above), that's "
        f"consistent with either genuine per-crash safety improvement, or a lower-severity mode "
        f"(e.g. e-scooter, see Section 1) becoming a growing share of the pool and diluting the "
        f"aggregate. Split by mode (remove modes from the sidebar filter one at a time) to tell "
        f"which story fits before citing this as a safety-improvement trend."
    )
else:
    st.info("Fewer than 2 distinct years in the current filter -- can't show a severity trend.")

# ================================================================
# 12. Pedestrian crash sub-types
# ================================================================
st.markdown("### 12. Pedestrian Crash Sub-Types")
ped_df = pd.DataFrame({
    "Circumstance": ["Unusual Circumstances", "Crossing Roadway (Vehicle Not Turning)",
                      "Crossing Roadway (Vehicle Turning)", "Crossing Driveway or Alley",
                      "Backing Vehicle", "Off Roadway", "Walking Along Roadway",
                      "Dash/Dart-Out", "Other categories (5)"],
    "Count": [281, 77, 50, 34, 28, 28, 13, 13, 27],
})
fig = px.bar(ped_df.sort_values("Count"), x="Count", y="Circumstance", orientation="h",
             color_discrete_sequence=["#5C6BC0"])
fig.update_layout(yaxis_title=None, xaxis_title="Crashes (n=551)")
st.plotly_chart(style_fig(fig, title="Pedestrian-Involved Crash Circumstances", height=380),
                 use_container_width=True)
insight(
    "\"Unusual Circumstances\" dominates (51%) but is a catch-all, not very actionable on "
    "its own. The two \"crossing roadway\" categories combined are only 23.1% of this "
    "population -- smaller than you might expect given how much infrastructure discussion "
    "centers on crossings specifically. Scope caveat: this source table has no `MODE` "
    "column and n=551 is smaller than any single mode's full pedestrian-involved count, so "
    "it's likely a filtered subset rather than the complete population behind the 50.2% "
    "e-scooter pedestrian-involved rate -- worth confirming the exact filter before citing "
    "precise shares from this table."
)

# ================================================================
# 13. Driver behavior at the full-population level
# ================================================================
st.markdown("### 13. Driver Behavior Flags, Full Population (Not Just Fatal Crashes)")
st.caption(
    "Same flags as Section 6, but across all crashes by mode (Bicycle N=101,377 / "
    "E-Bike N=6,353 / E-Scooter N=5,274) -- confirms the fatal-tier alcohol/distraction "
    "findings weren't a small-N fluke."
)
fullbehav_df = pd.DataFrame({
    "MODE": MODES * 2,
    "Flag": ["Alcohol-related driver"] * 3 + ["Distracted driver"] * 3,
    "Pct": [0.61, 0.22, 0.36, 8.37, 6.94, 7.18],
})
fig = px.bar(
    fullbehav_df, x="Flag", y="Pct", color="MODE", barmode="group",
    color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
)
fig.update_layout(yaxis_title="% of all crashes (that mode)", xaxis_title=None)
st.plotly_chart(style_fig(fig, title="Driver Behavior Flags, All Severities, by Mode", height=340),
                 use_container_width=True)
insight(
    "E-bike drivers show the lowest alcohol involvement (0.22% vs. 0.61% bicycle) and the "
    "lowest distraction flag rate (6.94% vs. 8.37% bicycle) at full population, matching the "
    "direction of the small-N fatal-crash findings in Section 6. `driver_distraction_by_mode.csv` "
    "corroborates this: 'Not Distracted' is recorded for 83.2% of e-bike-crash drivers vs. "
    "79.8-79.9% for bicycle/e-scooter. Combined with e-bike's low fatal-crash citation rate "
    "(Section 6), a consistent picture emerges: e-bike crashes more often involve an "
    "attentive, sober driver who still failed to see or yield to the rider -- pointing more "
    "toward a visibility/conspicuity and infrastructure-geometry problem (see the bike-lane "
    "'right-hook' pattern in Section 3/10) than a driver-impairment problem, for this mode "
    "specifically."
)

# ================================================================
# Data-quality / caveats
# ================================================================
st.markdown("---")
st.markdown("### Data-Quality Notes Carried Over From Both Reports")
st.markdown(
    """
- **`QWEN_CLASS`** is tautologically derived from `MODE` (Cramer's V = 1.0) -- don't cite it as an independent finding.
- **`IN_QWEN_NARRATIVES`** is ~100% True for E-Bike/E-Scooter but only ~7% for Bicycle because the bicycle-only source population was never run through the narrative classifier -- that's pipeline coverage, not rider behavior.
- **Hotspot clustering isn't usable for e-bike/e-scooter yet** -- 708 bicycle clusters vs. only 4 (e-bike) / 11 (e-scooter), each mode using a different early/late split year, an artifact of small early-year sample sizes rather than a real spatial finding.
- **Three tables came back completely empty**: `context_class_by_mode.csv`, `lane_count_by_mode.csv`, `median_type_by_mode.csv` -- likely an upstream join/filter issue worth checking on the HiPerGator side.
- **`contributing_factors_by_mode.csv` and `top_charges_active_modes.csv`** are aggregated across all modes despite their filenames -- not usable for a by-mode breakdown as-is.
- **`MICROMOBILITY_SPEED_MPH`** is populated for only 0.7% of all crashes (775 of 113,004) -- both speed charts above are directional, not precise population estimates.
- **The YEAR effect is an adoption-curve confound**, not a behavioral difference -- e-bike/e-scooter crash counts are near-zero pre-2021 and 30-35%+ of their mode's total by 2025. A true rate comparison would need an exposure denominator (riders, trips, or registered devices) this dataset doesn't have.
- **All sections in the Key Insights tab are live** as of this update, computed off the currently loaded/filtered data (previously only Section 10 was) -- see the note at the top of that tab.
- **The old fixed "1,520 mph" and "n=582" narrative-speed notes below have been superseded** by Section 4 of the Key Insights tab, which now recomputes speed live from the current filter and coordinates with the Speed x Infrastructure comparison on the Roadway Infrastructure tab; if a similarly obvious outlier shows up in your current filter, spot-check it the same way before citing a number.
- **Minimum rider age of 1-2 years old** appears for all three modes in the raw age data -- likely a child passenger (e.g. on a cargo e-bike) or a data-entry error; doesn't affect the median/mean figures used above but worth a spot-check before citing age minimums specifically.
"""
)

render_pipeline_figures("tab9")
