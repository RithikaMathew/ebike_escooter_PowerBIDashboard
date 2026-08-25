c1, c2 = st.columns(2)
with c1:
    dow_mode = df.groupby(["DOW", "MODE"], observed=True).size().reset_index(name="count")
    fig = px.bar(
        dow_mode, x="DOW", y="count", color="MODE",
        color_discrete_map=MODE_COLORS,
        category_orders={"DOW": DOW_ORDER, "MODE": MODES},
    )
    fig.update_traces(hovertemplate="%{fullData.name}, %{x}: %{y:,} crashes<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis_title="Crashes")
    st.plotly_chart(style_fig(fig, title="Crashes by Day of Week", n=total), width="stretch")

with c2:
    dn_mode = df.groupby(["MODE", "DAY_NIGHT"], observed=True).size().reset_index(name="count")
    dn_mode["pct"] = dn_mode["count"] / dn_mode.groupby("MODE")["count"].transform("sum") * 100
    fig = px.bar(
        dn_mode, x="MODE", y="pct", color="DAY_NIGHT",
        color_discrete_map={"Day": "#FDD835", "Night": "#283593"},
        category_orders={"MODE": MODES}, custom_data=["count"],
    )
    fig.update_traces(texttemplate="%{y:.0f}%", textposition="inside", textfont=dict(size=10, color="#12172b"),
                       hovertemplate="%{fullData.name}: %{y:.1f}%% (n=%{customdata[0]:,})<extra>%{x}</extra>")
    fig.update_layout(yaxis_title="% of crashes", xaxis_title=None, barmode="stack")
    dn_mode_n = dn_mode.groupby("MODE")["count"].sum().reindex(MODES).fillna(0).astype(int).to_dict()
    st.plotly_chart(
        style_fig(fig, title="Day vs. Night Share by Mode", n=dn_mode_n), width="stretch"
    )

c3, c4 = st.columns(2)
with c3:
    loc_mode = df.groupby(["MODE", "LOC_TYPE"], observed=True).size().reset_index(name="count")
    loc_mode["pct"] = loc_mode["count"] / loc_mode.groupby("MODE")["count"].transform("sum") * 100
    fig = px.bar(
        loc_mode, x="MODE", y="pct", color="LOC_TYPE",
        color_discrete_map={"Intersection": "#5C6BC0", "Segment": "#26A69A"},
        category_orders={"MODE": MODES}, custom_data=["count"],
    )
    fig.update_traces(texttemplate="%{y:.0f}%", textposition="inside", textfont=dict(size=10, color="white"),
                       hovertemplate="%{fullData.name}: %{y:.1f}%% (n=%{customdata[0]:,})<extra>%{x}</extra>")
    fig.update_layout(yaxis_title="% of crashes", xaxis_title=None, barmode="stack")
    loc_mode_n = loc_mode.groupby("MODE")["count"].sum().reindex(MODES).fillna(0).astype(int).to_dict()
    st.plotly_chart(
        style_fig(fig, title="Intersection vs. Segment by Mode", n=loc_mode_n), width="stretch"
    )

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
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES}, custom_data=["count"],
    )
    fig.update_traces(hovertemplate="%{y}, %{fullData.name}: %{x:.1f}%% (n=%{customdata[0]:,})<extra></extra>")
    fig.update_layout(yaxis_title=None, xaxis_title="% of that mode's crashes",
                       yaxis={"categoryorder": "total ascending"})
    light_n = {m: int(mode_totals.get(m, 0)) for m in MODES}
    st.plotly_chart(
        style_fig(fig, title="Light Conditions (% Within Mode)", n=light_n, height=420),
        width="stretch",
    )

c5, c6 = st.columns(2)
with c5:
    wthr_top = df["WEATHER_CONDITION"].value_counts().nlargest(5).index
    wthr_df = df[df["WEATHER_CONDITION"].isin(wthr_top)]
    wm = wthr_df.groupby(["WEATHER_CONDITION", "MODE"], observed=True).size().reset_index(name="count")
    wm["pct"] = wm.apply(lambda r: r["count"] / mode_totals.get(r["MODE"], 1) * 100, axis=1)
    fig = px.bar(
        wm, y="WEATHER_CONDITION", x="pct", color="MODE", orientation="h", barmode="group",
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES}, custom_data=["count"],
    )
    fig.update_traces(hovertemplate="%{y}, %{fullData.name}: %{x:.1f}%% (n=%{customdata[0]:,})<extra></extra>")
    fig.update_layout(yaxis_title=None, xaxis_title="% of that mode's crashes",
                       yaxis={"categoryorder": "total ascending"})
    wthr_n = {m: int(mode_totals.get(m, 0)) for m in MODES}
    st.plotly_chart(
        style_fig(fig, title="Weather Conditions (% Within Mode)", n=wthr_n, height=400),
        width="stretch",
    )

with c6:
    top_counties = df["COUNTY_NAME"].value_counts().nlargest(15).index
    co_df = df[df["COUNTY_NAME"].isin(top_counties)]
    cm = co_df.groupby(["COUNTY_NAME", "MODE"], observed=True).size().reset_index(name="count")
    fig = px.bar(
        cm, y="COUNTY_NAME", x="count", color="MODE", orientation="h",
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
    )
    fig.update_traces(hovertemplate="%{fullData.name}, %{y}: %{x:,} crashes<extra></extra>")
    fig.update_layout(yaxis_title=None, xaxis_title="Crashes",
                       yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(
        style_fig(fig, title="Top 15 Counties", height=430, n=int(co_df.shape[0])),
        width="stretch",
    )

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
        fig = px.scatter_map(
            geo, lat=LAT_COL, lon=LON_COL, color="MODE",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
            opacity=0.55, zoom=5.4, height=560,
        )
        fig = style_fig(
            fig, height=560, n=len(geo),
            title=f"Crash Locations by Mode ({len(geo):,} of {total:,} filtered crashes geocoded)",
        )
        fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=56, b=0))
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(fig, width="stretch")
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

st.markdown("---")
st.markdown("#### Crashes by Census Tract")
with st.expander("Glossary & methodology -- what a 'tract' is, how these maps and numbers are built", expanded=True):
    st.markdown(
        """
        **In short:**
        - **Census tract** = neighborhood-sized area (typically 1,200–8,000 residents; Florida has ~5,000)
        - **Map 1** = Where are the most crashes?
        - **Map 2** = Where are crashes high compared with population?
        - **Map 3** = What type of crash makes up the crashes in that area?
        - **DBSCAN hotspots** = Where are crashes repeatedly happening close together?
        - **Spatiotemporal growth** = Which of those concentrated areas appear to be getting worse over time?

        **How a crash gets assigned to a tract.** Each geocoded crash's point (lat/lon) is
        **spatially joined** to the Florida census tract polygon it falls inside of
        (point-in-polygon join against 2023 TIGER/Line tract boundaries). Every crash lands
        in exactly one tract, or none if its coordinates fall outside every tract polygon
        (bad/edge-of-state geocodes -- see the match-rate caption below). Crashes are
        aggregated to `GEOID` and joined to ACS total-population estimates.

        This is **statewide Florida**, not zoomed to any one metro -- the dark clusters you
        see are simply where population (and therefore ridership and crashes) concentrates:
        greater Jacksonville, Tampa-St. Pete, Orlando, and the whole
        Miami-Fort Lauderdale-West Palm Beach corridor down the southeast coast.

        **The three maps, and what each one actually answers:**
        - **Map 1 (raw counts)** -- "where do the most crashes happen." Use the mode selector
          to view Bicycle, E-Bike, E-Scooter, or all modes combined. Dominated by population
          density: a big city tract will out-count a small town even if the small town is more
          dangerous per rider.
        - **Map 2 (per 100,000 residents, colored by percentile rank)** -- "where is a
          resident most likely to be involved in a crash," controlling for how many people
          live there. Formula: `(crashes ÷ population) × 100,000`. That scaled rate is for
          comparing tracts — it is **not** the actual crash count (e.g. 28 crashes in a tract
          of 2,069 people → rate ≈ 1,353 per 100k means "if this ratio held in a city of
          100,000, we'd expect ~1,353 crashes," not that 1,353 crashes happened). Tracts under
          100 residents are dropped; the map is colored by **percentile rank** so tiny-population
          outliers don't blow out the scale. Hover for the actual rate.
        - **Map 3 (mode share %)** -- "of the micromobility crashes in this tract, what
          fraction were this mode." Always read alongside Map 1's count.

        **DBSCAN vs growth over time** (below Map 1):
        - **DBSCAN** asks: *"Are there specific places where crashes keep happening close
          together?"* — not merely "which neighborhood has the most crashes?" It groups nearby
          crash points without using tract boundaries (`eda_analysis_combined.py` §09d).
        - **Spatiotemporal growth** takes those same spatial clusters and splits each cluster's
          crashes into an **early** vs **late** period (at the median year `SPLIT_YEAR` in the
          hotspot file: years ≤ split = early, years > split = late) to ask which concentrations
          are getting worse over time. Primary label = Poisson rate-ratio test (p < 0.05);
          the 1.5× early→late heuristic is exploratory only.
        """
    )

if not GEOPANDAS_AVAILABLE:
    st.warning(
        "`geopandas` isn't installed in this environment, so the census-tract maps can't "
        "render. Install it (`pip install geopandas shapely`) and re-run the dashboard."
    )
elif tracts_raw is None:
    st.info(
        "No census tract boundary file loaded yet. Upload a `census_tracts.geojson` "
        "(GEOID + population + geometry, see the **Optional: census tract boundaries** "
        "panel in the sidebar) to enable these three maps."
    )
elif not (LAT_COL and LON_COL):
    st.info("No latitude/longitude columns in the loaded export, so points can't be joined to tracts.")
elif "GEOID" not in tracts_raw.columns:
    st.warning("The uploaded tract file has no `GEOID` column -- can't aggregate to it.")
else:
    geo_pts = df[[LAT_COL, LON_COL, "MODE"]].copy()
    geo_pts[LAT_COL] = pd.to_numeric(geo_pts[LAT_COL], errors="coerce")
    geo_pts[LON_COL] = pd.to_numeric(geo_pts[LON_COL], errors="coerce")
    geo_pts = geo_pts[geo_pts[LAT_COL].between(24, 31) & geo_pts[LON_COL].between(-88, -79)]

    if len(geo_pts) == 0:
        st.info("No crashes with valid Florida coordinates in the current filter selection.")
    else:
        pts_gdf = gpd.GeoDataFrame(
            geo_pts,
            geometry=gpd.points_from_xy(geo_pts[LON_COL], geo_pts[LAT_COL]),
            crs=4326,
        )
        joined = gpd.sjoin(pts_gdf, tracts_raw[["GEOID", "geometry"]], how="left", predicate="within")
        n_matched = joined["GEOID"].notna().sum()
        n_unmatched = len(joined) - n_matched
        st.caption(
            f"**{n_matched:,}** of **{len(joined):,}** geocoded, filtered crashes "
            f"(n = {len(joined):,}) matched to a tract; **{n_unmatched:,}** fell outside "
            f"every tract polygon (bad/edge-of-state geocodes) and are excluded from the maps below."
        )
        joined = joined.dropna(subset=["GEOID"])

        if len(joined) == 0:
            st.info("No crashes matched a census tract in the current filter selection.")
        else:
            # FIPS county codes -> names, decoded from GEOID for readable tables (2-digit
            # state + 3-digit county + 6-digit tract). Static Census reference data.
            FL_COUNTY_FIPS = {
                "001": "Alachua", "003": "Baker", "005": "Bay", "007": "Bradford",
                "009": "Brevard", "011": "Broward", "013": "Calhoun", "015": "Charlotte",
                "017": "Citrus", "019": "Clay", "021": "Collier", "023": "Columbia",
                "027": "DeSoto", "029": "Dixie", "031": "Duval", "033": "Escambia",
                "035": "Flagler", "037": "Franklin", "039": "Gadsden", "041": "Gilchrist",
                "043": "Glades", "045": "Gulf", "047": "Hamilton", "049": "Hardee",
                "051": "Hendry", "053": "Hernando", "055": "Highlands", "057": "Hillsborough",
                "059": "Holmes", "061": "Indian River", "063": "Jackson", "065": "Jefferson",
                "067": "Lafayette", "069": "Lake", "071": "Lee", "073": "Leon",
                "075": "Levy", "077": "Liberty", "079": "Madison", "081": "Manatee",
                "083": "Marion", "085": "Martin", "086": "Miami-Dade", "087": "Monroe",
                "089": "Nassau", "091": "Okaloosa", "093": "Okeechobee", "095": "Orange",
                "097": "Osceola", "099": "Palm Beach", "101": "Pasco", "103": "Pinellas",
                "105": "Polk", "107": "Putnam", "109": "St. Johns", "111": "St. Lucie",
                "113": "Santa Rosa", "115": "Sarasota", "117": "Seminole", "119": "Sumter",
                "121": "Suwannee", "123": "Taylor", "125": "Union", "127": "Volusia",
                "129": "Wakulla", "131": "Walton", "133": "Washington",
            }

            tract_counts = (
                joined.groupby(["GEOID", "MODE"], observed=True).size()
                .unstack(fill_value=0).reindex(columns=MODES, fill_value=0)
            )
            tract_counts["TOTAL_MICRO"] = tract_counts[MODES].sum(axis=1)
            tract_geo = tracts_raw.merge(tract_counts.reset_index(), on="GEOID", how="left")
            for c in list(MODES) + ["TOTAL_MICRO"]:
                tract_geo[c] = tract_geo[c].fillna(0)

            # Mode picker: Map 1 can be All or a single mode; maps 2–4 use a single mode.
            map1_options = ["All modes"] + list(MODES)
            map1_mode = st.radio(
                "Mode for Map 1 (raw counts)",
                map1_options, index=0,
                horizontal=True, key="tract_map1_mode",
            )
            map1_col = "TOTAL_MICRO" if map1_mode == "All modes" else map1_mode

            rate_mode = st.radio(
                "Mode for maps 2–4 (per-capita rate, mode share, spatial stats, EB)",
                MODES, index=MODES.index("E-Bike") if "E-Bike" in MODES else 0,
                horizontal=True, key="tract_rate_mode",
            )

            # Tracts with tiny population produce wildly unstable rates -- 1 crash in a
            # 40-person tract reads as a catastrophic "per capita" rate that isn't
            # meaningful, and it blows out the color scale so every normal tract gets
            # crushed to near-white by comparison. Exclude them (shown as blank/gray)
            # rather than letting one or two extreme tracts define the whole map.
            MIN_TRACT_POP = 100

            has_pop = tract_pop_col in tract_geo.columns
            if has_pop:
                tract_geo[tract_pop_col] = pd.to_numeric(tract_geo[tract_pop_col], errors="coerce")
                stable_pop = tract_geo[tract_pop_col] >= MIN_TRACT_POP
                n_small_pop = int((~stable_pop & tract_geo[tract_pop_col].notna()).sum())
                # Per 100,000 residents -- the standard convention for traffic-safety
                # rates (matches how NHTSA/CDC report crash and injury rates), rather
                # than the arbitrary 10,000 used before.
                tract_geo["RATE_PER_100K_POP"] = np.where(
                    stable_pop,
                    tract_geo[rate_mode] / tract_geo[tract_pop_col] * 100_000, np.nan,
                )
            tract_geo["MODE_SHARE_OF_MICRO"] = np.where(
                tract_geo["TOTAL_MICRO"] > 0,
                tract_geo[rate_mode] / tract_geo["TOTAL_MICRO"] * 100, np.nan,
            )

            def choropleth(gdf_col, title, colorbar_title, subtitle_n, colorscale="YlOrRd",
                            hotspot_df=None, zmin=None, zmax=None, customdata_col=None, hover_label=None):
                trace_kwargs = dict(
                    geojson=tract_geo.geometry.__geo_interface__,
                    locations=tract_geo.index, z=tract_geo[gdf_col],
                    zmin=zmin, zmax=zmax,
                    colorscale=colorscale, marker_opacity=0.7, marker_line_width=0.3,
                    colorbar_title=colorbar_title,
                )
                if customdata_col:
                    trace_kwargs["customdata"] = tract_geo[customdata_col]
                    trace_kwargs["hovertemplate"] = (
                        f"Percentile: %{{z:.0f}}<br>{hover_label}: %{{customdata:.1f}}<extra></extra>"
                    )
                fig = go.Figure(go.Choroplethmap(**trace_kwargs))
                if hotspot_df is not None and len(hotspot_df) and {"CENTER_LAT", "CENTER_LON"}.issubset(hotspot_df.columns):
                    hd = hotspot_df.copy()
                    hd["N_CRASHES"] = pd.to_numeric(hd.get("N_CRASHES"), errors="coerce").fillna(0)
                    hd = hd.dropna(subset=["CENTER_LAT", "CENTER_LON"])
                    max_n = hd["N_CRASHES"].max()
                    if len(hd) and pd.notna(max_n) and max_n > 0:
                        sizes = (hd["N_CRASHES"] / max_n * 22 + 6)
                    else:
                        sizes = 10  # flat fallback size if there's no usable N_CRASHES to scale by
                    fig.add_trace(go.Scattermap(
                        lat=hd["CENTER_LAT"], lon=hd["CENTER_LON"],
                        mode="markers",
                        marker=dict(size=sizes, color="#00BCD4", opacity=0.85),
                        text=hd.get("CLUSTER_ID"),
                        hovertemplate="Cluster %{text}: %{customdata} crashes<extra></extra>",
                        customdata=hd["N_CRASHES"],
                        name="Hotspot cluster center",
                        showlegend=True,
                    ))
                fig.update_layout(
                    map_style="open-street-map", map_zoom=5.4,
                    map_center={"lat": 27.8, "lon": -81.7},
                    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                )
                fig = style_fig(fig, height=520, title=title, n=subtitle_n)
                fig.update_layout(margin=dict(l=0, r=0, t=70, b=0))
                return fig

            show_hotspots = False
            hs_for_map = None
            if hotspot_raw is not None and "MODE" in hotspot_raw.columns:
                show_hotspots = st.checkbox(
                    "Overlay DBSCAN hotspot cluster centers on Map 1 (sized by crashes in cluster)",
                    value=True, key="tract_map_hotspot_overlay",
                )
                if show_hotspots:
                    if map1_mode == "All modes":
                        hs_for_map = hotspot_raw[hotspot_raw["MODE"].isin(sel_modes)].copy()
                    else:
                        hs_for_map = hotspot_raw[hotspot_raw["MODE"] == map1_mode].copy()

            m1, m2 = st.columns(2)
            with m1:
                map1_title = (
                    "1. Micromobility Crashes per Tract (all modes)"
                    if map1_mode == "All modes"
                    else f"1. {map1_mode} Crashes per Tract"
                )
                st.plotly_chart(
                    choropleth(
                        map1_col,
                        map1_title,
                        "Crashes", n_matched,
                        hotspot_df=hs_for_map,
                    ),
                    width="stretch",
                )
                st.caption(
                    (f"Raw {map1_mode} crash count per tract." if map1_mode != "All modes"
                     else "Raw crash count per tract, all three modes combined.")
                    + (" Teal bubbles = DBSCAN hotspot cluster centers "
                       "(places where crashes keep happening close together), sized by "
                       "crashes in that cluster." if show_hotspots else "")
                )
                if show_hotspots and hs_for_map is not None and len(hs_for_map) and "N_CRASHES" in hs_for_map.columns:
                    with st.expander("Top 10 hotspot clusters (recurring crash locations, not tied to tract boundaries)", expanded=True):
                        st.markdown(
                            "These are **DBSCAN** clusters of actual crash locations — asking "
                            "*“Are there specific places where crashes keep happening close together?”* "
                            "rather than *“Which neighborhood has the most crashes?”* "
                            "They are not diluted by population or forced into a census tract shape."
                        )
                        cluster_mode_options = ["All modes shown on Map 1"] + [m for m in MODES if m in hs_for_map["MODE"].unique()]
                        cluster_mode_filter = st.selectbox(
                            "Filter this table by mode", cluster_mode_options, index=0, key="tract_hotspot_table_mode",
                        )
                        top_clusters = hs_for_map.copy()
                        if cluster_mode_filter != "All modes shown on Map 1":
                            top_clusters = top_clusters[top_clusters["MODE"] == cluster_mode_filter]
                        top_clusters["N_CRASHES"] = pd.to_numeric(top_clusters["N_CRASHES"], errors="coerce").fillna(0)
                        top_clusters = top_clusters.nlargest(10, "N_CRASHES").copy()
                        # County from joining cluster center to tract polygons
                        if GEOPANDAS_AVAILABLE and len(top_clusters) and {"CENTER_LAT", "CENTER_LON"}.issubset(top_clusters.columns):
                            try:
                                centers = gpd.GeoDataFrame(
                                    top_clusters,
                                    geometry=gpd.points_from_xy(
                                        top_clusters["CENTER_LON"], top_clusters["CENTER_LAT"]
                                    ),
                                    crs=4326,
                                )
                                joined_c = gpd.sjoin(
                                    centers, tracts_raw[["GEOID", "geometry"]],
                                    how="left", predicate="within",
                                )
                                top_clusters["County"] = (
                                    joined_c["GEOID"].astype(str).str.slice(2, 5).map(FL_COUNTY_FIPS).values
                                )
                            except Exception:
                                top_clusters["County"] = None
                        cluster_cols = [c for c in ["CLUSTER_ID", "MODE", "County", "N_CRASHES", "CENTER_LAT", "CENTER_LON"]
                                        if c in top_clusters.columns]
                        if len(top_clusters):
                            st.dataframe(top_clusters[cluster_cols], width="stretch", hide_index=True)
                        else:
                            st.info(f"No {cluster_mode_filter} clusters in the current filter selection.")

            with m2:
                if has_pop:
                    # Raw per-capita rate is dominated by a handful of tiny-denominator
                    # tracts no matter how the scale is capped (1 crash on a 150-person
                    # tract vs. 5 crashes on an 8,000-person tract -- the first "wins" on
                    # raw rate despite being the less real problem). Coloring by percentile
                    # rank instead guarantees the color spread is visible across the whole
                    # state; the actual rate is still shown on hover.
                    tract_geo["RATE_PCTL"] = tract_geo["RATE_PER_100K_POP"].rank(pct=True) * 100
                    st.plotly_chart(
                        choropleth(
                            "RATE_PCTL",
                            f"2. {rate_mode} Crashes per 100,000 Residents (percentile rank)",
                            "Percentile", n_matched, colorscale="Reds",
                            zmin=0, zmax=100,
                            customdata_col="RATE_PER_100K_POP", hover_label=f"{rate_mode}/100k pop",
                        ),
                        width="stretch",
                    )
                    st.caption(
                        f"**Crash rate = (Number of {rate_mode} crashes ÷ Population) × 100,000.** "
                        f"That number is a *scaled rate for comparing tracts*, not the actual crash "
                        f"count. Example: 28 crashes ÷ 2,069 people × 100,000 ≈ 1,353.3 means "
                        f"“if this same ratio existed in a population of 100,000, we'd expect about "
                        f"1,353 crashes” — there were still only 28 actual crashes. "
                        f"Tracts colored by their **rank** among all tracts (hover for the actual "
                        f"rate). Tracts under {MIN_TRACT_POP} residents ({n_small_pop:,} of them) "
                        f"are excluded and shown blank."
                    )

                    risk_tbl = tract_geo[tract_geo["RATE_PER_100K_POP"].notna()].copy()
                    if len(risk_tbl):
                        risk_tbl = risk_tbl.nlargest(15, "RATE_PER_100K_POP")
                        # GEOID = 2-digit state + 3-digit county + 6-digit tract FIPS.
                        # Decode the county so the table reads as places, not just ID numbers.
                        risk_tbl["County"] = risk_tbl["GEOID"].astype(str).str.slice(2, 5).map(FL_COUNTY_FIPS)
                        cols = ["GEOID", "County", tract_pop_col, rate_mode, "TOTAL_MICRO", "RATE_PER_100K_POP"]
                        rename = {
                            tract_pop_col: "Population", rate_mode: f"{rate_mode} crashes",
                            "TOTAL_MICRO": "All micromobility crashes",
                            "RATE_PER_100K_POP": "Rate / 100k pop",
                        }
                        with st.expander(f"Top 15 highest-rate tracts for {rate_mode} (population \u2265 {MIN_TRACT_POP})", expanded=False):
                            st.dataframe(
                                risk_tbl[cols].rename(columns=rename).round({"Rate / 100k pop": 1}),
                                width="stretch", hide_index=True,
                            )
                            if (risk_tbl["County"] == "Monroe").any():
                                st.caption(
                                    "Monroe County (the Florida Keys) tracts showing up here are worth a "
                                    "second look: Census population only counts year-round residents, not "
                                    "the large tourist population that actually rides there, so a high rate "
                                    "in the Keys may partly reflect an undercounted denominator rather than "
                                    "locals being unusually at risk."
                                )
                else:
                    st.info(
                        f"Population column '{tract_pop_col}' not found in the tract file -- "
                        f"can't compute crashes-per-capita. Check the column name in the sidebar."
                    )

            st.plotly_chart(
                choropleth(
                    "MODE_SHARE_OF_MICRO",
                    f"3. {rate_mode} Share of All Micromobility Crashes per Tract (%)",
                    f"% {rate_mode}", n_matched, colorscale="Purples",
                ),
                width="stretch",
            )
            st.caption(
                f"{rate_mode} crashes \u00f7 (bicycle + e-bike + e-scooter crashes) in that tract, "
                f"as a %. Only meaningful where TOTAL_MICRO is non-trivial -- a tract with 1 total "
                f"crash that happens to be a {rate_mode.lower()} crash shows 100% here, so read "
                f"this alongside Map 1's raw count, not in isolation."
            )

            # --- Spatiotemporal growth explorer (moved from Narrative tab) ---
            st.markdown("---")
            st.markdown("#### Spatiotemporal growth — which concentrated areas are getting worse?")
            st.caption(
                "**DBSCAN** (overlay / Top 10 above) answers *where are crashes geographically "
                "concentrated?* This section adds the time dimension: among those clusters, "
                "*which appear to be getting worse over time?* "
                "Early vs late = years ≤ vs > the median `SPLIT_YEAR` stored in the hotspot file."
            )
            if hotspot_raw is not None and "MODE" in hotspot_raw.columns:
                hs = hotspot_raw[hotspot_raw["MODE"].isin(sel_modes)].copy()
                has_periods = {"N_EARLY_PERIOD", "N_LATE_PERIOD"}.issubset(hs.columns)
                stats_test_available = False
                if has_periods:
                    with st.expander("Classification: statistical test is primary; 1.5× is exploratory", expanded=False):
                        split_years = (
                            hs["SPLIT_YEAR"].dropna().unique().tolist()
                            if "SPLIT_YEAR" in hs.columns else []
                        )
                        split_note = (
                            f" Early/late split year(s) in this file: "
                            f"{', '.join(str(int(y)) for y in sorted(split_years))}."
                            if split_years else ""
                        )
                        st.markdown(
                            f"""
                            - **Primary:** two-sample Poisson rate-ratio test (exact binomial comparing
                              early vs late counts, equal period lengths assumed). Significant growth =
                              p < 0.05.
                            - **Secondary / exploratory:** late-period count > 1.5× early-period count
                              (with ≥5 late crashes). Small counts can clear 1.5× by chance (2→4), so
                              treat the heuristic as a screen, not the main label.
                            - **Early vs late:** crashes in each cluster are split at the median year
                              (`SPLIT_YEAR`). Years ≤ split = early; years > split = late.{split_note}
                            """
                        )
                    try:
                        from scipy import stats as _stats

                        def _rate_ratio_test(n1, n2):
                            n_total = n1 + n2
                            if n_total == 0 or pd.isna(n1) or pd.isna(n2):
                                return np.nan, np.nan
                            pval = _stats.binomtest(int(n2), int(n_total), 0.5, alternative="two-sided").pvalue
                            rr = (n2 / n1) if n1 > 0 else np.inf
                            return rr, pval

                        _rr = hs.apply(
                            lambda r: _rate_ratio_test(r["N_EARLY_PERIOD"], r["N_LATE_PERIOD"]), axis=1
                        )
                        hs["RATE_RATIO"] = [x[0] for x in _rr]
                        hs["GROWTH_PVAL"] = [x[1] for x in _rr]
                        hs["SIG_GROWTH"] = hs["GROWTH_PVAL"] < 0.05
                        stats_test_available = True
                    except ImportError:
                        st.warning(
                            "`scipy` isn't installed, so the Poisson rate-ratio test can't run. "
                            "Install it (`pip install scipy`) and re-run the dashboard."
                        )

                filter_options = ["All clusters"]
                if stats_test_available:
                    filter_options.append(
                        "Statistically significant growth (Poisson rate-ratio test, p<0.05)"
                    )
                if has_periods:
                    filter_options.append("Emerging (exploratory heuristic: >1.5x growth)")

                default_idx = 1 if stats_test_available else 0
                cluster_filter = st.selectbox(
                    "Filter clusters", filter_options, index=default_idx, key="hotspot_cluster_filter",
                )
                if cluster_filter.startswith("Statistically significant") and stats_test_available:
                    hs = hs[hs["SIG_GROWTH"] == True]  # noqa: E712
                elif "1.5x" in cluster_filter and "EMERGING" in hs.columns:
                    hs = hs[hs["EMERGING"] == True]  # noqa: E712

                st.caption(
                    f"**{len(hs):,}** clusters match the Mode filter in the sidebar "
                    f"(hotspot table isn't affected by Year/Severity filters — it's "
                    f"precomputed per mode over the full time range)."
                )
                if len(hs) and {"CENTER_LAT", "CENTER_LON"}.issubset(hs.columns):
                    hover_cols = [
                        "CLUSTER_ID", "N_CRASHES", "N_EARLY_PERIOD", "N_LATE_PERIOD", "GROWTH_RATIO",
                    ]
                    if stats_test_available:
                        hover_cols += ["RATE_RATIO", "GROWTH_PVAL"]
                    fig = px.scatter_map(
                        hs, lat="CENTER_LAT", lon="CENTER_LON", color="MODE",
                        size="N_CRASHES", size_max=28,
                        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
                        hover_data=[c for c in hover_cols if c in hs.columns],
                        zoom=5.4, height=560,
                    )
                    fig = style_fig(
                        fig, height=560, n=len(hs),
                        title="Cluster Centers — growth explorer (bubble size = crashes in cluster)",
                    )
                    fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=70, b=0))
                    st.plotly_chart(fig, width="stretch")

                    sort_col = (
                        "GROWTH_PVAL" if stats_test_available
                        else ("GROWTH_RATIO" if "GROWTH_RATIO" in hs.columns else hs.columns[0])
                    )
                    ascending = sort_col == "GROWTH_PVAL"
                    display_hs = hs.sort_values(sort_col, ascending=ascending).copy()
                    if stats_test_available:
                        display_hs = display_hs.rename(columns={
                            "RATE_RATIO": "Rate ratio (late/early)", "GROWTH_PVAL": "p-value",
                        })
                        display_hs = display_hs.round({"Rate ratio (late/early)": 2, "p-value": 4})
                    if GEOPANDAS_AVAILABLE and {"CENTER_LAT", "CENTER_LON"}.issubset(display_hs.columns):
                        try:
                            _c = gpd.GeoDataFrame(
                                display_hs,
                                geometry=gpd.points_from_xy(
                                    display_hs["CENTER_LON"], display_hs["CENTER_LAT"]
                                ),
                                crs=4326,
                            )
                            _j = gpd.sjoin(
                                _c, tracts_raw[["GEOID", "geometry"]], how="left", predicate="within",
                            )
                            display_hs["County"] = (
                                _j["GEOID"].astype(str).str.slice(2, 5).map(FL_COUNTY_FIPS).values
                            )
                        except Exception:
                            pass
                    st.dataframe(display_hs, width="stretch", hide_index=True)
                else:
                    st.info("No clusters match the current Mode selection.")
                st.caption(
                    "Exploratory spatial DBSCAN + early/late period split "
                    "(`eda_analysis_combined.py` §09d) — not a validated hotspot pipeline. "
                    "Primary: Poisson rate-ratio p<0.05. Exploratory: late > 1.5× early with ≥5 late crashes."
                )
            else:
                st.markdown(
                    f"""<div class="section-note">
                    No <code>{DEFAULT_HOTSPOT_PATH}</code> loaded — add it under <b>Data Source</b>
                    in the sidebar to enable the growth explorer.
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("#### Statistically Significant Hot/Cold Spots (Getis-Ord Gi*)")
            with st.expander("Why this map is different from Maps 1-3 and the DBSCAN clusters", expanded=False):
                st.markdown(
                    """
                    Maps 1-3 and the DBSCAN clusters above all show where **counts happen to be
                    higher** -- none of them test whether that's more clustering than you'd expect
                    from chance, given how crashes are scattered across the state. **Getis-Ord
                    Gi\\*** is the standard spatial-statistics answer to that question: for every
                    tract, it compares that tract's count and its immediate neighbors' counts
                    against what a random spatial arrangement would produce, and returns a
                    z-score / p-value for whether it's a genuine, **statistically significant**
                    hot spot (surrounded by other high tracts, more than chance would predict) or
                    cold spot (the reverse). A tract can have a high raw count and *not* be
                    flagged here if its neighbors are all low -- this map is about spatial
                    *clustering*, not individual tract magnitude.

                    Neighbors are defined by **Queen contiguity** (tracts sharing any border or
                    corner). Significance uses **999 conditional permutations** per tract (the
                    standard PySAL/esda approach) rather than a theoretical p-value, since crash
                    counts are skewed and small-sample theoretical approximations can be
                    unreliable. **Local Moran's I** (folded into the table below) adds one more
                    distinction: a "High-High" tract is part of an area-wide cluster, while a
                    "High-Low" tract is a single dangerous tract surrounded by otherwise-normal
                    ones -- worth investigating as a specific location (an intersection, a
                    corridor) rather than a broader area problem.
                    """
                )

            try:
                from esda.getisord import G_Local  # noqa: F401
                SPATIAL_STATS_AVAILABLE = True
            except ImportError:
                SPATIAL_STATS_AVAILABLE = False
                st.info(
                    "Spatial statistics need the `libpysal` and `esda` packages -- "
                    "`pip install libpysal esda` and re-run to enable this map."
                )

            if SPATIAL_STATS_AVAILABLE:
                stats = spatial_cluster_stats_for_mode(tract_geo, rate_mode)
                if stats:
                    tract_geo["GI_BUCKET"] = stats["gi_buckets"]
                    tract_geo["MORAN_Q"] = stats["moran_q"]
                    tract_geo["GI_Z"] = stats["gi_z"]
                    tract_geo["GI_P"] = stats["gi_p"]
                    n_hot = stats["n_hot_spots"]
                    n_cold = stats["n_cold_spots"]
                    n_hl_outlier = stats["n_high_low_outliers"]
                else:
                    tract_geo["GI_BUCKET"] = "Not significant"
                    tract_geo["MORAN_Q"] = "Not significant"
                    n_hot = n_cold = n_hl_outlier = 0

                bucket_order = [
                    "Hot spot (99% confidence)", "Hot spot (95% confidence)", "Hot spot (90% confidence)",
                    "Not significant",
                    "Cold spot (90% confidence)", "Cold spot (95% confidence)", "Cold spot (99% confidence)",
                ]
                bucket_colors = {
                    "Hot spot (99% confidence)": "#67000d", "Hot spot (95% confidence)": "#cb181d",
                    "Hot spot (90% confidence)": "#fc9272", "Not significant": "#f0f0f0",
                    "Cold spot (90% confidence)": "#9ecae1", "Cold spot (95% confidence)": "#3182bd",
                    "Cold spot (99% confidence)": "#08306b",
                }
                n_buckets = len(bucket_order)
                code_map = {b: i for i, b in enumerate(bucket_order)}
                tract_geo["GI_BUCKET_CODE"] = tract_geo["GI_BUCKET"].map(code_map).astype(float) + 0.5
                stepped_colorscale = []
                for i, b in enumerate(bucket_order):
                    stepped_colorscale.append([i / n_buckets, bucket_colors[b]])
                    stepped_colorscale.append([(i + 1) / n_buckets, bucket_colors[b]])

                fig = go.Figure(go.Choroplethmap(
                    geojson=tract_geo.geometry.__geo_interface__,
                    locations=tract_geo.index, z=tract_geo["GI_BUCKET_CODE"],
                    customdata=tract_geo["GI_BUCKET"],
                    colorscale=stepped_colorscale, zmin=0, zmax=n_buckets,
                    marker_opacity=0.75, marker_line_width=0.3,
                    hovertemplate="%{customdata}<extra></extra>", showscale=False,
                ))
                fig.update_layout(
                    map_style="open-street-map", map_zoom=5.4,
                    map_center={"lat": 27.8, "lon": -81.7},
                )
                fig = style_fig(
                    fig, height=560, n=n_matched,
                    title=f"4. {rate_mode} Statistically Significant Hot/Cold Spots (Getis-Ord Gi*)",
                )
                fig.update_layout(margin=dict(l=0, r=0, t=70, b=0))
                st.plotly_chart(fig, width="stretch")
                legend_html = " &nbsp;&nbsp; ".join(
                    f'<span style="color:{bucket_colors[b]}">\u25a0</span> {b}'
                    for b in bucket_order if (tract_geo["GI_BUCKET"] == b).any()
                )
                st.markdown(f"<div style='font-size:0.85em'>{legend_html}</div>", unsafe_allow_html=True)

                st.caption(
                    f"**{n_hot:,}** tracts are statistically significant {rate_mode} hot spots and "
                    f"**{n_cold:,}** are significant cold spots, out of {len(tract_geo):,} tracts "
                    f"statewide -- the rest show no significant spatial clustering either way. "
                    f"**{n_hl_outlier:,}** tracts are High-Low outliers (Local Moran's I) -- a single "
                    f"elevated tract surrounded by normal ones, worth investigating as a specific "
                    f"location rather than an area-wide pattern."
                )

                sig_hot = tract_geo[tract_geo["GI_BUCKET"].str.startswith("Hot spot")].copy()
                if len(sig_hot):
                    sig_hot["County"] = sig_hot["GEOID"].astype(str).str.slice(2, 5).map(FL_COUNTY_FIPS)
                    sig_hot = sig_hot.sort_values("GI_Z", ascending=False)
                    with st.expander(f"All {len(sig_hot)} statistically significant {rate_mode} hot-spot tracts", expanded=False):
                        st.dataframe(
                            sig_hot[["GEOID", "County", rate_mode, "GI_Z", "GI_P", "GI_BUCKET", "MORAN_Q"]]
                            .rename(columns={rate_mode: f"{rate_mode} crashes", "GI_Z": "Gi* z-score",
                                             "GI_P": "p-value", "MORAN_Q": "Local Moran's I quadrant"})
                            .round({"Gi* z-score": 2, "p-value": 3}),
                            width="stretch", hide_index=True,
                        )

            st.markdown("---")
            st.markdown("#### Empirical Bayes Excess-Crash Ranking (Highway Safety Manual method)")
            with st.expander("Formula, SPF scope, and why EB", expanded=False):
                st.markdown(
                    f"""
                    Ranking tracts by raw rate has a well-known problem called **regression to the
                    mean**: one unlucky year can look "high risk" and then normalize on its own.
                    **Empirical Bayes (EB)** (AASHTO Highway Safety Manual) blends each tract's
                    *observed* count with a *predicted* count from a safety performance function
                    (SPF).

                    **Exact formulas used here** (negative-binomial SPF, NB2):
                    - `predicted = exp(β₀ [+ county effects]) × population`
                    - `k = 1 / α` (α = dispersion from the fit)
                    - `w = k / (k + predicted)`
                    - `EB estimate = w × predicted + (1 − w) × observed`
                    - `Excess crashes = observed − EB estimate`

                    **SPF scope:** trained on **all Florida tracts** for the selected mode
                    (`{rate_mode}`), with population as exposure. When stable, the model also
                    includes **county fixed effects** so regional differences are absorbed while
                    still pooling information statewide — better for sparse modes than fitting
                    separate county SPFs. If the county-FE fit fails, we fall back to a
                    statewide intercept-only SPF and note that below.
                    """
                )
            try:
                import statsmodels.api as sm
                STATSMODELS_AVAILABLE = True
            except ImportError:
                STATSMODELS_AVAILABLE = False
                st.info("Empirical Bayes ranking needs `statsmodels` -- `pip install statsmodels` to enable it.")

            if STATSMODELS_AVAILABLE and has_pop:
                eb_df = tract_geo[["GEOID", tract_pop_col, rate_mode]].copy()
                eb_df = eb_df[(eb_df[tract_pop_col] > 0) & eb_df[tract_pop_col].notna()].copy()
                eb_df["County"] = eb_df["GEOID"].astype(str).str.slice(2, 5).map(FL_COUNTY_FIPS)
                try:
                    import warnings as _warnings
                    spf_used = "statewide intercept-only"
                    spf_family = "negative binomial"
                    y = eb_df[rate_mode].astype(float)
                    exp = eb_df[tract_pop_col].astype(float)
                    X_intercept = np.ones((len(eb_df), 1))

                    def _predict_mean(res, model_family):
                        if model_family == "negative binomial":
                            pred = np.asarray(res.predict(which="mean"))
                        else:
                            pred = np.asarray(res.predict())
                        return pred.ravel() if pred.ndim > 1 else pred

                    def _try_nb(design, label):
                        nb = sm.NegativeBinomial(
                            y, design, exposure=exp, loglike_method="nb2",
                        )
                        res = nb.fit(disp=0, maxiter=200)
                        pred = _predict_mean(res, "negative binomial")
                        if np.isfinite(pred).all():
                            return res, label, pred, "negative binomial"
                        return None

                    def _try_poisson(design, label):
                        po = sm.GLM(y, design, family=sm.families.Poisson(), exposure=exp)
                        res = po.fit(maxiter=200)
                        pred = _predict_mean(res, "poisson")
                        if np.isfinite(pred).all():
                            return res, label, pred, "poisson"
                        return None

                    with _warnings.catch_warnings(record=True) as _caught:
                        _warnings.simplefilter("always")
                        county_dummies = pd.get_dummies(eb_df["County"].fillna("Unknown"), drop_first=True)
                        use_county_fe = county_dummies.shape[1] >= 2 and y.sum() >= 50
                        fit = None
                        if use_county_fe:
                            X_fe = np.column_stack([np.ones(len(eb_df)), county_dummies.values.astype(float)])
                            fit = _try_nb(X_fe, "statewide with county fixed effects (NB)")
                            if fit is None:
                                fit = _try_poisson(X_fe, "statewide with county fixed effects (Poisson fallback)")
                        if fit is None:
                            fit = _try_nb(X_intercept, "statewide intercept-only (NB)")
                        if fit is None:
                            fit = _try_poisson(X_intercept, "statewide intercept-only (Poisson fallback)")
                        if fit is None:
                            raise ValueError("SPF fit failed for negative binomial and Poisson forms")
                        spf_res, spf_used, predicted, spf_family = fit

                        # EB shrinkage uses NB dispersion even when SPF mean came from Poisson.
                        nb_alpha = sm.NegativeBinomial(
                            y, X_intercept, exposure=exp, loglike_method="nb2",
                        ).fit(disp=0, maxiter=200)
                        alpha = max(float(nb_alpha.params.get("alpha", 1.0)), 1e-6)
                        if not np.isfinite(alpha):
                            alpha = 1.0

                    did_not_converge = any("did not converge" in str(w.message).lower() for w in _caught)
                    if not np.isfinite(predicted).all():
                        raise ValueError("SPF produced non-finite predictions after fallback chain")

                    k = 1 / alpha
                    weight = k / (k + predicted)
                    observed = y.values
                    eb_df["Predicted (SPF)"] = predicted
                    eb_df["EB estimate"] = weight * predicted + (1 - weight) * observed
                    eb_df["Excess crashes"] = observed - eb_df["EB estimate"]
                    eb_df["EB_PCTL"] = eb_df["EB estimate"].rank(pct=True) * 100

                    top_excess = eb_df.sort_values("Excess crashes", ascending=False).head(15)
                    st.dataframe(
                        top_excess[["GEOID", "County", tract_pop_col, rate_mode,
                                     "Predicted (SPF)", "EB estimate", "Excess crashes"]]
                        .rename(columns={tract_pop_col: "Population", rate_mode: f"Observed {rate_mode} crashes"})
                        .round({"Predicted (SPF)": 2, "EB estimate": 2, "Excess crashes": 2}),
                        width="stretch", hide_index=True,
                    )
                    with st.expander("What each column means (plain language)", expanded=True):
                        ex = top_excess.iloc[0] if len(top_excess) else None
                        if ex is not None:
                            st.markdown(
                                f"""
                                | Column | What it means |
                                |---|---|
                                | **Population** = {ex[tract_pop_col]:,.0f} | About {ex[tract_pop_col]:,.0f} people live in this census tract. |
                                | **Observed {rate_mode} crashes** = {ex[rate_mode]:,.0f} | You actually recorded {ex[rate_mode]:,.0f} {rate_mode.lower()} crashes in this tract. |
                                | **Predicted (SPF)** = {ex['Predicted (SPF)']:.2f} | Based on the Florida SPF (population ± county), the model says you'd normally expect about {ex['Predicted (SPF)']:.2f} crashes in a tract like this. |
                                | **EB estimate** = {ex['EB estimate']:.2f} | After shrinking the raw count toward the SPF (because {ex[rate_mode]:,.0f} might be unusually high or low by chance), EB estimates the underlying crash level at about {ex['EB estimate']:.2f}. |
                                | **Excess crashes** = {ex['Excess crashes']:.2f} | About {ex['Excess crashes']:.2f} crashes above (or below, if negative) what we'd consider the normal/expected level. |
                                """
                            )
                        else:
                            st.caption("No tracts available to illustrate.")

                    eb_map = tract_geo[["geometry", "GEOID"]].merge(
                        eb_df[["GEOID", "EB estimate", "EB_PCTL", "Predicted (SPF)",
                               "Excess crashes", rate_mode, tract_pop_col, "County"]],
                        on="GEOID", how="left",
                    )
                    hover_cd = np.column_stack([
                        eb_map["EB estimate"].fillna(0),
                        eb_map["Predicted (SPF)"].fillna(0),
                        eb_map["Excess crashes"].fillna(0),
                        eb_map[rate_mode].fillna(0),
                        eb_map[tract_pop_col].fillna(0),
                        eb_map["County"].fillna("").astype(str),
                    ])
                    fig = go.Figure(go.Choroplethmap(
                        geojson=eb_map.geometry.__geo_interface__,
                        locations=eb_map.index, z=eb_map["EB_PCTL"],
                        zmin=0, zmax=100, colorscale="YlOrRd",
                        marker_opacity=0.7, marker_line_width=0.3,
                        colorbar_title="EB percentile",
                        customdata=hover_cd,
                        hovertemplate=(
                            "County: %{customdata[5]}<br>"
                            "EB estimate: %{customdata[0]:.2f}<br>"
                            "Observed: %{customdata[3]:.0f}<br>"
                            "Predicted (SPF): %{customdata[1]:.2f}<br>"
                            "Excess: %{customdata[2]:.2f}<br>"
                            "Population: %{customdata[4]:,.0f}"
                            "<extra></extra>"
                        ),
                    ))
                    fig.update_layout(
                        map_style="open-street-map", map_zoom=5.4,
                        map_center={"lat": 27.8, "lon": -81.7},
                    )
                    fig = style_fig(
                        fig, height=520, n=int(eb_df.shape[0]),
                        title=f"5. {rate_mode} Empirical Bayes Estimate by Tract (percentile rank)",
                    )
                    fig.update_layout(margin=dict(l=0, r=0, t=70, b=0))
                    st.plotly_chart(fig, width="stretch")
                    st.caption(
                        "Colored by **EB estimate percentile** across tracts (0–100). Hover any tract "
                        "for the actual EB estimate, observed count, SPF prediction, excess, and population."
                    )

                    if did_not_converge:
                        st.warning(
                            f"The SPF for {rate_mode} didn't fully converge -- likely because "
                            f"{rate_mode} crashes are too sparse per tract. Treat this ranking as "
                            f"indicative rather than final."
                        )
                    st.caption(
                        f"Top 15 tracts by EB-adjusted excess {rate_mode} crashes. "
                        f"SPF: **{spf_used}**. Dispersion α={alpha:.3f} "
                        f"(from intercept-only NB; SPF mean from {spf_family})."
                    )
                except Exception as e:
                    st.warning(f"Empirical Bayes model failed to fit on the current filter selection: {e}")

render_pipeline_figures("tab3")
