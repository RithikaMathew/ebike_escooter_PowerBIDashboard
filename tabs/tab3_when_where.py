c1, c2 = st.columns(2)
with c1:
    dow_mode = df.groupby(["DOW", "MODE"], observed=True).size().reset_index(name="count")
    fig = px.bar(
        dow_mode, x="DOW", y="count", color="MODE",
        color_discrete_map=MODE_COLORS,
        category_orders={"DOW": DOW_ORDER, "MODE": MODES},
    )
    fig.update_layout(xaxis_title=None, yaxis_title="Crashes")
    st.plotly_chart(style_fig(fig, title="Crashes by Day of Week", n=total), use_container_width=True)

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
        style_fig(fig, title="Day vs. Night Share by Mode", n=dn_mode_n), use_container_width=True
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
        style_fig(fig, title="Intersection vs. Segment by Mode", n=loc_mode_n), use_container_width=True
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
        use_container_width=True,
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
        use_container_width=True,
    )

with c6:
    top_counties = df["COUNTY_NAME"].value_counts().nlargest(15).index
    co_df = df[df["COUNTY_NAME"].isin(top_counties)]
    cm = co_df.groupby(["COUNTY_NAME", "MODE"], observed=True).size().reset_index(name="count")
    fig = px.bar(
        cm, y="COUNTY_NAME", x="count", color="MODE", orientation="h",
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
    )
    fig.update_layout(yaxis_title=None, xaxis_title="Crashes",
                       yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(
        style_fig(fig, title="Top 15 Counties", height=430, n=int(co_df.shape[0])),
        use_container_width=True,
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
        fig = px.scatter_mapbox(
            geo, lat=LAT_COL, lon=LON_COL, color="MODE",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
            opacity=0.55, zoom=5.4, height=560,
        )
        fig = style_fig(
            fig, height=560,
            title=f"Crash Locations by Mode ({len(geo):,} of {total:,} filtered crashes geocoded)",
        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=56, b=0))
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True)
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
        **What's a census tract?** It's the Census Bureau's standard small-area unit --
        roughly a neighborhood, typically 1,200-8,000 residents. Florida is divided into
        about 5,000 of them. They're the building block for anything you see below;
        `GEOID` is just each tract's unique ID number.

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
        Scroll/zoom into any map to inspect a specific area.

        **The three maps, and what each one actually answers:**
        - **Map 1 (raw counts)** -- "where do the most crashes happen." Dominated by
          population density: a big city tract will out-count a small town even if the small
          town is more dangerous per rider.
        - **Map 2 (per 100,000 residents, colored by percentile rank)** -- "where is a
          resident most likely to be involved in a crash," controlling for how many people
          live there. Use the mode selector below to switch which mode this is computed for.
          Two caveats worth knowing: (1) tracts under 100 residents are dropped, since a rate
          off a tiny population is statistical noise, not signal; (2) it's colored by each
          tract's **rank** relative to other tracts (0-100), not the raw number, because a
          couple of small-population tracts would otherwise blow out the color scale and make
          the rest of the state look flat. Hover any tract to see its actual rate; the exact
          highest-rate tracts are also listed in the table beneath the map. A rate can also
          run high in tourist-heavy areas (like the Keys) simply because Census population
          only counts year-round residents, not visitors actually riding there -- worth
          keeping in mind before reading a high rate as "riskier for locals."
        - **Map 3 (mode share %)** -- "of the micromobility crashes in this tract, what
          fraction were this mode," independent of the tract's total crash volume. A tract
          with very few total crashes can show a misleadingly extreme % here (1 crash that
          happens to be an e-bike crash = 100%) -- always read it alongside Map 1's count.

        **Hotspot cluster centers** (optional overlay on Map 1, plus its own table) are a
        different, arguably more direct tool for "where will this happen again": they come
        from **DBSCAN**, a density-based clustering algorithm that groups crash points which
        are close together in space (and time) into a cluster without needing tract
        boundaries at all (`eda_analysis_combined.py` section 09d). A "cluster" is a
        real recurring location, not an administrative shape, so it isn't subject to the
        small-population noise that Map 2 has to work around. Bubble size / table rank =
        crashes in that cluster.
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

            # Mode picker for maps 2 & 3 -- these used to be hardcoded to E-Bike only.
            rate_mode = st.radio(
                "Mode for maps 2 & 3 (per-capita rate + mode share)",
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
                fig = go.Figure(go.Choroplethmapbox(**trace_kwargs))
                if hotspot_df is not None and len(hotspot_df) and {"CENTER_LAT", "CENTER_LON"}.issubset(hotspot_df.columns):
                    hd = hotspot_df.copy()
                    hd["N_CRASHES"] = pd.to_numeric(hd.get("N_CRASHES"), errors="coerce").fillna(0)
                    hd = hd.dropna(subset=["CENTER_LAT", "CENTER_LON"])
                    max_n = hd["N_CRASHES"].max()
                    if len(hd) and pd.notna(max_n) and max_n > 0:
                        sizes = (hd["N_CRASHES"] / max_n * 22 + 6)
                    else:
                        sizes = 10  # flat fallback size if there's no usable N_CRASHES to scale by
                    fig.add_trace(go.Scattermapbox(
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
                    mapbox_style="open-street-map", mapbox_zoom=5.4,
                    mapbox_center={"lat": 27.8, "lon": -81.7},
                    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                )
                fig = style_fig(fig, height=520, title=title, n=subtitle_n)
                fig.update_layout(margin=dict(l=0, r=0, t=70, b=0))
                return fig

            show_hotspots = False
            hs_for_map = None
            if hotspot_raw is not None and "MODE" in hotspot_raw.columns:
                show_hotspots = st.checkbox(
                    "Overlay hotspot cluster centers on Map 1 (sized by crashes in cluster)",
                    value=True, key="tract_map_hotspot_overlay",
                )
                if show_hotspots:
                    hs_for_map = hotspot_raw[hotspot_raw["MODE"].isin(sel_modes)].copy()

            m1, m2 = st.columns(2)
            with m1:
                st.plotly_chart(
                    choropleth(
                        "TOTAL_MICRO",
                        "1. Micromobility Crashes per Tract (all modes)",
                        "Crashes", n_matched,
                        hotspot_df=hs_for_map,
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "Raw crash count per tract, all three modes combined."
                    + (" Teal bubbles = DBSCAN hotspot cluster centers for the sidebar's "
                       "selected mode(s), sized by crashes in that cluster." if show_hotspots else "")
                )
                if show_hotspots and hs_for_map is not None and len(hs_for_map) and "N_CRASHES" in hs_for_map.columns:
                    with st.expander("Top 10 hotspot clusters (recurring crash locations, not tied to tract boundaries)", expanded=True):
                        st.markdown(
                            "These are DBSCAN spatiotemporal clusters of actual crash locations -- the most "
                            "direct read on *where crashes keep recurring*, since they're not diluted by "
                            "population or forced into a census tract's arbitrary shape."
                        )
                        cluster_mode_options = ["All modes shown on Map 1"] + [m for m in MODES if m in hs_for_map["MODE"].unique()]
                        cluster_mode_filter = st.selectbox(
                            "Filter this table by mode", cluster_mode_options, index=0, key="tract_hotspot_table_mode",
                        )
                        top_clusters = hs_for_map.copy()
                        if cluster_mode_filter != "All modes shown on Map 1":
                            top_clusters = top_clusters[top_clusters["MODE"] == cluster_mode_filter]
                        top_clusters["N_CRASHES"] = pd.to_numeric(top_clusters["N_CRASHES"], errors="coerce").fillna(0)
                        top_clusters = top_clusters.nlargest(10, "N_CRASHES")
                        cluster_cols = [c for c in ["CLUSTER_ID", "MODE", "N_CRASHES", "CENTER_LAT", "CENTER_LON"]
                                        if c in top_clusters.columns]
                        if len(top_clusters):
                            st.dataframe(top_clusters[cluster_cols], use_container_width=True, hide_index=True)
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
                        use_container_width=True,
                    )
                    st.caption(
                        f"Tracts colored by their **rank** among all tracts on {rate_mode} crashes "
                        f"\u00f7 population \u00d7 100,000 (hover a tract for its actual rate) -- "
                        f"percentile keeps the color spread readable, since the raw rate is dominated "
                        f"by whichever tract happens to have the smallest population. Tracts under "
                        f"{MIN_TRACT_POP} residents ({n_small_pop:,} of them) are excluded and shown "
                        f"blank, since a rate off a tiny population isn't reliable. See the exact "
                        f"highest-rate tracts in the table below."
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
                                use_container_width=True, hide_index=True,
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
                use_container_width=True,
            )
            st.caption(
                f"{rate_mode} crashes \u00f7 (bicycle + e-bike + e-scooter crashes) in that tract, "
                f"as a %. Only meaningful where TOTAL_MICRO is non-trivial -- a tract with 1 total "
                f"crash that happens to be a {rate_mode.lower()} crash shows 100% here, so read "
                f"this alongside Map 1's raw count, not in isolation."
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
                import libpysal
                from esda.getisord import G_Local
                from esda.moran import Moran_Local
                SPATIAL_STATS_AVAILABLE = True
            except ImportError:
                SPATIAL_STATS_AVAILABLE = False
                st.info(
                    "Spatial statistics need the `libpysal` and `esda` packages -- "
                    "`pip install libpysal esda` and re-run to enable this map."
                )

            if SPATIAL_STATS_AVAILABLE:
                @st.cache_resource(show_spinner="Building spatial weights (once per tract file)...")
                def _build_tract_weights(geoid_tuple, _geom_for_cache):
                    w = libpysal.weights.Queen.from_dataframe(
                        _geom_for_cache, use_index=True, silence_warnings=True,
                    )
                    w = libpysal.weights.fill_diagonal(w, 1.0)
                    w.transform = "r"
                    return w

                weights = _build_tract_weights(tuple(tract_geo["GEOID"].astype(str)), tract_geo[["GEOID", "geometry"]])

                gi_y = tract_geo[rate_mode].fillna(0).values.astype(float)
                gi = G_Local(gi_y, weights, star=None, permutations=999, seed=0)
                lm = Moran_Local(gi_y, weights, permutations=999, seed=0)

                tract_geo["GI_Z"] = gi.Zs
                tract_geo["GI_P"] = gi.p_sim

                def _gi_bucket(z, p):
                    if pd.isna(p) or p > 0.10:
                        return "Not significant"
                    conf = "99%" if p <= 0.01 else ("95%" if p <= 0.05 else "90%")
                    return f"Hot spot ({conf} confidence)" if z > 0 else f"Cold spot ({conf} confidence)"

                tract_geo["GI_BUCKET"] = [_gi_bucket(z, p) for z, p in zip(tract_geo["GI_Z"], tract_geo["GI_P"])]

                moran_labels = {1: "High-High (cluster)", 2: "Low-High (outlier)",
                                 3: "Low-Low (cluster)", 4: "High-Low (outlier)"}
                tract_geo["MORAN_Q"] = [
                    moran_labels.get(q, "n/a") if p <= 0.05 else "Not significant"
                    for q, p in zip(lm.q, lm.p_sim)
                ]

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

                fig = go.Figure(go.Choroplethmapbox(
                    geojson=tract_geo.geometry.__geo_interface__,
                    locations=tract_geo.index, z=tract_geo["GI_BUCKET_CODE"],
                    customdata=tract_geo["GI_BUCKET"],
                    colorscale=stepped_colorscale, zmin=0, zmax=n_buckets,
                    marker_opacity=0.75, marker_line_width=0.3,
                    hovertemplate="%{customdata}<extra></extra>", showscale=False,
                ))
                fig.update_layout(
                    mapbox_style="open-street-map", mapbox_zoom=5.4,
                    mapbox_center={"lat": 27.8, "lon": -81.7},
                )
                fig = style_fig(
                    fig, height=560, n=n_matched,
                    title=f"4. {rate_mode} Statistically Significant Hot/Cold Spots (Getis-Ord Gi*)",
                )
                fig.update_layout(margin=dict(l=0, r=0, t=70, b=0))
                st.plotly_chart(fig, use_container_width=True)
                legend_html = " &nbsp;&nbsp; ".join(
                    f'<span style="color:{bucket_colors[b]}">\u25a0</span> {b}'
                    for b in bucket_order if (tract_geo["GI_BUCKET"] == b).any()
                )
                st.markdown(f"<div style='font-size:0.85em'>{legend_html}</div>", unsafe_allow_html=True)

                n_hot = tract_geo["GI_BUCKET"].str.startswith("Hot spot").sum()
                n_cold = tract_geo["GI_BUCKET"].str.startswith("Cold spot").sum()
                n_hl_outlier = (tract_geo["MORAN_Q"] == "High-Low (outlier)").sum()
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
                            use_container_width=True, hide_index=True,
                        )

            st.markdown("---")
            st.markdown("#### Empirical Bayes Excess-Crash Ranking (Highway Safety Manual method)")
            with st.expander("Why this ranking is different from the raw-rate table above", expanded=False):
                st.markdown(
                    """
                    Ranking tracts by raw rate (or percentile) has a well-known problem in safety
                    analysis called **regression to the mean**: a tract with one genuinely unlucky
                    year looks "high risk" in a snapshot like this and will often look normal again
                    on its own next year, with no intervention needed. The **Empirical Bayes (EB)**
                    method from the AASHTO Highway Safety Manual -- the standard approach state DOTs
                    use to prioritize sites -- corrects for this by blending each tract's *observed*
                    count with a *predicted* count from a safety performance function (SPF, a
                    negative-binomial regression of crashes on population fit across every tract
                    statewide), weighted by how reliable each source is for that tract. Tracts that
                    stay high after this correction are the ones worth prioritizing; tracts that drop
                    out of the ranking were likely just unlucky in this snapshot.
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
                eb_df = eb_df[(eb_df[tract_pop_col] > 0) & eb_df[tract_pop_col].notna()]
                try:
                    import warnings as _warnings
                    with _warnings.catch_warnings(record=True) as _caught:
                        _warnings.simplefilter("always")
                        nb = sm.NegativeBinomial(
                            eb_df[rate_mode].astype(float), np.ones((len(eb_df), 1)),
                            exposure=eb_df[tract_pop_col].astype(float), loglike_method="nb2",
                        )
                        nb_res = nb.fit(disp=0)
                    did_not_converge = any("did not converge" in str(w.message).lower() for w in _caught)

                    const, alpha = nb_res.params["const"], max(nb_res.params["alpha"], 1e-6)
                    if not np.isfinite(const) or not np.isfinite(alpha):
                        raise ValueError("model produced non-finite parameters")
                    k = 1 / alpha
                    predicted = np.exp(const) * eb_df[tract_pop_col]
                    weight = k / (k + predicted)
                    eb_df["Predicted (SPF)"] = predicted
                    eb_df["EB estimate"] = weight * predicted + (1 - weight) * eb_df[rate_mode]
                    eb_df["Excess crashes"] = eb_df[rate_mode] - eb_df["EB estimate"]
                    eb_df["County"] = eb_df["GEOID"].astype(str).str.slice(2, 5).map(FL_COUNTY_FIPS)

                    top_excess = eb_df.sort_values("Excess crashes", ascending=False).head(15)
                    st.dataframe(
                        top_excess[["GEOID", "County", tract_pop_col, rate_mode,
                                     "Predicted (SPF)", "EB estimate", "Excess crashes"]]
                        .rename(columns={tract_pop_col: "Population", rate_mode: f"Observed {rate_mode} crashes"})
                        .round({"Predicted (SPF)": 2, "EB estimate": 2, "Excess crashes": 2}),
                        use_container_width=True, hide_index=True,
                    )
                    if did_not_converge:
                        st.warning(
                            f"The statewide SPF for {rate_mode} didn't fully converge -- likely because "
                            f"{rate_mode} crashes are too sparse per tract for this fit to be stable "
                            f"(common for a lower-volume mode). Treat this ranking as indicative rather "
                            f"than final; it'll be more reliable on Bicycle (higher counts) or on the "
                            f"combined `TOTAL_MICRO` count."
                        )
                    st.caption(
                        f"Top 15 tracts by EB-adjusted excess {rate_mode} crashes (observed minus what "
                        f"the statewide population-based model predicts, after EB shrinkage toward that "
                        f"prediction). Dispersion parameter alpha={alpha:.3f} (higher means crash counts "
                        f"are more overdispersed than a simple Poisson model assumes -- typical for crash "
                        f"data, and the reason a negative-binomial SPF is used instead of plain Poisson)."
                    )
                except Exception as e:
                    st.warning(f"Empirical Bayes model failed to fit on the current filter selection: {e}")

render_pipeline_figures("tab3")