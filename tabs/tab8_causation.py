st.markdown("## Crash Causation (LLM Narrative Classification)")
st.markdown(
    """<div class="section-note">
    Every crash here was read by an LLM classifier and assigned a
    <code>primary_cause</code>, an <code>infrastructure_type</code> (where the
    rider was at the moment of impact), and whether <code>speed_contributing</code>
    played a role -- the closest thing in this dataset to "why did this crash
    actually happen," as opposed to what conditions surrounded it. Source:
    <code>multilabel_RegBike_cause.xlsx</code> + <code>multilabel_ebike_cause.xlsx</code>.
    </div>""",
    unsafe_allow_html=True,
)

if cause_raw is None or "primary_cause" not in cause_raw.columns:
    st.markdown(
        f"""<div class="section-note">
        No <code>{DEFAULT_CAUSE_PATH}</code> loaded, so the causation
        breakdown isn't available -- add it under the <b>Data Source</b> panel
        in the sidebar.
        </div>""",
        unsafe_allow_html=True,
    )
else:
    cdf = cause_raw[cause_raw["MODE"].isin(sel_modes)].copy()
    if MAIN_CRASH_ID_COL:
        filtered_rns = set(df[MAIN_CRASH_ID_COL].astype(str))
        cdf = cdf[cdf["REPORT_NUMBER"].isin(filtered_rns)]

    st.caption(
        f"**{len(cdf):,}** narrative-classified crashes match the current sidebar "
        f"filters (Mode, Year, Severity, County, etc. all apply here too)."
    )

    if len(cdf) == 0:
        st.info("No causation-classified crashes in the current filter selection.")
    else:
        present_modes = [m for m in MODES if m in cdf["MODE"].unique()]
        # insight() now comes from dashboard_core.py (shared across tabs --
        # tab5 called it before it was ever defined in the old single-file
        # version, which would have crashed the app the first time that
        # branch ran).

        # ================================================================
        # 1. Fault attribution -- driver vs. non-motorist vs. ambiguous
        # ================================================================
        st.markdown("### 1. Fault Attribution")
        att = cdf.groupby(["MODE", "ATTRIBUTION"], observed=True).size().reset_index(name="count")
        att["pct"] = att.groupby("MODE")["count"].transform(lambda s: s / s.sum() * 100)
        fig = px.bar(
            att, x="MODE", y="pct", color="ATTRIBUTION", barmode="stack",
            category_orders={"MODE": present_modes, "ATTRIBUTION": list(ATTRIBUTION_COLORS.keys())},
            color_discrete_map=ATTRIBUTION_COLORS,
            text=att["pct"].round(1).astype(str) + "%",
        )
        fig.update_layout(yaxis_title="% of narrative-classified crashes", xaxis_title=None)
        st.plotly_chart(style_fig(fig, title="Fault Attribution by Mode"), use_container_width=True)
        st.caption(
            "Driver-attributable: failed to yield turning, ran stop/red light, following "
            "too close, distracted, speeding/reckless, impaired, dooring. Non-motorist-"
            "attributable: failed to yield entering roadway, ran stop/signal, wrong-way "
            "riding, sidewalk-driveway conflict."
        )
        overall_att = cdf["ATTRIBUTION"].value_counts(normalize=True) * 100
        top2 = cdf["CAUSE_LABEL"].value_counts(normalize=True).nlargest(2) * 100
        insight(
            f"Across the current selection, fault splits nearly evenly: "
            f"{overall_att.get('Driver-attributable', 0):.1f}% driver-attributable vs. "
            f"{overall_att.get('Non-motorist-attributable', 0):.1f}% non-motorist-attributable. "
            f"Just two categories -- <b>{top2.index[0]}</b> ({top2.iloc[0]:.1f}%) and "
            f"<b>{top2.index[1] if len(top2) > 1 else ''}</b> "
            f"({top2.iloc[1] if len(top2) > 1 else 0:.1f}%) -- account for "
            f"{top2.sum():.1f}% of all classified crashes. This is overwhelmingly a "
            f"yielding problem at points where paths cross, not a diverse mix of failure modes."
        )

        # ================================================================
        # 2. Where crashes happen
        # ================================================================
        st.markdown("### 2. Where the Rider Was at Impact")
        im = cdf.groupby(["INFRA_LABEL", "MODE"], observed=True).size().reset_index(name="count")
        fig = px.bar(
            im, y="INFRA_LABEL", x="count", color="MODE", orientation="h",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": present_modes},
        )
        fig.update_layout(
            yaxis_title=None, xaxis_title="Crashes",
            yaxis={"categoryorder": "total ascending"}, barmode="stack",
        )
        st.plotly_chart(
            style_fig(fig, title="Infrastructure Type at Impact, by Mode", height=440),
            use_container_width=True,
        )
        infra_pct = cdf["INFRA_LABEL"].value_counts(normalize=True) * 100
        off_road = infra_pct.get("Sidewalk", 0) + infra_pct.get("Crosswalk", 0)
        bike_lane_pct = infra_pct.get("Bike lane", 0)
        insight(
            f"Sidewalk + crosswalk together account for <b>{off_road:.1f}%</b> of "
            f"narrative-classified crashes in the current selection -- roughly half of all "
            f"crashes happen where the rider wasn't in the road at all. Only "
            f"<b>{bike_lane_pct:.1f}%</b> happened in a dedicated bike lane, despite how "
            f"much infrastructure conversation centers on bike lanes specifically."
        )

        # ================================================================
        # 3. Cause x location interaction (the "right hook" pattern)
        # ================================================================
        st.markdown("### 3. Primary Cause by Location -- Infrastructure Doesn't Remove the Yielding Problem, It Relocates It")
        mode_label = " + ".join(present_modes)
        cross = cdf.groupby(["INFRA_LABEL", "CAUSE_LABEL"], observed=True).size().reset_index(name="count")
        pivot = cross.pivot(index="INFRA_LABEL", columns="CAUSE_LABEL", values="count").fillna(0)
        top_cause_cols = cdf["CAUSE_LABEL"].value_counts().nlargest(8).index
        pivot_cols = [c for c in top_cause_cols if c in pivot.columns]
        pivot_pct = pivot[pivot_cols].div(pivot[pivot_cols].sum(axis=1).replace(0, 1), axis=0) * 100
        fig = go.Figure(go.Heatmap(
            z=pivot_pct.values, x=pivot_pct.columns, y=pivot_pct.index,
            colorscale="Reds", colorbar=dict(title="% of that location's crashes"),
            text=np.round(pivot_pct.values, 1), texttemplate="%{text}",
        ))
        st.plotly_chart(
            style_fig(fig, title=f"Primary Cause x Location -- {mode_label} (row %)", height=420),
            use_container_width=True,
        )
        bike_lane_row = pivot_pct.loc["Bike lane"] if "Bike lane" in pivot_pct.index else None
        if bike_lane_row is not None and len(bike_lane_row):
            top_bl_cause = bike_lane_row.idxmax()
            insight(
                f"For the current mode selection ({mode_label}), bike-lane crashes are "
                f"dominated by <b>{top_bl_cause}</b> ({bike_lane_row.max():.1f}% of bike-lane "
                f"crashes) -- a dedicated bike lane doesn't remove the 'driver turns across "
                f"the rider's path' problem, it's often where that pattern is most "
                f"concentrated (a 'right-hook' at intersections, since the lane puts the "
                f"rider exactly where a right-turning driver is most likely to miss them)."
            )

        # ================================================================
        # 4. Sidewalk riding is a driveway problem
        # ================================================================
        st.markdown("### 4. Sidewalk Crashes: a Driveway Problem More Than a Road-Crossing Problem")
        sw = cdf[cdf["INFRA_LABEL"] == "Sidewalk"]
        if len(sw):
            sw_causes = sw["CAUSE_LABEL"].value_counts(normalize=True).nlargest(6) * 100
            fig = px.bar(
                x=sw_causes.values, y=sw_causes.index, orientation="h",
                color_discrete_sequence=["#5C6BC0"],
            )
            fig.update_layout(yaxis_title=None, xaxis_title="% of sidewalk crashes",
                               yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(
                style_fig(fig, title=f"Top Causes of Sidewalk Crashes ({', '.join(present_modes)})", height=340),
                use_container_width=True,
            )
            dwc = sw["CAUSE_LABEL"].value_counts(normalize=True).get("Sidewalk driveway conflict", 0) * 100
            dft = sw["CAUSE_LABEL"].value_counts(normalize=True).get("Driver failed to yield turning", 0) * 100
            insight(
                f"Sidewalk crashes split mainly between driveway conflicts "
                f"({dwc:.1f}%) and drivers failing to yield while turning ({dft:.1f}%) -- "
                f"together {dwc + dft:.1f}% of sidewalk crashes. The risk of sidewalk "
                f"riding looks concentrated at the many driveway crossings a sidewalk route "
                f"passes, not from riding along the sidewalk itself -- pointing toward "
                f"driveway-crossing treatments as a more targeted fix than sidewalk-riding bans."
            )
        else:
            st.info("No sidewalk crashes in the current filter selection.")

        # ================================================================
        # 5. Speed as a documented factor
        # ================================================================
        st.markdown("### 5. Speed as a Documented Factor (Either Party)")
        sc = cdf.groupby(["MODE", "speed_contributing"], observed=True).size().reset_index(name="count")
        sc["pct"] = sc.groupby("MODE")["count"].transform(lambda s: s / s.sum() * 100)
        fig = px.bar(
            sc, x="MODE", y="pct", color="speed_contributing", barmode="stack",
            category_orders={"MODE": present_modes, "speed_contributing": ["yes", "unclear", "no"]},
            color_discrete_map={"yes": "#C0392B", "unclear": "#BDBDBD", "no": "#81C784"},
        )
        fig.update_layout(yaxis_title="% of narrative-classified crashes", xaxis_title=None)
        st.plotly_chart(style_fig(fig, title="Speed Flagged as Contributing (Driver or Rider), by Mode"), use_container_width=True)
        st.caption(
            "**Not micromobility-speed-specific.** This flags whether the narrative says "
            "*anyone's* speed -- the rider's or the driver's -- contributed to the crash; it "
            "isn't restricted to the micromobility rider. In a sample check, ~65% of 'yes' "
            "narratives referenced the rider's speed and ~38% referenced the driver's/"
            "vehicle's (some flag both). For rider-speed-only numbers, see the "
            "`MICROMOBILITY_SPEED_MPH` chart on the Insights tab (Section 4)."
        )
        yes_by_mode = cdf[cdf["speed_contributing"] == "yes"].groupby("MODE", observed=True).size() \
            / cdf.groupby("MODE", observed=True).size() * 100
        yes_text = "; ".join(f"{m}: {yes_by_mode.get(m, 0):.1f}%" for m in present_modes)
        insight(
            f"'Yes' rates by mode -- {yes_text}. Don't read the low overall rate as "
            f"'speed isn't a factor': officer narratives often just don't discuss speed "
            f"unless it was extreme or measured, so <b>unclear</b> (the largest bucket for "
            f"every mode) reflects a reporting gap, not evidence that speed was low."
        )

        # ================================================================
        # 6. Hit-and-run & wrong-way riding
        # ================================================================
        st.markdown("### 6. Hit-and-Run & Wrong-Way Riding")
        hr = cdf[cdf["primary_cause"] == "hit_and_run"]
        wwr = cdf[cdf["primary_cause"] == "wrong_way_riding"]
        hc1, hc2 = st.columns(2)
        with hc1:
            if len(hr):
                hr_infra = hr["INFRA_LABEL"].value_counts(normalize=True) * 100
                fig = px.bar(
                    x=hr_infra.values, y=hr_infra.index, orientation="h",
                    color_discrete_sequence=["#C0392B"],
                )
                fig.update_layout(yaxis_title=None, xaxis_title="% of hit-and-run crashes",
                                   yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(
                    style_fig(fig, title=f"Hit-and-Run Location (n={len(hr):,})", height=320),
                    use_container_width=True,
                )
            else:
                st.info("No hit-and-run crashes in the current filter selection.")
        with hc2:
            if len(wwr):
                wwr_infra = wwr["INFRA_LABEL"].value_counts(normalize=True) * 100
                fig = px.bar(
                    x=wwr_infra.values, y=wwr_infra.index, orientation="h",
                    color_discrete_sequence=["#FF9800"],
                )
                fig.update_layout(yaxis_title=None, xaxis_title="% of wrong-way-riding crashes",
                                   yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(
                    style_fig(fig, title=f"Wrong-Way Riding Location (n={len(wwr):,})", height=320),
                    use_container_width=True,
                )
            else:
                st.info("No wrong-way-riding crashes in the current filter selection.")
        insight(
            f"Hit-and-run ({len(hr):,} crashes, {len(hr) / len(cdf) * 100:.1f}% of the "
            f"current selection) is spread fairly evenly across location types -- a driver-"
            f"behavior issue independent of where the rider was, not a location-specific risk. "
            f"Wrong-way riding ({len(wwr):,} crashes, {len(wwr) / len(cdf) * 100:.1f}%) "
            f"concentrates in travel lanes and bike lanes rather than sidewalks -- mostly a "
            f"'riding against traffic in a facility meant for one-way travel' problem."
        )

        # ================================================================
        # Data quality notes
        # ================================================================
        st.markdown("### Data Quality Notes")
        noise = cause_raw[cause_raw["MODE"] == "Other"]
        cross_class = cause_raw.groupby("SOURCE_FILE")["MODE"].value_counts() if "SOURCE_FILE" in cause_raw.columns else None
        notes = [
            f"**Confidence is effectively capped** in the source classifier's output -- "
            f"not a strong signal for filtering 'high-quality' rows beyond excluding a small low tail.",
            f"**{len(noise):,} rows across both source files** landed as mode `Other` rather "
            f"than Bicycle/E-Bike/E-Scooter and are excluded from every chart on this tab.",
            "`cause_flag` is almost entirely empty in the source files and isn't a usable QA filter as-is.",
        ]
        for n in notes:
            st.markdown(f"- {n}")
        if cross_class is not None:
            with st.expander("Raw prediction counts by source file (mode-label cross-contamination)"):
                st.dataframe(cross_class.unstack(fill_value=0), use_container_width=True)

render_pipeline_figures("tab8")


