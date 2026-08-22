if demo is None or demo.empty:
    st.markdown(
        f"""<div class="section-note">
        No demographics data loaded. Add <code>{DEFAULT_DEMO_PATH}</code>
        (person-level age/gender export) under the <b>Data Source</b>
        panel in the sidebar to see age and gender breakdowns for the
        people involved in these crashes.
        </div>""",
        unsafe_allow_html=True,
    )
else:
    demo_mode_col = find_col(demo, ["MODE"])
    has_mode = demo_mode_col is not None

    if DEMO_AGE_AVAILABLE and has_mode:
        age_valid = demo[demo["_AGE"].notna()]
        if len(age_valid):
            fig = px.violin(
                age_valid, x=demo_mode_col, y="_AGE", color=demo_mode_col, box=True, points="outliers",
                color_discrete_map=MODE_COLORS, category_orders={demo_mode_col: MODES},
            )
            fig.update_layout(yaxis_title="Age", xaxis_title=None, showlegend=False)
            st.plotly_chart(style_fig(fig, title="Age Distribution by Mode (Violin Plot)", height=420), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if DEMO_AGE_AVAILABLE:
            if has_mode:
                age_band_df = demo.groupby([demo_mode_col, "_AGE_BAND"], observed=True).size().reset_index(name="count")
                fig = px.bar(
                    age_band_df, x="_AGE_BAND", y="count", color=demo_mode_col,
                    color_discrete_map=MODE_COLORS,
                    category_orders={"_AGE_BAND": AGE_BAND_ORDER},
                )
            else:
                age_band_df = demo.groupby("_AGE_BAND", observed=True).size().reset_index(name="count")
                fig = px.bar(age_band_df, x="_AGE_BAND", y="count", category_orders={"_AGE_BAND": AGE_BAND_ORDER})
            fig.update_layout(xaxis_title=None, yaxis_title="People involved")
            st.plotly_chart(style_fig(fig, title="Age Distribution by Band"), use_container_width=True)
        else:
            st.info("No recognizable age column found in the demographics file.")

    with c2:
        if DEMO_GENDER_AVAILABLE:
            gender_counts = demo["_GENDER"].value_counts()
            fig = go.Figure(go.Pie(
                labels=gender_counts.index, values=gender_counts.values, hole=0.55,
                marker=dict(colors=[GENDER_COLORS.get(g, "#B0BEC5") for g in gender_counts.index]),
                textinfo="label+percent",
            ))
            st.plotly_chart(style_fig(fig, title="Gender Breakdown"), use_container_width=True)
        else:
            st.info("No recognizable gender column found in the demographics file.")

    if DEMO_AGE_AVAILABLE and has_mode:
        sev_col = find_col(demo, ["S4_CRASH_SEVERITY", "SEVERITY", "INJURY_SEVERITY"])
        if sev_col:
            st.markdown("##### Age Distribution by Injury Severity, by Mode")
            st.caption(
                "Shown as three separate charts (one per mode) rather than one combined "
                "chart, since patterns like 'people in fatal crashes skew older' are easier "
                "to see within a single mode than averaged across all three."
            )
            age_sev_cols = st.columns(3)
            for slot, mode in zip(age_sev_cols, MODES):
                with slot:
                    sub = demo[(demo[demo_mode_col] == mode) & demo["_AGE"].notna() & demo[sev_col].notna()]
                    if len(sub) < 5:
                        st.info(f"Not enough {mode} records with age + severity in this filter.")
                        continue
                    fig = px.violin(
                        sub, x=sev_col, y="_AGE", color=sev_col, box=True, points=False,
                        color_discrete_map=SEVERITY_COLORS, category_orders={sev_col: SEVERITY_ORDER},
                    )
                    fig.update_layout(yaxis_title="Age", xaxis_title=None, showlegend=False)
                    st.plotly_chart(
                        style_fig(fig, title=f"{mode} (n={len(sub):,})", height=340),
                        use_container_width=True,
                    )

        if sev_col and DEMO_GENDER_AVAILABLE:
            gm = demo.groupby([demo_mode_col, "_GENDER"], observed=True).size().reset_index(name="count")
            gm["pct"] = gm["count"] / gm.groupby(demo_mode_col)["count"].transform("sum") * 100
            fig = px.bar(
                gm, x=demo_mode_col, y="pct", color="_GENDER",
                color_discrete_map=GENDER_COLORS, category_orders={demo_mode_col: MODES},
            )
            fig.update_layout(yaxis_title="% of people", xaxis_title=None, barmode="stack")
            st.plotly_chart(style_fig(fig, title="Gender Mix by Mode"), use_container_width=True)

    st.caption(
        f"Demographics reflect **{len(demo):,}** person-level records "
        f"matched to the currently filtered crashes."
    )

render_pipeline_figures("tab6")

