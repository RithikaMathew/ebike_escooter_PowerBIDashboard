c1, c2 = st.columns([1.4, 1])

with c1:
    rows = []
    for m in MODES:
        sub = df[df["MODE"] == m]
        n = len(sub)
        for f in DRIVER_FLAGS:
            rows.append({
                "Mode": m,
                "Flag": DRIVER_FLAG_LABELS.get(f, f),
                "Pct": sub[f].mean() * 100 if n else 0,
            })
    flag_df = pd.DataFrame(rows)
    flag_n = {m: int((df["MODE"] == m).sum()) for m in MODES}
    flag_df["N"] = flag_df["Mode"].map(flag_n)
    flag_df["count"] = (flag_df["Pct"] / 100 * flag_df["N"]).round().astype(int)
    fig = px.bar(
        flag_df, y="Flag", x="Pct", color="Mode", orientation="h", barmode="group",
        color_discrete_map=MODE_COLORS, category_orders={"Mode": MODES}, custom_data=["count"],
    )
    fig.update_traces(texttemplate="%{x:.0f}%", textposition="outside", textfont=dict(size=9),
                       hovertemplate="%{y}, %{fullData.name}: %{x:.1f}%% (n=%{customdata[0]:,})<extra></extra>")
    fig.update_layout(xaxis_title="% of crashes for that mode", yaxis_title=None)
    st.plotly_chart(
        style_fig(fig, title="Driver Behavior Flags by Mode", height=440, n=flag_n), use_container_width=True
    )

with c2:
    cite_mode_n = {m: int((df["MODE"] == m).sum()) for m in MODES}
    cite_n = {m: int(((df["MODE"] == m) & (df["CITED"] == True)).sum()) for m in MODES}  # noqa: E712
    cite_df = df.groupby("MODE", observed=True)["CITED"].mean().reindex(MODES).fillna(0) * 100
    fig = go.Figure(go.Bar(
        x=cite_df.index, y=cite_df.values,
        marker_color=[MODE_COLORS[m] for m in cite_df.index],
        text=[f"{v:.0f}%" for v in cite_df.values], textposition="outside",
        customdata=[[cite_n.get(m, 0)] for m in cite_df.index],
        hovertemplate="%{x}: %{y:.1f}%% (n=%{customdata[0]:,})<extra></extra>",
    ))
    fig.update_layout(yaxis_title="% crashes with citation", yaxis_range=[0, 110])
    st.plotly_chart(
        style_fig(fig, title="Citation Rate by Mode", n=cite_mode_n), use_container_width=True
    )

    cite_yr = df.groupby(["YEAR", "MODE"], observed=True)["CITED"].mean().reset_index()
    cite_yr["CITED"] *= 100
    fig2 = px.line(
        cite_yr, x="YEAR", y="CITED", color="MODE", markers=True,
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
    )
    fig2.update_layout(yaxis_title="% cited", xaxis_title=None)
    st.plotly_chart(style_fig(fig2, title="Citation Rate Over Time, by Mode", height=280), use_container_width=True)
    st.caption(
        "A declining rate here can reflect changing enforcement/charging practice, more "
        "crashes being self-reported without an officer response, or a reporting-lag "
        "artifact in the most recent year(s) (citations can be filed after the initial "
        "report) -- check whether the drop is concentrated in the latest year(s) before "
        "treating it as a real behavioral trend."
    )

st.markdown("---")
st.markdown("## Driver Distraction Type")
if "DISTRACTION_TYPE" in df.columns and df["DISTRACTION_TYPE"].notna().any():
    dist_df = df[df["DISTRACTION_TYPE"].notna()].copy()
    dist_df = dist_df[~dist_df["DISTRACTION_TYPE"].astype(str).str.lower().str.contains("not distracted")]
    if len(dist_df):
        dist_top_n = st.slider("Number of distraction types to show", 5, 15, 8, key="distraction_topn")
        dist_totals = dist_df["DISTRACTION_TYPE"].value_counts().nlargest(dist_top_n).index
        dsub2 = dist_df[dist_df["DISTRACTION_TYPE"].isin(dist_totals)]
        dm3 = dsub2.groupby(["DISTRACTION_TYPE", "MODE"], observed=True).size().reset_index(name="count")
        fig = px.bar(
            dm3, y="DISTRACTION_TYPE", x="count", color="MODE", orientation="h",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        fig.update_layout(
            yaxis_title=None, xaxis_title="Crashes",
            yaxis={"categoryorder": "total ascending"}, barmode="stack",
        )
        st.plotly_chart(
            style_fig(fig, title=f"Driver Distraction Type (top {dist_top_n}, by mode, excl. 'Not Distracted')", height=440),
            use_container_width=True,
        )
        st.caption(
            "Source: `driver.csv` (`DRIVER_DISTRACTION_CODE`) -- the specific distraction "
            "behind the Distracted Driving flag above. 'Not Distracted' is excluded so the "
            "chart focuses on actual distraction types. When a crash has more than one "
            "driver record, the non-'Not Distracted' code is preferred."
        )
    else:
        st.info("No distraction-type records (excl. 'Not Distracted') in the current filter selection.")
else:
    st.info(
        "No `DISTRACTION_TYPE` column in the loaded export. Re-run "
        "`eda_analysis_combined.py` to pick it up in `power_bi_export.csv`."
    )


render_pipeline_figures("tab4")

