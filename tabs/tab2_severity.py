c1, c2 = st.columns([1.3, 1])

with c1:
    sev_mode = (
        df.groupby(["MODE", "S4_CRASH_SEVERITY"], observed=True).size()
        .reset_index(name="count")
    )
    sev_mode["pct"] = sev_mode["count"] / sev_mode.groupby("MODE")["count"].transform("sum") * 100
    fig = px.bar(
        sev_mode, x="MODE", y="pct", color="S4_CRASH_SEVERITY",
        color_discrete_map=SEVERITY_COLORS,
        category_orders={"MODE": MODES, "S4_CRASH_SEVERITY": SEVERITY_ORDER},
        custom_data=["count"],
    )
    fig.update_traces(
        texttemplate="%{y:.0f}%", textposition="inside", textfont=dict(size=10, color="white"),
        hovertemplate="%{fullData.name}: %{y:.1f}%% (n=%{customdata[0]:,})<extra>%{x}</extra>",
    )
    fig.update_layout(yaxis_title="% of crashes", xaxis_title=None, barmode="stack")
    mode_n = df.groupby("MODE", observed=True).size().reindex(MODES).fillna(0).astype(int).to_dict()
    st.plotly_chart(
        style_fig(fig, title="Injury Severity Mix by Mode", n=mode_n), use_container_width=True
    )
    st.caption("Hover a segment for its raw crash count.")

with c2:
    mode_sizes = {m: int((df["MODE"] == m).sum()) for m in MODES}
    fatal_n = {m: int(((df["MODE"] == m) & (df["S4_CRASH_SEVERITY"] == "Fatality")).sum()) for m in MODES}
    serious_n = {m: int(((df["MODE"] == m) & (df["S4_CRASH_SEVERITY"] == "Serious Injury")).sum()) for m in MODES}
    rate_df = pd.DataFrame({
        "MODE": MODES,
        "Fatal %": [fatal_n[m] / mode_sizes[m] * 100 if mode_sizes[m] else 0 for m in MODES],
        "Serious Injury %": [serious_n[m] / mode_sizes[m] * 100 if mode_sizes[m] else 0 for m in MODES],
    })
    fig = go.Figure()
    fig.add_bar(
        name="Fatal %", x=rate_df["MODE"], y=rate_df["Fatal %"], marker_color="#B71C1C",
        text=[f"{v:.2f}%" for v in rate_df["Fatal %"]], textposition="outside",
        customdata=[[fatal_n[m]] for m in rate_df["MODE"]],
        hovertemplate="Fatal: %{y:.2f}%% (n=%{customdata[0]:,})<extra>%{x}</extra>",
    )
    fig.add_bar(
        name="Serious Injury %", x=rate_df["MODE"], y=rate_df["Serious Injury %"], marker_color="#EF9A9A",
        text=[f"{v:.1f}%" for v in rate_df["Serious Injury %"]], textposition="outside",
        customdata=[[serious_n[m]] for m in rate_df["MODE"]],
        hovertemplate="Serious Injury: %{y:.1f}%% (n=%{customdata[0]:,})<extra>%{x}</extra>",
    )
    fig.update_layout(barmode="group", yaxis_title="%")
    st.plotly_chart(
        style_fig(fig, title="Fatal & Serious Injury Rate", n=mode_sizes), use_container_width=True
    )
    st.caption("Hover a bar for its raw numerator count; denominator is that mode's total in the subtitle n=.")

c3, c4 = st.columns(2)
with c3:
    sev_yr = (
        df.groupby(["YEAR", "S4_CRASH_SEVERITY"], observed=True).size()
        .reset_index(name="count")
    )
    # % of that year's crashes, not raw counts -- raw counts conflate the
    # changing severity MIX with the separate fact that overall crash
    # volume also rose/fell year to year.
    sev_yr["pct"] = sev_yr["count"] / sev_yr.groupby("YEAR")["count"].transform("sum") * 100
    fig = px.area(
        sev_yr, x="YEAR", y="pct", color="S4_CRASH_SEVERITY",
        color_discrete_map=SEVERITY_COLORS,
        category_orders={"S4_CRASH_SEVERITY": SEVERITY_ORDER},
        custom_data=["count"],
    )
    fig.update_traces(hovertemplate="%{x}: %{y:.1f}%% (n=%{customdata[0]:,})<extra>%{fullData.name}</extra>")
    fig.update_layout(yaxis_title="% of that year's crashes")
    st.plotly_chart(
        style_fig(fig, title="Severity Mix Over Time (% of Crashes)", n=total),
        use_container_width=True,
    )
    st.caption("Hover a band for the raw crash count behind that year/severity slice.")

with c4:
    mode_sizes4 = {m: int((df["MODE"] == m).sum()) for m in MODES}
    mv_n = {m: int(((df["MODE"] == m) & (df["mv_involved"] == True)).sum()) for m in MODES}  # noqa: E712
    mv_df = df.groupby("MODE", observed=True)["mv_involved"].mean().reindex(MODES).fillna(0) * 100
    fig = go.Figure(go.Bar(
        x=mv_df.index, y=mv_df.values,
        marker_color=[MODE_COLORS[m] for m in mv_df.index],
        text=[f"{v:.0f}%" for v in mv_df.values], textposition="outside",
        customdata=[[mv_n.get(m, 0)] for m in mv_df.index],
        hovertemplate="%{x}: %{y:.1f}%% (n=%{customdata[0]:,})<extra></extra>",
    ))
    fig.update_layout(yaxis_title="% crashes with MV involved", yaxis_range=[0, 110])
    st.plotly_chart(
        style_fig(fig, title="Motor Vehicle Involvement by Mode", n=mode_sizes4), use_container_width=True
    )

if "FARS_LANDUSE" in df.columns:
    fars = df[df["FARS_LANDUSE"].notna()]
    if len(fars):
        urban = (fars["FARS_LANDUSE"] == "Urban").sum()
        rural = (fars["FARS_LANDUSE"] == "Rural").sum()
        st.markdown(
            f"""<div class="section-note">
            <b>FARS federal fatal-crash coding</b> (applies only to the
            {len(fars):,} fatal crashes in the current filter with a FARS
            match): {urban:,} occurred on <b>Urban</b> roads,
            {rural:,} on <b>Rural</b> roads.
            </div>""",
            unsafe_allow_html=True,
        )


render_pipeline_figures("tab2")