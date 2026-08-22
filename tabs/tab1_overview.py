c1, c2 = st.columns([1, 1.4])

with c1:
    mode_counts = df["MODE"].value_counts().reindex(MODES).fillna(0)
    fig = go.Figure(go.Pie(
        labels=mode_counts.index, values=mode_counts.values, hole=0.55,
        marker=dict(colors=[MODE_COLORS[m] for m in mode_counts.index]),
        textinfo="label+percent", textfont=dict(size=13),
    ))
    fig.add_annotation(text=f"{total:,}<br>crashes", showarrow=False, font=dict(size=15, color="#12172b"))
    st.plotly_chart(style_fig(fig, title="Crash Share by Mode"), use_container_width=True)

with c2:
    annual = df.groupby(["YEAR", "MODE"], observed=True).size().reset_index(name="count")
    fig = px.line(
        annual, x="YEAR", y="count", color="MODE", markers=True,
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
    )
    st.plotly_chart(style_fig(fig, title="Annual Crash Trend by Mode"), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    monthly = df.groupby(["MONTH", "MODE"], observed=True).size().reset_index(name="count")
    monthly["Month"] = monthly["MONTH"].apply(lambda m: MONTH_NAMES[m - 1])
    fig = px.line(
        monthly, x="Month", y="count", color="MODE", markers=True,
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES, "Month": MONTH_NAMES},
    )
    st.plotly_chart(style_fig(fig, title="Seasonal Pattern by Month"), use_container_width=True)

with c4:
    hour_mode = (
        df.groupby(["MODE", "HOUR"], observed=True).size()
        .reset_index(name="count")
        .pivot(index="MODE", columns="HOUR", values="count")
        .reindex(MODES).fillna(0)
    )
    fig = go.Figure(go.Heatmap(
        z=hour_mode.values, x=hour_mode.columns, y=hour_mode.index,
        colorscale="YlOrRd", colorbar=dict(title="Crashes"),
    ))
    st.plotly_chart(style_fig(fig, title="Crashes by Hour of Day"), use_container_width=True)


render_pipeline_figures("tab1")

