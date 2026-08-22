aadt_df = df[df["AVG_AADT"].notna()]
ictrl_df = df[df["INTERSECTION_CONTROL"].notna()]

infra_cols_present = {c: lbl for c, lbl in INFRA_COL_LABELS.items() if c in df.columns and df[c].notna().any()}
missing_infra = [lbl for c, lbl in INFRA_COL_LABELS.items() if c not in infra_cols_present]

note_text = (
    f"Roadway-infrastructure fields only match a subset of crashes via the "
    f"FDOT segment/intersection tables: AADT is available for "
    f"{len(aadt_df):,} of {total:,} filtered crashes "
    f"({len(aadt_df) / total * 100:.0f}%), intersection control type for "
    f"{len(ictrl_df):,} ({len(ictrl_df) / total * 100:.0f}%)."
)
if missing_infra:
    note_text += f" {', '.join(missing_infra)} {'is' if len(missing_infra) == 1 else 'are'} not populated in the current extract, so those charts are omitted."
st.markdown(f"""<div class="section-note">{note_text}</div>""", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    if len(aadt_df):
        fig = px.violin(
            aadt_df, x="MODE", y="AVG_AADT", color="MODE", box=True, points=False,
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        fig.update_layout(yaxis_title="Average Annual Daily Traffic", xaxis_title=None, showlegend=False)
        st.plotly_chart(style_fig(fig, title="Traffic Volume (AADT) by Mode"), use_container_width=True)
    else:
        st.info("No AADT data in the current filter selection.")

with c2:
    if len(ictrl_df):
        ic = ictrl_df.groupby(["INTERSECTION_CONTROL", "MODE"], observed=True).size().reset_index(name="count")
        ictrl_mode_totals = ictrl_df.groupby("MODE", observed=True).size()
        ic["pct"] = ic.apply(lambda r: r["count"] / ictrl_mode_totals.get(r["MODE"], 1) * 100, axis=1)
        fig = px.bar(
            ic, y="INTERSECTION_CONTROL", x="pct", color="MODE", orientation="h", barmode="group",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        fig.update_layout(yaxis_title=None, xaxis_title="% of that mode's crashes (with intersection-control data)",
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_fig(fig, title="Intersection Control Type by Mode (% Within Mode)"), use_container_width=True)
    else:
        st.info("No intersection-control data in the current filter selection.")

if SPEED_COL:
    spd_df = df[pd.to_numeric(df[SPEED_COL], errors="coerce").notna()].copy()
    spd_df[SPEED_COL] = pd.to_numeric(spd_df[SPEED_COL], errors="coerce")
    if len(spd_df):
        fig = px.violin(
            spd_df, x="MODE", y=SPEED_COL, color="MODE", box=True, points=False,
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        fig.update_layout(yaxis_title="Posted Speed Limit (mph)", xaxis_title=None, showlegend=False)
        st.plotly_chart(style_fig(fig, title="Posted Speed Limit Distribution by Mode"), use_container_width=True)
    else:
        st.info("No Posted Speed Limit data in the current filter selection.")

if MICRO_SPEED_COL:
    mspd_df = df[pd.to_numeric(df[MICRO_SPEED_COL], errors="coerce").notna()].copy()
    mspd_df[MICRO_SPEED_COL] = pd.to_numeric(mspd_df[MICRO_SPEED_COL], errors="coerce")
    if len(mspd_df):
        fig = px.violin(
            mspd_df, x="MODE", y=MICRO_SPEED_COL, color="MODE", box=True, points="all",
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
        )
        fig.update_layout(yaxis_title="Self-Reported Speed (mph)", xaxis_title=None, showlegend=False)
        st.plotly_chart(
            style_fig(fig, title=f"Micromobility Speed From Crash Narratives by Mode (n={len(mspd_df):,})"),
            use_container_width=True,
        )
        st.caption(
            "Extracted from the `micromobility_speed` narrative field. Crashes where the "
            "narrative gave no numeric speed (or only '0mph' placeholders) are excluded."
        )
    else:
        total_with_speed = pd.to_numeric(df_raw[MICRO_SPEED_COL], errors="coerce").notna().sum()
        st.info(
            f"No Micromobility Speed data in the current filter selection "
            f"({total_with_speed:,} crash(es) in the full dataset have a value -- "
            f"try widening the sidebar filters, e.g. reset Injury Severity to 'All' or widen the date range)."
        )
elif SPEED_COL:
    # Only mention the missing column once, right next to the sibling chart
    # that DID find its column -- avoids a confusing silent gap here.
    st.caption(
        "Micromobility Speed (self-reported speed from crash narratives) isn't present in "
        "the currently loaded `power_bi_export.csv` -- re-run `eda_analysis_combined.py` and "
        "reload the CSV to pick it up."
    )

st.markdown("---")
st.markdown("## Speed \u00d7 Infrastructure: Does Speed Differ by Where the Rider Was?")
st.caption(
    "Compares against the same infrastructure categories used on the Crash Causation tab "
    "(`infrastructure_type`, LLM-classified from the narrative) rather than a hand-picked "
    "keyword -- so every category the causation tab tracks (travel lane, sidewalk, crosswalk, "
    "bike lane, driveway/parking lot, shoulder, multi-use path) shows up here, not just sidewalk."
)
if cause_raw is None or "infrastructure_type" not in cause_raw.columns:
    st.info(
        f"No `{DEFAULT_CAUSE_PATH}` loaded (or it has no `infrastructure_type` column) -- add it "
        "under the Data Source panel in the sidebar to enable this comparison."
    )
elif not MAIN_CRASH_ID_COL:
    st.info("No crash-ID column found to join speed data to the causation table.")
else:
    cdf9 = cause_raw[cause_raw["MODE"].isin(sel_modes)].copy() if "MODE" in cause_raw.columns else cause_raw.copy()
    if "REPORT_NUMBER" in cdf9.columns:
        cdf9["REPORT_NUMBER"] = cdf9["REPORT_NUMBER"].astype(str)
        cdf9 = cdf9[cdf9["REPORT_NUMBER"].isin(set(df[MAIN_CRASH_ID_COL].astype(str)))]
    cdf9["INFRA_LABEL"] = cdf9["infrastructure_type"].map(INFRA_TYPE_LABELS).fillna(cdf9["infrastructure_type"])

    if len(cdf9) == 0:
        st.info("No causation-classified crashes in the current filter selection.")
    else:
        # ---- Robust view: % speed-flagged by infrastructure type (uses the
        # full causation table, not limited to the ~1% of crashes with a
        # numeric self-reported speed) ----
        if "speed_contributing" in cdf9.columns:
            sc9 = cdf9.groupby("INFRA_LABEL", observed=True).agg(
                n=("speed_contributing", "size"),
                pct_speed=("speed_contributing", lambda s: (s == "yes").mean() * 100),
            ).reset_index().sort_values("pct_speed", ascending=True)
            fig = go.Figure(go.Bar(
                x=sc9["pct_speed"], y=sc9["INFRA_LABEL"], orientation="h",
                marker_color="#5C6BC0",
                text=[f"{v:.0f}%" for v in sc9["pct_speed"]], textposition="outside",
                customdata=sc9[["n"]].values,
                hovertemplate="%{y}: %{x:.1f}%% flagged speed-contributing (n=%{customdata[0]:,})<extra></extra>",
            ))
            fig.update_layout(xaxis_title="% of crashes flagged speed-contributing", yaxis_title=None)
            st.plotly_chart(
                style_fig(
                    fig, title="Speed Flagged as Contributing, by Infrastructure Type (live)",
                    height=360, n=int(sc9["n"].sum()),
                ),
                use_container_width=True,
            )
            st.caption(
                "Either party's speed (driver or rider), from the LLM causation classifier -- "
                "not rider-speed-specific. This uses every causation-classified crash in the "
                "current filter, so it's the more reliable of the two charts here."
            )
            top_row = sc9.iloc[-1]
            bottom_row = sc9.iloc[0]
            insight(
                f"{top_row['INFRA_LABEL']} has the highest speed-contributing flag rate "
                f"({top_row['pct_speed']:.0f}%, n={int(top_row['n']):,}) of the infrastructure "
                f"types tracked here; {bottom_row['INFRA_LABEL']} has the lowest "
                f"({bottom_row['pct_speed']:.0f}%, n={int(bottom_row['n']):,}). Categories with "
                f"small n swing easily -- check the n before treating a specific ranking as solid."
            )

        # ---- Supplementary view: actual mph, where available ----
        if MICRO_SPEED_COL:
            spd9 = df[[MAIN_CRASH_ID_COL, MICRO_SPEED_COL]].copy()
            spd9[MAIN_CRASH_ID_COL] = spd9[MAIN_CRASH_ID_COL].astype(str)
            spd9[MICRO_SPEED_COL] = pd.to_numeric(spd9[MICRO_SPEED_COL], errors="coerce")
            spd9 = spd9.merge(
                cdf9[["REPORT_NUMBER", "INFRA_LABEL"]], left_on=MAIN_CRASH_ID_COL, right_on="REPORT_NUMBER", how="inner",
            )
            spd9 = spd9[spd9[MICRO_SPEED_COL].notna()]
            infra_n = spd9.groupby("INFRA_LABEL", observed=True).size()
            keep_infra = infra_n[infra_n >= 5].index  # drop categories too thin to plot meaningfully
            spd9 = spd9[spd9["INFRA_LABEL"].isin(keep_infra)]

            if len(spd9) < 10 or spd9["INFRA_LABEL"].nunique() < 2:
                st.info(
                    f"Only {len(spd9):,} filtered crashes have both a numeric self-reported "
                    f"speed and a causation-classified infrastructure type with n\u22655 -- too thin "
                    f"for a box plot. The chart above (which doesn't need a numeric speed value) "
                    f"is the more reliable comparison."
                )
            else:
                order = spd9.groupby("INFRA_LABEL", observed=True)[MICRO_SPEED_COL].median().sort_values(ascending=True).index.tolist()
                fig = px.box(
                    spd9, x=MICRO_SPEED_COL, y="INFRA_LABEL", orientation="h", points="outliers",
                    category_orders={"INFRA_LABEL": order}, color_discrete_sequence=["#B71C1C"],
                )
                fig.update_layout(xaxis_title="Self-reported speed (mph)", yaxis_title=None, showlegend=False)
                infra_speed_n = infra_n.reindex(order).fillna(0).astype(int).to_dict()
                st.plotly_chart(
                    style_fig(
                        fig, title="Self-Reported Crash Speed (mph), by Infrastructure Type (live)",
                        height=100 + 60 * len(order), n=infra_speed_n,
                    ),
                    use_container_width=True,
                )
                st.caption(
                    f"Categories with fewer than 5 matched crashes are dropped from this chart. "
                    f"Only {len(spd9):,} of {len(df):,} filtered crashes have both a populated "
                    f"speed value and a causation classification, so treat this as a directional "
                    f"supplement to the chart above, not the primary comparison."
                )
                try:
                    from scipy import stats as _stats9
                    groups = [spd9.loc[spd9["INFRA_LABEL"] == g, MICRO_SPEED_COL] for g in order if infra_speed_n.get(g, 0) >= 5]
                    if len(groups) >= 2:
                        h_stat, p_val9 = _stats9.kruskal(*groups)
                        st.caption(f"Kruskal-Wallis across infrastructure types: H = {h_stat:.2f}, p = {p_val9:.3f}.")
                except ImportError:
                    pass
        else:
            st.caption(
                "No self-reported speed column loaded, so only the speed-contributing-flag "
                "view above is available (it doesn't need a numeric speed value anyway)."
            )

if infra_cols_present:
    infra_items = list(infra_cols_present.items())
    for i in range(0, len(infra_items), 2):
        pair = infra_items[i:i + 2]
        cols = st.columns(2)
        for (col_name, col_label), slot in zip(pair, cols):
            with slot:
                sub = df[df[col_name].notna()]
                if not len(sub):
                    continue
                if pd.api.types.is_numeric_dtype(sub[col_name]):
                    fig = px.violin(
                        sub, x="MODE", y=col_name, color="MODE", box=True, points=False,
                        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
                    )
                    fig.update_layout(yaxis_title=col_label, xaxis_title=None, showlegend=False)
                else:
                    top_vals = sub[col_name].value_counts().nlargest(8).index
                    vt = sub[sub[col_name].isin(top_vals)].groupby([col_name, "MODE"], observed=True).size().reset_index(name="count")
                    fig = px.bar(
                        vt, y=col_name, x="count", color="MODE", orientation="h",
                        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
                    )
                    fig.update_layout(yaxis_title=None, xaxis_title="Crashes",
                                       yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(style_fig(fig, title=f"{col_label} by Mode"), use_container_width=True)

road_type_df = df[df[ROAD_TYPE_COL].notna()] if ROAD_TYPE_COL else pd.DataFrame()
if len(road_type_df):
    st.caption(
        "Road Type reflects the raw FDOT `TRAFFICWAY_CODE` -- the "
        "pipeline doesn't currently map these codes to readable labels."
    )
    rt = road_type_df.groupby([ROAD_TYPE_COL, "MODE"], observed=True).size().reset_index(name="count")
    rt_mode_totals = road_type_df.groupby("MODE", observed=True).size()
    rt["pct"] = rt.apply(lambda r: r["count"] / rt_mode_totals.get(r["MODE"], 1) * 100, axis=1)
    fig = px.bar(
        rt, y=ROAD_TYPE_COL, x="pct", color="MODE", orientation="h", barmode="group",
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
    )
    fig.update_layout(yaxis_title="Trafficway Code", xaxis_title="% of that mode's crashes",
                       yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_fig(fig, title="Road Type (Trafficway Code) by Mode (% Within Mode)"), use_container_width=True)


render_pipeline_figures("tab5")

