st.markdown("## Crash Typing")
if "CRASH_GROUP" in df.columns and df["CRASH_GROUP"].notna().any():
    top_n = st.slider("Number of crash-type groups to show", 5, 20, 12, key="typing_topn")
    grp_totals = df["CRASH_GROUP"].value_counts().nlargest(top_n).index
    gsub = df[df["CRASH_GROUP"].isin(grp_totals)]
    gm = gsub.groupby(["CRASH_GROUP", "MODE"], observed=True).size().reset_index(name="count")
    fig = px.bar(
        gm, y="CRASH_GROUP", x="count", color="MODE", orientation="h",
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
    )
    fig.update_traces(hovertemplate="%{fullData.name}, %{y}: %{x:,} crashes<extra></extra>")
    fig.update_layout(
        yaxis_title=None, xaxis_title="Crashes",
        yaxis={"categoryorder": "total ascending"}, barmode="stack",
    )
    st.plotly_chart(
        style_fig(fig, title=f"Crash Scenario / Type (top {top_n}, by mode)", height=480, n=len(gsub)),
        width="stretch",
    )
    st.caption(
        "Source: `bicycle_typing_20.csv` (`S4_CRASH_GROUP_DESCRIPTION`) -- describes "
        "what happened (e.g. who failed to yield), not road/environment conditions. "
        "Respects every sidebar filter, including the Mode selector."
    )
else:
    st.info(
        "No `CRASH_GROUP` column in the loaded export. Re-run "
        "`eda_analysis_combined.py` to pick it up in `power_bi_export.csv`."
    )

if "CRASH_TYPE_DESC" in df.columns and df["CRASH_TYPE_DESC"].notna().any():
    desc_top_n = st.slider("Number of crash-type descriptions to show", 5, 20, 12, key="typedesc_topn")
    desc_totals = df["CRASH_TYPE_DESC"].value_counts().nlargest(desc_top_n).index
    dsub = df[df["CRASH_TYPE_DESC"].isin(desc_totals)]
    dm2 = dsub.groupby(["CRASH_TYPE_DESC", "MODE"], observed=True).size().reset_index(name="count")
    fig = px.bar(
        dm2, y="CRASH_TYPE_DESC", x="count", color="MODE", orientation="h",
        color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
    )
    fig.update_traces(hovertemplate="%{fullData.name}, %{y}: %{x:,} crashes<extra></extra>")
    fig.update_layout(
        yaxis_title=None, xaxis_title="Crashes",
        yaxis={"categoryorder": "total ascending"}, barmode="stack",
    )
    st.plotly_chart(
        style_fig(fig, title=f"Crash Type Description (top {desc_top_n}, by mode)", height=480, n=len(dsub)),
        width="stretch",
    )
    st.caption(
        "Source: `bicycle_typing_20.csv` (`S4_CRASH_TYPE_DESCRIPTION`) -- the specific "
        "collision mechanics (e.g. right-hook, dooring, overtaking), narrower than the "
        "Crash Scenario / Type grouping above. Bicycle-typed crashes only."
    )

st.markdown("---")
st.markdown("## Contributing Factors")
cf_cols_present = [c for c in ("ROAD_CIRCUMSTANCE", "ENVIRONMENT_CIRCUMSTANCE") if c in df.columns]
if cf_cols_present and df[cf_cols_present].notna().any().any():
    cf_rows = []
    for c in cf_cols_present:
        src_label = "Road" if c == "ROAD_CIRCUMSTANCE" else "Environment"
        vc = df[c].dropna().value_counts()
        for factor, count in vc.items():
            cf_rows.append({"Factor": factor, "Count": count, "Source": src_label})
    cf_df = pd.DataFrame(cf_rows)
    cf_top_n = st.slider("Number of contributing factors to show", 5, 20, 12, key="contrib_topn")
    top_factors = cf_df.groupby("Factor")["Count"].sum().nlargest(cf_top_n).index
    cf_sub = cf_df[cf_df["Factor"].isin(top_factors)]
    fig = px.bar(
        cf_sub, y="Factor", x="Count", color="Source", orientation="h",
        color_discrete_sequence=["#42A5F5", "#FFA726"],
    )
    fig.update_traces(hovertemplate="%{fullData.name}, %{y}: %{x:,} crashes<extra></extra>")
    fig.update_layout(
        yaxis_title=None, xaxis_title="Crashes",
        yaxis={"categoryorder": "total ascending"}, barmode="stack",
    )
    st.plotly_chart(
        style_fig(fig, title=f"Top {cf_top_n} Contributing Factors -- All Active Modes", height=480, n=int(cf_sub["Count"].sum())),
        width="stretch",
    )
    st.caption(
        "Source: `crash_event.csv` (`ROAD_CIRCUMSTANCES_1` + `ENVIRONMENT_CIRCUMSTANCES_1`). "
        "**Road** = roadway circumstances present at the crash (e.g. work zone, debris, "
        "road surface issues) from `ROAD_CIRCUMSTANCE`. **Environment** = environmental "
        "circumstances (e.g. weather-related conditions) from `ENVIRONMENT_CIRCUMSTANCE`. "
        "These are *conditions at the scene*, not FDOT roadway infrastructure class "
        "(Roadway Infrastructure tab) and not the rider’s location type from narrative "
        "causation (Crash Causation tab). Covers Bicycle + E-Bike + E-Scooter and respects "
        "every sidebar filter."
    )
else:
    st.info(
        "No `ROAD_CIRCUMSTANCE`/`ENVIRONMENT_CIRCUMSTANCE` columns in the loaded export. "
        "Re-run `eda_analysis_combined.py` to pick them up in `power_bi_export.csv`."
    )

st.markdown("---")
st.markdown("## Qwen Narrative Classification")
if "IN_QWEN_NARRATIVES" in df.columns:
    qdf = df[df["IN_QWEN_NARRATIVES"] == True].copy()  # noqa: E712
    st.caption(
        f"**{len(qdf):,}** of the **{total:,}** currently filtered crashes have a "
        f"Signal4Data narrative that was run through the Qwen classifier "
        f"({len(qdf) / total * 100:.1f}%). The rest come from the S4_Crash_bicycle "
        f"population directly, or from Signal4Data crashes with no matched narrative."
    )
    if len(qdf) and "QWEN_CLASS" in qdf.columns:
        qc1, qc2 = st.columns(2)
        with qc1:
            qcounts = qdf["QWEN_CLASS"].value_counts()
            fig = go.Figure(go.Pie(
                labels=qcounts.index, values=qcounts.values, hole=0.5,
                textinfo="label+percent",
            ))
            st.plotly_chart(style_fig(fig, title="Qwen Raw Classification", n=len(qdf)), width="stretch")
        with qc2:
            cross = qdf.groupby(["QWEN_CLASS", "MODE"], observed=True).size().reset_index(name="count")
            pivot = cross.pivot(index="QWEN_CLASS", columns="MODE", values="count").fillna(0)
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns, y=pivot.index,
                colorscale="Blues", colorbar=dict(title="Crashes"),
            ))
            st.plotly_chart(
                style_fig(fig, title="Qwen Raw Class vs. Final Mode", n=len(qdf)), width="stretch"
            )
            st.caption(
                "Final Mode can differ from the raw Qwen label -- e.g. a Qwen "
                "'Bicyclist' call gets overridden to E-Bike/E-Scooter if that "
                "REPORT_NUMBER's S4_Crash_bicycle row was overridden (see About tab)."
            )

        if "CRASH_GROUP" in qdf.columns:
            val_df = qdf[["QWEN_CLASS", "CRASH_GROUP", "MODE"]].dropna(subset=["QWEN_CLASS", "CRASH_GROUP"])
            if len(val_df):
                st.markdown("#### Validating Qwen against the manually typed Crash Group")
                with st.expander("Methodology", expanded=False):
                    st.markdown(
                        """
                        Qwen's `QWEN_CLASS` labels are **micromobility mode** read from the
                        narrative (Bicyclist / E-bike / E-scooter / Other). Manual `CRASH_GROUP`
                        (`S4_CRASH_GROUP_DESCRIPTION`) is a **crash-geometry/fault typology**
                        (yield failures, merges, etc.) — the two taxonomies do not share category
                        text, so agreement is measured two ways:

                        1. **Mode κ:** compare `QWEN_CLASS` to final S4 `MODE` via the
                           `MODE_TO_QWEN_CLASS` crosswalk in `dashboard_core.py` (manual mode
                           is the ground truth).
                        2. **Fault κ:** collapse `CRASH_GROUP` to fault-party buckets with
                           `CRASH_GROUP_TO_FAULT_PARTY` and compare to the causation classifier's
                           `ATTRIBUTION` on the same crash (when the causation export is loaded).

                        Rough κ interpretation: <0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate,
                        0.61–0.80 substantial, >0.80 near-perfect.
                        """
                    )
                crosstab = pd.crosstab(val_df["CRASH_GROUP"], val_df["QWEN_CLASS"])
                fig = go.Figure(go.Heatmap(
                    z=crosstab.values, x=crosstab.columns, y=crosstab.index,
                    colorscale="Blues", colorbar=dict(title="Crashes"),
                ))
                st.plotly_chart(
                    style_fig(fig, title="Manual Crash Group vs. Qwen Classification", n=len(val_df),
                              height=max(360, 30 * len(crosstab))),
                    width="stretch",
                )
                qval = compute_qwen_validation_kappa(qdf, cause_raw)
                if qval.get("mode_kappa") is not None:
                    st.caption(
                        f"**Mode Cohen's κ = {qval['mode_kappa']:.2f}** "
                        f"(n={qval['mode_n']:,}) — Qwen class vs final S4 mode "
                        f"(`MODE_TO_QWEN_CLASS` crosswalk)."
                    )
                if qval.get("fault_kappa") is not None:
                    st.caption(
                        f"**Fault Cohen's κ = {qval['fault_kappa']:.2f}** "
                        f"(n={qval['fault_n']:,}) — manual `CRASH_GROUP` fault party vs "
                        f"causation `ATTRIBUTION`."
                    )
                with st.expander("Crosswalk tables (manual mappings)", expanded=False):
                    st.markdown("**Qwen mode ↔ S4 MODE**")
                    st.dataframe(
                        pd.DataFrame(
                            [{"S4 MODE": k, "Qwen class": v} for k, v in MODE_TO_QWEN_CLASS.items()]
                        ),
                        hide_index=True, width="stretch",
                    )
                    st.markdown("**CRASH_GROUP → fault party** (aligned with causation `ATTRIBUTION`)")
                    st.dataframe(
                        pd.DataFrame(
                            [{"CRASH_GROUP": k, "Fault party": v}
                             for k, v in sorted(CRASH_GROUP_TO_FAULT_PARTY.items())]
                        ),
                        hide_index=True, width="stretch",
                    )
    elif len(qdf) == 0:
        st.info("No narrative-classified crashes in the current filter selection.")
else:
    st.info(
        "No `QWEN_CLASS`/`IN_QWEN_NARRATIVES` columns in the loaded export. "
        "Re-run `eda_analysis_combined.py` to pick them up."
    )

st.markdown("---")
st.markdown("## Narrative Text Mining")
if narrative_raw is not None and "NARRATIVE_TEXT" in narrative_raw.columns:
    filtered_rns = set(df[MAIN_CRASH_ID_COL].astype(str)) if MAIN_CRASH_ID_COL else None
    ntext = narrative_raw.copy()
    if filtered_rns is not None:
        ntext = ntext[ntext["REPORT_NUMBER"].isin(filtered_rns)]
    text_mode_col = "MODE" if "MODE" in ntext.columns else ("QWEN_MODE" if "QWEN_MODE" in ntext.columns else None)
    if text_mode_col:
        ntext = ntext[ntext[text_mode_col].isin(sel_modes)]

    st.caption(
        f"**{len(ntext):,}** narratives match the current sidebar filters "
        f"(Mode, Year, Severity, etc. all apply here too)."
    )

    if len(ntext):
        default_keywords = "phone,texting,helmet,alcohol,dark,sidewalk,crosswalk,wrong way,speeding,failed to yield,intoxicated,fled"
        kw_input = st.text_input(
            "Keywords to search (comma-separated) -- edit freely and the chart updates live",
            value=default_keywords, key="keyword_search_input",
        )
        keywords = [k.strip().lower() for k in kw_input.split(",") if k.strip()]

        if keywords and text_mode_col:
            ntext_kw = ntext.copy()
            ntext_kw["_SEARCH_TEXT"] = ntext_kw["NARRATIVE_TEXT"].map(narrative_text_for_search)
            rows = []
            for m in [mm for mm in MODES if mm in ntext[text_mode_col].unique()]:
                sub_txt = ntext_kw[ntext_kw[text_mode_col] == m]["_SEARCH_TEXT"]
                n = len(sub_txt)
                for kw in keywords:
                    pct = sub_txt.str.contains(re.escape(kw), case=False, na=False).mean() * 100 if n else 0
                    rows.append({"Mode": m, "Keyword": kw, "Pct": pct, "N": n})
            kdf = pd.DataFrame(rows)
            if len(kdf):
                pivot = kdf.pivot(index="Keyword", columns="Mode", values="Pct").reindex(
                    columns=[m for m in MODES if m in kdf["Mode"].unique()]
                )
                fig = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns, y=pivot.index,
                    colorscale="YlOrRd", colorbar=dict(title="% of narratives"),
                    text=np.round(pivot.values, 1), texttemplate="%{text}",
                ))
                st.plotly_chart(
                    style_fig(fig, title="Keyword Mentions (% of narratives) by Mode", height=max(320, 34 * len(keywords)), n=len(ntext)),
                    width="stretch",
                )
                st.caption(
                    "Keyword search runs on narrative text **after stripping officer/report "
                    "header boilerplate** (e.g. badge/agency phone-number blocks), so "
                    "matches reflect crash content rather than report metadata."
                )

        st.markdown("#### Topic modeling (unsupervised)")
        with st.expander("Why add this on top of keyword search", expanded=False):
            st.markdown(
                """
                The keyword heatmap above only finds what you already thought to search for.
                **Latent Dirichlet Allocation (LDA)** instead looks at word co-occurrence patterns
                across all narratives and surfaces recurring themes automatically -- useful for
                catching scenario patterns you didn't think to search for. Each topic below is a
                cluster of words that tend to appear together; read the word list as a loose theme
                label, not a precise category, and re-run with a different topic count if topics
                look too broad (too few topics) or too fragmented (too many).
                """
            )
        if text_mode_col:
            topic_mode_options = [m for m in MODES if m in ntext[text_mode_col].unique()]
        else:
            topic_mode_options = []
        if topic_mode_options:
            tcol1, tcol2 = st.columns([1, 3])
            with tcol1:
                topic_mode = st.selectbox("Mode to model", topic_mode_options, key="lda_mode_select")
                n_topics = st.slider("Number of topics", 3, 10, 5, key="lda_n_topics")
            topic_texts = (
                ntext[ntext[text_mode_col] == topic_mode]["NARRATIVE_TEXT"]
                .dropna().astype(str).map(narrative_text_for_search)
            )
            min_needed = max(50, n_topics * 10)
            if len(topic_texts) >= min_needed:
                try:
                    from sklearn.feature_extraction.text import CountVectorizer
                    from sklearn.decomposition import LatentDirichletAllocation
                    vec = CountVectorizer(
                        stop_words=lda_stopword_list(), max_features=1000, min_df=5, max_df=0.6,
                    )
                    Xc = vec.fit_transform(topic_texts)
                    lda = LatentDirichletAllocation(
                        n_components=n_topics, random_state=0, max_iter=15, learning_method="online",
                    )
                    doc_topic = lda.fit_transform(Xc)
                    words = np.array(vec.get_feature_names_out())
                    topic_share = doc_topic.sum(axis=0) / doc_topic.sum()
                    n_disp_cols = min(n_topics, 5)
                    topic_cols = st.columns(n_disp_cols)
                    # Rank topics highest → lowest share of narratives
                    topic_order = np.argsort(topic_share)[::-1]
                    for rank, ti in enumerate(topic_order):
                        top_idx = np.argsort(lda.components_[ti])[::-1][:8]
                        top_words = words[top_idx]
                        with topic_cols[rank % n_disp_cols]:
                            st.markdown(
                                f"**Topic {rank + 1}** ({topic_share[ti] * 100:.0f}% of narratives)"
                            )
                            st.caption(", ".join(top_words))
                    st.caption(
                        f"LDA fit on {len(topic_texts):,} {topic_mode} narratives ({n_topics} topics, "
                        f"unigrams, report boilerplate stripped + extended stopword list, terms in "
                        f"fewer than 5 narratives or more than 60% of narratives excluded)."
                    )
                except Exception as e:
                    st.warning(f"Topic model failed to fit: {e}")
            else:
                st.info(
                    f"Only {len(topic_texts)} {topic_mode} narratives match the current filters -- "
                    f"need at least {min_needed} for a stable {n_topics}-topic model. Try fewer "
                    f"topics or a mode/filter combination with more narratives."
                )

        st.markdown("#### Top words by mode")
        st.caption("Reflects the Mode filter in the sidebar.")
        STOPWORDS = set((
            "the a an and or of to in on at for with was were is are be been being this that "
            "it its he she they them his her their who was driver vehicle crash report "
            "not no did do does had have has as by from into out up down "
            "1 2 3 4 5 6 7 8 9 0"
        ).split())
        word_modes = [m for m in MODES if text_mode_col and m in ntext[text_mode_col].unique()]
        if word_modes:
            word_cols = st.columns(len(word_modes))
            for wmode, wcol in zip(word_modes, word_cols):
                sub_txt = ntext[ntext[text_mode_col] == wmode]["NARRATIVE_TEXT"]
                words = re.findall(r"[a-z']{3,}", " ".join(sub_txt.tolist()))
                words = [w for w in words if w not in STOPWORDS]
                top_words = Counter(words).most_common(20)
                with wcol:
                    if top_words:
                        wdf = pd.DataFrame(top_words, columns=["word", "count"])
                        fig = px.bar(wdf.sort_values("count"), x="count", y="word", orientation="h")
                        fig.update_layout(yaxis_title=None, xaxis_title="Mentions")
                        st.plotly_chart(
                            style_fig(fig, title=f"Top 20 Words -- {wmode}", height=460, n=len(sub_txt)),
                            width="stretch",
                        )
                    else:
                        st.info(f"No words for {wmode}.")
    else:
        st.info("No narratives match the current filter selection.")
else:
    st.markdown(
        f"""<div class="section-note">
        No <code>{DEFAULT_NARRATIVE_PATH}</code> loaded, so the interactive keyword
        tool isn't available -- add it under the <b>Data Source</b> panel in the
        sidebar. It's produced by <code>eda_analysis_combined.py</code> alongside
        <code>power_bi_export.csv</code>.
        </div>""",
        unsafe_allow_html=True,
    )

render_pipeline_figures("tab7")
