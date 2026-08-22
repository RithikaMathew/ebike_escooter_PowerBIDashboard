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
    fig.update_layout(
        yaxis_title=None, xaxis_title="Crashes",
        yaxis={"categoryorder": "total ascending"}, barmode="stack",
    )
    st.plotly_chart(
        style_fig(fig, title=f"Crash Scenario / Type (top {top_n}, by mode)", height=480),
        use_container_width=True,
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
    fig.update_layout(
        yaxis_title=None, xaxis_title="Crashes",
        yaxis={"categoryorder": "total ascending"}, barmode="stack",
    )
    st.plotly_chart(
        style_fig(fig, title=f"Crash Type Description (top {desc_top_n}, by mode)", height=480),
        use_container_width=True,
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
    fig.update_layout(
        yaxis_title=None, xaxis_title="Crashes",
        yaxis={"categoryorder": "total ascending"}, barmode="stack",
    )
    st.plotly_chart(
        style_fig(fig, title=f"Top {cf_top_n} Contributing Factors -- All Active Modes", height=480),
        use_container_width=True,
    )
    st.caption(
        "Source: `crash_event.csv` (`ROAD_CIRCUMSTANCES_1` + `ENVIRONMENT_CIRCUMSTANCES_1`) -- "
        "road/environment CONDITIONS present at the crash, covering Bicycle + E-Bike + "
        "E-Scooter. Different from the crash-typing charts above, which show what happened "
        "rather than the conditions it happened under. Respects every sidebar filter."
    )
else:
    st.info(
        "No `ROAD_CIRCUMSTANCE`/`ENVIRONMENT_CIRCUMSTANCE` columns in the loaded export. "
        "Re-run `eda_analysis_combined.py` to pick them up in `power_bi_export.csv`."
    )

st.markdown("---")
st.markdown("## Severity Risk Factors")
SEVERITY_CANDIDATES = [
    "SEVERITY", "INJURY_SEVERITY", "CRASH_SEVERITY", "MOST_SEVERE_INJURY",
    "HIGHEST_INJURY_SEVERITY_DESC", "INJURY_SEVERITY_DESC", "INJSEVER",
]
severity_col = next((c for c in SEVERITY_CANDIDATES if c in df.columns and df[c].notna().any()), None)
if severity_col is None:
    st.info(
        f"No severity column found under any of the usual names "
        f"({', '.join(SEVERITY_CANDIDATES)}). If your export uses a different column name for "
        f"crash/injury severity, point me at it and I'll wire this section up to it."
    )
else:
    with st.expander("Methodology", expanded=False):
        st.markdown(
            """
            This fits a **binary logistic regression** predicting whether a crash falls into the
            "severe" categories you pick below, using the road/environment/timing factors shown
            in the charts above -- but jointly, so each factor's effect controls for the others
            (e.g. "is fog actually associated with severity, or do fog crashes just also happen
            to skew toward night?"). Results are reported as **odds ratios**: an OR of 2.0 means
            that factor is associated with roughly double the odds of a severe outcome, holding
            everything else in the model fixed. A **random forest permutation importance** is
            shown alongside as a model-free cross-check, since logistic regression assumes each
            factor's effect is additive on the log-odds scale, which may not actually hold.
            """
        )
    sev_values = sorted(df[severity_col].dropna().astype(str).unique().tolist())
    default_severe = [
        v for v in sev_values if any(k in v.lower() for k in ("fatal", "incapacitat", "severe", "kill"))
    ]
    severe_values = st.multiselect(
        f"Which `{severity_col}` categories count as \"severe\" for this model?",
        sev_values, default=default_severe or sev_values[:1], key="severity_model_severe_values",
    )
    candidate_predictors = [
        c for c in ("DAY_NIGHT", "LOC_TYPE", "LIGHT_CONDITION", "WEATHER_CONDITION",
                     "ROAD_CIRCUMSTANCE", "ENVIRONMENT_CIRCUMSTANCE", "MODE")
        if c in df.columns
    ]
    predictors = st.multiselect(
        "Predictors to include", candidate_predictors, default=candidate_predictors,
        key="severity_model_predictors",
    )
    if severe_values and predictors:
        model_df = df[predictors + [severity_col]].dropna().copy()
        model_df[severity_col] = model_df[severity_col].astype(str)
        model_df["SEVERE_OUTCOME"] = model_df[severity_col].isin(severe_values).astype(int)
        if model_df["SEVERE_OUTCOME"].nunique() < 2:
            st.warning(
                "The selected severe categories produce no variation (every crash is/isn't "
                "severe) -- pick a different split."
            )
        elif len(model_df) < 50:
            st.warning(f"Only {len(model_df)} complete rows for the selected columns -- too few to fit a reliable model.")
        else:
            try:
                import statsmodels.api as sm
                X = pd.get_dummies(model_df[predictors], drop_first=True)
                X = sm.add_constant(X).astype(float)
                y = model_df["SEVERE_OUTCOME"].astype(float)
                logit_res = sm.Logit(y, X).fit(disp=0, maxiter=100)
                or_table = pd.DataFrame({
                    "Odds Ratio": np.exp(logit_res.params),
                    "CI low": np.exp(logit_res.conf_int()[0]),
                    "CI high": np.exp(logit_res.conf_int()[1]),
                    "p-value": logit_res.pvalues,
                })
                if "const" in or_table.index:
                    or_table = or_table.drop("const")
                or_table = or_table.sort_values("Odds Ratio", ascending=True)
                or_table["Significant (p<0.05)"] = or_table["p-value"] < 0.05

                sc1, sc2 = st.columns(2)
                with sc1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=or_table["Odds Ratio"], y=or_table.index, mode="markers",
                        marker=dict(size=9, color=np.where(or_table["Significant (p<0.05)"], "#c62828", "#9e9e9e")),
                        error_x=dict(
                            type="data", symmetric=False,
                            array=(or_table["CI high"] - or_table["Odds Ratio"]).clip(lower=0),
                            arrayminus=(or_table["Odds Ratio"] - or_table["CI low"]).clip(lower=0),
                        ),
                    ))
                    fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
                    fig.update_layout(xaxis_title="Odds ratio (log scale, dashed line = no effect)", xaxis_type="log", yaxis_title=None)
                    st.plotly_chart(
                        style_fig(fig, title="Logistic Regression Odds Ratios (95% CI)", height=max(360, 32 * len(or_table))),
                        use_container_width=True,
                    )
                with sc2:
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.inspection import permutation_importance
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import roc_auc_score
                    Xrf = X.drop(columns="const")
                    strat = y if y.value_counts().min() >= 2 else None
                    Xtr, Xte, ytr, yte = train_test_split(Xrf, y, test_size=0.3, random_state=0, stratify=strat)
                    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=0, class_weight="balanced")
                    rf.fit(Xtr, ytr)
                    if yte.nunique() >= 2:
                        perm = permutation_importance(rf, Xte, yte, n_repeats=15, random_state=0, scoring="roc_auc")
                        test_auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
                        imp_df = pd.DataFrame({"Feature": Xtr.columns, "Importance": perm.importances_mean}).sort_values("Importance")
                        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h")
                        fig.update_layout(yaxis_title=None, xaxis_title="Permutation importance (drop in AUC)")
                        st.plotly_chart(
                            style_fig(fig, title=f"Random Forest Cross-Check (test AUC={test_auc:.2f})", height=max(360, 32 * len(imp_df))),
                            use_container_width=True,
                        )
                    else:
                        st.info("Test split ended up with only one outcome class -- try a larger dataset or different severe-category split.")
                st.caption(
                    f"n={len(model_df):,} crashes with complete data on the selected predictors; "
                    f"**{int(model_df['SEVERE_OUTCOME'].sum()):,}** ({model_df['SEVERE_OUTCOME'].mean() * 100:.1f}%) "
                    f"classified as severe under the current category selection. Red points in the left "
                    f"chart are statistically significant (p<0.05) after controlling for every other "
                    f"factor in the model; gray points are not. Right chart bars are dummy-encoded "
                    f"categories (e.g. `WEATHER_CONDITION_Fog` = Fog vs. the reference category), "
                    f"matching the logistic regression's encoding -- AUC of 0.5 = no better than chance, "
                    f"1.0 = perfect separation."
                )
            except Exception as e:
                st.warning(f"Severity model failed to fit: {e}")

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
            st.plotly_chart(style_fig(fig, title="Qwen Raw Classification"), use_container_width=True)
        with qc2:
            cross = qdf.groupby(["QWEN_CLASS", "MODE"], observed=True).size().reset_index(name="count")
            pivot = cross.pivot(index="QWEN_CLASS", columns="MODE", values="count").fillna(0)
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns, y=pivot.index,
                colorscale="Blues", colorbar=dict(title="Crashes"),
            ))
            st.plotly_chart(
                style_fig(fig, title="Qwen Raw Class vs. Final Mode"), use_container_width=True
            )
            st.caption(
                "Final Mode can differ from the raw Qwen label -- e.g. a Qwen "
                "'Bicyclist' call gets overridden to E-Bike/E-Scooter if that "
                "REPORT_NUMBER's S4_Crash_bicycle row was overridden (see About tab)."
            )

        if "CRASH_GROUP" in qdf.columns:
            val_df = qdf[["QWEN_CLASS", "CRASH_GROUP"]].dropna()
            if len(val_df):
                st.markdown("#### Validating Qwen against the manually typed Crash Group")
                with st.expander("Methodology", expanded=False):
                    st.markdown(
                        """
                        The Qwen classification above is used throughout this dashboard, but its
                        accuracy has never been checked against a ground truth. `CRASH_GROUP`
                        (`S4_CRASH_GROUP_DESCRIPTION`) is the closest thing available -- a
                        separately, manually typed label for the same crash. This crosstab and
                        **Cohen's kappa** compare the two directly. Kappa accounts for the
                        agreement you'd expect by chance alone (unlike raw % agreement) and only
                        gets computed on rows where both labels use literally the same category
                        text, since kappa isn't meaningful across two different taxonomies without
                        a manual crosswalk between them. Rough interpretation: <0.20 slight,
                        0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial, >0.80 near-perfect.
                        """
                    )
                crosstab = pd.crosstab(val_df["CRASH_GROUP"], val_df["QWEN_CLASS"])
                fig = go.Figure(go.Heatmap(
                    z=crosstab.values, x=crosstab.columns, y=crosstab.index,
                    colorscale="Blues", colorbar=dict(title="Crashes"),
                ))
                st.plotly_chart(
                    style_fig(fig, title="Manual Crash Group vs. Qwen Classification",
                              height=max(360, 30 * len(crosstab))),
                    use_container_width=True,
                )
                qwen_cats = set(val_df["QWEN_CLASS"].str.strip().str.lower())
                manual_cats = set(val_df["CRASH_GROUP"].str.strip().str.lower())
                overlap = qwen_cats & manual_cats
                if overlap:
                    common_mask = (
                        val_df["QWEN_CLASS"].str.strip().str.lower().isin(overlap)
                        & val_df["CRASH_GROUP"].str.strip().str.lower().isin(overlap)
                    )
                    common_df = val_df[common_mask]
                    if len(common_df) >= 10:
                        try:
                            from sklearn.metrics import cohen_kappa_score
                            kappa = cohen_kappa_score(
                                common_df["CRASH_GROUP"].str.strip().str.lower(),
                                common_df["QWEN_CLASS"].str.strip().str.lower(),
                            )
                            st.caption(
                                f"**Cohen's kappa = {kappa:.2f}**, computed on the "
                                f"**{len(common_df):,}** crashes (of {len(val_df):,} total) where "
                                f"Qwen's label and the manual `CRASH_GROUP` label use directly "
                                f"matching category text ({len(overlap)} shared category names out "
                                f"of {len(qwen_cats)} Qwen / {len(manual_cats)} manual categories)."
                            )
                        except ImportError:
                            st.info("`scikit-learn` is needed to compute Cohen's kappa.")
                    else:
                        st.caption(
                            f"Only {len(common_df)} crashes have exactly matching category text "
                            f"between the two taxonomies -- too few for a reliable kappa. The "
                            f"heatmap above is still informative for eyeballing correspondence."
                        )
                else:
                    st.caption(
                        "Qwen's category names and `CRASH_GROUP`'s category names don't share any "
                        "identical text, so Cohen's kappa can't be computed directly -- a real "
                        "validation would need a manual mapping table between the two taxonomies. "
                        "The heatmap above still shows the raw correspondence."
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
            rows = []
            for m in [mm for mm in MODES if mm in ntext[text_mode_col].unique()]:
                sub_txt = ntext[ntext[text_mode_col] == m]["NARRATIVE_TEXT"]
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
                    style_fig(fig, title="Keyword Mentions (% of narratives) by Mode", height=max(320, 34 * len(keywords))),
                    use_container_width=True,
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
            topic_texts = ntext[ntext[text_mode_col] == topic_mode]["NARRATIVE_TEXT"].dropna().astype(str)
            min_needed = max(50, n_topics * 10)
            if len(topic_texts) >= min_needed:
                try:
                    from sklearn.feature_extraction.text import CountVectorizer
                    from sklearn.decomposition import LatentDirichletAllocation
                    vec = CountVectorizer(stop_words="english", max_features=1000, min_df=5, max_df=0.6)
                    Xc = vec.fit_transform(topic_texts)
                    lda = LatentDirichletAllocation(
                        n_components=n_topics, random_state=0, max_iter=15, learning_method="online",
                    )
                    doc_topic = lda.fit_transform(Xc)
                    words = np.array(vec.get_feature_names_out())
                    topic_share = doc_topic.sum(axis=0) / doc_topic.sum()
                    n_disp_cols = min(n_topics, 5)
                    topic_cols = st.columns(n_disp_cols)
                    for ti in range(n_topics):
                        top_idx = np.argsort(lda.components_[ti])[::-1][:8]
                        top_words = words[top_idx]
                        with topic_cols[ti % n_disp_cols]:
                            st.markdown(f"**Topic {ti + 1}** ({topic_share[ti] * 100:.0f}% of narratives)")
                            st.caption(", ".join(top_words))
                    st.caption(
                        f"LDA fit on {len(topic_texts):,} {topic_mode} narratives ({n_topics} topics, "
                        f"unigrams, English stopwords removed, terms in fewer than 5 narratives or "
                        f"more than 60% of narratives excluded)."
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
                            style_fig(fig, title=f"Top 20 Words -- {wmode} (n={len(sub_txt):,})", height=460),
                            use_container_width=True,
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

st.markdown("---")
st.markdown("## Spatiotemporal Hotspot Clusters")
if hotspot_raw is not None and "MODE" in hotspot_raw.columns:
    hs = hotspot_raw[hotspot_raw["MODE"].isin(sel_modes)].copy()

    has_periods = {"N_EARLY_PERIOD", "N_LATE_PERIOD"}.issubset(hs.columns)
    if has_periods:
        with st.expander("Why there's a statistical test here now, not just the 1.5x heuristic", expanded=False):
            st.markdown(
                """
                "Late-period count > 1.5x early-period count" flags noise as often as a real
                trend when counts are small -- a cluster going from 2 crashes to 4 clears that
                bar but could easily be chance. This adds a **two-sample Poisson rate-ratio
                test** (an exact binomial test comparing the two period counts, assuming equal
                period lengths): it asks "if this cluster's true rate hadn't changed, how likely
                is a split this lopsided?" and gives you a p-value instead of an arbitrary
                multiplier. A cluster can pass the old 1.5x heuristic and still not be
                statistically significant if both counts are small -- that's the point.
                """
            )
        from scipy import stats as _stats

        def _rate_ratio_test(n1, n2):
            n_total = n1 + n2
            if n_total == 0 or pd.isna(n1) or pd.isna(n2):
                return np.nan, np.nan
            pval = _stats.binomtest(int(n2), int(n_total), 0.5, alternative="two-sided").pvalue
            rr = (n2 / n1) if n1 > 0 else np.inf
            return rr, pval

        _rr = hs.apply(lambda r: _rate_ratio_test(r["N_EARLY_PERIOD"], r["N_LATE_PERIOD"]), axis=1)
        hs["RATE_RATIO"] = [x[0] for x in _rr]
        hs["GROWTH_PVAL"] = [x[1] for x in _rr]
        hs["SIG_GROWTH"] = hs["GROWTH_PVAL"] < 0.05

        filter_options = ["All clusters", "Emerging (heuristic: >1.5x growth)",
                           "Statistically significant growth (Poisson rate-ratio test, p<0.05)"]
    else:
        filter_options = ["All clusters", "Emerging (heuristic: >1.5x growth)"]

    cluster_filter = st.selectbox("Filter clusters", filter_options, index=0, key="hotspot_cluster_filter")
    if cluster_filter == "Emerging (heuristic: >1.5x growth)" and "EMERGING" in hs.columns:
        hs = hs[hs["EMERGING"] == True]  # noqa: E712
    elif cluster_filter.startswith("Statistically significant") and has_periods:
        hs = hs[hs["SIG_GROWTH"] == True]  # noqa: E712

    st.caption(
        f"**{len(hs):,}** clusters match the Mode filter in the sidebar "
        f"(hotspot table isn't affected by Year/Severity/etc. filters -- it's "
        f"precomputed per mode over the full time range)."
    )
    if len(hs) and {"CENTER_LAT", "CENTER_LON"}.issubset(hs.columns):
        hover_cols = ["CLUSTER_ID", "N_CRASHES", "N_EARLY_PERIOD", "N_LATE_PERIOD", "GROWTH_RATIO"]
        if has_periods:
            hover_cols += ["RATE_RATIO", "GROWTH_PVAL"]
        fig = px.scatter_mapbox(
            hs, lat="CENTER_LAT", lon="CENTER_LON", color="MODE",
            size="N_CRASHES", size_max=28,
            color_discrete_map=MODE_COLORS, category_orders={"MODE": MODES},
            hover_data=[c for c in hover_cols if c in hs.columns],
            zoom=5.4, height=560,
        )
        fig = style_fig(fig, height=560, title="Cluster Centers (bubble size = crashes in cluster)")
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=56, b=0))
        st.plotly_chart(fig, use_container_width=True)

        sort_col = "GROWTH_PVAL" if has_periods else ("GROWTH_RATIO" if "GROWTH_RATIO" in hs.columns else hs.columns[0])
        ascending = sort_col == "GROWTH_PVAL"
        display_hs = hs.sort_values(sort_col, ascending=ascending).copy()
        if has_periods:
            display_hs = display_hs.rename(columns={"RATE_RATIO": "Rate ratio (late/early)", "GROWTH_PVAL": "p-value"})
            display_hs = display_hs.round({"Rate ratio (late/early)": 2, "p-value": 4})
        st.dataframe(display_hs, use_container_width=True, hide_index=True)
    else:
        st.info("No clusters match the current Mode selection.")
    st.caption(
        "Exploratory DBSCAN clustering (see `eda_analysis_combined.py` section 09d) -- "
        "not a validated hotspot-detection pipeline. 'Emerging' (heuristic) = late-period "
        "count > 1.5x early-period count, with at least 5 late-period crashes. "
        + ("'Statistically significant growth' = Poisson rate-ratio test p<0.05, assuming "
           "equal-length early/late periods." if has_periods else "")
    )
else:
    st.markdown(
        f"""<div class="section-note">
        No <code>{DEFAULT_HOTSPOT_PATH}</code> loaded, so the interactive hotspot
        explorer isn't available -- add it under the <b>Data Source</b> panel in the
        sidebar.
        </div>""",
        unsafe_allow_html=True,
    )

render_pipeline_figures("tab7")