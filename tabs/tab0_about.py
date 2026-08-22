st.markdown("## What this dashboard shows")
st.markdown(
    """
This dashboard tracks **crashes involving people on bicycles, e-bikes, and
e-scooters** ("active-mode" or "micromobility" road users) in Florida,
built from **the S4_Crash_bicycle population** (Bicycle) plus **Signal4
crash reports** (E-Bike, E-Scooter, Other) matched against **FDOT roadway
tables**. It's produced by the Just & Green Transportation Lab at the
University of Florida as part of a Complete Streets road-safety analysis.

**Scope:** every chart here is scoped to the three active modes -- Bicycle,
E-Bike, E-Scooter. Everything else (pedestrian-only, motor-vehicle-only,
single-vehicle, animal, etc. -- labeled "Other" in the source pipeline) is
excluded from the dashboard entirely, not just hidden by a default filter.

**Why e-bikes and e-scooters get special attention:** these are the two
fastest-growing and least-studied categories of active-mode travel.
Traditional crash datasets and roadway-safety literature were built around
conventional bicycles and pedestrians; e-bikes and e-scooters ride at
higher speeds, on different infrastructure (sidewalks, bike lanes,
shared-use paths), and are involved in crash patterns that don't map
cleanly onto older bicycle-safety research. Understanding how their crash
profile differs from conventional bicycles is central to this project.
    """
)

st.markdown("## How each mode is classified")
st.markdown(
    """
- **Bicycle** -- sourced from **S4_Crash_bicycle**, a larger, dedicated
  bicycle-crash population (bigger N than relying on Signal4Data alone).
  Every row in that source is already a bicycle crash, so no structural
  crash-form code is needed to identify it. The only exception: if Qwen's
  narrative classifier says a specific crash is actually **E-Bike** or
  **E-Scooter**, that overrides the Bicycle label (see below).
- **E-Bike / E-Scooter** -- identified **exclusively** through Qwen LLM
  narrative classification of Signal4Data crash narratives. Florida crash
  forms have no structural code for either, so if the narrative wasn't
  classified, the crash can't be identified as E-Bike/E-Scooter. This Qwen
  narrative call is authoritative even when a crash also shows up in the
  S4_Crash_bicycle population.
    """
)

st.markdown("## Why the numbers don't match the raw crash count")
st.markdown(
    """
This dashboard's active-mode export is built from **two source systems**,
not one flat pull, so a raw report-number count from either source alone
won't match what you see here:
    """
)
st.markdown(
    """
- **S4_Crash_bicycle** -- a dedicated, larger bicycle-crash population.
  Every record here is a bicycle crash; the only thing that can move a
  record out of Bicycle is a Qwen narrative override to E-Bike/E-Scooter.
- **Signal4Data** -- a separate crash-report pull covering *all* crash
  types (bicycle, pedestrian, single-vehicle, MV-only, animal, etc). It
  supplies E-Bike, E-Scooter, and everything classified **Other**.
- **Overlap** -- the same physical crash can legitimately appear in both
  source systems. When a REPORT_NUMBER shows up in both, it's counted
  **once**: Qwen's E-Bike/E-Scooter call wins if present, otherwise it's
  counted as Bicycle from S4_Crash_bicycle. Nothing is double-counted.
- **Final export** -- whatever's left after mode classification is what's
  loaded into this dashboard right now.
    """
)

st.markdown("### What actually happens at the classification step")
st.markdown(
    """
Every crash in the combined crash_event table runs through one
classification rule, in this order:

1. **Qwen narrative says E-Bike or E-Scooter** (from `multilabel_ebike.xlsx`
   + `multilabel_RegBike.xlsx`, Signal4Data narratives) -- authoritative,
   even if that REPORT_NUMBER is also present in S4_Crash_bicycle.
2. **REPORT_NUMBER is in S4_Crash_bicycle** (and wasn't overridden above)
   &rarr; classified as **Bicycle**.
3. **Everything else &rarr; "Other"**, including pedestrian-only,
   single-vehicle, and motor-vehicle-only crashes from Signal4Data, plus
   non-motorist records coded `'Other Cyclist'` -- these are deliberately
   **not** guessed into E-Bike or E-Scooter, since that code also covers
   unicycles, tricycles, cargo bikes, and para-cycles, and there isn't
   enough evidence to know which.

So the gap between the combined **crash_event** total and this dashboard's
**active-mode export** isn't missing or lost data -- it's every crash that
landed in "Other" at this step, mostly pedestrian-only and MV-only crashes
from Signal4Data that were never bicycle/e-bike/e-scooter to begin with.
E-Bike and E-Scooter can **only** come from an explicit Qwen narrative
match -- there's no structural fallback for those two modes, since Florida
crash forms have no dedicated code for either.
    """
)

if meta_raw is not None:
    metric_col = find_col(meta_raw, ["metric"]) or meta_raw.columns[0]
    value_col = find_col(meta_raw, ["value"]) or (meta_raw.columns[1] if len(meta_raw.columns) > 1 else meta_raw.columns[0])
    note_col = find_col(meta_raw, ["note"])

    m = meta_raw.set_index(metric_col)[value_col]

    # Two source inputs merge into crash_event now (S4_Crash_bicycle +
    # Signal4Data), so they're shown as parallel KPI cards rather than
    # forced into a single linear funnel -- a funnel implies each step
    # narrows the one before it, which no longer holds once two separate
    # source populations combine.
    source_metrics = [s for s in ["bicycle_population_s4_crash_bicycle", "query_params_report_numbers"] if s in m.index]
    if source_metrics:
        st.markdown("### Two source populations")
        sc = st.columns(len(source_metrics))
        for col, s in zip(sc, source_metrics):
            col.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-label">{META_METRIC_LABELS.get(s, s)}</div>
                    <div class="kpi-value">{int(m[s]):,}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        st.caption(
            "S4_Crash_bicycle is the larger, dedicated bicycle population. "
            "Signal4Data supplies E-Bike/E-Scooter (via Qwen narrative) and "
            "everything else. They merge into one combined crash_event table below."
        )

    st.markdown("### Pipeline funnel (from `dashboard_meta.csv`)")
    funnel_steps = [s for s in META_FUNNEL_ORDER if s in m.index]
    # Final export count isn't a row in dashboard_meta.csv itself --
    # it's however many rows made it into the currently loaded
    # power_bi_export.csv, so we append it as the last funnel stage.
    if funnel_steps:
        labels = [META_METRIC_LABELS.get(s, s) for s in funnel_steps] + ["Final dashboard export"]
        values = [m[s] for s in funnel_steps] + [len(df_raw)]
        fig = go.Figure(go.Funnel(
            y=labels, x=values, marker=dict(color="#2b3f8c"),
        ))
        st.plotly_chart(style_fig(fig, title="Records Retained at Each Pipeline Step", height=340), use_container_width=True)
    else:
        st.dataframe(meta_raw, use_container_width=True, hide_index=True)

    mode_metrics = [s for s in ["bicycle_crashes", "ebike_crashes", "escooter_crashes"] if s in m.index]
    if mode_metrics:
        st.markdown("**Active-mode breakdown**")
        bc1, bc2, bc3 = st.columns(len(mode_metrics))
        for col, s in zip([bc1, bc2, bc3][:len(mode_metrics)], mode_metrics):
            col.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-label">{META_METRIC_LABELS.get(s, s)}</div>
                    <div class="kpi-value">{int(m[s]):,}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    if note_col:
        with st.expander("What each metric means"):
            for _, row in meta_raw.iterrows():
                label = META_METRIC_LABELS.get(row[metric_col], row[metric_col])
                st.markdown(f"**{label}** ({int(row[value_col]):,}) -- {row[note_col]}")
else:
    st.markdown(
        f"""<div class="section-note">
        No <code>{DEFAULT_META_PATH}</code> loaded yet, so the exact
        step-by-step funnel isn't shown here -- add it under the
        <b>Data Source</b> panel in the sidebar to see records retained
        at each pipeline stage (S4_Crash_bicycle + Signal4Data query pulled
        &rarr; combined crash_event &rarr; active-mode classified &rarr;
        final export). In the meantime, the current loaded export contains
        <b>{len(df_raw):,}</b> crash records after all pipeline steps.
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("## Data sources")
st.markdown(
    """
- **S4_Crash_bicycle** -- the larger, dedicated bicycle-crash population.
  Source of every Bicycle-mode record, minus any Qwen E-Bike/E-Scooter
  override.
- **Signal4** crash report tables (crash-level fields: date/time, location,
  severity, contributing factors, citations) -- source of E-Bike, E-Scooter,
  and Other.
- **FDOT** roadway tables (AADT, intersection control, functional
  classification -- matched by location where available), pulled from
  whichever source system (S4_Crash_bicycle or Signal4Data) each crash
  came from.
- **Non-motorist person-level records** (age, gender, and other
  demographics of the person on the bicycle/e-bike/e-scooter, when a
  demographics export is loaded)
    """
)


render_pipeline_figures("tab0")

