"""LLM vs FLHSMV tab -- how the LLM e-bike/e-scooter label compares with FLHSMV's
"E-Bike Crash Likeliness" rating.

Everything on this tab is read from results/flhsmv_comparison.json, which
build_flhsmv_comparison.py computes from ebike_llm_vs_flhsmv_combined.xlsx (the
JSON holds counts and generated text only, no narratives). No number below is
typed in by hand. This tab is a fixed comparison on the matched sample, so it
deliberately ignores the sidebar filters.
"""
import json

_FL_PATH = os.path.join(_HERE, "results", "flhsmv_comparison.json")
_PRED_COLORS = {"E-bike": "#4CAF50", "E-scooter": "#FF9800", "Bicyclist": "#2196F3", "Other": "#9E9E9E"}


@st.cache_data(ttl=3600)
def _load_flhsmv(path, _mtime=None):
    with open(path) as f:
        return json.load(f)


def _about(what, how=None, takeaway=None):
    """Plain-language caption shown under a chart or table."""
    parts = [f"<b>What it shows:</b> {what}"]
    if how:
        parts.append(f"<b>How to read it:</b> {how}")
    if takeaway:
        parts.append(f"<b>Takeaway:</b> {takeaway}")
    st.markdown('<div class="section-note">' + "<br>".join(parts) + "</div>", unsafe_allow_html=True)


_FL_SCHEMA = 2   # must match "schema" written by build_flhsmv_comparison.py
_FL_DATA = None
if os.path.exists(_FL_PATH):
    _FL_DATA = _load_flhsmv(_FL_PATH, _mtime=os.path.getmtime(_FL_PATH))

st.markdown("## LLM vs FLHSMV: E-Bike Classification")

if _FL_DATA is not None and _FL_DATA.get("schema") != _FL_SCHEMA:
    st.markdown(
        """<div class="section-note">
        <code>results/flhsmv_comparison.json</code> is from an older build and does not match this version of the tab.
        Replace it with the newest <code>flhsmv_comparison.json</code>, or rebuild it from the project folder with
        <code>python build_flhsmv_comparison.py /full/path/to/ebike_llm_vs_flhsmv_combined.xlsx</code>, then reload.
        </div>""",
        unsafe_allow_html=True,
    )
elif _FL_DATA is None:
    st.markdown(
        """<div class="section-note">
        <code>results/flhsmv_comparison.json</code> was not found. Build it from the project folder with
        <code>python build_flhsmv_comparison.py /path/to/ebike_llm_vs_flhsmv_combined.xlsx</code> and reload.
        </div>""",
        unsafe_allow_html=True,
    )
else:
    FL = _FL_DATA
    K, LABELS, PREDS, CT, LT = FL["kpis"], FL["labels"], FL["preds"], FL["crosstab"], FL["label_totals"]
    LD, LP, SH = FL["ladder"], FL["llm_pos_share"], FL["sheets"]

    # ------------------------------------------------------------------ intro
    st.markdown(
        f"""<div class="section-note">
        This tab checks a <b>large language model (LLM)</b> against <b>FLHSMV</b> (Florida Highway Safety and Motor Vehicles).
        For {K['ids_total']:,} crash reports, the LLM read the written crash narrative and decided what kind of vehicle the
        person was riding. FLHSMV's reviewers separately rated, for the same crash reports, how likely it was that the crash involved an
        e-bike or e-scooter, using keywords in the narrative and property-damage fields. We line the two up to see where they agree, where they differ, and why. Only the {K['matched']:,} reports that exist in
        both sources can be compared. <b>This tab ignores the sidebar filters</b> (fixed sample; FLHSMV records loaded
        {FL['load_dt_range'][0]} to {FL['load_dt_range'][1]}). Built {FL['generated']} from <code>{FL['source_workbook']}</code>.
        </div>""",
        unsafe_allow_html=True,
    )

    with st.expander("Key terms (read this first)", expanded=True):
        st.markdown("**FLHSMV \"E-Bike Likeliness\" rating** – assigned by FLHSMV reviewers using keyword criteria. FLHSMV's own definitions, with how many of the comparable reports carry each rating:")
        fd = pd.DataFrame(FL["flhsmv_defs"]).rename(columns={"label": "Rating", "definition": "FLHSMV definition", "reports": "Comparable reports"})
        st.dataframe(fd, width="stretch", hide_index=True)
        if not FL["labels_unexpected"] and FL["flhsmv_defs"][-1]["reports"] == 0:
            st.caption("\"Not Enough Information\" is part of FLHSMV's legend, but no comparable report in this data carries it, so it does not appear in the charts.")
        st.markdown(
            """
**"FLHSMV positive"** means rated **Likely or higher** (Likely, Highly Likely or Definitely). That cut-off is a choice made for this analysis;
"FLHSMV negative" means Less Likely or Not Likely. Note that FLHSMV's definitions cover <b>"an E-Bike or E-Scooter"</b> together.

**Matched / unmatched** – a report is *matched* if its REPORT_NUMBER appears in the FLHSMV files, so it has a rating to compare with.
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
**LLM labels** – the model reads the crash narrative and picks one, following Florida Statute 316.003 and a numbered rule list:
- **E-bike**: a bicycle with working pedals and an electric motor under 750 W. The narrative must state it is electric (or name a known e-bike brand).
- **E-scooter**: a motor-powered scooter-type device with no pedals, standing or seated. The narrative must state it is electric, or name a known e-scooter brand.
- **Bicyclist**: only human-powered cycling terms (bicycle, cyclist, tricycle) with no power-related wording anywhere. "Bike" on its own is never Bicyclist.
- **Other**: everything else. This includes vague wording ("bike" or "scooter" alone, "motorized scooter" alone), two different devices in one report, no device named, and non-qualifying vehicles such as mopeds, motorcycles, electric dirt bikes, hoverboards and wheelchairs.

**What the LLM saw**: the narrative text only. It was **not given the TYPE_PROPERTY_DAMAGE column** (the officer's note of what was damaged, often "electric bicycle"), which FLHSMV's criteria do use.

**Strict vs lenient** – *strict* counts only an LLM "E-bike" as a match for FLHSMV positive; *lenient* counts "E-bike" or "E-scooter". Because FLHSMV's ratings cover both, <b>lenient is the like-for-like comparison</b>.
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
**Agreement** is the share of reports where the two sources land on the same side (both say e-bike/e-scooter, or both say not).
**Cohen's kappa** is agreement adjusted for chance (0 means no better than chance, 1 is perfect).
**Precision** (here): of the reports the LLM calls e-bike or e-scooter, the share FLHSMV also rated positive.
**Recall** (here): of the reports FLHSMV rated positive, the share the LLM also called e-bike or e-scooter. **F1** combines the two.
            """
        )

    # ------------------------------------------------------------------- KPIs
    def _kpi(col, label, value, sub, color="#7a86d4"):
        col.markdown(
            f"""<div class="kpi-card" style="border-left-color:{color};">
                <div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    c = st.columns(4)
    _kpi(c[0], "Reports that can be compared", f"{K['matched']:,}",
         f"of {K['ids_total']:,} ({K['match_rate']}%); {K['unmatched']:,} not found in FLHSMV")
    _kpi(c[1], "Agreement, strict", f"{K['strict']['agree']}%", f"LLM E-bike only &middot; kappa {K['strict']['kappa']:.2f}", "#4CAF50")
    _kpi(c[2], "Agreement, lenient", f"{K['lenient']['agree']}%", f"like-for-like: LLM E-bike or E-scooter &middot; kappa {K['lenient']['kappa']:.2f}", "#FF9800")
    _kpi(c[3], "LLM precision / recall", f"{K['precision']}% / {K['recall']}%",
         f"F1 {K['f1']}% &middot; {K['pos_n']:,} FLHSMV-positive reports", "#2196F3")
    st.write("")
    c = st.columns(4)
    _kpi(c[0], "E-scooter scope difference", f"{K['n_esc']:,}", "FLHSMV Definitely/Highly Likely; LLM separates e-scooters", "#FF9800")
    _kpi(c[1], "LLM blind spot", f"{K['n_blind']:,}", "FLHSMV Definitely, LLM regular bicycle, property field says electric", "#E53935")
    _kpi(c[2], "FLHSMV-positive, LLM Other", f"{K['n_pos_other']:,}", "FLHSMV Likely or higher, LLM sent it to Other", "#9E9E9E")
    _kpi(c[3], "FLHSMV misses", f"{K['n_miss']:,}", f"FLHSMV Less/Not Likely, LLM found an e-bike or e-scooter", "#4CAF50")
    st.write("")

    st.markdown(f'<div class="section-note">💡 <b>Bottom line:</b> {FL["bottom_line"]}</div>', unsafe_allow_html=True)

    # ---------------------------------------------------- how the labels line up
    st.markdown("### How the two labels line up")
    cc = st.columns(2)
    with cc[0]:
        rows = [{"FLHSMV rating": lab, "LLM label": p, "Reports": CT[lab][p],
                 "pct": CT[lab][p] / LT[lab] * 100 if LT[lab] else 0} for lab in LABELS for p in PREDS]
        fig = px.bar(pd.DataFrame(rows), y="FLHSMV rating", x="pct", color="LLM label", orientation="h",
                     color_discrete_map=_PRED_COLORS, category_orders={"FLHSMV rating": LABELS, "LLM label": PREDS},
                     custom_data=["Reports"])
        fig.update_traces(hovertemplate="%{fullData.name}: %{x:.1f}% (%{customdata[0]:,} reports)<extra></extra>")
        fig.update_layout(barmode="stack", xaxis_title="% of reports with that FLHSMV rating", yaxis_title=None,
                          yaxis={"autorange": "reversed"})
        st.plotly_chart(style_fig(fig, title="What the LLM says within each FLHSMV rating", height=380, n=K["matched"]), width="stretch")
        _about(
            "For each FLHSMV rating (rows), how the LLM labeled those same reports (colors).",
            "Each bar adds up to 100% of the reports with that rating. Hover for counts.",
            f"The top two ratings are mostly E-bike or E-scooter. \"Likely\" is mostly \"Other\" ({CT['Likely']['Other'] / LT['Likely'] * 100:.0f}%). "
            f"\"Less Likely\" and \"Not Likely\" are mostly ordinary bicycles, as expected.",
        )
    with cc[1]:
        fig = go.Figure()
        fig.add_bar(x=LABELS, y=[LD[l]["strict"] for l in LABELS], name="Strict (LLM E-bike only)", marker_color="#4CAF50")
        fig.add_bar(x=LABELS, y=[LD[l]["lenient"] for l in LABELS], name="Lenient (LLM E-bike or E-scooter)", marker_color="#FF9800")
        fig.add_bar(x=LABELS, y=[LD[l]["credited"] for l in LABELS], name="Lenient + blind-spot cases credited", marker_color="#9DC3E6")
        fig.update_layout(barmode="group", yaxis_title="% of reports with that FLHSMV rating", xaxis_title=None, yaxis_range=[0, 100])
        fig.update_traces(hovertemplate="%{x}: %{y:.1f}%<extra>%{fullData.name}</extra>")
        st.plotly_chart(style_fig(fig, title="Share of each FLHSMV rating the LLM confirms", height=380, n=K["matched"]), width="stretch")
        _about(
            "For each FLHSMV rating, the share of its reports where the LLM also found an electric device.",
            f"Green counts only LLM \"E-bike\". Orange also counts \"E-scooter\". Light blue is the orange bar plus the {K['n_blind']:,} <b>blind-spot cases</b>: "
            f"reports FLHSMV rated \"Definitely\" where the LLM said ordinary bicycle, but the property-damage column (which the LLM was not given) says "
            f"electric bicycle. \"Credited\" means we count those as if the LLM had agreed, to show what the LLM would score with that column. Only the \"Definitely\" bar changes.",
            f"\"Definitely\" rises from {LD['Definitely']['lenient']}% to {LD['Definitely']['credited']}% once credited, level with \"Highly Likely\" ({LD['Highly Likely']['lenient']}%). "
            f"\"Likely\" is only {LD['Likely']['lenient']}%.",
        )

    cc = st.columns(2)
    with cc[0]:
        fig = go.Figure(go.Bar(x=PREDS, y=[LP[p]["pct"] for p in PREDS], marker_color=[_PRED_COLORS[p] for p in PREDS],
                               customdata=[[LP[p]["n"], LP[p]["pos"]] for p in PREDS],
                               text=[f"{LP[p]['pct']}%" for p in PREDS], textposition="outside"))
        fig.update_traces(hovertemplate="%{x}: %{y:.1f}% (%{customdata[1]:,} of %{customdata[0]:,} reports)<extra></extra>")
        fig.update_layout(yaxis_title="% rated Likely or higher by FLHSMV", xaxis_title="LLM label", yaxis_range=[0, 110])
        st.plotly_chart(style_fig(fig, title="Share of each LLM label that FLHSMV also rates positive", height=360, n=K["matched"]), width="stretch")
        _about(
            "The same comparison from the other side: for each LLM label, the share of those reports FLHSMV rated Likely or higher.",
            "A tall bar means FLHSMV also considers those reports e-bike crashes.",
            f"FLHSMV agrees with {LP['E-bike']['pct']}% of LLM \"E-bike\" and {LP['E-scooter']['pct']}% of LLM \"E-scooter\" calls. "
            f"But it also rates {LP['Other']['pct']}% of the LLM's \"Other\" and {LP['Bicyclist']['pct']}% of its \"Bicyclist\" reports as positive, "
            f"so FLHSMV's positive group is broader than the LLM's.",
        )
    with cc[1]:
        cf = FL["confidence"]
        fig = go.Figure(go.Bar(x=list(cf.keys()), y=list(cf.values()), marker_color=[_PRED_COLORS.get(k, "#9E9E9E") for k in cf],
                               text=[f"{v:.2f}" for v in cf.values()], textposition="outside"))
        fig.update_layout(yaxis_title="Average LLM confidence (0 to 1)", yaxis_range=[0.5, 1.02], xaxis_title="LLM label")
        st.plotly_chart(style_fig(fig, title="How confident the LLM says it is", height=360), width="stretch")
        _about(
            "The average confidence score the LLM reported with each label.",
            "Closer to 1 means the model says it is more certain.",
            f"Confidence is about {cf['Bicyclist']:.2f} for Bicyclist, E-bike and E-scooter but {cf['Other']:.2f} for \"Other\", so lower confidence does flag ambiguous reports. "
            f"It does not flag the blind-spot cases: all {K['n_blind']:,} were labeled at {K['blind_conf']:.2f}.",
        )

    # ---------------------------------------------------- where the disagreements come from
    st.markdown("### Where the disagreements come from")
    bd = pd.DataFrame(FL["breakdown"])
    cc = st.columns([3, 2])
    with cc[0]:
        fig = px.bar(bd, y="case", x="ids", orientation="h", custom_data=["desc"],
                     color="kind", color_discrete_map={"scope": "#FF9800", "llm": "#E53935", "disputed": "#9E9E9E", "flhsmv": "#7a86d4"})
        fig.update_traces(texttemplate="%{x:,}", textposition="outside",
                          hovertemplate="%{y}: %{x:,} reports<br>%{customdata[0]}<extra></extra>")
        fig.update_layout(showlegend=False, xaxis_title="Reports", yaxis_title=None, yaxis={"autorange": "reversed"})
        st.plotly_chart(style_fig(fig, title="Types of disagreement between the LLM and FLHSMV", height=380), width="stretch")
    with cc[1]:
        lines = "".join(f"<li><b>{r['case']} ({r['ids']:,}):</b> {r['desc']}</li>" for r in FL["breakdown"])
        _about(
            "Every report where the two sources disagree, grouped by the likely reason.",
            "Each bar is a count of reports. The first bar only counts as a disagreement under the <i>strict</i> view; the other four are disagreements under both views "
            f"(together {K['disagree_lenient']:,} reports, {100 - K['lenient']['agree']:.1f}% of the comparable sample).",
        )
        st.markdown(f'<div class="section-note"><b>What each bar means</b><ul style="margin:0.3rem 0 0 1rem;padding:0">{lines}</ul></div>', unsafe_allow_html=True)
    _about(
        "Why the e-scooter bar is not an error: FLHSMV's rating definitions cover \"an E-Bike or E-Scooter\" as one group, while the LLM keeps the two separate "
        "(an e-bike has pedals and a motor; an e-scooter has no pedals).",
        None,
        "Both sources see an electric device; they differ only on the category. That is why agreement jumps from "
        f"{K['strict']['agree']}% to {K['lenient']['agree']}% when e-scooters count as a match.",
    )

    st.markdown(f"#### Why the LLM said \"Other\" for {K['n_pos_other']:,} reports FLHSMV rated likely")
    rd = pd.DataFrame([{
        "Rule code": r["code"], "Reports": r["ids"], "Share": r["share"],
        "Rule as written in the LLM prompt": r["definition"],
        "Reasoning contains the rule's trigger wording": (f"{r['trigger_pct']:.0f}%" if r["trigger_pct"] is not None else "n/a (catch-all)"),
        "Device wording the LLM cited (most common)": ", ".join(f"{t['term']} ({t['pct']:.0f}%)" for t in r["top_terms"]) or "mixed",
    } for r in FL["rules"]])
    st.dataframe(rd, width="stretch", hide_index=True,
                 column_config={"Reports": st.column_config.NumberColumn(format="%d"),
                                "Share": st.column_config.ProgressColumn("Share of these reports (%)", min_value=0, max_value=100, format="%.1f")})
    top3 = sum(r["ids"] for r in FL["rules"] if r["code"] in ("R9", "RC3", "R8"))
    _about(
        f"The LLM follows a numbered rule list in its prompt. Whenever the rules say the device is not clearly an e-bike, e-scooter or bicycle, it answers \"Other\" and writes the rule code (for example R8) in its reasoning. "
        f"This table counts the codes across all {FL['rules_total']:,} reports where FLHSMV rated Likely or higher and the LLM answered \"Other\".",
        "The middle column is the rule text from the prompt. The last two columns are counted from the LLM's own reasoning: the share of cases whose reasoning contains the wording that rule is meant for, and the device words it cited most often. "
        "Where the first percentage is low (for example R11 and R14), the LLM is citing a code for cases the rule text does not describe, which may be worth checking in the prompt.",
        f"R9, R8 and RC3 account for {top3:,} of the {FL['rules_total']:,}: a bare \"scooter\", a \"motorized scooter\" with nothing else, or no device named. "
        f"In these the LLM is following its instruction not to guess. FLHSMV may be right or the LLM may be too cautious; only a human review of a sample would say.",
    )

    # --------------------------------------------------------------- coverage
    st.markdown("### Coverage: reports that could not be compared")
    cc = st.columns(2)
    with cc[0]:
        ub = FL["unmatched_by_sheet"]
        names = {"EBike": "E-bike file", "RegBike": "Regular-bike file"}
        rows = [{"File": names[s], "LLM label": p, "Reports": ub[s][p]} for s in ub for p in PREDS]
        fig = px.bar(pd.DataFrame(rows), x="File", y="Reports", color="LLM label", color_discrete_map=_PRED_COLORS,
                     category_orders={"LLM label": PREDS})
        fig.update_layout(barmode="stack", xaxis_title=None, yaxis_title="Reports not found in FLHSMV")
        fig.update_traces(hovertemplate="%{x}, %{fullData.name}: %{y:,}<extra></extra>")
        st.plotly_chart(style_fig(fig, title="Reports not found in the FLHSMV files", height=360, n=K["unmatched"]), width="stretch")
        _about(
            "The reports whose REPORT_NUMBER does not appear in the FLHSMV files, split by the file they came from (the project's e-bike file and regular-bike file) and by the LLM's label.",
            "Taller bars mean more reports with nothing to compare against.",
            f"{K['un_pos']:,} of the {K['unmatched']:,} ({K['un_pos'] / K['unmatched'] * 100:.1f}%) are reports the LLM called E-bike or E-scooter, "
            f"so a meaningful number of likely e-bike crashes sit outside this comparison.",
        )
    with cc[1]:
        ids = pd.DataFrame(FL["id_series"])
        fig = go.Figure(go.Bar(x=ids["series"], y=ids["match_rate"], marker_color=["#E57373", "#4CAF50", "#E57373"],
                               customdata=ids[["matched", "ids"]].values,
                               text=[f"{v}%" for v in ids["match_rate"]], textposition="outside"))
        fig.update_traces(hovertemplate="%{x}: %{y:.1f}% found (%{customdata[0]:,} of %{customdata[1]:,})<extra></extra>")
        fig.update_layout(yaxis_title="% found in FLHSMV files", yaxis_range=[0, 110], xaxis_title="REPORT_NUMBER begins with")
        st.plotly_chart(style_fig(fig, title="How often a report is found in FLHSMV, by ID series", height=360, n=K["ids_total"]), width="stretch")
        s_ = {x["series"]: x for x in FL["id_series"]}
        _about(
            "REPORT_NUMBERs fall into two numeric families (starting 21 to 28, and starting 80 to 90). This groups reports by that start and shows how many were found in the FLHSMV files.",
            "A short bar means most reports in that group are missing from FLHSMV.",
            f"The missing reports are concentrated in two groups: IDs starting 80 to 90 ({s_['Starts with 80-90']['match_rate']}% found) and the lowest IDs, starting 21 to 24 ({s_['Starts with 21-24']['match_rate']}% found), "
            f"versus {s_['Starts with 25-28']['match_rate']}% for 25 to 28. That points to which reports are missing, not yet to why. "
            f"The FLHSMV records that matched were loaded between {FL['zip_window'][0]} and {FL['zip_window'][1]}, so a different year range is one possible reason; this workbook cannot confirm it.",
        )
    cy = FL.get("crash_year_check")
    if cy:
        cyd = pd.DataFrame(cy["rows"])
        fig = go.Figure()
        fig.add_bar(x=cyd["year"], y=cyd["matched"], name="Found in FLHSMV", marker_color="#4CAF50")
        fig.add_bar(x=cyd["year"], y=cyd["unmatched"], name="Not found", marker_color="#E57373")
        fig.update_layout(barmode="stack", xaxis_title="Crash year", yaxis_title="Reports")
        st.plotly_chart(style_fig(fig, title="Found vs not found, by crash year", height=340, n=cy["ids_found"]), width="stretch")
        _about(
            f"Each report's crash year (from <code>{cy['file']}</code>) against whether it was found in the FLHSMV files. {cy['ids_found']:,} of {cy['ids_total']:,} reports were found in that file.",
            "If the red part is concentrated in particular years, a year mismatch between the sources explains the gap.",
        )

    cc = st.columns(2)
    with cc[0]:
        names2 = ["EBike", "RegBike"]
        fig = go.Figure(go.Bar(x=["E-bike file", "Regular-bike file", "Both (blended)"],
                               y=[SH[s]["agree_pct"] for s in names2] + [K["lenient"]["agree"]],
                               marker_color=["#4CAF50", "#2196F3", "#7a86d4"],
                               text=[f"{SH[s]['agree_pct']}%" for s in names2] + [f"{K['lenient']['agree']}%"], textposition="outside"))
        fig.update_layout(yaxis_title="Agreement (lenient), %", yaxis_range=[0, 110], xaxis_title=None)
        st.plotly_chart(style_fig(fig, title="Agreement with FLHSMV, by file", height=360, n=K["matched"]), width="stretch")
        _about(
            "Agreement between the two sources (lenient view) inside each of the project's two files, and for both together.",
            "Higher means the LLM and FLHSMV land on the same side more often.",
            f"The e-bike file is almost entirely FLHSMV-positive ({SH['EBike']['pos']:,} of {SH['EBike']['matched']:,}), so its {SH['EBike']['agree_pct']}% mostly measures how many of those the LLM confirms. "
            f"The regular-bike file is almost entirely FLHSMV-negative ({SH['RegBike']['matched'] - SH['RegBike']['pos']:,} of {SH['RegBike']['matched']:,}), so its {SH['RegBike']['agree_pct']}% mostly measures how rarely the LLM finds an e-bike there. "
            f"The blended {K['lenient']['agree']}% depends on that mix.",
        )

    # ------------------------------------------------------- time and agency
    with cc[1]:
        qd = pd.DataFrame(FL["quarters"])
        fig = go.Figure(go.Scatter(x=qd["q"], y=qd["pct"], mode="lines+markers", line=dict(color="#2b3f8c", width=3),
                                   customdata=qd[["c", "n"]].values))
        fig.update_traces(hovertemplate="%{x}: %{y:.1f}% (%{customdata[0]:,} of %{customdata[1]:,})<extra></extra>")
        fig.update_layout(yaxis_title="% the LLM also calls e-bike or e-scooter", xaxis_title="Quarter (FLHSMV load date)", yaxis_range=[60, 90])
        st.plotly_chart(style_fig(fig, title="FLHSMV-positive reports the LLM confirms, over time", height=360, n=int(qd["n"].sum())), width="stretch")
        _about(
            "Within the e-bike file (all FLHSMV-positive), the share of reports per quarter that the LLM also labeled E-bike or E-scooter.",
            "A rising line means the LLM confirms more of FLHSMV's positives in later quarters. The vertical axis starts at 60%. The last quarter is partial, and the date is when FLHSMV loaded the record, not necessarily the crash date.",
            f"The share rose from {FL['quarters'][0]['pct']}% to {FL['quarters'][-1]['pct']}%. The data cannot say why.",
        )

    st.markdown("### By agency")
    ad = pd.DataFrame(FL["agencies"])
    top_n = st.slider("Agencies to show (highest disagreement first)", 5, len(ad), min(12, len(ad)), key="fl_agency_n")
    ad_top = ad.head(top_n)
    fig = go.Figure(go.Bar(y=ad_top["agency"], x=ad_top["disagree_pct"], orientation="h", marker_color="#7a86d4",
                           customdata=ad_top[["positives", "blind_spot", "escooter_pct"]].values,
                           text=[f"{v}%" for v in ad_top["disagree_pct"]], textposition="outside"))
    fig.update_traces(hovertemplate="%{y}: %{x:.1f}% disagree<br>%{customdata[0]:,} FLHSMV-positive reports"
                                    "<br>%{customdata[1]:,} blind-spot cases<br>%{customdata[2]:.1f}% LLM E-scooter<extra></extra>")
    fig.update_layout(xaxis_title="% of FLHSMV-positive reports the LLM does not call e-bike or e-scooter", yaxis_title=None,
                      yaxis={"autorange": "reversed"}, xaxis_range=[0, max(60, ad_top["disagree_pct"].max() + 10)])
    st.plotly_chart(style_fig(fig, title=f"Disagreement by reporting agency (agencies with {K['agency_min']}+ FLHSMV-positive reports)",
                              height=max(340, 28 * top_n + 120)), width="stretch")
    _about(
        f"For each agency with at least {K['agency_min']} FLHSMV-positive reports, the share of those reports where the LLM did <i>not</i> find an e-bike or e-scooter.",
        "Longer bars mean more disagreement. Hover to see the agency's report count, its blind-spot cases and its e-scooter share.",
        f"Agencies differ widely. Part of the gap is where each agency records \"electric\": if it writes it only in the property-damage column, the LLM (which never saw that column) misses it. "
        f"Overall the LLM disagrees with {K['overall_dis']}% of FLHSMV-positive reports.",
    )

    # ---------------------------------------------------------------- insights
    st.markdown("### Key insights")
    st.markdown(
        '<div class="section-note">What the numbers above mean, in plain terms. The supporting figures sit under each point.</div>',
        unsafe_allow_html=True,
    )
    for ins in FL["insights"]:
        st.markdown(f"**{ins['title']}**")
        st.markdown(ins["text"])
        st.caption("Evidence: " + ins["evidence"])

    st.markdown("### Miscellaneous data notes")
    for mi in FL["misc"]:
        with st.expander(f"{mi['theme']}: {mi['figure']}"):
            st.markdown(mi["text"])

    # ---------------------------------------------------------------- examples
    st.markdown("### Example cases")
    ex = pd.DataFrame(FL["examples"]).rename(columns={
        "category": "Category", "id": "REPORT_NUMBER", "agency": "Agency", "flhsmv": "FLHSMV rating",
        "llm": "LLM label", "note": "Evidence in the data"})
    st.dataframe(ex[["Category", "REPORT_NUMBER", "Agency", "FLHSMV rating", "LLM label", "Evidence in the data"]],
                 width="stretch", hide_index=True)
    _about(
        "A handful of real REPORT_NUMBERs for each type of disagreement, with the evidence found in the data for that report.",
        "The crash narratives are kept in the local workbook (they are not published here); look the REPORT_NUMBER up in the matching sheet.",
    )
    es = pd.DataFrame(FL["example_sheets"]).rename(columns={"sheet": "Workbook sheet", "rows": "Reports", "what": "What it holds", "note": "Notes"})
    st.dataframe(es, width="stretch", hide_index=True)

    st.markdown("### Caveats")
    for t in FL["caveats"]:
        st.markdown(f"- {t}")

    with st.expander("Where do these numbers come from?"):
        st.markdown("Every figure on this tab is computed by `build_flhsmv_comparison.py` from the workbook and stored in `results/flhsmv_comparison.json`.")
        st.dataframe(pd.DataFrame(FL["provenance"]).rename(columns={"item": "Figure", "source": "Workbook sheet(s)", "how": "How it is computed"}),
                     width="stretch", hide_index=True)
        ck = pd.DataFrame(FL["checks"])
        ck["ok"] = ck["ok"].map({True: "matches", False: "DIFFERS"})
        st.markdown("Cross-checks against the workbook's own sheets:")
        st.dataframe(ck.rename(columns={"check": "Check", "computed": "Recomputed here", "workbook": "In workbook", "ok": "Result"}),
                     width="stretch", hide_index=True)
