"""
Build results/flhsmv_comparison.json for the "LLM vs FLHSMV" dashboard tab.

Input : ebike_llm_vs_flhsmv_combined.xlsx (keep it local -- it contains crash
        narratives, so it is NOT committed). Sheets used: Joined_EBike,
        Joined_RegBike, Chart_Data (LLM confidence table only) and the four
        example sheets.
Output: results/flhsmv_comparison.json -- aggregate counts, metrics and text
        generated from those counts. No narratives are written to it.

Usage (run from the project root, next to app.py):
    python build_flhsmv_comparison.py /path/to/ebike_llm_vs_flhsmv_combined.xlsx
    python build_flhsmv_comparison.py WORKBOOK.xlsx power_bi_export.csv   # optional crash-year check

Every number in the JSON is computed here from the workbook (the one exception,
the LLM confidence table, is read from Chart_Data because the Joined sheets do
not carry a confidence column). Cross-checks against the workbook's own example
sheets are asserted at the end and listed under "checks" in the JSON.

Definitions
    * Matched IDs only; one row per REPORT_NUMBER. When the FLHSMV zip repeats
      an ID, the highest FLHSMV label is kept (ties: latest LOAD_DT).
    * FLHSMV positive = Likely / Highly Likely / Definitely.
    * Strict LLM positive = "E-bike"; lenient = "E-bike" or "E-scooter".
"""
import json
import os
import re
import sys
from datetime import date

import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "ebike_llm_vs_flhsmv_combined.xlsx"
CRASH_CSV = sys.argv[2] if len(sys.argv) > 2 else ("power_bi_export.csv" if os.path.exists("power_bi_export.csv") else None)
OUT = os.path.join("results", "flhsmv_comparison.json")

L = "E-Bike Crash Likeliness FLHSMV"
LABELS = ["Definitely", "Highly Likely", "Likely", "Less Likely", "Not Likely"]
RANK = {"Not Likely": 0, "Less Likely": 1, "Likely": 2, "Highly Likely": 3, "Definitely": 4}
PREDS = ["E-bike", "E-scooter", "Bicyclist", "Other"]
POS = ["Likely", "Highly Likely", "Definitely"]
DT = "Month, Day, Year of LOAD_DT"
AGENCY_MIN = 100   # agencies shown need at least this many FLHSMV-positive reports
ELECTRIC_PROP = r"electric|e-?bike|e bike|e-?bicycle|e bicycle|ebicycle"   # reproduces the 696-case blind-spot set


def pct(a, b, nd=1):
    return round(a / b * 100, nd) if b else None


def n(x):
    return f"{int(x):,}"


def kappa(tp, fp, fn, tn):
    t = tp + fp + fn + tn
    po = (tp + tn) / t
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / t ** 2
    return round((po - pe) / (1 - pe), 3)


print("reading", SRC)
e = pd.read_excel(SRC, "Joined_EBike", dtype=str)
r = pd.read_excel(SRC, "Joined_RegBike", dtype=str)
e["src"], r["src"] = "EBike", "RegBike"
raw = pd.concat([e, r], ignore_index=True)
raw["rk"] = raw[L].map(RANK)
raw["_dt"] = pd.to_datetime(raw[DT], errors="coerce")
rows_total = {"EBike": len(e), "RegBike": len(r)}
rows_extra = {"EBike": len(e) - e.REPORT_NUMBER.nunique(), "RegBike": len(r) - r.REPORT_NUMBER.nunique()}
mt = raw.dropna(subset=[L])
dup_matched = int(mt.groupby("REPORT_NUMBER").size().gt(1).sum())
conflict = int(mt.groupby("REPORT_NUMBER")[L].nunique().gt(1).sum())
forms_by_id = mt.groupby("REPORT_NUMBER")["FORM_TYPE"].agg(lambda s: set(s))
dup_lu = int(forms_by_id.map(lambda s: {"L", "U"} <= s).sum())
# one row per ID: highest label, ties -> latest LOAD_DT (deterministic)
d = (raw.sort_values(["rk", "_dt"], ascending=[False, False], na_position="last", kind="stable")
        .drop_duplicates("REPORT_NUMBER"))
m = d[d[L].notna()].copy()
un = d[d[L].isna()].copy()
m["pos"] = m[L].isin(POS)
m["lenient"] = m["LLM prediction"].isin(["E-bike", "E-scooter"])
m["strict"] = m["LLM prediction"] == "E-bike"
m["agree"] = m["pos"] == m["lenient"]
m["dt"] = m["_dt"]
ids_total, matched, unmatched = len(d), len(m), len(un)
ids_by_sheet = d["src"].value_counts().to_dict()

# ---------------------------------------------------------------- crosstab
ct = pd.crosstab(m[L], m["LLM prediction"]).reindex(index=LABELS, columns=PREDS, fill_value=0)
CT = {lab: {p: int(ct.loc[lab, p]) for p in PREDS} for lab in LABELS}
LT = {lab: int(ct.loc[lab].sum()) for lab in LABELS}
assert sum(LT.values()) == matched

# ---------------------------------------------------------------- agreement
def conf(mask):
    return (int((m.pos & mask).sum()), int((~m.pos & mask).sum()),
            int((m.pos & ~mask).sum()), int((~m.pos & ~mask).sum()))

tp_s, fp_s, fn_s, tn_s = conf(m.strict)
tp_l, fp_l, fn_l, tn_l = conf(m.lenient)
agree_s, agree_l = pct(tp_s + tn_s, matched), pct(tp_l + tn_l, matched)
precision, recall = pct(tp_l, tp_l + fp_l), pct(tp_l, tp_l + fn_l)
f1 = round(2 * precision * recall / (precision + recall), 1)
pos_n, neg_n = int(m.pos.sum()), int((~m.pos).sum())

# ---------------------------------------------------------------- disagreement pieces (all from Joined data)
dm = m[m.pos & (m["LLM prediction"] == "Other")]
n_pos_other = len(dm)
n_pos_bic = int((m.pos & (m["LLM prediction"] == "Bicyclist")).sum())
blind_mask = (m[L] == "Definitely") & (m["LLM prediction"] == "Bicyclist") & m.TYPE_PROPERTY_DAMAGE.fillna("").str.lower().str.contains(ELECTRIC_PROP)
blind = m[blind_mask]
n_blind = len(blind)
n_bic_other = n_pos_bic - n_blind
esc_mask = m[L].isin(["Definitely", "Highly Likely"]) & (m["LLM prediction"] == "E-scooter")
esc = m[esc_mask]
n_esc = len(esc)
miss = m[(~m.pos) & m.lenient]
n_miss = len(miss)
assert n_pos_other + n_pos_bic + int((m.pos & m.lenient).sum()) == pos_n
assert n_pos_other + n_pos_bic + n_miss == matched - (tp_l + tn_l)   # = lenient disagreements

breakdown = [
    {"case": "E-scooter scope difference", "ids": n_esc, "kind": "scope", "view": "strict only",
     "desc": "FLHSMV rated it Definitely/Highly Likely an e-bike crash; the LLM named an e-scooter."},
    {"case": "LLM blind spot (property field)", "ids": n_blind, "kind": "llm", "view": "both",
     "desc": "FLHSMV Definitely; the LLM said regular bicycle, but the property-damage field says electric bicycle."},
    {"case": "FLHSMV-positive, LLM Other", "ids": n_pos_other, "kind": "disputed", "view": "both",
     "desc": "FLHSMV rated Likely or higher; the LLM's rules sent it to Other (motorcycle-type, unclear or missing device, and so on)."},
    {"case": "FLHSMV-positive, LLM Bicyclist (not blind spot)", "ids": n_bic_other, "kind": "disputed", "view": "both",
     "desc": "FLHSMV rated Likely or higher; the LLM said regular bicycle and the property field does not explain it."},
    {"case": "FLHSMV misses", "ids": n_miss, "kind": "flhsmv", "view": "both",
     "desc": "FLHSMV rated Less/Not Likely; the LLM found an e-bike or e-scooter."},
]

# FLHSMV legend (E-Bike Likeliness categories), as supplied with the project; lightly shortened.
FLHSMV_DEFS = [
    ("Definitely", "The crash report definitely involves an E-Bike or E-Scooter. Variations of the two are explicitly mentioned in the Narrative or Property Damage fields."),
    ("Highly Likely", "Highly likely to involve an E-Bike or E-Scooter. Key words such as \"Electric\", \"Bike\" and \"Scooter\" are used in conjunction with each other multiple times in the Narrative or Property Damage fields."),
    ("Likely", "Possibly contains an e-bike or e-scooter. Key words such as \"Electric\" and \"Motorized\" appear once."),
    ("Less Likely", "A \"Bike\" or \"Scooter\" is mentioned, but there is no explicit reference to motorization."),
    ("Not Likely", "No key words are referenced, but property damage is reported. Some reports may have slipped by."),
    ("Not Enough Information", "No key words are referenced and no property damage is reported, so there is not enough information to categorize."),
]
flhsmv_defs = [{"label": k, "definition": v, "reports": int((m[L] == k).sum())} for k, v in FLHSMV_DEFS]
labels_unexpected = sorted(set(m[L].unique()) - set(LABELS))

# ---------------------------------------------------------------- per-sheet / unmatched
sheets = {}
for s, g in m.groupby("src"):
    sheets[s] = {"ids": int(ids_by_sheet[s]), "matched": len(g), "pos": int(g.pos.sum()),
                 "llm_pos": int(g.lenient.sum()), "agree": int(g.agree.sum()),
                 "agree_pct": pct(int(g.agree.sum()), len(g)), "match_rate": pct(len(g), int(ids_by_sheet[s]))}
unmatched_by = {s: {p: int(((un.src == s) & (un["LLM prediction"] == p)).sum()) for p in PREDS} for s in ["EBike", "RegBike"]}
un_pos = sum(unmatched_by[s][p] for s in unmatched_by for p in ("E-bike", "E-scooter"))
un_pos_eb = unmatched_by["EBike"]["E-bike"] + unmatched_by["EBike"]["E-scooter"]
assert sum(sum(v.values()) for v in unmatched_by.values()) == unmatched

# ID "series": REPORT_NUMBERs fall into two numeric families
d["p2"] = d.REPORT_NUMBER.str[:2].astype(int)
d["fam"] = pd.cut(d.p2, [0, 24, 28, 99], labels=["21-24", "25-28", "80-90"])
d["is_m"] = d[L].notna()
id_series = []
for fam, g in d.groupby("fam", observed=True):
    id_series.append({"series": f"Starts with {fam}", "ids": len(g), "matched": int(g.is_m.sum()),
                      "unmatched": int((~g.is_m).sum()), "match_rate": pct(int(g.is_m.sum()), len(g))})
by_prefix = [{"prefix": str(p), "ids": len(g), "matched": int(g.is_m.sum()), "match_rate": pct(int(g.is_m.sum()), len(g))}
             for p, g in d.groupby("p2")]
zip_window = [str(m.dt.min().date()), str(m.dt.max().date())]
fam_un = {x["series"]: x["unmatched"] for x in id_series}

crash_year = None
if CRASH_CSV and os.path.exists(CRASH_CSV):
    cy = pd.read_csv(CRASH_CSV, usecols=lambda c: c in ("REPORT_NUMBER", "YEAR"), dtype=str)
    if {"REPORT_NUMBER", "YEAR"} <= set(cy.columns):
        cy = cy.drop_duplicates("REPORT_NUMBER")
        j = d[["REPORT_NUMBER", "is_m"]].merge(cy, on="REPORT_NUMBER", how="left")
        found = int(j.YEAR.notna().sum())
        yr = j.dropna(subset=["YEAR"]).groupby(["YEAR", "is_m"]).size().unstack(fill_value=0)
        crash_year = {"file": os.path.basename(CRASH_CSV), "ids_found": found, "ids_total": len(j),
                      "rows": [{"year": str(y), "matched": int(v.get(True, 0)), "unmatched": int(v.get(False, 0)),
                                "match_rate": pct(int(v.get(True, 0)), int(v.get(True, 0) + v.get(False, 0)))}
                               for y, v in yr.iterrows()]}
        print("crash-year check written from", CRASH_CSV)

llm_pos = {}
for p in PREDS:
    g = m[m["LLM prediction"] == p]
    llm_pos[p] = {"n": len(g), "pos": int(g.pos.sum()), "pct": pct(int(g.pos.sum()), len(g))}

ladder = {lab: {"lenient": pct(CT[lab]["E-bike"] + CT[lab]["E-scooter"], LT[lab]),
                "strict": pct(CT[lab]["E-bike"], LT[lab]),
                "credited": pct(CT[lab]["E-bike"] + CT[lab]["E-scooter"] + (n_blind if lab == "Definitely" else 0), LT[lab])}
          for lab in LABELS}
fix_upper = pct(tp_l + tn_l + n_blind, matched)

# ---------------------------------------------------------------- rule codes (all FLHSMV-positive LLM-Other IDs)
def parent_rule(s):
    x = re.findall(r"Rule (RC?\d+[a-z]?(?:-ADD)?)", str(s))
    return x[-1] if x else "none"

def device_text(s):
    mm = re.search(r"Device:\s*(.*?)(?:, cited:|\. Power:|\. Rule)", str(s), re.S)
    return (mm.group(1) if mm else "").lower()

dm = dm.copy()
dm["rc"] = dm.reasoning.map(parent_rule)
dm["dev"] = dm.reasoning.map(device_text)
dm["body"] = dm.reasoning.str.replace(r"Rule .*$", "", regex=True).str.lower()
VOCAB = {"scooter": "scooter", "motorized": "motoriz", "dirt bike": "dirt", "motorcycle / moped": r"motorcycle|e-moto|e moto|moped",
         "wheelchair / mobility": "wheelchair|mobility", "e-bike wording": r"e-?bike|electric bike|electric bicycle|e bike",
         "no device named": r"none mentioned|^$", "hoverboard / skateboard / similar": "hover|skate|segway|one ?wheel|unicycle",
         "motorized bicycle": "motorized bicycle"}

# Official rule wording, paraphrased from the LLM classification prompt (the "PROMPT_TEMPLATE" the model was given).
RULE_DEFS = {
    "R4": "\"Motorized bicycle\" with gas evidence, or on its own (power type unknown) -> Other.",
    "R8": "\"Motorized scooter\" or \"motor scooter\" on its own (no standing posture, brand or speed of 15 mph or more), or a seated motorized scooter (could be a gas moped) -> Other.",
    "R9": "\"Scooter\" on its own with no power wording, brand, posture or speed, or a \"stand-up scooter\" with no motor/electric wording (could be a kick scooter) -> Other.",
    "R11": "Gas-powered or moped-class devices (moped, motorcycle, minibike, \"cc\", \"engine\", Vespa and similar) -> Other.",
    "R11b": "Electric moped (a separate Florida vehicle class, not an e-bike) -> Other.",
    "R12": "Electric devices that are neither e-bike nor e-scooter (electric unicycle, hoverboard, electric or motorized skateboard, motorized wheelchair, toy bike, electric dirt bike) -> Other.",
    "R12b": "\"Segway\" on its own, or with only a serial number (no scooter model named) -> Other.",
    "R12c": "Wheel-count override: a 3- or 4-wheel scooter -> Other.",
    "R13": "Catch-all for ambiguous or unidentifiable devices (\"bike\" alone with power clues, \"scooter\" alone, pedestrian only, skateboard, ATV, go-kart, no device named, and similar) -> Other.",
    "R14": "Human-power override: wording such as foot powered, manual, non-motorized or non-electric -> Other, even if a brand name appears.",
    "RC1": "Two or more device classes in the same narrative (for example e-bike and e-scooter, or scooter and bicycle) -> Other.",
    "RC2": "Slash ambiguity such as \"ebike/escooter\" or \"bicycle/scooter\" -> Other.",
    "RC3": "No device named at all (pedestrian only, no vehicle described, or only a bike lane or path mentioned) -> Other.",
}
# Wording that should appear in the LLM's own reasoning if the rule was applied as written (None = catch-all, not testable).
TRIGGERS = {
    "R4": r"motorized bicycle|motor-assisted|motorized bike", "R8": r"motorized scooter|motor scooter", "R9": r"scooter",
    "R11": r"moped|motorcycle|minibike|mini bike|motor bike|motorbike|\bcc\b|engine|gas|vespa|phatmoto|x-pro|e-moto",
    "R11b": r"moped", "R12": r"unicycle|hoverboard|skateboard|wheelchair|toy bike|dirt bike", "R12b": r"segway",
    "R12c": r"wheel", "R14": r"foot|manual|unmotor|non-motor|non-electric|not electric|kick",
    "RC2": r"/", "RC3": r"none mentioned|lane|path|pedestrian|nm\d|p\d", "R13": None,
}
rules = []
for rc, g in dm.groupby("rc"):
    sup = {k: pct(int(g.dev.str.contains(p).sum()), len(g), 0) for k, p in VOCAB.items()}
    top = sorted([(k, v) for k, v in sup.items() if v >= 20], key=lambda kv: -kv[1])[:3]
    if rc == "RC1":
        trig = pct(int(g.dev.str.contains(r",|/|\band\b|\+").sum()), len(g), 0)
    elif TRIGGERS.get(rc):
        trig = pct(int(g.body.str.contains(TRIGGERS[rc]).sum()), len(g), 0)
    else:
        trig = None
    rules.append({"code": rc, "ids": len(g), "share": pct(len(g), len(dm)), "definition": RULE_DEFS.get(rc, ""),
                  "trigger_pct": trig, "top_terms": [{"term": k, "pct": v} for k, v in top]})
rules.sort(key=lambda x: -x["ids"])
assert sum(x["ids"] for x in rules) == n_pos_other
n_multi = int(dm.reasoning.str.startswith("Device: [").sum())
lift_mask = dm.narrative.fillna("").str.lower().str.contains("scooter lift|scooter carrier|scooter rack")
n_lift = int(lift_mask.sum())

# ---------------------------------------------------------------- misc facts used in text
miss_lab = miss[L].value_counts().to_dict()
miss_pred = miss["LLM prediction"].value_counts().to_dict()
miss_src = miss.src.value_counts().to_dict()
miss_reg = int(miss_src.get("RegBike", 0))
miss_ag = miss["Agency Proper Name"].value_counts().head(3)
bs_top = blind.TYPE_PROPERTY_DAMAGE.value_counts().head(6)
bs_power = int(blind.reasoning.str.contains("Power: none mentioned").sum())
_bs_conf = pd.read_excel(SRC, "LLM_blindspot_propfield", dtype=str).confidence.astype(float)
blind_conf = float(_bs_conf.mean())
assert _bs_conf.min() == _bs_conf.max(), "blind-spot confidence is not a single value"
bs_ag = blind["Agency Proper Name"].value_counts()
bs_fhp = int(bs_ag.get("Florida Highway Patrol", 0))
es_scooter_prop = int(esc.TYPE_PROPERTY_DAMAGE.fillna("").str.lower().str.contains("scooter").sum())
es_ag = esc["Agency Proper Name"].value_counts().head(5)

eb_pos = m[(m.src == "EBike") & m.pos].copy()
eb_pos["q"] = eb_pos.dt.dt.to_period("Q").astype(str).str.replace("Q", " Q")
qs = eb_pos.groupby("q").agg(n=("lenient", "size"), c=("lenient", "sum")).reset_index()
quarters = [{"q": x.q, "n": int(x.n), "c": int(x.c), "pct": pct(int(x.c), int(x.n))} for x in qs.itertuples()]
q_first, q_last = quarters[0], quarters[-1]
q_min = min(quarters[:-1], key=lambda x: x["pct"])
date_min, date_max = str(m.dt.min().date()), str(m.dt.max().date())

pos_df = m[m.pos]
agencies = []
for a, g in pos_df.groupby("Agency Proper Name"):
    if len(g) >= AGENCY_MIN:
        agencies.append({"agency": a, "positives": len(g), "disagree_pct": pct(int((~g.lenient).sum()), len(g)),
                         "blind_spot": int(bs_ag.get(a, 0)), "escooter_pct": pct(int((g["LLM prediction"] == "E-scooter").sum()), len(g))})
agencies.sort(key=lambda x: -x["disagree_pct"])
ag = {x["agency"]: x for x in agencies}
fhp, miami = ag["Florida Highway Patrol"], ag["Miami Police Department"]
fhp_dis_n = round(fhp["disagree_pct"] / 100 * fhp["positives"])
fhp_fixed = pct(fhp_dis_n - bs_fhp, fhp["positives"])
overall_dis = pct(int((~pos_df.lenient).sum()), len(pos_df))
n_agencies_pos = int(pos_df["Agency Proper Name"].nunique())

forms = [{"form": k, "n": len(g), "agree": int(g.agree.sum()), "pct": pct(int(g.agree.sum()), len(g))} for k, g in m.groupby("FORM_TYPE")]
fm = {x["form"]: x for x in forms}

cdx = pd.read_excel(SRC, "Chart_Data", header=None)
ci = cdx.index[cdx[0] == "LLM prediction"][0]
confidence = {str(cdx.iloc[ci + 1 + i, 0]): float(cdx.iloc[ci + 1 + i, 1]) for i in range(4)}

kpis = {
    "ids_total": ids_total, "matched": matched, "unmatched": unmatched, "match_rate": pct(matched, ids_total),
    "strict": {"agree": agree_s, "kappa": kappa(tp_s, fp_s, fn_s, tn_s)},
    "lenient": {"agree": agree_l, "kappa": kappa(tp_l, fp_l, fn_l, tn_l)},
    "precision": precision, "recall": recall, "f1": f1, "pos_n": pos_n, "neg_n": neg_n,
    "flhsmv_pos_rate": pct(pos_n, matched), "llm_pos_rate": pct(int(m.lenient.sum()), matched),
    "n_miss": n_miss, "n_esc": n_esc, "n_blind": n_blind, "n_pos_other": n_pos_other, "n_pos_bic": n_pos_bic,
    "n_bic_other": n_bic_other, "hidden_pct": pct(n_miss, neg_n), "fix_upper": fix_upper,
    "dup_matched_ids": dup_matched, "conflicting": conflict, "un_pos": un_pos,
    "disagree_lenient": matched - (tp_l + tn_l), "overall_dis": overall_dis, "blind_conf": blind_conf, "agency_min": AGENCY_MIN,
}
E_, R_ = sheets["EBike"], sheets["RegBike"]
Dh = LT["Definitely"] + LT["Highly Likely"]
lp = llm_pos

# ---------------------------------------------------------------- narrative insights (big picture)
bottom_line = (
    f"The LLM and FLHSMV tell broadly the same story, and most differences trace back to how each is defined. FLHSMV's ratings are keyword-based (\"electric\", \"bike\", \"scooter\", \"motorized\"), "
    f"look at both the narrative and the property-damage field, and group e-bikes and e-scooters together. The LLM reads only the narrative and needs explicit evidence of both the device and its electric power before it will say e-bike or e-scooter. "
    f"So when the LLM does name one, FLHSMV had already rated it likely {precision}% of the time. The gaps come from three places: the LLM was never shown the property-damage column, where officers often wrote \"electric bicycle\" ({n(n_blind)} reports); "
    f"the LLM's rules deliberately send vague wording (a bare \"scooter\" or \"bike\", \"motorized scooter\" alone, several devices at once, no device) to \"Other\" where FLHSMV's keywords gave a likely rating ({n(n_pos_other)} reports); "
    f"and the LLM separates e-scooters from e-bikes while FLHSMV does not ({n(n_esc)} reports, a disagreement only under the strict view). A small group ({n_miss}) went the other way: FLHSMV rated them unlikely, yet the narrative describes an e-bike or e-scooter. "
    f"Neither source is treated as the final truth; settling the disputed cases would take a human review."
)
insights = [
 dict(title="When the LLM spots an e-bike or e-scooter, FLHSMV almost always agrees",
  text=f"Of every {n(lp['E-bike']['n'] + lp['E-scooter']['n'])} reports the LLM called an e-bike or e-scooter, FLHSMV had rated all but a handful as at least \"Likely\". "
       f"So the LLM's positive calls are very dependable. Its weakness is the other direction: it recovers about {recall}% of the reports FLHSMV rated likely, because it is stricter about what counts.",
  evidence=f"Precision {precision}%, recall {recall}%, F1 {f1}% (FLHSMV treated as the reference)."),
 dict(title="Much of the \"disagreement\" is a difference in definition, not a mistake",
  text=f"FLHSMV's own rating definition covers \"an E-Bike or E-Scooter\" together, so the LLM's e-scooter calls are consistent with it. The LLM keeps them separate, following the Florida legal split (an e-bike has pedals and a motor; an e-scooter has no pedals). "
       f"That is why the comparison jumps from {agree_s}% (strict, only LLM E-bike counts) to {agree_l}% (lenient, E-bike or E-scooter counts). The lenient figure is the like-for-like one. "
       f"In {pct(es_scooter_prop, n_esc, 0):.0f}% of these cases FLHSMV's own property-damage field also says \"scooter\".",
  evidence=f"{n(n_esc)} reports FLHSMV rated Definitely/Highly Likely that the LLM called E-scooter; {n(es_scooter_prop)} mention a scooter in the property-damage field."),
 dict(title="The LLM was blind to a column that held the answer",
  text=f"The LLM read the crash narrative but was not given the property-damage field (TYPE_PROPERTY_DAMAGE). In {n(n_blind)} reports FLHSMV rated the crash \"Definitely\" an e-bike, the narrative did not mention a motor (true for {n(bs_power)} of them), and the property field said things like \"electric bicycle\". "
       f"The LLM called these regular bicycles with high confidence. Here FLHSMV is right and the LLM simply lacked the information. Giving the LLM that column would likely lift agreement to about {fix_upper}%.",
  evidence=f"{n(n_blind)} reports; {n(bs_power)} of them say the narrative mentions no power source; best case agreement {agree_l}% to about {fix_upper}%."),
 dict(title="FLHSMV's \"Likely\" rating is loose; its top two ratings are solid",
  text=f"The LLM confirms most \"Highly Likely\" reports ({ladder['Highly Likely']['lenient']}%) and, once the blind spot is credited, most \"Definitely\" reports ({ladder['Definitely']['credited']}%). "
       f"\"Likely\" is different: the LLM confirms only {ladder['Likely']['lenient']}% and calls {pct(CT['Likely']['Other'], LT['Likely'], 0):.0f}% of them \"Other\". "
       f"FLHSMV defines \"Likely\" as keywords such as \"electric\" or \"motorized\" appearing just once, a weaker standard than the LLM's requirement for explicit evidence of both the device and its electric power. "
       f"Counting \"Likely\" as a yes therefore adds many reports that are questionable.",
  evidence=f"Share the LLM confirms: Definitely {ladder['Definitely']['lenient']}% ({ladder['Definitely']['credited']}% with blind spot credited), Highly Likely {ladder['Highly Likely']['lenient']}%, Likely {ladder['Likely']['lenient']}%."),
 dict(title="Where the LLM says \"Other\", it is usually following a rule not to guess",
  text=f"For {n(n_pos_other)} reports FLHSMV rated likely but the LLM said \"Other\". The LLM's prompt tells it to answer \"Other\" whenever the device or its power type is unclear: a bare \"scooter\" or \"bike\", \"motorized scooter\" on its own (it could be gas-powered), several devices in one report, or no device at all. "
       f"About {pct(sum(x['ids'] for x in rules if x['code'] in ('R9','RC3','R8')), n_pos_other, 0):.0f}% fall under just three of those rules. FLHSMV's keyword criteria, by contrast, can reach a likely rating from a single \"motorized\" or \"electric\". "
       f"Keyword matching can also pick up incidental mentions, for example {n_lift} reports where the narrative mentions a scooter lift or carrier on a vehicle. "
       f"In many of these FLHSMV may be right and the LLM too cautious, but the data cannot say which; a human review of a sample would settle it.",
  evidence=f"{n(n_pos_other)} reports ({pct(n_pos_other, pos_n)}% of FLHSMV positives); see the rule table."),
 dict(title="FLHSMV occasionally rates a real e-bike as unlikely",
  text=f"In {n_miss} reports FLHSMV said \"Less Likely\" or \"Not Likely\", yet the narrative describes an e-bike ({miss_pred.get('E-bike', 0)}) or e-scooter ({miss_pred.get('E-scooter', 0)}). Almost all ({miss_reg} of {n_miss}) sit in the regular-bike file. "
       f"That is only {pct(n_miss, neg_n)}% of everything FLHSMV rated unlikely, so FLHSMV's low ratings are mostly reliable, but the LLM could help catch the few it misses.",
  evidence=f"{n_miss} reports; mean LLM confidence {round(pd.read_excel(SRC, 'FLHSMV_missed_LLM_found', dtype=str).confidence.astype(float).mean(), 2)}."),
 dict(title="The two sources disagree more for some agencies than others",
  text=f"Among agencies with at least {AGENCY_MIN} FLHSMV-positive reports, the LLM disagrees most often for {agencies[0]['agency']} ({agencies[0]['disagree_pct']}%), {agencies[1]['agency']} ({agencies[1]['disagree_pct']}%) and {agencies[2]['agency']} ({agencies[2]['disagree_pct']}%), and least for {agencies[-1]['agency']} ({agencies[-1]['disagree_pct']}%). "
       f"Part of the gap is where an agency records \"electric\": for Florida Highway Patrol, {pct(bs_fhp, fhp_dis_n, 0):.0f}% of its disagreements are the property-field blind spot, which would shrink its rate from {fhp['disagree_pct']}% to about {fhp_fixed}%.",
  evidence=f"Overall, the LLM disagrees with {overall_dis}% of FLHSMV-positive reports."),
 dict(title="The LLM sounds equally sure when it is right and when it lacks information",
  text=f"Its self-reported confidence is about {confidence['Bicyclist']:.2f} for Bicyclist, E-bike and E-scooter, and lower ({confidence['Other']:.2f}) for \"Other\", so low confidence does point to ambiguous reports. "
       f"But it cannot catch the blind spot: those {n(n_blind)} reports were all labeled at {blind_conf:.2f}. Confidence is a useful flag for \"Other\", not a safety net.",
  evidence=f"Mean confidence: Bicyclist {confidence['Bicyclist']:.2f}, E-bike {confidence['E-bike']:.2f}, E-scooter {confidence['E-scooter']:.2f}, Other {confidence['Other']:.2f}."),
 dict(title="The share of FLHSMV-positive reports the LLM confirms has risen over time",
  text=f"In the e-bike file, the LLM confirmed {q_first['pct']}% of FLHSMV-positive reports in {q_first['q']} and {q_last['pct']}% in {q_last['q']} (a partial quarter), with a low of {q_min['pct']}% in {q_min['q']}. "
       f"The data cannot say why: narratives may have become more explicit, FLHSMV coding may have changed, or the mix of e-scooters may have shifted.",
  evidence=f"{q_first['q']}: {q_first['pct']}%; {q_last['q']}: {q_last['pct']}%."),
]

# ---------------------------------------------------------------- miscellaneous (numeric notes)
ser = {x["series"]: x for x in id_series}
s24, s28, s80 = ser["Starts with 21-24"], ser["Starts with 25-28"], ser["Starts with 80-90"]
misc = [
 dict(theme="Coverage",
  text=f"Of {n(ids_total)} unique REPORT_NUMBERs ({n(E_['ids'])} in the e-bike file, {n(R_['ids'])} in the regular-bike file, no overlap), {n(matched)} ({pct(matched, ids_total)}%) were found in the FLHSMV files and {n(unmatched)} ({pct(unmatched, ids_total)}%) were not. "
       f"Match rate is {E_['match_rate']}% for the e-bike file and {R_['match_rate']}% for the regular-bike file.",
  figure=f"{n(matched)} matched / {n(unmatched)} unmatched"),
 dict(theme="Why IDs are unmatched",
  text=f"The unmatched IDs are clean 8-digit numbers that simply do not appear in the FLHSMV files, so this is not a formatting problem. They cluster in two places: IDs starting with 80 to 90 ({n(s80['unmatched'])} unmatched, only {s80['match_rate']}% found) and IDs starting with 21 to 24 ({n(s24['unmatched'])} unmatched, {s24['match_rate']}% found), against {s28['match_rate']}% found for IDs starting with 25 to 28. "
       f"The FLHSMV records that did match were loaded between {zip_window[0]} and {zip_window[1]}, so crashes outside that window could be absent. "
       f"Whether the two sources cover different years cannot be tested from this workbook, because unmatched rows carry no FLHSMV date. The cause is therefore not confirmed; "
       + ("the crash-year check below uses your crash export." if crash_year else "rerunning the build script with power_bi_export.csv adds a crash-year check."),
  figure=f"80-90 series {s80['match_rate']}% found; 21-24 series {s24['match_rate']}%; 25-28 series {s28['match_rate']}%"),
 dict(theme="Unmatched reports that still look like e-bikes",
  text=f"Among the {n(unmatched)} unmatched IDs, {n(un_pos)} ({pct(un_pos, unmatched)}%) are ones the LLM called E-bike or E-scooter. These crashes exist in the LLM data but cannot be compared with FLHSMV. "
       f"The FLHSMV files may be a partial extract, so confirm before calling these FLHSMV omissions.",
  figure=f"{n(un_pos)} ({pct(un_pos, unmatched)}% of unmatched)"),
 dict(theme="Sample design",
  text=f"The two files are almost pure slices of the FLHSMV rating. In the e-bike file, {n(E_['pos'])} of {n(E_['matched'])} matched IDs are FLHSMV-positive; in the regular-bike file, {n(R_['matched'] - R_['pos'])} of {n(R_['matched'])} are FLHSMV-negative. "
       f"Agreement is {E_['agree_pct']}% in the first and {R_['agree_pct']}% in the second, and the {agree_l}% headline blends them. These numbers describe this sample, not all Florida e-bike crashes.",
  figure=f"{E_['agree_pct']}% / {R_['agree_pct']}%"),
 dict(theme="Duplicate rows",
  text=f"The FLHSMV files repeat some REPORT_NUMBERs (for example a crash with both an L and a U form). {n(dup_matched)} matched IDs have more than one row ({n(dup_lu)} have both an L and a U form), and {n(conflict)} of them carry different FLHSMV ratings. "
       f"The workbook's Joined sheets keep every row ({n(rows_total['EBike'])} and {n(rows_total['RegBike'])} rows; {n(rows_extra['EBike'] + rows_extra['RegBike'])} extra). Everything on this tab uses one row per ID, keeping the highest FLHSMV rating.",
  figure=f"{n(dup_matched)} IDs with duplicates"),
 dict(theme="Form type",
  text=f"Agreement is similar across FLHSMV form types: {fm['L']['pct']}% for L, {fm['U']['pct']}% for U, {fm['S']['pct']}% for S (V has only {fm['V']['n']} IDs), so form type does not explain the disagreement.",
  figure=f"L {fm['L']['pct']}% / U {fm['U']['pct']}% / S {fm['S']['pct']}%"),
 dict(theme="Agreement statistics",
  text=f"Cohen's kappa corrects agreement for chance. It is {kpis['strict']['kappa']:.2f} for the strict definition and {kpis['lenient']['kappa']:.2f} when e-scooters count. It is low even at {agree_l}% agreement because FLHSMV rates {kpis['flhsmv_pos_rate']}% of matched reports positive while the LLM rates {kpis['llm_pos_rate']}%, and because the sample is built from two slices of the FLHSMV rating.",
  figure=f"k {kpis['strict']['kappa']:.2f} / {kpis['lenient']['kappa']:.2f}"),
 dict(theme="E-scooters by agency",
  text=f"E-scooter cases are concentrated in a few agencies: " + ", ".join(f"{k} ({v})" for k, v in es_ag.items()) + f". At Miami Police Department about {miami['escooter_pct']:.0f}% of FLHSMV-positive reports are E-scooter, yet its disagreement rate is only {miami['disagree_pct']}% because e-scooters count as a match there; under the strict definition it would look very different.",
  figure=f"Miami PD ~{miami['escooter_pct']:.0f}% E-scooter"),
]

# ---------------------------------------------------------------- examples (evidence computed from the data)
def kw(text, pats):
    t = str(text).lower()
    for p in pats:
        x = re.search(p, t)
        if x:
            return x.group(0)
    return None

def ex_row(rid, category, evidence_fn):
    row = d[d.REPORT_NUMBER == rid]
    if row.empty:
        return None
    x = row.iloc[0]
    return {"id": rid, "category": category, "agency": x["Agency Proper Name"], "flhsmv": x[L], "llm": x["LLM prediction"],
            "note": evidence_fn(x)}

narr_kw = lambda pats: (lambda x: f"The narrative contains the phrase \"{kw(x.narrative, pats)}\"." if kw(x.narrative, pats) else "See narrative in the workbook.")
prop_note = lambda x: f"Property-damage field: {x.TYPE_PROPERTY_DAMAGE}."
rule_note = lambda x: f"LLM rule {parent_rule(x.reasoning)} sent it to Other; the reasoning lists {'more than one device' if str(x.reasoning).startswith('Device: [') else 'one device'}."
examples = [e_ for e_ in [
    ex_row("27371310", "FLHSMV missed, LLM found", narr_kw([r"e[- ]?bicycle", r"e[- ]?bike", r"electric bi\w+"])),
    ex_row("26858191", "FLHSMV missed, LLM found", narr_kw([r"electric bicycl\w+", r"e[- ]?bike"])),
    ex_row("26336300", "FLHSMV missed, LLM found", narr_kw([r"electronic bicycle", r"electric bicycle", r"e[- ]?bike"])),
    ex_row("26790524", "FLHSMV missed, LLM found", narr_kw([r"e[- ]?bicycle", r"e[- ]?bike", r"electric bicycle"])),
    ex_row("26330175", "FLHSMV-positive, LLM Other", prop_note),
    ex_row("25930833", "FLHSMV-positive, LLM Other", prop_note),
    ex_row("26198526", "FLHSMV-positive, LLM Other (several devices)", rule_note),
    ex_row(str(dm[lift_mask].REPORT_NUMBER.iloc[0]) if n_lift else "", "FLHSMV-positive, LLM Other (incidental mention)",
           lambda x: f"The narrative mentions a scooter lift (equipment on a vehicle); LLM rule {parent_rule(x.reasoning)} sent it to Other."),
    ex_row("27786286", "LLM blind spot (property field)", prop_note),
    ex_row("26676979", "E-scooter scope difference", prop_note),
] if e_]
assert len(examples) >= 8, "some example IDs were not found in the workbook"

# example-sheet guide (row counts read from the sheets themselves)
sh_len = {s: len(pd.read_excel(SRC, s, dtype=str)) for s in
          ["FLHSMV_missed_LLM_found", "FLHSMV_false_pos_LLM_Other", "FLHSMV_ebike_LLM_escooter", "LLM_blindspot_propfield"]}
example_sheets = [
    {"sheet": "FLHSMV_missed_LLM_found", "rows": sh_len["FLHSMV_missed_LLM_found"],
     "what": "FLHSMV rated Less/Not Likely, LLM found an e-bike or e-scooter.", "note": "Matches the \"FLHSMV misses\" count."},
    {"sheet": "FLHSMV_false_pos_LLM_Other", "rows": sh_len["FLHSMV_false_pos_LLM_Other"],
     "what": "FLHSMV rated Likely or higher, LLM said Other.",
     "note": f"A subset of the {n(n_pos_other)} FLHSMV-positive, LLM-Other reports; how this subset was chosen is not recorded in the workbook. The tab's numbers use all {n(n_pos_other)}."},
    {"sheet": "FLHSMV_ebike_LLM_escooter", "rows": sh_len["FLHSMV_ebike_LLM_escooter"],
     "what": "FLHSMV Definitely/Highly Likely, LLM E-scooter.", "note": "Matches the scope-difference count."},
    {"sheet": "LLM_blindspot_propfield", "rows": sh_len["LLM_blindspot_propfield"],
     "what": "FLHSMV Definitely, LLM Bicyclist, and the property-damage field names an electric bicycle.", "note": "Matches the blind-spot count."},
]

caveats = [
    "FLHSMV's rating is assigned by FLHSMV reviewers using the keyword criteria in the legend above, and the LLM is a model reading the narrative. Neither is treated as the final truth: where they disagree, this tab shows the disagreement and the likely reason, not a verdict.",
    "FLHSMV's criteria use the narrative and the property-damage fields; the LLM was given only the narrative. Some disagreement is therefore about what each source was allowed to see.",
    "Rule codes and their definitions come from the LLM's classification prompt. The LLM sometimes cites a code whose trigger wording does not appear in its own reasoning (see the last column of the rule table), so a code is a guide to why, not proof.",
    f"The {n(n_pos_other)} FLHSMV-positive reports the LLM called Other, and the {n(n_bic_other)} FLHSMV-positive reports it called Bicyclist without a property-field explanation, have not been hand-reviewed.",
    "This tab is a fixed comparison on the matched sample and ignores the sidebar filters.",
    f"Unmatched IDs ({n(unmatched)}) are excluded from every agreement figure, and the matched sample is made of two slices of the FLHSMV rating.",
]

# ---------------------------------------------------------------- provenance
provenance = [
    {"item": "All counts, agreement, kappa, precision/recall", "source": "Joined_EBike + Joined_RegBike", "how": "One row per REPORT_NUMBER (highest FLHSMV rating, ties to latest LOAD_DT); FLHSMV positive = Likely or higher; LLM positive = E-bike (strict) or E-bike/E-scooter (lenient)."},
    {"item": "Scope difference", "source": "Joined sheets (cross-checked with FLHSMV_ebike_LLM_escooter)", "how": "FLHSMV Definitely/Highly Likely and LLM E-scooter."},
    {"item": "Blind spot", "source": "Joined sheets (cross-checked with LLM_blindspot_propfield)", "how": f"FLHSMV Definitely, LLM Bicyclist, TYPE_PROPERTY_DAMAGE matches /{ELECTRIC_PROP}/."},
    {"item": "FLHSMV-positive, LLM Other / Bicyclist", "source": "Joined sheets", "how": "FLHSMV Likely or higher and LLM Other (or Bicyclist)."},
    {"item": "FLHSMV misses", "source": "Joined sheets (cross-checked with FLHSMV_missed_LLM_found)", "how": "FLHSMV Less/Not Likely and LLM E-bike or E-scooter."},
    {"item": "Rule codes", "source": "reasoning column, Joined sheets", "how": "Last \"Rule ...\" code in the LLM reasoning for each FLHSMV-positive, LLM-Other report; device wording counted from the reasoning's Device field."},
    {"item": "Unmatched IDs and ID series", "source": "Joined sheets", "how": "Rows with blank FLHSMV columns; IDs grouped by their first two digits."},
    {"item": "FLHSMV rating definitions", "source": "FLHSMV E-Bike Likeliness legend", "how": "Quoted from the legend; report counts per rating come from the Joined sheets."},
    {"item": "Rule definitions", "source": "LLM classification prompt", "how": "Paraphrased from the prompt given to the model; trigger-wording percentages are counted from the reasoning column."},
    {"item": "Trend and agencies", "source": "Joined sheets", "how": f"LOAD_DT quarter of the FLHSMV record; agency = Agency Proper Name; agencies with {AGENCY_MIN}+ FLHSMV-positive reports."},
    {"item": "LLM confidence by prediction", "source": "Chart_Data sheet", "how": "Read from the workbook's static confidence table (the Joined sheets have no confidence column)."},
]

# ---------------------------------------------------------------- cross-checks against the workbook's own sheets
checks = []
def check(name, mine, theirs):
    ok = mine == theirs
    checks.append({"check": name, "computed": mine, "workbook": theirs, "ok": ok})
    print(("OK   " if ok else "DIFF ") + f"{name}: computed {mine} vs workbook {theirs}")
check("scope-difference IDs vs FLHSMV_ebike_LLM_escooter rows", n_esc, sh_len["FLHSMV_ebike_LLM_escooter"])
check("blind-spot IDs vs LLM_blindspot_propfield rows", n_blind, sh_len["LLM_blindspot_propfield"])
check("FLHSMV misses vs FLHSMV_missed_LLM_found rows", n_miss, sh_len["FLHSMV_missed_LLM_found"])
sm = pd.read_excel(SRC, "Summary", header=None)
sm_map = {str(a): b for a, b in zip(sm[0], sm[1])}
for k_, v_ in sm_map.items():
    kl = k_.lower()
    if "matched" in kl and "unmatched" not in kl and "ids" in kl:
        check(f"Summary: {k_}", matched, int(v_))
    if "unmatched" in kl:
        check(f"Summary: {k_}", unmatched, int(v_))
assert all(c["ok"] for c in checks if "Summary" not in c["check"] or True), "cross-check failed -- see output above"

out = {
    "schema": 2, "generated": str(date.today()), "source_workbook": os.path.basename(SRC),
    "load_dt_range": [date_min, date_max], "labels": LABELS, "preds": PREDS,
    "kpis": kpis, "crosstab": CT, "label_totals": LT, "ladder": ladder, "llm_pos_share": llm_pos,
    "breakdown": breakdown, "sheets": sheets, "unmatched_by_sheet": unmatched_by,
    "id_series": id_series, "id_prefixes": by_prefix, "zip_window": zip_window, "crash_year_check": crash_year,
    "quarters": quarters, "agencies": agencies, "forms": forms, "rules": rules, "rules_total": n_pos_other, "rules_multi_device": n_multi,
    "flhsmv_defs": flhsmv_defs, "labels_unexpected": labels_unexpected,
    "confidence": confidence, "bottom_line": bottom_line, "insights": insights, "misc": misc,
    "examples": examples, "example_sheets": example_sheets, "caveats": caveats,
    "provenance": provenance, "checks": checks,
}
os.makedirs("results", exist_ok=True)
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", OUT, f"({os.path.getsize(OUT) / 1024:.0f} KB)")
