"""
Complete Streets Active-Mode Crash Dashboard -- entry point.
Just & Green Transportation Lab | University of Florida

Run with:
    streamlit run app.py

This file just wires together dashboard_core.py (imports, constants, data
loading, sidebar, filters, shared helpers) and tabs/*.py (one file per tab).
Each tab file is exec()'d inside its own `with tabN:` block so it can
reference df / sel_modes / MODE_COLORS / etc. directly as plain names,
exactly like the original single-file dashboard.py did -- no context object
or import juggling, and it still reruns correctly on every Streamlit
interaction (dashboard_core.py's top-level code -- including the sidebar
widgets and filtering -- re-executes fresh on every run too, same as before).
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_HERE, "dashboard_core.py")) as f:
    exec(compile(f.read(), "dashboard_core.py", "exec"), globals())

# ============================================================================
# TABS
# ============================================================================
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "\u2139\ufe0f About This Dashboard",
    "\U0001F4C8 Overview & Trends",
    "\U0001F6A8 Severity & Outcomes",
    "\U0001F55B When & Where",
    "\U0001F464 Driver Behavior & Citations",
    "\U0001F6E3 Roadway Infrastructure",
    "\U0001F9D1 Demographics",
    "\U0001F4DD Narrative, Typing & Hotspots",
    "\U0001F50D Crash Causation",
    "\U0001F4CC Insights",
])

TAB_FILES = [
    (tab0, "tab0_about.py"),
    (tab1, "tab1_overview.py"),
    (tab2, "tab2_severity.py"),
    (tab3, "tab3_when_where.py"),
    (tab4, "tab4_driver_behavior.py"),
    (tab5, "tab5_infrastructure.py"),
    (tab6, "tab6_demographics.py"),
    (tab7, "tab7_narrative.py"),
    (tab8, "tab8_causation.py"),
    (tab9, "tab9_insights.py"),
]
_TABS_DIR = os.path.join(_HERE, "tabs")

for _tab, _filename in TAB_FILES:
    with _tab:
        _path = os.path.join(_TABS_DIR, _filename)
        with open(_path) as _f:
            exec(compile(_f.read(), _filename, "exec"), globals())

st.markdown(
    """<div style="text-align:center; color:#9aa0b8; font-size:0.78rem; padding: 1.2rem 0 0.4rem 0;">
    Data: Signal4 crash tables + FDOT roadway tables, processed via eda_analysis_combined.py &middot;
    Just & Green Transportation Lab, University of Florida
    </div>""",
    unsafe_allow_html=True,
)
