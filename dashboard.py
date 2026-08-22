"""
Complete Streets Active-Mode Crash Dashboard -- legacy entry point.

Streamlit Cloud is configured to run this file. It delegates to app.py,
which wires together dashboard_core.py and tabs/*.py.

Run with either:
    streamlit run dashboard.py
    streamlit run app.py
"""

import os
import runpy

runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"),
    run_name="__main__",
)
