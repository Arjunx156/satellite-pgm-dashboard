"""Predictive Satellite Telemetry Monitoring: Streamlit entry point.

Run with:  streamlit run app.py
"""

import streamlit as st

from ui import theme
from ui.state import load_model_or_stop, sidebar

st.set_page_config(page_title="Predictive Satellite Telemetry Monitoring", page_icon="assets/icon.svg",
                   layout="wide", initial_sidebar_state="expanded")
theme.apply()

pages = [
    st.Page("views/predictive_monitoring.py", title="Predictive Monitoring", default=True),
    st.Page("views/telemetry_analysis.py", title="Telemetry Analysis"),
    st.Page("views/model_analysis.py", title="Model Analysis"),
    st.Page("views/decision_support.py", title="Decision Support"),
]
nav = st.navigation(pages)

model = load_model_or_stop()
sidebar(model)
nav.run()
