"""Colour tokens and the small amount of CSS the dashboard needs."""

import html

import streamlit as st

BG = "#0B111C"
SURFACE = "#121A28"
BORDER = "#22304A"
TEXT = "#E4E9F2"
MUTED = "#8B97AD"
ACCENT = "#5CC8E6"      # the one accent: HMM belief, nominal state, interactive elements
ANOMALY = "#F0735A"     # semantic only: anomalous state / high risk
TAN_LINE = "#A9B4CA"

MONO = "ui-monospace, 'Cascadia Mono', 'SF Mono', Consolas, monospace"
SANS = "'Segoe UI Variable', 'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif"

CSS = f"""
<style>
html, body, [class*="css"] {{ font-family: {SANS}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1320px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h1 {{ font-weight: 650; font-size: 1.9rem !important; }}
h2 {{ font-weight: 600; font-size: 1.3rem !important; margin-top: 1.6rem !important; }}
h3 {{ font-weight: 600; font-size: 1.05rem !important; }}
h4 {{ font-weight: 600; font-size: 0.98rem !important; padding-top: 0.6rem !important; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}

.kicker {{ color: {MUTED}; font-size: 0.85rem; margin: -0.4rem 0 1.4rem 0; }}
.kicker b {{ color: {TEXT}; font-weight: 600; }}

.kpi {{ border: 1px solid {BORDER}; background: {SURFACE}; border-radius: 6px;
        padding: 18px 20px 16px 20px; min-height: 132px; }}
.kpi .label {{ color: {MUTED}; font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; }}
.kpi .value {{ font-family: {MONO}; font-size: 2.6rem; font-weight: 600; line-height: 1.1;
               margin: 6px 0 4px 0; font-variant-numeric: tabular-nums; }}
.kpi .value.small {{ font-size: 1.55rem; font-family: {SANS}; letter-spacing: 0.02em; }}
.kpi .sub {{ color: {MUTED}; font-size: 0.82rem; line-height: 1.4; }}
.kpi .sub code {{ font-size: 0.78rem; }}
.kpi.primary {{ border-color: {ACCENT}55; min-height: 150px; }}
.bar {{ height: 4px; background: {BORDER}; border-radius: 2px; margin-top: 10px; overflow: hidden; }}
.bar > span {{ display: block; height: 100%; border-radius: 2px; }}

.steps {{ counter-reset: s; list-style: none; padding-left: 0; margin: 0; }}
.steps li {{ counter-increment: s; position: relative; padding: 0 0 12px 34px; line-height: 1.5; }}
.steps li::before {{ content: counter(s); position: absolute; left: 0; top: 0; width: 22px; height: 22px;
                    border: 1px solid {BORDER}; border-radius: 50%; font-family: {MONO};
                    font-size: 0.75rem; text-align: center; line-height: 21px; color: {MUTED}; }}
.steps li span.num {{ font-family: {MONO}; color: {TEXT}; }}

.note {{ color: {MUTED}; font-size: 0.85rem; line-height: 1.55; border-left: 2px solid {BORDER};
         padding: 2px 0 2px 12px; margin: 8px 0 16px 0; max-width: 78ch; }}
.mono {{ font-family: {MONO}; font-variant-numeric: tabular-nums; }}
footer {{ visibility: hidden; }}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def state_colour(state: str) -> str:
    return ANOMALY if state == "Anomalous" else ACCENT


def risk_colour(p: float) -> str:
    return ANOMALY if p >= 0.5 else ACCENT


def esc(text) -> str:
    return html.escape(str(text))
