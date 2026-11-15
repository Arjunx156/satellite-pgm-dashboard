"""Reusable dashboard pieces: KPI cards, notes, the shared header and empty states."""

import streamlit as st

from backend import config

from .theme import ACCENT, ANOMALY, MUTED, esc, risk_colour, state_colour

DISCLAIMER = ("Utility values are hypothetical project decision-support assumptions. "
              "They are not real spacecraft control commands.")
TARGET = ("The prediction is the probability that the next telemetry segment is anomalous. "
          "It does not predict spacecraft, component or mission failure.")


def kpi(label, value, sub="", colour=None, small=False, primary=False, bar=None):
    style = f' style="color:{colour}"' if colour else ""
    bar_html = ""
    if bar is not None:
        bar_html = (f'<div class="bar"><span style="width:{100 * bar:.1f}%;'
                    f'background:{risk_colour(bar)}"></span></div>')
    st.markdown(
        f'<div class="kpi{" primary" if primary else ""}"><div class="label">{esc(label)}</div>'
        f'<div class="value{" small" if small else ""}"{style}>{esc(value)}</div>'
        f'<div class="sub">{sub}</div>{bar_html}</div>', unsafe_allow_html=True)


def pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def note(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def page_header(title: str, result=None):
    st.title(title)
    if result is None:
        return
    if result.mode == "dataset":
        mode = "Historical Telemetry Analysis (replay of recorded segments)"
        where = (f"segment <b>{result.segment_id}</b> &middot; channel <b>{esc(result.channel)}</b> "
                 f"&middot; position <b>{result.position}</b> in channel &middot; {result.split} split")
        if result.split == "train":
            where += (" &middot; this segment was used to learn the models, so its probabilities are "
                      "in-sample. Pick a test-split segment for an out-of-sample view.")
    else:
        mode = "Telemetry Scenario Analysis (manually entered features)"
        ctx = "after recorded channel history" if result.has_history else "isolated segment, HMM starts from &pi;"
        where = f"channel <b>{esc(result.channel)}</b> &middot; {ctx}"
    st.markdown(f'<div class="kicker"><b>{mode}</b><br>{where}</div>', unsafe_allow_html=True)


def show_input_messages(result):
    if result.filled:
        names = ", ".join(config.FEATURE_LABELS[f].lower() for f in result.filled)
        st.info(f"Missing values were filled with the training median, as in the notebook: {names}.")
    for w in result.warnings:
        st.warning(w)


def need_result():
    """Return the current prediction, or show why there isn't one and stop the page."""
    result = st.session_state.get("result")
    error = st.session_state.get("input_error")
    if error:
        st.error(error)
    if result is None:
        if not error:
            st.info("Choose a dataset segment or enter telemetry features in the sidebar, "
                    "then run the prediction.")
        st.stop()
    return result


def state_kpi(label, state, sub):
    kpi(label, state.upper(), sub, colour=state_colour(state), small=True)


def action_kpi(action, sub):
    colour = {"Continue": ACCENT, "Monitor": "#E4E9F2", "Safe Mode": ANOMALY}[action]
    kpi("Recommended action", action.upper(), sub, colour=colour, small=True)


__all__ = ["kpi", "pct", "note", "page_header", "show_input_messages", "need_result",
           "state_kpi", "action_kpi", "DISCLAIMER", "TARGET", "MUTED"]
