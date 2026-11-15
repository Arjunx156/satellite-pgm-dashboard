"""Headless smoke test of every page and both input modes, using Streamlit's AppTest.

AppTest does not re-run an st.navigation entrypoint on later reruns, so each case presets the
sidebar state and renders in one pass. Widget interaction is checked by hand in the browser.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"
PAGES = ["views/predictive_monitoring.py", "views/telemetry_analysis.py",
         "views/model_analysis.py", "views/decision_support.py"]
MANUAL = "Manual telemetry features"
RECORD = {"channel": "CADC0872", "sampling": 1, "kurtosis": "-1.0", "skew": "0.5", "n_peaks": "3",
          "smooth10_n_peaks": "2", "smooth20_n_peaks": "2", "diff2_peaks": "10", "diff_var": "1e-11",
          "var": "4e-10", "len": "280"}


def render(page, state=None):
    at = AppTest.from_file(str(APP), default_timeout=120)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    if page == PAGES[0]:
        at.run()
    else:
        at.run()
        at.switch_page(page).run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_dataset_mode_pages(page):
    at = render(page)
    assert at.session_state.result.mode == "dataset"


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("history", [False, True])
def test_manual_mode_pages(page, history):
    at = render(page, {"input_mode": MANUAL, "manual_record": (RECORD, history)})
    r = at.session_state.result
    assert r.mode == "manual" and r.has_history == history


def test_manual_missing_values_are_filled():
    record = {"channel": "CADC0873", "sampling": 5, "var": None}
    at = render(PAGES[0], {"input_mode": MANUAL, "manual_record": (record, False)})
    assert "var" in at.session_state.result.filled
    assert any("training median" in i.value for i in at.info)


def test_manual_invalid_value_shows_error_not_traceback():
    at = render(PAGES[0], {"input_mode": MANUAL, "manual_record": ({**RECORD, "var": "abc"}, False)})
    assert at.session_state.result is None
    assert any("valid number" in e.value for e in at.error)


def test_out_of_range_value_warns():
    at = render(PAGES[0], {"input_mode": MANUAL, "manual_record": ({**RECORD, "kurtosis": "900"}, False)})
    assert any("outside the training range" in w.value for w in at.warning)


def test_first_and_last_segment_of_channel():
    for pos in (0, 193):
        at = render(PAGES[0], {"ds_channel": "CADC0874", "ds_pos": pos})
        assert at.session_state.result.position == pos
