"""Model loading (once per server process) and the sidebar that produces the current prediction."""

import numpy as np
import pandas as pd
import streamlit as st

from backend import TelemetryModel, config
from backend.data_loader import DataError

MODE_DATASET = "Select existing dataset segment"
MODE_MANUAL = "Manual telemetry features"


@st.cache_resource(show_spinner="Learning TAN, Markov network, HMM and Decision Tree from the training split...")
def get_model() -> TelemetryModel:
    return TelemetryModel.build()


def load_model_or_stop() -> TelemetryModel:
    try:
        return get_model()
    except DataError as exc:
        st.error(f"Could not load the dataset. {exc}")
    except Exception as exc:                          # surface, don't hide, unexpected failures
        st.error("Model initialisation failed.")
        st.exception(exc)
    st.stop()


def _default_segment(model):
    """The notebook's end-to-end example: the highest-risk test segment (Section 14)."""
    te = np.where(model.data.test_mask)[0]
    row = int(te[np.argmax(model.p_filt[te])])
    channel = model.data.df.at[row, "channel"]
    position = int(np.where(model.channel_rows(channel) == row)[0][0])
    return channel, position


def _fmt(x) -> str:
    return f"{x:.6g}"


def _dataset_inputs(model):
    df = model.data.df
    channels = model.channels()
    if "ds_channel" not in st.session_state:
        st.session_state.ds_channel, st.session_state.ds_pos = _default_segment(model)

    channel = st.selectbox("Channel", channels, key="ds_channel",
                           format_func=lambda c: f"{c}  ({len(model.channel_rows(c))} segments)")
    ix = model.channel_rows(channel)
    n = len(ix)
    st.session_state.ds_pos = min(int(st.session_state.get("ds_pos", 0)), n - 1)

    c1, c2 = st.columns(2)
    if c1.button("Previous", width="stretch", disabled=st.session_state.ds_pos <= 0):
        st.session_state.ds_pos -= 1
    if c2.button("Next segment", width="stretch", disabled=st.session_state.ds_pos >= n - 1):
        st.session_state.ds_pos += 1
    position = st.number_input(f"Position in channel (0 to {n - 1})", min_value=0, max_value=n - 1,
                               step=1, key="ds_pos")
    row = ix[position]
    st.caption(f"Segment id {int(df.at[row, 'segment'])}, starts "
               f"{df.at[row, 'start_time']:%Y-%m-%d %H:%M} UTC, "
               f"{'train' if model.data.train_mask[row] else 'test'} split.")
    return model.predict_segment(channel, int(position))


def _parse(text: str):
    text = (text or "").strip()
    if text == "":
        return None
    return text            # numeric validation happens in the backend


def _manual_inputs(model):
    disc = model.disc
    if "manual_values" not in st.session_state:
        st.session_state.manual_values = {f: _fmt(disc.fill[f]) for f in disc.feats}
        st.session_state.manual_values.update(channel=model.channels()[0], sampling=disc.levels["sampling"][0])

    prefill_from = st.selectbox(
        "Prefill from", ["Training medians"] + [f"Segment {s}" for s in model.data.df["segment"].sort_values()],
        help="Load a starting point, then edit any value.", key="prefill_from")
    if st.button("Load values", width="stretch"):
        if prefill_from == "Training medians":
            vals = {f: _fmt(disc.fill[f]) for f in disc.feats}
            vals.update(channel=st.session_state.manual_values["channel"],
                        sampling=st.session_state.manual_values["sampling"])
        else:
            row = model.data.df.loc[model.data.df["segment"] == int(prefill_from.split()[-1])].iloc[0]
            vals = {f: _fmt(row[f]) for f in disc.feats}
            vals.update(channel=row["channel"], sampling=row["sampling"])
        st.session_state.manual_values = vals
        for k in list(st.session_state):
            if k.startswith("mf_"):
                del st.session_state[k]
        st.session_state.pop("manual_record", None)

    vals = st.session_state.manual_values
    with st.form("manual_form", border=False):
        channel = st.selectbox("Channel", model.channels(), index=model.channels().index(vals["channel"]),
                               key="mf_channel")
        levels = disc.levels["sampling"]
        sampling = st.selectbox("Sampling rate", levels, index=levels.index(vals["sampling"]), key="mf_sampling")
        record = {"channel": channel, "sampling": sampling}
        for f in disc.feats:
            lo, hi = disc.ranges[f]
            record[f] = _parse(st.text_input(
                config.FEATURE_LABELS[f], value=vals[f], key=f"mf_{f}",
                help=f"`{f}`. Training range {lo:.4g} to {hi:.4g}. Leave empty to use the training median "
                     f"({disc.fill[f]:.4g}). Scientific notation such as 3.9e-10 is accepted."))
        continue_history = st.radio(
            "Sequence context", ["Isolated segment", "Append after channel history"], key="mf_context",
            help="Isolated: the HMM starts from the learned initial distribution π. Append: the HMM starts "
                 "from the filtered belief of the channel's last recorded segment.") != "Isolated segment"
        submitted = st.form_submit_button("Run prediction", type="primary", width="stretch")

    if submitted:
        st.session_state.manual_record = (record, continue_history)
    if "manual_record" not in st.session_state:
        return None
    record, continue_history = st.session_state.manual_record
    return model.predict_manual(record, continue_history)


def sidebar(model):
    """Draw the input panel and store the prediction (or a readable error) in session state."""
    with st.sidebar:
        st.markdown("### Telemetry input")
        mode = st.radio("Input mode", [MODE_DATASET, MODE_MANUAL], key="input_mode", label_visibility="collapsed")
        st.session_state.input_error = None
        try:
            result = _dataset_inputs(model) if mode == MODE_DATASET else _manual_inputs(model)
        except ValueError as exc:
            result = None
            st.session_state.input_error = str(exc)
            st.error(str(exc))
        st.session_state.result = result
        st.divider()
        st.caption("Historical ESA OPS-SAT-AD telemetry, replayed offline. "
                   "This is not a live satellite feed.")


def model_inputs_frame(model) -> pd.DataFrame:
    disc = model.disc
    rows = []
    for f in disc.feats:
        lo, hi = disc.ranges[f]
        e = disc.edges[f]
        rows.append({"Feature": config.FEATURE_LABELS[f], "Column": f, "Training min": lo,
                     "LOW | MEDIUM cut": e[0], "MEDIUM | HIGH cut": e[1], "Training max": hi,
                     "Median fill": disc.fill[f]})
    return pd.DataFrame(rows)
