"""Page 1: current and next-segment anomaly risk, the timeline, and how the numbers were made."""

import streamlit as st

from ui import charts, theme
from ui.components import TARGET, action_kpi, kpi, need_result, note, page_header, pct, show_input_messages, state_kpi
from ui.state import load_model_or_stop

model = load_model_or_stop()
result = need_result()
A = model.A

page_header("Predictive Satellite Telemetry Monitoring", result)
show_input_messages(result)

c1, c2 = st.columns(2, gap="medium")
with c1:
    kpi("Current anomaly risk", pct(result.current_hmm_risk),
        f"HMM filtering on this and earlier segments &middot; "
        f"TAN alone {pct(result.current_tan_risk)} &middot; {result.risk_band.lower()}",
        colour=theme.risk_colour(result.current_hmm_risk), primary=True, bar=result.current_hmm_risk)
with c2:
    kpi("Predicted next-segment risk", pct(result.next_anomaly_risk),
        f"current state projected one segment ahead &middot; {result.next_risk_band.lower()}",
        colour=theme.risk_colour(result.next_anomaly_risk), primary=True, bar=result.next_anomaly_risk)

st.write("")
c3, c4, c5 = st.columns(3, gap="medium")
with c3:
    state_kpi("Current state", result.current_state,
              "more likely state right now")
with c4:
    state_kpi("Predicted next state", result.predicted_next_state,
              "more likely state of the next segment")
with c5:
    action_kpi(result.recommended_action,
               f"highest expected utility at current risk &middot; next segment: "
               f"{result.next_recommended_action}")

if result.mode == "dataset":
    def lab(v):
        return "anomalous" if v == 1 else "nominal"
    nxt = (f"the recorded next segment is labelled <b>{lab(result.next_actual_label)}</b>"
           if result.next_actual_label is not None else "this is the last recorded segment of the channel")
    note(f"Replay check: this segment is labelled <b>{lab(result.actual_label)}</b> in the dataset, and {nxt}. "
         "Labels are shown for evaluation only; the prediction never reads them.")

st.subheader("Anomaly risk timeline")
if result.has_history:
    left, right = st.columns([3, 1])
    with right:
        window = st.slider("Segments of history", 10, 120, 30, step=5)
        show_labels = st.toggle("Show recorded labels", value=True, disabled=result.mode == "manual")
    with left:
        note("Past and current points are recorded segments of this channel. The hollow ring at +1 is the "
             "model's prediction for the next segment; it is not observed data.")
    st.plotly_chart(charts.risk_timeline(result, window, show_labels), width="stretch",
                    config=charts.PLOT_CONFIG)
else:
    note("A manually entered isolated segment has no recorded history, so there is no timeline to draw. "
         "The chart compares the current estimate with the one-step prediction instead. Choose "
         "<b>Append after channel history</b> in the sidebar to place the scenario after a real channel history.")
    st.plotly_chart(charts.current_vs_next(result), width="stretch", config=charts.PLOT_CONFIG)

lo, hi = A[0, 1], A[1, 1]
note(f"With the learned transition matrix, next-segment risk always lies between "
     f"{pct(lo)} and {pct(hi)}, however high or low the current risk is. One step ahead, the HMM regresses towards its long-run anomaly rate. "
     "That is why the predicted next state is nominal unless the transition matrix itself says otherwise. "
     "Dotted lines mark the Decision Network action boundaries.")

st.subheader("How was this prediction made?")
weights = result.features.sort_values("Evidence weight", ascending=False)
top = ", ".join(f"{r.Feature.lower()} {r.State}" for r in weights.head(2).itertuples())
eu = result.expected_utilities
st.markdown(f"""
<ol class="steps">
<li>The {len(result.features)} model variables were prepared. Numeric features were binned into LOW / MEDIUM / HIGH
using cut points learned from the 1,594 training segments{"; missing values took the training median" if result.filled else ""}.</li>
<li>The TAN Bayesian network put this segment's anomaly probability at <b>{pct(result.current_tan_risk)}</b>.
The strongest evidence towards anomalous was {top}.</li>
<li>HMM filtering combined that evidence with {"the channel's earlier segments" if result.has_history else "the model's starting state probabilities"},
giving a current anomaly risk of <b>{pct(result.current_hmm_risk)}</b>.</li>
<li>The HMM's learned state-change probabilities projected that one segment ahead, giving a
next-segment risk of <b>{pct(result.next_anomaly_risk)}</b>.</li>
<li>The Decision Network compared expected utilities at the current risk
(Continue {eu["Continue"]:.2f}, Monitor {eu["Monitor"]:.2f}, Safe Mode {eu["Safe Mode"]:.2f})
and recommends <b>{result.recommended_action}</b>.</li>
</ol>
""", unsafe_allow_html=True)
note(TARGET)
