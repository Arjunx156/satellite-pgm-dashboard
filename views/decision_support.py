"""Page 4: Decision Network expected utilities and the recommended action."""

import streamlit as st

from backend import decision_network
from ui import charts, theme
from ui.components import DISCLAIMER, action_kpi, kpi, need_result, note, page_header, pct
from ui.state import load_model_or_stop

model = load_model_or_stop()
result = need_result()

page_header("Decision Support", result)
st.warning(DISCLAIMER)

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    kpi("Current anomaly probability", pct(result.current_hmm_risk), "HMM filtering on this and earlier segments",
        colour=theme.risk_colour(result.current_hmm_risk), primary=True, bar=result.current_hmm_risk)
with c2:
    kpi("Predicted next-segment probability", pct(result.next_anomaly_risk),
        "current state projected one segment ahead", colour=theme.risk_colour(result.next_anomaly_risk),
        primary=True, bar=result.next_anomaly_risk)
with c3:
    action_kpi(result.recommended_action, f"{result.risk_band} band at the current probability")

st.subheader("Expected utility")
note("Each action's expected utility weighs its utility in each state by how likely that state is. "
     "The recommended action is the one with the highest expected utility.")
left, right = st.columns(2, gap="large")
with left:
    st.markdown("**At the current belief**: used for the recommendation, as in the notebook")
    st.plotly_chart(charts.eu_bars(result.expected_utilities, result.recommended_action),
                    width="stretch", config=charts.PLOT_CONFIG)
with right:
    st.markdown("**At the predicted next-segment belief**: forward-looking view")
    st.plotly_chart(charts.eu_bars(result.next_expected_utilities, result.next_recommended_action),
                    width="stretch", config=charts.PLOT_CONFIG)
    st.markdown(f"Best action for the next segment: **{result.next_recommended_action}**.")

note("The recommended action is the action with the highest expected utility under the state probabilities. "
     "The headline recommendation uses the filtered current belief, the operational input the notebook's "
     "Decision Network was built and evaluated on. The next-segment view applies the same utilities to "
     f"the next-segment probability. Because the transition matrix keeps that probability between "
     f"{pct(model.A[0, 1])} and {pct(model.A[1, 1])}, it always falls in the Monitor region.")

st.subheader("Utility table and action regions")
left, right = st.columns([2, 3], gap="large")
with left:
    st.dataframe(decision_network.utility_table(), width="stretch")
    regions = decision_network.action_regions()
    st.markdown("<br>".join(
        f"<b>{a}</b> when the anomaly probability is between {pct(float(lo))} and {pct(float(hi))}"
        for a, lo, hi in regions), unsafe_allow_html=True)
    note("Low / moderate / high risk bands on this dashboard are these action regions renamed. They come from "
         "the utility table, not from validated spacecraft safety thresholds.")
with right:
    st.plotly_chart(charts.eu_lines(result.current_hmm_risk, result.next_anomaly_risk),
                    width="stretch", config=charts.PLOT_CONFIG)

st.subheader("Influence diagram")
left, right = st.columns([3, 2], gap="large")
with left:
    st.graphviz_chart(charts.influence_dot(), width="stretch")
with right:
    st.markdown("#### Recommended actions on the test split")
    dist = model.decision_distribution()
    st.dataframe(dist, width="stretch",
                 column_config={"Percentage": st.column_config.NumberColumn(format="%.2f%%")})
    st.caption("Decision Network applied to the HMM filtered probability of each of the 529 test segments.")
