"""Page 2: the segment's features, their discrete states and the TAN evidence."""

import pandas as pd
import streamlit as st

from ui import charts, theme
from ui.components import kpi, need_result, note, page_header, pct, show_input_messages
from ui.state import load_model_or_stop, model_inputs_frame

model = load_model_or_stop()
result = need_result()

page_header("Telemetry Analysis", result)
show_input_messages(result)

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    kpi("TAN  P(Anomaly | evidence)", pct(result.current_tan_risk),
        "this segment's features only, no temporal context",
        colour=theme.risk_colour(result.current_tan_risk), primary=True, bar=result.current_tan_risk)
with c2:
    kpi("Prior P(Anomaly)", pct(model.bn.posterior({})[1]),
        "before any evidence: the training anomaly rate (Laplace smoothed)")
with c3:
    kpi("Decision Tree benchmark", pct(result.dt_risk),
        "same discretised inputs; ML comparison, not the primary model",
        colour=theme.risk_colour(result.dt_risk))

note("The TAN probability is Bayes' rule over the learned network: "
     "P(A | e) ∝ P(A) ∏<sub>i</sub> P(x<sub>i</sub> | A, parent<sub>i</sub>). "
     "It says how strongly this combination of discrete feature states resembled anomalous training segments.")

st.subheader("Model inputs and discrete states")
table = result.features.copy()
table["Value"] = [f"{v:.6g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
                  for v in table["Value"]]
st.dataframe(
    table[["Feature", "Variable", "Value", "State", "Evidence weight"]],
    hide_index=True, width="stretch", height=36 * (len(table) + 1) + 4,
    column_config={"Evidence weight": st.column_config.NumberColumn(
        format="%+.3f", help="log P(x | parents, Anomalous) − log P(x | parents, Nominal). "
                             "Positive pushes towards anomalous.")})

st.subheader("Feature evidence")
st.plotly_chart(charts.evidence_bars(result.features), width="stretch", config=charts.PLOT_CONFIG)
note("Each bar is a log-likelihood ratio from the learned conditional probability tables. Red bars made the "
     "observation more likely under the anomalous class, blue under the nominal class. These are evidence "
     "weights inside the probabilistic model. They do not say that a feature causes an anomaly.")

st.subheader("Discretisation cut points")
note("Learned from the training split only. Continuous features use training terciles; peak counts use "
     "fixed bins (1 = LOW, 2 = MEDIUM, 3 or more = HIGH). Values below the first cut are LOW, values at or "
     "above the second cut are HIGH.")
cuts = model_inputs_frame(model)
st.dataframe(cuts, hide_index=True, width="stretch", height=36 * (len(cuts) + 1) + 4,
             column_config={c: st.column_config.NumberColumn(format="%.4g")
                            for c in ["Training min", "LOW | MEDIUM cut", "MEDIUM | HIGH cut",
                                      "Training max", "Median fill"]})

if result.mode == "dataset":
    with st.expander("All recorded columns for this segment"):
        row = model.data.df.iloc[model.channel_rows(result.channel)[result.position]]
        used = set(model.vars)
        other = pd.DataFrame({"Column": row.index, "Value": [str(v) for v in row.values],
                              "Used by the TAN model": [c in used for c in row.index]})
        st.dataframe(other, hide_index=True, width="stretch")
        st.caption("The notebook's TAN uses 9 numeric features plus sampling and channel. The other "
                   "extracted features are kept in the dataset but are not model inputs.")
