"""Page 3: every model in the project and its evaluation on the official test split."""

import pandas as pd
import streamlit as st

from backend.evaluation import confusion
from ui import charts
from ui.components import kpi, note, page_header
from ui.state import load_model_or_stop

model = load_model_or_stop()
result = st.session_state.get("result")
data = model.data
y_test = data.y[data.test_mask]
results = model.results()
probs = model.test_probabilities()

page_header("Model Analysis")
st.markdown(f'<div class="kicker">Learned from the official training split '
            f'(<b>{data.train_mask.sum():,}</b> segments) and evaluated on the official test split '
            f'(<b>{data.test_mask.sum():,}</b> segments, {y_test.mean():.1%} anomalous) at a fixed 0.5 threshold.'
            f'</div>', unsafe_allow_html=True)

tab_tan, tab_mn, tab_hmm, tab_dt, tab_eval = st.tabs(
    ["TAN Bayesian network", "Markov network", "Hidden Markov model", "Decision Tree", "Evaluation"])

# ------------------------------------------------------------------ TAN
with tab_tan:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.graphviz_chart(charts.tan_dot(model), width="stretch")
    with right:
        st.markdown("#### Structure")
        note("Anomaly is a parent of every variable (grey). The blue arrows are the feature tree: the maximum "
             "spanning tree of conditional mutual information between feature pairs, given the anomaly label, on the "
             f"training split. Its root is <b>{model.tan_root}</b>, the variable most informative about anomalies.")
        parents = pd.DataFrame([{"Variable": v, "Parents": ", ".join(p) or "none"}
                                for v, p in model.tan_parents.items()])
        st.dataframe(parents, hide_index=True, width="stretch", height=36 * (len(parents) + 1) + 4)

    st.markdown("#### Exact inference by Variable Elimination")
    post = model.bn.posterior
    cases = [("None (prior)", {}), ("kurtosis = HIGH, n_peaks = HIGH", {"kurtosis": 2, "n_peaks": 2}),
             ("kurtosis = LOW, n_peaks = LOW", {"kurtosis": 0, "n_peaks": 0}),
             ("kurtosis, n_peaks, smooth10_n_peaks = HIGH",
              {"kurtosis": 2, "n_peaks": 2, "smooth10_n_peaks": 2})]
    st.dataframe(pd.DataFrame({"Evidence": [c[0] for c in cases],
                               "Anomaly probability": [post(c[1])[1] for c in cases]}),
                 hide_index=True, width="stretch",
                 column_config={"Anomaly probability": st.column_config.NumberColumn(format="%.3f")})

    st.markdown("#### Conditional probability tables")
    var = st.selectbox("Variable", model.vars, key="cpt_var")
    f = model.bn.cpt[var]
    states = ["LOW", "MEDIUM", "HIGH"] if var in model.disc.feats else [str(s) for s in model.disc.levels[var]]
    pa = f.vars[:-1]
    def level_names(v):
        if v == "anomaly":
            return ["Nominal", "Anomalous"]
        return ["LOW", "MEDIUM", "HIGH"] if v in model.disc.feats else [str(s) for s in model.disc.levels[v]]
    idx = pd.MultiIndex.from_product([level_names(p) for p in pa], names=pa)
    st.dataframe(pd.DataFrame(f.table.reshape(-1, f.table.shape[-1]), index=idx, columns=states),
                 width="stretch", column_config={s: st.column_config.NumberColumn(format="%.3f")
                                                          for s in states})
    st.caption(f"Probability of each {var} state given {' and '.join(pa)}. Laplace smoothed; each row sums to 1.")

# ------------------------------------------------------------------ Markov network
with tab_mn:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.graphviz_chart(charts.markov_dot(model), width="stretch")
    with right:
        st.markdown("#### Undirected dependency structure")
        note("A tree over Anomaly and the telemetry variables: the maximum spanning tree of pairwise mutual "
             "information on the training split. It represents dependencies "
             "and conditional independencies. It is not used to compute the TAN anomaly probability.")
        kpi("Markov blanket of Anomaly", ", ".join(model.blanket), "its neighbours in the undirected graph",
            small=True)
    st.markdown("#### Conditional independence by graph separation")
    note("Two variables are independent given a third when removing the third from the graph leaves no path "
         "between them. In a tree, any node on the path between two variables separates them. As a check, "
         "their mutual information should drop once the middle variable is known.")
    ind = model.independencies.rename(columns={"I(X;Y)": "Mutual information",
                                               "I(X;Y | Z)": "Given the middle variable"})
    ind["Independence statement"] = [s.replace(" ⊥ ", " independent of ").replace(" | ", " given ")
                                     for s in ind["Independence statement"]]
    st.dataframe(ind, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f")
                                for c in ["Mutual information", "Given the middle variable"]})

# ------------------------------------------------------------------ HMM
with tab_hmm:
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("#### Transition matrix A")
        st.plotly_chart(charts.transition_heatmap(model.A), width="stretch", config=charts.PLOT_CONFIG)
        st.markdown(f"Each row is the current state and each column the next state. A sequence starts nominal "
                    f"with probability <b>{model.pi[0]:.3f}</b> and anomalous with <b>{model.pi[1]:.3f}</b>.",
                    unsafe_allow_html=True)
        note("Counted from consecutive training segments of the same channel, in time order, with Laplace "
             "smoothing. How likely each state makes the observed features comes from the TAN network.")
    with right:
        st.markdown("#### Filtering and smoothing")
        st.markdown("""
| | Uses | Role |
|---|---|---|
| **HMM filtering** | past and current segments | current / online state estimation, feeds the prediction and the Decision Network |
| **HMM smoothing** | the whole sequence, including future segments | retrospective / offline analysis only |
""", unsafe_allow_html=True)
        if result is not None:
            b = result.current_hmm_risk
            c1, c2 = st.columns(2)
            with c1:
                kpi("Filtered nominal probability", f"{1 - b:.3f}", "selected segment", small=True)
            with c2:
                kpi("Filtered anomalous probability", f"{b:.3f}", "selected segment", small=True)

    st.markdown("#### Retrospective view of a channel")
    channels = model.channels()
    default_ch = result.channel if result is not None and result.channel in channels else channels[0]
    c1, c2 = st.columns([1, 3])
    with c1:
        ch = st.selectbox("Channel", channels, index=channels.index(default_ch), key="hmm_channel")
    n = len(model.channel_rows(ch))
    with c2:
        span = st.slider("Segment range", 0, n, (0, min(n, 80)), key=f"hmm_span_{ch}")
    if span[1] - span[0] < 2:
        st.info("Pick a range of at least two segments.")
    else:
        st.plotly_chart(charts.filter_vs_smooth(model, ch, *span), width="stretch",
                        config=charts.PLOT_CONFIG)
    note("Smoothing (purple) can react before an anomaly starts because it sees future segments. That is "
         "useful for labelling history after the fact, and it is the reason smoothing is never used for the "
         "dashboard's prediction.")

# ------------------------------------------------------------------ Decision Tree
with tab_dt:
    note("The Decision Tree is a supervised ML benchmark trained on the same discretised, one-hot encoded "
         "variables (entropy criterion, max depth 5, at least 10 samples per leaf, balanced class weights). "
         "It is a comparison model, not the project's primary probabilistic graphical model.")
    c = st.columns(5, gap="small")
    dt = results.loc["Decision Tree"]
    for col, name in zip(c, ["Precision", "Recall", "F1", "PR-AUC", "Brier Score"]):
        with col:
            kpi(name, f"{dt[name]:.4f}", "test split", small=True)
    st.write("")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("#### Top feature importances")
        st.plotly_chart(charts.importance_bars(model.dt.importance()), width="stretch",
                        config=charts.PLOT_CONFIG)
    with c2:
        st.markdown("#### Tree")
        st.markdown(f"Depth <span class='mono'>{model.dt.model.get_depth()}</span>, "
                    f"<span class='mono'>{model.dt.model.get_n_leaves()}</span> leaves, "
                    f"<span class='mono'>{len(model.dt.columns)}</span> one-hot inputs.", unsafe_allow_html=True)
        cm = confusion(y_test, probs["Decision Tree"])
        st.plotly_chart(charts.confusion_heatmap(cm, "Decision Tree, test split"), width="stretch",
                        config=charts.PLOT_CONFIG)

# ------------------------------------------------------------------ Evaluation
with tab_eval:
    st.markdown("#### Final model comparison, official test split")
    st.dataframe(results, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in results.columns})
    best_brier = results["Brier Score"].idxmin()
    best_f1 = results["F1"].idxmax()
    note(f"The Brier score is the average squared gap between predicted probability and true label, so lower is better: it scores the probabilities themselves. "
         f"<b>{best_brier}</b> has the lowest Brier score ({results.loc[best_brier, 'Brier Score']:.4f}), "
         f"while <b>{best_f1}</b> has the highest F1 ({results.loc[best_f1, 'F1']:.4f}). The highest F1 at one "
         "threshold does not make a model the best probabilistic model. The Decision Network is not in this "
         "table because it selects actions rather than classifying segments.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### F1")
        st.plotly_chart(charts.metric_bars(results, "F1"), width="stretch", config=charts.PLOT_CONFIG)
    with c2:
        st.markdown("#### Brier score (lower is better)")
        st.plotly_chart(charts.metric_bars(results, "Brier Score"), width="stretch",
                        config=charts.PLOT_CONFIG)

    st.markdown("#### Confusion matrices")
    cols = st.columns(2, gap="large")
    for i, (name, p) in enumerate(probs.items()):
        with cols[i % 2]:
            st.plotly_chart(charts.confusion_heatmap(confusion(y_test, p), name), width="stretch",
                            config=charts.PLOT_CONFIG)

    st.markdown("#### Precision–recall curves")
    st.plotly_chart(charts.pr_curves(y_test, probs, results), width="stretch", config=charts.PLOT_CONFIG)
    note("Similar F1 values do not mean the models behave the same. The confusion matrices show different "
         "false-positive / false-negative trade-offs; the curves show behaviour across every threshold. "
         "The Decision Tree's training-split probabilities are in-sample, so only test rows are scored here.")
