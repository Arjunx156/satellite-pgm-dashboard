"""Plotly figures and Graphviz diagrams. Every number plotted comes from the backend."""

import numpy as np
import plotly.graph_objects as go

from backend import config, decision_network
from backend.evaluation import pr_curve

from .theme import ACCENT, ANOMALY, BORDER, MONO, MUTED, SURFACE, TAN_LINE, TEXT

MODEL_COLOURS = {"TAN Bayesian Network": TAN_LINE, "HMM filtering": ACCENT,
                 "HMM smoothing": "#C9A0F0", "Decision Tree": "#E8C26A"}
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def _layout(fig, height=360, **kw):
    kw.setdefault("margin", dict(l=56, r=24, t=36, b=48))
    kw.setdefault("legend", dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11, color=MUTED)))
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=12), hoverlabel=dict(bgcolor=SURFACE, font_family=MONO),
        **kw)
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED))
    return fig


def _band_lines(fig):
    for action, lo, _ in decision_network.action_regions()[1:]:
        fig.add_hline(y=float(lo), line=dict(color=BORDER, width=1, dash="dot"),
                      annotation_text=f"{action} above {float(lo):.3f}",
                      annotation_position="top left", annotation_font=dict(size=10, color=MUTED))


def risk_timeline(result, window=40, show_labels=True):
    """Past -> current -> predicted, from real recorded history only."""
    h = result.history
    manual_tail = result.mode == "manual"
    if manual_tail:
        h = h.iloc[-(window - 1):]
        steps_past = np.arange(-len(h), 0)
    else:
        h = h.iloc[-window:]
        steps_past = np.arange(-(len(h) - 1), 1)
    fig = go.Figure()
    fig.add_vrect(x0=0.5, x1=3.2, fillcolor=ACCENT, opacity=0.07, line_width=0)
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    _band_lines(fig)

    tan = list(h["tan"])
    filt = list(h["filtered"])
    x = list(steps_past)
    if manual_tail:                                    # the scenario is the current step
        x, tan, filt = x + [0], tan + [result.current_tan_risk], filt + [result.current_hmm_risk]

    fig.add_trace(go.Scatter(x=x, y=tan, name="TAN  P(A<sub>t</sub> | e<sub>t</sub>)", mode="lines+markers",
                             line=dict(color=TAN_LINE, width=1.2, dash="dash"), marker=dict(size=4),
                             hovertemplate="step %{x}<br>TAN %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=filt, name="HMM filtered  P(A<sub>t</sub> | O<sub>1:t</sub>)", mode="lines+markers",
                             line=dict(color=ACCENT, width=2.2), marker=dict(size=5),
                             hovertemplate="step %{x}<br>filtered %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[result.current_hmm_risk, result.next_anomaly_risk],
                             mode="lines", line=dict(color=ACCENT, width=2, dash="dot"),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[1], y=[result.next_anomaly_risk], name="Predicted  P(A<sub>t+1</sub> | O<sub>1:t</sub>)",
                             mode="markers", marker=dict(size=14, color="rgba(0,0,0,0)",
                                                         line=dict(color=ACCENT, width=2.5)),
                             hovertemplate="next segment<br>predicted %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0], y=[result.current_hmm_risk], mode="markers", showlegend=False,
                             marker=dict(size=12, color=ACCENT, line=dict(color=TEXT, width=1.5)),
                             hovertemplate="current<br>filtered %{y:.3f}<extra></extra>"))
    if show_labels and not manual_tail:
        lab = h["label"].to_numpy()
        test = (h["split"] == "test").to_numpy()
        for mask, name, symbol in ((~test, "Recorded label (train)", "line-ew-open"),
                                   (test, "Recorded label (test)", "line-ew")):
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=np.asarray(x)[mask], y=lab[mask] * 1.0, mode="markers", name=name,
                    marker=dict(symbol=symbol, size=9, color=ANOMALY if name.endswith("(test)") else MUTED,
                                line=dict(width=2, color=ANOMALY if name.endswith("(test)") else MUTED)),
                    hovertemplate="label %{y:.0f}<extra></extra>"))
    lo = min(x)
    for xpos, text, anchor in ((lo / 2, "PAST", "center"), (-0.2, "CURRENT", "right"),
                               (0.55, "PREDICTED", "left")):
        fig.add_annotation(x=xpos, y=1.13, text=text, showarrow=False, yref="y", xanchor=anchor,
                           font=dict(size=10, color=ACCENT if text == "PREDICTED" else MUTED, family=MONO))
    fig.update_yaxes(range=[-0.04, 1.2], title="P(Anomalous)", tickvals=[0, 0.25, 0.5, 0.75, 1])
    fig.update_xaxes(title="segment step relative to current (0 = current, +1 = next)",
                     range=[lo - 0.6, 3.2])
    return _layout(fig, height=420)


def current_vs_next(result):
    fig = go.Figure()
    labels = ["TAN  P(A<sub>t</sub> | e<sub>t</sub>)", "HMM filtered  P(A<sub>t</sub> | O<sub>1:t</sub>)", "Predicted  P(A<sub>t+1</sub> | O<sub>1:t</sub>)"]
    vals = [result.current_tan_risk, result.current_hmm_risk, result.next_anomaly_risk]
    colours = [TAN_LINE, ACCENT, "rgba(92,200,230,0.35)"]
    fig.add_trace(go.Bar(x=labels, y=vals, marker=dict(color=colours, line=dict(color=ACCENT, width=[0, 0, 2])),
                         text=[f"{v:.3f}" for v in vals], textposition="outside",
                         textfont=dict(family=MONO), hoverinfo="skip"))
    _band_lines(fig)
    fig.update_yaxes(range=[0, 1.12], title="P(Anomalous)")
    return _layout(fig, height=340, showlegend=False)


def evidence_bars(features):
    f = features.iloc[::-1]
    w = f["Evidence weight"].to_numpy()
    fig = go.Figure(go.Bar(
        x=w, y=[f"{a}  ({s})" for a, s in zip(f["Feature"], f["State"])], orientation="h",
        marker_color=[ANOMALY if v > 0 else ACCENT for v in w],
        hovertemplate="%{y}<br>log-likelihood ratio %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title="log P(x | parents, Anomalous) − log P(x | parents, Nominal)")
    return _layout(fig, height=40 + 32 * len(f), showlegend=False)


def transition_heatmap(A):
    names = ["Nominal", "Anomalous"]
    fig = go.Figure(go.Heatmap(z=A, x=names, y=names, colorscale=[[0, SURFACE], [1, "#2E6F86"]],
                               zmin=0, zmax=1, showscale=False,
                               text=[[f"{v:.3f}" for v in row] for row in A], texttemplate="%{text}",
                               textfont=dict(family=MONO, size=16),
                               hovertemplate="from %{y} to %{x}: %{z:.4f}<extra></extra>"))
    fig.update_xaxes(title="Next state S<sub>t+1</sub>", side="bottom")
    fig.update_yaxes(title="Current state S<sub>t</sub>", autorange="reversed")
    return _layout(fig, height=300, margin=dict(l=90, r=20, t=20, b=60))


def filter_vs_smooth(model, channel, start, stop):
    ix = model.channel_rows(channel)[start:stop]
    x = np.arange(start, start + len(ix))
    y = model.data.y[ix]
    test = model.data.test_mask[ix]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=model.p_bn[ix], name="TAN", line=dict(color=TAN_LINE, width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=model.p_filt[ix], name="HMM filtering (online)", line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=x, y=model.p_smooth[ix], name="HMM smoothing (retrospective)",
                             line=dict(color=MODEL_COLOURS["HMM smoothing"], width=2)))
    fig.add_trace(go.Scatter(x=x[~test], y=y[~test], mode="markers", name="label (train)",
                             marker=dict(color=MUTED, size=6)))
    fig.add_trace(go.Scatter(x=x[test], y=y[test], mode="markers", name="label (test)",
                             marker=dict(color="rgba(0,0,0,0)", size=9, line=dict(color=TEXT, width=1.5))))
    fig.update_yaxes(range=[-0.05, 1.05], title="P(Anomalous)")
    fig.update_xaxes(title=f"segment position in channel {channel}")
    return _layout(fig, height=380)


def confusion_heatmap(cm, title):
    labels = [["TN", "FP"], ["FN", "TP"]]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred. nominal", "Pred. anomalous"], y=["Actual nominal", "Actual anomalous"],
        colorscale=[[0, SURFACE], [1, "#2E6F86"]], showscale=False,
        text=[[f"{labels[i][j]}  {cm[i][j]}" for j in range(2)] for i in range(2)],
        texttemplate="%{text}", textfont=dict(family=MONO, size=15), hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=250, title=dict(text=title, font=dict(size=13)),
                   margin=dict(l=110, r=10, t=40, b=40))


def pr_curves(y_test, probabilities, results):
    fig = go.Figure()
    base = float(np.mean(y_test))
    fig.add_hline(y=base, line=dict(color=BORDER, dash="dot"),
                  annotation_text=f"no-skill baseline {base:.3f}", annotation_position="bottom right",
                  annotation_font=dict(size=10, color=MUTED))
    for name, p in probabilities.items():
        prec, rec = pr_curve(y_test, p)
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", line=dict(color=MODEL_COLOURS[name], width=2),
                                 name=f"{name} (PR-AUC {results.loc[name, 'PR-AUC']:.3f})",
                                 line_shape="hv"))
    fig.update_xaxes(range=[0, 1.01], title="Recall")
    fig.update_yaxes(range=[0, 1.05], title="Precision")
    return _layout(fig, height=440, legend=dict(orientation="v", x=0.02, y=0.05, yanchor="bottom",
                                                bgcolor="rgba(11,17,28,0.8)", font=dict(size=11)))


def metric_bars(results, column):
    fig = go.Figure(go.Bar(x=results.index, y=results[column],
                           marker_color=[MODEL_COLOURS[m] for m in results.index],
                           text=[f"{v:.4f}" for v in results[column]], textposition="outside",
                           textfont=dict(family=MONO)))
    top = results[column].max() * 1.2
    fig.update_yaxes(range=[0, top], title=column)
    return _layout(fig, height=300, showlegend=False)


def importance_bars(series):
    s = series.iloc[::-1]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color="#E8C26A",
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Gini-style importance (entropy criterion)")
    return _layout(fig, height=420, showlegend=False)


def eu_lines(p_cur, p_next):
    p = np.linspace(0, 1, 201)
    fig = go.Figure()
    colours = {"Continue": ACCENT, "Monitor": TEXT, "Safe Mode": ANOMALY}
    for action, (lo, hi) in ((a, config.UTILITY[a]) for a in config.ACTIONS):
        fig.add_trace(go.Scatter(x=p, y=(1 - p) * lo + p * hi, name=action,
                                 line=dict(color=colours[action], width=2)))
    for action, lo, hi in decision_network.action_regions():
        if float(lo) > 0:
            fig.add_vline(x=float(lo), line=dict(color=BORDER, dash="dot"),
                          annotation_text=f"{float(lo):.3f}", annotation_font=dict(size=10, color=MUTED))
    for x, name, sym in ((p_cur, "current filtered risk", "circle"), (p_next, "next-segment risk", "circle-open")):
        _, eu = decision_network.recommend(x)
        best = max(eu.values())
        fig.add_trace(go.Scatter(x=[x], y=[best], mode="markers", name=name,
                                 marker=dict(symbol=sym, size=13, color=ACCENT, line=dict(width=2.5, color=ACCENT)),
                                 hovertemplate=f"{name}<br>p = %{{x:.3f}}<br>max EU = %{{y:.2f}}<extra></extra>"))
    fig.update_xaxes(title="P(Anomalous | evidence)", range=[-0.02, 1.02])
    fig.update_yaxes(title="Expected utility")
    return _layout(fig, height=380)


def eu_bars(eu, best):
    acts = list(eu)
    fig = go.Figure(go.Bar(x=[eu[a] for a in acts], y=acts, orientation="h",
                           marker_color=[ACCENT if a == best else BORDER for a in acts],
                           text=[f"{eu[a]:.2f}" for a in acts], textposition="outside", cliponaxis=False,
                           textfont=dict(family=MONO)))
    lo, hi = min(0, min(eu.values())), max(0, max(eu.values()))
    pad = 0.22 * (hi - lo) + 1
    fig.update_xaxes(range=[lo - pad, hi + pad], title="EU(a | E)")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=200, showlegend=False, margin=dict(l=90, r=30, t=10, b=44))


# ----------------------------------------------------------------- Graphviz (rendered in the browser)

_GRAPH = (f'bgcolor="transparent"; fontname="Helvetica"; '
          f'node [fontname="Helvetica", fontsize=11, fontcolor="{TEXT}", color="{BORDER}", '
          f'style=filled, fillcolor="{SURFACE}", penwidth=1.4]; edge [color="{MUTED}", arrowsize=0.7];')


def tan_dot(model):
    lines = [f"digraph TAN {{ rankdir=TB; {_GRAPH} nodesep=0.25; ranksep=0.55;",
             f'anomaly [label="Anomaly", shape=ellipse, color="{ANOMALY}", fontcolor="{ANOMALY}", penwidth=2];']
    for v in model.vars:
        lines.append(f'"{v}" [label="{v}", shape=box, style="rounded,filled"];')
    for child, parents in model.tan_parents.items():
        for p in parents:
            if p == "anomaly":
                lines.append(f'anomaly -> "{child}" [color="{BORDER}", penwidth=0.8];')
            else:
                lines.append(f'"{p}" -> "{child}" [color="{ACCENT}", penwidth=2];')
    lines.append("}")
    return "\n".join(lines)


def markov_dot(model):
    lines = [f"graph MN {{ layout=dot; rankdir=TB; {_GRAPH}",
             f'anomaly [label="Anomaly", shape=ellipse, color="{ANOMALY}", fontcolor="{ANOMALY}", penwidth=2];']
    for v in model.vars:
        lines.append(f'"{v}" [shape=ellipse];')
    for a, b in model.mn.edges:
        lines.append(f'"{a}" -- "{b}" [penwidth=1.6];')
    lines.append("}")
    return "\n".join(lines)


def influence_dot():
    return f"""digraph ID {{ rankdir=LR; {_GRAPH} nodesep=0.6; ranksep=0.9;
evidence [label="Telemetry evidence\\nO_1:t", shape=box, style="rounded,filled"];
state [label="Satellite state S_t\\nNominal / Anomalous\\n(chance)", shape=ellipse, color="{ACCENT}"];
action [label="Action\\nContinue / Monitor / Safe Mode\\n(decision)", shape=box];
utility [label="Utility\\nU(a, s)", shape=diamond, color="{ANOMALY}"];
evidence -> state [label=" HMM filtering", fontcolor="{MUTED}", fontsize=9];
evidence -> action; state -> utility; action -> utility;
}}"""
