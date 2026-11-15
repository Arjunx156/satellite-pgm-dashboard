"""The predictive pipeline the dashboard calls.

Telemetry input -> discretisation -> TAN P(A_t | e_t) -> HMM filtering P(S_t | O_1:t)
-> transition model P(S_t+1 | O_1:t) -> Decision Network -> recommended action.

`TelemetryModel.build()` learns every parameter from the official training split once.
The test split is touched only for evaluation, and only features (never labels) feed the HMM
forward pass, exactly as in the notebook.
"""

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from . import config, decision_network, evaluation, hmm_model
from .data_loader import Dataset, load_dataset
from .decision_tree import DecisionTreeBenchmark
from .markov_model import independence_table, learn_markov_tree, markov_blanket
from .preprocessing import Discretizer, PreparedInput, prepare_record
from .tan_model import BayesNet, learn_tan


@dataclass
class PredictionResult:
    mode: str                          # "dataset" or "manual"
    title: str
    current_tan_risk: float            # P(A_t | e_t), TAN on this segment only
    current_hmm_risk: float            # P(A_t | O_1:t), HMM filtering
    next_anomaly_risk: float           # P(A_t+1 | O_1:t), filtering + transition matrix
    current_state: str
    predicted_next_state: str
    expected_utilities: dict           # on the current filtered belief (notebook's decision input)
    recommended_action: str
    next_expected_utilities: dict      # on the next-segment belief (forward-looking view)
    next_recommended_action: str
    risk_band: str
    next_risk_band: str
    dt_risk: float                     # Decision Tree benchmark on the same segment
    features: pd.DataFrame             # Feature | Value | State | Evidence weight
    filled: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    has_history: bool = False
    history: pd.DataFrame | None = None   # columns: step, segment, tan, filtered, label, split
    actual_label: int | None = None
    next_actual_label: int | None = None
    channel: str | None = None
    position: int | None = None
    segment_id: int | None = None
    split: str | None = None

    def as_dict(self) -> dict:
        keys = ["current_tan_risk", "current_hmm_risk", "next_anomaly_risk", "current_state",
                "predicted_next_state", "expected_utilities", "recommended_action"]
        return {k: getattr(self, k) for k in keys}


def _state(p):
    return "Anomalous" if p >= config.THRESHOLD else "Nominal"


class TelemetryModel:
    """All learned objects, built once and reused for every prediction."""

    def __init__(self, data: Dataset):
        self.data = data
        df, tr, y = data.df, data.train_mask, data.y

        # Section 3: discretisation, training rows only
        self.disc = Discretizer.fit(df, tr)
        self.vars = self.disc.variables
        self.D = self.disc.transform(df)
        self.D["anomaly"] = y
        D_train = self.D[tr]

        # Section 4: TAN structure (CMI spanning tree) and CPTs, training rows only
        self.tan_parents, self.tan_tree, self.tan_root = learn_tan(D_train, self.vars)
        self.bn = BayesNet(self.tan_parents, self.disc.card, config.CPT_ALPHA).fit(D_train)
        self.dag = nx.DiGraph([(p, c) for c, ps in self.tan_parents.items() for p in ps])
        assert nx.is_directed_acyclic_graph(self.dag)
        assert all(np.allclose(f.table.sum(axis=-1), 1) for f in self.bn.cpt.values())

        # Sections 6-7: Markov network, training rows only
        self.mn_nodes = ["anomaly"] + self.vars
        self.mn = learn_markov_tree(D_train, self.mn_nodes)
        self.independencies = independence_table(self.mn, D_train, self.mn_nodes)
        self.blanket = markov_blanket(self.mn)

        # Section 8: HMM. pi, A from training labels; emissions from the TAN network
        self.seqs = {ch: ix for ch, ix in df.groupby("channel").indices.items()}
        self.pi, self.A = hmm_model.fit_transitions(list(self.seqs.values()), y, tr)
        self.loge = self.bn.loglik(self.D)
        self.p_bn = self.bn.predict_proba(self.D)
        self.p_filt, self.p_smooth = np.zeros(len(df)), np.zeros(len(df))
        self.alpha = np.zeros((len(df), 2))
        for ix in self.seqs.values():
            self.p_filt[ix], self.p_smooth[ix] = hmm_model.forward_backward(self.loge[ix], self.A, self.pi)
            self.alpha[ix] = hmm_model.forward(self.loge[ix], self.A, self.pi)

        # Section 10: Decision Tree benchmark, training rows only
        self.dt = DecisionTreeBenchmark(self.vars).fit(D_train, y[tr])
        self.p_dt = self.dt.predict_proba(self.D)

    @classmethod
    def build(cls, **paths) -> "TelemetryModel":
        return cls(load_dataset(**paths))

    # ---------------------------------------------------------------- evaluation (test split only)

    def test_probabilities(self) -> dict:
        te = self.data.test_mask
        return {"TAN Bayesian Network": self.p_bn[te], "HMM filtering": self.p_filt[te],
                "HMM smoothing": self.p_smooth[te], "Decision Tree": self.p_dt[te]}

    def results(self) -> pd.DataFrame:
        return evaluation.results_table(self.data.y[self.data.test_mask], self.test_probabilities())

    def decision_distribution(self) -> pd.DataFrame:
        te = np.where(self.data.test_mask)[0]
        actions = [decision_network.recommend(float(np.clip(self.p_filt[i], 0, 1)))[0] for i in te]
        counts = pd.Series(actions).value_counts().reindex(config.ACTIONS, fill_value=0)
        return pd.DataFrame({"Test segments": counts, "Percentage": 100 * counts / len(te)})

    # ---------------------------------------------------------------- helpers

    def channels(self):
        return list(self.disc.levels["channel"])

    def channel_rows(self, channel):
        return self.seqs[channel]

    def _feature_table(self, raw: dict, codes: dict) -> pd.DataFrame:
        weights = self.bn.contributions(codes)
        rows = []
        for v in self.vars:
            state = config.STATE_NAMES[codes[v]] if v in self.disc.feats else str(raw[v])
            rows.append({"Variable": v, "Feature": config.FEATURE_LABELS[v], "Value": raw[v],
                         "State": state, "Evidence weight": weights[v]})
        return pd.DataFrame(rows)

    def _history_frame(self, ix) -> pd.DataFrame:
        df = self.data.df
        return pd.DataFrame({
            "step": np.arange(len(ix)),
            "segment": df["segment"].to_numpy()[ix],
            "tan": self.p_bn[ix],
            "filtered": self.alpha[ix, 1],
            "label": self.data.y[ix],
            "split": np.where(self.data.train_mask[ix], "train", "test"),
        })

    def _assemble(self, *, mode, title, tan, belief, codes, raw, **extra) -> PredictionResult:
        p_cur = float(np.clip(belief[1], 0, 1))
        p_next = float(np.clip(hmm_model.predict_next(belief, self.A)[1], 0, 1))
        action, eu = decision_network.recommend(p_cur)
        next_action, next_eu = decision_network.recommend(p_next)
        dt = float(self.dt.predict_proba(pd.DataFrame([codes]))[0])
        return PredictionResult(
            mode=mode, title=title, current_tan_risk=float(tan), current_hmm_risk=p_cur,
            next_anomaly_risk=p_next, current_state=_state(p_cur), predicted_next_state=_state(p_next),
            expected_utilities=eu, recommended_action=action,
            next_expected_utilities=next_eu, next_recommended_action=next_action,
            risk_band=decision_network.risk_band(p_cur), next_risk_band=decision_network.risk_band(p_next),
            dt_risk=dt, features=self._feature_table(raw, codes), **extra)

    # ---------------------------------------------------------------- public API

    def predict_segment(self, channel: str, position: int) -> PredictionResult:
        """Historical replay: segment `position` of `channel`, filtered on O_1..O_position only."""
        if channel not in self.seqs:
            raise ValueError(f"Unknown channel {channel!r}.")
        ix = self.seqs[channel]
        if not 0 <= position < len(ix):
            raise ValueError(f"Channel {channel} has segments 0 to {len(ix) - 1}.")
        past = ix[: position + 1]
        # causal by construction: the forward pass is run on the past only
        belief = hmm_model.forward(self.loge[past], self.A, self.pi)[-1]
        row = ix[position]
        df = self.data.df
        raw = {v: (df.at[row, v].item() if hasattr(df.at[row, v], "item") else df.at[row, v]) for v in self.vars}
        codes = {v: int(self.D.at[row, v]) for v in self.vars}
        nxt = int(self.data.y[ix[position + 1]]) if position + 1 < len(ix) else None
        return self._assemble(
            mode="dataset", title=f"Segment {int(df.at[row, 'segment'])}, channel {channel}",
            tan=self.p_bn[row], belief=belief, codes=codes, raw=raw,
            has_history=True, history=self._history_frame(past),
            actual_label=int(self.data.y[row]), next_actual_label=nxt,
            channel=channel, position=position, segment_id=int(df.at[row, "segment"]),
            split="train" if self.data.train_mask[row] else "test")

    def predict_manual(self, record: dict, continue_history: bool = False) -> PredictionResult:
        """Scenario analysis for a hand-entered feature vector.

        continue_history=False: the segment starts a new sequence, so the HMM uses pi.
        continue_history=True: the segment is appended after the last recorded segment of the
        chosen channel, so the HMM starts from that segment's filtered belief.
        """
        prep: PreparedInput = prepare_record(record, self.disc)
        D_row = pd.DataFrame([prep.codes])
        loge = self.bn.loglik(D_row)[0]
        tan = float(self.bn.predict_proba(D_row)[0])
        history = None
        if continue_history:
            ix = self.seqs[prep.raw["channel"]]
            belief = hmm_model.filter_step(self.alpha[ix[-1]], loge, self.A)
            history = self._history_frame(ix)
        else:
            belief = hmm_model.initial_step(loge, self.pi)
        return self._assemble(
            mode="manual", title=f"Scenario on channel {prep.raw['channel']}",
            tan=tan, belief=belief, codes=prep.codes, raw=prep.raw,
            filled=prep.filled, warnings=prep.warnings, has_history=continue_history,
            history=history, channel=prep.raw["channel"])


def predict_telemetry(model: TelemetryModel, record: dict, continue_history: bool = False) -> dict:
    """Convenience wrapper returning the plain result dictionary."""
    return model.predict_manual(record, continue_history).as_dict()
