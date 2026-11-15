"""Backend checks against the outputs recorded in final_satellite_pgm_project.ipynb."""

import numpy as np
import pandas as pd
import pytest

from backend import TelemetryModel, decision_network, hmm_model, predict_telemetry
from backend.preprocessing import Discretizer

# values printed by the notebook (rounded as displayed there)
NB_RESULTS = {
    "TAN Bayesian Network": (0.8624, 0.8319, 0.8468, 0.9267, 0.0481),
    "HMM filtering": (0.8868, 0.8319, 0.8584, 0.9259, 0.0500),
    "HMM smoothing": (0.8796, 0.8407, 0.8597, 0.9197, 0.0515),
    "Decision Tree": (0.8110, 0.9115, 0.8583, 0.9279, 0.0556),
}
NB_CONFUSION = {
    "TAN Bayesian Network": [[401, 15], [19, 94]],
    "HMM filtering": [[404, 12], [19, 94]],
    "HMM smoothing": [[403, 13], [18, 95]],
    "Decision Tree": [[392, 24], [10, 103]],
}


@pytest.fixture(scope="module")
def model():
    return TelemetryModel.build()


def test_split_sizes(model):
    assert model.data.train_mask.sum() == 1594
    assert model.data.test_mask.sum() == 529


def test_cut_points_match_notebook(model):
    cuts = model.disc.cut_table()
    assert cuts.loc["kurtosis"].tolist() == ["-1.41", "-1.12"]
    assert cuts.loc["skew"].tolist() == ["-0.06", "0.204"]
    assert cuts.loc["diff2_peaks"].tolist() == ["6", "24"]
    assert cuts.loc["len"].tolist() == ["46", "178"]


def test_cut_points_ignore_test_rows(model):
    df = model.data.df.copy()
    df.loc[model.data.test_mask, model.disc.feats] = 1e9      # corrupt test features
    other = Discretizer.fit(df, model.data.train_mask)
    for f in model.disc.feats:
        assert np.allclose(other.edges[f], model.disc.edges[f])


def test_tan_inference_matches_notebook(model):
    post = model.bn.posterior
    assert round(post({})[1], 3) == 0.202
    assert round(post({"kurtosis": 2, "n_peaks": 2})[1], 3) == 0.702
    assert round(post({"kurtosis": 0, "n_peaks": 0})[1], 3) == 0.019
    assert round(post({"kurtosis": 2, "n_peaks": 2, "smooth10_n_peaks": 2})[1], 3) == 0.985


def test_variable_elimination_equals_bayes_rule(model):
    for i in np.where(model.data.test_mask)[0][:20]:
        ev = {v: int(model.D.loc[i, v]) for v in model.vars}
        assert np.isclose(model.bn.posterior(ev)[1], model.p_bn[i])


def test_markov_network_matches_notebook(model):
    assert model.blanket == ["n_peaks"]
    t = model.independencies.round(4)
    assert t["Independence statement"].iloc[0] == "anomaly ⊥ smooth10_n_peaks | n_peaks"
    assert t["I(X;Y)"].iloc[0] == 0.1354 and t["I(X;Y | Z)"].iloc[0] == 0.0337


def test_transition_matrix_matches_notebook(model):
    assert np.allclose(model.A.round(3), [[0.844, 0.156], [0.628, 0.372]])


def test_metrics_match_notebook(model):
    res = model.results().round(4)
    for name, values in NB_RESULTS.items():
        assert tuple(res.loc[name]) == values, name


def test_confusion_matrices_match_notebook(model):
    from backend.evaluation import confusion
    y_te = model.data.y[model.data.test_mask]
    for name, p in model.test_probabilities().items():
        assert confusion(y_te, p).tolist() == NB_CONFUSION[name], name


def test_decision_distribution_matches_notebook(model):
    counts = model.decision_distribution()["Test segments"].tolist()
    assert counts == [397, 27, 105]


def test_decision_boundaries():
    regions = decision_network.action_regions()
    assert [r[0] for r in regions] == ["Continue", "Monitor", "Safe Mode"]
    assert regions[0][2] == pytest.approx(1 / 7) and regions[1][2] == pytest.approx(5 / 9)


def test_highest_risk_example(model):
    te = np.where(model.data.test_mask)[0]
    i = te[np.argmax(model.p_filt[te])]
    assert i == 1292 and model.data.df.at[i, "channel"] == "CADC0874"


def test_filtering_is_causal(model):
    # changing future observations must not change the current filtered belief
    ch = "CADC0872"
    ix = model.channel_rows(ch)
    t = 40
    before = model.predict_segment(ch, t).current_hmm_risk
    loge = model.loge.copy()
    model.loge[ix[t + 1:]] = model.loge[ix[t + 1:]][:, ::-1]
    try:
        assert model.predict_segment(ch, t).current_hmm_risk == pytest.approx(before)
    finally:
        model.loge = loge
    assert before == pytest.approx(model.p_filt[ix[t]])


def test_next_state_formula(model):
    r = model.predict_segment("CADC0873", 100)
    A = model.A
    expected = (1 - r.current_hmm_risk) * A[0, 1] + r.current_hmm_risk * A[1, 1]
    assert r.next_anomaly_risk == pytest.approx(expected)
    assert A[0, 1] <= r.next_anomaly_risk <= A[1, 1]


def test_dataset_mode_reproduces_notebook_arrays(model):
    for ch, ix in model.seqs.items():
        for pos in (0, len(ix) // 2, len(ix) - 1):
            r = model.predict_segment(ch, pos)
            assert r.current_tan_risk == pytest.approx(model.p_bn[ix[pos]])
            assert r.current_hmm_risk == pytest.approx(model.p_filt[ix[pos]])


def test_manual_mode_equals_dataset_row(model):
    row = model.data.df.iloc[model.channel_rows("CADC0888")[0]]
    record = {v: row[v] for v in model.vars}
    r = predict_telemetry(model, record)
    assert r["current_tan_risk"] == pytest.approx(model.p_bn[model.channel_rows("CADC0888")[0]])
    # first segment of a channel: filtering from pi gives the same belief as the notebook
    assert r["current_hmm_risk"] == pytest.approx(model.p_filt[model.channel_rows("CADC0888")[0]])
    assert set(r) == {"current_tan_risk", "current_hmm_risk", "next_anomaly_risk", "current_state",
                      "predicted_next_state", "expected_utilities", "recommended_action"}


def test_manual_continue_history(model):
    ch = "CADC0872"
    ix = model.channel_rows(ch)
    row = model.data.df.iloc[ix[-1]]
    r = model.predict_manual({v: row[v] for v in model.vars}, continue_history=True)
    expected = hmm_model.filter_step(model.alpha[ix[-1]], model.loge[ix[-1]], model.A)[1]
    assert r.current_hmm_risk == pytest.approx(expected)
    assert len(r.history) == len(ix)


def test_missing_values_use_training_median(model):
    record = {"channel": "CADC0872", "sampling": 1}
    r = model.predict_manual(record)
    assert set(r.filled) == set(model.disc.feats)
    assert 0 <= r.current_tan_risk <= 1


@pytest.mark.parametrize("bad, message", [
    ({"channel": "XYZ", "sampling": 1}, "valid channel"),
    ({"channel": "CADC0872", "sampling": 3}, "valid sampling"),
    ({"channel": "CADC0872", "sampling": 1, "var": "abc"}, "valid number"),
    ({"channel": "CADC0872", "sampling": 1, "var": -1.0}, "cannot be negative"),
    ({"channel": "CADC0872", "sampling": 1, "kurtosis": float("inf")}, "finite"),
])
def test_invalid_inputs_raise_clear_errors(model, bad, message):
    with pytest.raises(ValueError, match=message):
        model.predict_manual(bad)


def test_out_of_range_warns(model):
    r = model.predict_manual({"channel": "CADC0872", "sampling": 1, "kurtosis": 500.0})
    assert any("outside the training range" in w for w in r.warnings)


def test_hmm_smoothing_never_feeds_prediction(model):
    # zeroing smoothed probabilities must not change any operational output
    saved = model.p_smooth.copy()
    model.p_smooth[:] = 0
    try:
        r = model.predict_segment("CADC0874", 50)
        assert r.current_hmm_risk == pytest.approx(model.p_filt[model.channel_rows("CADC0874")[50]])
    finally:
        model.p_smooth[:] = saved


def test_feature_table_columns(model):
    r = model.predict_segment("CADC0872", 3)
    assert isinstance(r.features, pd.DataFrame)
    assert list(r.features["Variable"]) == model.vars
