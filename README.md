# Predictive Satellite Telemetry Monitoring

**Probabilistic Satellite Telemetry Anomaly Risk Assessment and Decision Support Using Probabilistic Graphical Models**

**Live dashboard:** https://satellite-pgm-dashboard.streamlit.app

A Streamlit dashboard built on the project's validated notebook (`final_satellite_pgm_project.ipynb`). It takes one telemetry segment from the ESA OPS-SAT-AD dataset, or a set of hand-entered features, and answers three questions:

1. How likely is it that this segment is anomalous? That is P(Anomaly | Telemetry Evidence), from a TAN Bayesian network.
2. Given everything seen so far on this channel, what is the current state? That is P(S<sub>t</sub> | O<sub>1:t</sub>), from HMM filtering.
3. How likely is it that the **next** segment is anomalous, and which action has the highest expected utility? That is P(S<sub>t+1</sub> | O<sub>1:t</sub>), from the HMM transition model and a Decision Network.

The prediction is the probability that the next telemetry segment is anomalous. It is not a prediction of satellite, component or mission failure. The data is historical telemetry replayed offline, not a live satellite feed.

Engineered by Arjun Krishnan.

---

## System architecture

```
Telemetry input (dataset segment or manual features)
      |
Feature preparation      fill missing values with the training median
      |
Discretisation           LOW / MEDIUM / HIGH, cut points from the training split
      |
TAN Bayesian network     P(A_t | e_t)
      |
HMM filtering            P(S_t | O_1:t)
      |
HMM transition model     P(S_t+1 | O_1:t) = P(S_t | O_1:t) · A
      |
Decision Network         EU(a | E) = sum_s P(s | E) U(a, s)
      |
Recommended action       argmax_a EU(a | E)
```

The frontend never computes a probability itself. Every page reads one `PredictionResult` produced by `backend/prediction.py`:

```python
from backend import TelemetryModel, predict_telemetry

model = TelemetryModel.build()            # learns everything from the training split, once
result = predict_telemetry(model, {"channel": "CADC0872", "sampling": 1, "kurtosis": 0.4, ...})
# {'current_tan_risk': ..., 'current_hmm_risk': ..., 'next_anomaly_risk': ...,
#  'current_state': ..., 'predicted_next_state': ..., 'expected_utilities': {...},
#  'recommended_action': ...}
```

`model.predict_segment(channel, position)` does the same for a recorded segment and also returns its real history for the timeline.

## Models used

| Model | Role | Source in notebook |
|---|---|---|
| TAN Bayesian network | Primary model. P(Anomaly \| evidence) by Bayes' rule and exact Variable Elimination | Sections 4–5 |
| Markov network | Undirected dependency structure and conditional-independence analysis. Not used to compute the anomaly probability | Sections 6–7 |
| Hidden Markov model | Two states (0 = Nominal, 1 = Anomalous). Filtering for online use, smoothing for retrospective analysis, transition matrix for the one-step prediction | Section 8 |
| Decision Tree | Supervised ML benchmark on the same discretised inputs. Not the primary model | Section 10 |
| Decision Network | Expected utility over Continue / Monitor / Safe Mode | Sections 11–12 |

All model code is ported line for line from the notebook. Nothing was swapped for a different ML method, and no pgmpy is needed because the notebook writes every model in NumPy.

## Dataset

ESA OPS-SAT-AD (Zenodo, DOI 10.5281/zenodo.15108715). The two CSV files in `data/` are redistributed unchanged under the dataset's CC BY 4.0 licence; credit belongs to the dataset authors.

- `data/dataset.csv`: one row per telemetry segment, holding the extracted features, the `anomaly` label and the official `train` flag. 2,123 segments across 9 channels.
- `data/segments.csv`: raw readings. Used only to find each segment's start time (to order segments chronologically within a channel) and to cross-check the labels, exactly as in the notebook.

The official split gives 1,594 training segments and 529 test segments. 20.1% of training segments and 21.4% of test segments are anomalous.

The TAN uses 11 variables, read from the notebook rather than guessed: `kurtosis, skew, n_peaks, smooth10_n_peaks, smooth20_n_peaks, diff2_peaks, diff_var, var, len` plus the categorical `sampling` and `channel`. The other columns in `dataset.csv` (mean, std, duration and so on) are shown on the Telemetry Analysis page but are not model inputs.

## Installation

Python 3.12 is recommended. The pinned packages were tested on 3.12 and 3.14, and pandas 3 needs Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Folder structure

```
satellite_pgm_dashboard/
├── app.py                      entry point: page setup, navigation, sidebar, cached model
├── requirements.txt            pinned runtime packages
├── requirements-dev.txt        adds pytest
├── README.md
├── .gitignore
├── .streamlit/config.toml      dark theme, telemetry off
├── assets/icon.svg             browser-tab icon
├── data/
│   ├── dataset.csv
│   └── segments.csv
├── backend/                    all model logic, no Streamlit imports
│   ├── config.py               feature list, utility table, thresholds: one place to change them
│   ├── data_loader.py          load, order chronologically, keep the official split
│   ├── preprocessing.py        training-only cut points, missing values, input validation
│   ├── tan_model.py            Factor, BayesNet, CMI, TAN learning, Variable Elimination
│   ├── markov_model.py         Markov tree, graph separation, independence table
│   ├── hmm_model.py            transitions, forward filtering, forward-backward, one-step prediction
│   ├── decision_tree.py        Decision Tree benchmark
│   ├── decision_network.py     expected utility, action regions, risk bands
│   ├── evaluation.py           precision, recall, F1, PR-AUC, Brier, confusion matrices
│   └── prediction.py           TelemetryModel and the predictive pipeline the UI calls
├── ui/                         presentation only
│   ├── theme.py                colour tokens and CSS
│   ├── components.py           KPI cards, notes, page header, empty states
│   ├── charts.py               Plotly figures and Graphviz diagrams
│   └── state.py                cached model loading and the sidebar input panel
├── views/                      the four pages, routed by st.navigation in app.py
│   ├── predictive_monitoring.py
│   ├── telemetry_analysis.py
│   ├── model_analysis.py
│   └── decision_support.py
└── tests/
    ├── test_backend.py         every number checked against the notebook's recorded output
    └── test_app.py             headless render of every page in both input modes
```

## How to run

The page scripts live in `views/`, not `pages/`. Streamlit treats a folder named `pages/` as a legacy multipage app and can bypass `app.py` when someone opens a page URL directly, so the name is deliberate.

```powershell
python -m streamlit run app.py
```

The dashboard opens at http://localhost:8501. The first load takes about 5–10 seconds while the models are learned from the training split. After that the model is cached for the life of the server, so changing an input only reruns inference (around 25 ms).

To run the tests:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

## How to use the dashboard

The sidebar holds the input. Whatever you choose there drives all four pages.

| Page | What it shows |
|---|---|
| **Predictive Monitoring** | Current anomaly risk, predicted next-segment risk, current and predicted state, recommended action, the past → current → predicted timeline, and a step-by-step "How was this prediction made?" with the real numbers |
| **Telemetry Analysis** | Every model input with its value, its LOW / MEDIUM / HIGH state and its evidence weight. Also the TAN probability, the prior, the Decision Tree benchmark and the training cut points |
| **Model Analysis** | TAN graph, conditional probability tables, Variable Elimination examples, Markov network with independencies, HMM transition matrix, filtering vs smoothing, Decision Tree, the final evaluation table, confusion matrices and PR curves |
| **Decision Support** | Expected utility of each action at the current and the predicted risk, the utility table, exact action regions, the influence diagram, and the action distribution over the test split |

### Dataset segment mode: Historical Telemetry Analysis

Pick a channel and a position in that channel, or step through segments with **Previous** and **Next segment**. The HMM forward pass runs on that channel's segments up to and including the selected one, and nothing after it. The dashboard opens on the notebook's own end-to-end example: the highest-risk test segment (segment 292, channel CADC0874).

Recorded labels appear on the timeline and in a "replay check" note, so you can compare the prediction with what actually happened. The prediction never reads them.

### Manual input mode: Telemetry Scenario Analysis

Type values for the 11 model variables. Scientific notation such as `3.9e-10` is accepted, because the variance features are tiny. **Prefill from** loads the training medians or any recorded segment as a starting point.

- Leave a box empty and it is filled with the training median, as in the notebook.
- Text that isn't a number, negative variances or lengths, and infinities are rejected with a plain message.
- Values outside the training range are accepted but flagged, because the model has never seen them.
- **Sequence context** decides where the HMM starts. *Isolated segment* starts from the learned initial distribution π, so there is no history to plot. *Append after channel history* places the scenario after the chosen channel's last recorded segment, so the timeline shows real history followed by your scenario and the prediction.

Click **Run prediction** to compute.

## Prediction methodology

For the current segment t with discrete evidence e<sub>t</sub>:

**TAN.** P(A<sub>t</sub> | e<sub>t</sub>) ∝ P(A) ∏<sub>i</sub> P(x<sub>i</sub> | A, parent<sub>i</sub>). The structure is the maximum spanning tree of conditional mutual information I(X<sub>i</sub>; X<sub>j</sub> | A), rooted at the variable with the highest I(X; A), which is `n_peaks`. CPTs are counted from the training split with Laplace smoothing α = 1.

**HMM filtering.** The TAN likelihoods P(e<sub>t</sub> | S<sub>t</sub>) are the emissions:

α<sub>t</sub>(s) ∝ P(e<sub>t</sub> | s) Σ<sub>s'</sub> α<sub>t−1</sub>(s') A<sub>s's</sub>, &nbsp; giving P(S<sub>t</sub> | O<sub>1:t</sub>).

**Next-state prediction.**

P(A<sub>t+1</sub> | O<sub>1:t</sub>) = Σ<sub>s</sub> P(A<sub>t+1</sub> | S<sub>t</sub> = s) P(S<sub>t</sub> = s | O<sub>1:t</sub>) = 0.156 · P(N<sub>t</sub> | O<sub>1:t</sub>) + 0.372 · P(A<sub>t</sub> | O<sub>1:t</sub>)

The learned transition matrix (training labels only, consecutive segments of the same channel) is:

| from \ to | Nominal | Anomalous |
|---|---|---|
| Nominal | 0.844 | 0.156 |
| Anomalous | 0.628 | 0.372 |

### Current risk vs next-segment risk

*Current risk* P(A<sub>t</sub> | O<sub>1:t</sub>) answers "is this segment anomalous?". It combines the segment's own evidence with what the channel looked like before.

*Next-segment risk* P(A<sub>t+1</sub> | O<sub>1:t</sub>) answers "before seeing the next segment, how likely is it to be anomalous?". It has no evidence about segment t+1 yet, so it can only use the transition matrix. That leads to an important property of this model: next-segment risk always lies between **15.6%** (current state certainly nominal) and **37.2%** (current state certainly anomalous). Anomalies in this data rarely last more than a couple of segments, so one step ahead the model leans back towards nominal. As a result:

- the predicted next state is always Nominal at the 0.5 threshold, and
- the Decision Network applied to next-segment risk always lands in the Monitor region (1/7 to 5/9).

This is what the learned HMM says, not a bug, and the dashboard says so where it shows the numbers. The headline recommendation therefore uses the current filtered belief, the same input the notebook's Decision Network was built and evaluated on. The next-segment recommendation is shown beside it as a forward-looking view.

## HMM filtering vs smoothing

| | Filtering | Smoothing |
|---|---|---|
| Quantity | P(S<sub>t</sub> \| O<sub>1:t</sub>) | P(S<sub>t</sub> \| O<sub>1:T</sub>) |
| Uses | past and current segments | the whole sequence, including future segments |
| Use in this dashboard | current risk, next-segment prediction, Decision Network | retrospective analysis on the Model Analysis page only |

Smoothing can react before an anomaly starts because it has seen the future. That helps when labelling history after the fact, and it is exactly why smoothing never feeds a prediction here. A test (`test_hmm_smoothing_never_feeds_prediction`) checks this.

## Decision Network

Chance node: satellite state (Nominal / Anomalous). Decision node: Continue / Monitor / Safe Mode. Utility node: U(action, state). Utilities are copied from the notebook:

| Action | Nominal | Anomalous |
|---|---|---|
| Continue | 10 | −20 |
| Monitor | 6 | 4 |
| Safe Mode | 1 | 8 |

EU(a | E) = P(N | E) U(a, N) + P(A | E) U(a, A), and a* = argmax<sub>a</sub> EU(a | E). Because each EU is linear in p = P(A | E), the action regions are exact:

- Continue for p < 1/7 (0.143)
- Monitor for 1/7 ≤ p < 5/9 (0.556)
- Safe Mode for p ≥ 5/9

The dashboard's low, moderate and high risk bands are these three regions renamed. They come from the utility table, not from validated spacecraft safety thresholds. Edit `UTILITY` in `backend/config.py` and the regions, bands and recommendations all update.

**The utility values are hypothetical project decision-support assumptions, not real spacecraft control commands.**

## Evaluation

Official test split (529 segments), fixed threshold 0.5. The dashboard recomputes these at start-up, and `tests/test_backend.py` asserts they equal the notebook's recorded output to four decimals:

| Model | Precision | Recall | F1 | PR-AUC | Brier score |
|---|---|---|---|---|---|
| TAN Bayesian Network | 0.8624 | 0.8319 | 0.8468 | 0.9267 | **0.0481** |
| HMM filtering | 0.8868 | 0.8319 | 0.8584 | 0.9259 | 0.0500 |
| HMM smoothing | 0.8796 | 0.8407 | **0.8597** | 0.9197 | 0.0515 |
| Decision Tree | 0.8110 | **0.9115** | 0.8583 | **0.9279** | 0.0556 |

A lower Brier score means better probabilistic accuracy. TAN gives the best-calibrated probabilities, while HMM smoothing has the highest F1. The highest F1 at one threshold does not make a model the best probabilistic model. The Decision Network on the filtered probabilities recommends Continue for 397 test segments, Monitor for 27 and Safe Mode for 105, matching the notebook.

The tests also confirm the cut points, the TAN inference values (prior 0.202, 0.702, 0.019, 0.985), the Markov blanket of Anomaly ({n_peaks}), the transition matrix, all four confusion matrices, and that filtering is causal: changing future segments does not change the current belief.

### Leakage checks

- Cut points, medians, TAN structure, CPTs, the Markov network, π, A and the Decision Tree are all fitted on `train == 1` rows only. `test_cut_points_ignore_test_rows` corrupts every test feature and confirms the cut points do not move.
- The HMM forward pass uses test segments' *features* to carry context along a channel, exactly as the notebook does, but never their labels.
- Online quantities use filtering, never smoothing.

## Differences from the notebook

These are worth knowing for the viva:

1. In notebook cell 25, a comment says "Use the HMM smoothed probability as the main temporal risk estimate", but the code beneath it uses `p_filt` (filtering). The dashboard follows the code, which is also what the notebook's markdown describes.
2. The notebook factorises `channel` and `sampling` over the full dataset. That only assigns integer codes to category names, uses no labels, and gives the same codes as the training split alone, so it is preserved unchanged.
3. The notebook builds the TAN tree with `nx.bfs_predecessors`, which NetworkX is deprecating. The dashboard uses the equivalent `nx.bfs_edges`. The structure is identical, and the tests confirm the numbers are unchanged.
4. Added for the dashboard and not in the notebook: the one-step prediction P(S<sub>t+1</sub> | O<sub>1:t</sub>), manual-input validation, per-variable evidence weights (log-likelihood ratios read from the learned CPTs), and the risk-band naming of the Decision Network regions.

## Limitations

- Next-segment risk is a one-step projection through a 2×2 transition matrix, so it is bounded to [0.156, 0.372] and can never forecast a strongly anomalous next segment on its own.
- The HMM is first order and its transitions are shared by all channels.
- Discretising into three bins throws away detail inside each bin. A value far outside the training range falls into the same bin as an ordinary LOW or HIGH value, which is why the dashboard warns about it.
- The utility table is an assumption made for the project, not operational guidance.
- The data is historical. Nothing here monitors a spacecraft live.

## Future scope

- Multi-step forecasts P(S<sub>t+k</sub> | O<sub>1:t</sub>) = P(S<sub>t</sub> | O<sub>1:t</sub>) A<sup>k</sup>, and the expected time until the next anomalous segment.
- Channel-specific or duration-dependent transition models (hidden semi-Markov models).
- Learning the utility table from operator preferences, or adding a value-of-information node for "request more telemetry".
- Replaying the dataset as a simulated stream, clearly labelled as a simulation.
- Calibration plots alongside the Brier score.
