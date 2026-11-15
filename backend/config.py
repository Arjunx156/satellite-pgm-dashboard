"""Project constants, copied from the final notebook. Change them here, nowhere else."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASET_CSV = DATA_DIR / "dataset.csv"
SEGMENTS_CSV = DATA_DIR / "segments.csv"

# Section 3: discrete model variables (order matters for the TAN learner)
PREFERRED = ["kurtosis", "skew", "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
             "diff2_peaks", "diff_var", "var", "len"]
COUNT_FEATS = {"n_peaks", "smooth10_n_peaks", "smooth20_n_peaks"}
CATEGORICAL = ["sampling", "channel"]
CLASS_VAR = "anomaly"
STATE_NAMES = ["LOW", "MEDIUM", "HIGH"]

# Section 4: Laplace smoothing for CPTs; Section 8: Laplace smoothing for pi and A
CPT_ALPHA = 1.0

# Section 9: fixed classification threshold, no tuning
THRESHOLD = 0.5

# Section 10 (Decision Tree benchmark)
DT_PARAMS = dict(criterion="entropy", max_depth=5, min_samples_leaf=10,
                 class_weight="balanced", random_state=42)

# Section 11: Decision Network utilities. Rows = actions, columns = (Nominal, Anomalous).
# Project decision-support assumptions, not spacecraft operating commands.
ACTIONS = ["Continue", "Monitor", "Safe Mode"]
STATES = ["Nominal", "Anomalous"]
UTILITY = {
    "Continue": (10, -20),
    "Monitor": (6, 4),
    "Safe Mode": (1, 8),
}

# Human-readable labels for the dashboard
FEATURE_LABELS = {
    "kurtosis": "Kurtosis",
    "skew": "Skewness",
    "n_peaks": "Number of peaks",
    "smooth10_n_peaks": "Peaks after 10-pt smoothing",
    "smooth20_n_peaks": "Peaks after 20-pt smoothing",
    "diff2_peaks": "Peaks in 2nd difference",
    "diff_var": "Variance of 1st difference",
    "var": "Variance",
    "len": "Segment length (samples)",
    "sampling": "Sampling rate",
    "channel": "Channel",
    "anomaly": "Anomaly",
}
