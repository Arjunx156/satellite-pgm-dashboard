"""Section 9 of the notebook: metrics on the official test split, fixed 0.5 threshold."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from . import config


def evaluate(y_test, p_test) -> dict:
    yh = (np.asarray(p_test) >= config.THRESHOLD).astype(int)
    return {
        "Precision": precision_score(y_test, yh, zero_division=0),
        "Recall": recall_score(y_test, yh, zero_division=0),
        "F1": f1_score(y_test, yh, zero_division=0),
        "PR-AUC": average_precision_score(y_test, p_test),
        "Brier Score": brier_score_loss(y_test, p_test),
    }


def results_table(y_test, test_probabilities: dict) -> pd.DataFrame:
    return pd.DataFrame({name: evaluate(y_test, p) for name, p in test_probabilities.items()}).T.rename_axis("Model")


def confusion(y_test, p_test) -> np.ndarray:
    return confusion_matrix(y_test, (np.asarray(p_test) >= config.THRESHOLD).astype(int), labels=[0, 1])


def pr_curve(y_test, p_test):
    precision, recall, _ = precision_recall_curve(y_test, p_test)
    return precision, recall
