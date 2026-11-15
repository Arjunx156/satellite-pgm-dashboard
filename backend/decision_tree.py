"""Section 10 of the notebook: Decision Tree, a supervised ML benchmark, not the primary model."""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from . import config


class DecisionTreeBenchmark:
    def __init__(self, variables):
        self.variables = variables
        self.columns = None
        self.model = DecisionTreeClassifier(**config.DT_PARAMS)

    def _encode(self, D):
        # one-hot so LOW/MEDIUM/HIGH are not treated as ordered numbers
        X = pd.get_dummies(D[self.variables].astype(str), columns=self.variables, dtype=float)
        return X if self.columns is None else X.reindex(columns=self.columns, fill_value=0.0)

    def fit(self, D_train, y_train):
        X = self._encode(D_train)
        self.columns = X.columns
        self.model.fit(X, y_train)
        return self

    def predict_proba(self, D):
        return self.model.predict_proba(self._encode(D))[:, 1]

    def importance(self, top=12):
        return (pd.Series(self.model.feature_importances_, index=self.columns)
                .sort_values(ascending=False).head(top))
