"""Section 3 of the notebook: LOW / MEDIUM / HIGH discretisation fitted on training rows only."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


def fit_edges(x, count=False):
    # two cut points -> 3 states. Fitted on training values only.
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([])
    if count:
        return np.array([1.5, 2.5])
    e = np.unique(np.quantile(x, [1 / 3, 2 / 3]))
    if len(e) < 2:                                   # repeated values: use value-based midpoints
        u = np.unique(x)
        e = ((u[:-1] + u[1:]) / 2)[:2]
    return e


@dataclass
class Discretizer:
    feats: list
    cats: list
    edges: dict = field(default_factory=dict)        # cut points per numeric feature
    fill: dict = field(default_factory=dict)         # training median, used for missing values
    ranges: dict = field(default_factory=dict)       # training min / max, used for input warnings
    levels: dict = field(default_factory=dict)       # sorted category values per categorical var
    card: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pd.DataFrame, train_mask: np.ndarray) -> "Discretizer":
        feats = [f for f in config.PREFERRED if f in df.columns]
        cats = [c for c in config.CATEGORICAL if c in df.columns]
        d = cls(feats=feats, cats=cats)
        for f in feats:
            x = df.loc[train_mask, f].to_numpy(float)
            d.edges[f] = fit_edges(x, f in config.COUNT_FEATS)
            d.fill[f] = np.nanmedian(x) if np.isfinite(x).any() else 0.0
            d.ranges[f] = (float(np.nanmin(x)), float(np.nanmax(x)))
        for c in cats:
            # notebook: pd.factorize(df[c], sort=True) -> codes follow sorted category values
            d.levels[c] = list(pd.factorize(df[c], sort=True)[1])
        d.card = {f: 3 for f in feats}
        d.card.update({c: len(d.levels[c]) for c in cats})
        d.card[config.CLASS_VAR] = 2
        return d

    @property
    def variables(self):
        return self.feats + self.cats

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for f in self.feats:
            out[f] = np.digitize(df[f].fillna(self.fill[f]).to_numpy(float), self.edges[f])
        for c in self.cats:
            codes = pd.Categorical(df[c], categories=self.levels[c]).codes
            if (codes < 0).any():
                raise ValueError(f"Unknown {c} value(s): {sorted(set(df[c][codes < 0]))}")
            out[c] = codes
        return out

    def cut_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {f: [f"{e:.3g}" for e in self.edges[f]] + ["-"] * (2 - len(self.edges[f])) for f in self.feats},
            index=["LOW | MEDIUM cut", "MEDIUM | HIGH cut"]).T


@dataclass
class PreparedInput:
    raw: dict             # value per variable after filling missing values
    codes: dict           # discrete code per variable
    filled: list          # variables that were missing and got the training median
    warnings: list        # human-readable range warnings


def prepare_record(record: dict, disc: Discretizer) -> PreparedInput:
    """Validate and discretise one manually entered segment."""
    raw, filled, warnings = {}, [], []
    for c in disc.cats:
        value = record.get(c)
        if value is None or value not in disc.levels[c]:
            raise ValueError(f"Please choose a valid {config.FEATURE_LABELS[c].lower()} "
                             f"({', '.join(map(str, disc.levels[c]))}).")
        raw[c] = value
    for f in disc.feats:
        value = record.get(f)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raw[f] = float(disc.fill[f])
            filled.append(f)
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Please enter a valid number for {config.FEATURE_LABELS[f].lower()}.") from None
        if not np.isfinite(value):
            raise ValueError(f"{config.FEATURE_LABELS[f]} must be a finite number.")
        lo, hi = disc.ranges[f]
        if value < lo or value > hi:
            warnings.append(f"{config.FEATURE_LABELS[f]} = {value:.4g} is outside the training range "
                            f"[{lo:.4g}, {hi:.4g}]. It is binned as "
                            f"{'LOW' if value < lo else 'HIGH'}, but the model has not seen such values.")
        # the training median itself may be fractional (e.g. len = 68.5); don't warn about it
        if f in config.COUNT_FEATS | {"len", "diff2_peaks"} and value != int(value) and value != disc.fill[f]:
            warnings.append(f"{config.FEATURE_LABELS[f]} is a count; {value} is not a whole number.")
        if f in {"var", "diff_var", "len"} and value < 0:
            raise ValueError(f"{config.FEATURE_LABELS[f]} cannot be negative.")
        raw[f] = value
    frame = pd.DataFrame([raw])
    codes = disc.transform(frame).iloc[0].astype(int).to_dict()
    return PreparedInput(raw=raw, codes=codes, filled=filled, warnings=warnings)
