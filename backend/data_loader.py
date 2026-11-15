"""Section 2 of the notebook: load segments, order them in time, keep the official split."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


class DataError(RuntimeError):
    """Raised when the dataset files are missing or inconsistent."""


@dataclass(frozen=True)
class Dataset:
    df: pd.DataFrame          # one row per segment, sorted by channel then start time
    train_mask: np.ndarray    # official `train == 1` flag
    test_mask: np.ndarray
    y: np.ndarray             # 0 = nominal, 1 = anomalous


def load_dataset(dataset_csv=config.DATASET_CSV, segments_csv=config.SEGMENTS_CSV) -> Dataset:
    for path in (dataset_csv, segments_csv):
        if not path.exists():
            raise DataError(f"Missing data file: {path}. Put dataset.csv and segments.csv in the data/ folder.")

    df = pd.read_csv(dataset_csv)
    required = set(config.PREFERRED) | set(config.CATEGORICAL) | {"segment", "anomaly", "train"}
    missing = required - set(df.columns)
    if missing:
        raise DataError(f"dataset.csv is missing columns: {sorted(missing)}")

    seg = pd.read_csv(segments_csv, usecols=["segment", "timestamp", "anomaly"])
    seg["timestamp"] = pd.to_datetime(seg["timestamp"], utc=True, format="ISO8601")

    df["start_time"] = df["segment"].map(seg.groupby("segment")["timestamp"].min())
    seg_labels = seg.groupby("segment")["anomaly"].first().reindex(df["segment"]).to_numpy()
    if not (seg_labels == df["anomaly"].to_numpy()).all():
        raise DataError("Labels differ between dataset.csv and segments.csv.")

    # chronological inside each channel, exactly as the notebook
    df = df.sort_values(["channel", "start_time", "segment"]).reset_index(drop=True)

    train_mask = (df["train"] == 1).to_numpy()
    return Dataset(df=df, train_mask=train_mask, test_mask=~train_mask,
                   y=df["anomaly"].astype(int).to_numpy())
