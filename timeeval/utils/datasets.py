from pathlib import Path

import numpy as np
import pandas as pd


def extract_labels(df: pd.DataFrame) -> np.ndarray:
    labels: np.ndarray = df.values[:, -1].astype(np.float64)
    return labels


def extract_features(df: pd.DataFrame) -> np.ndarray:
    nb_channels = (len(df.columns) - 1) // 2
    features: np.ndarray = df.values[:, 1:-nb_channels]
    return features


def load_dataset(path: Path, target_channels = None) -> pd.DataFrame:
    columns = pd.read_csv(path, index_col=0, nrows=0).columns.tolist()

    if target_channels is not None:
        columns = [x for x in target_channels]
        columns = ["timestamp", *columns]

    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp", usecols=columns, infer_datetime_format=True)


def load_labels_only(path: Path, target_channels = None) -> np.ndarray:

    columns = pd.read_csv(path, index_col=0, nrows=0).columns.tolist()
    anomaly_columns = [x for x in columns if x.startswith("is_anomaly")]

    if target_channels is not None and anomaly_columns[0] != "is_anomaly":
        anomaly_columns = [f"is_anomaly_{x}" for x in target_channels]

    labels = pd.read_csv(path, usecols=anomaly_columns, dtype=np.uint8).values

    labels[labels > 1] = 1  # decide what to do with rare events and time gaps
    return labels
