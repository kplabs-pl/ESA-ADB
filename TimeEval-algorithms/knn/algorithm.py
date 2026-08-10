#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
import pickle

from typing import List, Tuple
from dataclasses import dataclass
from pyod.models.knn import KNN


@dataclass
class CustomParameters:
    n_neighbors: int = 5
    leaf_size: int = 30
    method: str = "largest"
    radius: float = 1.0
    distance_metric_order: int = 2
    n_jobs: int = 1
    algorithm: str = "auto"  # using default is fine
    distance_metric: str = "minkowski"  # using default is fine
    random_state: int = 42
    target_channels: List[str] = None
    target_channel_indices: List[int] = None  # do not use, automatically handled


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


def get_valid_channels(raw_channels: List[str], data_cols: List[str]) -> List[str]:
    if not raw_channels:
        print(f"No target_channels provided. Using all data columns (original order): {data_cols}")
        return data_cols
    else:
        filtered = [ch for ch in raw_channels if ch in data_cols]
        valid_channels = list(dict.fromkeys(filtered))

        if not valid_channels:
            print("No valid target channels found. Falling back to all data columns.")
            return data_cols

        return valid_channels

def get_columns_names(filepath: str) -> tuple[list[str], list[str]]:
    columns = pd.read_csv(filepath, index_col="timestamp", nrows=0).columns.tolist()
    anomaly_columns = [col for col in columns if col.startswith("is_anomaly")]
    data_columns = columns[:-len(anomaly_columns)] if anomaly_columns else columns
    return data_columns, anomaly_columns


def read_dataset(filepath: str, data_cols: list[str], anomaly_cols: list[str]) -> pd.DataFrame:
    dtypes = {col: np.float32 for col in data_cols}
    dtypes.update({col: np.uint8 for col in anomaly_cols})
    return pd.read_csv(filepath, index_col="timestamp", parse_dates=True, dtype=dtypes)


def unravel_global_annotation(dataset: pd.DataFrame, anomaly_columns: List[str],
                              used_channels: List[str]) -> pd.DataFrame:
    if len(anomaly_columns) == 1 and anomaly_columns[0] == "is_anomaly":  # Handle datasets with only one global is_anomaly column
        for ch in used_channels:
            dataset[f"is_anomaly_{ch}"] = dataset["is_anomaly"]
        dataset = dataset.drop(columns="is_anomaly")
    return dataset


def filter_to_used_columns(dataset: pd.DataFrame, used_channels: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    anomaly_cols = [f"is_anomaly_{ch}" for ch in used_channels]
    # Remove unused columns from dataset
    dataset = dataset.loc[:, used_channels + anomaly_cols]
    return dataset, anomaly_cols


def get_labels(dataset: pd.DataFrame, anomaly_cols: List[str]) -> np.ndarray:
    return dataset[anomaly_cols].to_numpy()


def normalize_data(
    dataset: np.ndarray,
    labels: np.ndarray,
    model_output_path: str,
    mode: str
) -> Tuple:

    means_path = model_output_path + ".means.txt"
    stds_path = model_output_path + ".stds.txt"

    if mode == "train":
        train_means = [np.mean(dataset[:, i][labels[:, i] == 0]) for i in range(dataset.shape[-1])]
        np.savetxt(means_path, train_means)

        train_stds = [np.std(dataset[:, i][labels[:, i] == 0].astype(float)) for i in range(dataset.shape[-1])]
        train_stds = np.asarray(train_stds)
        train_stds = np.where(train_stds == 0, 1, train_stds)  # do not divide constant signals by zero
        np.savetxt(stds_path, train_stds)
    elif mode == "execute":
        train_means = np.atleast_1d(np.loadtxt(means_path))
        train_stds = np.atleast_1d(np.loadtxt(stds_path))

    normalized_data = (dataset - train_means) / train_stds
    return normalized_data, train_means, train_stds


def load_data(config) -> Tuple[np.ndarray, float]:
    print(f"Loading: {config.dataInput}")

    data_columns, anomaly_columns = get_columns_names(config.dataInput)
    dataset = read_dataset(config.dataInput, data_columns, anomaly_columns)

    raw_channels = config.customParameters.target_channels
    target_channels = get_valid_channels(raw_channels, data_columns)
    config.customParameters.target_channels = target_channels

    used_channels = [ch for ch in data_columns if ch in set(target_channels)]
    used_anomaly_cols = [f"is_anomaly_{ch}" for ch in used_channels]

    dataset = unravel_global_annotation(dataset, anomaly_columns, used_channels)

    dataset, used_anomaly_cols = filter_to_used_columns(dataset, used_channels)

    labels_matrix = get_labels(dataset, used_anomaly_cols)

    data_matrix = dataset[used_channels].to_numpy()

    data_matrix, train_means, train_stds = normalize_data(
        data_matrix, labels_matrix, str(config.modelOutput), config.executionType
    )

    labels = labels_matrix.max(axis=1)
    labels[labels > 0] = 1

    contamination = labels.sum() / len(labels)
    # Use smallest positive float as contamination if there are no anomalies in dataset
    contamination = np.nextafter(0, 1) if contamination == 0. else contamination

    return data_matrix, contamination




def train(config: AlgorithmArgs):
    set_random_state(config)
    data, contamination = load_data(config)

    clf = KNN(
        contamination=contamination,
        n_neighbors=config.customParameters.n_neighbors,
        method=config.customParameters.method,
        radius=config.customParameters.radius,
        leaf_size=config.customParameters.leaf_size,
        n_jobs=config.customParameters.n_jobs,
        algorithm=config.customParameters.algorithm,
        metric=config.customParameters.distance_metric,
        metric_params=None,
        p=config.customParameters.distance_metric_order,
    )
    clf.fit(data)
    clfPickle = open(config.modelOutput, "wb")
    pickle.dump(clf, clfPickle)
    clfPickle.close()
    print(f"Model saved to {config.modelOutput}")


def execute(config: AlgorithmArgs):
    data, _ = load_data(config)

    clfPickle = open(config.modelInput, "rb")
    clf = pickle.load(clfPickle)
    clfPickle.close()
    print(f"Model loaded {config.modelInput}")

    result = clf.predict(data)

    np.savetxt(config.dataOutput, result, delimiter=",")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()
    print(f"Config: {config}")

    if config.executionType == "train":
        train(config)
    elif config.executionType == "execute":
        execute(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected either 'train' or 'execute'!")
