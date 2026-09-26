#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
import pickle

from typing import List
from typing import Optional
from dataclasses import dataclass
from pyod.models.pca import PCA


@dataclass
class CustomParameters:
    n_components: Optional[int] = None
    n_selected_components: Optional[int] = None
    whiten: bool = False
    svd_solver: str = 'auto'
    tol: float = 0.0
    max_iter: Optional[int] = None
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


def get_columns_names(filepath: str) -> tuple[list[str], list[str]]:
    columns = pd.read_csv(filepath, index_col="timestamp", nrows=0).columns.tolist()
    anomaly_columns = [col for col in columns if col.startswith("is_anomaly")]
    data_columns = columns[:-len(anomaly_columns)] if anomaly_columns else columns
    return data_columns, anomaly_columns


def read_dataset(filepath: str, data_cols: list[str], anomaly_cols: list[str]) -> pd.DataFrame:
    dtypes = {col: np.float32 for col in data_cols}
    dtypes.update({col: np.uint8 for col in anomaly_cols})
    return pd.read_csv(filepath, index_col="timestamp", parse_dates=True, dtype=dtypes)


def get_valid_channels(raw_channels: list[str], data_cols: list[str], sort: bool) -> list[str]:
    if not raw_channels:
        print(f"No target_channels provided. Using all data columns: {data_cols}")
        valid_channels = data_cols
    else:
        valid_channels = list(dict.fromkeys([ch for ch in raw_channels if ch in data_cols]))
        if not valid_channels:
            print("No valid target channels found in dataset, falling back to all data columns.")
            valid_channels = data_cols

    if sort:
        valid_channels.sort()

    return valid_channels


def unravel_global_annotation(dataset: pd.DataFrame, original_anomaly_cols: list[str],
                              target_anomaly_cols: list[str]) -> pd.DataFrame:
    if len(original_anomaly_cols) == 1 and original_anomaly_cols[0] == "is_anomaly":  # Handle datasets with only one global is_anomaly column
        for col in target_anomaly_cols:
            dataset[col] = dataset["is_anomaly"]
        dataset = dataset.drop(columns="is_anomaly")
    return dataset


def get_means_stds(data: np.ndarray, labels: np.ndarray, config: AlgorithmArgs) -> tuple[np.ndarray, np.ndarray]:
    means_path = str(config.modelOutput) + ".means.txt"
    stds_path = str(config.modelOutput) + ".stds.txt"

    if config.executionType == "train":
        means = [np.mean(data[:, i][labels[:, i] == 0]) for i in range(data.shape[1])]
        stds = [np.std(data[:, i][labels[:, i] == 0]) for i in range(data.shape[1])]
        stds = np.where(np.asarray(stds) == 0, 1, stds)  # do not divide constant signals by zero

        np.savetxt(means_path, means)
        np.savetxt(stds_path, stds)

    elif config.executionType == "execute":
        means = np.atleast_1d(np.loadtxt(means_path))
        stds = np.atleast_1d(np.loadtxt(stds_path))

    return means, stds


def load_data(config: AlgorithmArgs) -> tuple[np.ndarray, float]:
    print(f"Loading: {config.dataInput}")

    data_columns, anomaly_columns = get_columns_names(config.dataInput)
    dataset = read_dataset(config.dataInput, data_columns, anomaly_columns)

    target_channels = get_valid_channels(
        config.customParameters.target_channels, data_columns, sort=True
    )
    config.customParameters.target_channels = target_channels

    target_anomaly_columns = [f"is_anomaly_{ch}" for ch in target_channels]
    dataset = unravel_global_annotation(dataset, anomaly_columns, target_anomaly_columns)
    dataset = dataset.loc[:, target_channels + target_anomaly_columns]

    data = dataset[target_channels].to_numpy()
    labels = dataset[target_anomaly_columns].to_numpy()

    means, stds = get_means_stds(data, labels, config)

    # Binary labels: 0 or 1 (if any anomaly is present)
    labels = labels.max(axis=1)
    labels[labels > 0] = 1
    contamination = labels.sum() / len(labels)
    contamination = np.nextafter(0, 1) if contamination == 0 else contamination

    data = (data - means) / stds

    return data, contamination


def train(config: AlgorithmArgs):
    data, contamination = load_data(config)

    clf = PCA(
        contamination=contamination,
        n_components=config.customParameters.n_components,
        n_selected_components=config.customParameters.n_selected_components,
        whiten=config.customParameters.whiten,
        svd_solver=config.customParameters.svd_solver,
        tol=config.customParameters.tol,
        iterated_power=config.customParameters.max_iter or "auto",
        random_state=config.customParameters.random_state,
        copy=True,
        weighted=True,
        standardization=False,
    )
    clf.fit(data)
    clfPickle = open(config.modelOutput, "wb")
    pickle.dump(clf, clfPickle)
    clfPickle.close()
    print(f"Model saved to {config.modelOutput}")


def execute(config: AlgorithmArgs):
    data, contamination = load_data(config)

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
