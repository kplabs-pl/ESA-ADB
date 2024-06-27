#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
import pickle

from typing import List
from dataclasses import dataclass
from pyod.models.hbos import HBOS


@dataclass
class CustomParameters:
    n_bins: int = 10
    alpha: float = 0.1
    bin_tol: float = 0.5
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


def load_data(config: AlgorithmArgs) -> np.ndarray:
    print(f"Loading: {config.dataInput}")
    columns = pd.read_csv(config.dataInput, index_col="timestamp", nrows=0).columns.tolist()
    anomaly_columns = [x for x in columns if x.startswith("is_anomaly")]
    data_columns = columns[:-len(anomaly_columns)]

    dtypes = {col: np.float32 for col in data_columns}
    dtypes.update({col: np.uint8 for col in anomaly_columns})
    dataset = pd.read_csv(config.dataInput, index_col="timestamp", parse_dates=True, dtype=dtypes)

    if config.customParameters.target_channels is None or len(
            set(config.customParameters.target_channels).intersection(data_columns)) == 0:
        config.customParameters.target_channels = data_columns
        print(
            f"Input channels not given or not present in the data, selecting all the channels: {config.customParameters.target_channels}")
        all_used_channels = [x for x in data_columns if x in set(config.customParameters.target_channels)]
        all_used_anomaly_columns = [f"is_anomaly_{channel}" for channel in all_used_channels]
    else:
        config.customParameters.target_channels = [x for x in config.customParameters.target_channels if x in data_columns]

        # Remove unused columns from dataset
        all_used_channels = [x for x in data_columns if x in set(config.customParameters.target_channels)]
        all_used_anomaly_columns = [f"is_anomaly_{channel}" for channel in all_used_channels]
        if len(anomaly_columns) == 1 and anomaly_columns[0] == "is_anomaly":  # Handle datasets with only one global is_anomaly column
            for c in all_used_anomaly_columns:
                dataset[c] = dataset["is_anomaly"]
            dataset = dataset.drop(columns="is_anomaly")
        dataset = dataset.loc[:, all_used_channels + all_used_anomaly_columns]
        data_columns = dataset.columns.tolist()[:len(all_used_channels)]

    # Change channel names to index for further processing
    config.customParameters.target_channel_indices = [data_columns.index(x) for x in config.customParameters.target_channels]

    labels = dataset[all_used_anomaly_columns].to_numpy()
    dataset = dataset.to_numpy()[:, config.customParameters.target_channel_indices]
    labels = labels.max(axis=1)
    labels[labels > 0] = 1
    contamination = labels.sum() / len(labels)
    # Use smallest positive float as contamination if there are no anomalies in dataset
    contamination = np.nextafter(0, 1) if contamination == 0. else contamination

    return dataset, contamination


def train(config: AlgorithmArgs):
    set_random_state(config)
    data, contamination = load_data(config)

    clf = HBOS(
        contamination=contamination,
        n_bins=config.customParameters.n_bins,
        alpha=config.customParameters.alpha,
        tol=config.customParameters.bin_tol
    )
    clf.fit(data)
    clfPickle = open(config.modelOutput, "wb")
    pickle.dump(clf, clfPickle)
    clfPickle.close()
    print(f"Model saved to {config.modelOutput}")


def execute(config: AlgorithmArgs):
    set_random_state(config)
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
