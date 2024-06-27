#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
import pickle

from typing import List
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from model import RandomBlackForestAnomalyDetector


@dataclass
class CustomParameters:
    train_window_size: int = 250
    n_estimators: int = 100  # number of forests
    max_features_per_estimator: float = 0.5  # fraction of features per forest
    n_trees: float = 100  # number of trees per forest
    max_features_method: str = "auto"  # "sqrt", "log2"
    bootstrap: bool = True
    max_samples: Optional[float] = None  # fraction of all samples
    # standardize: bool = False  # does not really influence the quality
    random_state: int = 42
    verbose: int = 0
    n_jobs: int = 1
    # the following parameters control the tree size
    max_depth: Optional[int] = 4
    min_samples_split: int = 2
    min_samples_leaf: int = 1
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
    df = pd.read_csv(config.dataInput)
    data = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -1].values
    return data, labels

def load_data(config: AlgorithmArgs, train=True) -> np.ndarray:
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

    if train:
        dataset = dataset[labels == 0] # cut-out anomalies

    return dataset


def train(config: AlgorithmArgs):
    np.random.seed(config.customParameters.random_state)
    data = load_data(config, train=True)

    print("Training random forest classifier")
    args = asdict(config.customParameters)
    del args["target_channels"]
    del args["target_channel_indices"]
    model = RandomBlackForestAnomalyDetector(**args).fit(data)
    print(f"Saving model to {config.modelOutput}")
    model.save(Path(config.modelOutput))

def execute(config: AlgorithmArgs):
    np.random.seed(config.customParameters.random_state)
    data = load_data(config, train=False)
    print(f"Loading model from {config.modelInput}")
    model = RandomBlackForestAnomalyDetector.load(Path(config.modelInput))

    print("Forecasting and calculating errors")
    scores = model.detect(data)
    np.savetxt(config.dataOutput, scores, delimiter=",")
    print(f"Stored anomaly scores at {config.dataOutput}")


def plot(data, predictions, scores, labels):
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler

    # for better visuals, align scores to value range of labels (0, 1)
    scores = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).reshape(-1)

    fig, axs = plt.subplots(data.shape[1]+1, sharex=True)
    for i in range(data.shape[1]):
        axs[i].plot(data[:, i], label="truth")
        axs[i].plot(predictions[:, i], label="predict")
        axs[i].legend()
    axs[-1].plot(labels, label="label")
    axs[-1].plot(scores, label="score")
    axs[-1].legend()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()
    set_random_state(config)
    print(f"Config: {config}")

    if config.executionType == "train":
        train(config)
    elif config.executionType == "execute":
        execute(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected either 'train' or 'execute'!")
