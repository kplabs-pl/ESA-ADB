#!/usr/bin/env python3
import json
import sys
import pickle

from typing import List
import numpy as np
import eif as iso
import pandas as pd
from pathlib import Path
from typing import Optional


class Config:
    dataInput: Path
    dataOutput: Path
    modelInput: Path
    modelOutput: Path
    executionType: str
    n_trees: int
    max_samples: Optional[float]
    extension_level: int
    limit: int
    random_state: int
    target_channels: List[str] = None
    target_channel_indices: List[int] = None  # do not use, automatically handled

    def __init__(self, params):
        self.dataInput = Path(params.get("dataInput", "/data/dataset.csv"))
        self.dataOutput = Path(
            params.get("dataOutput", "/results/anomaly_window_scores.ts")
        )
        self.modelInput = Path(params.get("modelInput", "/results/model.h5"))
        self.modelOutput = Path(
            params.get("modelOutput", "/results/model.h5")
        )
        self.executionType = params.get("executionType", "execute")
        try:
            customParameters = params["customParameters"]
        except KeyError:
            customParameters = {}
        self.n_trees = customParameters.get("n_trees", 200)
        self.max_samples = customParameters.get("max_samples", None)
        self.extension_level = customParameters.get("extension_level", None)
        self.limit = customParameters.get("limit", None)
        self.random_state = customParameters.get("random_state", 42)
        self.customParameters = customParameters


def set_random_state(config) -> None:
    seed = config.random_state
    import random

    random.seed(seed)
    np.random.seed(seed)


def read_data(data_path: Path, config):
    print(f"Loading: {data_path}")
    columns = pd.read_csv(data_path, index_col="timestamp", nrows=0).columns.tolist()
    anomaly_columns = [x for x in columns if x.startswith("is_anomaly")]
    data_columns = columns[:-len(anomaly_columns)]

    dtypes = {col: np.float32 for col in data_columns}
    dtypes.update({col: np.uint8 for col in anomaly_columns})
    dataset = pd.read_csv(data_path, index_col="timestamp", parse_dates=True, dtype=dtypes)

    if config.customParameters['target_channels'] is None or len(
            set(config.customParameters['target_channels']).intersection(data_columns)) == 0:
        config.customParameters['target_channels'] = data_columns
        print(
            f"Input channels not given or not present in the data, selecting all the channels: {config.customParameters['target_channels']}")
        all_used_channels = [x for x in data_columns if x in set(config.customParameters['target_channels'])]
        all_used_anomaly_columns = [f"is_anomaly_{channel}" for channel in all_used_channels]
    else:
        config.customParameters['target_channels'] = [x for x in config.customParameters['target_channels'] if
                                                  x in data_columns]

        # Remove unused columns from dataset
        all_used_channels = [x for x in data_columns if x in set(config.customParameters['target_channels'])]
        all_used_anomaly_columns = [f"is_anomaly_{channel}" for channel in all_used_channels]
        if len(anomaly_columns) == 1 and anomaly_columns[0] == "is_anomaly":  # Handle datasets with only one global is_anomaly column
            for c in all_used_anomaly_columns:
                dataset[c] = dataset["is_anomaly"]
            dataset = dataset.drop(columns="is_anomaly")
        dataset = dataset.loc[:, all_used_channels + all_used_anomaly_columns]
        data_columns = dataset.columns.tolist()[:len(all_used_channels)]

    # Change channel names to index for further processing
    config.customParameters['target_channel_indices'] = [data_columns.index(x) for x in config.customParameters['target_channels']]

    labels = dataset[all_used_anomaly_columns].to_numpy()
    dataset = dataset.to_numpy()[:, config.customParameters['target_channel_indices']]
    labels = labels.max(axis=1)
    labels[labels > 0] = 1
    contamination = labels.sum() / len(labels)
    # Use smallest positive float as contamination if there are no anomalies in dataset
    contamination = np.nextafter(0, 1) if contamination == 0. else contamination

    return dataset, contamination


def create_forest(X: int, config: Config):
    print("Creating forest")
    n_trees = config.n_trees
    if config.max_samples:
        sample_size = int(config.max_samples * X.shape[0])
    else:
        sample_size = min(256, X.shape[0])
    limit = config.limit or int(np.ceil(np.log2(sample_size)))
    extension_level = config.extension_level or X.shape[1] - 1
    forest = iso.iForest(
        X,
        ntrees=n_trees,
        sample_size=sample_size,
        limit=limit,
        ExtensionLevel=extension_level,
    )
    return forest

def train(config: Config):
    print(iso.__version__)
    set_random_state(config)
    X, contamination = read_data(config.dataInput, config)
    print(f"Data shape {X.shape}")
    forest = create_forest(X, config)

    print("Computing scores")
    clfPickle = open(config.modelOutput, "wb")
    pickle.dump(forest, clfPickle)
    clfPickle.close()
    np.savetxt(str(config.modelOutput) + ".contamination.csv", [contamination], delimiter=",", fmt="%f")
    print(f"Model saved to {config.modelOutput}")

def execute(config: Config):
    print(iso.__version__)
    set_random_state(config)
    X, _ = read_data(config.dataInput, config)
    print(f"Data shape {X.shape}")
    contamination = np.loadtxt(str(config.modelOutput) + ".contamination.csv")

    clfPickle = open(config.modelInput, "rb")
    forest = pickle.load(clfPickle)
    clfPickle.close()
    print(f"Model loaded {config.modelInput}")

    print("Computing scores")
    scores = forest.compute_paths(X_in=X)

    threshold = np.percentile(scores, 100 * (1 - contamination))
    scores[scores >= threshold] = 1
    scores[scores < threshold] = 0

    np.savetxt(config.dataOutput, scores, delimiter=",")
    print(f"Results saved to {config.dataOutput}")

def train_n_execute(config: Config):
    print(iso.__version__)
    set_random_state(config)
    X, contamination = read_data(config.dataInput, config)
    print(f"Data shape {X.shape}")
    forest = create_forest(X.astype(np.float64), config)

    set_random_state(config)
    test_data_input = config.dataInput.parents[0].joinpath("All_channels_50.test.csv")
    X, _ = read_data(test_data_input, config)
    print(f"Data shape {X.shape}")

    print("Computing scores")
    scores = forest.compute_paths(X_in=X.astype(np.float64))

    threshold = np.percentile(scores, 100 * (1 - contamination))
    scores[scores >= threshold] = 1
    scores[scores < threshold] = 0

    np.savetxt(config.dataOutput, scores, delimiter=",")
    print(f"Results saved to {config.dataOutput}")


def parse_args():
    print(sys.argv)
    if len(sys.argv) < 2:
        print("No arguments supplied, using default arguments!", file=sys.stderr)
        params = {}
    elif len(sys.argv) > 2:
        print("Wrong number of arguments supplied! Single JSON-String expected!", file=sys.stderr)
        exit(1)
    else:
        params = json.loads(sys.argv[1])
    return Config(params)


if __name__ == "__main__":
    config = parse_args()
    if config.executionType == "train":
        train_n_execute(config)
    elif config.executionType == "execute":
        pass
    else:
        raise Exception("Invalid Execution type given")
