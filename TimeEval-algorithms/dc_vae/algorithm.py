import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import argparse
import pandas as pd
import numpy as np
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from dateutil.parser import parse as parse_date
from typing import List
from tensorflow.compat.v1 import set_random_seed

from dc_vae.helpers import Config
from dc_vae.channel import Channel
from dc_vae.dc_vae import DCVAE


@dataclass
class CustomParameters:
    alpha: float = 3
    max_std: float = 7
    T: int = 128
    cnn_units: List[int] = field(default_factory=lambda: [64, 64, 64, 64, 64, 64])
    dil_rate: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    kernel: int = 2
    strs: int = 1
    batch_size: int = 64
    J: int = 2
    epochs: int = 100
    lr: float = 1e-3
    lr_decay: bool = False
    decay_rate: float = 0.96
    decay_step: int = 7000
    val_percent: float = 0.2
    validation_date_split: str = None
    seed: int = 123
    input_channels: List[str] = None
    input_channel_indices: List[int] = None  # do not use, automatically handled
    target_channels: List[str] = None
    target_channel_indices: List[int] = None  # do not use, automatically handled
    threshold_scores: bool = False


class AlgorithmArgs(argparse.Namespace):

    @property
    def ts(self) -> np.ndarray:
        print(f"Loading: {self.dataInput}")
        columns = pd.read_csv(self.dataInput, index_col="timestamp", nrows=0).columns.tolist()
        anomaly_columns = [x for x in columns if x.startswith("is_anomaly")]
        data_columns = columns[:-len(anomaly_columns)]

        dtypes = {col: np.float32 for col in data_columns}
        dtypes.update({col: np.uint8 for col in anomaly_columns})
        dataset = pd.read_csv(self.dataInput, index_col="timestamp", parse_dates=True, dtype=dtypes)

        if self.customParameters.input_channels is None or len(set(self.customParameters.input_channels).intersection(data_columns)) == 0:
            self.customParameters.input_channels = data_columns
            print(f"Input channels not given or not present in the data, selecting all the channels: {self.customParameters.input_channels}")
        else:
            self.customParameters.input_channels = [x for x in self.customParameters.input_channels if x in data_columns]

        if self.customParameters.target_channels is None or len(set(self.customParameters.target_channels).intersection(data_columns)) == 0:
            self.customParameters.target_channels = data_columns
            print(f"Target channels not given or not present in the data, selecting all the channels: {self.customParameters.target_channels}")
        else:
            self.customParameters.target_channels = [x for x in self.customParameters.target_channels if x in data_columns]

        # Remove unused columns from dataset
        all_used_channels = [x for x in data_columns if x in set(self.customParameters.input_channels + self.customParameters.target_channels)]
        all_used_anomaly_columns = [f"is_anomaly_{channel}" for channel in all_used_channels]
        if len(anomaly_columns) == 1 and anomaly_columns[0] == "is_anomaly":  # Handle datasets with only one global is_anomaly column
            for c in all_used_anomaly_columns:
                dataset[c] = dataset["is_anomaly"]
            dataset = dataset.drop(columns="is_anomaly")
        dataset = dataset.loc[:, all_used_channels + all_used_anomaly_columns]
        data_columns = dataset.columns.tolist()[:len(all_used_channels)]

        # Change channel names to index for further processing
        self.customParameters.input_channel_indices = [data_columns.index(x) for x in self.customParameters.input_channels]
        self.customParameters.target_channel_indices = [data_columns.index(x) for x in self.customParameters.target_channels]

        if self.executionType == "train":
            # Check if data should and can be splitted into train/val at some special timestamp
            validation_date_split = self.customParameters.validation_date_split
            if validation_date_split is not None:
                try:
                    validation_date_split = parse_date(validation_date_split)
                    if validation_date_split < dataset.index[0] or validation_date_split > dataset.index[-1]:
                        print(f"Cannot use validation_date_split '{validation_date_split}' because it is outside the data range")
                        validation_date_split = None
                except:
                    print(f"Cannot use validation_date_split '{validation_date_split}' because timestamp is not datetime")
                    validation_date_split = None

            # Find start and end points of fragments without anomalies
            target_anomaly_column = "is_anomaly"
            dataset[target_anomaly_column] = 0
            for channel in self.customParameters.target_channels:
                dataset.loc[dataset[f"is_anomaly_{channel}"] > 0, f"is_anomaly_{channel}"] = 1
                dataset[target_anomaly_column] |= dataset[f"is_anomaly_{channel}"]

            for col in all_used_anomaly_columns:
                dataset.drop(columns=[col], inplace=True)

            labels_groups = dataset.groupby(
                (dataset[target_anomaly_column].shift() != dataset[target_anomaly_column]).cumsum()
            )
            start_end_points = [
                (group[0], group[-1])
                for group in labels_groups.groups.values()
                if dataset.loc[group[0], target_anomaly_column] == 0
            ]
            dataset.drop(columns=[target_anomaly_column], inplace=True)  # at this point label columns are no longer needed

            # Binary channel has only two unique integer values
            binary_channels_mask = [np.sum(dataset.values[..., i].astype(np.int64) - dataset.values[..., i]) == 0 and
                                    len(np.unique(dataset.values[..., i])) == 2
                                    for i in range(dataset.values.shape[-1])]
            channels_minimums = np.min(dataset.values, axis=0)
            channels_maximums = np.max(dataset.values, axis=0)

            if validation_date_split is None:
                dataset = np.array([dataset.loc[start:end].values for start, end in start_end_points], dtype=object)
            else:
                train_data = []
                val_data = []
                for start_date, end_date in start_end_points:
                    if start_date < validation_date_split and end_date < validation_date_split:
                        train_data.append(dataset[start_date:end_date].values)
                    elif start_date > validation_date_split and end_date > validation_date_split:
                        val_data.append(dataset[start_date:end_date].values)
                    else:
                        train_data.append(dataset[start_date:validation_date_split].values)
                        val_data.append(dataset[validation_date_split:end_date].values)
                dataset = [np.array(train_data), np.array(val_data)]

        else:
            dataset = np.expand_dims(dataset.values[:, :-len(all_used_anomaly_columns)], axis=0).astype(np.float32)
            binary_channels_mask = None
            channels_minimums = None
            channels_maximums = None

        return dataset, binary_channels_mask, channels_minimums, channels_maximums

    @property
    def ts_for_alpha_selection(self):
        print(f"Loading: {self.dataInput}")
        all_used_channels = [x for x in set(self.customParameters.input_channels + self.customParameters.target_channels)]
        all_used_anomaly_columns = [f"is_anomaly_{channel}" for channel in all_used_channels]

        dtypes = {col: np.float32 for col in all_used_channels}
        dtypes.update({col: np.uint8 for col in all_used_anomaly_columns})
        dataset = pd.read_csv(self.dataInput, index_col="timestamp", parse_dates=True, dtype=dtypes,
                              usecols=["timestamp", *all_used_channels, *all_used_anomaly_columns])

       # dataset = dataset[dataset.index > parse_date(self.customParameters.validation_date_split)]

        for anomaly_col in all_used_anomaly_columns:
            dataset.loc[dataset[anomaly_col] > 0, anomaly_col] = 1

        X = dataset[all_used_channels].values
        y = dataset[all_used_anomaly_columns].values

        return X, y

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        hyper_params_path = os.path.join(os.path.dirname(args["dataOutput"]), "hyper_params.json")
        if os.path.isfile(hyper_params_path):
            with open(hyper_params_path, "r") as fh:
                hyper_params = json.load(fh)
            for key, value in hyper_params.items():
                filtered_parameters[key] = value
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def adapt_config_yaml(args: AlgorithmArgs) -> Config:
    params = asdict(args.customParameters)

    params["window_size"] = params["T"]
    params["meansOutput"] = f"{args.modelOutput}.means"
    params["stdsOutput"] = f"{args.modelOutput}.stds"

    config = Config.from_dict(params)
    if args.executionType == "train":
        config["train"] = True
        config["predict"] = False
    elif args.executionType == "execute":
        config["train"] = False
        config["predict"] = True

    return config


def train(args: AlgorithmArgs, model, channel: Channel):
    model.fit(channel, args.modelOutput)

    X, y = args.ts_for_alpha_selection
    train_means = np.atleast_1d(np.loadtxt(channel.config.meansOutput))
    train_stds = np.atleast_1d(np.loadtxt(channel.config.stdsOutput))
    X = (X - train_means) / train_stds
    model.alpha_selection(X, y, args.modelOutput, channel, load_model=False, custom_metrics=True)


def execute(args: AlgorithmArgs, model, channel: Channel, thresholded: bool = False):
    alpha = args.customParameters.alpha
    model.predict(args, channel,
                  load_model=True,
                  alpha_set_up=np.ones(channel.nb_output_channels) * alpha if alpha is not None else [],
                  alpha_set_down=np.ones(channel.nb_output_channels) * alpha if alpha is not None else [])


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.seed
    import random
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)


def main():
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    ts, binary_channels_mask, channels_minimums, channels_maximums = args.ts

    config = adapt_config_yaml(args)
    is_train = args.executionType == "train"

    channels = Channel(config)
    channels.shape_data(ts, binary_channels_mask, channels_minimums, channels_maximums, train=is_train)

    print(args)

    config = config.__dict__
    model = DCVAE(
        config['T'],
        channels.nb_input_channels,
        config["target_channel_indices"],
        config['cnn_units'],
        config['dil_rate'],
        config['kernel'],
        config['strs'],
        config['batch_size'],
        config['J'],
        config['epochs'],
        config['lr'],
        config['lr_decay'],
        config['decay_rate'],
        config['decay_step'],
        "model"
    )

    if is_train:
        train(args, model, channels)
    else:
        execute(args, model, channels, args.customParameters.threshold_scores)


if __name__ == "__main__":
    main()
