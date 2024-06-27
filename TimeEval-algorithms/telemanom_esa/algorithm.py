import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from telemanom.detector import Detector
from telemanom.modeling import Model
from telemanom.helpers import Config
from telemanom.channel import Channel
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


@dataclass
class CustomParameters:
    batch_size: int = 70
    smoothing_window_size: int = 30
    smoothing_perc: float = 0.05
    error_buffer: int = 100
    loss_metric: str = 'mse'
    optimizer: str = 'adam'
    split: float = 0.8
    validation_date_split: str = None
    dropout: float = 0.3
    lstm_batch_size: int = 64
    epochs: int = 35
    layers: List[int] = field(default_factory=lambda: [80, 80])
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.0003
    window_size: int = 250
    prediction_window_size: int = 10
    p: float = 0.13
    min_error_value: float = 0.05
    use_id: str = "internal-run-id"
    random_state: int = 42
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
                # dataset.loc[dataset[f"is_anomaly_{channel}"] == 2, f"is_anomaly_{channel}"] = 0
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
    # remap config keys
    params["validation_split"] = 1 - params["split"]
    params["patience"] = params["early_stopping_patience"]
    params["min_delta"] = params["early_stopping_delta"]
    params["l_s"] = params["window_size"]
    for k in ["split", "early_stopping_patience", "early_stopping_delta"]:
        del params[k]

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


def train(args: AlgorithmArgs, config: Config, channel: Channel):
    Model(config, config.use_id, channel, model_path=args.modelOutput)  # trains and saves model


def execute(args: AlgorithmArgs, config: Config, channels: Channel, thresholded: bool = False):
    detector = Detector(config=config, model_path=args.modelInput, result_path=args.dataOutput)
    errors = detector.predict(channels, args, thresholded)
    np.savetxt(args.dataOutput, errors, delimiter=",")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
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

    if is_train:
        train(args, config, channels)
    else:
        execute(args, config, channels, args.customParameters.threshold_scores)


if __name__ == "__main__":
    main()
