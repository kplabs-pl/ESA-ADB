from telemanom.detector import Detector
from telemanom.modeling import Model
from telemanom.helpers import Config
from telemanom.channel import Channel
import argparse
import pandas as pd
import numpy as np
import json
import sys
from dataclasses import dataclass, asdict, field
from dateutil.parser import parse as parse_date
from typing import List
from tensorflow.compat.v1 import set_random_seed


CHANNEL_ID = "0"


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
    use_id: str = "internal-run-id"
    random_state: int = 42
    target_channel: str = None


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        dataset = pd.read_csv(self.dataInput)
        nb_channels = (len(dataset.columns) - 1) // 2
        if self.customParameters.target_channel is None or self.customParameters.target_channel not in dataset.columns:
            self.customParameters.target_channel = dataset.columns[1]
            print(f"Target channel not given or not present in the data, selecting the default: {self.customParameters.target_channel}")

        if self.executionType == "train":
            # Check if data should and can be splitted into train/val at some special timestamp
            validation_date_split = self.customParameters.validation_date_split
            if validation_date_split is not None:
                try:
                    validation_date_split = parse_date(validation_date_split)
                    timestamps = dataset.timestamp.map(parse_date)
                    if validation_date_split < timestamps.iloc[0] or validation_date_split > timestamps.iloc[-1]:
                        print(f"Cannot use validation_date_split '{validation_date_split}' because it is outside the data range")
                        validation_date_split = None
                except:
                    print(f"Cannot use validation_date_split '{validation_date_split}' because timestamp is not datetime")
                    validation_date_split = None

            # Find start and end points of fragments without anomalies
            dataset = dataset.reset_index()
            labels_groups = dataset.groupby(
                (dataset[f"is_anomaly_{self.customParameters.target_channel}"].shift() != dataset[f"is_anomaly_{self.customParameters.target_channel}"]).cumsum()
            )
            start_end_points = [
                (group[0], group[-1])
                for group in labels_groups.groups.values()
                if dataset.loc[group[0], f"is_anomaly_{self.customParameters.target_channel}"] == 0
            ]

            if validation_date_split is None:
                data = []
                values = dataset.values[:, 2:-nb_channels]
                for start, end in start_end_points:
                    data.append(values[start:end])
                data = np.array(data)
            else:
                dataset = dataset.set_index(timestamps)
                train_data = []
                val_data = []
                values = dataset.iloc[:, 2:-nb_channels]
                for start, end in start_end_points:
                    start_date = parse_date(dataset.timestamp[start])
                    end_date = parse_date(dataset.timestamp[end])
                    if start_date < validation_date_split and end_date < validation_date_split:
                        train_data.append(values[start_date:end_date].values)
                    elif start_date > validation_date_split and end_date > validation_date_split:
                        val_data.append(values[start_date:end_date].values)
                    else:
                        train_data.append(values[start_date:validation_date_split].values)
                        val_data.append(values[validation_date_split:end_date].values)
                data = [np.array(train_data), np.array(val_data)]

            # Change target channel name to index for further processing
            self.customParameters.target_channel = dataset.columns.get_loc(self.customParameters.target_channel) - 2
        else:
            data = np.expand_dims(dataset.values[:, 1:-nb_channels], axis=0)

            # Change target channel name to index for further processing
            self.customParameters.target_channel = dataset.columns.get_loc(self.customParameters.target_channel) - 1

        return data

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
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


def execute(args: AlgorithmArgs, config: Config, channel: Channel):
    detector = Detector(config=config, model_path=args.modelInput, result_path=args.dataOutput)
    errors = detector.predict([channel])[0]
    errors.tofile(args.dataOutput, sep="\n")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)


def main():
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    ts = args.ts

    config = adapt_config_yaml(args)
    is_train = args.executionType == "train"

    single_channel = Channel(config)
    single_channel.set_data(ts, train=is_train)

    if is_train:
        train(args, config, single_channel)
    else:
        execute(args, config, single_channel)


if __name__ == "__main__":
    main()
