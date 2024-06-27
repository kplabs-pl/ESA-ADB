import argparse
import os
import statistics
from pathlib import Path
from glob import glob

import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.parser import parse as parse_date

from timeeval import DatasetManager, Datasets
from timeeval.datasets import DatasetAnalyzer, DatasetRecord, AnomalyLength
from utils import AnnotationLabel, encode_telecommands, find_full_time_range


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a folder with ESA Mission2 dataset.",
    )
    return parser.parse_args()


dataset_splits = {"1_months": "2000-02-01",
                  "5_months": "2000-06-01",
                  "10_months": "2000-11-01",
                  "21_months": "2001-10-01"}
test_data_split = "2001-10-01"
data_raw_folder = parse_args().input_path

current_dir = os.path.dirname(os.path.realpath(__file__))
data_processed_folder = os.path.abspath(os.path.join(current_dir, "../../data/preprocessed"))

dataset_collection_name = os.path.basename(data_raw_folder)
source_folder = Path(data_raw_folder)
target_folder = Path(data_processed_folder)

print(f"Looking for source datasets in {source_folder.absolute()} and saving processed datasets in {target_folder.absolute()}")

# set dataset features
dataset_type = "real"
input_type = "multivariate"
datetime_index = True
train_is_normal = False
learning_type = "semi-supervised"

# create target directory
dataset_subfolder = Path(input_type) / f"{dataset_collection_name}-{learning_type}"
target_subfolder = target_folder / dataset_subfolder
target_subfolder.mkdir(parents=True, exist_ok=True)
print(f"Created directories {target_subfolder}")


def process_dataset(dm: DatasetManager, dataset_name: str, split_at: str, resampling_rule=pd.Timedelta(seconds=18)):

    labels_df = pd.read_csv(os.path.join(source_folder, "labels.csv"), parse_dates=["StartTime", "EndTime"], date_parser=lambda x: parse_date(x, ignoretz=True))
    anomaly_types_df = pd.read_csv(os.path.join(source_folder, "anomaly_types.csv"))
    telecommands_df = pd.read_csv(os.path.join(source_folder, "telecommands.csv"))
    telecommands_min_priority = 3

    extension = ".zip"
    all_parameter_names = sorted([
        os.path.basename(file)[: -len(extension)]
        for file in glob(os.path.join(source_folder, "channels", f"*{extension}"))
    ])

    telecommands_df = telecommands_df.loc[telecommands_df["Priority"] >= telecommands_min_priority]
    all_telecommands_names = sorted(telecommands_df.Telecommand.to_list())

    is_anomaly_columns = [f"is_anomaly_{param}" for param in all_parameter_names + all_telecommands_names]
    train_test_paths = {"train": None, "test": None}

    target_meta_filepath = target_subfolder / f"{dataset_name}.{Datasets.METADATA_FILENAME_SUFFIX}"

    for train_test_type in train_test_paths.keys():
        if train_test_type == "test":
            train_test_name = "21_months"
        else:
            train_test_name = dataset_name

        processed_filename = f"{train_test_name}.{train_test_type}.csv"
        train_test_paths[train_test_type] = str(dataset_subfolder / processed_filename).replace(os.sep, '/')
        target_filepath = target_subfolder / processed_filename

        # Prepare datasets
        if not target_filepath.exists() or not target_meta_filepath.exists():
            params_dict = {}

            for param in tqdm(all_parameter_names):
                print(param)
                param_df = pd.read_pickle(os.path.join(source_folder, "channels", f"{param}{extension}"))
                param_df["label"] = np.uint8(0)
                param_df = param_df.rename(columns={param: "value"})

                # Take derivative of monotonic channels - part of preprocessing
                if 29 <= int(param.split("_")[1]) <= 46:
                    param_df.value = np.diff(param_df.value, append=param_df.value[-1])

                # change string values to categorical integers
                if param_df["value"].dtype == "O":
                    print(f"{param} is not numeric!")
                    param_df["value"] = pd.factorize(param_df["value"])[0]

                # Fill labels
                is_param_annotated = False
                for _, row in labels_df.iterrows():
                    if row["Channel"] == param:
                        anomaly_type = anomaly_types_df.loc[anomaly_types_df["ID"] == row["ID"]]["Category"].values[0]
                        if anomaly_type == "Anomaly":
                            label_value = AnnotationLabel.ANOMALY.value
                        elif anomaly_type == "Rare Event":
                            label_value = AnnotationLabel.RARE_EVENT.value
                        param_df.loc[row["StartTime"]:row["EndTime"], "label"] = label_value
                        is_param_annotated = True

                if split_at is not None:
                    if train_test_type == "train":
                        param_df = param_df[param_df.index <= parse_date(split_at)].copy()
                    else:
                        param_df = param_df[param_df.index > parse_date(test_data_split)].copy()

                if len(param_df) == 0:
                    params_dict[param] = []
                    continue

                # Resample using zero order hold
                first_index_resampled = pd.Timestamp(param_df.index[0]).floor(freq=resampling_rule)
                last_index_resampled = pd.Timestamp(param_df.index[-1]).ceil(freq=resampling_rule)
                resampled_range = pd.date_range(first_index_resampled, last_index_resampled, freq=resampling_rule)
                params_dict[param] = param_df.reindex(resampled_range, method="ffill")
                params_dict[param].iloc[0] = param_df.iloc[0]  # Initialize the first sample

                # Restore annotated samples if not present in the resampled series
                if is_param_annotated:
                    grouper = param_df.groupby(pd.Grouper(freq=resampling_rule))
                    for timestamp, group in grouper.indices.items():
                        if len(group) <= 1:
                            continue
                        org_elements = param_df.iloc[group]
                        if org_elements.label.values[-1] != AnnotationLabel.NOMINAL.value:
                            continue
                        is_annotated = (org_elements.label > 0)
                        if is_annotated.any():
                            print(timestamp, org_elements[is_annotated].iloc[-1])
                            params_dict[param].loc[timestamp + pd.Timedelta(resampling_rule)] = org_elements[is_annotated].iloc[-1]

            for param in tqdm(all_telecommands_names):
                print(param)
                param_df = pd.read_pickle(os.path.join(source_folder, "telecommands", f"{param}{extension}"))
                param_df["label"] = 0
                param_df = param_df.rename(columns={param: "value"})

                param_df.index = pd.to_datetime(param_df.index)
                param_df = param_df[~param_df.index.duplicated()]
                param_df = encode_telecommands(param_df, resampling_rule)

                if split_at is not None:
                    if train_test_type == "train":
                        param_df = param_df[param_df.index <= parse_date(split_at)].copy()
                    else:
                        param_df = param_df[param_df.index > parse_date(test_data_split)].copy()

                if len(param_df) == 0:
                    params_dict[param] = []
                    continue

                # Resample using zero order hold
                first_index_resampled = pd.Timestamp(param_df.index[0]).floor(freq=resampling_rule)
                last_index_resampled = pd.Timestamp(param_df.index[-1]).ceil(freq=resampling_rule)
                resampled_range = pd.date_range(first_index_resampled, last_index_resampled, freq=resampling_rule)
                params_dict[param] = param_df.reindex(resampled_range, method="ffill")
                params_dict[param].iloc[0] = param_df.iloc[0]

            # Initialize dataframe
            start_time, end_time = find_full_time_range(params_dict)
            full_index = pd.date_range(start_time, end_time, freq=resampling_rule)
            data_df = pd.DataFrame(index=full_index)

            all_params = list(params_dict.keys())
            for param in all_params:
                df = params_dict.pop(param)
                if len(df) == 0:
                    data_df[param] = np.uint8(0)
                    data_df[f"is_anomaly_{param}"] = np.uint8(0)
                    continue
                df = df.rename(columns={"value": param, "label": f"is_anomaly_{param}"})
                data_df[df.columns] = df.reindex(data_df.index)
                data_df[param] = data_df[param].astype(np.float64).ffill().bfill()
                data_df[f"is_anomaly_{param}"] = data_df[f"is_anomaly_{param}"].ffill().bfill().astype(np.uint8)

            new_columns_order = [*all_parameter_names, *all_telecommands_names, *is_anomaly_columns]
            data_df = data_df[new_columns_order]
            data_df.insert(0, "timestamp", data_df.index.strftime('%Y-%m-%d %H:%M:%S'))

            data_df.to_csv(target_filepath, index=False, lineterminator='\n')
            print(f"  written dataset {train_test_name}")
        else:
            data_df = None
            print(f"  skipped writing dataset {train_test_name} to disk, because it already exists.")

        if train_test_type == "train":
            # Prepare metadata
            def analyze(df_test):
                da = DatasetAnalyzer((dataset_collection_name, dataset_name), is_train=True,
                                     df=df_test, ignore_stationarity=True, ignore_trend=True)
                da.save_to_json(target_meta_filepath, overwrite=(train_test_type == "train"))
                print(f"  analyzed dataset {dataset_name}")
                return da.metadata

            if target_meta_filepath.exists():
                try:
                    meta = DatasetAnalyzer.load_from_json(target_meta_filepath, train=True)
                    for channel, ano_len in meta.anomaly_length.items():  # dict to AnomalyLength
                        meta.anomaly_length[channel] = AnomalyLength(**ano_len)
                    print(f"  skipped analyzing dataset {dataset_name}, because metadata already exists.")
                except ValueError:
                    if data_df is None:
                        data_df = pd.read_csv(target_filepath)
                    meta = analyze(data_df)
            else:
                meta = analyze(data_df)

    dm.add_dataset(DatasetRecord(
          collection_name=dataset_collection_name,
          dataset_name=dataset_name,
          train_path=train_test_paths["train"],
          test_path=train_test_paths["test"],
          dataset_type=dataset_type,
          datetime_index=datetime_index,
          split_at=split_at,
          train_type=learning_type,
          train_is_normal=train_is_normal,
          input_type=input_type,
          length=meta.length,
          dimensions=meta.dimensions,
          contamination=statistics.mean([m for m in meta.contamination.values()]),
          num_anomalies=statistics.mean([m for m in meta.num_anomalies.values()]),
          min_anomaly_length=min([m.min for m in meta.anomaly_length.values()]),
          median_anomaly_length=statistics.median([m.median for m in meta.anomaly_length.values()]),
          max_anomaly_length=max([m.max for m in meta.anomaly_length.values()]),
          mean=meta.mean,
          stddev=meta.stddev,
          trend=meta.trend,
          stationarity=meta.get_stationarity_name(),
          period_size=0
    ))
    print(f"... processed source dataset: {dataset_name}")


dm = DatasetManager(target_folder, create_if_missing=True)
for name, split in dataset_splits.items():
    process_dataset(dm, name, split)
dm.save()

