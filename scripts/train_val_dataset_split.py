import os
import argparse
import statistics

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd

from dateutil.parser import parse as parse_date

from timeeval import Datasets, DatasetManager
from timeeval.datasets import DatasetAnalyzer, DatasetRecord, AnomalyLength


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "collection",
        type=str,
        choices=["ESA-Mission1", "ESA-Mission2"],
        help="Datataset specification.",
    )
    parser.add_argument(
        "--index_filename",
        type=str,
        default="datasets.csv",
        help="Persist train-val splits. If not specified, overwrites the default 'datasets.csv'.",
    )
    parser.add_argument(
        "--split_suffix",
        type=str,
        default="split",
        help="Suffix to append to the default train-test dataset name to form a dataset name for train-val splits.",
    )
    return parser.parse_args()


def analyze_dataset(
    dataset_collection_name: str,
    dataset_name: str,
    data_df: pd.DataFrame,
    target_filepath: str,
    target_meta_filepath: str,
    dataset_split: str = "",
):

    # Prepare metadata
    def analyze(df):
        da = DatasetAnalyzer(
            (dataset_collection_name, dataset_name),
            is_train=True,
            df=df,
            ignore_stationarity=True,
            ignore_trend=True,
        )
        da.save_to_json(target_meta_filepath, overwrite=True)
        print(
            f"Analyzed {dataset_split} dataset {dataset_name}, saved at {target_meta_filepath}."
        )
        return da.metadata

    if os.path.exists(target_meta_filepath):
        try:
            meta = DatasetAnalyzer.load_from_json(target_meta_filepath, train=True)
            for (
                channel,
                ano_len,
            ) in meta.anomaly_length.items():  # dict to AnomalyLength
                meta.anomaly_length[channel] = AnomalyLength(**ano_len)
            print(
                f"Skipped analyzing {dataset_split} dataset {dataset_name} because metadata already exists."
            )
        except ValueError:
            if data_df is None:
                data_df = pd.read_csv(target_filepath)
            meta = analyze(data_df)
    else:
        meta = analyze(data_df)

    return meta


esa_mission_1_validation_splits = {
    "3_months": "2000-03-11",
    "10_months": "2000-09-01",
    "21_months": "2001-07-01",
    "42_months": "2003-04-01",
    "84_months": "2006-10-01",
}

esa_mission_2_validation_splits = {
    "1_months": "2000-01-24",
    "5_months": "2000-05-01",
    "10_months": "2000-09-01",
    "21_months": "2001-07-01",
}

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_raw_folder = os.path.abspath(os.path.join(base_dir, "data"))
    data_processed_folder = os.path.abspath(
        os.path.join(data_raw_folder, "preprocessed")
    )

    assert os.path.exists(
        data_processed_folder
    ), f"Folder {data_processed_folder} does not exist."

    args = parse_args()
    collection = args.collection
    index_filename = args.index_filename
    split_suffix = args.split_suffix

    # set correct validation splits
    validations_splits = (
        esa_mission_1_validation_splits
        if collection == "ESA-Mission1"
        else esa_mission_2_validation_splits
    )

    # set dataset features
    dataset_type = "real"
    input_type = "multivariate"
    datetime_index = True
    train_is_normal = False
    learning_type = "semi-supervised"

    dataset_subfolder = os.path.join(input_type, f"{collection}-{learning_type}")
    target_subfolder = os.path.abspath(
        os.path.join(data_processed_folder, dataset_subfolder)
    )
    assert os.path.exists(target_subfolder), f"{target_subfolder} does not exist."

    print(
        f"Initializing DatasetManager from {data_processed_folder} for {collection}..."
    )
    dm = DatasetManager(data_processed_folder)
    datasets = dm.select(collection=collection)
    if len(datasets) == 0:
        print(
            f"No datasets recorded in {data_processed_folder}/datasets.csv, exiting..."
        )

    for ix, (dataset_collection_name, dataset_name) in enumerate(datasets):
        # skip dataset names not corresponding to collections
        if dataset_name not in validations_splits:
            print(
                f"Skipped {dataset_name} in 'datasets.csv' - not listed for validation split."
            )
            continue

        data_df = dm.get_dataset_df((dataset_collection_name, dataset_name), train=True)
        assert (
            data_df is not None and not data_df.empty
        ), f"Invalid training dataset {dataset_name}. Verify correctness of 'datasets.csv' under {data_processed_folder}."

        val_split_at = parse_date(validations_splits[dataset_name])
        assert (
            val_split_at >= data_df["timestamp"].iloc[0]
            and val_split_at <= data_df["timestamp"].iloc[-1]
        ), f"Timestamp {val_split_at} out of range on training dataset {dataset_name}."

        # train-val splits
        df_train = data_df[data_df["timestamp"] <= val_split_at]
        df_val = data_df[data_df["timestamp"] > val_split_at]

        # book-keeping for adding datasets via DatasetManager
        metas = {}
        target_filepaths = {}

        for dataset_split in ["train", "val"]:
            df_split = df_train if dataset_split == "train" else df_val

            # train-val csv filepaths
            target_filepath = os.path.abspath(
                os.path.join(
                    target_subfolder,
                    f"{dataset_name}.{dataset_split}.{split_suffix}.csv",
                )
            )
            target_filepaths[dataset_split] = target_filepath
            df_split.to_csv(target_filepath, index=False, lineterminator="\n")

            # train-val metadata.json filepaths
            target_meta_filepath = os.path.abspath(
                os.path.join(
                    target_subfolder,
                    f"{dataset_name}.{dataset_split}.{split_suffix}.{Datasets.METADATA_FILENAME_SUFFIX}",
                )
            )

            # differentiate between train-val and default train-test splits
            split_dataset_name = "_".join([dataset_name, split_suffix])

            meta = analyze_dataset(
                dataset_collection_name,
                split_dataset_name,
                df_split,
                target_filepath,
                target_meta_filepath,
                dataset_split,
            )
            metas[dataset_split] = meta

        meta = metas["train"]

        # (train, val)-splits as (train, test)-Dataset
        datasets_df = pd.DataFrame.from_records(
            [
                {
                    "collection_name": dataset_collection_name,
                    "dataset_name": split_dataset_name,
                    "train_path": target_filepaths["train"],
                    "test_path": target_filepaths["val"],
                    "dataset_type": dataset_type,
                    "datetime_index": datetime_index,
                    "split_at": validations_splits[dataset_name],
                    "train_type": learning_type,
                    "train_is_normal": train_is_normal,
                    "input_type": input_type,
                    "length": meta.length,
                    "dimensions": meta.dimensions,
                    "contamination": statistics.mean(
                        [m for m in meta.contamination.values()]
                    ),
                    "num_anomalies": statistics.mean(
                        [m for m in meta.num_anomalies.values()]
                    ),
                    "min_anomaly_length": min(
                        [m.min for m in meta.anomaly_length.values()]
                    ),
                    "median_anomaly_length": statistics.median(
                        [m.median for m in meta.anomaly_length.values()]
                    ),
                    "max_anomaly_length": max(
                        [m.max for m in meta.anomaly_length.values()]
                    ),
                    "mean": meta.mean,
                    "stddev": meta.stddev,
                    "trend": meta.trend,
                    "stationarity": meta.get_stationarity_name(),
                    "period_size": 0,
                }
            ],
            index=["collection_name", "dataset_name"],
        )

        datasets_df = datasets_df[~datasets_df.index.duplicated(keep="last")]
        datasets_df = datasets_df.sort_index()

        target_index_filepath = os.path.abspath(
            os.path.join(data_processed_folder, index_filename)
        )
        datasets_df.to_csv(
            target_index_filepath, mode="w" if (ix == 0) else "a", header=(ix == 0)
        )
        print(
            f"Processed source dataset: {split_dataset_name}, written to {target_index_filepath}"
        )
