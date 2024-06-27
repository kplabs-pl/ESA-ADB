import argparse
import os
import pandas as pd
import numpy as np

from natsort import natsorted, natsort_keygen
from dateutil.parser import parse as parse_date
from glob import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer anomaly types from annotations."
    )
    parser.add_argument(
        "--input_path",
        "-i",
        type=str,
        required=True,
        help="Path to a folder with dataset. It must contain anomaly_types.csv, channels.csv, labels.csv and 'channels' folder.",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default=".",
        help="Output folder for modified anomaly_types.csv.",
    )
    parser.add_argument(
        "--point_anomaly_threshold",
        "-t",
        type=float,
        default=30,
        help="Maximal length of point anomaly in seconds.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input_path
    output_path = args.output_path
    point_anomaly_threshold = args.point_anomaly_threshold

    anomaly_types_csv = pd.read_csv(os.path.join(input_path, "anomaly_types.csv"))
    anomaly_types_csv.sort_values(by=["ID"], key=natsort_keygen(), inplace=True)

    labels_df = pd.read_csv(os.path.join(input_path, "labels.csv"))
    all_anomaly_labels = labels_df["ID"].astype(str).unique().tolist()

    channels_csv = pd.read_csv(os.path.join(input_path, "channels.csv"))
    channels_csv = channels_csv.set_index("Channel")

    extension = ".zip"
    all_channel_names = [
        os.path.basename(file)[: -len(extension)]
        for file in glob(os.path.join(input_path, "channels", f"*{extension}"))
    ]

    print("Loading files with channels to extract...")
    channels_df = dict()
    for channel in tqdm(all_channel_names):
        if channels_csv.loc[channel]["Target"] == "NO":
            continue
        df = pd.read_pickle(
            os.path.join(input_path, "channels", f"{channel}{extension}")
        )
        df = df.rename(columns={channel: "value"})
        # change string values to categorical integers
        if df["value"].dtype == "O":
            df["value"] = pd.factorize(df["value"])[0]
        df["label"] = np.uint8(0)
        channels_df[channel] = df
    print("Data for channels loaded")

    print("Loading annotations...")
    for _, row in labels_df.iterrows():
        channel = str(row["Channel"])
        if channel in channels_df:
            anomaly_start = parse_date(row["StartTime"], ignoretz=True)
            anomaly_end = parse_date(row["EndTime"], ignoretz=True)
            channels_df[channel].loc[anomaly_start:anomaly_end, "label"] = 1
    print("Annotations loaded")

    print("Calculating statistics...")
    channels_stats = dict()
    for channel in channels_df:
        channel_data = channels_df[channel].copy()
        nominal_data = channel_data.loc[channel_data["label"] == 0]["value"].to_numpy()
        if len(nominal_data) == 0:
            nominal_data = channel_data["value"].to_numpy()
        channels_stats[channel] = {
            "min": np.min(nominal_data),
            "max": np.max(nominal_data),
        }

    anomaly_types_dict = dict()
    for anomaly in all_anomaly_labels:
        anomaly_types_dict[anomaly] = ["", "", ""]
        filter_categories = anomaly_types_csv.loc[
            (anomaly_types_csv["ID"] == anomaly)
            & (anomaly_types_csv["Category"].isin(["Anomaly", "Rare Event"]))
        ]
        if len(filter_categories) == 0:
            continue

        mask = labels_df["ID"] == anomaly
        if len(labels_df[mask]) == 0:
            continue

        affected_channels = labels_df[mask]["Channel"].to_list()
        affected_channels = [ch for ch in affected_channels if ch in channels_df]  # just in case, e.g., when manually changing all_channel_names
        if len(affected_channels) == 0:
            continue
        print(f"Process anomaly {anomaly}")

        anomaly_starts = [
            parse_date(d, ignoretz=True) for d in labels_df[mask]["StartTime"]
        ]
        anomaly_ends = [
            parse_date(d, ignoretz=True) for d in labels_df[mask]["EndTime"]
        ]

        if len(set(affected_channels)) == 1:
            dimensionality = "Univariate"
        else:
            dimensionality = "Multivariate"

        anomaly_length = "Point"
        locality = "Local"
        for channel in affected_channels:
            for start, end, ap in zip(anomaly_starts, anomaly_ends, affected_channels):
                if ap != channel:
                    continue
                fragment = channels_df[channel].loc[start:end, "value"].to_numpy()
                if len(fragment) > 1 and (end - start) > pd.Timedelta(
                    seconds=point_anomaly_threshold
                ):
                    anomaly_length = "Subsequence"
                if np.any(fragment < channels_stats[channel]["min"]) or np.any(
                    fragment > channels_stats[channel]["max"]
                ):
                    locality = "Global"
            if anomaly_length == "Subsequence" and locality == "Global":
                break

        anomaly_types_dict[anomaly] = [dimensionality, locality, anomaly_length]
        print(anomaly_types_dict[anomaly])

    types_df = pd.DataFrame.from_dict(
        anomaly_types_dict,
        orient="index",
        columns=["Dimensionality", "Locality", "Length"],
    )
    types_df = types_df.reindex(index=natsorted(types_df.index))

    anomaly_types_csv[["Dimensionality", "Locality", "Length"]] = types_df.to_numpy()
    anomaly_types_csv.to_csv(
        os.path.join(output_path, "anomaly_types.csv"), index=False
    )


if __name__ == "__main__":
    main()
