"""
Script for preparing fragments for OXI annotator oxi.kplabs.pl
"""
import argparse
import fnmatch
import os
import pandas as pd

from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from glob import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract specific fragments from mission for OXI anomaly annotator."
    )
    parser.add_argument(
        "--input_path",
        "-i",
        type=str,
        required=True,
        help="Path to a folder with dataset. It must contain labels.csv file and 'channels' folder.",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="fragments",
        help="Output folder for fragments.",
    )
    parser.add_argument(
        "--channels_to_extract",
        "-c",
        type=str,
        nargs="+",
        default="*",
        help="A list of channels to include in the fragments. It is possible to use an asterisk * as a wildcard "
        "to select a group of channels. "
        "In default, all channels will be included.",
    )
    parser.add_argument(
        "--telecommands",
        "-t",
        type=str,
        nargs="+",
        default=[],
        help="A list of telecommands to include in the fragments. It is possible to use an asterisk * as a wildcard "
        "to select a group of channels. "
        "In default, no telecommands will be included.",
    )
    parser.add_argument(
        "--anomalies_to_extract",
        "-a",
        type=str,
        nargs="+",
        default="*",
        help="A list of anomaly IDs around which the fragments should be extracted. It is possible to use an asterisk "
        "* as a wildcard to select a group of anomaly IDs."
        "In default, all anomalies will be included.",
    )
    parser.add_argument(
        "--dates_to_extract",
        "-d",
        type=str,
        nargs="+",
        default=[],
        help="A list of dates in YYYY-MM-DD format around which the fragments should be extracted. "
        "They will be generated independently from --anomalies_to_extract argument.",
    )
    parser.add_argument(
        "--days_before",
        "-db",
        type=float,
        default=1,
        help="Number of days before anomaly to include in the fragment.",
    )
    parser.add_argument(
        "--days_after",
        "-da",
        type=float,
        default=1,
        help="Number of days after anomaly to include in the fragment.",
    )
    return parser.parse_args()


class TextColors:
    RED = "\033[91m"
    ENDC = "\033[0m"


def to_iso_date(date: datetime):
    return date.replace(tzinfo=None).isoformat(timespec="milliseconds") + "Z"


def select_options_by_patterns(patterns: list, all_options: list) -> list:
    parsed_options = []
    for pattern in patterns:
        matching_options = [opt for opt in fnmatch.filter(all_options, pattern)]
        if len(matching_options) == 0:
            print(f"No options matching the pattern: {pattern}")
        else:
            parsed_options.extend(matching_options)

    return list(set(parsed_options))


def parse_dates_to_extract(dates_to_extract: list) -> list:
    parsed_dates = []
    for date_str in dates_to_extract:
        try:
            date_object = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            print(f"Impossible to cast '{date_str}' as YYYY-MM-DD date")
            continue

        parsed_dates.append(date_object)

    return list(set(parsed_dates))


def main():
    args = parse_args()

    input_path = args.input_path
    output_path = args.output_path
    days_before = timedelta(days=args.days_before)
    days_after = timedelta(days=args.days_after)

    labels_df = pd.read_csv(os.path.join(input_path, "labels.csv"))
    all_anomaly_labels = labels_df["ID"].astype(str).unique().tolist()

    extension = ".zip"
    all_channel_names = [
        os.path.basename(file)[: -len(extension)]
        for file in glob(os.path.join(input_path, "channels", f"*{extension}"))
    ]

    anomalies_to_extract = select_options_by_patterns(
        args.anomalies_to_extract, all_anomaly_labels
    )
    channels_to_extract = select_options_by_patterns(
        args.channels_to_extract, all_channel_names
    )

    if len(channels_to_extract) == 0:
        raise ValueError("No channels to extract")

    all_telecommands_names = [
        os.path.basename(file)[: -len(extension)]
        for file in glob(os.path.join(input_path, "telecommands", f"*{extension}"))
    ]

    telecommands_to_extract = select_options_by_patterns(
        args.telecommands, all_telecommands_names
    )

    dates_to_extract = parse_dates_to_extract(args.dates_to_extract)

    print("Loading files with channels to extract...")
    channels_df = dict()
    for channel in tqdm(sorted(channels_to_extract)):
        df = pd.read_pickle(
            os.path.join(input_path, "channels", f"{channel}{extension}")
        )
        df = df.rename(columns={channel: "value"})
        # change string values to categorical integers
        if df["value"].dtype == "O":
            df["value"] = pd.factorize(df["value"])[0]
        df["label"] = ""
        channels_df[channel] = df
    print("Data for channels loaded")

    print("Loading annotations...")
    for _, row in labels_df.iterrows():
        channel = str(row["Channel"])
        if channel in channels_to_extract:
            anomaly_start = parse_date(row["StartTime"], ignoretz=True)
            anomaly_end = parse_date(row["EndTime"], ignoretz=True)
            channels_df[channel].loc[anomaly_start:anomaly_end, "label"] = row["ID"]
    print("Annotations loaded")

    num_executions = []
    if len(telecommands_to_extract) > 0:
        print("Loading files with telecommands to extract...")
        d_t = pd.Timedelta(milliseconds=1)  # impulse width
        for telecommand in tqdm(telecommands_to_extract):
            df = pd.read_pickle(
                os.path.join(input_path, "telecommands", f"{telecommand}{extension}")
            )
            # Encode telecommands as impulses
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated()]
            num_executions.append(len(df.index))
            extended_timestamps = []
            for t in df.index:
                extended_timestamps.extend([t - d_t, t, t + d_t])
            extended_timestamps = sorted(list(set(extended_timestamps)))
            df = df.reindex(extended_timestamps, fill_value=0)
            df = df.rename(columns={telecommand: "value"})
            df["label"] = ""
            channels_df[telecommand] = df
        print(
            f"Data for {len(num_executions)} telecommands loaded with {sum(num_executions)} executions"
        )

    # select relevant labels from annotations
    if len(anomalies_to_extract) > 0:
        labels_df = labels_df[labels_df["Channel"].isin(channels_to_extract)]
        labels_df = labels_df[labels_df["ID"].isin(anomalies_to_extract)]
    else:
        labels_df = labels_df.iloc[0:0]

    for date in dates_to_extract:
        labels_df.loc[
            0 if pd.isnull(labels_df.index.max()) else labels_df.index.max() + 1
        ] = date.isoformat()
        all_anomaly_labels.append(date.isoformat())

    if len(labels_df) == 0:
        print("No fragments meet the criteria")
        return

    os.makedirs(output_path, exist_ok=True)

    for anomaly in all_anomaly_labels:
        mask = labels_df["ID"] == anomaly
        if len(labels_df[mask]) == 0:
            continue
        print(f"Process anomaly {anomaly}")

        anomaly_starts = [
            parse_date(d, ignoretz=True) for d in labels_df[mask]["StartTime"]
        ]
        anomaly_ends = [
            parse_date(d, ignoretz=True) for d in labels_df[mask]["EndTime"]
        ]
        affected_channels = labels_df[mask]["Channel"].tolist()

        fragment_start = min(anomaly_starts) - days_before
        fragment_end = max(anomaly_ends) + days_after

        fragment_df = pd.DataFrame(columns=["channel", "timestamp", "value", "label"])
        for channel, df in channels_df.items():
            channel_df = df.loc[fragment_start:fragment_end].copy()
            channel_df["channel"] = channel
            channel_df["timestamp"] = channel_df.index.map(to_iso_date)

            # overwrite any other anomalies with the dominant anomaly
            for start, end, ap in zip(anomaly_starts, anomaly_ends, affected_channels):
                if ap == channel:
                    channel_df.loc[start:end, "label"] = anomaly

            fragment_df = pd.concat(
                [fragment_df, channel_df], axis=0, ignore_index=True
            )

        fragment_id = f"{anomaly}_{fragment_start.strftime('%Y-%m-%d')}_{fragment_end.strftime('%Y-%m-%d')}"
        if len(fragment_df) == 0:
            print(f"Empty fragment {fragment_id}")
            continue

        fragment_file = os.path.join(output_path, f"fragment_{fragment_id}.csv")
        fragment_df.to_csv(fragment_file, index=False)
        print(f"Fragment saved: {fragment_file}")


if __name__ == "__main__":
    main()
