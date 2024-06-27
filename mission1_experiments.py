#!/usr/bin/env python3
from pathlib import Path

from timeeval import TimeEval, DatasetManager, ResourceConstraints
from timeeval.metrics import ESAScores, ChannelAwareFScore, ADTQC
from timeeval.params import FixedParameters, FullParameterGrid

from timeeval_experiments.algorithms import *
from durations import Duration

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

data_raw_folder = os.path.abspath(os.path.join(current_dir, "data"))
data_processed_folder = os.path.abspath(os.path.join(data_raw_folder, "preprocessed"))


def main():
    dm = DatasetManager(data_processed_folder)  # or test-cases directory
    collection = "ESA-Mission1"
    datasets = dm.select(collection=collection)

    target_channels = ["channel_12", "channel_13", "channel_14", "channel_15", "channel_16", "channel_17", "channel_18", "channel_19", "channel_20", "channel_21", "channel_22", "channel_23", "channel_24", "channel_25", "channel_26", "channel_27", "channel_28", "channel_29", "channel_30", "channel_31", "channel_32", "channel_33", "channel_34", "channel_35", "channel_36", "channel_37", "channel_38", "channel_39", "channel_40", "channel_41", "channel_42", "channel_43", "channel_44", "channel_45", "channel_46", "channel_47", "channel_48", "channel_49", "channel_50", "channel_51", "channel_52", "channel_57", "channel_58", "channel_59", "channel_60", "channel_61", "channel_62", "channel_63", "channel_64", "channel_65", "channel_66", "channel_70", "channel_71", "channel_72", "channel_73", "channel_74", "channel_75", "channel_76"]
    subset_channels = ["channel_41", "channel_42", "channel_43", "channel_44", "channel_45", "channel_46"]

    limits = ResourceConstraints(
        train_timeout=Duration("120h"),
        execute_timeout=Duration("120h")
    )

    beta = 0.5
    metrics = [ESAScores(betas=beta, select_labels={"Category": ["Rare Event", "Anomaly"]}),
               ESAScores(betas=beta, select_labels={"Category": ["Anomaly"]})]
    ranking_metrics = [ChannelAwareFScore(beta=beta, select_labels={"Category": ["Rare Event", "Anomaly"]}),
                       ChannelAwareFScore(beta=beta, select_labels={"Category": ["Anomaly"]}),
                       ADTQC(select_labels={"Category": ["Rare Event", "Anomaly"]}),
                       ADTQC(select_labels={"Category": ["Anomaly"]})]

    labels_csv = Path(os.path.join(data_raw_folder, collection, "labels.csv"))
    test_dataset_path = Path(os.path.join(data_processed_folder, "multivariate", f"{collection}-semi-supervised", "84_months.test.csv"))

    algorithms = [
        pcc(params=FullParameterGrid({"target_channels": [subset_channels, target_channels]})),
        hbos(params=FullParameterGrid({"target_channels": [subset_channels, target_channels], "n_bins": [50]})),
        std(params=FullParameterGrid({"target_channels": [subset_channels, target_channels], "tol": [3, 5]})),
        iforest(params=FullParameterGrid({"target_channels": [subset_channels, target_channels]})),
        subsequence_if(params=FullParameterGrid({"target_channels": [subset_channels, target_channels], "n_trees": [200], "window_size": [17]})),
        knn(params=FullParameterGrid({"target_channels": [subset_channels, target_channels]}))
    ]

    timeeval = TimeEval(dm, datasets, algorithms,
                        metrics=metrics,
                        ranking_metrics=ranking_metrics,
                        resource_constraints=limits,
                        labels_csv_path=labels_csv,
                        test_dataset_path=test_dataset_path)

    timeeval.run()
    results = timeeval.get_results(aggregated=False)
    print(results)

    # Deep learning-based algorithms
    validation_splits = {"3_months": "2000-03-11",
                         "10_months": "2000-09-01",
                         "21_months": "2001-07-01",
                         "42_months": "2003-04-01",
                         "84_months": "2006-10-01"}

    for dataset, split in validation_splits.items():
        datasets = dm.select(collection=collection, dataset=dataset)
        dc_vae_units = [32 if dataset != "3_months" else 16] * 6
        algorithms = [
            dc_vae(params=FixedParameters({"epochs": 1000, "alpha": None, "J": len(subset_channels) // 3, "T": 256, "cnn_units": dc_vae_units, "batch_size": 64, "validation_date_split": split, "input_channels": subset_channels, "target_channels": subset_channels})),
            dc_vae(params=FixedParameters({"epochs": 1000, "alpha": None, "J": 87 // 3, "T": 256, "cnn_units": dc_vae_units, "batch_size": 64, "validation_date_split": split, "target_channels": target_channels})),
            telemanom_esa(params=FixedParameters({"epochs": 1000, "early_stopping_patience": 20, "layers": [80, 80], "validation_date_split": split, "input_channels": subset_channels, "target_channels": subset_channels, "threshold_scores": 1, "min_error_value": 0})),
            telemanom_esa(params=FixedParameters({"epochs": 1000, "early_stopping_patience": 20, "layers": [134, 134], "validation_date_split": split, "target_channels": target_channels, "threshold_scores": 1, "min_error_value": 0})),
        ]
        timeeval = TimeEval(dm, datasets, algorithms,
                            metrics=metrics,
                            ranking_metrics=ranking_metrics,
                            resource_constraints=limits,
                            labels_csv_path=labels_csv,
                            test_dataset_path=test_dataset_path)

        timeeval.run()
        results = timeeval.get_results(aggregated=False)
        print(results)


if __name__ == "__main__":
    main()
