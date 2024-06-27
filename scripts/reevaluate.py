import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from timeeval import Algorithm, Status, MultiChannelMetric, Metric, DefaultMetrics, MultiDatasetManager
from timeeval.adapters.docker import SCORES_FILE_NAME as DOCKER_SCORES_FILE_NAME
from timeeval.constants import DEVIATION_CSV, RESULTS_CSV, HYPER_PARAMETERS, METRICS_CSV, ANOMALY_SCORES_TS, RECONSTRUCTION_CSV
from timeeval.data_types import ExecutionType
from timeeval.core.experiments import Experiment as TimeEvalExperiment
from timeeval.metrics import FScoreAtK, PrecisionAtK, ESAScores, \
    ChannelAwareFScore, ADTQC, DcVaeThresholding, DcVaeAnomalyScoring, TelemanomThresholding, ThresholdingStrategy
from timeeval.utils.datasets import load_labels_only

# required to build a lookup-table for algorithm implementations
import timeeval_experiments.algorithms as algorithms
# noinspection PyUnresolvedReferences
from timeeval_experiments.algorithms import *
from timeeval_experiments.baselines import Baselines

REEVALUATE_LOG = "reevaluate.log"
INITIAL_WAITING_SECONDS = 0


def path_is_empty(path: Path) -> bool:
    return not any(path.iterdir())


class Evaluator:

    def __init__(self, results_path: Path, data_paths: List[Path], metrics: List[Metric],
                 ranking_metrics: List[MultiChannelMetric] = None, labels_csv_path: Path = None,
                 threshold: ThresholdingStrategy = None):
        log_filehandler = logging.FileHandler(results_path / REEVALUATE_LOG, 'w')
        self._logger = logging.getLogger(self.__class__.__name__)
        for hdlr in self._logger.handlers[:]:  # remove the existing file handlers
            if isinstance(hdlr, logging.FileHandler):
                self._logger.removeHandler(hdlr)
        self._logger.addHandler(log_filehandler)
        sys.stdout.write = self._logger.info

        self.results_path = results_path
        self.data_paths = data_paths
        self.metrics = metrics
        self.ranking_metrics = ranking_metrics if ranking_metrics is not None else []
        self.algos = self._build_algorithm_dict()
        self.dmgr = MultiDatasetManager(data_paths)
        self.threshold = threshold

        self.labels_csv_path = labels_csv_path
        self.labels_df = None
        self.test_data = None

        if labels_csv_path is not None:
            anomaly_types_path = str(labels_csv_path).replace("labels.csv", "anomaly_types.csv")
            if not os.path.isfile(labels_csv_path):
                self._logger.warning(f"There is no labels CSV file ({labels_csv_path})! "
                                     "Continuing with standard labels in data")
            elif not os.path.isfile(anomaly_types_path):
                self._logger.warning(f"There is no anomaly types CSV file ({anomaly_types_path})! "
                                     "Continuing with standard labels in data")
            else:
                self.labels_df: pd.DataFrame = pd.read_csv(labels_csv_path, parse_dates=["StartTime", "EndTime"])
                self.labels_df["StartTime"] = self.labels_df["StartTime"].apply(lambda t: t.tz_localize(None))
                self.labels_df["EndTime"] = self.labels_df["EndTime"].apply(lambda t: t.tz_localize(None))

                anomaly_types_df: pd.DataFrame = pd.read_csv(anomaly_types_path)
                columns_to_copy = anomaly_types_df.columns[-4:]
                for col in columns_to_copy:
                    self.labels_df[col] = ""
                for _, row in anomaly_types_df.iterrows():
                    self.labels_df.loc[self.labels_df["ID"] == row["ID"], columns_to_copy] = row[columns_to_copy].values

            channels_path = str(labels_csv_path).replace("labels.csv", "channels.csv")
            if not os.path.isfile(channels_path):
                self.subsystems_mapping = None
                self._logger.warning(f"There is no channels CSV file ({channels_path})! "
                                     "Metrics at the subsystem level won't be calculated")
            else:
                channels_df = pd.read_csv(channels_path)
                self.subsystems_mapping = {s: [*v] for s, v in channels_df.groupby("Subsystem")["Channel"]}

        self.df: pd.DataFrame = pd.read_csv(results_path / RESULTS_CSV)

        self._logger.warning(f"The Evaluator changes the results folder ({self.results_path}) in-place! "
                             "If you do not want this, cancel this script using Ctrl-C! "
                             f"Waiting {INITIAL_WAITING_SECONDS} seconds before continuing ...")
        time.sleep(INITIAL_WAITING_SECONDS)

    @staticmethod
    def _build_algorithm_dict() -> Dict[str, Algorithm]:
        algo_names = [a for a in dir(algorithms) if not a.startswith("__")]
        algo_list: List[Algorithm] = [eval(f"{a}()") for a in algo_names]
        algos: Dict[str, Algorithm] = {}
        for a in algo_list:
            algos[a.name] = a
        # add baselines
        increasing_baseline = Baselines.increasing()
        algos[increasing_baseline.name] = increasing_baseline
        random_baseline = Baselines.random()
        algos[random_baseline.name] = random_baseline
        normal_baseline = Baselines.normal()
        algos[normal_baseline.name] = normal_baseline
        # aliases for some renamed algorithms:
        algos["Image-embedding-CAE"] = algos["ImageEmbeddingCAE"]
        algos["LTSM-VAE"] = algos["LSTM-VAE"]
        return algos

    def evaluate(self, select_index: Optional[Path], evaluate_successful: bool = False):
        if select_index is None:
            exp_indices = self.df.index.values
        else:
            exp_indices = pd.read_csv(select_index).iloc[:, 0]
        self._logger.info(f"Re-evaluating {len(exp_indices)} experiments from {len(self.df)} experiments of "
                          f"folder {self.results_path}")
        for i in exp_indices:
            s_exp: pd.Series = self.df.iloc[i]
            if not evaluate_successful and s_exp.status == "Status.OK":
                self._logger.info(f"Exp-{i:06d}: Skipping, because experiment was successful.")
                continue

            exp_path = self._exp_path(s_exp)
            self._logger.info(f"Exp-{i:06d}: Starting processing {exp_path}")
            docker_scores_path = exp_path / DOCKER_SCORES_FILE_NAME
            processed_scores_path = exp_path / ANOMALY_SCORES_TS
            params_path = exp_path / HYPER_PARAMETERS
            metrics_path = exp_path / METRICS_CSV
            reconstruction_csv = exp_path / RECONSTRUCTION_CSV
            deviation_csv = exp_path / DEVIATION_CSV
            if not params_path.exists():
                self._logger.error(f"Exp-{i:06d}: Experiment ({s_exp.algorithm}-{s_exp.collection}-{s_exp.dataset}) "
                                   "does not contain any results to start with (scores or hyper params are missing)!")
                continue

            with open(params_path) as json_file:
                params = json.load(json_file)

            if "target_channels" in params:
                channel_names = params["target_channels"]
            else:
                columns = pd.read_csv(self.dmgr.get_dataset_path((s_exp.collection, s_exp.dataset)), index_col=0, nrows=0).columns.tolist()
                anomaly_columns = [x for x in columns if x.startswith("is_anomaly")]
                channel_names = columns[:-len(anomaly_columns)]

            self.test_data = pd.read_csv(self.dmgr.get_dataset_path((s_exp.collection, s_exp.dataset)), usecols=["timestamp", *channel_names], parse_dates=[0])
            self.test_data.rename(columns={"timestamp": "Timestamp"}, inplace=True)

            if self.labels_df is None:
                y_true = load_labels_only(self.dmgr.get_dataset_path((s_exp.collection, s_exp.dataset)), target_channels=channel_names)
                if y_true.ndim == 1:
                    y_true = np.expand_dims(y_true, -1)
            else:
                self.test_data["Score"] = np.uint8(0)

                y_true = self.labels_df[self.labels_df["Channel"].isin(channel_names) &
                                        (self.labels_df["StartTime"] >= self.test_data["Timestamp"].min()) &
                                        (self.labels_df["EndTime"] <= self.test_data["Timestamp"].max())]

            if processed_scores_path.exists():
                self._logger.debug(f"Exp-{i:06d}: Skipping reprocessing of anomaly scores, they are present.")
                if self.labels_df is not None:
                    # Optimization for large files instead of np.genfromtxt
                    y_scores = None
                    with open(processed_scores_path, 'r') as file:
                        for num, line in enumerate(file.readlines()):
                            line = [float(e) for e in line.split(",")]
                            if y_scores is None:
                                y_scores = np.zeros((len(self.test_data), len(line)), dtype=float)
                            y_scores[num] = line
                else:
                    y_scores = np.genfromtxt(processed_scores_path, delimiter=",")
            else:
                self._logger.debug(f"Exp-{i:06d}: Processing anomaly scores.")
                y_scores = np.loadtxt(str(docker_scores_path), delimiter=",")
                post_fn = self.algos[s_exp.algorithm].postprocess
                if post_fn is not None:
                    with params_path.open("r") as fh:
                        hyper_params = json.load(fh)
                    dataset = self.dmgr.get(s_exp.collection, s_exp.dataset)
                    args = {
                        "executionType": ExecutionType.EXECUTE,
                        "results_path": exp_path,
                        "hyper_params": hyper_params,
                        "dataset_details": dataset
                    }
                    y_scores = post_fn(y_scores, args)
                y_scores = TimeEvalExperiment.scale_scores(y_scores)
                self._logger.info(f"Exp-{i:06d}: Writing anomaly scores to {processed_scores_path}.")
                np.savetxt(str(processed_scores_path), y_scores, delimiter=",")

            if y_scores.ndim == 1:
                y_scores = np.expand_dims(y_scores, -1)

            if self.threshold is not None and os.path.isfile(reconstruction_csv):
                reconstruction = pd.read_csv(reconstruction_csv).values
                y_scores = np.zeros_like(y_scores)
                if isinstance(threshold, TelemanomThresholding):
                    for y_channel in range(y_scores.shape[-1]):
                        y_scores[..., y_channel] = self.threshold.find_threshold(self.test_data[channel_names].values[..., y_channel], reconstruction[..., y_channel])
                elif isinstance(threshold, DcVaeThresholding) or isinstance(threshold, DcVaeAnomalyScoring):
                    deviation = pd.read_csv(deviation_csv).values
                    y_scores = self.threshold.find_threshold(self.test_data[channel_names].values, [reconstruction, deviation])
                np.savetxt(str(processed_scores_path), y_scores, delimiter=",")

            metric_scores = {}

            if not evaluate_successful and all(m.name in metric_scores for m in self.metrics) and all(m.name in metric_scores for m in self.ranking_metrics):
                self._logger.debug(f"Exp-{i:06d}: Skipping re-assessment of metrics, they are all present.")
                errors = 0
            else:
                self._logger.debug(f"Exp-{i:06d}: Re-assessing metrics.")
                results = {}
                errors = 0

                only_global_scores = False
                only_global_gt = False
                if y_scores.shape[1] == 1:
                    only_global_scores = True
                    print("Only global scores available")
                if self.labels_df is None and y_true.shape[1] == 1:
                    only_global_gt = True
                    print("Only global GT available")

                # First calculate global metrics
                for metric in self.metrics:
                    if hasattr(metric, '_plot_store'):
                        metric._plot_store = exp_path

                    try:
                        if self.labels_df is not None:
                            self.test_data["Score"] = y_scores.max(axis=1).astype(np.uint8)
                            score = metric.score(y_true.drop(columns=["Channel"]), self.test_data[["Timestamp", "Score"]])
                        else:
                            score = metric(y_true.max(axis=1), y_scores.max(axis=1))

                        if isinstance(score, dict):
                            for submetric, value in score.items():
                                print(f"{metric.name}_{submetric} calculated: {value}")
                                results[f"{metric.name}_{submetric}"] = value
                        else:
                            print(f"{metric.name} calculated: {score}")
                            results[f"{metric.name}"] = score
                    except Exception as e:
                        self._logger.warning(
                            f"Exp-{i:06d}: Exception while computing global metric {metric.name}!", exc_info=e)
                        errors += 1
                        continue

                for metric in self.ranking_metrics:
                    if not isinstance(metric, ADTQC):
                        continue

                    try:
                        y_pred = dict()
                        self.test_data["Score"] = y_scores.max(axis=1).astype(np.uint8)
                        y_pred["global"] = self.test_data[["Timestamp", "Score"]].copy()

                        y_true_global = y_true.copy()
                        y_true_global["Channel"] = "global"
                        score = metric.score(y_true_global, y_pred)
                        for submetric, value in score.items():
                            print(f"Global_{metric.name}_{submetric} calculated: {value}")
                            results[f"Global_{metric.name}_{submetric}"] = value

                    except Exception as e:
                        self._logger.warning(
                            f"Exp-{i:06d}: Exception while computing global metric {metric.name}!", exc_info=e)
                        errors += 1
                        continue

                # Calculate per_channel evaluation if possible
                if not only_global_scores and not only_global_gt:
                    for metric in self.ranking_metrics:
                        if isinstance(metric, ADTQC):
                            continue
                        try:
                            y_pred = dict()
                            for c, channel_name in enumerate(channel_names):
                                self.test_data["Score"] = y_scores[..., c].astype(np.uint8)
                                y_pred[channel_name] = self.test_data[["Timestamp", "Score"]].copy()

                            if isinstance(metric, ChannelAwareFScore):
                                score = metric.score(y_true, y_pred, self.subsystems_mapping)
                            else:
                                score = metric.score(y_true, y_pred)

                            if isinstance(score, dict):
                                for submetric, value in score.items():
                                    print(f"{metric.name}_{submetric} calculated: {value}")
                                    results[f"{metric.name}_{submetric}"] = value
                            else:
                                print(f"{metric.name} calculated: {score}")
                                results[f"{metric.name}"] = score
                        except Exception as e:
                            self._logger.warning(
                                f"Exp-{i:06d}: Exception while computing metric {metric.name}!",
                                exc_info=e)
                            errors += 1
                            continue

                    for metric in self.metrics:
                        for c, channel_name in enumerate(channel_names):
                            try:
                                if self.labels_df is not None:
                                    self.test_data["Score"] = y_scores[..., c].astype(np.uint8)
                                    score = metric.score(y_true[y_true["Channel"] == channel_name].drop(columns=["Channel"]), self.test_data[["Timestamp", "Score"]])
                                else:
                                    score = metric(y_true[..., c], y_scores[..., c])

                                if isinstance(score, dict):
                                    for submetric, value in score.items():
                                        print(f"{metric.name}_{submetric}_{channel_name} calculated: {value}")
                                        results[f"{metric.name}_{submetric}_{channel_name}"] = value
                                else:
                                    print(f"{metric.name}_{channel_name} calculated: {score}")
                                    results[f"{metric.name}_{channel_name}"] = score
                            except Exception as e:
                                self._logger.warning(f"Exp-{i:06d}: Exception while computing metric {metric.name}_{channel_name}!", exc_info=e)
                                errors += 1
                                continue

                # update metrics and write them to disk
                metric_scores.update(results)
                if metric_scores:
                    self._logger.info(f"Exp-{i:06d}: Writing new metrics to {metrics_path}!")
                    pd.DataFrame([metric_scores]).to_csv(metrics_path, index=False)
                else:
                    self._logger.warning(f"Exp-{i:06d}: No metrics computed!")

            if metric_scores and errors == 0:
                self._logger.debug(f"Exp-{i:06d}: Updating status to success (Status.OK).")
                s_update = s_exp.copy()
                s_update["status"] = Status.OK
                s_update["error_message"] = "(fixed)"
                for metric in metric_scores:
                    if metric in s_update:
                        s_update[metric] = metric_scores[metric]
                self.df.iloc[i] = s_update
            self._logger.info(f"Exp-{i:06d}: ... finished processing.")

        self._logger.info(f"Overwriting results file at {self.results_path / RESULTS_CSV}")
        self.df.to_csv(self.results_path / RESULTS_CSV, index=False)

    def _exp_path(self, exp: pd.Series) -> Path:
        return (self.results_path
                / exp.algorithm
                / exp.hyper_params_id
                / exp.collection
                / exp.dataset
                / str(exp.repetition))


_metrics = {
    DefaultMetrics.ROC_AUC.name: DefaultMetrics.ROC_AUC,
    DefaultMetrics.PR_AUC.name: DefaultMetrics.PR_AUC,
    DefaultMetrics.FIXED_RANGE_PR_AUC.name: DefaultMetrics.FIXED_RANGE_PR_AUC,
    DefaultMetrics.AVERAGE_PRECISION.name: DefaultMetrics.AVERAGE_PRECISION,
    PrecisionAtK().name: PrecisionAtK(),
    FScoreAtK().name: FScoreAtK()
}


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Takes an experiment result folder and re-evaluates (standardization and metric calculation) "
                    "selected experiments."
    )
    parser.add_argument("result_folder", type=Path,
                        help="Folder of the experiment")
    parser.add_argument("data_folders", type=Path, nargs="*",
                        help="Folders, where the datasets from the experiment are stored.")
    parser.add_argument("--labels", type=Path, default=None,
                        help="Path to the labels.csv file from ESA Anomaly Benchmark structure.")
    parser.add_argument("--select", type=Path, default=None,
                        help="Experiments to reevaluate (indices to results.csv; single column with header 'index').")
    parser.add_argument("--loglevel", default="DEBUG", choices=("ERROR", "WARNING", "INFO", "DEBUG"),
                        help="Set logging verbosity (default: %(default)s)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Set this flag if successful experiments (Status.OK) should be reevaluated as well")
    parser.add_argument("--metrics", type=str, nargs="*", default=["ROC_AUC", "PR_AUC", "RANGE_PR_AUC"],
                        choices=list(_metrics.keys()),
                        help="Metrics to re-calculate. (default: %(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _create_arg_parser()
    logging.basicConfig(level=args.loglevel)

    #threshold = TelemanomThresholding(min_error_value=0.05/7)
    #threshold = DcVaeThresholding(alpha=5)
    threshold = None

    betas = [0.5]
    selected_metrics = [ESAScores(betas=betas, select_labels={"Category": ["Rare Event", "Anomaly"]}),
                        ESAScores(betas=betas, select_labels={"Category": ["Anomaly"]})]
    ranking_metrics = [ChannelAwareFScore(beta=0.5, select_labels={"Category": ["Rare Event", "Anomaly"]}),
                       ChannelAwareFScore(beta=0.5, select_labels={"Category": ["Anomaly"]}),
                       ADTQC(select_labels={"Category": ["Rare Event", "Anomaly"]}),
                       ADTQC(select_labels={"Category": ["Anomaly"]})]

    rs = Evaluator(args.result_folder, args.data_folders, selected_metrics, ranking_metrics, args.labels, threshold=threshold)
    rs.evaluate(args.select, args.force)
