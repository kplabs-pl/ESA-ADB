from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Generator, Optional

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..algorithm import Algorithm
from ..constants import EXECUTION_LOG, ANOMALY_SCORES_TS, METRICS_CSV, HYPER_PARAMETERS
from ..core.times import Times
from ..data_types import AlgorithmParameter, TrainingType, InputDimensionality
from ..datasets import Datasets, Dataset
from ..heuristics import inject_heuristic_values
from ..metrics import MultiChannelMetric, Metric, DefaultMetrics, ChannelAwareFScore, ADTQC
from ..resource_constraints import ResourceConstraints
from ..utils.datasets import extract_features, load_dataset, load_labels_only
from ..utils.encode_params import dump_params
from ..utils.hash_dict import hash_dict
from ..utils.results_path import generate_experiment_path


@dataclass
class Experiment:
    dataset: Dataset
    algorithm: Algorithm
    params: dict
    params_id: str
    repetition: int
    base_results_dir: Path
    resource_constraints: ResourceConstraints
    metrics: List[Metric]
    ranking_metrics: Optional[List[MultiChannelMetric]]
    resolved_train_dataset_path: Optional[Path]
    resolved_test_dataset_path: Path
    labels_df: pd.DataFrame
    test_data_scores: pd.DataFrame
    subsystems_mapping: dict

    @property
    def name(self) -> str:
        return f"{self.algorithm.name}-{self.dataset.collection_name}-{self.dataset.name}-{self.params_id}-{self.repetition}"

    @property
    def dataset_collection(self) -> str:
        return self.dataset.collection_name

    @property
    def dataset_name(self) -> str:
        return self.dataset.name

    @property
    def results_path(self) -> Path:
        return generate_experiment_path(self.base_results_dir, self.algorithm.name, self.params_id,
                                        self.dataset_collection, self.dataset_name, self.repetition)

    def build_args(self) -> dict:
        return {
            "results_path": self.results_path,
            "resource_constraints": self.resource_constraints,
            "hyper_params": self.params,
            "dataset_details": self.dataset
        }

    @staticmethod
    def scale_scores(y_scores: np.ndarray) -> np.ndarray:
        y_scores = np.asarray(y_scores, dtype=np.float32)

        if y_scores.ndim == 1:
            y_scores = np.expand_dims(y_scores, -1)

        for i in range(y_scores.shape[-1]):
            # mask NaNs and Infs
            mask = np.isinf(y_scores[..., i]) | np.isneginf(y_scores[..., i]) | np.isnan(y_scores[..., i])

            # scale all other scores to [0, 1]
            scores = y_scores[..., i][~mask]
            if scores.size != 0:
                if len(scores.shape) == 1:
                    scores = scores.reshape(-1, 1)
                y_scores[..., i][~mask] = MinMaxScaler().fit_transform(scores).ravel()

        return y_scores

    def evaluate(self) -> dict:
        """
        Using TimeEval distributed, this method is executed on the remote node.
        """
        with (self.results_path / EXECUTION_LOG).open("a") as logs_file:
            print(f"Starting evaluation of experiment {self.name}\n=============================================\n",
                  file=logs_file)

        # persist hyper parameters to disk
        dump_params(self.params, self.results_path / HYPER_PARAMETERS)

        # perform training if necessary
        results = self._perform_training()

        # perform execution
        y_scores, execution_times = self._perform_execution()
        results.update(execution_times)
        # backup results to disk
        pd.DataFrame([results]).to_csv(self.results_path / METRICS_CSV, index=False)

        if "target_channels" in self.params:
            channel_names = self.params["target_channels"]
        else:
            columns = pd.read_csv(self.resolved_test_dataset_path, index_col=0, nrows=0).columns.tolist()
            anomaly_columns = [x for x in columns if x.startswith("is_anomaly")]
            channel_names = columns[:-len(anomaly_columns)]

        if self.labels_df is None:
            y_true = load_labels_only(self.resolved_test_dataset_path, channel_names)
            if y_true.ndim == 1:
                y_true = np.expand_dims(y_true, -1)
        else:
            y_true = self.labels_df[self.labels_df["Channel"].isin(channel_names)]

        if y_scores.ndim == 1:
            y_scores = np.expand_dims(y_scores, -1)

        y_scores = self.scale_scores(y_scores)
        # persist scores to disk
        np.savetxt(str(self.results_path / ANOMALY_SCORES_TS), y_scores, delimiter=",")

        with (self.results_path / EXECUTION_LOG).open("a") as logs_file:
            print(f"Scoring algorithm {self.algorithm.name} with {','.join([m.name for m in self.metrics])} metrics",
                  file=logs_file)
            if self.ranking_metrics is not None:
                print(f"Scoring algorithm {self.algorithm.name} with {','.join([m.name for m in self.ranking_metrics])} ranking metrics",
                      file=logs_file)

            errors = 0
            last_exception = None

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
                    metric._plot_store = self.results_path

                try:
                    if self.test_data_scores is not None:
                        self.test_data_scores["Score"] = y_scores.max(axis=1).astype(np.uint8)
                        score = metric.score(y_true.drop(columns=["Channel"]), self.test_data_scores)
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
                    print(f"Exception while computing metric {metric.name}: {e}", file=logs_file)
                    errors += 1
                    if str(e):
                        last_exception = e
                    continue

            for metric in self.ranking_metrics:
                if not isinstance(metric, ADTQC):
                    continue

                try:
                    y_pred = dict()
                    self.test_data_scores["Score"] = y_scores.max(axis=1).astype(np.uint8)
                    y_pred["global"] = self.test_data_scores[["Timestamp", "Score"]].copy()

                    y_true_global = y_true.copy()
                    y_true_global["Channel"] = "global"
                    score = metric.score(y_true_global, y_pred)
                    for submetric, value in score.items():
                        print(f"Global_{metric.name}_{submetric} calculated: {value}")
                        results[f"Global_{metric.name}_{submetric}"] = value

                except Exception as e:
                    print(f"Exception while computing global metric {metric.name}!")
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
                            self.test_data_scores["Score"] = y_scores[..., c].astype(np.uint8)
                            y_pred[channel_name] = self.test_data_scores.copy()

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
                        print(f"Exception while computing metric {metric.name}: {e}", file=logs_file)
                        errors += 1
                        continue

                for metric in self.metrics:
                    for c, channel_name in enumerate(channel_names):
                        try:
                            if self.test_data_scores is not None:
                                self.test_data_scores["Score"] = y_scores[..., c].astype(np.uint8)
                                score = metric.score(y_true[y_true["Channel"] == channel_name].drop(columns=["Channel"]), self.test_data_scores)
                            else:
                                score = metric(y_true[..., c], y_scores[..., c])

                            if isinstance(score, dict):
                                for submetric, value in score.items():
                                    print(f"{metric.name}_{submetric}_{channel_names[c]} calculated: {value}")
                                    results[f"{metric.name}_{submetric}_{channel_names[c]}"] = value
                            else:
                                print(f"{metric.name}_{channel_names[c]} calculated: {score}")
                                results[f"{metric.name}_{channel_names[c]}"] = score
                            print(f"  = {score}", file=logs_file)
                            logs_file.flush()
                        except Exception as e:
                            print(f"Exception while computing metric {metric.name}: {e}", file=logs_file)
                            errors += 1
                            if str(e):
                                last_exception = e
                            continue

        # write all results to disk (overwriting backup)
        pd.DataFrame([results]).to_csv(self.results_path / METRICS_CSV, index=False)

        # rethrow exception if no metric could be calculated
        if errors == len(self.metrics) and last_exception is not None:
            raise last_exception

        return results

    def _perform_training(self) -> dict:
        if self.algorithm.training_type == TrainingType.UNSUPERVISED:
            return {}

        if not self.resolved_train_dataset_path:
            raise ValueError(f"No training dataset was provided. Algorithm cannot be trained!")

        if self.algorithm.data_as_file:
            X: AlgorithmParameter = self.resolved_train_dataset_path
        else:
            X = load_dataset(self.resolved_train_dataset_path).values

        with (self.results_path / EXECUTION_LOG).open("a") as logs_file, redirect_stdout(logs_file):
            print(f"Performing training for {self.algorithm.training_type.name} algorithm {self.algorithm.name}")
            times = Times.from_train_algorithm(self.algorithm, X, self.build_args())
        return times.to_dict()

    def _perform_execution(self) -> Tuple[np.ndarray, dict]:
        if self.algorithm.data_as_file:
            X: AlgorithmParameter = self.resolved_test_dataset_path
        else:
            dataset = load_dataset(self.resolved_test_dataset_path)
            if dataset.shape[1] >= 3:
                X = extract_features(dataset)
            else:
                raise ValueError(
                    f"Dataset '{self.resolved_test_dataset_path.name}' has a shape that was not expected: {dataset.shape}")

        with (self.results_path / EXECUTION_LOG).open("a") as logs_file, redirect_stdout(logs_file):
            print(f"Performing execution for {self.algorithm.training_type.name} algorithm {self.algorithm.name}")
            y_scores, times = Times.from_execute_algorithm(self.algorithm, X, self.build_args())
        return y_scores, times.to_dict()


class Experiments:
    def __init__(self,
                 dmgr: Datasets,
                 datasets: List[Dataset],
                 algorithms: List[Algorithm],
                 base_result_path: Path,
                 resource_constraints: ResourceConstraints = ResourceConstraints.default_constraints(),
                 repetitions: int = 1,
                 metrics: Optional[List[Metric]] = None,
                 ranking_metrics: Optional[List[MultiChannelMetric]] = None,
                 skip_invalid_combinations: bool = False,
                 force_training_type_match: bool = False,
                 force_dimensionality_match: bool = False,
                 experiment_combinations_file: Optional[Path] = None,
                 labels_csv_path: Optional[Path] = None,
                 test_dataset_path: Optional[Path] = None
                 ):
        self.dmgr = dmgr
        self.datasets = datasets
        self.algorithms = algorithms
        self.repetitions = repetitions
        self.base_result_path = base_result_path
        self.resource_constraints = resource_constraints
        self.metrics = metrics or DefaultMetrics.default_list()
        self.ranking_metrics = ranking_metrics
        self.skip_invalid_combinations = skip_invalid_combinations or force_training_type_match or force_dimensionality_match
        self.force_training_type_match = force_training_type_match
        self.force_dimensionality_match = force_dimensionality_match
        self.experiment_combinations: Optional[pd.DataFrame] = pd.read_csv(experiment_combinations_file) if experiment_combinations_file else None
        if self.skip_invalid_combinations:
            self._N: Optional[int] = None
        else:
            self._N = sum(
                [len(algo.param_config) for algo in self.algorithms]
            ) * len(self.datasets) * self.repetitions

        self.labels_csv_path = labels_csv_path
        self.labels_df = None
        self.test_data_scores = None

        if labels_csv_path is not None:
            anomaly_types_path = str(labels_csv_path).replace("labels.csv", "anomaly_types.csv")
            if not os.path.isfile(labels_csv_path):
                print(f"There is no labels CSV file ({labels_csv_path})! Continuing with standard labels in data")
            elif not os.path.isfile(anomaly_types_path):
                print(f"There is no anomaly types CSV file ({anomaly_types_path})! Continuing with standard labels in data")
            elif not os.path.isfile(test_dataset_path):
                print(f"There is no test dataset ({test_dataset_path})! Continuing with standard labels in data")
            else:
                self.labels_df: pd.DataFrame = pd.read_csv(labels_csv_path, parse_dates=["StartTime", "EndTime"])
                self.labels_df["StartTime"] = self.labels_df["StartTime"].apply(lambda t: t.tz_localize(None))
                self.labels_df["EndTime"] = self.labels_df["EndTime"].apply(lambda t: t.tz_localize(None))

                self.test_data_scores = pd.read_csv(test_dataset_path, usecols=["timestamp"], parse_dates=[0])
                self.test_data_scores.rename(columns={"timestamp": "Timestamp"}, inplace=True)
                self.test_data_scores["Score"] = np.uint8(0)

                self.labels_df = self.labels_df[self.labels_df["StartTime"] >= self.test_data_scores["Timestamp"].min()]
                self.labels_df = self.labels_df[self.labels_df["EndTime"] <= self.test_data_scores["Timestamp"].max()]

                anomaly_types_df: pd.DataFrame = pd.read_csv(anomaly_types_path)
                columns_to_copy = anomaly_types_df.columns[-4:]
                for col in columns_to_copy:
                    self.labels_df[col] = ""
                for index, row in anomaly_types_df.iterrows():
                    self.labels_df.loc[self.labels_df["ID"] == row["ID"], columns_to_copy] = row[columns_to_copy].values

            channels_path = str(labels_csv_path).replace("labels.csv", "channels.csv")
            if not os.path.isfile(channels_path):
                self.subsystems_mapping = None
                print(f"There is no channels CSV file ({channels_path})! Metrics at the subsystem level won't be calculated")
            else:
                channels_df = pd.read_csv(channels_path)
                self.subsystems_mapping = {s: [*v] for s, v in channels_df.groupby("Subsystem")["Channel"]}

    def _should_be_run(self, algorithm: Algorithm, dataset: Dataset, params_id: str) -> bool:
        return self.experiment_combinations is None or \
                not self.experiment_combinations[
                    (self.experiment_combinations.algorithm == algorithm.name) &
                    (self.experiment_combinations.collection == dataset.datasetId[0]) &
                    (self.experiment_combinations.dataset == dataset.datasetId[1]) &
                    (self.experiment_combinations.hyper_params_id == params_id)
                ].empty

    def __iter__(self) -> Generator[Experiment, None, None]:
        for algorithm in self.algorithms:
            for algorithm_config in algorithm.param_config:
                for dataset in self.datasets:
                    if self._check_compatible(dataset, algorithm):
                        test_path, train_path = self._resolve_dataset_paths(dataset, algorithm)
                        # create parameter hash before executing heuristics
                        # (they replace the parameter values, but we want to be able to group by original configuration)
                        params_id = hash_dict(algorithm_config)
                        params = inject_heuristic_values(algorithm_config, algorithm, dataset, test_path)
                        if self._should_be_run(algorithm, dataset, params_id):
                            for repetition in range(1, self.repetitions + 1):
                                yield Experiment(
                                    algorithm=algorithm,
                                    dataset=dataset,
                                    params=params,
                                    params_id=params_id,
                                    repetition=repetition,
                                    base_results_dir=self.base_result_path,
                                    resource_constraints=self.resource_constraints,
                                    metrics=self.metrics,
                                    ranking_metrics=self.ranking_metrics,
                                    resolved_test_dataset_path=test_path,
                                    resolved_train_dataset_path=train_path,
                                    labels_df=self.labels_df,
                                    test_data_scores=self.test_data_scores,
                                    subsystems_mapping=self.subsystems_mapping
                                )

    def __len__(self) -> int:
        if self._N is None:
            self._N = sum([
                int(self._should_be_run(algorithm, dataset, hash_dict(algorithm_config)))
                for algorithm in self.algorithms
                for algorithm_config in algorithm.param_config
                for dataset in self.datasets
                if self._check_compatible(dataset, algorithm)
                for _repetition in range(1, self.repetitions + 1)
            ])
        return self._N  # type: ignore

    def _resolve_dataset_paths(self, dataset: Dataset, algorithm: Algorithm) -> Tuple[Path, Optional[Path]]:
        test_dataset_path = self.dmgr.get_dataset_path(dataset.datasetId, train=False)
        train_dataset_path: Optional[Path] = None
        if algorithm.training_type != TrainingType.UNSUPERVISED:
            try:
                train_dataset_path = self.dmgr.get_dataset_path(dataset.datasetId, train=True)
            except KeyError:
                pass
        return test_dataset_path, train_dataset_path

    def _check_compatible(self, dataset: Dataset, algorithm: Algorithm) -> bool:
        if not self.skip_invalid_combinations:
            return True

        if (self.force_training_type_match or
                algorithm.training_type in [TrainingType.SUPERVISED, TrainingType.SEMI_SUPERVISED]):
            train_compatible = algorithm.training_type == dataset.training_type
        else:
            train_compatible = True

        if self.force_dimensionality_match:
            dim_compatible = algorithm.input_dimensionality == dataset.input_dimensionality
        else:
            """
            m = multivariate, u = univariate
            algo | data | res
              u  |  u   | 1
              u  |  m   | 0 <-- not compatible
              m  |  u   | 1
              m  |  m   | 1
            """
            dim_compatible = not (algorithm.input_dimensionality == InputDimensionality.UNIVARIATE and dataset.input_dimensionality == InputDimensionality.MULTIVARIATE)
        return dim_compatible and train_compatible
