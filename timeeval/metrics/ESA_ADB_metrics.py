from typing import Optional

import numpy as np
import pandas as pd
import portion as P
from collections import defaultdict

from .affiliation_based_metrics_repo.affiliation import pr_from_events
from .metric import Metric
from .utils import convert_time_series_to_events, NANOSECONDS_IN_SECOND


class ESAScores(Metric):
    """Computes the corrected event-wise F-score, alarming precision, and affiliation-based F-score
    used in the ESA Anomaly Detection Benchmark.

    Parameters
    ----------
    betas : float or list
        beta determines the weight of recall in the combined F-beta-scores.
        beta < 1 lends more weight to precision, while beta > 1 favors recall.
        Multiple betas can be provided as a list.
    select_labels : dict
        Optional dictionary of event categories, classes or types to include in the calculation.
        Dictionary should contain column names and values from anomaly_types.csv as keys and values.
        If None, all events are included.
    full_range : tuple of datetimes
        Optional tuple of (start time, end time) of the original data.
        If None, it is automatically inferred from the data.
    name : str
        Optional custom name for the metric.
    """

    def __init__(self,
                 betas: float = 1,
                 select_labels: Optional[dict] = None,
                 full_range: Optional[tuple] = None,
                 name: Optional[str] = None) -> None:
        self._betas = np.atleast_1d(betas)
        self.full_range = full_range

        if select_labels is None or len(select_labels) == 0:
            self.selected_labels = dict()
            filter_string = "ALL"
        else:
            select_labels = {col: np.atleast_1d(val) for col, val in select_labels.items()}
            self.selected_labels = select_labels
            filter_string = "_".join(["_".join(val) for val in select_labels.values()])
        self._name = filter_string if name is None else name

    def score(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> dict:
        """
        Calculate scores.
        :param y_true: DataFrame representing labels.csv from ESA-ADB without the "Channel" column
        :param y_pred: list of pairs (timestamp, is_anomaly), where is_anomaly is binary, 0 - nominal, 1 - anomaly
        :return: dictionary of calculated scores
        """
        y_pred = np.asarray(y_pred)

        if self.full_range is None:  # automatic full range
            self.full_range = (min(y_true["StartTime"].min(), min(y_pred[..., 0])), max(y_true["EndTime"].max(), max(y_pred[..., 0])))
        else:
            assert self.full_range[0] <= y_true["StartTime"].min()
            assert self.full_range[1] >= y_true["EndTime"].max()
            assert self.full_range[0] <= min(y_pred[..., 0])
            assert self.full_range[1] >= max(y_pred[..., 0])
        if y_pred[0, 0] > self.full_range[0]:
            y_pred = np.array([np.array([self.full_range[0], y_pred[0, 1]]), *y_pred])
        if y_pred[-1, 0] < self.full_range[1]:
            y_pred = np.array([*y_pred, np.array([self.full_range[1], y_pred[-1, 1]])])

        events_pred = convert_time_series_to_events(y_pred)

        filtered_y_true = y_true.copy()
        for col, val in self.selected_labels.items():
            filtered_y_true = filtered_y_true[filtered_y_true[col].isin(val)]

        # event-wise scores
        true_positives = 0
        false_positives = 0
        redundant_detections = 0
        false_negatives = 0
        matched_events_pred = [False for _ in events_pred]
        for aid in filtered_y_true["ID"].unique():
            gt = filtered_y_true[filtered_y_true["ID"] == aid]

            gt_intervals = []
            for _, row in gt[["StartTime", "EndTime"]].iterrows():
                gt_intervals.append(P.closed(*row))
            gt_intervals = P.Interval(*gt_intervals)

            already_detected = [0 for _ in gt_intervals]
            at_least_one_detected = False
            for p, pred in enumerate(events_pred):
                if pred.upper < gt_intervals.lower or pred.lower > gt_intervals.upper:
                    continue
                intersections = [not (pred & g).empty for g in gt_intervals]
                if not any(intersections):
                    continue
                matched_events_pred[p] = True
                if not at_least_one_detected:
                    true_positives += 1
                    at_least_one_detected = True
                for i, val in enumerate(intersections):
                    if val:
                        already_detected[i] += 1

            for det in already_detected:
                if det > 1:
                    redundant_detections += (det - 1)

            if not at_least_one_detected:
                false_negatives += 1

        events_gt = []
        for _, row in y_true[["StartTime", "EndTime"]].iterrows():
            events_gt.append(P.closed(*row))
        events_gt = P.Interval(*events_gt)

        for pred, matched in zip(events_pred, matched_events_pred):
            if not matched and (pred & events_gt).empty:
                false_positives += 1

        divider = true_positives + false_positives
        if divider == 0:
            precision = 0.0
        else:
            precision = true_positives / divider

        divider = true_positives + redundant_detections
        if divider == 0:
            precision_one_detection = 0.0
        else:
            precision_one_detection = true_positives / divider

        # Correction from (Sehili et al., 2023) http://arxiv.org/abs/2308.13068
        if precision > 0:
            nominal_interval = P.closed(*self.full_range) - events_gt
            false_positives_interval = nominal_interval & events_pred

            nominal_seconds = 0
            for interval in nominal_interval:
                nominal_seconds += (interval.upper - interval.lower).value / NANOSECONDS_IN_SECOND
            false_positive_seconds = 0
            for interval in false_positives_interval:
                false_positive_seconds += (interval.upper - interval.lower).value / NANOSECONDS_IN_SECOND
            tnr = (1 - false_positive_seconds / nominal_seconds)
            precision *= tnr

        divider = true_positives + false_negatives
        if divider == 0:
            recall = 0.0
        else:
            recall = true_positives / divider

        result_dict = {"alarming_precision": precision_one_detection, "EW_precision": precision, "EW_recall": recall}

        for b in self._betas:
            divider = b ** 2 * precision + recall
            if divider == 0:
                result_dict[f"EW_F_{b:.2f}"] = 0
            else:
                result_dict[f"EW_F_{b:.2f}"] = ((1 + b ** 2) * precision * recall) / divider

        # Affiliation-based
        # Transform timestamps to nanseconds and manage point anomalies (as 1 ns long anomalies)
        events_pred_ns = [(e.lower.value, e.upper.value if e.lower.value != e.upper.value else e.upper.value + 1)
                          for e in events_pred]
        events_gt_ns = [(e.lower.value, e.upper.value if e.lower.value != e.upper.value else e.upper.value + 1)
                        for e in events_gt]

        # correct full range if there is a point anomaly at the end
        corrected_full_upper_range = max(np.max(events_pred_ns), np.max(events_gt_ns), self.full_range[1].value)
        score_dict = pr_from_events(events_pred_ns, events_gt_ns,
                                    (self.full_range[0].value, corrected_full_upper_range))

        precision_dict = defaultdict(list)
        recall_dict = defaultdict(list)
        for pr, rec, zone in zip(score_dict["individual_precision_probabilities"],
                                 score_dict["individual_recall_probabilities"], events_gt):
            intersections = [P.closed(*row) & zone for _, row in y_true[["StartTime", "EndTime"]].iterrows()]

            y_true_in_zone = y_true[[not inter.empty for inter in intersections]]

            filtered_y_true_in_zone = y_true_in_zone.copy()
            for col, val in self.selected_labels.items():
                filtered_y_true_in_zone = filtered_y_true_in_zone[filtered_y_true_in_zone[col].isin(val)]

            # ignore affiliation zone if there are any non-selected events
            if len(y_true_in_zone) > len(filtered_y_true_in_zone):
                continue

            ids = y_true_in_zone["ID"]
            for id_ in ids:
                precision_dict[id_].append(pr)
                recall_dict[id_].append(rec)
        precision_list = []
        recall_list = []
        for pr, rec in zip(precision_dict.values(), recall_dict.values()):
            precision_list.append(np.mean(pr))
            recall_list.append(np.mean(rec))

        print(precision_list)
        print(recall_list)
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)

        result_dict["AFF_precision"] = precision
        result_dict["AFF_recall"] = recall

        for b in self._betas:
            result_dict[f"AFF_F_{b:.2f}"] = ((1 + b ** 2) * precision * recall) / (b ** 2 * precision + recall)

        return result_dict

    def supports_continuous_scorings(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._name


if __name__ == "__main__":
    full_range = [pd.to_datetime("2015-01-01"), pd.to_datetime("2015-01-15")]
    y_true = [["id_0", pd.to_datetime("2015-01-01"), pd.to_datetime("2015-01-02"), "Anomaly", "Multivariate", "Global",
               "Point"],
              ["id_1", pd.to_datetime("2015-01-04"), pd.to_datetime("2015-01-05"), "Anomaly", "Univariate", "Local",
               "Subsequence"],
              ["id_2", pd.to_datetime("2015-01-07"), pd.to_datetime("2015-01-08"), "Anomaly", "Multivariate", "Local",
               "Subsequence"]]
    y_true = pd.DataFrame(y_true,
                          columns=["ID", "StartTime", "EndTime", "Category", "Dimensionality", "Locality", "Length"])
    y_pred = [[pd.to_datetime("2015-01-01"), 0],
              [pd.to_datetime("2015-01-04"), 1],
              [pd.to_datetime("2015-01-09"), 0]]
    metrics = [ESAScores(betas=0.5, full_range=full_range, select_labels={"Dimensionality": "Multivariate", "Length": "Subsequence"})]
    for metric in metrics:
        print(metric.name, metric.score(y_true, y_pred))



