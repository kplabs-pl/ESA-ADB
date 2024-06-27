import math
from typing import Optional

import numpy as np
import pandas as pd
import portion as P

from .ranking_metrics import MultiChannelMetric
from .utils import convert_time_series_to_events


class ADTQC(MultiChannelMetric):
    """Computes anomaly detection timing quality curve (ADTQC) scores used in the ESA Anomaly Detection Benchmark.

    Parameters
    ----------
    exponent : float
        Value of exponent of ADTQC. The default is math.e
    full_range : tuple of datetimes
        Optional tuple of (start time, end time) of the original data.
        If None, it is automatically inferred from the data.
    select_labels : dict
        Optional dictionary of event categories, classes or types to include in the calculation.
        Dictionary should contain column names and values from anomaly_types.csv as keys and values.
        If None, all events are included.
    name : str
        Optional custom name for the metric.
    """

    def __init__(self,
                 exponent: float = math.e,
                 full_range: Optional[tuple] = None,
                 select_labels: Optional[dict] = None,
                 name: Optional[str] = None) -> None:
        self.exponent = exponent
        self.full_range = full_range

        if select_labels is None or len(select_labels) == 0:
            self.selected_labels = dict()
            filter_string = "ALL"
        else:
            select_labels = {col: np.atleast_1d(val) for col, val in select_labels.items()}
            self.selected_labels = select_labels
            filter_string = "_".join(["_".join(val) for val in select_labels.values()])
        self._name = f"ADTQC_{filter_string}" if name is None else name

    def timing_curve(self, x, a, b):
        assert a >= pd.Timedelta(0)
        assert b >= pd.Timedelta(0)
        if (a == pd.Timedelta(0) or b == pd.Timedelta(0)) and x == pd.Timedelta(0):
            return 1
        if x <= -a or x >= b:
            return 0
        if -a < x <= pd.Timedelta(0):
            return ((x + a)/a)**self.exponent
        if pd.Timedelta(0) < x < b:
            denom_part = x/(b - x)
            return 1. / (1. + denom_part**self.exponent)

    def score(self, y_true: pd.DataFrame, y_pred: dict) -> dict:
        """
        Calculate scores.
        :param y_true: DataFrame representing labels.csv from ESA-ADB
        :param y_pred: dict of {channel_name: list of pairs (timestamp, is_anomaly)}, where is_anomaly is binary, 0 - nominal, 1 - anomaly
        :return: dictionary of calculated scores
        """
        for channel, values in y_pred.items():
            y_pred[channel] = np.asarray(values)

        # Adjust to full range
        min_y_pred = min(np.concatenate([y[..., 0] for y in y_pred.values()]))
        max_y_pred = max(np.concatenate([y[..., 0] for y in y_pred.values()]))
        if self.full_range is None:  # automatic full range
            self.full_range = (min(y_true["StartTime"].min(), min_y_pred), max(y_true["EndTime"].max(), max_y_pred))
        else:
            assert self.full_range[0] <= y_true["StartTime"].min()
            assert self.full_range[1] >= y_true["EndTime"].max()
            assert self.full_range[0] <= min_y_pred
            assert self.full_range[1] >= max_y_pred

        for channel, values in y_pred.items():
            if y_pred[channel][0, 0] > self.full_range[0]:
                y_pred[channel] = np.array([np.array([self.full_range[0], y_pred[channel][0, 1]]), *y_pred[channel]])
            if y_pred[channel][-1, 0] < self.full_range[1]:
                y_pred[channel] = np.array([*y_pred[channel], np.array([self.full_range[1], y_pred[channel][-1, 1]])])

        # Find prediction intervals per channel
        events_pred_dict = dict()
        for channel, pred in y_pred.items():
            events_pred_dict[channel] = convert_time_series_to_events(np.asarray(pred))

        # Analyze only selected anomaly types
        filtered_y_true = y_true.copy()
        for col, val in self.selected_labels.items():
            filtered_y_true = filtered_y_true[filtered_y_true[col].isin(val)]

        unique_anomaly_ids = filtered_y_true["ID"].unique()
        start_times = []
        for aid in unique_anomaly_ids:
            gt = filtered_y_true[filtered_y_true["ID"] == aid]
            start_times.append(min(gt["StartTime"]))
        start_times = sorted(start_times)

        before_tps = []
        after_tps = []
        curve_scores = []
        for aid in unique_anomaly_ids:
            gt = filtered_y_true[filtered_y_true["ID"] == aid]

            affected_channels = np.sort(gt["Channel"].unique())
            channels_intervals = dict()
            for channel in affected_channels:
                c_gt = gt[gt["Channel"] == channel]
                c_gt_intervals = []
                for _, row in c_gt[["StartTime", "EndTime"]].iterrows():
                    c_gt_intervals.append(P.closed(*row))
                channels_intervals[channel] = P.Interval(*c_gt_intervals)

            global_preds = []
            global_gts = []
            for channel in affected_channels:
                events_pred = [pred for pred in events_pred_dict[channel] if not (pred & channels_intervals[channel]).empty]
                global_preds.extend(events_pred)
                global_gts.append(channels_intervals[channel])
            global_preds = P.Interval(*global_preds)
            if global_preds.empty:  # no detection no score
                continue
            global_gts = P.Interval(*global_gts)

            anomaly_length = global_gts.upper - global_gts.lower
            current_anomaly_idx = start_times.index(global_gts.lower)
            previous_anomaly_start = start_times[current_anomaly_idx - 1] if current_anomaly_idx > 0 else global_gts.lower - anomaly_length
            alpha = min(anomaly_length, global_gts.lower - previous_anomaly_start)

            latency = global_preds.lower - global_gts.lower
            metric_value = self.timing_curve(latency, alpha, anomaly_length)
            curve_scores.append(metric_value)

            if latency < pd.Timedelta(0):
                before_tps.append(metric_value)
            else:
                after_tps.append(metric_value)

        print(before_tps)
        print(after_tps)
        print(curve_scores)

        before_tps = np.array(before_tps)
        after_tps = np.array(after_tps)
        curve_scores = np.array(curve_scores)

        result_dict = {"Nb_Before": len(before_tps),
                       "Nb_After": len(after_tps),
                       "AfterRate": len(after_tps) / len(curve_scores) if len(curve_scores) > 0 else np.nan,
                       "Total": np.mean(curve_scores) if len(curve_scores) > 0 else np.nan}

        return result_dict

    def supports_continuous_scorings(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._name


if __name__ == "__main__":

    full_range = [pd.to_datetime("8:10:10"), pd.to_datetime("8:11:24")]
    y_true = [
        ["id_0", "ch1", pd.to_datetime("8:10:16"), pd.to_datetime("8:10:35"), "", "", "", ""],
        ["id_0", "ch2", pd.to_datetime("8:10:10"), pd.to_datetime("8:10:24"), "", "", "", ""],
        ["id_1", "ch3", pd.to_datetime("8:10:30"), pd.to_datetime("8:10:34"), "", "", "", ""],
        ["id_1", "ch3", pd.to_datetime("8:10:40"), pd.to_datetime("8:10:45"), "", "", "", ""],
        ["id_2", "ch2", pd.to_datetime("8:10:54"), pd.to_datetime("8:11:06"), "", "", "", ""],
        ["id_2", "ch3", pd.to_datetime("8:10:54"), pd.to_datetime("8:11:06"), "", "", "", ""],
        ["id_3", "ch1", pd.to_datetime("8:11:08"), pd.to_datetime("8:11:24"), "", "", "", ""]]
    y_true = pd.DataFrame(y_true,
                          columns=["ID", "Channel", "StartTime", "EndTime", "Category", "Dimensionality", "Locality",
                                   "Length"])

    y_pred = np.array([[[pd.to_datetime("8:10:10"), 0],
                        [pd.to_datetime("8:10:14"), 1],
                        [pd.to_datetime("8:10:31"), 0],
                        [pd.to_datetime("8:10:41"), 1]],
                           [[pd.to_datetime("8:10:10"), 0],
                            [pd.to_datetime("8:10:16"), 1],
                            [pd.to_datetime("8:10:22"), 0]],
                               [[pd.to_datetime("8:10:10"), 0],
                                [pd.to_datetime("8:10:25"), 1],
                                [pd.to_datetime("8:10:41"), 0]]
                       ], dtype=object)
    y_pred_channels_order = ["ch1", "ch2", "ch3"]
    y_pred = {key: value for key, value in zip(y_pred_channels_order, y_pred)}


    metric = ADTQC(full_range=full_range)
    print(metric.name, metric.score(y_true, y_pred))
