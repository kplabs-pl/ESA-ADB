import argparse
from typing import Callable
import math
import copy
from timeeval.metrics.eTaPR_pkg.DataManage import File_IO  #, Time_Plot
from timeeval.metrics.eTaPR_pkg.DataManage import Range as rng


class TaPR:
    def __init__(self, theta: float, delta: int):
        self._predictions = []  # list of Ranges
        self._anomalies = []    # list of Ranges
        self._ambiguous_inst = [] # list of Ranges

        self._set_predictions = False
        self._set_anomalies = False

        #self._rho = theta
        #self._pi = theta
        self._theta = theta
        self._delta = delta

        pass

    def set_anomalies(self, anomaly_list: list) -> None:
        self._anomalies = copy.deepcopy(anomaly_list)
        self._gen_ambiguous()
        self._set_anomalies = True

    def set_predictions(self, prediction_list: list) -> None:
        self._predictions = copy.deepcopy(prediction_list)
        self._set_predictions = True

    def _gen_ambiguous(self):
        for i in range(len(self._anomalies)):
            start_id = self._anomalies[i].get_time()[1] + 1
            end_id = start_id + self._delta

            #if the next anomaly occurs during the theta, update the end_id
            if i+1 < len(self._anomalies) and end_id > self._anomalies[i+1].get_time()[0]:
                end_id = self._anomalies[i+1].get_time()[0] - 1

            if start_id > end_id:
                start_id = -2
                end_id = -1

            self._ambiguous_inst.append(rng.Range(start_id, end_id, str(i)))

    def get_n_predictions(self):
        return len(self._predictions)

    def get_n_anomalies(self):
        return len(self._anomalies)

    def _ids_2_objects(self, id_list, range_list):
        result = []
        for id in id_list:
            result.append(range_list[id])
        return result

    def TaR_d(self) -> float and list:
        score, detected_id_list = self._TaR_d(self._anomalies, self._ambiguous_inst, self._predictions, self._theta)
        return score, self._ids_2_objects(detected_id_list, self._anomalies)

    def _TaR_d(self, anomalies: list, ambiguous_inst: list, predictions: list, threshold: float) -> float and list:
        total_score = 0.0
        detected_anomalies = []
        for anomaly_id in range(len(anomalies)):
            anomaly = anomalies[anomaly_id]
            ambiguous = ambiguous_inst[anomaly_id]

            max_score = self._sum_of_func(anomaly.get_time()[0], anomaly.get_time()[1],
                                          anomaly.get_time()[0], anomaly.get_time()[1], self._uniform_func)

            score = 0.0
            for prediction in predictions:
                score += self._overlap_and_subsequent_score(anomaly, ambiguous, prediction)

            if min(1.0, score / max_score) >= threshold:
                total_score += 1.0
                detected_anomalies.append(anomaly_id)

        if len(anomalies) == 0:
            return 0.0, []
        else:
            return total_score / len(anomalies), detected_anomalies

    def TaP_d(self) -> float and list:
        score, correct_id_list = self._TaP_d(self._anomalies, self._ambiguous_inst, self._predictions, self._theta)
        return score, self._ids_2_objects(correct_id_list, self._predictions)

    def _TaP_d(self, anomalies, ambiguous_inst, predictions, threshold):
        correct_predictions = []
        total_score = 0.0
        for prediction_id in range(len(predictions)):
            max_score = predictions[prediction_id].get_time()[1] - predictions[prediction_id].get_time()[0] + 1

            score = 0.0
            for anomaly_id in range(len(anomalies)):
                anomaly = anomalies[anomaly_id]
                ambiguous = ambiguous_inst[anomaly_id]

                score += self._overlap_and_subsequent_score(anomaly, ambiguous, predictions[prediction_id])

            if (score/max_score) >= threshold:
                total_score += 1.0
                correct_predictions.append(prediction_id)

        if len(predictions) == 0:
            return 0.0, []
        else:
            return total_score / len(predictions), correct_predictions

    def _detect(self, src_range: rng.Range, ranges: list, theta: int) -> bool:
        rest_len = src_range.get_time()[1] - src_range.get_time()[0] + 1
        for dst_range in ranges:
            len = self._overlapped_len(src_range, dst_range)
            if len != -1:
                rest_len -= len
        return (float)(rest_len) / (src_range.get_time()[1] - src_range.get_time()[0] + 1) <= (1.0 - theta)

    def _overlapped_len(self, range1: rng.Range, range2: rng.Range) -> int:
        detected_start = max(range1.get_time()[0], range2.get_time()[0])
        detected_end = min(range1.get_time()[1], range2.get_time()[1])

        if detected_end < detected_start:
            return 0
        else:
            return detected_end - detected_start + 1

    def _min_max_norm(self, value: int, org_min: int, org_max: int, new_min: int, new_max: int) -> float:
        if org_min == org_max:
            return new_min
        else:
            return (float)(new_min) + (float)(value - org_min) * (new_max - new_min) / (org_max - org_min)

    def _decaying_func(self, val: float) -> float:
        assert (-6 <= val <= 6)
        return 1 / (1 + math.exp(val))

    def _ascending_func(self, val: float) -> float:
        assert (-6 <= val <= 6)
        return 1 / (1 + math.exp(val * -1))

    def _uniform_func(self, val: float) -> float:
        return 1.0

    def _sum_of_func(self, start_time: int, end_time: int, org_start: int, org_end: int,
                     func: Callable[[float], float]) -> float:
        val = 0.0
        for timestamp in range(start_time, end_time + 1):
            val += func(self._min_max_norm(timestamp, org_start, org_end, -6, 6))
        return val

    def _overlap_and_subsequent_score(self, anomaly: rng.Range, ambiguous: rng.Range, prediction: rng.Range) -> float:
        score = 0.0

        detected_start = max(anomaly.get_time()[0], prediction.get_time()[0])
        detected_end = min(anomaly.get_time()[1], prediction.get_time()[1])

        score += self._sum_of_func(detected_start, detected_end,
                                   anomaly.get_time()[0], anomaly.get_time()[1], self._uniform_func)

        if ambiguous.get_time()[0] < ambiguous.get_time()[1]:
            detected_start = max(ambiguous.get_time()[0], prediction.get_time()[0])
            detected_end = min(ambiguous.get_time()[1], prediction.get_time()[1])

            score += self._sum_of_func(detected_start, detected_end,
                                       ambiguous.get_time()[0], ambiguous.get_time()[1], self._decaying_func)

        return score

    def TaR_p(self) -> float:
        total_score = 0.0
        for anomaly_id in range(len(self._anomalies)):
            anomaly = self._anomalies[anomaly_id]
            ambiguous = self._ambiguous_inst[anomaly_id]

            max_score = self._sum_of_func(anomaly.get_time()[0], anomaly.get_time()[1],
                                          anomaly.get_time()[0], anomaly.get_time()[1], self._uniform_func)

            score = 0.0
            for prediction in self._predictions:
                score += self._overlap_and_subsequent_score(anomaly, ambiguous, prediction)

            total_score += min(1.0, score/max_score)

        if len(self._anomalies) == 0:
            return 0.0
        else:
            return total_score / len(self._anomalies)

    def TaP_p(self) -> float:
        total_score = 0.0
        for prediction in self._predictions:
            max_score = prediction.get_time()[1] - prediction.get_time()[0] + 1

            score = 0.0
            for anomaly_id in range(len(self._anomalies)):
                anomaly = self._anomalies[anomaly_id]
                ambiguous = self._ambiguous_inst[anomaly_id]

                score += self._overlap_and_subsequent_score(anomaly, ambiguous, prediction)

            total_score += score/max_score

        if len(self._predictions) == 0:
            return 0.0
        else:
            return total_score / len(self._predictions)


def compute(anomalies: list, predictions: list, alpha: float, theta: float, delta: int) -> dict:
    ev = TaPR(theta, delta)

    ev.set_anomalies(anomalies)
    ev.set_predictions(predictions)

    tard_value, detected_list = ev.TaR_d()
    tarp_value = ev.TaR_p()

    tapd_value, correct_list = ev.TaP_d()
    tapp_value = ev.TaP_p()

    result = {}
    tar_value = alpha * tard_value + (1 - alpha) * tarp_value
    result['TaR'] = tar_value
    result['TaRd'] = tard_value
    result['TaRp'] = tarp_value

    tap_value = alpha * tapd_value + (1 - alpha) * tapp_value
    result['TaP'] = tap_value
    result['TaPd'] = tapd_value
    result['TaPp'] = tapp_value

    detected_anomalies = []
    for value in detected_list:
        detected_anomalies.append(value.get_name())

    result['Detected_Anomalies'] = detected_anomalies
    result['Detected_Anomalies_Ranges'] = detected_list
    result['Correct_Predictions_Ranges'] = correct_list

    if tar_value + tap_value == 0:
        result['f1'] = 0.0
    else:
        result['f1'] = (2 * tar_value * tap_value) / (tar_value + tap_value)

    return result


def compute_with_load(anomaly_file: str, prediction_file: str, file_type: str, alpha: float, theta: float, delta: int) -> dict:
    anomalies = File_IO.load_file(anomaly_file, file_type)
    predictions = File_IO.load_file(prediction_file, file_type)
    return compute(anomalies, predictions, alpha, theta, delta)


def print_result(anomalies: list, predictions: list, alpha: float, theta: float, delta: int, verbose: bool, graph: str) -> None:
    org_predictions = copy.deepcopy(predictions)
    result = compute(anomalies, predictions, alpha, theta, delta)

    print("The parameters (alpha, theta, delta) are set as %g, %g, and %d." % (alpha, theta, delta))

    print('\n[TaR]:', "%0.5f" % result['TaR'])
    print("\t* Detection score:", "%0.5f" % result['TaRd'])
    print("\t* Portion score:", "%0.5f" % result['TaRp'])
    if verbose:
        buf = '\t\tdetected anomalies: '
        if len(result['Detected_Anomalies_Ranges']) == 0:
            buf += "None  "
        else:
            for value in result['Detected_Anomalies_Ranges']:
                buf += value.get_name() + '(' + str(value.get_time()[0]) + ':' + str(value.get_time()[1]) + '), '
        print(buf[:-2])


    print('\n[TaP]:', "%0.5f" % result['TaP'])
    print("\t* Detection score:", "%0.5f" % result['TaPd'])
    print("\t* Portion score:", "%0.5f" % result['TaPp'])
    if verbose:
        buf = '\t\tcorrect predictions: '
        if len(result['Correct_Predictions_Ranges']) == 0:
            buf += "None  "
        else:
            for value in result['Correct_Predictions_Ranges']:
                buf += value.get_name() + '(' + str(value.get_time()[0]) + ':' + str(value.get_time()[1]) + '), '
        print(buf[:-2])


    assert(graph == 'screen' or graph == 'file' or graph == 'none' or graph == 'all')
    # if graph == 'screen' or graph == 'file' or graph == 'all':
    #     Time_Plot.draw_graphs(anomalies, org_predictions, graph)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--anomalies", help="anomaly file name (ground truth)", required=True)
    argument_parser.add_argument("--predictions", help="prediction file name", required=True)
    argument_parser.add_argument("--filetype", help="choose the file type between range and stream", required=True)
    argument_parser.add_argument("--graph", help="show graph of results")

    argument_parser.add_argument("--verbose", help="show detail results", action='store_true')
    argument_parser.add_argument("--theta", help="set parameter theta")
    argument_parser.add_argument("--alpha", help="set parameter alpha")
    argument_parser.add_argument("--delta", help="set parameter delta")
    arguments = argument_parser.parse_args()

    arguments = argument_parser.parse_args()
    theta, alpha, delta, graph = 0.5, 0.8, 600, 'none'  #default values
    if arguments.theta is not None:
        theta = float(arguments.theta)
    if arguments.alpha is not None:
        alpha = float(arguments.alpha)
    if arguments.delta is not None:
        delta = int(arguments.delta)
    if arguments.graph is not None:
        graph = arguments.graph

    assert(0.0 <= theta <= 1.0)
    assert(0.0 <= alpha <= 1.0)
    assert(isinstance(delta, int))
    assert(graph == 'screen' or graph == 'file' or graph == 'none' or graph == 'all')

    anomalies = File_IO.load_file(arguments.anomalies, arguments.filetype)
    predictions = File_IO.load_file(arguments.predictions, arguments.filetype)

    print_result(anomalies, predictions, alpha, theta, delta, arguments.verbose, graph)
