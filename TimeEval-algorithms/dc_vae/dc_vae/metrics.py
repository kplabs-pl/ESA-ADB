import numpy as np
from itertools import groupby
from operator import itemgetter


def interval_intersection(I=(1, 3), J=(2, 4)):
    """
    Intersection between two intervals I and J
    I and J should be either empty or represent a positive interval (no point)

    :param I: an interval represented by start and stop
    :param J: a second interval of the same form
    :return: an interval representing the start and stop of the intersection (or None if empty)
    """
    if I is None:
        return (None)
    if J is None:
        return (None)

    I_inter_J = (max(I[0], J[0]), min(I[1], J[1]))
    if I_inter_J[0] >= I_inter_J[1]:
        return (None)
    else:
        return (I_inter_J)


def convert_vector_to_events(vector=[0, 1, 1, 0, 0, 1, 0]):
    """
    Convert a binary vector (indicating 1 for the anomalous instances)
    to a list of events. The events are considered as durations,
    i.e. setting 1 at index i corresponds to an anomalous interval [i, i+1).

    :param vector: a list of elements belonging to {0, 1}
    :return: a list of couples, each couple representing the start and stop of
    each event
    """
    positive_indexes = [idx for idx, val in enumerate(vector) if val > 0]
    events = []
    for k, g in groupby(enumerate(positive_indexes), lambda ix: ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))

    # Consistent conversion in case of range anomalies (for indexes):
    # A positive index i is considered as the interval [i, i+1),
    # so the last index should be moved by 1
    events = [(x, y + 1) for (x, y) in events]

    return events


def per_event_f_score(y_true: np.ndarray, y_score: np.ndarray, beta=1.0):
    y_pred = y_score

    events_pred = convert_vector_to_events(y_pred)  # [(4, 5), (8, 9)]
    events_gt = convert_vector_to_events(y_true)  # [(3, 4), (7, 10)]

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_events_pred = [False for _ in events_pred]
    for gt in events_gt:
        for p, pred in enumerate(events_pred):
            if pred[0] > gt[1]:
                break
            if interval_intersection(pred, gt) is None:
                continue
            matched_events_pred[p] = True
            true_positives += 1
            break
        else:
            false_negatives += 1

    for p, pred in enumerate(events_pred):
        if matched_events_pred[p]:
            continue
        for gt in events_gt:
            if gt[0] > pred[1]:
                false_positives += 1
                break
            if interval_intersection(pred, gt) is not None:
                break
        else:
            false_positives += 1

    if true_positives + false_positives == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)
        # Correction from (Sehili et al., 2023) http://arxiv.org/abs/2308.13068
        nominal_samples = (y_true == 0)
        detected_samples = (y_pred == 1)
        false_positive_samples = (detected_samples & nominal_samples)
        precision *= (1 - sum(false_positive_samples) / sum(nominal_samples))

    if true_positives + false_negatives == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    divider = beta ** 2 * precision + recall
    if divider == 0.0:
        result = 0.0
    else:
        result = ((1 + beta ** 2) * precision * recall) / divider

    return result
