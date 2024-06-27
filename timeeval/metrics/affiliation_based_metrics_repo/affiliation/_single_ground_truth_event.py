#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from ._affiliation_zone import (
        get_all_E_gt_func,
        affiliation_partition)
from ._integral_interval import (
        integral_interval_distance,
        integral_interval_probaCDF_precision,
        integral_interval_probaCDF_recall,
        interval_length,
        sum_interval_lengths)

def affiliation_precision_distance(Is = [(1,2),(3,4),(5,6)], J = (2,5.5)):
    """
    Compute the individual average distance from Is to a single ground truth J

    :param Is: list of predicted events within the affiliation zone of J
    :param J: couple representating the start and stop of a ground truth interval
    :return: individual average precision directed distance number
    """
    if all([I is None for I in Is]): # no prediction in the current area
        return(math.nan) # undefined
    return(sum([integral_interval_distance(I, J) for I in Is]) / sum_interval_lengths(Is))

def affiliation_precision_proba(Is = [(1,2),(3,4),(5,6)], J = (2,5.5), E = (0,8)):
    """
    Compute the individual precision probability from Is to a single ground truth J

    :param Is: list of predicted events within the affiliation zone of J
    :param J: couple representating the start and stop of a ground truth interval
    :param E: couple representing the start and stop of the zone of affiliation of J
    :return: individual precision probability in [0, 1], or math.nan if undefined
    """
    if all([I is None for I in Is]): # no prediction in the current area
        return(0.5) # undefined
    return(sum([integral_interval_probaCDF_precision(I, J, E) for I in Is]) / sum_interval_lengths(Is))


def affiliation_specificity_distance(Is=[(1, 2), (3, 4), (5, 6)], J=(2, 5.5), E = (0,8)):
    """
    Compute the individual average distance from reverted Is to a single reverted ground truth J

    :param Is: list of predicted events within the affiliation zone of J
    :param J: couple representating the start and stop of a ground truth interval
    :param E: couple representing the start and stop of the zone of affiliation of J
    :return: individual average precision directed distance number
    """
    if all([I is None for I in Is]):
        reverted_Is = [E]
    else:
        Is = [I for I in Is if I is not None]  # filter possible None in Is
        reverted_Is = [(E[0], Is[0][0])]
        for i in range(1, len(Is)):
            reverted_Is.append((Is[i - 1][1], Is[i][0]))
        reverted_Is.append((Is[-1][1], E[1]))
        reverted_Is = [I for I in reverted_Is if I[1] - I[0] > 0]

    reverted_J = [(E[0], J[0]), (J[1], E[1])]
    reverted_J = [(x, y) for x, y in reverted_J if y-x > 0]

    calculation_zones = get_all_E_gt_func(reverted_Is, E)
    aff_partition = affiliation_partition(reverted_J, calculation_zones)

    return sum([sum([integral_interval_distance(rJ, I) for rJ in ap]) for ap, I in zip(aff_partition, reverted_Is)]) / sum_interval_lengths(reverted_J)

def affiliation_specificity_proba(Is=[(1, 2), (3, 4), (5, 6)], J=(2, 5.5), E = (0,8)):
    """
    Compute the individual average distance from reverted Is to a single reverted ground truth J

    :param Is: list of predicted events within the affiliation zone of J
    :param J: couple representating the start and stop of a ground truth interval
    :param E: couple representing the start and stop of the zone of affiliation of J
    :return: individual average precision directed distance number
    """
    if all([I is None for I in Is]):
        reverted_Is = [E]
    else:
        Is = [I for I in Is if I is not None]  # filter possible None in Is
        reverted_Is = [(E[0], Is[0][0])]
        for i in range(1, len(Is)):
            reverted_Is.append((Is[i - 1][1], Is[i][0]))
        reverted_Is.append((Is[-1][1], E[1]))
        reverted_Is = [I for I in reverted_Is if I[1] - I[0] > 0]

    reverted_J = [(E[0], J[0]), (J[1], E[1])]
    reverted_J = [(x, y) for x, y in reverted_J if y-x > 0]

    calculation_zones = get_all_E_gt_func(reverted_Is, E)
    aff_partition = affiliation_partition(reverted_J, calculation_zones)

    return sum([sum([integral_interval_probaCDF_recall(I, rJ, E) for rJ in ap]) for ap, I in zip(aff_partition, reverted_Is)]) / sum_interval_lengths(reverted_J)

def affiliation_recall_distance(Is = [(1,2),(3,4),(5,6)], J = (2,5.5)):
    """
    Compute the individual average distance from a single J to the predictions Is

    :param Is: list of predicted events within the affiliation zone of J
    :param J: couple representating the start and stop of a ground truth interval
    :return: individual average recall directed distance number
    """
    Is = [I for I in Is if I is not None] # filter possible None in Is
    if len(Is) == 0: # there is no prediction in the current area
        return(math.inf)
    E_gt_recall = get_all_E_gt_func(Is, (-math.inf, math.inf))  # here from the point of view of the predictions
    Js = affiliation_partition([J], E_gt_recall) # partition of J depending of proximity with Is
    Js = [J for J in Js if len(J) > 0]
    return(sum([integral_interval_distance(J[0], I) for I, J in zip(Is, Js)]) / interval_length(J))

def affiliation_recall_proba(Is = [(1,2),(3,4),(5,6)], J = (2,5.5), E = (0,8)):
    """
    Compute the individual recall probability from a single ground truth J to Is

    :param Is: list of predicted events within the affiliation zone of J
    :param J: couple representating the start and stop of a ground truth interval
    :param E: couple representing the start and stop of the zone of affiliation of J
    :return: individual recall probability in [0, 1]
    """
    Is = [I for I in Is if I is not None] # filter possible None in Is
    if len(Is) == 0: # there is no prediction in the current area
        return(0)
    E_gt_recall = get_all_E_gt_func(Is, E) # here from the point of view of the predictions
    Js = affiliation_partition([J], E_gt_recall) # partition of J depending of proximity with Is
    Js = [J for J in Js if len(J) > 0]
    return(sum([integral_interval_probaCDF_recall(I, J[0], E) for I, J in zip(Is, Js)]) / interval_length(J))
