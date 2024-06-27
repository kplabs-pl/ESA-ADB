#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import math

from metrics.affiliation_based_metrics_repo.affiliation.generics import convert_vector_to_events, infer_Trange, \
    has_point_anomalies, _sum_wo_nan, _len_wo_nan

"""
Function `convert_vector_to_events`
"""
class Test_convert_vector_to_events(unittest.TestCase):
    def test_empty(self):
        """
        Empty vector gives empty events
        """
        tested = convert_vector_to_events([])
        expected = []
        self.assertEqual(tested, expected)
        
    def test_zeros(self):
        """
        Zeros vector gives empty events
        """
        tested = convert_vector_to_events([0])
        expected = []
        self.assertEqual(tested, expected)
        
        tested = convert_vector_to_events([0, 0, 0, 0, 0])
        expected = []
        self.assertEqual(tested, expected)

    def test_ones(self):
        """
        Ones vector gives single long event
        """
        tested = convert_vector_to_events([1])
        expected = [(0, 1)] # of length of one index by convention
        self.assertEqual(tested, expected)
        
        tested = convert_vector_to_events([1, 1, 1, 1, 1])
        expected = [(0, 5)]
        self.assertEqual(tested, expected)
        
    def test_border(self):
        """
        Test with elements on the border
        """
        tested = convert_vector_to_events([1, 1, 0, 0, 0])
        expected = [(0, 2)]
        self.assertEqual(tested, expected)
        
        tested = convert_vector_to_events([0, 0, 1, 1, 1])
        expected = [(2, 5)]
        self.assertEqual(tested, expected)
        
        tested = convert_vector_to_events([1, 0, 1, 1, 1])
        expected = [(0, 1), (2, 5)]
        self.assertEqual(tested, expected)
        
        tested = convert_vector_to_events([1, 1, 0, 1, 0, 1, 1, 1])
        expected = [(0, 2), (3, 4), (5, 8)]
        self.assertEqual(tested, expected)

    def test_generic(self):
        """
        Test without elements on the border, generic case
        """
        tested = convert_vector_to_events([0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                                           1, 1, 0, 0, 0, 0, 0, 1, 1, 1,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        expected = [(6, 12), (17, 20), (60, 63)]
        self.assertEqual(tested, expected)

"""
Function `infer_Trange`
"""
class Test_infer_Trange(unittest.TestCase):
    def test_empty_predictions(self):
        """
        Infer Trange with empty predictions
        """
        tested = infer_Trange([], [(-1,2), (3,4), (6,20)])
        expected = (-1, 20)
        self.assertEqual(tested, expected)

    def test_empty_groundtruth(self):
        """
        Raise error for empty ground truth
        """
        with self.assertRaises(ValueError):
            infer_Trange([(-1,2), (3,4), (6,20)], [])

    def test_generic(self):
        """
        Infer Trange with generic predictions/groundtruth
        """
        tested = infer_Trange([(-3, 4), (5, 6)], [(-1,2), (3,4), (6,20)])
        expected = (-3, 20)
        self.assertEqual(tested, expected)
        
        tested = infer_Trange([(-3, 4), (5, 6)], [(-1,2), (3,4)])
        expected = (-3, 6)
        self.assertEqual(tested, expected)
        
        tested = infer_Trange([(0, 4), (5, 6)], [(-1,2), (3,4), (6,20)])
        expected = (-1, 20)
        self.assertEqual(tested, expected)
        
        tested = infer_Trange([(-1,2), (3,4), (6,20)], [(-3, 4), (5, 6)])
        expected = (-3, 20)
        self.assertEqual(tested, expected)

"""
Function `has_point_anomalies`
"""
class Test_has_point_anomalies(unittest.TestCase):
    def test_empty_event(self):
        """
        Test with an empty event
        """
        tested = has_point_anomalies([])
        expected = False # no event, so no point anomaly too
        self.assertEqual(tested, expected)

    def test_generic(self):
        """
        Check whether contain point anomalies with generic events
        """
        tested = has_point_anomalies([(1, 2)])
        expected = False
        self.assertEqual(tested, expected)
        
        tested = has_point_anomalies([(1, 2), (3, 4), (8, 10)])
        expected = False
        self.assertEqual(tested, expected)
        
        tested = has_point_anomalies([(1, 2), (3, 3), (8, 10)])
        expected = True
        self.assertEqual(tested, expected)
        
        tested = has_point_anomalies([(1, 1), (3, 3), (8, 8)])
        expected = True
        self.assertEqual(tested, expected)
        
        tested = has_point_anomalies([(1, 1)])
        expected = True
        self.assertEqual(tested, expected)

"""
Functions `_sum_wo_nan` and `_len_wo_nan`
"""
class Test_sum_len_wo_nan(unittest.TestCase):
    def test_empty_event(self):
        """
        Test with an empty event
        """
        tested = _sum_wo_nan([])
        expected = 0
        self.assertEqual(tested, expected)
        
        tested = _len_wo_nan([])
        expected = 0
        self.assertEqual(tested, expected)

    def test_generic(self):
        """
        Check with either math.nan or not
        """
        vec = [1, 4, 3]
        tested = _sum_wo_nan(vec)
        expected = 8
        self.assertEqual(tested, expected)
        tested = _len_wo_nan(vec)
        expected = 3
        self.assertEqual(tested, expected)
        
        vec = [1, math.nan, 3]
        tested = _sum_wo_nan(vec)
        expected = 4
        self.assertEqual(tested, expected)
        tested = _len_wo_nan(vec)
        expected = 2
        self.assertEqual(tested, expected)
        
        vec = [math.nan, math.nan, 3]
        tested = _sum_wo_nan(vec)
        expected = 3
        self.assertEqual(tested, expected)
        tested = _len_wo_nan(vec)
        expected = 1
        self.assertEqual(tested, expected)
        
        vec = [math.nan, math.nan, math.nan] # like an empty vec after removing math.nan
        tested = _sum_wo_nan(vec)
        expected = 0
        self.assertEqual(tested, expected)
        tested = _len_wo_nan(vec)
        expected = 0
        self.assertEqual(tested, expected)
        
if __name__ == '__main__':
    unittest.main()
