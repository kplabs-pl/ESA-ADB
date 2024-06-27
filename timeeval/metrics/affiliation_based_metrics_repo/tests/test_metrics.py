#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import math

from metrics.affiliation_based_metrics_repo.affiliation.metrics import test_events
from metrics.affiliation_based_metrics_repo.affiliation import pr_from_events

"""
Function `test_events` prevents some mistakes as input
"""
class Test_test_events(unittest.TestCase):
    def test_generic(self):
        with self.assertRaises(TypeError):
            events = (1,3)
            test_events(events)
        with self.assertRaises(TypeError):
            events = [[1,3],[4,5]]
            test_events(events)
        with self.assertRaises(ValueError):
            events = [(1,3),(4,5,6)]
            test_events(events)
        with self.assertRaises(ValueError):
            events = [(1,3),(5,4)]
            test_events(events)
        with self.assertRaises(ValueError):
            events = [(4,6),(1,2)]
            test_events(events)
        with self.assertRaises(ValueError):
            events = [(4,6),(6,7)] # borders are not disjoint
            test_events(events)

"""
Function `pr_from_events`
"""
class Test_pr_from_events(unittest.TestCase):
    def test_empty(self):
        """
        With empty entries for predictions,
        the recall should be 0, and the predictions undefined
        (corresponding to resp. infinite and undefined distances)
        
        Note: It is not allowed to have events_gt empty
        """
        events_pred = []
        events_gt = [(1,10)]
        Trange = (1,10)
        results = pr_from_events(events_pred, events_gt, Trange)
        self.assertTrue(math.isnan(results['precision']))
        self.assertEqual(results['recall'], 0)
        
        self.assertEqual(len(results['individual_precision_probabilities']), 1)
        self.assertEqual(len(results['individual_recall_probabilities']), 1)
        self.assertEqual(len(results['individual_precision_distances']), 1)
        self.assertEqual(len(results['individual_recall_distances']), 1)
        
        self.assertEqual(results['individual_recall_distances'][0], math.inf)
        self.assertEqual(results['individual_recall_probabilities'][0], 0)
        self.assertTrue(math.isnan(results['individual_precision_distances'][0]))
        self.assertTrue(math.isnan(results['individual_precision_probabilities'][0]))

    def test_generic_precision_distance(self):
        """ Example 1 for precision distance """
        events_pred = [(1,3), (6,18), (25,26)]
        events_gt = [(1,8), (16,17), (25,28), (29,31)]
        Trange = (1, 31)
        results = pr_from_events(events_pred, events_gt, Trange)
        self.assertEqual(results['individual_precision_distances'],
                        [8/8, 8.5/6, 0/1, math.nan])
        # Explanation
        # For first ground truth, we group the elements of the first column 
        # for all predictions:
        # * Prediction 1: [1,3), which is fully inside [1,8) so the distance is 0
        # * Prediction 2: pred=[6, 12) vs gt1=[1,8) so distance of 8
        # * Prediction 3: not affiliated with gt1
        # In total, total distance is 8, for a total interval of 2+6=8, so the mean precision for gt1 is 8/8=1
        #
        # For second gt:
        # * Prediction 1: not affiliated with gt2
        # * Prediction 2: pred=[12, 18) vs gt=[16,17) so distance of 8.5
        # * Prediction 3: not affiliated with gt2
        # In total, total distance is 8.5, for a total interval of 6, so the mean precision for gt1 is 8.5/6
        # 
        # For third gt:
        # Only third prediction [25,26) is affiliated to the gt3=[25,28), and distance is 0 over a predicted
        # interval of 1, so the distance is 0/1
        #
        # For last gt:
        # No prediction on the affiliated interval, so distance is 0/0 = NaN

        """ Example 2 for precision distance with one gt and one pred only """
        events_pred = [(1,3), (5,10)]
        events_gt = [(1,8), (16,17)]
        Trange = (1, 31)
        results0 = pr_from_events(events_pred, events_gt, Trange)['individual_precision_distances']
        
        events_pred = [(1,3)]
        events_gt = [(1,8), (16,17)]
        results1 = pr_from_events(events_pred, events_gt, Trange)['individual_precision_distances']
        
        events_pred = [(1,3), (5,10)]
        events_gt = [(1,8)]
        results2 = pr_from_events(events_pred, events_gt, Trange)['individual_precision_distances']
        
        events_pred = [(2,3)]
        events_gt = [(1,8)]
        results3 = pr_from_events(events_pred, events_gt, Trange)['individual_precision_distances']

        self.assertEqual(results0, [(0+2**2/2)/7, math.nan])
        self.assertEqual(results1, [0, math.nan])
        self.assertEqual(results2, [results0[0]]) # answer is still a list, of length 1
        self.assertEqual(results3, [results1[0]])

    def test_generic_recall_distance(self):
        """ Example 1 for recall distance """
        events_pred = [(1,3), (6,18), (25,26)]
        events_gt = [(1,8), (16,17), (25,28), (29,31)]
        Trange = (1, 31)
        results = pr_from_events(events_pred, events_gt, Trange)
        self.assertEqual(results['individual_recall_distances'],
                        [2.25/7, 0/1, 2/3, math.inf])
        # Explanation
        #
        # For the first gt:
        # * Recall regarding prediction 1: gt1@aff_pred1=[1,(3+6)/2), and pred1=[1,3), so the distance is (4.5-3)^2/2
        # * Recall regarding prediction 2: gt1@aff_pred2=[(3+6)/2,8), and pred2=[6,18), so the distance is (6-4.5)^2/2
        # * gt1 is not affiliated to the other predictions
        # In total, total distance is (4.5-3)^2/2+(6-4.5)^2/2 = 2.25
        # And the length of gt1 is 7, hence the result.
        #
        # For the second gt:
        # * Recall regarding prediction 2: gt2@aff_pred2=gt2=[16,17), and pred2=[6,18), so the recall distance is 0
        # * gt2 is not affiliated to the other predictions
        # In total, total distance is 0
        # And the length of gt2 is 1, hence the result.
        #
        # For the third gt:
        # * Recall regarding prediction 3: gt3@aff_pred3=gt3=[25,28), and pred3=[25,26), so the recall distance is 2^2/2
        # * gt3 is not affiliated to the other predictions
        # In total, total distance is 2
        # And the length of gt3 is 3, hence the result.
        #
        # For the last gt, there is no affiliated prediction, so the recall distance
        # is infinite.

        """ Example 2 for recall distance with one gt and one pred only """
        events_pred = [(1,3), (5,10)]
        events_gt = [(1,8), (16,17)]
        Trange = (1, 31)
        results0 = pr_from_events(events_pred, events_gt, Trange)['individual_recall_distances']
        
        events_pred = [(1,3)]
        events_gt = [(1,8), (16,17)]
        results1 = pr_from_events(events_pred, events_gt, Trange)['individual_recall_distances']
        
        events_pred = [(1,3), (5,10)]
        events_gt = [(1,8)]
        results2 = pr_from_events(events_pred, events_gt, Trange)['individual_recall_distances']
        
        events_pred = [(1,3)]
        events_gt = [(1,8)]
        results3 = pr_from_events(events_pred, events_gt, Trange)['individual_recall_distances']

        self.assertEqual(results0, [(1**2/2+1**2/2)/7, math.inf])
        self.assertEqual(results1, [(5**2/2)/7, math.inf])
        self.assertEqual(results2, [results0[0]]) # answer is still a list, of length 1
        self.assertEqual(results3, [results1[0]])
        # Explanation:
        #   for results0:
        # gt1: 4 is the cut of affiliation of gt1 between pred1 and pred2, on both side distance from 0 to 1, for a gt1 of length 7
        # gt2: distance is outside the zone of affiliation, hence infinite
        # 
        #   for results1:
        # gt1: on [3,8), distance from 0 to 5, so 5^2/2, over a total length of gt1 of 7
        # gt2: infinite too
        #
        #   for results2:
        # gt1: like first part of results0
        #
        #   for results3:
        # gt1: like first part of results1
    
    def test_check_coherence(self):
        """
        Check coherence of the results in one example
        """
        events_pred = [(1,3), (6,18), (25,26)]
        events_gt = [(1,8), (16,17), (25,28), (29,31)]
        Trange = (1,40)
        results = pr_from_events(events_pred, events_gt, Trange)
        
        # around the third gt (25,28), only (25,26) is affiliated, with a
        # which is fully included, hence a precision probability of 1
        self.assertEqual(results['individual_precision_probabilities'][2], 1)

        # around the fourth gt (29,31), there is no prediction
        # hence a precision probability which is undefined
        # and also gives a recall probability of 0 (and a distance of math.inf)
        self.assertTrue(math.isnan(results['individual_precision_probabilities'][3]))
        self.assertEqual(results['individual_recall_probabilities'][3], 0)
        self.assertEqual(results['individual_recall_distances'][3], math.inf)
        
        # The second gt (16,17) is fully recalled by (6,18), so the recall is 1
        # and the corresponding distance is 0
        self.assertEqual(results['individual_recall_probabilities'][1], 1)
        self.assertEqual(results['individual_recall_distances'][1], 0)
        
    def test_paper(self):
        """
        Example of the paper
        """
        events_gt = [(0, 10*60), (50*60, 70*60), (170*60, 175*60)]
        events_pred = [(5*60,6*60), (7*60,10*60), (11*60,12*60), 
                       (40*60, 60*60), (115*60, 130*60), (135*60, 140*60), 
                       (165*60,170*60)]
        Trange = (0, 180*60)
        results = pr_from_events(events_pred, events_gt, Trange)
        self.assertAlmostEqual(results['individual_precision_distances'],
                               [18, 60*11.5, 60*31.25])
        self.assertAlmostEqual(results['individual_recall_distances'],
                               [76.5, 60*2.5, 60*2.5])
        self.assertAlmostEqual(results['individual_precision_probabilities'][1],
                               0.672222222)
        self.assertAlmostEqual(results['individual_recall_probabilities'][1],
                               0.944444444)

if __name__ == '__main__':
    unittest.main()
