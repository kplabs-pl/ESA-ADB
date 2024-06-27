#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import math

from metrics.affiliation_based_metrics_repo.affiliation._single_ground_truth_event import \
    affiliation_precision_distance, affiliation_recall_distance, affiliation_precision_proba, affiliation_recall_proba

"""
Function `affiliation_precision_distance` is correct
"""
class Test_affiliation_precision_distance(unittest.TestCase):
    def test_generic(self):
        J = (1, 8)        
        self.assertEqual(affiliation_precision_distance([(1,3)], J), 0)
        self.assertEqual(affiliation_precision_distance([(1,8)], J), 0)
        self.assertEqual(affiliation_precision_distance([(1,9)], J), (1/2)/8)
        self.assertEqual(affiliation_precision_distance([(0,9)], J), 1/9)
        self.assertEqual(affiliation_precision_distance([(1,2), (3,4)], J), 0)
        self.assertEqual(affiliation_precision_distance([(7,9)], J), 1/4)
        self.assertEqual(affiliation_precision_distance([(8,9)], J), 1/2)
        self.assertEqual(affiliation_precision_distance([(9,10)], J), 3/2)        
        self.assertEqual(affiliation_precision_distance([(1,2),(9,10)], J), 3/4) 
       
        # previous tests
        # with pred=[6, 12) vs gt=[1,8): 0 at first, then int tdt from 0 to 4 (on [8, 12)), which is 8,
        # then divided by the length 12-6=6
        self.assertEqual(affiliation_precision_distance([(6,12)], (1,8)), 8/6)
        # with pred=[12, 18) vs gt=[16,17): 0 at the middle, on the left (4^2)/2, on the right (1^2)/2, sum is 8.5 
        # then divided by the length 6
        self.assertEqual(affiliation_precision_distance([(12,18)], (16,17)), 8.5/6)
    
    def test_empty(self):
        """
        With empty or None entries, return undefined (represented with math.nan)
        """
        J = (1, 8)
        self.assertTrue(math.isnan(affiliation_precision_distance([], J)))
        self.assertTrue(math.isnan(affiliation_precision_distance([None, None], J)))

    def test_paper(self):
        """
        Example of the paper
        """
        J = (0, 10*60)
        Is = [(5*60,6*60), (7*60,10*60), (11*60,12*60)]
        self.assertEqual(affiliation_precision_distance(Is, J), 18)

        J = (50*60, 70*60)
        Is = [(40*60, 60*60), (115*60,120*60)]
        self.assertEqual(affiliation_precision_distance(Is, J), 60*11.5)

        J = (170*60, 175*60)
        Is = [(120*60, 130*60), (135*60, 140*60), (165*60,170*60)]
        self.assertEqual(affiliation_precision_distance(Is, J), 60*31.25)

"""
Function `affiliation_recall_distance` is correct
"""
class Test_affiliation_recall_distance(unittest.TestCase):
    def test_generic(self):
        J = (1, 8)
        self.assertEqual(affiliation_recall_distance([(1,3)], J), 0*(2/7) + 2.5*(5/7))
        self.assertEqual(affiliation_recall_distance([(1,8)], J), 0)
        self.assertEqual(affiliation_recall_distance([(1,9)], J), 0)
        self.assertEqual(affiliation_recall_distance([(0,9)], J), 0)
    
    def test_empty(self):
        """
        With empty or None entries, return +inf (recall is always defined)
        but here there is no prediction in the zone, meaning that the recall
        is bad
        """
        J = (1, 8)
        self.assertEqual(affiliation_recall_distance([], J), math.inf)
        self.assertEqual(affiliation_recall_distance([None, None], J), math.inf)

    def test_paper(self):
        """
        Example of the paper
        """
        J = (0, 10*60)
        Is = [(5*60,6*60), (7*60,10*60), (11*60,12*60)]
        self.assertEqual(affiliation_recall_distance(Is, J), 76.5)

        J = (50*60, 70*60)
        Is = [(40*60, 60*60), (115*60,120*60)]
        self.assertEqual(affiliation_recall_distance(Is, J), 60*2.5)

        J = (170*60, 175*60)
        Is = [(120*60, 130*60), (135*60, 140*60), (165*60,170*60)]
        self.assertEqual(affiliation_recall_distance(Is, J), 60*2.5)

"""
Function `affiliation_precision_proba`
"""
class Test_affiliation_precision_proba(unittest.TestCase):
    def test_empty(self):
        """
        With empty or None entries, return undefined (represented with math.nan)
        """
        J = (1, 8)
        E = J
        self.assertTrue(math.isnan(affiliation_precision_proba([], J, E)))
        self.assertTrue(math.isnan(affiliation_precision_proba([None, None], J, E)))

    def test_paper(self):
        """
        Example of the paper
        """
        J = (50*60, 70*60)
        Is = [(40*60, 60*60), (115*60,120*60)]
        E = (30*60, 120*60)
        self.assertAlmostEqual(affiliation_precision_proba(Is, J, E), 0.672222222)

"""
Function `affiliation_recall_proba`
"""
class Test_affiliation_recall_proba(unittest.TestCase):
    def test_empty(self):
        """
        With empty or None entries, return 0
        """
        J = (1, 8)
        E = J
        self.assertEqual(affiliation_recall_proba([], J, E), 0)
        self.assertEqual(affiliation_recall_proba([None, None], J, E), 0)

    def test_paper(self):
        """
        Example of the paper
        """
        J = (50*60, 70*60)
        Is = [(40*60, 60*60), (115*60,120*60)]
        E = (30*60, 120*60)
        self.assertAlmostEqual(affiliation_recall_proba(Is, J, E), 0.944444444)

"""
Misc
"""
class Test_single_ground_truth_event_misc(unittest.TestCase):
    def test_generic(self):
        """
        Check accordance with previous values
        """
        E = (1, 90+1)
        Is = [(11,30+1),(86,90+1)]
        J = (21, 40+1)
        self.assertAlmostEqual(affiliation_recall_distance(Is, J), 2.5)
        self.assertAlmostEqual(affiliation_precision_distance(Is, J), 11.5)
        self.assertAlmostEqual(affiliation_recall_proba(Is, J, E), 0.944444444)
        self.assertAlmostEqual(affiliation_precision_proba(Is, J, E), 0.672222222)

if __name__ == '__main__':
    unittest.main()
