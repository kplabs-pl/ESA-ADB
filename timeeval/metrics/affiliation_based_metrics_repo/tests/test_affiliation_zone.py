#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import math

from metrics.affiliation_based_metrics_repo.affiliation._affiliation_zone import E_gt_func, get_all_E_gt_func, \
    affiliation_partition

"""
Function `E_gt_func` is correct, even for the borders
"""
class Test_E_gt_func(unittest.TestCase):
    def test_generic(self):
        Trange = (1, 30)
        events_gt = [(3,7), (10,18), (20,21)]
        j = 0
        self.assertEqual(E_gt_func(j, events_gt, Trange)[0], min(Trange))
        self.assertEqual(E_gt_func(j, events_gt, Trange)[1], (10+7)/2)
        j = 1
        self.assertEqual(E_gt_func(j, events_gt, Trange)[0], (10+7)/2)
        self.assertEqual(E_gt_func(j, events_gt, Trange)[1], (18+20)/2)
        j = 2
        self.assertEqual(E_gt_func(j, events_gt, Trange)[0], (18+20)/2)
        self.assertEqual(E_gt_func(j, events_gt, Trange)[1], max(Trange))
      
        # Case j = 1
        Trange = (1, 30)
        events_gt = [(3,20)]
        j = 0
        self.assertEqual(E_gt_func(j, events_gt, Trange)[0], min(Trange))
        self.assertEqual(E_gt_func(j, events_gt, Trange)[1], max(Trange))

"""
Function `get_all_E_gt_func` is correct
"""
class Test_get_all_E_gt_func(unittest.TestCase):
    def test_generic(self):
        Trange = (1, 30)
        events_gt = [(3,7), (10,18), (20,21)]
        cut_aff2 = get_all_E_gt_func(events_gt, Trange)
        self.assertEqual(cut_aff2[0], (min(Trange), (10+7)/2))
        self.assertEqual(cut_aff2[1], ((10+7)/2, (18+20)/2))
        self.assertEqual(cut_aff2[2], ((18+20)/2, max(Trange)))

"""
Function `affiliation_partition` is correct
"""
class Test_affiliation_partition(unittest.TestCase):
    def test_precision_direction(self):
        """
        Test of the function in the 'precision' direction I --> J  in one example
        """
        events_pred = [(1,3), (6,18), (25,26)]
        events_gt = [(1,8), (16,17), (25,28), (29,31)]
        Trange = (-math.inf, math.inf)
        E_gt = get_all_E_gt_func(events_gt, Trange)
        M = affiliation_partition(events_pred, E_gt)
        
        # Check of dimension of the lists
        self.assertEqual(len(M), len(events_gt))
        self.assertEqual(len(M[0]), len(events_pred))
        
        # First element, related to the first affiliation zone (-inf, 12)
        self.assertEqual(M[0][0], (1, 3)) # zone1, first prediction
        self.assertEqual(M[0][1], (6, 12)) # zone1, 2nd prediction
        self.assertEqual(M[0][2], None) # zone1, 3rd prediction
        
        # Second element, related to the second affiliation zone (12, 21)
        self.assertEqual(M[1][0], None) # zone2, first prediction
        self.assertEqual(M[1][1], (12, 18)) # zone2, 2nd prediction
        self.assertEqual(M[1][2], None) # zone2, 3rd prediction

        # Third element, related to the third affiliation zone (25, 28)
        self.assertEqual(M[2][0], None) # zone3, first prediction
        self.assertEqual(M[2][1], None) # zone3, 2nd prediction
        self.assertEqual(M[2][2], (25, 26)) # zone3, 3rd prediction
        
        # Fourth element, related to the fourth affiliation zone (29, 31)
        self.assertEqual(M[3][0], None) # zone4, first prediction
        self.assertEqual(M[3][1], None) # zone4, 2nd prediction
        self.assertEqual(M[3][2], None) # zone4, 3rd prediction

    def test_single_gt_and_pred(self):
        """
        Test of shape of the output of the function with only 
        one prediction and one ground truth intervals
        """
        Trange = (-math.inf, math.inf)
        
        events_pred = [(1,3), (5,10)]
        events_gt = [(1,8), (16,17)]
        E_gt = get_all_E_gt_func(events_gt, Trange)
        M0 = affiliation_partition(events_pred, E_gt)
        
        # One pred, more than one gt
        events_pred = [(1,3)]
        events_gt = [(1,8), (16,17)]
        E_gt = get_all_E_gt_func(events_gt, Trange)
        M1 = affiliation_partition(events_pred, E_gt)
        
        # One gt, more than one pred
        events_pred = [(1,3), (5,10)]
        events_gt = [(1,8)]
        E_gt = get_all_E_gt_func(events_gt, Trange)
        M2 = affiliation_partition(events_pred, E_gt)

        events_pred = [(2,3)]
        events_gt = [(1,8)]
        E_gt = get_all_E_gt_func(events_gt, Trange)
        M3 = affiliation_partition(events_pred, E_gt)
        
        # Check of dimension of the lists (here like gt)
        self.assertEqual(len(M0), 2)
        self.assertEqual(len(M1), 2)
        self.assertEqual(len(M2), 1)
        self.assertEqual(len(M3), 1)
        # Check of dimension of the lists (here like pred)
        self.assertEqual(len(M0[0]), 2)
        self.assertEqual(len(M1[0]), 1)
        self.assertEqual(len(M2[0]), 2)
        self.assertEqual(len(M3[0]), 1)
        
    def test_zero_pred(self):
        """
        Test of shape of the output of the function with no
        prediction
        """
        Trange = (-math.inf, math.inf)
        
        events_pred = []
        events_gt = [(1,8), (16,17)]
        E_gt = get_all_E_gt_func(events_gt, Trange)
        M = affiliation_partition(events_pred, E_gt)
        
        # Check of dimension of the lists
        self.assertEqual(len(M), len(events_gt))
        self.assertEqual(len(M[0]), len(events_pred)) # here with len(events_pred) == 0

if __name__ == '__main__':
    unittest.main()
