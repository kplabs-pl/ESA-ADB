#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

from metrics.affiliation_based_metrics_repo.affiliation._integral_interval import interval_length, sum_interval_lengths, \
    interval_intersection, interval_subset, cut_into_three_func, get_pivot_j, integral_mini_interval, \
    integral_interval_distance, integral_mini_interval_P_CDFmethod__min_piece, \
    integral_mini_interval_Pprecision_CDFmethod, integral_interval_probaCDF_precision, cut_J_based_on_mean_func, \
    integral_interval_probaCDF_recall

"""
Function `interval_length`
"""
class Test_interval_length(unittest.TestCase):
    def test_empty(self):
        """
        None interval gives 0 length
        """
        tested = interval_length(None)
        expected = 0
        self.assertEqual(tested, expected)
        
    def test_generic(self):
        """
        Test correct length of the interval
        """
        tested = interval_length((1, 2))
        expected = 1
        self.assertEqual(tested, expected)
        
        tested = interval_length((-1, 3.5))
        expected = 4.5
        self.assertEqual(tested, expected)

"""
Function `sum_interval_lengths`
"""
class Test_sum_interval_lengths(unittest.TestCase):
    def test_empty(self):
        """
        Empty event gives 0 length
        """
        tested = sum_interval_lengths([])
        expected = 0
        self.assertEqual(tested, expected)
        
    def test_generic(self):
        """
        Test correct sum of length of the intervals
        """
        tested = sum_interval_lengths([(1, 2)])
        expected = 1
        self.assertEqual(tested, expected)
        
        tested = sum_interval_lengths([(1, 2), (3.5, 4)])
        expected = 1+0.5
        self.assertEqual(tested, expected)

"""
Function `interval_intersection`
"""
class Test_interval_intersection(unittest.TestCase):
    def test_empty(self):
        """
        None when one or more interval is None
        """
        tested = interval_intersection(None, None)
        expected = None
        self.assertEqual(tested, expected)
        
        tested = interval_intersection(None, (1,2))
        expected = None
        self.assertEqual(tested, expected)
        
        tested = interval_intersection((1,2), None)
        expected = None
        self.assertEqual(tested, expected)
        
    def test_generic(self):
        """
        Test correct intersection of the intervals
        """
        tested = interval_intersection((1, 2), (1, 2))
        expected = (1, 2)
        self.assertEqual(tested, expected)
        
        tested = interval_intersection((1, 2), (2, 3))
        expected = None # because intersection or [1, 2) and [2, 3)
        self.assertEqual(tested, expected)
        
        tested = interval_intersection((1, 2), (3, 4))
        expected = None
        self.assertEqual(tested, expected)
        
        tested = interval_intersection((1, 3), (2, 4))
        expected = (2, 3)
        self.assertEqual(tested, expected)
        
        tested = interval_intersection((2, 4), (1, 3))
        expected = (2, 3)
        self.assertEqual(tested, expected)
        
        tested = interval_intersection((-1, 5), (1, 3))
        expected = (1, 3)
        self.assertEqual(tested, expected)
                
        tested = interval_intersection((1, 3), (-1, 5))
        expected = (1, 3)
        self.assertEqual(tested, expected)
        
        tested = interval_intersection((1, 10), (0, 5))
        expected = (1, 5)
        self.assertEqual(tested, expected)

"""
Function `interval_subset`
"""
class Test_interval_subset(unittest.TestCase):
    def test_empty(self):
        """
        Error for empty interval
        """
        with self.assertRaises(TypeError):
            interval_subset(None, None)
        
        with self.assertRaises(TypeError):
            interval_subset(None, (1,2))
            
        with self.assertRaises(TypeError):
            interval_subset((1,2), None)
        
    def test_generic(self):
        """
        Test correct check of subset
        """
        tested = interval_subset((1, 2), (1, 2))
        expected = True
        self.assertEqual(tested, expected)
        
        tested = interval_subset((1, 2), (1, 3))
        expected = True
        self.assertEqual(tested, expected)

        tested = interval_subset((1, 2), (0, 3))
        expected = True
        self.assertEqual(tested, expected)
        
        tested = interval_subset((1, 3), (2, 3))
        expected = False
        self.assertEqual(tested, expected)

        tested = interval_subset((1, 3), (-1, 2))
        expected = False
        self.assertEqual(tested, expected)
        
        tested = interval_subset((1, 3), (-1, 0))
        expected = False
        self.assertEqual(tested, expected)
        
"""
Function `cut_into_three_func`
"""
class Test_cut_into_three_func(unittest.TestCase):
    def test_examples(self):
        """
        Example 1
        """
        I = (0, 1.5)
        J = (1, 2)
        tested = cut_into_three_func(I, J)
        self.assertEqual(len(tested), 3)
        self.assertEqual(tested[0], (0, 1))
        self.assertEqual(tested[1], (1, 1.5))
        self.assertEqual(tested[2], None)
        
        """
        Example 2 with elements both before and after
        """
        I = (-1, 10)
        J = (1.4, 2.4)
        tested = cut_into_three_func(I, J)
        self.assertEqual(len(tested), 3)
        self.assertEqual(tested[0], (-1, 1.4))
        self.assertEqual(tested[1], (1.4, 2.4))
        self.assertEqual(tested[2], (2.4, 10))
               
        """
        Example 3 with only elements before
        """
        I = (-1, 1)
        J = (1.4, 2.4)
        tested = cut_into_three_func(I, J)
        self.assertEqual(len(tested), 3)
        self.assertEqual(tested[0], (-1, 1))
        self.assertEqual(tested[1], None)
        self.assertEqual(tested[2], None)
        
        """
        Example 4 with only elements at middle
        """
        I = (1.6, 2)
        J = (1.4, 2.4)
        tested = cut_into_three_func(I, J)
        self.assertEqual(len(tested), 3)
        self.assertEqual(tested[0], None)
        self.assertEqual(tested[1], (1.6, 2))
        self.assertEqual(tested[2], None)

        """
        Example 5 with only elements after
        """
        I = (4, 5)
        J = (1.4, 2.4)
        tested = cut_into_three_func(I, J)
        self.assertEqual(len(tested), 3)
        self.assertEqual(tested[0], None)
        self.assertEqual(tested[1], None)
        self.assertEqual(tested[2], (4,5))

"""
Function `get_pivot_j`
"""
class Test_get_pivot_j(unittest.TestCase):
    def test_examples(self):
        """
        Examples
        """
        I = (4, 5)
        J = (1.4, 2.4)
        self.assertEqual(get_pivot_j(I, J), 2.4) # max(J)
        
        I = (0, 1)
        J = (1.4, 2.4)
        self.assertEqual(get_pivot_j(I, J), 1.4) # min(J)
        
        I = (0, 1.5)
        J = (1.4, 2.4)
        with self.assertRaises(ValueError):
            # intersection I inter J is not void
            get_pivot_j(I, J)

"""
Function `integral_mini_interval`
"""
class Test_integral_mini_interval(unittest.TestCase):
    def test_examples(self):
        """
        Examples
        """
        I = (4, 5)
        J = (1.4, 2.4)
        # We look at sum distance between every element of [4,5] to 2.4 the closest element of J
        # Distance is going from 4-2.4 to 5-2.4 i.e. 1.6 to 2.6, and increases linearly
        # There is 1.6 with a time duration of 1, and in addition the triangle 1/2 (integral from 0 to 1 of tdt)
        # Globally 1.6+1/2
        self.assertEqual(integral_mini_interval(I, J), 1.6 + 1/2)
        
        I = (0.1, 1.2)
        J = (1.4, 2.4)
        # We look at sum distance between every element of [0.1,1.2] to 1.4 the closest element of J
        # Distance is going from 1.3 to 0.2 and decreases linearly
        # There is 0.2 with a time duration of deltaI=1.1, and in addition
        # a decreases from 1.1 to 0 during 1.1 (integral from 0 to 1.1 of tdt) which is 1.1^2/2
        # Globally 0.2*1.1+1.1^2/2
        self.assertAlmostEqual(integral_mini_interval(I, J), 0.2*1.1+1.1**2/2)
        
        I = (0, 1.5)
        J = (1.4, 2.4)
        with self.assertRaises(ValueError):
            # intersection I inter J is not void
            integral_mini_interval(I, J)

"""
Function `integral_interval`
"""
class Test_integral_interval(unittest.TestCase):
    def test_examples(self):
        """
        Function integral_interval *for distance* verifies some tests
        """
        ## For I included in J, it's 0
        I = (0, 1.5)
        J = (-1, 2.4)
        self.assertEqual(integral_interval_distance(I, J), 0)
        
        J = (-1, 2.4)
        I = J
        self.assertEqual(integral_interval_distance(I, J), 0)
        
        ## The integral is same from I or I\J
        I = (-10, 1.5)
        J = (-1, 2.4)
        I_minus_J = (-10, -1) # I \ J
        self.assertEqual(integral_interval_distance(I, J),
                         integral_interval_distance(I_minus_J, J))
        
        # previous test
        I = (-10, 20)
        J = (-1, 2.4)
        self.assertEqual(integral_interval_distance(I, J),
                         195.38)

"""
Function `integral_mini_interval_P_CDFmethod__min_piece`
"""
class Test_integral_mini_interval_P_CDFmethod__min_piece(unittest.TestCase):
    def test_examples(self):
        """
        Check this component of `integral_mini_interval_Pprecision_CDFmethod`
        in three cases, by recomputing the formulas
        
        It is three cases with I totally outside J
        I = (i_min, i_max)
        J = (j_min, j_max)
        E = (e_min, e_max)
        """        
        # Closed-form for $C = \int_{d_min}^{d_max} \min(m, x) dx$
        # Case 1: $d_max <= m$:
        # C = \int_{d_min}^{d_max} x dx = (1/2)*(d_max^2 - d_min^2)
        # Case 2: $d_min < m < d_max$:
        # C = \int_{d_min}^{m} x dx + \int_{m}^{d_max} m dx
        #   = (1/2)*(m^2 - d_min^2) + m (d_max - m)
        # Case 3: $m <= d_min$:
        # C = \int_{d_min}^{d_max} m dx = m (d_max - d_min)
        #
        # For the combinated of all three cases, we see first that:
        # A = min(d_max,m)^2 - min(d_min, m)^2 is: d_max^2 - d_min^2 (case1); m^2 - d_min^2 (case2); 0           (case3)
        # and then that:
        # B = max(d_max, m) - max(d_min, m)    is: 0 (case1);                 d_max - m (case2);     d_max-d_min (case3)
        # so that:
        # C = (1/2)*A + m*B
        # It is checked below, for each case C1, C2, C3:

        """ Case 1 """
        e_min = 0.7176185
        j_min = 1.570739
        j_max = 1.903998
        e_max = 2.722883
        i_min = 0.924204
        i_max = 1.376826
        d_min = max(i_min - j_max, j_min - i_max) # 0.1939125
        d_max = max(i_max - j_max, j_min - i_min) # 0.6465346
        m = min(j_min - e_min, e_max - j_max) # 0.8188856
        C_case1 = (1/2)*(d_max - d_min)*(d_max + d_min) # 0.1902024
        C_case2 = (1/2)*(m**2 - d_min**2) + m * (d_max - m) # 0.17535
        C_case3 = (d_max - d_min)*m # 0.3706457
        A = min(d_max, m)**2 - min(d_min, m)**2 # 0.3804049
        B = max(d_max, m) - max(d_min, m) # 0
        C = (1/2)*A + m*B # 0.1902024
        # Actual test
        I = (i_min, i_max)
        J = (j_min, j_max)
        E = (e_min, e_max)
        self.assertTrue(d_max <= m) # it is the case 1
        self.assertEqual(C_case1, C)
        self.assertEqual(integral_mini_interval_P_CDFmethod__min_piece(I, J, E), C)
  
        """ Case 2 """
        e_min = 0.3253522
        j_min = 0.5569796
        j_max = 0.8238064
        e_max = 1.403741
        i_min = 0.8751017
        i_max = 1.116294
        d_min = max(i_min - j_max, j_min - i_max) # 0.05129532
        d_max = max(i_max - j_max, j_min - i_min) # 0.2924877
        m = min(j_min - e_min, e_max - j_max) # 0.2316275
        C_case1 = (1/2)*(d_max - d_min)*(d_max + d_min) # 0.04145893
        C_case2 = (1/2)*(m**2 - d_min**2) + m * (d_max - m) # 0.03960695
        C_case3 = (d_max - d_min)*m # 0.05586679
        A = min(d_max, m)**2 - min(d_min, m)**2 # 0.05102008
        B = max(d_max, m) - max(d_min, m) # 0.06086027
        C = (1/2)*A + m*B # 0.03960695
        # Actual test
        I = (i_min, i_max)
        J = (j_min, j_max)
        E = (e_min, e_max)
        self.assertTrue(d_min < m) # it is the case 2
        self.assertTrue(m < d_max) # it is the case 2
        self.assertEqual(C_case2, C)
        self.assertEqual(integral_mini_interval_P_CDFmethod__min_piece(I, J, E), C)

        """ Case 3 """
        e_min = 0.6516738
        j_min = 1.523338
        j_max = 1.958426
        e_max = 2.435003
        i_min = 0.767282
        i_max = 0.7753016
        d_min = max(i_min - j_max, j_min - i_max) # 0.7480365
        d_max = max(i_max - j_max, j_min - i_min) # 0.7560561
        m = min(j_min - e_min, e_max - j_max) # 0.4765765
        C_case1 = (1/2)*(d_max - d_min)*(d_max + d_min) # 0.006031113
        C_case2 = (1/2)*(m**2 - d_min**2) + m * (d_max - m) # -0.03302331
        C_case3 = (d_max - d_min)*m # 0.003821954
        A = min(d_max, m)**2 - min(d_min, m)**2 # 0
        B = max(d_max, m) - max(d_min, m) # 0.008019604
        C = (1/2)*A + m*B # 0.003821954
        # Actual test
        I = (i_min, i_max)
        J = (j_min, j_max)
        E = (e_min, e_max)
        self.assertTrue(m <= d_min) # it is the case 3
        self.assertEqual(C_case3, C)
        self.assertEqual(integral_mini_interval_P_CDFmethod__min_piece(I, J, E), C)

"""
Function `integral_mini_interval_Pprecision_CDFmethod`
"""
class Test_integral_mini_interval_Pprecision_CDFmethod(unittest.TestCase):
    def test_symmetric(self):
        """
        Check function `integral_mini_interval_Pprecision_CDFmethod`
        in the symmetric case i.e. J is centered on E,
        and when I goes from the min of E to the max of E.
        
        Check in three cases
        """    
        lists = [dict({'e_min': 0.2655087,
                        'j_min': 0.9202326,
                        'j_max': 1.187741,
                        'e_max': 1.842465}),
                 dict({'e_min': 0.3721239,
                        'j_min': 0.7253212,
                        'j_max': 0.9439665,
                        'e_max': 1.297164}),
                 dict({'e_min': 0.5728534,
                        'j_min': 0.8431135,
                        'j_max': 1.35991,
                        'e_max': 1.63017})]

        for my_dict in lists:
            e_min = my_dict['e_min']
            j_min = my_dict['j_min']
            j_max = my_dict['j_max']
            e_max = my_dict['e_max']
            
            # on the left
            i_min_left = e_min
            i_max_left = j_min
            # on the right
            i_min_right = j_max
            i_max_right = e_max        
            m = min(j_min - e_min, e_max - j_max)
            M = max(j_min - e_min, e_max - j_max) # same because symmetric
    
            # Actual test
            I_left = (i_min_left, i_max_left)
            I_right = (i_min_right, i_max_right)
            J = (j_min, j_max)
            E = (e_min, e_max)
            integral_left = integral_mini_interval_Pprecision_CDFmethod(I_left, J, E)
            integral_middle = max(J) - min(J)
            integral_right = integral_mini_interval_Pprecision_CDFmethod(I_right, J, E)
            m = min(J) - min(E)
            M = max(E) - max(J)
            DeltaJ = max(J) - min(J)
            DeltaE = max(E) - min(E)
            self.assertAlmostEqual((1-DeltaJ/DeltaE)*m/2, integral_left)
            self.assertAlmostEqual(DeltaJ, integral_middle)
            self.assertAlmostEqual((1-DeltaJ/DeltaE)*M/2, integral_right)
            # Explanation:
            # In case of symmetry the value is 1 for elements in J,
            # outside, it goes from (1 - DeltaJ/DeltaE) the closer to J,
            # until 0 at min(E) and max(E).
            # Since it's symmetric it decreases always linearly, on both side
            # It is (1 - DeltaJ/DeltaE) and not 1 as the border of J because
            # there is already DeltaJ/DeltaE of the probability took on the interval J
            #
            # So e.g. on the left, it's a triangle of height (1 - DeltaJ/DeltaE) and length
            # m (or M, it's the same since it's symmetric), so the answer.
        
    def test_almost_point(self):
        """
        Check a property of the function 
        `integral_mini_interval_Pprecision_CDFmethod` in the almost point 
        case, i.e. J of duration 1e-9
        Check in three cases
        """    
        lists = [dict({'e_min': 0.2655087,
                        'j_min': 0.9202326,
                        'e_max': 1.842465}),
                 dict({'e_min': 0.3721239,
                        'j_min': 0.7253212,
                        'e_max': 1.297164}),
                 dict({'e_min': 0.5728534,
                        'j_min': 0.8431135,
                        'e_max': 1.63017})]

        for my_dict in lists:
            e_min = my_dict['e_min']
            j_min = my_dict['j_min']
            j_max = j_min + 1e-9 # almost point case
            e_max = my_dict['e_max']
            
            # on the left
            i_min_left = e_min
            i_max_left = j_min
            # on the right
            i_min_right = j_max
            i_max_right = e_max        

            # Actual test
            I_left = (i_min_left, i_max_left)
            I_right = (i_min_right, i_max_right)
            J = (j_min, j_max)
            E = (e_min, e_max)
            integral_left = integral_mini_interval_Pprecision_CDFmethod(I_left, J, E)
            # integral_middle = max(J) - min(J)
            integral_right = integral_mini_interval_Pprecision_CDFmethod(I_right, J, E)
            DeltaE = max(E) - min(E)
            self.assertAlmostEqual((integral_left + integral_right)/DeltaE, 1/2)
            # Explanation: for point anomaly, the mean value should be 1/2

"""
Function `integral_interval_probaCDF_precision`
"""
class Test_integral_interval_probaCDF_precision(unittest.TestCase):
    def test_basics(self):
        """
        Some tests *for proba_CDF precision* integral
        """
        ## For I close to the border of E, it's close to 0%
        # (after taking the mean i.e. dividing by |I|)
        E = (-3, 3)
        J = (-1, 2.4)
        x = -2.5
        I1 = (-3, x)
        DeltaI1 = max(I1) - min(I1)
        self.assertTrue(integral_interval_probaCDF_precision(I1, J, E) / DeltaI1 < 0.05)

        x = -2.8
        I2 = (-3, x)
        DeltaI2 = max(I2) - min(I2)
        self.assertTrue(integral_interval_probaCDF_precision(I2, J, E) / DeltaI2 < integral_interval_probaCDF_precision(I1, J, E) / DeltaI1)

        x = -2.99
        I3 = (-3, x)
        DeltaI3 = max(I3) - min(I3)
        self.assertTrue(integral_interval_probaCDF_precision(I3, J, E) / DeltaI3 < integral_interval_probaCDF_precision(I2, J, E) / DeltaI2)
  
    def test_closed(self):
        """
        proba_CDF precision integral verifies closed form integral when I=E
        """
        def closed_form_for_I_equals_to_E_proba_CDF(J, E):
            # The total integral (when I is the whole interval E) is given by the sum:
            # I = (1-DeltaJ/DeltaE)*m/2 + (1-DeltaJ/DeltaE)*M/2 + DeltaJ
            # and M+m = DeltaE - DeltaJ so
            # I = (1-DeltaJ/DeltaE)*(DeltaE - DeltaJ)/2 + DeltaJ
            #   = (DeltaE - DeltaJ - DeltaJ + DeltaJ^2/DeltaE + 2*DeltaJ)/2         (*)
            #   = (DeltaE + DeltaJ^2/DeltaE)/2
            DeltaE = max(E) - min(E)
            DeltaJ = max(J) - min(J)
            return((DeltaE + DeltaJ**2/DeltaE)/2)

        E = (-3, 3)
        J = (-1, 2.4)
        I = E
        self.assertAlmostEqual(integral_interval_probaCDF_precision(I, J, E),
                               closed_form_for_I_equals_to_E_proba_CDF(J, E))

        E = (-10, 3)
        J = (0, 2.9)
        I = E
        self.assertAlmostEqual(integral_interval_probaCDF_precision(I, J, E),
                               closed_form_for_I_equals_to_E_proba_CDF(J, E))

"""
Function `cut_J_based_on_mean_func`
"""
class Test_cut_J_based_on_mean_func(unittest.TestCase):
    def test_generic(self):
        J = None
        e_mean = 1.5
        tested = cut_J_based_on_mean_func(J, e_mean)
        expected = (None, None)
        self.assertEqual(tested, expected)
        
        J = (2, 3)
        e_mean = 1.5
        tested = cut_J_based_on_mean_func(J, e_mean)
        expected = (None, J)
        self.assertEqual(tested, expected)
        
        J = (0, 1)
        e_mean = 1.5
        tested = cut_J_based_on_mean_func(J, e_mean)
        expected = (J, None)
        self.assertEqual(tested, expected)
        
        J = (0, 5)
        e_mean = 1.5
        tested = cut_J_based_on_mean_func(J, e_mean)
        expected = ((0, 1.5), (1.5, 5))
        self.assertEqual(tested, expected)
        
        J = (0, 1.5)
        e_mean = 1.5
        tested = cut_J_based_on_mean_func(J, e_mean)
        expected = (J, None)
        self.assertEqual(tested, expected)
        
        J = (1.5, 2)
        e_mean = 1.5
        tested = cut_J_based_on_mean_func(J, e_mean)
        expected = (None, J)
        self.assertEqual(tested, expected)


"""
Functions `integral_interval_probaCDF_recall` and `integral_mini_interval_Precall_CDFmethod`
"""
class Test_integral_interval_probaCDF_recall(unittest.TestCase):
    def test_almost_point(self):
        """
        Check a property of the function 
        `integral_interval_probaCDF_recall` in the almost point 
        case, i.e. when both I and J are both almost-point anomalies
        Check in three cases
        """    
        size_event = 1e-9 # almost a point anomaly
        # J is an interval of length 2*size_event
        # I is also an interval of 2*length size_event
        # E is a (longer) interval
        # The recall of J from I should be 1 when I is close to J, then decrease to 0 when I is closer and closer to E,
        # and keep to be 0 outside E.
        
        ## We take J at the middle of E
        E = (1, 3)
        J = (2-size_event, 2+size_event)
        DeltaJ = max(J) - min(J) # divide by J the size to obtain the mean
      
        # a. I is at position J, so the recall should be 1
        I = (2-size_event, 2+size_event)
        self.assertAlmostEqual(integral_interval_probaCDF_recall(I, J, E) / DeltaJ, 1)
      
        # b. I is close to J, so the recall should be high
        I = (1.98-size_event, 1.98+size_event)
        self.assertTrue(integral_interval_probaCDF_recall(I, J, E) / DeltaJ > 0.95)
      
        # c. I is at middle between max(E) and min(J), so the recall should be 0.5
        I = (1.5-size_event, 1.5+size_event)
        self.assertAlmostEqual(integral_interval_probaCDF_recall(I, J, E) / DeltaJ, 0.5)
        # c'. Same for I at the other side
        I = (2.5-size_event, 2.5+size_event)
        self.assertAlmostEqual(integral_interval_probaCDF_recall(I, J, E) / DeltaJ, 0.5)
      
        # d. I is close to the edge of E, the recall should be low
        I = (1.01-size_event, 1.01+size_event)
        self.assertTrue(integral_interval_probaCDF_recall(I, J, E) / DeltaJ < 0.1)
        I = (2.99-size_event, 2.99+size_event)
        self.assertTrue(integral_interval_probaCDF_recall(I, J, E) / DeltaJ < 0.1)
      
        # e. I is at the edge of E, the recall should be 0
        I = (1-size_event, 1+size_event)
        self.assertAlmostEqual(integral_interval_probaCDF_recall(I, J, E) / DeltaJ, 0)
        I = (3-size_event, 3+size_event)
        self.assertAlmostEqual(integral_interval_probaCDF_recall(I, J, E) / DeltaJ, 0)
      
        # f. I is outside E, the recall should be 0
        I = (-4-size_event, -4+size_event)
        self.assertAlmostEqual(integral_interval_probaCDF_recall(I, J, E) / DeltaJ, 0)
        I = (10-size_event, 10+size_event)
        self.assertAlmostEqual(integral_interval_probaCDF_recall(I, J, E) / DeltaJ, 0)
      
    def test_partially_almost_point(self):
        """
        Check the recall probability when J is almost-point anomaly 
        and I is growing
        """    
        size_event = 1e-9 # almost a point anomaly
        # J is an interval of length 2*size_event
        # E is a (longer) interval
        # The recall of J from I should be 1 when I is close to J, then decrease to 0 when I is closer and closer to E,
        # and keep to be 0 outside E.
      
        # In the following, the pivot is has in the previous test, so it does not change anything
        # to have I not a point anomaly for the recall
      
        ## We take J at the middle of E
        E = (1, 3)
        J = (2-size_event, 2+size_event)
        DeltaJ = max(J) - min(J) # divide by J the size to obtain the mean
      
        # J is included in I, so the recall should be 1
        I = (1-size_event, 3+size_event)
        self.assertAlmostEqual(integral_interval_probaCDF_recall(I, J, E) / DeltaJ, 1)
      
        #  I is close to J, so the recall should be high
        I = (1-size_event, 1.98+size_event)
        self.assertTrue(integral_interval_probaCDF_recall(I, J, E) / DeltaJ > 0.95)
      
        # c. I is at middle between max(E) and min(J), so the recall should be 0.5
        I = (1-size_event, 1.5+size_event)
        self.assertAlmostEqual(integral_interval_probaCDF_recall(I, J, E) / DeltaJ, 0.5)
        # c'. Same for I at the other side
        I = (2.5-size_event, 3+size_event)
        self.assertAlmostEqual(integral_interval_probaCDF_recall(I, J, E) / DeltaJ, 0.5)
      
        # d. I is close to the edge of E, the recall should be low
        I = (1-size_event, 1.01+size_event)
        self.assertTrue(integral_interval_probaCDF_recall(I, J, E) / DeltaJ < 0.1)
        I = (2.99-size_event, 3+size_event)
        self.assertTrue(integral_interval_probaCDF_recall(I, J, E) / DeltaJ < 0.1)
      
        # e. I is at the edge of E, the recall should be 0
        I = (0-size_event, 1+size_event)
        self.assertAlmostEqual(integral_interval_probaCDF_recall(I, J, E) / DeltaJ, 0)
        I = (3-size_event, 5+size_event)
        self.assertAlmostEqual(integral_interval_probaCDF_recall(I, J, E) / DeltaJ, 0)
      
    def test_special_cases(self):
        """
        Check that when J is as large as E and I is a point-anomaly 
        at the middle of J, the recall is 0.625, 
        and when I is at the border, the recall is 0.25
        (p = 1 in the article)
        """    
        size_event = 1e-9 # almost a point anomaly
        x_center = 2
        delta = 1
        E = (x_center - delta, x_center + delta)
        J = (x_center - delta, x_center + delta)
        DeltaJ = max(J) - min(J) # divide by J the size to obtain the mean
      
        I = (x_center-size_event, x_center+size_event)
        p_recall = integral_interval_probaCDF_recall(I, J, E) / DeltaJ # it's the constant 5/8 when |J|=|E|
        self.assertAlmostEqual(p_recall, 0.625) # 5/8 == 0.625
      
        I = (x_center - delta-size_event, x_center - delta+size_event)
        p_recall = integral_interval_probaCDF_recall(I, J, E) / DeltaJ
        self.assertAlmostEqual(p_recall, 0.25)
      
        I = (x_center + delta - size_event, x_center + delta +size_event)
        p_recall = integral_interval_probaCDF_recall(I, J, E) / DeltaJ
        self.assertAlmostEqual(p_recall, 0.25)

    def test_behavior_when_I_increases(self):
        """
        Check that recall is better and better for a prediction I 
        centered in the middle of J that grows symmetrically
        """    
        size_event = 1e-9
        E = (-5, 5)
        J = (-3, 3)
        DeltaJ = max(J) - min(J) # divide by J the size to obtain the mean
        I = (0-size_event, 0+size_event)
        # 0.708 in that case, because more possibility for a random pred to miss the gt event
        # compared to the 0.625 constant when |E|=|J|
        self.assertTrue(integral_interval_probaCDF_recall(I, J, E) / DeltaJ > 0.625)
      
        I2 = (-2, 2) # 0.9666 in that case
        self.assertTrue(integral_interval_probaCDF_recall(I2, J, E) / DeltaJ > 0.625)
      
        I1 = (-1, 1) # 0.8666 in that case
        self.assertTrue(integral_interval_probaCDF_recall(I1, J, E) / DeltaJ > 0.625)
      
        # Better recall for I2 compared to I1
        self.assertTrue(integral_interval_probaCDF_recall(I2, J, E) / DeltaJ > integral_interval_probaCDF_recall(I1, J, E) / DeltaJ)
      
        # Better recall for I29 compared to I2
        I29 = (-2.9, 2.9) # 0.999666
        self.assertTrue(integral_interval_probaCDF_recall(I29, J, E) / DeltaJ > integral_interval_probaCDF_recall(I2, J, E) / DeltaJ)
        
    def test_behavior_when_E_increases(self):
        """
        Check that recall goes to 1 when |E| increases to the right 
        without chaning I
        """    
        size_event = 1e-9
        J = (-3, 3)
        DeltaJ = max(J) - min(J) # divide by J the size to obtain the mean
        I = (10, 10+size_event)
      
        # |E| is growing to the right until infinity, recall should be better and better
        E10 = (-10, 10)
        integral_interval_probaCDF_recall(I, J, E10) / DeltaJ # 0
        E12 = (-10, 12)
        integral_interval_probaCDF_recall(I, J, E12) / DeltaJ # 0.1590909
        E18 = (-10, 18)
        integral_interval_probaCDF_recall(I, J, E18) / DeltaJ # 0.3392857
        E30 = (-10, 30)
        integral_interval_probaCDF_recall(I, J, E30) / DeltaJ # 0.5375
        E100 = (-10, 100)
        integral_interval_probaCDF_recall(I, J, E100) / DeltaJ # 0.8318182
        E10000 = (-10, 10000)
        integral_interval_probaCDF_recall(I, J, E10000) / DeltaJ # 0.9981518
      
        self.assertTrue(integral_interval_probaCDF_recall(I, J, E10) / DeltaJ < integral_interval_probaCDF_recall(I, J, E12) / DeltaJ)
        self.assertTrue(integral_interval_probaCDF_recall(I, J, E12) / DeltaJ < integral_interval_probaCDF_recall(I, J, E18) / DeltaJ)
        self.assertTrue(integral_interval_probaCDF_recall(I, J, E18) / DeltaJ < integral_interval_probaCDF_recall(I, J, E30) / DeltaJ)
        self.assertTrue(integral_interval_probaCDF_recall(I, J, E30) / DeltaJ < integral_interval_probaCDF_recall(I, J, E100) / DeltaJ)
        self.assertTrue(integral_interval_probaCDF_recall(I, J, E100) / DeltaJ < integral_interval_probaCDF_recall(I, J, E10000) / DeltaJ)

if __name__ == '__main__':
    unittest.main()
