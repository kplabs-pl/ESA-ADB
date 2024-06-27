#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import glob
import re
import math

from metrics.affiliation_based_metrics_repo.affiliation.generics import convert_vector_to_events, f1_func, read_gz_data
from metrics.affiliation_based_metrics_repo.affiliation.metrics import produce_all_results


"""
Check reproducibility of the results
"""
class Test_data(unittest.TestCase):       
    def test_description_data(self):
        """
        Check description of the data sets:
        - number of instances,
        - proportion of anomalous instances,
        - number of events in the ground truth
        """
        filepaths = glob.glob('data/*.gz')
        for filepath in filepaths:
            vector = read_gz_data(filepath)
            if re.search('machinetemp_', filepath):
                self.assertEqual(len(vector), 17682)
                if re.search('_groundtruth', filepath):
                    self.assertAlmostEqual(sum(vector)/len(vector), 0.0641, places=3)
                    events = convert_vector_to_events(vector)
                    self.assertEqual(len(events), 2)
            if re.search('nyctaxi_', filepath):
                self.assertEqual(len(vector), 2307)
                if re.search('_groundtruth', filepath):
                    self.assertAlmostEqual(sum(vector)/len(vector), 0.2691, places=3)
                    events = convert_vector_to_events(vector)
                    self.assertEqual(len(events), 3)
            if re.search('twitteraapl_', filepath):
                self.assertEqual(len(vector), 11889)
                if re.search('_groundtruth', filepath):
                    self.assertAlmostEqual(sum(vector)/len(vector), 0.0667, places=3)
                    events = convert_vector_to_events(vector)
                    self.assertEqual(len(events), 2)
            if re.search('swat_', filepath):
                self.assertEqual(len(vector), 449919)
                if re.search('_groundtruth', filepath):
                    self.assertAlmostEqual(sum(vector)/len(vector), 0.1214, places=3)
                    events = convert_vector_to_events(vector)
                    self.assertEqual(len(events), 35)

    def test_cells(self):
        """
        Check each cell by applying the affiliation metrics
        """
        # table from the article to be checked
        table_in_article = {'machinetemp': {},
                            'nyctaxi': {},
                            'twitteraapl': {},
                            'swat': {}}
        
        table_in_article['machinetemp']['trivial']    = '1.00/0.50/0.66'
        table_in_article['machinetemp']['adversary']  = '0.49/1.00/0.66'
        table_in_article['machinetemp']['greenhouse'] = '0.71/0.99/0.83'
        table_in_article['machinetemp']['lstmad']     = '0.50/1.00/0.67'
        table_in_article['machinetemp']['luminol']    = '0.54/0.99/0.70'

        table_in_article['nyctaxi']['trivial']        = '1.00/0.30/0.46'
        table_in_article['nyctaxi']['adversary']      = '0.54/1.00/0.70'
        table_in_article['nyctaxi']['greenhouse']     = '0.51/0.99/0.67'
        table_in_article['nyctaxi']['lstmad']         = '0.51/1.00/0.67'
        table_in_article['nyctaxi']['luminol']        = '0.38/0.79/0.51'

        table_in_article['twitteraapl']['trivial']    = '1.00/0.49/0.66'
        table_in_article['twitteraapl']['adversary']  = '0.50/1.00/0.67'
        table_in_article['twitteraapl']['greenhouse'] = '0.78/0.98/0.87'
        table_in_article['twitteraapl']['lstmad']     = '0.66/0.99/0.79'
        table_in_article['twitteraapl']['luminol']    = '0.73/0.98/0.83'

        table_in_article['swat']['trivial']           = '1.00/0.03/0.06'
        table_in_article['swat']['adversary']         = '0.53/1.00/0.69'
        table_in_article['swat']['iforest']           = '0.52/0.84/0.64'
        table_in_article['swat']['ocsvm']             = '0.65/0.70/0.68'
        table_in_article['swat']['seq2seq']           = '0.86/0.79/0.83'

        # checking the table
        results = produce_all_results() # produce results
        
        # Check results related to `best_algos` and `pr_of_best_algo`
        for data_name in results.keys():
            for algo_name in results[data_name].keys():
                p = results[data_name][algo_name]['precision']
                r = results[data_name][algo_name]['recall']
                f1 = f1_func(p, r)
                # convert to a string with two decimals
                p, r, f1 = ['%.2f' % x for x in [p, r, f1]]
                cell_obtained_from_results = str(p) + '/' + str(r) + '/' + str(f1)
                self.assertEqual(cell_obtained_from_results, table_in_article[data_name][algo_name])

    def test_single_events_results(self):
        """
        Check single events results for swat with iforest/seq2seq
        """        
        results = produce_all_results() # produce results
        p_precis = dict()
        p_recall = dict()
        p_f1 = dict()
        for algo_name in ['iforest', 'seq2seq']:
            p_precis_raw = results['swat'][algo_name]['individual_precision_probabilities']
            p_recall_raw = results['swat'][algo_name]['individual_recall_probabilities']
            p_precis[algo_name] = [round(x, 2) for x in p_precis_raw]
            p_recall[algo_name] = [round(x, 2) for x in p_recall_raw]
            p_f1[algo_name] = [round(f1_func(x[0], x[1]), 2) for x in zip(p_precis_raw, p_recall_raw)]

        p_out = dict()
        for algo_name in ['iforest', 'seq2seq']:
            p_out[algo_name] = ['%.2f' % x[0] + '/' + '%.2f' % x[1] + '/' + '%.2f' % x[2] for x in zip(p_precis[algo_name], p_recall[algo_name], p_f1[algo_name])]

        self.assertEqual(p_out['iforest'][0:6],
                         ['0.37/0.53/0.44',
                          '1.00/0.91/0.95',
                          '0.76/0.99/0.86',
                          'nan/0.00/nan',
                          '0.38/0.60/0.46',
                          '0.09/0.21/0.12'])
        self.assertEqual(p_out['seq2seq'][0:6],
                         ['0.96/1.00/0.98', 
                          '0.86/1.00/0.93', 
                          '0.73/0.78/0.75', 
                          '0.39/0.71/0.50', 
                          '0.71/0.97/0.82', 
                          '0.88/1.00/0.94'])
        
        """
        Check the number of events for which seq2seq is better than iforest
        """
        nb_events = len(p_precis['seq2seq'])
        
        nb_seq2seq_nan_pred = sum([math.isnan(x) for x in p_precis['seq2seq']])
        nb_iforest_nan_pred = sum([math.isnan(x) for x in p_precis['iforest']])
        
        idx_nan1 = [idx for idx, val in enumerate(p_precis['seq2seq']) if math.isnan(val)]
        idx_nan2 = [idx for idx, val in enumerate(p_precis['iforest']) if math.isnan(val)]
        idx_nan = idx_nan1 + idx_nan2
        p_precis_seq2seq_not_nan = [val for idx, val in enumerate(p_precis['seq2seq']) if idx not in idx_nan]
        p_precis_iforest_not_nan = [val for idx, val in enumerate(p_precis['iforest']) if idx not in idx_nan]
        p_recall_seq2seq_not_nan = [val for idx, val in enumerate(p_recall['seq2seq']) if idx not in idx_nan]
        p_recall_iforest_not_nan = [val for idx, val in enumerate(p_recall['iforest']) if idx not in idx_nan]

        nb_both_better = sum([(p_seq > p_ifo) and (r_seq > r_ifo) for p_seq, p_ifo, r_seq, r_ifo in zip(p_precis_seq2seq_not_nan, p_precis_iforest_not_nan, p_recall_seq2seq_not_nan, p_recall_iforest_not_nan)])
        nb_better_precision_only = sum([(p_seq > p_ifo) and (r_seq == r_ifo) for p_seq, p_ifo, r_seq, r_ifo in zip(p_precis_seq2seq_not_nan, p_precis_iforest_not_nan, p_recall_seq2seq_not_nan, p_recall_iforest_not_nan)])

        nb_better_precision_worse_recall = sum([(p_seq > p_ifo) and (r_seq < r_ifo) for p_seq, p_ifo, r_seq, r_ifo in zip(p_precis_seq2seq_not_nan, p_precis_iforest_not_nan, p_recall_seq2seq_not_nan, p_recall_iforest_not_nan)])
        nb_better_recall_worse_precision = sum([(p_seq < p_ifo) and (r_seq > r_ifo) for p_seq, p_ifo, r_seq, r_ifo in zip(p_precis_seq2seq_not_nan, p_precis_iforest_not_nan, p_recall_seq2seq_not_nan, p_recall_iforest_not_nan)])
        nb_equivocal = nb_better_precision_worse_recall + nb_better_recall_worse_precision
        
        nb_both_worse = sum([(p_seq < p_ifo) and (r_seq < r_ifo) for p_seq, p_ifo, r_seq, r_ifo in zip(p_precis_seq2seq_not_nan, p_precis_iforest_not_nan, p_recall_seq2seq_not_nan, p_recall_iforest_not_nan)])

        # check that number of events is 35
        self.assertEqual(nb_events, 35)
        # check the number of zone without predictions are 6 for seq2seq and 2 for iforest
        self.assertEqual(nb_seq2seq_nan_pred, 6)
        self.assertEqual(nb_iforest_nan_pred, 2)  
        # check that we covered all the possibilities for those specific results
        self.assertEqual(nb_seq2seq_nan_pred + nb_iforest_nan_pred + nb_both_better + nb_better_precision_only + nb_equivocal + nb_both_worse, nb_events)
        # check the number of better elements
        self.assertEqual(nb_both_better + nb_better_precision_only, 21)
        self.assertEqual(nb_both_better, 13)
        self.assertEqual(nb_better_precision_only, 8)
        # check the number of equivocal results
        self.assertEqual(nb_equivocal, 4)
        # check the number of worse results
        self.assertEqual(nb_both_worse, 2)
        
if __name__ == '__main__':
    unittest.main()