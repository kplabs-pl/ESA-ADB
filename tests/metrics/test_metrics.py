import unittest
import warnings

import numpy as np
import pandas as pd
from pandas import Timedelta

from timeeval import DefaultMetrics
from timeeval.metrics import (RangeFScore, RangePrecision, RangeRecall, F1Score, Precision, Recall, FScoreAtK,
                              PrecisionAtK, RangePrAUC, RangeRocAUC, RangePrVUS, RangeRocVUS, ESAScores,
                              ChannelAwareFScore, ADTQC)
from timeeval.metrics.thresholding import FixedValueThresholding, NoThresholding
from timeeval.metrics.utils import NANOSECONDS_IN_HOUR


class TestMetrics(unittest.TestCase):

    def test_regards_nan_as_wrong(self):
        y_scores = np.array([np.nan, 0.1, 0.9])
        y_true = np.array([0, 0, 1])
        result = DefaultMetrics.ROC_AUC(y_true, y_scores, nan_is_0=False)
        self.assertEqual(0.5, result)

        y_true = np.array([1, 0, 1])
        result = DefaultMetrics.ROC_AUC(y_true, y_scores, nan_is_0=False)
        self.assertEqual(0.5, result)

    def test_regards_inf_as_wrong(self):
        y_scores = np.array([0.1, np.inf, 0.9])
        y_true = np.array([0, 0, 1])
        result = DefaultMetrics.ROC_AUC(y_true, y_scores, inf_is_1=False)
        self.assertEqual(0.5, result)

        y_true = np.array([0, 1, 1])
        result = DefaultMetrics.ROC_AUC(y_true, y_scores, inf_is_1=False)
        self.assertEqual(0.5, result)

    def test_regards_neginf_as_wrong(self):
        y_scores = np.array([0.1, -np.inf, 0.9])
        y_true = np.array([0, 0, 1])
        result = DefaultMetrics.ROC_AUC(y_true, y_scores, neginf_is_0=False)
        self.assertEqual(0.5, result)

        y_true = np.array([0, 1, 1])
        result = DefaultMetrics.ROC_AUC(y_true, y_scores, neginf_is_0=False)
        self.assertEqual(0.5, result)

    def test_range_based_f1(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = DefaultMetrics.RANGE_F1(y_true, y_pred)
        self.assertAlmostEqual(result, 0.66666, places=4)

    def test_range_based_f_score_thresholding(self):
        y_score = np.array([0.1, 0.9, 0.8, 0.2])
        y_true = np.array([0, 1, 0, 0])
        result = RangeFScore(thresholding_strategy=FixedValueThresholding(), beta=1)(y_true, y_score)
        self.assertAlmostEqual(result, 0.66666, places=4)

    def test_range_based_precision(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = DefaultMetrics.RANGE_PRECISION(y_true, y_pred)
        self.assertEqual(result, 0.5)

    def test_range_based_precision_thresholding(self):
        y_score = np.array([0.1, 0.9, 0.8, 0.2])
        y_true = np.array([0, 1, 0, 0])
        result = RangePrecision(thresholding_strategy=FixedValueThresholding())(y_true, y_score)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_range_based_recall(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = DefaultMetrics.RANGE_RECALL(y_true, y_pred)
        self.assertEqual(result, 1)

    def test_range_based_recall_thresholding(self):
        y_score = np.array([0.1, 0.9, 0.8, 0.2])
        y_true = np.array([0, 1, 0, 0])
        result = RangeRecall(thresholding_strategy=FixedValueThresholding())(y_true, y_score)
        self.assertEqual(result, 1)

    def test_rf1_value_error(self):
        y_pred = np.array([0, .2, .7, 0])
        y_true = np.array([0, 1, 0, 0])
        with self.assertRaises(ValueError):
            DefaultMetrics.RANGE_F1(y_true, y_pred)

    def test_range_based_p_range_based_r_curve_auc(self):
        y_pred = np.array([0, 0.1, 1., .5, 0.1, 0])
        y_true = np.array([0, 1, 1, 1, 0, 0])
        result = DefaultMetrics.RANGE_PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9583, places=4)

    def test_range_based_p_range_based_r_auc_perfect_hit(self):
        y_pred = np.array([0, 0, 0.5, 0.5, 0, 0])
        y_true = np.array([0, 0, 1, 1, 0, 0])
        result = DefaultMetrics.RANGE_PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 1.0000, places=4)

    def test_pr_curve_auc(self):
        y_pred = np.array([0, 0.1, 1., .5, 0, 0])
        y_true = np.array([0, 0, 1, 1, 0, 0])
        result = DefaultMetrics.PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 1.0000, places=4)

    def test_average_precision(self):
        y_pred = np.array([0, 0.1, 1., .5, 0, 0])
        y_true = np.array([0, 1, 1, 0, 0, 0])
        result = DefaultMetrics.AVERAGE_PRECISION(y_true, y_pred)
        self.assertAlmostEqual(result, 0.8333, places=4)

    def test_fixed_range_based_pr_auc(self):
        y_pred = np.array([0, 0.1, 1., .5, 0.1, 0])
        y_true = np.array([0, 1, 1, 1, 0, 0])
        result = DefaultMetrics.FIXED_RANGE_PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9792, places=4)

    def test_range_based_pr_auc_discrete(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = DefaultMetrics.RANGE_PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 1.000, places=4)

    def test_fixed_range_based_pr_auc_discrete(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = DefaultMetrics.FIXED_RANGE_PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 1.000, places=4)

    def test_precision_at_k(self):
        y_pred = np.array([0, 0.1, 1., .6, 0.1, 0, 0.4, 0.5])
        y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        result = PrecisionAtK()(y_true, y_pred)
        self.assertAlmostEqual(result, 0.5000, places=4)
        result = PrecisionAtK(k=1)(y_true, y_pred)
        self.assertAlmostEqual(result, 1.0000, places=4)

    def test_fscore_at_k(self):
        y_pred = np.array([0.4, 0.1, 1., .5, 0.1, 0, 0.4, 0.5])
        y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        result = FScoreAtK()(y_true, y_pred)
        self.assertAlmostEqual(result, 0.500, places=4)
        result = FScoreAtK(k=3)(y_true, y_pred)
        self.assertAlmostEqual(result, 0.800, places=4)

    def test_edge_cases(self):
        y_true = np.zeros(10, dtype=np.int_)
        y_true[2:4] = 1
        y_true[6:8] = 1
        y_zeros = np.zeros_like(y_true, dtype=np.float_)
        y_flat = np.full_like(y_true, fill_value=0.5, dtype=np.float_)
        y_ones = np.ones_like(y_true, dtype=np.float_)
        y_inverted = (y_true * -1 + 1).astype(np.float_)

        pr_metrics = [DefaultMetrics.PR_AUC, DefaultMetrics.RANGE_PR_AUC, DefaultMetrics.FIXED_RANGE_PR_AUC]
        range_metrics = [RangeRocAUC(), RangePrAUC(), RangeRocVUS(), RangePrVUS()]
        other_metrics = [DefaultMetrics.ROC_AUC, PrecisionAtK(), FScoreAtK()]
        metrics = [*pr_metrics, *range_metrics, *other_metrics]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="Cannot compute metric for a constant value in y_score, returning 0.0!")
            for y_pred in [y_zeros, y_flat, y_ones]:
                for m in metrics:
                    self.assertAlmostEqual(m(y_true, y_pred), 0, msg=m.name)

            for m in pr_metrics:
                score = m(y_true, y_inverted)
                self.assertTrue(score <= 0.2, msg=f"{m.name}(y_true, y_inverted)={score} is not <= 0.2")
            # range metrics can deal with lag and this inverted score
            for m in other_metrics:
                score = m(y_true, y_inverted)
                self.assertAlmostEqual(score, 0, msg=m.name)

    def test_f1(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = F1Score(NoThresholding())(y_true, y_pred)
        self.assertAlmostEqual(result, 0.66666, places=4)

    def test_f_score_thresholding(self):
        y_score = np.array([0.1, 0.9, 0.8, 0.2])
        y_true = np.array([0, 1, 0, 0])
        result = F1Score(thresholding_strategy=FixedValueThresholding())(y_true, y_score)
        self.assertAlmostEqual(result, 0.66666, places=4)

    def test_precision(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = Precision(NoThresholding())(y_true, y_pred)
        self.assertEqual(result, 0.5)

    def test_precision_thresholding(self):
        y_score = np.array([0.1, 0.9, 0.8, 0.2])
        y_true = np.array([0, 1, 0, 0])
        result = Precision(thresholding_strategy=FixedValueThresholding())(y_true, y_score)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_recall(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = Recall(NoThresholding())(y_true, y_pred)
        self.assertEqual(result, 1)

    def test_recall_thresholding(self):
        y_score = np.array([0.1, 0.9, 0.8, 0.2])
        y_true = np.array([0, 1, 0, 0])
        result = Recall(thresholding_strategy=FixedValueThresholding())(y_true, y_score)
        self.assertEqual(result, 1)


class TestVUSMetrics(unittest.TestCase):
    def setUp(self) -> None:
        y_true = np.zeros(200)
        y_true[10:20] = 1
        y_true[28:33] = 1
        y_true[110:120] = 1
        y_score = np.random.default_rng(41).random(200) * 0.5
        y_score[16:22] = 1
        y_score[33:38] = 1
        y_score[160:170] = 1
        self.y_true = y_true
        self.y_score = y_score
        self.expected_range_pr_auc = 0.3737854660
        self.expected_range_roc_auc = 0.7108527197
        self.expected_range_pr_volume = 0.7493254559  # max_buffer_size = 200
        self.expected_range_roc_volume = 0.8763382130  # max_buffer_size = 200

    def test_range_pr_auc_compat(self):
        result = RangePrAUC(compatibility_mode=True)(self.y_true, self.y_score)
        self.assertAlmostEqual(result, self.expected_range_pr_auc, places=10)

    def test_range_roc_auc_compat(self):
        result = RangeRocAUC(compatibility_mode=True)(self.y_true, self.y_score)
        self.assertAlmostEqual(result, self.expected_range_roc_auc, places=10)

    def test_edge_case_existence_reward_compat(self):
        result = RangePrAUC(compatibility_mode=True, buffer_size=4)(self.y_true, self.y_score)
        self.assertAlmostEqual(result, 0.2506464391, places=10)
        result = RangeRocAUC(compatibility_mode=True, buffer_size=4)(self.y_true, self.y_score)
        self.assertAlmostEqual(result, 0.6143220816, places=10)

    def test_range_pr_volume_compat(self):
        result = RangePrVUS(max_buffer_size=200, compatibility_mode=True)(self.y_true, self.y_score)
        self.assertAlmostEqual(result, self.expected_range_pr_volume, places=10)

    def test_range_roc_volume_compat(self):
        result = RangeRocVUS(max_buffer_size=200, compatibility_mode=True)(self.y_true, self.y_score)
        self.assertAlmostEqual(result, self.expected_range_roc_volume, places=10)

    def test_range_pr_auc(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = RangePrAUC()(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9636, places=4)

    def test_range_roc_auc(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = RangeRocAUC()(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9653, places=4)

    def test_range_pr_volume(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = RangePrVUS(max_buffer_size=200)(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9937, places=4)

    def test_range_roc_volume(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = RangeRocVUS(max_buffer_size=200)(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9904, places=4)


class TestAffiliationMetrics(unittest.TestCase):

    def test_time_aware(self):
        full_range = [pd.to_datetime("2015-01-01"), pd.to_datetime("2015-01-15")]
        y_true = [["id_0", pd.to_datetime("2015-01-01"), pd.to_datetime("2015-01-02"), "Rare Event", "", "", ""],
                  ["id_1", pd.to_datetime("2015-01-04"), pd.to_datetime("2015-01-05"), "Anomaly", "", "", ""],
                  ["id_1", pd.to_datetime("2015-01-06"), pd.to_datetime("2015-01-07"), "Anomaly", "", "", ""],
                  ["id_2", pd.to_datetime("2015-01-09"), pd.to_datetime("2015-01-10"), "Communication Gap", "", "", ""],
                  ["id_3", pd.to_datetime("2015-01-12"), pd.to_datetime("2015-01-14"), "Rare Event", "", "", ""]]
        y_true = pd.DataFrame(y_true, columns=["ID", "StartTime", "EndTime", "Category", "Dimensionality", "Locality",
                                               "Length"])
        y_pred = np.array([[pd.to_datetime("2015-01-01"), 1],
                           [pd.to_datetime("2015-01-02"), 0],
                           [pd.to_datetime("2015-01-04"), 1],
                           [pd.to_datetime("2015-01-05"), 0],
                           [pd.to_datetime("2015-01-09"), 1],
                           [pd.to_datetime("2015-01-10"), 0]])

        select_rare_and_anomalies = {"Category": ["Rare Event", "Anomaly"]}
        select_anomalies_only = {"Category": ["Anomaly"]}

        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_rare_and_anomalies)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.6818181818181818, 'AFF_precision': 0.75, 'AFF_recall': 0.5, 'EW_F_0.50': 0.9090909090909091, 'EW_precision': 1.0, 'EW_recall': 0.6666666666666666, 'alarming_precision': 1.0},
                             result)
        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_anomalies_only)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.6818181818181818, 'AFF_precision': 0.75, 'AFF_recall': 0.5, 'EW_F_0.50': 1.0, 'EW_precision': 1.0, 'EW_recall': 1.0, 'alarming_precision': 1.0},
                             result)

        # Add additional detection for id_1
        y_pred = np.array([[pd.to_datetime("2015-01-01"), 1],
                           [pd.to_datetime("2015-01-02"), 0],
                           [pd.to_datetime("2015-01-04"), 1],
                           [pd.to_datetime("2015-01-05"), 0],
                           [pd.to_datetime("2015-01-06"), 1],
                           [pd.to_datetime("2015-01-07"), 0],
                           [pd.to_datetime("2015-01-09"), 1],
                           [pd.to_datetime("2015-01-10"), 0]])

        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_rare_and_anomalies)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.7936507936507936, 'AFF_precision': 0.8333333333333334, 'AFF_recall': 0.6666666666666666, 'EW_F_0.50': 0.9090909090909091, 'EW_precision': 1.0, 'EW_recall': 0.6666666666666666, 'alarming_precision': 1.0},
                             result)
        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_anomalies_only)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 1.0, 'AFF_precision': 1.0, 'AFF_recall': 1.0, 'EW_F_0.50': 1.0, 'EW_precision': 1.0, 'EW_recall': 1.0, 'alarming_precision': 1.0},
                             result)

        # Add false positive
        y_pred = np.array([[pd.to_datetime("2015-01-01"), 1],
                           [pd.to_datetime("2015-01-02"), 0],
                           [pd.to_datetime("2015-01-04"), 1],
                           [pd.to_datetime("2015-01-05"), 0],
                           [pd.to_datetime("2015-01-06"), 1],
                           [pd.to_datetime("2015-01-07"), 0],
                           [pd.to_datetime("2015-01-09"), 1],
                           [pd.to_datetime("2015-01-10"), 0],
                           [pd.to_datetime("2015-01-11"), 1],
                           [pd.to_datetime("2015-01-12"), 0]])

        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_rare_and_anomalies)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.7670454543744435, 'AFF_precision': 0.75, 'AFF_recall': 0.8437499989653835, 'EW_F_0.50': 0.5982905982905982, 'EW_precision': 0.5833333333333333, 'EW_recall': 0.6666666666666666, 'alarming_precision': 1.0},
                             result)
        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_anomalies_only)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 1.0, 'AFF_precision': 1.0, 'AFF_recall': 1.0, 'EW_F_0.50': 0.49295774647887325, 'EW_precision': 0.4375, 'EW_recall': 1.0, 'alarming_precision': 1.0},
                             result)

        # more than one detection per id_1
        y_pred = np.array([[pd.to_datetime("2015-01-01"), 0],
                           [pd.to_datetime("2015-01-04"), 1],
                           [pd.to_datetime("2015-01-06 12:00"), 0],
                           [pd.to_datetime("2015-01-06 14:00"), 1],
                           [pd.to_datetime("2015-01-08"), 0]])

        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_rare_and_anomalies)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.4934259707731491, 'AFF_precision': 0.5609195402298851, 'AFF_recall': 0.3331018508608397, 'EW_F_0.50': 0.6000000000000001, 'EW_precision': 0.75, 'EW_recall': 0.3333333333333333, 'alarming_precision': 0.5},
                             result)
        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_anomalies_only)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.7289393967240779, 'AFF_precision': 0.6827586206896552, 'AFF_recall': 0.9993055525825192, 'EW_F_0.50': 0.7894736842105263, 'EW_precision': 0.75, 'EW_recall': 1.0, 'alarming_precision': 0.5},
                             result)

        # overlapping anomalies in GT
        y_true = [["id_0", pd.to_datetime("2015-01-01"), pd.to_datetime("2015-01-07"), "Anomaly", "", "", ""],
                  ["id_1", pd.to_datetime("2015-01-02"), pd.to_datetime("2015-01-05"), "Anomaly", "", "", ""],
                  ["id_2", pd.to_datetime("2015-01-02"), pd.to_datetime("2015-01-03"), "Anomaly", "", "", ""]]
        y_true = pd.DataFrame(y_true, columns=["ID", "StartTime", "EndTime", "Category", "Dimensionality", "Locality",
                                               "Length"])
        y_pred = np.array([[pd.to_datetime("2015-01-01"), 0],
                           [pd.to_datetime("2015-01-02"), 1],
                           [pd.to_datetime("2015-01-03"), 0]])

        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_rare_and_anomalies)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.9525495750645814, 'AFF_precision': 1.0, 'AFF_recall': 0.8005952380731983, 'EW_F_0.50': 1.0, 'EW_precision': 1.0, 'EW_recall': 1.0, 'alarming_precision': 1.0},
                             result)
        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_anomalies_only)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.9525495750645814, 'AFF_precision': 1.0, 'AFF_recall': 0.8005952380731983, 'EW_F_0.50': 1.0, 'EW_precision': 1.0, 'EW_recall': 1.0, 'alarming_precision': 1.0},
                             result)

        y_pred = np.array([[pd.to_datetime("2015-01-01"), 1],
                           [pd.to_datetime("2015-01-02"), 0],
                           [pd.to_datetime("2015-01-04"), 1],
                           [pd.to_datetime("2015-01-05"), 0]])

        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_rare_and_anomalies)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.9848484850010343, 'AFF_precision': 1.0, 'AFF_recall': 0.9285714292494953, 'EW_F_0.50': 0.9090909090909091, 'EW_precision': 1.0, 'EW_recall': 0.6666666666666666, 'alarming_precision': 0.6666666666666666},
                             result)
        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_anomalies_only)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.9848484850010343, 'AFF_precision': 1.0, 'AFF_recall': 0.9285714292494953, 'EW_F_0.50': 0.9090909090909091, 'EW_precision': 1.0, 'EW_recall': 0.6666666666666666, 'alarming_precision': 0.6666666666666666},
                             result)

        # select by multiple classes
        y_true = [
            ["id_0", pd.to_datetime("2015-01-01"), pd.to_datetime("2015-01-02"), "Anomaly", "Multivariate", "Global",
             "Point"],
            ["id_1", pd.to_datetime("2015-01-04"), pd.to_datetime("2015-01-05"), "Anomaly", "Univariate", "Local",
             "Subsequence"],
            ["id_2", pd.to_datetime("2015-01-07"), pd.to_datetime("2015-01-08"), "Anomaly", "Multivariate", "Local",
             "Subsequence"]]
        y_true = pd.DataFrame(y_true, columns=["ID", "StartTime", "EndTime", "Category", "Dimensionality", "Locality",
                                               "Length"])
        y_pred = np.array([[pd.to_datetime("2015-01-01"), 0],
                           [pd.to_datetime("2015-01-04"), 1],
                           [pd.to_datetime("2015-01-09"), 0]])

        select_labels = {"Dimensionality": ["Univariate"], "Length": ["Subsequence"]}
        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_labels)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.7142857142857142, 'AFF_precision': 0.6666666666666666, 'AFF_recall': 1.0, 'EW_F_0.50': 0.7692307692307693, 'EW_precision': 0.7272727272727273, 'EW_recall': 1.0, 'alarming_precision': 1.0},
                             result)

        select_labels = {"Dimensionality": ["Multivariate"], "Length": ["Subsequence"]}
        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_labels)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.8778625954198473, 'AFF_precision': 0.8518518518518519, 'AFF_recall': 1.0, 'EW_F_0.50': 0.7692307692307693, 'EW_precision': 0.7272727272727273, 'EW_recall': 1.0, 'alarming_precision': 1.0},
                             result)

        select_labels = {"Locality": ["Global"], "Length": ["Point"]}
        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_labels)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 0.0, 'AFF_precision': 0.5, 'AFF_recall': 0.0, 'EW_F_0.50': 0, 'EW_precision': 0.0, 'EW_recall': 0.0, 'alarming_precision': 0.0},
                             result)

        # Add detection as the last sample in the full range
        y_true = [["id_0", pd.to_datetime("2015-01-15"), pd.to_datetime("2015-01-15"), "Anomaly", "", "", ""]]
        y_true = pd.DataFrame(y_true, columns=["ID", "StartTime", "EndTime", "Category", "Dimensionality", "Locality",
                                               "Length"])
        y_pred = np.array([[full_range[0], 0],
                           [full_range[1], 1]])

        metric = ESAScores(betas=0.5, full_range=full_range, select_labels=select_anomalies_only)
        result = metric.score(y_true, y_pred)
        self.assertDictEqual({'AFF_F_0.50': 1.0, 'AFF_precision': 1.0, 'AFF_recall': 1.0, 'EW_F_0.50': 1.0, 'EW_precision': 1.0, 'EW_recall': 1.0, 'alarming_precision': 1.0},
                             result)


class TestRankingMetrics(unittest.TestCase):
    def test_from_report(self):

        y_true = [["id_0", "ch1", pd.to_datetime("2015-01-01"), pd.to_datetime("2015-01-04"), "Anomaly", "", "", ""],
                  ["id_0", "ch1", pd.to_datetime("2015-01-05"), pd.to_datetime("2015-01-08"), "Anomaly", "", "", ""],
                  ["id_0", "ch2", pd.to_datetime("2015-01-02"), pd.to_datetime("2015-01-04"), "Anomaly", "", "", ""],
                  ["id_0", "ch3", pd.to_datetime("2015-01-02"), pd.to_datetime("2015-01-03"), "Anomaly", "", "", ""],
                  ["id_0", "ch3", pd.to_datetime("2015-01-06"), pd.to_datetime("2015-01-07"), "Anomaly", "", "", ""]]
        y_true = pd.DataFrame(y_true, columns=["ID", "Channel", "StartTime", "EndTime", "Category",
                                               "Dimensionality", "Locality", "Length"])
        y_pred = np.array([[[pd.to_datetime("2015-01-01"), 1],
                            [pd.to_datetime("2015-01-04"), 0]],
                           [[pd.to_datetime("2015-01-01"), 0],
                            [pd.to_datetime("2015-01-03"), 1],
                            [pd.to_datetime("2015-01-04"), 0]],
                           [[pd.to_datetime("2015-01-01"), 0],
                            [pd.to_datetime("2015-01-02"), 1],
                            [pd.to_datetime("2015-01-05"), 0],
                            [pd.to_datetime("2015-01-06"), 1],
                            [pd.to_datetime("2015-01-08"), 0]]], dtype=object)
        y_pred_channels_order = ["ch1", "ch2", "ch3"]

        y_pred = {key: value for key, value in zip(y_pred_channels_order, y_pred)}

        subsystems_mapping = {"s1": ["ch1"], "s2": ["ch2", "ch3"]}
        expected_result = {'channel_precision': 1.0, 'channel_recall': 1.0, 'channel_F0.50': 1.0,
                           'subsystem_precision': 1.0, 'subsystem_recall': 1.0, 'subsystem_F0.50': 1.0}

        result = ChannelAwareFScore(beta=0.5).score(y_true, y_pred, subsystems_mapping)
        for name, value in expected_result.items():
            self.assertAlmostEqual(result[name], value)

        y_true = [["id_0", "ch1", pd.to_datetime("2015-01-01"), pd.to_datetime("2015-01-04"), "Anomaly", "", "", ""],
                  ["id_0", "ch1", pd.to_datetime("2015-01-05"), pd.to_datetime("2015-01-08"), "Anomaly", "", "", ""],
                  ["id_0", "ch2", pd.to_datetime("2015-01-02"), pd.to_datetime("2015-01-04"), "Anomaly", "", "", ""],
                  ["id_0", "ch3", pd.to_datetime("2015-01-05 12:00"), pd.to_datetime("2015-01-05 14:00"), "Anomaly", "", "", ""]]
        y_true = pd.DataFrame(y_true, columns=["ID", "Channel", "StartTime", "EndTime", "Category",
                                               "Dimensionality", "Locality", "Length"])
        y_pred = np.array([[[pd.to_datetime("2015-01-01"), 1],
                            [pd.to_datetime("2015-01-04"), 0]],
                           [[pd.to_datetime("2015-01-01"), 0]],
                           [[pd.to_datetime("2015-01-01"), 0],
                            [pd.to_datetime("2015-01-02"), 1],
                            [pd.to_datetime("2015-01-05"), 0],
                            [pd.to_datetime("2015-01-06"), 1],
                            [pd.to_datetime("2015-01-08"), 0]]], dtype=object)
        y_pred_channels_order = ["ch1", "ch2", "ch3"]

        y_pred = {key: value for key, value in zip(y_pred_channels_order, y_pred)}

        subsystems_mapping = {"s1": ["ch1"], "s2": ["ch2", "ch3"]}
        expected_result = {'channel_precision': 1.0, 'channel_recall': 0.6666666666666666, 'channel_F0.50': 0.9090909090909091,
                           'subsystem_precision': 1.0, 'subsystem_recall': 1.0, 'subsystem_F0.50': 1.0}

        result = ChannelAwareFScore(beta=0.5).score(y_true, y_pred, subsystems_mapping)
        for name, value in expected_result.items():
            self.assertAlmostEqual(result[name], value)

        y_true = [["id_0", "ch1", pd.to_datetime("2015-01-01"), pd.to_datetime("2015-01-06"), "Anomaly", "", "", ""],
                  ["id_0", "ch2", pd.to_datetime("2015-01-01"), pd.to_datetime("2015-01-03"), "Anomaly", "", "", ""],
                  ["id_1", "ch1", pd.to_datetime("2015-01-05"), pd.to_datetime("2015-01-09"), "Anomaly", "", "", ""],
                  ["id_1", "ch3", pd.to_datetime("2015-01-04"), pd.to_datetime("2015-01-09"), "Anomaly", "", "", ""],
                  ["id_1", "ch4", pd.to_datetime("2015-01-07"), pd.to_datetime("2015-01-09"), "Anomaly", "", "", ""]]
        y_true = pd.DataFrame(y_true, columns=["ID", "Channel", "StartTime", "EndTime", "Category",
                                               "Dimensionality", "Locality", "Length"])
        y_pred = np.array([[[pd.to_datetime("2015-01-01"), 1],
                            [pd.to_datetime("2015-01-05 12:00"), 0]],
                           [[pd.to_datetime("2015-01-01"), 0]],
                           [[pd.to_datetime("2015-01-01"), 0],
                            [pd.to_datetime("2015-01-04"), 1],
                            [pd.to_datetime("2015-01-08"), 0]],
                           [[pd.to_datetime("2015-01-01"), 0],
                            [pd.to_datetime("2015-01-08"), 1]]], dtype=object)
        y_pred_channels_order = ["ch1", "ch2", "ch3", "ch4"]
        y_pred = {key: value for key, value in zip(y_pred_channels_order, y_pred)}

        subsystems_mapping = {"s1": ["ch1", "ch2"], "s2": ["ch3", "ch4"]}
        expected_result = {'channel_precision': 1.0, 'channel_recall': 0.75, 'channel_F0.50': 0.9166666666666667,
                           'subsystem_precision': 1.0, 'subsystem_recall': 1.0, 'subsystem_F0.50': 1.0}

        result = ChannelAwareFScore(beta=0.5).score(y_true, y_pred, subsystems_mapping)
        for name, value in expected_result.items():
            self.assertAlmostEqual(result[name], value)


        y_pred["ch4"][..., 1] = 1
        expected_result = {'channel_precision': 0.75, 'channel_recall': 0.75, 'channel_F0.50': 0.75,
                           'subsystem_precision': 0.75, 'subsystem_recall': 1.0, 'subsystem_F0.50': 0.7777777777777778}

        result = ChannelAwareFScore(beta=0.5).score(y_true, y_pred, subsystems_mapping)
        for name, value in expected_result.items():
            self.assertAlmostEqual(result[name], value)


class TestLatencyMetrics(unittest.TestCase):
    def test_from_slides(self):
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
                              columns=["ID", "Channel", "StartTime", "EndTime", "Category", "Dimensionality",
                                       "Locality",
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
        expected_result = {'Nb_Before': 2, "Nb_After": 1, "AfterRate": 0.3333333333333333, 'Total': 0.4404148830816283}
        result = metric.score(y_true, y_pred)
        for name, value in expected_result.items():
            self.assertAlmostEqual(result[name], value)

        y_pred = np.array([[[pd.to_datetime("8:10:10"), 1]],
                           [[pd.to_datetime("8:10:10"), 1]],
                           [[pd.to_datetime("8:10:10"), 1]]], dtype=object)
        y_pred = {key: value for key, value in zip(y_pred_channels_order, y_pred)}

        metric = ADTQC(full_range=full_range)
        expected_result = {'Nb_Before': 3, "Nb_After": 1, "AfterRate": 0.25, 'Total': 0.25}
        result = metric.score(y_true, y_pred)
        for name, value in expected_result.items():
            self.assertAlmostEqual(result[name], value)

        y_pred = np.array([[[pd.to_datetime("8:10:10"), 1]],
                           [[pd.to_datetime("8:10:10"), 0]],
                           [[pd.to_datetime("8:10:10"), 0]]], dtype=object)
        y_pred = {key: value for key, value in zip(y_pred_channels_order, y_pred)}

        metric = ADTQC(full_range=full_range)
        expected_result = {'Nb_Before': 1, "Nb_After": 1, "AfterRate": 0.5, 'Total': 0.5}
        result = metric.score(y_true, y_pred)
        for name, value in expected_result.items():
            self.assertAlmostEqual(result[name], value)

        y_pred = np.array([[[pd.to_datetime("8:10:10"), 0],
                            [pd.to_datetime("8:10:39"), 1],
                            [pd.to_datetime("8:10:41"), 0]],
                           [[pd.to_datetime("8:10:10"), 0],
                            [pd.to_datetime("8:10:39"), 1],
                            [pd.to_datetime("8:10:58"), 0]],
                           [[pd.to_datetime("8:10:10"), 0],
                            [pd.to_datetime("8:10:39"), 1],
                            [pd.to_datetime("8:10:41"), 0]]
                           ], dtype=object)
        y_pred = {key: value for key, value in zip(y_pred_channels_order, y_pred)}

        metric = ADTQC(full_range=full_range)
        expected_result = {'Nb_Before': 1, "Nb_After": 1, "AfterRate": 0.5, 'Total': 0.12466690772198988}
        result = metric.score(y_true, y_pred)
        for name, value in expected_result.items():
            self.assertAlmostEqual(result[name], value)

