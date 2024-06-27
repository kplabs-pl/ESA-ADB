from typing import Optional, Tuple

import numpy as np
import time
from prts import ts_precision, ts_recall, ts_fscore

from .auc_metrics import AucMetric
from .metric import Metric
from .thresholding import ThresholdingStrategy, NoThresholding, FixedValueThresholding

from timeeval.metrics.eTaPR_pkg.etapr import eTaPR
from timeeval.metrics.eTaPR_pkg.DataManage import Range
from tqdm import tqdm


class RangePrecision(Metric):
    """Computes the range-based precision metric introduced by Tatbul et al. at NeurIPS 2018 [TatbulEtAl2018]_.

    Parameters
    ----------
    thresholding_strategy : ThresholdingStrategy
        Strategy used to find a threshold over continuous anomaly scores to get binary labels.
        Use :class:`timeeval.metrics.thresholding.NoThresholding` for results that already contain binary labels.
    alpha : float
        Weight of the existence reward. For most - when not all - cases, `p_alpha` should be set to 0.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    name : str
        Custom name for this metric (e.g. including your parameter changes).
    """

    def __init__(self, thresholding_strategy: ThresholdingStrategy = NoThresholding(), alpha: float = 0,
                 cardinality: str = "reciprocal", bias: str = "flat", name: str = "RANGE_PRECISION") -> None:
        self._thresholding_strategy = thresholding_strategy
        self._alpha = alpha
        self._cardinality = cardinality
        self._bias = bias
        self._name = name

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_pred = self._thresholding_strategy.fit_transform(y_true, y_score)
        score: float = ts_precision(y_true, y_pred, alpha=self._alpha, cardinality=self._cardinality, bias=self._bias)
        return score

    def supports_continuous_scorings(self) -> bool:
        return not isinstance(self._thresholding_strategy, NoThresholding)

    @property
    def name(self) -> str:
        return self._name


class RangeRecall(Metric):
    """Computes the range-based recall metric introduced by Tatbul et al. at NeurIPS 2018 [TatbulEtAl2018]_.

    Parameters
    ----------
    thresholding_strategy : ThresholdingStrategy
        Strategy used to find a threshold over continuous anomaly scores to get binary labels.
        Use :class:`timeeval.metrics.thresholding.NoThresholding` for results that already contain binary labels.
    alpha : float
        Weight of the existence reward. If 0: no existence reward, if 1: only existence reward.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    name : str
        Custom name for this metric (e.g. including your parameter changes).
    """

    def __init__(self, thresholding_strategy: ThresholdingStrategy = NoThresholding(), alpha: float = 0,
                 cardinality: str = "reciprocal", bias: str = "flat", name: str = "RANGE_RECALL") -> None:
        self._thresholding_strategy = thresholding_strategy
        self._alpha = alpha
        self._cardinality = cardinality
        self._bias = bias
        self._name = name

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_pred = self._thresholding_strategy.fit_transform(y_true, y_score)
        score: float = ts_recall(y_true, y_pred, alpha=self._alpha, cardinality=self._cardinality, bias=self._bias)
        return score

    def supports_continuous_scorings(self) -> bool:
        return not isinstance(self._thresholding_strategy, NoThresholding)

    @property
    def name(self) -> str:
        return self._name


class RangeFScore(Metric):
    """Computes the range-based F-score using the recall and precision metrics by Tatbul et al. at NeurIPS 2018
    [TatbulEtAl2018]_.

    The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its
    worst value at 0. This implementation uses the range-based precision and range-based recall as basis.

    Parameters
    ----------
    thresholding_strategy : ThresholdingStrategy
        Strategy used to find a threshold over continuous anomaly scores to get binary labels.
        Use :class:`timeeval.metrics.thresholding.NoThresholding` for results that already contain binary labels.
    beta : float
        F-score beta determines the weight of recall in the combined score.
        beta < 1 lends more weight to precision, while beta > 1 favors recall.
    p_alpha : float
        Weight of the existence reward for the range-based precision. For most - when not all - cases, `p_alpha`
        should be set to 0.
    r_alpha : float
        Weight of the existence reward. If 0: no existence reward, if 1: only existence reward.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    p_bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    r_bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    name : str
        Custom name for this metric (e.g. including your parameter changes). If `None`, will include the beta-value in
        the name: "RANGE_F{beta}_SCORE".
    """

    def __init__(self,
                 thresholding_strategy: ThresholdingStrategy = NoThresholding(),
                 beta: float = 1,
                 p_alpha: float = 0,
                 r_alpha: float = 0.5,
                 cardinality: str = "reciprocal",
                 p_bias: str = "flat",
                 r_bias: str = "flat",
                 name: Optional[str] = None) -> None:
        self._thresholding_strategy = thresholding_strategy
        self._beta = beta
        self._p_alpha = p_alpha
        self._r_alpha = r_alpha
        self._cardinality = cardinality
        self._p_bias = p_bias
        self._r_bias = r_bias
        self._name = f"RANGE_F{self._beta:.2f}_SCORE" if name is None else name

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_pred = self._thresholding_strategy.fit_transform(y_true, y_score)
        score: float = ts_fscore(y_true, y_pred,
                                 beta=self._beta,
                                 p_alpha=self._p_alpha, r_alpha=self._r_alpha,
                                 cardinality=self._cardinality,
                                 p_bias=self._p_bias, r_bias=self._p_bias)
        return score

    def supports_continuous_scorings(self) -> bool:
        return not isinstance(self._thresholding_strategy, NoThresholding)

    @property
    def name(self) -> str:
        return self._name


class RangePrecisionRangeRecallAUC(AucMetric):
    """Computes the area under the precision recall curve when using the range-based precision and range-based
    recall metric introduced by Tatbul et al. at NeurIPS 2018 [TatbulEtAl2018]_.

    Parameters
    ----------
    max_samples: int
        TimeEval uses a community implementation of the range-based precision and recall metrics, which is quite slow.
        To prevent long runtimes caused by scorings with high precision (many thresholds), just a specific amount of
        possible thresholds is sampled. This parameter controls the maximum number of thresholds; too low numbers
        degrade the metrics' quality.
    r_alpha : float
        Weight of the existence reward for the range-based recall.
    p_alpha : float
        Weight of the existence reward for the range-based precision. For most - when not all - cases, `p_alpha`
        should be set to 0.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    plot : bool
    plot_store : bool
    name : str
        Custom name for this metric (e.g. including your parameter changes).


    .. rubric:: References

    .. [TatbulEtAl2018] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin Gottschlich. "Precision and Recall for
       Time Series." In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS),
       1920–30. 2018. http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """

    def __init__(self, max_samples: int = 50, r_alpha: float = 0.5, p_alpha: float = 0, cardinality: str = "reciprocal",
                 bias: str = "flat", plot: bool = False, plot_store: bool = False, name: str = "RANGE_PR_AUC") -> None:
        super().__init__(plot, plot_store)
        self._max_samples = max_samples
        self._r_alpha = r_alpha
        self._p_alpha = p_alpha
        self._cardinality = cardinality
        self._bias = bias
        self._name = name

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, self._range_precision_recall_curve)

    def _range_precision_recall_curve(self, y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        thresholds = np.unique(y_score)
        thresholds.sort()
        # The first precision and recall values are precision=class balance and recall=1.0, which corresponds to a
        # classifier that always predicts the positive class, independently of the threshold. This means that we can
        # skip the first threshold!
        p0 = y_true.sum() / len(y_true)
        r0 = 1.0
        thresholds = thresholds[1:]

        # sample thresholds
        n_thresholds = thresholds.shape[0]
        if n_thresholds > self._max_samples:
            every_nth = n_thresholds // (self._max_samples - 1)
            sampled_thresholds = thresholds[::every_nth]
            if thresholds[-1] == sampled_thresholds[-1]:
                thresholds = sampled_thresholds
            else:
                thresholds = np.r_[sampled_thresholds, thresholds[-1]]

        recalls = np.zeros_like(thresholds)
        precisions = np.zeros_like(thresholds)
        for i, threshold in enumerate(thresholds):
            y_pred = (y_score >= threshold).astype(np.int64)
            recalls[i] = ts_recall(y_true, y_pred,
                                   alpha=self._r_alpha,
                                   cardinality=self._cardinality,
                                   bias=self._bias)
            precisions[i] = ts_precision(y_true, y_pred,
                                         alpha=self._p_alpha,
                                         cardinality=self._cardinality,
                                         bias=self._bias)
        # first sort by recall, then by precision to break ties (important for noisy scorings)
        sorted_idx = np.lexsort((precisions * (-1), recalls))[::-1]
        return np.r_[p0, precisions[sorted_idx], 1], np.r_[r0, recalls[sorted_idx], 0], thresholds

    @property
    def name(self) -> str:
        return self._name


class eTaPR_Fscore(Metric):
    """Computes a threshold-agnostic version of eTaPR (Hwang et al., 2022) by computing mean or max F-score for a series
    of thresholds.
    (Hwang et al., 2022) https://doi.org/10.1145/3477314.3507024

    The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its
    worst value at 0. This implementation uses the affiliation-based precision and affiliation-based recall as basis.

    Parameters
    ----------
    theta_r : float
        minimal overlap of prediction with anomaly which is sufficient to detect that anomaly,
        i.e., if too small part of the anomaly is predicted, the expert will not find it.
        The higher the theta_r (from 0 to 1) the harder it is to achieve high eTaR(ecall)
    theta_p : float
        minimal overlap of detected anomaly with prediction which is sufficient to identify that anomaly based on that
        prediction, i.e., if most part of the prediction is irrelevant for the anomaly, the expert will regard this
        prediction as false positive.
        The higher the theta_p (from 0 to 1) the harder it is to achieve high eTaP(recision)
    beta : float
        F-score beta determines the weight of recall in the combined score. Default is 1.
        beta < 1 lends more weight to precision, while beta > 1 favors recall.
    aggregation : str
        Whether to take maximum ("max" - default) or mean ("mean") of F-scores for thresholds.
        beta < 1 lends more weight to precision, while beta > 1 favors recall.
    max_samples : int
        maximum number of thresholds to check. Default is None which means no limits (i.e., the number of unique anomaly
         scores), but it may be useful to set some limit for larger datasets with millions of unique anomaly scores.
    min_threshold : float
        minimal threshold to check. The default is 0, but it may be useful to set a larger
        value for large datasets where low thresholds result with millions of separate anomaly ranges to check
    """

    def __init__(self, theta_r: float = 0.5, theta_p: float = 0.5, delta: float = 0, beta: float = 1,
                 aggregation: str = "max", max_samples: int = None, min_threshold: float = 0) -> None:
        assert 0 <= theta_r <= 1
        self.theta_r = theta_r
        assert 0 <= theta_p <= 1
        self.theta_p = theta_p
        self.delta = delta
        self.beta = beta
        self._aggregation = aggregation
        self._max_samples = max_samples
        self._min_threshold = min_threshold
        beta_name = 1 if beta == 1 else np.round(beta, 2)
        self._name = f"eTaPR_{aggregation}F{beta_name}"

    def _anomaly_bounds(self, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """corresponds to range_convers_new"""
        # convert to boolean/binary
        labels = y_true > 0
        # deal with start and end of time series
        labels = np.diff(np.r_[0, labels, 0])
        # extract begin and end of anomalous regions
        index = np.arange(0, labels.shape[0])
        starts = index[labels == 1]
        ends = index[labels == -1]
        return starts, ends

    def _get_thresholds_from_scores(self, scores):
        thresholds = np.sort(np.unique(scores))
        thresholds = thresholds[(thresholds > self._min_threshold) & (thresholds < 1)]

        if self._max_samples is not None and len(thresholds) > self._max_samples:
            thresholds = thresholds[np.round(np.linspace(0, len(thresholds) - 1, self._max_samples)).astype(int)]

        return thresholds

    def score(self, y_true: np.ndarray, y_score: np.ndarray):
        start_time = time.time()
        starts, ends = self._anomaly_bounds(y_true)
        anomalies = [Range.Range(s, e, '') for s, e in zip(starts, ends)]

        thresholds = self._get_thresholds_from_scores(y_score)

        f_scores = []
        for t in tqdm(thresholds):
            thresholding_strategy = FixedValueThresholding(t)
            y_pred = thresholding_strategy.fit_transform(y_true, y_score)

            starts, ends = self._anomaly_bounds(y_pred)
            pred_ranges = [Range.Range(s, e, '') for s, e in zip(starts, ends)]

            ev = eTaPR(self.theta_p, self.theta_r, self.delta)
            ev.set(anomalies, pred_ranges)

            precision = ev.eTaP()
            recall = ev.eTaR()

            denom = self.beta**2 * precision + recall
            if denom > 0:
                f_score = ((1 + self.beta**2) * precision * recall) / denom
                f_scores.append(f_score)

        if self._aggregation == "mean":
            result = np.mean(f_scores)
        else:
            result = np.max(f_scores)

        print(f"{self.name} calculation time: {time.time() - start_time}")

        return result

    def supports_continuous_scorings(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self._name


class eTaPR_PR_AUC(AucMetric):
    """Computes a threshold-agnostic version of eTaPR (Hwang et al., 2022) by computing PR-AUC.
    (Hwang et al., 2022) https://doi.org/10.1145/3477314.3507024


    Parameters
    ----------
    theta_r : float
        minimal overlap of prediction with anomaly which is sufficient to detect that anomaly,
        i.e., if too small part of the anomaly is predicted, the expert will not find it.
        The higher the theta_r (from 0 to 1) the harder it is to achieve high eTaR(ecall)
    theta_p : float
        minimal overlap of detected anomaly with prediction which is sufficient to identify that anomaly based on that
        prediction, i.e., if most part of the prediction is irrelevant for the anomaly, the expert will regard this
        prediction as false positive.
        The higher the theta_p (from 0 to 1) the harder it is to achieve high eTaP(recision)
    delta : float
        the delta ratio for creating additional ambiguous regions around anomalies that can be rewarded in scoring (for
        the details see definitions in
        'Hwang,W.-S. et al. (2019) Time-Series Aware Precision and Recall for Anomaly Detection: Considering Variety of
        Detection Result and Addressing Ambiguous Labeling.')

    """

    def __init__(self, plot: bool = False, plot_store: bool = False, theta_r: float = 0.1, theta_p: float = 0.1, delta: float = 0, max_samples: int = None,
                 min_threshold: float = 0) -> None:
        super().__init__(plot, plot_store, max_samples, min_threshold)
        assert 0 <= theta_r <= 1
        self.theta_r = theta_r
        assert 0 <= theta_p <= 1
        self.theta_p = theta_p
        self.delta = delta

    def _anomaly_bounds(self, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """corresponds to range_convers_new"""
        # convert to boolean/binary
        labels = y_true > 0
        # deal with start and end of time series
        labels = np.diff(np.r_[0, labels, 0])
        # extract begin and end of anomalous regions
        index = np.arange(0, labels.shape[0])
        starts = index[labels == 1]
        ends = index[labels == -1]
        return starts, ends

    def _init_pr_auc_calculation(self, y_true: np.array):
        starts, ends = self._anomaly_bounds(y_true)
        self.anomalies = [Range.Range(s, e, '') for s, e in zip(starts, ends)]

    def _calculate_precision_and_recall(self, y_score: np.array, threshold: float):
        y_pred = (y_score >= threshold).astype(int)
        starts, ends = self._anomaly_bounds(y_pred)
        pred_ranges = [Range.Range(s, e, '') for s, e in zip(starts, ends)]

        ev = eTaPR(self.theta_p, self.theta_r, self.delta)
        ev.set(self.anomalies, pred_ranges)

        return ev.eTaP(), ev.eTaR()

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, self._precision_recall_curve)

    @property
    def name(self) -> str:
        return "eTaPR_PR_AUC"


class point_adjust_PR_AUC(AucMetric):
    """Computes the point-adjust PR-AUC

    Parameters
    ----------
    theta_r : float
        minimal overlap of prediction with anomaly which is sufficient to detect that anomaly,
        i.e., if too small part of the anomaly is predicted, the expert will not find it.
        The higher the theta_r (from 0 to 1) the harder it is to achieve high Recall
    """

    def __init__(self, plot: bool = False, plot_store: bool = False, theta_r: float = 0.1, max_samples: int = None, min_threshold: float = 0) -> None:
        super().__init__(plot, plot_store, max_samples, min_threshold)
        assert 0 <= theta_r <= 1
        self.theta_r = theta_r

    def _anomaly_bounds(self, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """corresponds to range_convers_new"""
        # convert to boolean/binary
        labels = y_true > 0
        # deal with start and end of time series
        labels = np.diff(np.r_[0, labels, 0])
        # extract begin and end of anomalous regions
        index = np.arange(0, labels.shape[0])
        starts = index[labels == 1]
        ends = index[labels == -1]
        return starts, ends

    def _init_pr_auc_calculation(self, y_true: np.array):
        starts, ends = self._anomaly_bounds(y_true)
        self.anomalies = [Range.Range(s, e, '') for s, e in zip(starts, ends)]

    def _calculate_precision_and_recall(self, y_score: np.array, threshold: float):
        y_pred = (y_score >= threshold).astype(int)
        starts, ends = self._anomaly_bounds(y_pred)
        pred_ranges = [Range.Range(s, e, '') for s, e in zip(starts, ends)]

        ev = eTaPR(0, self.theta_r)
        ev.set(self.anomalies, pred_ranges)

        return ev.point_adjust_precision(self.theta_r), ev.point_adjust_recall(self.theta_r)

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, self._precision_recall_curve)

    @property
    def name(self) -> str:
        return "POINT_ADJUST_PR_AUC"


if __name__ == "__main__":
    gt = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1])
    pr = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
    metric = point_adjust_PR_AUC(max_samples=50)
    print(metric.score(gt, pr))
