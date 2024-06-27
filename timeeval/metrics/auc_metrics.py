from abc import ABC
from typing import Iterable, Callable
import time
import os
from tqdm import tqdm

import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve

from .metric import Metric


class AucMetric(Metric, ABC):
    """Base class for area-under-curve-based metrics.

    All AUC-Metrics support continuous scorings, calculate the area under a curve function, and allow plotting this
    curve function. See the subclasses' documentation for a detailed explanation of the corresponding curve and metric.

    Parameters
    ----------
    max_samples : int
        maximum number of points on Precision-Recall curve. Default is None which means no limits (i.e., the number of
        unique anomaly scores), but it may be useful to set some limit for larger datasets with millions of unique
        anomaly scores.
    min_threshold : float
        minimal threshold for creating Precision-Recall curve. The default is 0, but it may be useful to set a larger
        value for large datasets where low thresholds result with millions of separate anomaly ranges to check
    """
    def __init__(self, plot: bool = False, plot_store: str = ".", max_samples: int = 100, min_threshold: float = 0) -> None:
        self._plot = plot
        self._plot_store = plot_store
        self._max_samples = max_samples
        self._min_threshold = min_threshold

    def _auc(self, y_true: np.ndarray, y_score: Iterable[float], _curve_function: Callable) -> float:
        x, y, thresholds = _curve_function(y_true, y_score)
        name = _curve_function.__name__
        if "precision_recall" in name:
            # swap x and y
            x, y = y, x
        area: float = auc(x, y)
        if self._plot:
            import matplotlib.pyplot as plt

            plt.plot(x, y, label=self.name, drawstyle="steps-post")
            plt.title(f"{self.name} | area = {area:.4f}")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            if "precision_recall" in name:
                plt.xlabel("Recall")
                plt.ylabel("Precision")
            plt.savefig(os.path.join(self._plot_store, f"fig-{self.name}.png"))
            plt.close()
        return area

    def _init_pr_auc_calculation(self, y_true) -> bool:
        raise "Not implemented"

    def _calculate_precision_and_recall(self, y_score, threshold) -> bool:
        raise "Not implemented"

    def _precision_recall_curve(self, y_true: np.ndarray, y_score: np.ndarray):
        start_time = time.time()
        self._init_pr_auc_calculation(y_true)

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
        for i, threshold in tqdm(enumerate(thresholds)):
            precision, recall = self._calculate_precision_and_recall(y_score, threshold)
            precisions[i] = precision
            recalls[i] = recall

        # first sort by recall, then by precision to break ties (important for noisy scorings)
        sorted_idx = np.lexsort((precisions * (-1), recalls))[::-1]
        precisions = np.r_[p0, precisions[sorted_idx], 1]

        # There may be some NaN precisions for large thresholds because both TP=0 and FP=0
        # We fill them with values for previous thresholds
        mask = np.isnan(precisions)
        if np.sum(mask) > 0:
            idx = np.where(~mask, np.arange(len(mask)), 0)
            np.maximum.accumulate(idx, out=idx)
            precisions = precisions[idx]

        recalls = np.r_[r0, recalls[sorted_idx], 0]

        print(f"{self.name} calculation time: {time.time() - start_time}")

        return precisions, recalls, thresholds

    def supports_continuous_scorings(self) -> bool:
        return True


class RocAUC(AucMetric):
    """Computes the area under the receiver operating characteristic curve.

    Parameters
    ----------
    plot : bool
        Set this parameter to ``True`` to plot the curve.
    plot_store : bool
        If this parameter is ``True`` the curve plot will be saved in the current working directory under the name
        template "fig-{metric-name}.pdf".

    See Also
    --------
    `https://en.wikipedia.org/wiki/Receiver_operating_characteristic <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_ : Explanation of the ROC-curve.
    """
    def __init__(self, plot: bool = False, plot_store: bool = False) -> None:
        super().__init__(plot, plot_store)

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, roc_curve)

    @property
    def name(self) -> str:
        return "ROC_AUC"


class PrAUC(AucMetric):
    """Computes the area under the precision recall curve.

    Parameters
    ----------
    plot : bool
        Set this parameter to ``True`` to plot the curve.
    plot_store : bool
        If this parameter is ``True`` the curve plot will be saved in the current working directory under the name
        template "fig-{metric-name}.pdf".
    """
    def __init__(self, plot: bool = False, plot_store: bool = False) -> None:
        super().__init__(plot, plot_store)

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, precision_recall_curve)

    @property
    def name(self) -> str:
        return "PR_AUC"
