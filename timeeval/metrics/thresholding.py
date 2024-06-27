import abc
import os
import pickle
from typing import Optional, Tuple, Any
import more_itertools as mit

import numpy as np
import pandas as pd


class ThresholdingStrategy(abc.ABC):
    """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.

    Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
    All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
    The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
    already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
    classification results.
    """
    def __int__(self) -> None:
        self.threshold: Optional[float] = None

    def fit(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        """Calls :func:`~timeeval.metrics.thresholding.ThresholdingStrategy.find_threshold` to compute and set the
        threshold.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).
        """
        self.threshold = self.find_threshold(y_true, y_score)

    def transform(self, y_score: np.ndarray) -> np.ndarray:
        """Applies the threshold to the anomaly scoring and returns the corresponding binary labels.

        Parameters
        ----------
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        y_pred : np.ndarray
            Array of binary labels; 0 for normal points and 1 for anomalous points.
        """
        return (y_score >= self.threshold).astype(np.int_)

    def fit_transform(self, y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
        """Determines the threshold and applies it to the scoring in one go.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        y_pred : np.ndarray
            Array of binary labels; 0 for normal points and 1 for anomalous points.

        See Also
        --------
        ~timeeval.metrics.thresholding.ThresholdingStrategy.fit : fit-function to determine the threshold.
        ~timeeval.metrics.thresholding.ThresholdingStrategy.transform :
            transform-function to calculate the binary predictions.
        """
        self.fit(y_true, y_score)
        return self.transform(y_score)

    @abc.abstractmethod
    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Abstract method containing the actual code to determine the threshold. Must be overwritten by subclasses!"""
        pass


class NoThresholding(ThresholdingStrategy):
    """Special no-op strategy that checks for already existing binary labels and keeps them untouched. This allows
    applying the metrics on existing binary classification results.
    """

    def fit(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        """Does nothing (no-op).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).
        """
        pass

    def transform(self, y_score: np.ndarray) -> np.ndarray:
        """Checks if the provided scoring `y_score` is actually a binary classification prediction of integer type. If
        this is the case, the prediction is returned. If not, a :class:`ValueError` is raised.

        Parameters
        ----------
        y_score : np.ndarray
            Anomaly scoring with binary predictions.

        Returns
        -------
        y_pred : np.ndarray
            Array of binary labels; 0 for normal points and 1 for anomalous points.
        """
        if y_score.dtype.kind not in ["i", "u"]:
            raise ValueError("The NoThresholding strategy can only be used for binary predictions (either 0 or 1). "
                             "Continuous anomaly scorings are not supported, please use any other thresholding "
                             "strategy for this!")
        return y_score

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Does nothing (no-op).

        Parameters
        ----------
        y_true : np.ndarray
            Ignored.
        y_score : np.ndarray
            Ignored.

        Returns
        -------
        None
        """
        pass

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"NoThresholding()"


class FixedValueThresholding(ThresholdingStrategy):
    """Thresholding approach using a fixed threshold value.

    Parameters
    ----------
    threshold : float
        Fixed threshold to use. All anomaly scorings are scaled to the interval [0, 1]
    """
    def __init__(self, threshold: float = 0.8):
        if threshold > 1 or threshold < 0:
            raise ValueError(f"Threshold must be in the interval [0, 1], but was {threshold}!")
        self.threshold = threshold

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Returns the fixed threshold."""
        return self.threshold  # type: ignore

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"FixedValueThresholding(threshold={repr(self.threshold)})"


class PercentileThresholding(ThresholdingStrategy):
    """Use the xth-percentile of the anomaly scoring as threshold.

    Parameters
    ----------
    percentile : int
        The percentile of the anomaly scoring to use. Must be between 0 and 100.
    """
    def __init__(self, percentile: int = 90):
        if percentile < 0 or percentile > 100:
            raise ValueError(f"Percentile must be within [0, 100], but was {percentile}!")
        self._percentile = percentile

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Computes the xth-percentile ignoring NaNs and using a linear interpolation.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        threshold : float
            The xth-percentile of the anomaly scoring as threshold.
        """
        return np.nanpercentile(y_score, self._percentile)  # type: ignore

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"PercentileThresholding(percentile={repr(self._percentile)})"


class TopKPointsThresholding(ThresholdingStrategy):
    """Calculates a threshold so that exactly `k` points are marked anomalous.

    Parameters
    ----------
    k : optional int
        Number of expected anomalous points. If `k` is `None`, the ground truth data is used to calculate the real
        number of anomalous points.
    """
    def __init__(self, k: Optional[int] = None):
        if k is not None and k <= 0:
            raise ValueError(f"K must be greater than 0, but was {k}!")
        self._k: Optional[int] = k

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Computes a threshold based on the number of expected anomalous points.

        The threshold is determined by taking the reciprocal ratio of expected anomalous points to all points as target
        percentile. We, again, ignore NaNs and use a linear interpolation.
        If `k` is `None`, the ground truth data is used to calculate the real ratio of anomalous points to all points.
        Otherwise, `k` is used as the number of expected anomalous points.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        threshold : float
            Threshold that yields k anomalous points.
        """
        if self._k is None:
            return np.nanpercentile(y_score, (1 - y_true.sum() / y_true.shape[0])*100)  # type: ignore
        else:
            return np.nanpercentile(y_score, (1 - self._k / y_true.shape[0]) * 100)  # type: ignore

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"TopKPointsThresholding(k={repr(self._k)})"


class TopKRangesThresholding(ThresholdingStrategy):
    """Calculates a threshold so that exactly `k` anomalies are found. The anomalies are either single-points anomalies
    or continuous anomalous ranges.

    Parameters
    ----------
    k : optional int
        Number of expected anomalies. If `k` is `None`, the ground truth data is used to calculate the real number of
        anomalies.
    """
    def __init__(self, k: Optional[int] = None):
        if k is not None and k <= 0:
            raise ValueError(f"K must be greater than 0, but was {k}!")
        self._k: Optional[int] = k

    @staticmethod
    def _count_anomaly_ranges(y_pred: np.ndarray) -> int:
        return int(np.sum(np.diff(np.r_[0, y_pred, 0]) == 1))

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Computes a threshold based on the number of expected anomalous subsequences / ranges (number of anomalies).

        This method iterates over all possible thresholds from high to low to find the first threshold that yields `k`
        or more continuous anomalous ranges.

        If `k` is `None`, the ground truth data is used to calculate the real number of anomalies (anomalous ranges).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        threshold : float
            Threshold that yields k anomalies.
        """
        if self._k is None:
            self._k = self._count_anomaly_ranges(y_true)
        thresholds: Tuple[float] = np.unique(y_score)[::-1]
        t = thresholds[0]
        y_pred = np.array(y_score >= t, dtype=np.int_)
        # exclude minimum from thresholds, because all points are >= minimum!
        for t in thresholds[1:-1]:
            y_pred = np.array(y_score >= t, dtype=np.int_)
            detected_n = self._count_anomaly_ranges(y_pred)
            if detected_n >= self._k:
                break
        return t

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"TopKRangesThresholding(k={repr(self._k)})"


class SigmaThresholding(ThresholdingStrategy):
    """Computes a threshold :math:`\\theta` based on the anomaly scoring's mean :math:`\\mu_s` and the
    standard deviation :math:`\\sigma_s`:

    .. math::
       \\theta = \\mu_{s} + x \\cdot \\sigma_{s}

    Parameters
    ----------
    factor: float
        Multiples of the standard deviation to be added to the mean to compute the threshold (:math:`x`).
    """
    def __init__(self, factor: float = 3.0):
        if factor <= 0:
            raise ValueError(f"factor must be greater than 0, but was {factor}!")
        self._factor = factor

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Determines the mean and standard deviation ignoring NaNs of the anomaly scoring and computes the
        threshold using the mentioned equation.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        threshold : float
            Computed threshold based on mean and standard deviation.
        """
        return np.nanmean(y_score) + self._factor * np.nanstd(y_score)  # type: ignore

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"SigmaThresholding(factor={repr(self._factor)})"


class PyThreshThresholding(ThresholdingStrategy):
    """Uses a thresholder from the `PyThresh <https://github.com/KulikDM/pythresh>`_ package to find a scoring
    threshold and to transform the continuous anomaly scoring into binary anomaly predictions.

    .. warning::
      You need to install PyThresh before you can use this thresholding strategy:

      .. code-block:: bash

        pip install pythresh

      Please note the additional package requirements for some available thresholders of PyThresh.

    Parameters
    ----------
    pythresh_thresholder : pythresh.thresholds.base.BaseThresholder
        Initiated PyThresh thresholder.
    random_state: Any
        Seed used to seed the numpy random number generator used in some thresholders of PyThresh. Note that PyThresh
        uses the legacy global RNG (``np.random``) and we try to reset the global RNG after calling PyThresh. Can be
        left at its default value for most thresholders that don't use random numbers or provide their own way of
        seeding. Please consult the `PyThresh Documentation <https://pythresh.readthedocs.io/en/latest/index.html>`_
        for details about the individual thresholders.

    Examples
    --------
    .. code-block:: python

      from timeeval.metrics.thresholding import PyThreshThresholding
      from pythresh.thresholds.regr import REGR
      import numpy as np

      thresholding = PyThreshThresholding(
          REGR(method="theil")
      )

      y_scores = np.random.default_rng().random(1000)
      y_labels = np.zeros(1000)
      y_pred = thresholding.fit_transform(y_labels, y_scores)
    """

    def __init__(self, pythresh_thresholder: 'BaseThresholder', random_state: Any = None):  # type: ignore
        self._thresholder = pythresh_thresholder
        self._predictions: Optional[np.ndarray] = None
        self._random_state: Any = random_state

    @staticmethod
    def _make_finite(y_score: np.ndarray) -> np.ndarray:
        """Replaces NaNs with 0 and (Neg)Infs with 1."""
        nan_mask = np.isnan(y_score)
        inf_mask = np.isinf(y_score)
        neginf_mask = np.isneginf(y_score)
        tmp = y_score.copy()
        tmp[nan_mask] = 0
        tmp[inf_mask | neginf_mask] = 1
        return tmp

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Uses the passed thresholder from the `PyThresh <https://github.com/KulikDM/pythresh>`_ package to determine
        the threshold. Beforehand, the scores are forced to be finite by replacing NaNs with 0 and (Neg)Infs with 1.

        PyThresh thresholders directly compute the binary predictions. Thus, we cache the predictions in the member
        ``_predictions`` and return them when calling
        :func:`~timeeval.metrics.thresholding.PyThreshThresholding.transform`.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        threshold : float
            Threshold computed by the internal thresholder.
        """
        y_score = self._make_finite(y_score)

        # seed legacy np.random for reproducibility
        old_state = np.random.get_state()
        np.random.seed(self._random_state)

        # call PyThresh
        self._predictions = self._thresholder.eval(y_score)
        threshold: float = self._thresholder.thresh_

        # reset np.random state
        np.random.set_state(old_state)
        return threshold

    def transform(self, y_score: np.ndarray) -> np.ndarray:
        if self._predictions is not None:
            return self._predictions
        else:
            return (y_score >= self.threshold).astype(np.int_)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"PyThreshThresholding(pythresh_thresholding={repr(self._thresholder)})"


class DynamicThresholdingStrategy(abc.ABC):
    """Takes an anomaly scoring and ground truth labels to compute dynamically thresholded output.

    Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
    All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
    The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
    already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
    classification results.
    """
    def __int__(self) -> None:
        self.threshold: Optional[float] = None

    def fit_transform(self, y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
        """Determines the threshold and applies it to the scoring in one go.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        y_pred : np.ndarray
            Array of binary labels; 0 for normal points and 1 for anomalous points.

        See Also
        --------
        ~timeeval.metrics.thresholding.ThresholdingStrategy.fit : fit-function to determine the threshold.
        ~timeeval.metrics.thresholding.ThresholdingStrategy.transform :
            transform-function to calculate the binary predictions.
        """
        return self.find_threshold(y_true, y_score)

    @abc.abstractmethod
    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
        """Abstract method containing the actual code to determine the threshold. Must be overwritten by subclasses!"""
        pass

class ErrorWindow:
    def __init__(self, e_s, y_true, start_idx, end_idx, window_num, batch_size, error_buffer, p, window_size,
                 min_error_value = 0.05):
        """
        Data and calculations for a specific window of prediction errors.
        Includes finding thresholds, pruning, and scoring anomalous sequences
        for errors and inverted errors (flipped around mean) - significant drops
        in values can also be anomalous.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            config (obj): Config object containing parameters for processing
            start_idx (int): Starting index for window within full set of
                channel test values
            end_idx (int): Ending index for window within full set of channel
                test values
            errors (arr): Errors class object
            window_num (int): Current window number within channel test values
            min_error_value (float): minimum reconstruction error value to consider

        Attributes:
            i_anom (arr): indices of anomalies in window
            i_anom_inv (arr): indices of anomalies in window of inverted
                telemetry values
            E_seq (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window
            E_seq_inv (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window of inverted telemetry
                values
            non_anom_max (float): highest smoothed error value below epsilon
            non_anom_max_inv (float): highest smoothed error value below
                epsilon_inv
            config (obj): see Args
            anom_scores (arr): score indicating relative severity of each
                anomaly sequence in E_seq within a window
            window_num (int): see Args
            sd_lim (int): default number of standard deviations to use for
                threshold if no winner or too many anomalous ranges when scoring
                candidate thresholds
            sd_threshold (float): number of standard deviations for calculation
                of best anomaly threshold
            sd_threshold_inv (float): same as above for inverted channel values
            e_s (arr): exponentially-smoothed prediction errors in window
            e_s_inv (arr): inverted e_s
            sd_e_s (float): standard deviation of e_s
            mean_e_s (float): mean of e_s
            epsilon (float): threshold for e_s above which an error is
                considered anomalous
            epsilon_inv (float): threshold for inverted e_s above which an error
                is considered anomalous
            y_test (arr): Actual telemetry values for window
            sd_values (float): st dev of y_test
            perc_high (float): the 95th percentile of y_test values
            perc_low (float): the 5th percentile of y_test values
            inter_range (float): the range between perc_high - perc_low
            num_to_ignore (int): number of values to ignore initially when
                looking for anomalies
        """
        self.batch_size = batch_size
        self.error_buffer = error_buffer
        self.p = p
        self.min_error_value = min_error_value
        self.i_anom = np.array([])
        self.E_seq = np.array([])
        self.non_anom_max = -np.inf
        self.i_anom_inv = np.array([])
        self.E_seq_inv = np.array([])
        self.non_anom_max_inv = -np.inf

        self.anom_scores = []

        self.window_num = window_num

        self.sd_lim = 12.0
        self.sd_threshold = self.sd_lim
        self.sd_threshold_inv = self.sd_lim

        self.e_s = e_s[start_idx:end_idx]

        self.mean_e_s = np.mean(self.e_s)
        self.sd_e_s = np.std(self.e_s)
        self.e_s_inv = np.array([self.mean_e_s + (self.mean_e_s - e)
                                 for e in self.e_s])

        self.epsilon = self.mean_e_s + self.sd_lim * self.sd_e_s
        self.epsilon_inv = self.mean_e_s + self.sd_lim * self.sd_e_s

        self.y_test = y_true[start_idx:end_idx]
        self.sd_values = np.std(self.y_test)

        self.perc_high, self.perc_low = np.percentile(self.y_test, [95, 5])
        self.inter_range = self.perc_high - self.perc_low

        # ignore initial error values until enough history for processing
        self.num_to_ignore = window_size * 2
        # if y_test is small, ignore fewer
        if len(y_true) < 2500:
            self.num_to_ignore = window_size
        if len(y_true) < 1800:
            self.num_to_ignore = 0

    def find_epsilon(self, inverse=False):
        """
        Find the anomaly threshold that maximizes function representing
        tradeoff between:
            a) number of anomalies and anomalous ranges
            b) the reduction in mean and st dev if anomalous points are removed
            from errors
        (see https://arxiv.org/pdf/1802.04431.pdf)

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """
        if self.sd_e_s == 0.0:
            return

        e_s = self.e_s if not inverse else self.e_s_inv

        max_score = -np.inf

        for z in np.arange(2.5, self.sd_lim, 0.5):
            epsilon = self.mean_e_s + (self.sd_e_s * z)

            pruned_e_s = e_s[e_s < epsilon]

            i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
            buffer = np.arange(1, self.error_buffer)
            i_anom = np.sort(np.concatenate((i_anom,
                                            np.array([i+buffer for i in i_anom])
                                             .flatten(),
                                            np.array([i-buffer for i in i_anom])
                                             .flatten())))
            i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
            i_anom = np.sort(np.unique(i_anom))

            if len(i_anom) > 0:
                # group anomalous indices into continuous sequences
                groups = [list(group) for group
                          in mit.consecutive_groups(i_anom)]
                E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

                mean_perc_decrease = (self.mean_e_s - np.mean(pruned_e_s)) \
                                     / self.mean_e_s
                sd_perc_decrease = (self.sd_e_s - np.std(pruned_e_s)) \
                                   / self.sd_e_s
                score = (mean_perc_decrease + sd_perc_decrease) \
                        / (len(E_seq) ** 2 + len(i_anom))

                # sanity checks / guardrails
                if score >= max_score and len(E_seq) <= 5 and \
                        len(i_anom) < (len(e_s) * 0.5):
                    max_score = score
                    if not inverse:
                        self.sd_threshold = z
                        self.epsilon = self.mean_e_s + z * self.sd_e_s
                    else:
                        self.sd_threshold_inv = z
                        self.epsilon_inv = self.mean_e_s + z * self.sd_e_s
            else:
                break

    def compare_to_epsilon(self, errors_all, inverse=False):
        """
        Compare smoothed error values to epsilon (error threshold) and group
        consecutive errors together into sequences.

        Args:
            errors_all (obj): Errors class object containing list of all
            previously identified anomalies in test set
        """

        e_s = self.e_s if not inverse else self.e_s_inv
        epsilon = self.epsilon if not inverse else self.epsilon_inv

        # Check: scale of errors compared to values too small?
        if not (self.sd_e_s > (.05 * self.sd_values) or max(self.e_s) > (.05 * self.inter_range)) \
                or len(e_s) < self.error_buffer or not max(self.e_s) > self.min_error_value:
            return

        i_anom = np.argwhere((e_s >= epsilon) &
                             (e_s > 0.05 * self.inter_range)).reshape(-1,)

        if len(i_anom) == 0:
            return
        buffer = np.arange(1, self.error_buffer+1)
        i_anom = np.sort(np.concatenate((i_anom,
                                         np.array([i + buffer for i in i_anom])
                                         .flatten(),
                                         np.array([i - buffer for i in i_anom])
                                         .flatten())))
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]

        # if it is first window, ignore initial errors (need some history)
        if self.window_num == 0:
            i_anom = i_anom[i_anom >= self.num_to_ignore]
        else:
            i_anom = i_anom[i_anom >= len(e_s) - self.batch_size]

        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) == 0:
            return

        # capture max of non-anomalous values below the threshold
        # (used in filtering process)
        batch_position = self.window_num * self.batch_size
        window_indices = np.arange(0, len(e_s)) + batch_position
        adj_i_anom = i_anom + batch_position
        window_indices = np.setdiff1d(window_indices,
                                      np.append(errors_all.i_anom, adj_i_anom))
        candidate_indices = np.unique(window_indices - batch_position)
        non_anom_max = np.max(np.take(e_s, candidate_indices))

        # group anomalous indices into continuous sequences
        groups = [list(group) for group in mit.consecutive_groups(i_anom)]
        E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

        if inverse:
            self.i_anom_inv = i_anom
            self.E_seq_inv = E_seq
            self.non_anom_max_inv = non_anom_max
        else:
            self.i_anom = i_anom
            self.E_seq = E_seq
            self.non_anom_max = non_anom_max

    def prune_anoms(self, inverse=False):
        """
        Remove anomalies that don't meet minimum separation from the next
        closest anomaly or error value

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """

        E_seq = self.E_seq if not inverse else self.E_seq_inv
        e_s = self.e_s if not inverse else self.e_s_inv
        non_anom_max = self.non_anom_max if not inverse \
            else self.non_anom_max_inv

        if len(E_seq) == 0:
            return

        E_seq_max = np.array([max(e_s[e[0]:e[1]+1]) for e in E_seq])
        E_seq_max_sorted = np.sort(E_seq_max)[::-1]
        E_seq_max_sorted = np.append(E_seq_max_sorted, [non_anom_max])

        i_to_remove = np.array([], dtype=bool)
        for i in range(0, len(E_seq_max_sorted)-1):
            if (E_seq_max_sorted[i] - E_seq_max_sorted[i+1]) \
                    / E_seq_max_sorted[i] < self.p:
                i_to_remove = np.append(i_to_remove, np.argwhere(
                    E_seq_max == E_seq_max_sorted[i]))
            else:
                i_to_remove = np.array([], dtype=bool)
        i_to_remove[::-1].sort()

        if len(i_to_remove) > 0:
            E_seq = np.delete(E_seq, i_to_remove, axis=0)

        if len(E_seq) == 0 and inverse:
            self.i_anom_inv = np.array([])
            return
        elif len(E_seq) == 0 and not inverse:
            self.i_anom = np.array([])
            return

        indices_to_keep = np.concatenate([range(e_seq[0], e_seq[-1]+1)
                                          for e_seq in E_seq])

        if not inverse:
            mask = np.isin(self.i_anom, indices_to_keep)
            self.i_anom = self.i_anom[mask]
        else:
            mask_inv = np.isin(self.i_anom_inv, indices_to_keep)
            self.i_anom_inv = self.i_anom_inv[mask_inv]

    def score_anomalies(self, prior_idx):
        """
        Calculate anomaly scores based on max distance from epsilon
        for each anomalous sequence.

        Args:
            prior_idx (int): starting index of window within full set of test
                values for channel
        """

        groups = [list(group) for group in mit.consecutive_groups(self.i_anom)]

        for e_seq in groups:

            score_dict = {
                "start_idx": e_seq[0] + prior_idx,
                "end_idx": e_seq[-1] + prior_idx,
                "score": 0
            }

            score = max([abs(self.e_s[i] - self.epsilon)
                         / (self.mean_e_s + self.sd_e_s) for i in
                         range(e_seq[0], e_seq[-1] + 1)])
            inv_score = max([abs(self.e_s_inv[i] - self.epsilon_inv)
                             / (self.mean_e_s + self.sd_e_s) for i in
                             range(e_seq[0], e_seq[-1] + 1)])

            # the max score indicates whether anomaly was from regular
            # or inverted errors
            score_dict['score'] = max([score, inv_score])
            self.anom_scores.append(score_dict)


class TelemanomThresholding(ThresholdingStrategy):
    """Computes a threshold using non-parametric dynamic thresholding (NDT)

    Parameters
    ----------
    smoothing_window_size: int
        Window size used for smoothing the errors
    """
    def __init__(self, batch_size: int = 70, smoothing_window_size: int = 30, smoothing_perc: float = 0.05, error_buffer: int = 100, p: float = 0.13, window_size: int = 250, min_error_value: float = 0.05):
        self.batch_size = batch_size
        self.smoothing_window_size = smoothing_window_size
        self.smoothing_perc = smoothing_perc
        self.error_buffer = error_buffer
        self.p = p
        self.window_size = window_size
        self.min_error_value = min_error_value
        self.i_anom = np.array([])
        self.E_seq = []
        self.anom_scores = []

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Parameters
        ----------
        y_true : np.ndarray
            Test data values
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        threshold : float
            Computed threshold based on mean and standard deviation.
        """
        # smoothed prediction error
        smoothing_window = max(1, int(self.batch_size * self.smoothing_window_size * self.smoothing_perc))
        e_s = pd.DataFrame(np.abs(y_score - y_true)).ewm(span=smoothing_window, axis=0).mean().values.flatten()

        self.nb_samples = len(y_true)
        self.n_windows = int(np.ceil(self.nb_samples / self.batch_size))

        while self.n_windows < 0:
            self.smoothing_window_size -= 1
            self.n_windows = int((len(y_true) - (self.batch_size * self.smoothing_window_size)) / self.batch_size)
            if self.smoothing_window_size == 1 and self.n_windows < 0:
                raise ValueError('Batch_size ({}) larger than y_true (len={}). '
                                 .format(self.batch_size, len(y_true)))

        for i in range(0, self.n_windows):
            prior_idx = i * self.batch_size
            idx = prior_idx + (self.smoothing_window_size * self.batch_size)
            if i == self.n_windows - 1:
                idx = len(y_true)

            window = ErrorWindow(e_s, y_true, prior_idx, idx, i, self.batch_size, self.error_buffer, self.p, self.window_size, self.min_error_value)

            window.find_epsilon()
            window.find_epsilon(inverse=True)

            window.compare_to_epsilon(self)
            window.compare_to_epsilon(self, inverse=True)

            if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                continue

            window.prune_anoms()
            window.prune_anoms(inverse=True)

            if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                continue

            window.i_anom = np.sort(np.unique(np.append(window.i_anom, window.i_anom_inv))).astype('int')
            window.score_anomalies(prior_idx)

            # update indices to reflect true indices in full set of values
            self.i_anom = np.append(self.i_anom, window.i_anom + prior_idx)
            self.anom_scores = self.anom_scores + window.anom_scores

        if len(self.i_anom) > 0:
            # group anomalous indices into continuous sequences
            groups = [list(group) for group in
                      mit.consecutive_groups(self.i_anom)]
            self.E_seq = [(int(g[0]), int(g[-1])) for g in groups
                          if not g[0] == g[-1]]

        thresholded_scores = np.zeros_like(y_true)
        for start_idx, end_idx in self.E_seq:
            thresholded_scores[start_idx:end_idx] = 1

        return thresholded_scores  # type: ignore

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"TelemanomThresholding(batch_size={repr(self.batch_size)}, smoothing_window_size={repr(self.smoothing_window_size)}," \
               f"smoothing_perc={repr(self.smoothing_perc)}, error_buffer={repr(self.error_buffer)}, p={repr(self.p)})"


class DcVaeThresholding(ThresholdingStrategy):
    """Thresholding using alpha deviations from mean from DC-VAE algorithm

    Parameters
    ----------
    alpha: float
        Window size used for smoothing the errors
    """
    def __init__(self, alpha: float = None, alpha_up_pickle: str = "", alpha_down_pickle: str = ""):
        if alpha is not None:
            self.alpha_up = alpha
            self.alpha_down = alpha
        else:
            self.alpha_up = 3
            self.alpha_down = 3

        if os.path.isfile(alpha_up_pickle):
            with open(alpha_up_pickle, 'rb') as f:
                self.alpha_up = pickle.load(f)

        if os.path.isfile(alpha_down_pickle):
            with open(alpha_down_pickle, 'rb') as f:
                self.alpha_down = pickle.load(f)


    def find_threshold(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Parameters
        ----------
        y_true : np.ndarray
            Test data values
        y_pred : np.ndarray
            Anomaly scoring with continuous anomaly scores and deviations for each sample (2D array)

        Returns
        -------
        threshold : float
            Computed threshold based on mean and standard deviation.
        """

        y_mean = y_pred[0]
        y_dev = y_pred[1]

        return ((y_true < y_mean - self.alpha_down * y_dev) | (y_true > y_mean + self.alpha_up * y_dev)).astype(np.uint8)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"DcVaeThresholding(alpha_up={repr(self.alpha_up)},alpha_down={repr(self.alpha_down)})"


class DcVaeAnomalyScoring(ThresholdingStrategy):
    """Thresholding using alpha deviations from mean from DC-VAE algorithm

    Parameters
    ----------
    alpha: float
        Window size used for smoothing the errors
    """
    def __init__(self, min_alpha: float = 2):
        self.min_alpha = min_alpha


    def find_threshold(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Parameters
        ----------
        y_true : np.ndarray
            Test data values
        y_pred : np.ndarray
            Anomaly scoring with continuous anomaly scores and deviations for each sample (2D array)

        Returns
        -------
        threshold : float
            Computed threshold based on mean and standard deviation.
        """

        y_mean = y_pred[0]
        y_dev = y_pred[1]

        anomaly_score = np.abs(y_true - y_mean)
        anomaly_score[anomaly_score < self.min_alpha * y_dev] = 0
        anomaly_score[anomaly_score > 0] -= self.min_alpha * y_dev[anomaly_score > 0]

        return anomaly_score

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"DcVaeAnomalyScoring(min_alpha={repr(self.min_alpha)})"
