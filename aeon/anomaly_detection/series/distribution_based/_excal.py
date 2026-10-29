"""ExCal anomaly detector."""

__maintainer__ = []
__all__ = ["ExCal"]

import numpy as np
from sklearn.decomposition import PCA

from aeon.anomaly_detection.series.base import BaseSeriesAnomalyDetector
from aeon.utils.windowing import reverse_windowing, sliding_windows

# fixed constants of the base configuration (Table II of [1]_)
_SHORT_WINDOW = 8  # window length of the short raw-state retrieval bank
_MIN_LONG_WINDOW = 16  # clip range of the period-scale retrieval window
_MAX_LONG_WINDOW = 128
_FALLBACK_LONG_WINDOW = 32  # used when no dominant period is found
_SCALES = (4, 16, 64, 256)  # scales of the rolling robust statistics
_BANK_CAP = 10_000  # maximum number of exemplar windows per retrieval bank
_PCA_SAMPLE_CAP = 4_096  # windows used to fit the retrieval PCA models
_QUERY_CHUNK = 1_024  # query chunk size of the nearest neighbor searches
_PCA_ENERGY = 0.95  # variance kept by the retrieval PCA projections
_SUBSPACE_ENERGY = 0.90  # variance kept by the signal/derivative subspace
_SUBSPACE_MAX_DIMS = 32  # maximum dimension of the signal/derivative subspace
_TOP_COORDS = 3  # number of top coordinates/channels averaged per expert
_EPS = 1e-6

# constants of the dominant period estimation
_PERIOD_MIN_LAG = 4
_PERIOD_MAX_LAG = 400
_PERIOD_THRESHOLD = 0.1
_PERIOD_MAX_POINTS = 20_000


def _estimate_period(Z: np.ndarray) -> int | None:
    """Estimate the dominant period of a normalized series via FFT.

    The autocorrelation of each channel is estimated from the inverse power
    spectrum (Wiener-Khinchin theorem), averaged over channels and normalized by the
    zero-lag value. The lag with the largest autocorrelation among the local maxima
    within lags 4 to ``min(400, n_timepoints / 3)`` is accepted if it exceeds 0.1.

    Parameters
    ----------
    Z : np.ndarray
        Normalized series with shape ``(n_timepoints, n_channels)``.

    Returns
    -------
    int or None
        The dominant period, or None if no candidate exceeds the threshold.
    """
    A = Z[: _PERIOD_MAX_POINTS]
    A = A - A.mean(axis=0)
    n_fft = 1 << int(2 * A.shape[0] - 1).bit_length()
    spec = np.fft.rfft(A, n=n_fft, axis=0)
    acov = np.fft.irfft(spec.real**2 + spec.imag**2, n=n_fft, axis=0)[: A.shape[0]]
    with np.errstate(divide="ignore", invalid="ignore"):
        acf_per_channel = np.where(acov[0] > 0, acov / acov[0], 0.0)
    acf = acf_per_channel.mean(axis=1)
    if not np.isfinite(acf).all():
        return None
    max_lag = min(_PERIOD_MAX_LAG, A.shape[0] // 3)
    if max_lag < _PERIOD_MIN_LAG:
        return None
    lags = np.arange(_PERIOD_MIN_LAG, max_lag + 1)
    is_local_max = (acf[lags] >= acf[lags - 1]) & (acf[lags] >= acf[lags + 1])
    candidates = lags[is_local_max]
    if candidates.size == 0:
        candidates = lags
    best = candidates[np.argmax(acf[candidates])]
    return int(best) if acf[best] > _PERIOD_THRESHOLD else None


def _top_coords_mean(values: np.ndarray) -> np.ndarray:
    """Mean of the largest ``_TOP_COORDS`` entries of each row (all if fewer)."""
    n_top = min(_TOP_COORDS, values.shape[1])
    return np.sort(values, axis=1)[:, -n_top:].mean(axis=1)


class _RetrievalExpert:
    """Raw-state retrieval expert map.

    Scores windows of a series by the distance to the ``k``-th nearest exemplar
    window of a bank of normal training windows, both projected into a PCA space
    retaining ``_PCA_ENERGY`` of the training variance.
    """

    def __init__(
        self,
        Z: np.ndarray,
        window_size: int,
        k: int,
        rng: np.random.Generator,
    ) -> None:
        windows, _ = sliding_windows(Z, window_size=window_size, stride=1, axis=0)
        n_windows = windows.shape[0]
        self.window_size = window_size
        self.k = k
        if n_windows < 2:
            self.pca = None
            self.bank = np.empty((0, window_size * Z.shape[1]))
            self.bank_starts = np.empty(0, dtype=int)
            return
        sample = rng.choice(
            n_windows, size=min(n_windows, _PCA_SAMPLE_CAP), replace=False
        )
        # constant windows make the explained variance ratios undefined
        with np.errstate(invalid="ignore", divide="ignore"):
            self.pca = PCA(n_components=_PCA_ENERGY, svd_solver="full").fit(
                windows[sample]
            )
        projected = self.pca.transform(windows)
        if n_windows > _BANK_CAP:
            selection = np.sort(rng.choice(n_windows, size=_BANK_CAP, replace=False))
        else:
            selection = np.arange(n_windows)
        self.bank = projected[selection]
        # start index of each exemplar window within the training series
        self.bank_starts = selection

    def score(self, Z: np.ndarray, self_overlap: bool = False) -> np.ndarray:
        """Return the k-NN distance score of each window of ``Z``.

        Parameters
        ----------
        Z : np.ndarray
            Normalized series with shape ``(n_timepoints, n_channels)``.
        self_overlap : bool, default=False
            Whether ``Z`` is the training series itself. If so, exemplar windows
            overlapping the query window are excluded from the search.

        Returns
        -------
        np.ndarray
            Array of scores with shape ``(n_timepoints,)``.
        """
        windows, _ = sliding_windows(Z, window_size=self.window_size, stride=1, axis=0)
        n_windows = windows.shape[0]
        if self.pca is None or n_windows == 0:
            return np.zeros(Z.shape[0])
        queries = self.pca.transform(windows)
        bank_sq = (self.bank**2).sum(axis=1)
        k = min(self.k, self.bank.shape[0])
        window_scores = np.empty(n_windows)
        for lo in range(0, n_windows, _QUERY_CHUNK):
            hi = min(lo + _QUERY_CHUNK, n_windows)
            chunk = queries[lo:hi]
            d2 = (
                (chunk**2).sum(axis=1)[:, None]
                + bank_sq[None, :]
                - 2.0 * chunk @ self.bank.T
            )
            np.maximum(d2, 0.0, out=d2)
            if self_overlap:
                d2_raw = d2.copy()
                starts = np.arange(lo, hi)
                overlap = np.abs(starts[:, None] - self.bank_starts[None, :]) < (
                    self.window_size
                )
                d2[overlap] = np.inf
                # fall back to the full bank if the exclusion removes too many
                # exemplars, e.g. for short series
                starved = np.isfinite(d2).sum(axis=1) < k
                if starved.any():
                    d2[starved] = d2_raw[starved]
            window_scores[lo:hi] = np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])
        return reverse_windowing(
            window_scores, window_size=self.window_size, reduction=np.nanmean
        )


class _SubspaceExpert:
    """Signal/derivative subspace expert map.

    Augments each point with its first difference, projects the augmented points
    into a PCA subspace retaining ``_SUBSPACE_ENERGY`` of the training variance
    (at most ``_SUBSPACE_MAX_DIMS`` dimensions) and scores each point by the mean
    of the largest robustly standardized reconstruction residual coordinates.
    """

    def __init__(self, Z: np.ndarray) -> None:
        augmented = np.hstack([Z, np.diff(Z, axis=0, prepend=Z[:1])])
        self.center = augmented.mean(axis=0)
        # a PCA on a single point is undefined; the expert then scores zero
        self._valid = augmented.shape[0] >= 2
        if not self._valid:
            self.components = np.empty((0, augmented.shape[1]))
            self.residual_median = np.zeros(augmented.shape[1])
            self.residual_iqr = np.full(augmented.shape[1], _EPS)
            return
        # constant points make the explained variance ratios undefined
        with np.errstate(invalid="ignore", divide="ignore"):
            pca = PCA(n_components=_SUBSPACE_ENERGY, svd_solver="full").fit(augmented)
        self.components = pca.components_[:_SUBSPACE_MAX_DIMS]
        residual = self._residuals(augmented)
        self.residual_median = np.median(residual, axis=0)
        self.residual_iqr = np.maximum(
            np.percentile(residual, 75, axis=0) - np.percentile(residual, 25, axis=0),
            _EPS,
        )

    def _residuals(self, augmented: np.ndarray) -> np.ndarray:
        centered = augmented - self.center
        return centered - (centered @ self.components.T) @ self.components

    def score(self, Z: np.ndarray, self_overlap: bool = False) -> np.ndarray:
        """Return the subspace residual score of each point of ``Z``.

        Parameters
        ----------
        Z : np.ndarray
            Normalized series with shape ``(n_timepoints, n_channels)``.
        self_overlap : bool, default=False
            Unused, present for a uniform expert interface.

        Returns
        -------
        np.ndarray
            Array of scores with shape ``(n_timepoints,)``.
        """
        augmented = np.hstack([Z, np.diff(Z, axis=0, prepend=Z[:1])])
        if not self._valid:
            return np.zeros(Z.shape[0])
        standardized = np.abs(
            (self._residuals(augmented) - self.residual_median) / self.residual_iqr
        )
        return _top_coords_mean(standardized)


class _RobustStatsExpert:
    """Multi-scale robust statistics expert map.

    Computes a rolling statistic (mean or log standard deviation) per channel at
    each scale, robustly standardizes each channel against its training median
    and IQR, averages the largest standardized channel deviations, and takes the
    maximum over the scales.
    """

    def __init__(self, Z: np.ndarray, use_log_std: bool) -> None:
        self.use_log_std = use_log_std
        self.stats = []
        for scale in _SCALES:
            if scale > Z.shape[0]:
                continue
            stat = self._rolling_stat(Z, scale)
            self.stats.append(
                (
                    scale,
                    np.median(stat, axis=0),
                    np.maximum(
                        np.percentile(stat, 75, axis=0)
                        - np.percentile(stat, 25, axis=0),
                        _EPS,
                    ),
                )
            )

    def _rolling_stat(self, Z: np.ndarray, scale: int) -> np.ndarray:
        windows = np.lib.stride_tricks.sliding_window_view(Z, scale, axis=0)
        if self.use_log_std:
            return np.log(np.maximum(windows.std(axis=-1), _EPS))
        return windows.mean(axis=-1)

    def score(self, Z: np.ndarray, self_overlap: bool = False) -> np.ndarray:
        """Return the robust deviation score of each point of ``Z``.

        Parameters
        ----------
        Z : np.ndarray
            Normalized series with shape ``(n_timepoints, n_channels)``.
        self_overlap : bool, default=False
            Unused, present for a uniform expert interface.

        Returns
        -------
        np.ndarray
            Array of scores with shape ``(n_timepoints,)``.
        """
        n_timepoints = Z.shape[0]
        scores = np.zeros(n_timepoints)
        for scale, median, iqr in self.stats:
            if scale > n_timepoints:
                continue
            stat = self._rolling_stat(Z, scale)
            standardized = np.abs((stat - median) / iqr)
            per_scale = np.full(n_timepoints, -np.inf)
            # a rolling window ending at time point t scores time point t
            per_scale[scale - 1 :] = _top_coords_mean(standardized)
            scores = np.maximum(scores, per_scale)
        return np.maximum(scores, 0.0)


class ExCal(BaseSeriesAnomalyDetector):
    """Excess calibration (ExCal) anomaly detector.

    ExCal [1]_ is an optimization-free, normal-only anomaly detector. It combines
    five "expert maps", each turning a time point into a non-negative normality
    score. During fitting, the experts are built on a normal training series and
    each map is calibrated by two normality floors, its median and a high
    quantile on the training data. When scoring, only the calibrated excess above
    the high floor is retained (Eq. 10), attenuated for maps firing on a large
    fraction of the scored sequence (Eq. 11), and combined across maps with a
    small rank term (Eq. 12). No gradient-based training is involved; fitting
    only builds PCA projections, quantile floors, and exemplar windows.

    The expert maps of the base configuration are:

    1. short raw-state retrieval: distance to the ``k``-th nearest exemplar of a
       bank of PCA-projected training windows of length 8,
    2. period-scale raw-state retrieval: as above, but with windows of length
       ``clip(period, 16, 128)`` derived from the dominant period,
    3. signal/derivative subspace: reconstruction residual of points augmented
       with their first difference in a PCA subspace,
    4. multi-scale rolling mean robust statistics at scales {4, 16, 64, 256},
    5. multi-scale rolling log standard deviation robust statistics.

    Scoring is transductive: the prevalence damping and the rank term are
    computed over the full scored sequence.

    This is a clean-room implementation from the paper; no reference
    implementation was used or exists. It deviates from the paper as follows:

    - The PCA projections use scikit-learn, which centers the data, while the
      paper projects uncentered windows.
    - The two robust statistics experts (rolling mean and rolling log standard
      deviation) are counted as two of the five expert maps.
    - The optional shape expert of the paper (using the context window ``W`` and
      horizon ``H`` parameters) is disabled in the paper's base configuration
      and is not implemented.
    - Rolling statistics are aligned to the end of their window; time points
      before the first full window of a scale score zero.
    - When the self-overlap exclusion of the retrieval experts would remove too
      many exemplars (short series), the full exemplar bank is searched instead.

    Parameters
    ----------
    k : int, default=3
        The number of nearest exemplars of the retrieval experts. The score is
        the distance to the ``k``-th nearest exemplar window.
    period : int or None, default=None
        The dominant period of the series. If None, it is estimated from the
        training series via FFT-based autocorrelation. The window length of the
        period-scale retrieval expert is ``clip(period, 16, 128)``.
    rho : float, default=0.1
        The prevalence knee of Eq. 11. Maps with a positive excess for more than
        this fraction of the scored sequence are attenuated.
    tau : float, default=0.02
        The weight of the rank term of Eq. 12.
    quantile : float, default=0.995
        The quantile of the normality floor of each expert map, as a fraction in
        (0.5, 1).
    random_state : int or None, default=None
        Seed for the subsampling of PCA training samples and exemplar windows of
        the retrieval experts.

    See Also
    --------
    COPOD : Copula-based outlier detection anomaly detector.
    DWT_MLEAD : Discrete wavelet transform and MLE anomaly detector.

    References
    ----------
    .. [1] Özer, Kadir-Kaan, René Ebeling, and Markus Enzweiler. "ExCal:
           Optimization-Free Multivariate Time Series Anomaly Detection via
           Excess Calibration." arXiv preprint arXiv:2610.35568, 2026.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.anomaly_detection.series.distribution_based import ExCal
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(100)
    >>> X = np.sin(2 * np.pi * t / 20) + 0.1 * rng.normal(size=100)
    >>> X[80:84] += 3.0  # inject a spike anomaly
    >>> detector = ExCal(period=20, random_state=0)
    >>> scores = detector.fit(X[:60]).predict(X)
    >>> int(scores.argmax())
    84
    >>> np.round(scores[78:88], 2)
    array([ 9.93, 12.5 , 23.33, 18.81, 22.31, 23.93, 28.21, 18.46, 17.17,
           15.86])
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "anomaly_output_type": "anomaly_scores",
        "learning_type:unsupervised": True,
        "learning_type:semi_supervised": True,
    }

    def __init__(
        self,
        k: int = 3,
        period: int | None = None,
        rho: float = 0.1,
        tau: float = 0.02,
        quantile: float = 0.995,
        random_state: int | None = None,
    ):
        self.k = k
        self.period = period
        self.rho = rho
        self.tau = tau
        self.quantile = quantile
        self.random_state = random_state

        super().__init__(axis=0)

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ExCal":
        self._check_params()
        # reference to the training input to detect scoring of the training series
        self._X_train = X
        self.mean_ = X.mean(axis=0)
        self.std_ = np.maximum(X.std(axis=0), _EPS)
        Z = (X - self.mean_) / self.std_

        if self.period is not None:
            self.period_ = int(self.period)
        else:
            self.period_ = _estimate_period(Z)
        if self.period_ is not None:
            long_window = int(np.clip(self.period_, _MIN_LONG_WINDOW, _MAX_LONG_WINDOW))
        else:
            long_window = _FALLBACK_LONG_WINDOW
        long_window = min(long_window, Z.shape[0])
        short_window = min(_SHORT_WINDOW, Z.shape[0])

        rng = np.random.default_rng(self.random_state)
        self.experts_ = [
            _RetrievalExpert(Z, short_window, self.k, rng),
            _RetrievalExpert(Z, long_window, self.k, rng),
            _SubspaceExpert(Z),
            _RobustStatsExpert(Z, use_log_std=False),
            _RobustStatsExpert(Z, use_log_std=True),
        ]

        # normality floors of each expert map on the training series
        train_scores = self._score_experts(X)
        self.floors_ = np.column_stack(
            [
                np.median(train_scores, axis=0),
                np.quantile(train_scores, self.quantile, axis=0),
            ]
        )
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._score_experts(X)
        # Eq. 10: calibrated excess above the normality floor
        spread = np.maximum(self.floors_[:, 1] - self.floors_[:, 0], _EPS)
        excess = np.clip((scores - self.floors_[:, 1]) / spread, 0.0, None)
        # Eq. 11: prevalence damping of broadly firing maps
        prevalence = (excess > 0).mean(axis=0)
        damping = np.sqrt(self.rho / np.maximum(prevalence, self.rho))
        # Eq. 12: damped excess plus a small rank term per map
        n_timepoints = scores.shape[0]
        ranks = np.empty_like(scores)
        for i in range(scores.shape[1]):
            order = np.argsort(scores[:, i], kind="stable")
            ranks[order, i] = np.arange(n_timepoints) / max(n_timepoints - 1, 1)
        return (damping * excess + self.tau * ranks).sum(axis=1)

    def _score_experts(self, X: np.ndarray) -> np.ndarray:
        """Evaluate all expert maps on ``X`` with shape ``(n_timepoints, 5)``."""
        Z = (X - self.mean_) / self.std_
        self_overlap = X.shape == self._X_train.shape and np.array_equal(
            X, self._X_train
        )
        return np.column_stack(
            [expert.score(Z, self_overlap) for expert in self.experts_]
        )

    def _check_params(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.period is not None and self.period < 1:
            raise ValueError("period must be >= 1 or None")
        if self.rho <= 0:
            raise ValueError("rho must be > 0")
        if self.tau < 0:
            raise ValueError("tau must be >= 0")
        if self.quantile <= 0.5 or self.quantile >= 1:
            raise ValueError("quantile must be in (0.5, 1)")

    @classmethod
    def _get_test_params(cls, parameter_set="default") -> dict:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid
            test instance.
        """
        return {"k": 2, "period": 8, "random_state": 0}
