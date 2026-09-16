"""AutoPlait automatic multi-regime segmentation.

Clean-room implementation of AutoPlait, a parameter-free, MDL-based
multi-regime segmenter for co-evolving (multivariate) time series. The series
is split into segments, each segment is assigned to one of an automatically
chosen number of *regimes*, and each regime is modelled by a multivariate
Gaussian hidden Markov model (HMM). The number of regimes, the number of
segments and the per-regime model complexity are all selected by the Minimum
Description Length (MDL) principle, so the method has no threshold to tune.

This module was written **from the SIGMOD 2014 paper only** (see the class
``References``). No reference implementation source was read, adapted or
copied; the only public references are unlicensed and depend on ``hmmlearn``,
which aeon does not use. The Gaussian HMM (Baum-Welch fitting, Viterbi
decoding and log-space scoring) is implemented from scratch with
``numpy``/``scipy`` only.
"""

__maintainer__ = []
__all__ = ["AutoPlaitSegmenter"]

import numpy as np
from scipy.special import logsumexp

from aeon.segmentation.base import BaseSegmenter

# Number of bits used to encode a single continuous model parameter. AutoPlait
# uses a constant float cost in its model description length; the exact value is
# a coding convention rather than an algorithmic parameter (see VALIDATION.md).
_COST_FLOAT = 32.0
_LOG2 = np.log(2.0)
# Constant of Rissanen's universal code for the integers.
_LOG_STAR_C = np.log2(2.865064)


def _log_star(n):
    """Universal code length (in bits) for a positive integer ``n``.

    Implements Rissanen's ``log*`` code, ``log2(n) + log2(log2(n)) + ...``
    summing the successive positive iterated logarithms plus a constant. Used
    to encode integer quantities (number of regimes, segments and states) in
    the MDL model cost.

    Parameters
    ----------
    n : int
        Positive integer to encode. Values below 1 are treated as 1.

    Returns
    -------
    float
        Description length in bits.
    """
    n = max(int(n), 1)
    total = _LOG_STAR_C
    logx = np.log2(n)
    while logx > 0:
        total += logx
        logx = np.log2(logx)
    return float(total)


class _GaussianHMM:
    """Multivariate Gaussian HMM with diagonal covariance (from scratch).

    A hidden Markov model over ``D``-dimensional observations with ``k`` hidden
    states. Each state has a Gaussian emission with its own mean and diagonal
    covariance. Fitting uses Baum-Welch (the forward-backward EM algorithm);
    decoding uses Viterbi; scoring returns the forward log-likelihood. All the
    recursions are computed in log-space with ``log-sum-exp`` for numerical
    stability.

    Parameters
    ----------
    n_states : int
        Number of hidden states ``k``.
    n_dims : int
        Observation dimensionality ``D``.
    random_state : numpy.random.Generator
        Random generator used to initialise the emission means.
    max_iter : int, default=30
        Maximum number of Baum-Welch iterations.
    tol : float, default=1e-4
        Relative log-likelihood improvement below which EM stops.
    var_floor : float, default=1e-4
        Lower bound applied to every variance to avoid singular emissions.

    Attributes
    ----------
    startprob_ : numpy.ndarray, shape (n_states,)
        Initial state distribution ``pi``.
    transmat_ : numpy.ndarray, shape (n_states, n_states)
        Row-stochastic transition matrix ``A``.
    means_ : numpy.ndarray, shape (n_states, n_dims)
        Per-state emission means.
    variances_ : numpy.ndarray, shape (n_states, n_dims)
        Per-state diagonal emission variances.
    """

    def __init__(
        self,
        n_states,
        n_dims,
        random_state,
        max_iter=30,
        tol=1e-4,
        var_floor=1e-4,
    ):
        self.n_states = int(n_states)
        self.n_dims = int(n_dims)
        self.random_state = random_state
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.var_floor = float(var_floor)

        k = self.n_states
        self.startprob_ = np.full(k, 1.0 / k)
        self.transmat_ = np.full((k, k), 1.0 / k)
        self.means_ = np.zeros((k, self.n_dims))
        self.variances_ = np.ones((k, self.n_dims))

    def _init_params(self, sequences):
        """Initialise emission parameters from the pooled observations.

        Means are seeded from randomly chosen observations and the variance is
        set to the global per-dimension variance (floored). The initial and
        transition distributions start uniform.

        Parameters
        ----------
        sequences : list of numpy.ndarray
            Observation sequences, each of shape ``(T_i, n_dims)``.
        """
        data = np.concatenate(sequences, axis=0)
        k = self.n_states
        n = data.shape[0]
        idx = self.random_state.choice(n, size=k, replace=(n < k))
        self.means_ = data[idx].astype(float).copy()
        # Perturb duplicated seeds so identical rows do not collapse states.
        self.means_ += self.random_state.normal(scale=1e-6, size=self.means_.shape)
        global_var = np.maximum(data.var(axis=0), self.var_floor)
        self.variances_ = np.tile(global_var, (k, 1))
        self.startprob_ = np.full(k, 1.0 / k)
        self.transmat_ = np.full((k, k), 1.0 / k)

    def _log_emission(self, X):
        """Log Gaussian emission density for every observation and state.

        Parameters
        ----------
        X : numpy.ndarray, shape (T, n_dims)
            Observation sequence.

        Returns
        -------
        numpy.ndarray, shape (T, n_states)
            ``log p(x_t | state=j)`` in natural log.
        """
        # (T, 1, D) - (1, k, D) -> (T, k, D)
        diff = X[:, None, :] - self.means_[None, :, :]
        var = self.variances_[None, :, :]
        log_det = np.sum(np.log(2.0 * np.pi * self.variances_), axis=1)  # (k,)
        quad = np.sum(diff * diff / var, axis=2)  # (T, k)
        return -0.5 * (log_det[None, :] + quad)

    def _forward(self, log_startprob, log_transmat, log_b):
        """Run the forward recursion in log-space.

        Returns
        -------
        log_alpha : numpy.ndarray, shape (T, n_states)
        loglik : float
            Sequence log-likelihood.
        """
        t_len = log_b.shape[0]
        log_alpha = np.empty((t_len, self.n_states))
        log_alpha[0] = log_startprob + log_b[0]
        for t in range(1, t_len):
            log_alpha[t] = (
                logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0) + log_b[t]
            )
        return log_alpha, logsumexp(log_alpha[-1])

    def _backward(self, log_transmat, log_b):
        """Run the backward recursion in log-space.

        Returns
        -------
        numpy.ndarray, shape (T, n_states)
            Log-beta values.
        """
        t_len = log_b.shape[0]
        log_beta = np.zeros((t_len, self.n_states))
        for t in range(t_len - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_transmat + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1
            )
        return log_beta

    def fit(self, sequences):
        """Fit the HMM to one or more observation sequences by Baum-Welch.

        Parameters
        ----------
        sequences : numpy.ndarray or list of numpy.ndarray
            A single ``(T, n_dims)`` array or a list of such arrays. Sufficient
            statistics are accumulated across all supplied sequences.

        Returns
        -------
        self : _GaussianHMM
            The fitted model.
        """
        if isinstance(sequences, np.ndarray):
            sequences = [sequences]
        sequences = [np.asarray(s, dtype=float) for s in sequences if len(s) > 0]
        if len(sequences) == 0:
            raise ValueError("Cannot fit a HMM to empty data.")
        self._init_params(sequences)

        k, d = self.n_states, self.n_dims
        prev_ll = -np.inf
        for _ in range(self.max_iter):
            log_startprob = np.log(self.startprob_ + 1e-300)
            log_transmat = np.log(self.transmat_ + 1e-300)

            start_acc = np.zeros(k)
            trans_acc = np.zeros((k, k))
            gamma_acc = np.zeros(k)
            mean_acc = np.zeros((k, d))
            var_acc = np.zeros((k, d))
            total_ll = 0.0

            for X in sequences:
                log_b = self._log_emission(X)
                log_alpha, ll = self._forward(log_startprob, log_transmat, log_b)
                log_beta = self._backward(log_transmat, log_b)
                total_ll += ll

                log_gamma = log_alpha + log_beta - ll
                gamma = np.exp(log_gamma)  # (T, k)

                start_acc += gamma[0]
                gamma_acc += gamma.sum(axis=0)
                mean_acc += gamma.T @ X
                var_acc += gamma.T @ (X * X)

                if X.shape[0] > 1:
                    # log_xi[t,i,j] summed over t via log-sum-exp then exp.
                    log_xi = (
                        log_alpha[:-1, :, None]
                        + log_transmat[None, :, :]
                        + (log_b[1:] + log_beta[1:])[:, None, :]
                        - ll
                    )
                    trans_acc += np.exp(logsumexp(log_xi, axis=0))

            # M-step.
            self.startprob_ = start_acc / max(start_acc.sum(), 1e-300)
            row_sums = trans_acc.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self.transmat_ = trans_acc / row_sums
            safe_gamma = np.maximum(gamma_acc, 1e-300)[:, None]
            self.means_ = mean_acc / safe_gamma
            self.variances_ = np.maximum(
                var_acc / safe_gamma - self.means_**2, self.var_floor
            )

            if prev_ll > -np.inf:
                denom = abs(prev_ll) if prev_ll != 0 else 1.0
                if (total_ll - prev_ll) / denom < self.tol:
                    prev_ll = total_ll
                    break
            prev_ll = total_ll

        self.loglik_ = prev_ll
        return self

    def score(self, X):
        """Return the forward log-likelihood (natural log) of ``X``.

        Parameters
        ----------
        X : numpy.ndarray, shape (T, n_dims)
            Observation sequence.

        Returns
        -------
        float
            ``log p(X | model)``.
        """
        X = np.asarray(X, dtype=float)
        log_startprob = np.log(self.startprob_ + 1e-300)
        log_transmat = np.log(self.transmat_ + 1e-300)
        _, ll = self._forward(log_startprob, log_transmat, self._log_emission(X))
        return float(ll)

    def decode(self, X):
        """Viterbi-decode the most likely hidden-state path for ``X``.

        Parameters
        ----------
        X : numpy.ndarray, shape (T, n_dims)
            Observation sequence.

        Returns
        -------
        states : numpy.ndarray, shape (T,)
            Most likely state index at each time point.
        logprob : float
            Log-probability of the decoded path.
        """
        X = np.asarray(X, dtype=float)
        log_startprob = np.log(self.startprob_ + 1e-300)
        log_transmat = np.log(self.transmat_ + 1e-300)
        log_b = self._log_emission(X)
        t_len = log_b.shape[0]

        delta = np.empty((t_len, self.n_states))
        psi = np.zeros((t_len, self.n_states), dtype=int)
        delta[0] = log_startprob + log_b[0]
        for t in range(1, t_len):
            scores = delta[t - 1][:, None] + log_transmat
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(self.n_states)] + log_b[t]

        states = np.empty(t_len, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(t_len - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states, float(np.max(delta[-1]))

    def model_cost_bits(self):
        """Description length (bits) of encoding this HMM's parameters.

        Encodes the number of states with ``log*`` and every continuous
        parameter (``pi``, ``A``, the means and the diagonal variances) at the
        fixed float cost.

        Returns
        -------
        float
            Model description length in bits.
        """
        k, d = self.n_states, self.n_dims
        n_floats = k + k * k + 2 * k * d  # pi, A, means, variances
        return _log_star(k) + _COST_FLOAT * n_floats


def _regime_data_cost_bits(hmm, sequences):
    """Negative log-likelihood (in bits) of ``sequences`` under one HMM.

    Parameters
    ----------
    hmm : _GaussianHMM
        Fitted regime model.
    sequences : list of numpy.ndarray
        Observation sequences assigned to the regime.

    Returns
    -------
    float
        ``CostC`` contribution for these sequences.
    """
    total = 0.0
    for s in sequences:
        if len(s) > 0:
            total += -hmm.score(s)
    return total / _LOG2


def _segment_cost_bits(seg_lengths, n_regimes):
    """Description length (bits) of a segmentation's structure.

    Encodes the number of segments, each segment length and the regime
    identity of each segment.

    Parameters
    ----------
    seg_lengths : list of int
        Length of every segment.
    n_regimes : int
        Number of regimes the segments may be assigned to.

    Returns
    -------
    float
        Segmentation description length in bits.
    """
    m = len(seg_lengths)
    if m == 0:
        return 0.0
    cost = _log_star(m)
    cost += sum(_log_star(length) for length in seg_lengths)
    cost += m * np.log2(max(n_regimes, 1))
    return float(cost)


def _cut_point_search(X, hmms, switch_cost):
    """Jointly segment and assign ``X`` to a set of regime HMMs.

    A nested Viterbi decode over the augmented state space ``(regime, hidden
    state)``: within-regime transitions use each HMM's own transition matrix,
    and switching regimes incurs ``switch_cost`` (the per-segment encoding
    penalty) plus re-entry through the new regime's initial distribution. The
    regime component of the optimal path is the segmentation.

    Parameters
    ----------
    X : numpy.ndarray, shape (T, n_dims)
        Observation sequence.
    hmms : list of _GaussianHMM
        Candidate regime models.
    switch_cost : float
        Penalty (in nats) charged whenever the regime changes between
        consecutive time points.

    Returns
    -------
    numpy.ndarray, shape (T,)
        Regime index assigned to each time point.
    """
    t_len = X.shape[0]
    r = len(hmms)
    # Flatten (regime, state) into a single index space.
    offsets = np.cumsum([0] + [h.n_states for h in hmms])
    n_total = offsets[-1]
    regime_of = np.empty(n_total, dtype=int)
    for g in range(r):
        regime_of[offsets[g] : offsets[g + 1]] = g

    log_b = np.empty((t_len, n_total))
    log_start = np.empty(n_total)
    for g, h in enumerate(hmms):
        sl = slice(offsets[g], offsets[g + 1])
        log_b[:, sl] = h._log_emission(X)
        log_start[sl] = np.log(h.startprob_ + 1e-300)

    # Full augmented transition matrix in log-space.
    log_trans = np.full((n_total, n_total), -np.inf)
    for g, h in enumerate(hmms):
        sl = slice(offsets[g], offsets[g + 1])
        log_trans[sl, sl] = np.log(h.transmat_ + 1e-300)
    # Cross-regime switches: pay switch_cost and re-enter via target startprob.
    for g in range(r):
        for g2 in range(r):
            if g == g2:
                continue
            src = slice(offsets[g], offsets[g + 1])
            tgt = slice(offsets[g2], offsets[g2 + 1])
            log_trans[src, tgt] = (
                -switch_cost + np.log(hmms[g2].startprob_ + 1e-300)[None, :]
            )

    delta = np.empty((t_len, n_total))
    psi = np.zeros((t_len, n_total), dtype=int)
    delta[0] = log_start + log_b[0]
    for t in range(1, t_len):
        scores = delta[t - 1][:, None] + log_trans
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(n_total)] + log_b[t]

    path = np.empty(t_len, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(t_len - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return regime_of[path]


def _labels_to_segments(labels):
    """Split a per-point label array into contiguous ``(start, end, label)``.

    Parameters
    ----------
    labels : numpy.ndarray, shape (T,)
        Per-point regime label.

    Returns
    -------
    list of tuple
        ``(start, end, label)`` for each maximal run, with ``end`` exclusive.
    """
    segments = []
    start = 0
    for t in range(1, len(labels)):
        if labels[t] != labels[t - 1]:
            segments.append((start, t, int(labels[start])))
            start = t
    segments.append((start, len(labels), int(labels[start])))
    return segments


class AutoPlaitSegmenter(BaseSegmenter):
    """AutoPlait parameter-free multi-regime segmenter.

    AutoPlait [1]_ segments a (multivariate) time series into variable-length
    segments and assigns each segment to one of an automatically chosen number
    of *regimes*, each modelled by a multivariate Gaussian HMM. It is
    parameter-free: the Minimum Description Length (MDL) principle selects the
    number of regimes, the number of segments and the per-regime model
    complexity, so no similarity threshold has to be set. Because regimes are
    reused, the same regime label can recur non-contiguously.

    The algorithm starts with the whole series as a single regime and
    repeatedly attempts to split a regime into two: two seed HMMs are fitted, a
    nested-Viterbi cut-point search re-assigns time points, and the two HMMs are
    re-estimated until convergence. A split is accepted only if it lowers the
    total description length; otherwise the regime is finalised. This continues
    until no split reduces the cost.

    The output is a dense array of per-point regime labels (the tag
    ``returns_dense`` is ``False``), matching aeon segmenters such as
    :class:`GreedyGaussianSegmenter` that expose repeatable segment labels.

    Parameters
    ----------
    max_regimes : int, default=10
        Upper bound on the number of regimes discovered. Acts purely as a
        runtime safeguard; MDL usually selects fewer.
    n_states : int, default=3
        Number of hidden states per regime HMM.
    max_iter : int, default=20
        Maximum Baum-Welch iterations for each HMM fit.
    n_restarts : int, default=1
        Number of random seedings tried per regime split; the best (lowest
        cost) is kept.
    min_segment : int, default=10
        Minimum number of time points a discovered segment may contain. Splits
        producing shorter segments are rejected.
    random_state : int or None, default=None
        Seed for HMM initialisation, for reproducibility.

    Attributes
    ----------
    n_regimes_ : int
        Number of regimes discovered.
    labels_ : numpy.ndarray
        Per-point regime labels from the last call to ``predict``.
    change_points_ : list of int
        Sorted change-point locations (segment boundaries) discovered.

    References
    ----------
    .. [1] Matsubara, Y., Sakurai, Y., & Faloutsos, C. "AutoPlait: Automatic
       Mining of Co-evolving Time Sequences", Proceedings of the 2014 ACM
       SIGMOD International Conference on Management of Data, 2014.
       DOI 10.1145/2588555.2588556.

    Notes
    -----
    Clean-room implementation written from the SIGMOD 2014 paper only. No
    reference implementation was consulted; the Gaussian HMM (Baum-Welch,
    Viterbi and scoring) is implemented from scratch using ``numpy``/``scipy``.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.segmentation import AutoPlaitSegmenter
    >>> rng = np.random.default_rng(0)
    >>> a = rng.normal(-3, 0.3, size=(80, 1))
    >>> b = rng.normal(3, 0.3, size=(80, 1))
    >>> X = np.concatenate([a, b, a])
    >>> seg = AutoPlaitSegmenter(n_states=2, random_state=0)
    >>> labels = seg.fit_predict(X, axis=0)  # doctest: +SKIP
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:missing_values": False,
        "returns_dense": False,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        max_regimes: int = 10,
        n_states: int = 3,
        max_iter: int = 20,
        n_restarts: int = 1,
        min_segment: int = 10,
        random_state=None,
    ):
        self.max_regimes = max_regimes
        self.n_states = n_states
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.min_segment = min_segment
        self.random_state = random_state
        super().__init__(axis=0, n_segments=None)

    def _make_hmm(self, n_states, n_dims, rng):
        """Construct an unfitted regime HMM with the configured settings."""
        return _GaussianHMM(
            n_states=n_states,
            n_dims=n_dims,
            random_state=rng,
            max_iter=self.max_iter,
        )

    def _fit_regime(self, sequences, n_dims, rng):
        """Fit a single-regime HMM and return it with its total MDL cost.

        Parameters
        ----------
        sequences : list of numpy.ndarray
            Observation sequences forming the regime.
        n_dims : int
            Observation dimensionality.
        rng : numpy.random.Generator
            Random generator for initialisation.

        Returns
        -------
        hmm : _GaussianHMM
            The fitted model.
        cost : float
            ``CostM + CostC`` for modelling these sequences with one regime.
        """
        k = min(self.n_states, max(1, min(len(s) for s in sequences)))
        hmm = self._make_hmm(k, n_dims, rng).fit(sequences)
        seg_lengths = [len(s) for s in sequences]
        cost = (
            hmm.model_cost_bits()
            + _regime_data_cost_bits(hmm, sequences)
            + _segment_cost_bits(seg_lengths, 1)
        )
        return hmm, cost

    def _try_split(self, sequences, n_dims, rng):
        """Attempt to split a regime's sequences into two sub-regimes.

        Seeds two HMMs, alternates nested-Viterbi cut-point search and
        Baum-Welch re-estimation until the assignment stabilises, and returns
        the best two-regime model found across restarts.

        Parameters
        ----------
        sequences : list of numpy.ndarray
            Observation sequences currently assigned to the regime.
        n_dims : int
            Observation dimensionality.
        rng : numpy.random.Generator
            Random generator for the seedings.

        Returns
        -------
        result : dict or None
            ``None`` if no valid two-regime split was found, otherwise a dict
            with keys ``"hmms"``, ``"labels"`` (per-sequence label arrays) and
            ``"cost"`` (the two-regime MDL cost).
        """
        total_len = sum(len(s) for s in sequences)
        switch_cost = _LOG2 * (np.log2(2.0) + _log_star(total_len))
        best = None

        for _ in range(self.n_restarts):
            k = min(self.n_states, max(1, min(len(s) for s in sequences)))
            hmm0 = self._make_hmm(k, n_dims, rng)
            hmm1 = self._make_hmm(k, n_dims, rng)
            # Seed the two HMMs on the lower/upper half of the pooled data by
            # projecting onto the first principal direction (mean split here).
            pooled = np.concatenate(sequences, axis=0)
            proj = pooled @ (pooled.mean(axis=0) - pooled[0] + 1e-9)
            median = np.median(proj)
            low = pooled[proj <= median]
            high = pooled[proj > median]
            if len(low) == 0 or len(high) == 0:
                low, high = pooled[: len(pooled) // 2], pooled[len(pooled) // 2 :]
            hmm0.fit([low])
            hmm1.fit([high])

            prev_labels = None
            for _ in range(10):
                per_seq_labels = [
                    _cut_point_search(s, [hmm0, hmm1], switch_cost) for s in sequences
                ]
                flat = np.concatenate(per_seq_labels)
                if prev_labels is not None and np.array_equal(flat, prev_labels):
                    break
                prev_labels = flat

                seqs0, seqs1 = [], []
                for s, lab in zip(sequences, per_seq_labels):
                    for start, end, g in _labels_to_segments(lab):
                        (seqs0 if g == 0 else seqs1).append(s[start:end])
                if len(seqs0) == 0 or len(seqs1) == 0:
                    break
                k0 = min(self.n_states, max(1, min(len(s) for s in seqs0)))
                k1 = min(self.n_states, max(1, min(len(s) for s in seqs1)))
                hmm0 = self._make_hmm(k0, n_dims, rng).fit(seqs0)
                hmm1 = self._make_hmm(k1, n_dims, rng).fit(seqs1)

            per_seq_labels = [
                _cut_point_search(s, [hmm0, hmm1], switch_cost) for s in sequences
            ]
            seqs0, seqs1, all_lengths = [], [], []
            for s, lab in zip(sequences, per_seq_labels):
                for start, end, g in _labels_to_segments(lab):
                    all_lengths.append(end - start)
                    (seqs0 if g == 0 else seqs1).append(s[start:end])
            if len(seqs0) == 0 or len(seqs1) == 0:
                continue
            if min(all_lengths) < self.min_segment:
                continue

            cost = (
                hmm0.model_cost_bits()
                + hmm1.model_cost_bits()
                + _regime_data_cost_bits(hmm0, seqs0)
                + _regime_data_cost_bits(hmm1, seqs1)
                + _segment_cost_bits(all_lengths, 2)
            )
            if best is None or cost < best["cost"]:
                best = {"hmms": [hmm0, hmm1], "labels": per_seq_labels, "cost": cost}

        return best

    def _predict(self, X):
        """Segment ``X`` into MDL-selected regimes.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_timepoints, n_channels)
            Time series to segment.

        Returns
        -------
        numpy.ndarray, shape (n_timepoints,)
            Regime label of every time point. Labels may repeat
            non-contiguously across segments assigned to the same regime.
        """
        X = np.asarray(X, dtype=float)
        n_timepoints, n_dims = X.shape
        rng = np.random.default_rng(self.random_state)

        labels = np.zeros(n_timepoints, dtype=np.int32)
        if n_timepoints < 2 * self.min_segment:
            self.n_regimes_ = 1
            self.labels_ = labels
            self.change_points_ = []
            return labels

        # Each stack entry is the list of global index arrays forming a regime.
        root_indices = [np.arange(n_timepoints)]
        stack = [root_indices]
        next_label = 0

        while stack:
            if next_label >= self.max_regimes - 1:
                # Finalise every remaining candidate as its own regime.
                for indices in stack:
                    for idx in indices:
                        labels[idx] = next_label
                    next_label += 1
                break

            indices = stack.pop()
            sequences = [X[idx] for idx in indices]
            if min(len(s) for s in sequences) < 1:
                continue

            _, base_cost = self._fit_regime(sequences, n_dims, rng)
            split = None
            if sum(len(s) for s in sequences) >= 2 * self.min_segment:
                split = self._try_split(sequences, n_dims, rng)

            if split is not None and split["cost"] < base_cost:
                child0, child1 = [], []
                for idx, lab in zip(indices, split["labels"]):
                    for start, end, g in _labels_to_segments(lab):
                        (child0 if g == 0 else child1).append(idx[start:end])
                stack.append(child0)
                stack.append(child1)
            else:
                for idx in indices:
                    labels[idx] = next_label
                next_label += 1

        self.n_regimes_ = int(labels.max()) + 1 if n_timepoints > 0 else 0
        self.labels_ = labels
        self.change_points_ = [
            end for (_, end, _) in _labels_to_segments(labels)[:-1]
        ]
        return labels

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return small, fast parameter settings for estimator tests.

        Parameters
        ----------
        parameter_set : str, default="default"
            Unused; present for API compatibility.

        Returns
        -------
        dict
            Parameters constructing a fast test instance.
        """
        return {
            "max_regimes": 3,
            "n_states": 2,
            "max_iter": 5,
            "min_segment": 3,
            "random_state": 0,
        }
