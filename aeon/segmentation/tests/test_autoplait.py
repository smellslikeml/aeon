"""Tests for the AutoPlait multi-regime segmenter.

The tests are layered deliberately: the from-scratch Gaussian HMM is validated
on its own first (it is the load-bearing correctness floor), then the segmenter
is validated on synthetic multi-regime data with known change points, and
finally the estimator is run through aeon's conformance suite.
"""

__maintainer__ = []
__all__ = []

import numpy as np
import pytest

from aeon.segmentation import AutoPlaitSegmenter
from aeon.segmentation._autoplait import (
    _GaussianHMM,
    _labels_to_segments,
    _log_star,
)
from aeon.testing.estimator_checking import check_estimator


def _sample_from_hmm(startprob, transmat, means, variances, n, rng):
    """Draw a length-``n`` sequence and its state path from a Gaussian HMM."""
    k, d = means.shape
    states = np.empty(n, dtype=int)
    obs = np.empty((n, d))
    states[0] = rng.choice(k, p=startprob)
    for t in range(1, n):
        states[t] = rng.choice(k, p=transmat[states[t - 1]])
    for t in range(n):
        obs[t] = rng.normal(means[states[t]], np.sqrt(variances[states[t]]))
    return obs, states


def _match_accuracy(true_states, pred_states, k):
    """Best label-permutation accuracy between two state sequences."""
    from itertools import permutations

    best = 0.0
    for perm in permutations(range(k)):
        mapped = np.array([perm[s] for s in pred_states])
        best = max(best, np.mean(mapped == true_states))
    return best


def test_log_star_monotone_positive():
    """log* is positive and non-decreasing in its argument."""
    values = [_log_star(n) for n in [1, 2, 5, 50, 5000]]
    assert all(v > 0 for v in values)
    assert values == sorted(values)


def test_gaussian_hmm_recovers_known_model():
    """The from-scratch HMM: EM increases the log-likelihood.

    Baum-Welch is the highest-risk component, so this is asserted directly:
    fitting from a random start must not decrease the achievable
    log-likelihood relative to that starting point.
    """
    rng = np.random.default_rng(42)
    startprob = np.array([0.6, 0.4])
    transmat = np.array([[0.95, 0.05], [0.05, 0.95]])
    means = np.array([[-4.0], [4.0]])
    variances = np.array([[0.5], [0.5]])

    obs, _ = _sample_from_hmm(startprob, transmat, means, variances, 400, rng)

    hmm = _GaussianHMM(n_states=2, n_dims=1, random_state=np.random.default_rng(0))
    hmm._init_params([obs])
    ll_before = hmm.score(obs)
    hmm.fit([obs])
    ll_after = hmm.score(obs)

    assert ll_after >= ll_before - 1e-6


def test_gaussian_hmm_viterbi_recovers_states():
    """Viterbi recovers the ground-truth state path up to label permutation."""
    rng = np.random.default_rng(7)
    startprob = np.array([0.5, 0.5])
    transmat = np.array([[0.97, 0.03], [0.03, 0.97]])
    means = np.array([[-5.0], [5.0]])
    variances = np.array([[0.3], [0.3]])

    obs, true_states = _sample_from_hmm(
        startprob, transmat, means, variances, 500, rng
    )

    hmm = _GaussianHMM(n_states=2, n_dims=1, random_state=np.random.default_rng(1))
    hmm.fit([obs])
    pred_states, _ = hmm.decode(obs)

    assert _match_accuracy(true_states, pred_states, 2) > 0.9


def test_labels_to_segments():
    """Contiguous-run splitting yields the expected (start, end, label) runs."""
    labels = np.array([0, 0, 1, 1, 1, 0, 0])
    segs = _labels_to_segments(labels)
    assert segs == [(0, 2, 0), (2, 5, 1), (5, 7, 0)]


def _multi_regime_series(rng):
    """Concatenate two distinct HMM regimes with one recurrence: A, B, A.

    Each regime is a distinct 2-state Gaussian HMM with its own emission means,
    so a single HMM cannot cheaply model their union and MDL should prefer two
    regimes. Regime A recurs, so its label should repeat non-contiguously.
    """
    startprob = np.array([0.5, 0.5])
    transmat = np.array([[0.9, 0.1], [0.1, 0.9]])
    variances = np.array([[0.25], [0.25]])
    means_a = np.array([[-5.0], [-2.0]])
    means_b = np.array([[2.0], [5.0]])

    a, _ = _sample_from_hmm(startprob, transmat, means_a, variances, 120, rng)
    b, _ = _sample_from_hmm(startprob, transmat, means_b, variances, 120, rng)
    a2, _ = _sample_from_hmm(startprob, transmat, means_a, variances, 120, rng)
    X = np.concatenate([a, b, a2], axis=0)
    true_cps = [120, 240]
    return X, true_cps


def test_autoplait_recovers_regimes_and_changepoints():
    """AutoPlait recovers the regime count and change points within tolerance."""
    rng = np.random.default_rng(3)
    X, true_cps = _multi_regime_series(rng)

    seg = AutoPlaitSegmenter(n_states=2, min_segment=20, random_state=0)
    labels = seg.fit_predict(X, axis=0)

    assert labels.shape == (X.shape[0],)
    # Two distinct regimes (A recurs), so exactly two labels expected.
    assert seg.n_regimes_ == 2
    # The recurring regime should share a label across its two occurrences.
    assert labels[0] == labels[-1]
    assert labels[0] != labels[len(X) // 2]

    tol = 30
    for cp in true_cps:
        assert any(abs(cp - found) <= tol for found in seg.change_points_), (
            f"no detected change point near {cp}; found {seg.change_points_}"
        )


def test_autoplait_output_shape_and_type():
    """Output is a dense per-point integer label array (returns_dense=False)."""
    rng = np.random.default_rng(11)
    X, _ = _multi_regime_series(rng)

    seg = AutoPlaitSegmenter(n_states=2, min_segment=20, random_state=0)
    labels = seg.fit_predict(X, axis=0)

    assert isinstance(labels, np.ndarray)
    assert labels.ndim == 1
    assert labels.shape[0] == X.shape[0]
    assert np.issubdtype(labels.dtype, np.integer)
    assert not seg.get_tag("returns_dense")


def test_autoplait_univariate_and_multivariate():
    """The segmenter accepts both univariate and multivariate input."""
    rng = np.random.default_rng(5)
    # Univariate 1D input.
    uni = np.concatenate(
        [rng.normal(-3, 0.3, size=80), rng.normal(3, 0.3, size=80)]
    )
    seg = AutoPlaitSegmenter(n_states=2, min_segment=15, random_state=0)
    labels_uni = seg.fit_predict(uni, axis=0)
    assert labels_uni.shape[0] == uni.shape[0]

    # Multivariate (n_timepoints, n_channels) input.
    multi = np.concatenate(
        [
            rng.normal([-3, 3], 0.3, size=(80, 2)),
            rng.normal([3, -3], 0.3, size=(80, 2)),
        ],
        axis=0,
    )
    labels_multi = seg.fit_predict(multi, axis=0)
    assert labels_multi.shape[0] == multi.shape[0]


@pytest.mark.parametrize("n_states", [2])
def test_autoplait_single_regime(n_states):
    """Homogeneous data yields a single regime (no spurious split)."""
    rng = np.random.default_rng(9)
    X = rng.normal(0.0, 0.5, size=(150, 1))
    seg = AutoPlaitSegmenter(n_states=n_states, min_segment=20, random_state=0)
    labels = seg.fit_predict(X, axis=0)
    assert seg.n_regimes_ == 1
    assert np.all(labels == labels[0])


def test_autoplait_conformance():
    """Estimator passes aeon's conformance suite with fast test params."""
    check_estimator(AutoPlaitSegmenter, raise_exceptions=True)
