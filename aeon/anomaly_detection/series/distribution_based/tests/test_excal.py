"""Tests for the ExCal class."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from aeon.anomaly_detection.series.distribution_based import ExCal
from aeon.testing.data_generation import make_example_2d_numpy_series


def _make_periodic_series(n_timepoints: int, seed: int) -> np.ndarray:
    """Generate a noisy two-channel periodic series with a dominant period of 20."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_timepoints)
    base = np.column_stack(
        [np.sin(2 * np.pi * t / 20), 0.5 * np.cos(2 * np.pi * t / 20 + 1.0)]
    )
    return base + 0.1 * rng.normal(size=(n_timepoints, 2))


def test_excal_output():
    """Test ExCal output shape and dtype."""
    X = make_example_2d_numpy_series(n_timepoints=100, n_channels=2, axis=0)

    detector = ExCal(random_state=0)
    pred = detector.fit(X[:80], axis=0).predict(X[80:], axis=0)

    assert pred.shape == (20,)
    assert np.issubdtype(pred.dtype, np.floating)
    assert np.isfinite(pred).all()


def test_excal_detects_anomalies():
    """Test that ExCal separates injected anomalies from normal points."""
    X_train = _make_periodic_series(500, seed=42)
    X_test = _make_periodic_series(1000, seed=43)[500:].copy()
    y_test = np.zeros(500, dtype=bool)

    # rare anomalies: a spike and a variance burst
    X_test[150:155] += 3.5
    X_test[300:310] += np.random.default_rng(44).normal(0.0, 2.5, size=(10, 2))
    y_test[150:155] = True
    y_test[300:310] = True

    detector = ExCal(random_state=0)
    scores = detector.fit(X_train, axis=0).predict(X_test, axis=0)

    assert scores.shape == (500,)
    auc = roc_auc_score(y_test, scores)
    assert auc > 0.9
    # the injected anomaly points rank within the top 15 scores
    top_k = y_test[np.argsort(scores)[-15:]]
    assert top_k.mean() >= 0.6


def test_excal_univariate():
    """Test ExCal on a univariate series with a spike anomaly."""
    X = _make_periodic_series(300, seed=7)[:, 0]
    X_test = X[200:].copy()
    X_test[30:34] += 3.5

    detector = ExCal(random_state=0)
    scores = detector.fit(X[:200]).predict(X_test)

    assert scores.shape == (100,)
    assert 28 <= np.argmax(scores) <= 36


def test_excal_transductive():
    """Test the transductive (unsupervised) usage via fit_predict."""
    X = _make_periodic_series(200, seed=7)

    detector = ExCal(random_state=0)
    pred = detector.fit_predict(X, axis=0)

    assert pred.shape == (200,)
    assert np.issubdtype(pred.dtype, np.floating)
    assert np.isfinite(pred).all()
    # scoring the training series itself must reproduce fit_predict
    assert np.allclose(detector.predict(X, axis=0), pred)


def test_excal_period_estimation():
    """Test that the dominant period is estimated from the training series."""
    X = _make_periodic_series(100, seed=0)

    detector = ExCal(random_state=0)
    detector.fit(X, axis=0)

    assert detector.period_ == 20


def test_excal_incorrect_params():
    """Test ExCal with incorrect parameters."""
    X = _make_periodic_series(50, seed=0)
    with pytest.raises(ValueError, match="k must be >= 1"):
        ExCal(k=0).fit(X, axis=0)
    with pytest.raises(ValueError, match="period must be >= 1"):
        ExCal(period=0).fit(X, axis=0)
    with pytest.raises(ValueError, match="rho must be > 0"):
        ExCal(rho=0.0).fit(X, axis=0)
    with pytest.raises(ValueError, match="tau must be >= 0"):
        ExCal(tau=-1.0).fit(X, axis=0)
    with pytest.raises(ValueError, match="quantile must be in"):
        ExCal(quantile=0.5).fit(X, axis=0)
    with pytest.raises(ValueError, match="quantile must be in"):
        ExCal(quantile=1.0).fit(X, axis=0)
