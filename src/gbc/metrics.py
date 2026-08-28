"""Evaluation metrics for probabilistic surrogates: CRPS, coverage, RMSE.

Faithful port of the reference implementation released with Polson & Sokolov
(2026). Default behavior is unchanged; optional arguments are opt-in.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["crps_gaussian", "crps_samples", "coverage", "rmse", "ms",
           "interval_score", "summarize"]


def crps_gaussian(y, mu, sigma) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution.

    Args:
        y: ``(n,)`` observed values.
        mu: ``(n,)`` predictive means.
        sigma: ``(n,)`` predictive standard deviations.

    Returns:
        Mean CRPS over the ``n`` points (lower is better).
    """
    from scipy.stats import norm

    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = (y - mu) / np.maximum(sigma, 1e-12)
    return float(np.mean(
        sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z)
                 - 1.0 / np.sqrt(np.pi))))


def crps_samples(y, samples, rng=None) -> float:
    """Sample-based CRPS via the energy-score estimator.

    ``CRPS(F, y) = E|Y - y| - 0.5 * E|Y - Y'|``, with the second term
    estimated from a single random pairing of the sample set.

    Args:
        y: ``(n,)`` observed values.
        samples: ``(B, n)`` predictive draws — e.g. the output of
            :meth:`~gbc.GBCRegressor.sample`.
        rng: optional :class:`numpy.random.Generator` or integer seed for the
            pairing permutation. ``None`` (default) uses the global NumPy
            random state, matching the reference implementation.

    Returns:
        Mean CRPS over the ``n`` points (lower is better).
    """
    y = np.asarray(y, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2-D (B, n); got {samples.shape}")
    if samples.shape[1] != len(y):
        raise ValueError(
            f"samples has {samples.shape[1]} columns but y has {len(y)}")

    term1 = np.mean(np.abs(samples - y[np.newaxis, :]), axis=0)
    if rng is None:
        idx = np.random.permutation(samples.shape[0])
    else:
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        idx = rng.permutation(samples.shape[0])
    term2 = 0.5 * np.mean(np.abs(samples - samples[idx, :]), axis=0)
    return float(np.mean(term1 - term2))


def coverage(y, samples, alpha: float = 0.90) -> float:
    """Empirical coverage of the central ``alpha`` prediction interval.

    Args:
        y: ``(n,)`` observed values.
        samples: ``(B, n)`` predictive draws.
        alpha: nominal coverage, e.g. ``0.90``.

    Returns:
        Fraction of points falling inside the interval. Compare to ``alpha``.
    """
    y = np.asarray(y, dtype=float)
    samples = np.asarray(samples, dtype=float)
    lo = np.quantile(samples, (1 - alpha) / 2, axis=0)
    hi = np.quantile(samples, 1 - (1 - alpha) / 2, axis=0)
    return float(np.mean((y >= lo) & (y <= hi)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ms(arr) -> Tuple[float, float]:
    """Mean and standard error of an array, ignoring NaNs.

    Returns:
        ``(mean, standard_error)`` — the format used in the paper's tables.
    """
    a = np.array(arr, dtype=float)
    return (float(np.nanmean(a)),
            float(np.nanstd(a) / np.sqrt(np.sum(~np.isnan(a)))))


def interval_score(y, samples, alpha: float = 0.90) -> float:
    """Winkler interval score for the central ``alpha`` interval.

    Penalizes width plus ``2/(1-alpha)`` times any miss. Lower is better.
    Not part of the paper's tables; provided as a sharpness-aware companion
    to :func:`coverage`, which on its own says nothing about interval width.
    """
    y = np.asarray(y, dtype=float)
    samples = np.asarray(samples, dtype=float)
    a = 1 - alpha
    lo = np.quantile(samples, a / 2, axis=0)
    hi = np.quantile(samples, 1 - a / 2, axis=0)
    score = (hi - lo)
    score += (2 / a) * np.where(y < lo, lo - y, 0.0)
    score += (2 / a) * np.where(y > hi, y - hi, 0.0)
    return float(np.mean(score))


def summarize(y, samples, alpha: float = 0.90, rng=None) -> dict:
    """Convenience: the standard metric set for one set of predictive draws.

    Args:
        y: ``(n,)`` observed values.
        samples: ``(B, n)`` predictive draws.
        alpha: nominal coverage for the interval metrics.
        rng: passed to :func:`crps_samples`.

    Returns:
        ``{"rmse": ..., "crps": ..., "coverage": ..., "interval_score": ...}``
        where RMSE is computed against the sample mean.
    """
    samples = np.asarray(samples, dtype=float)
    return {
        "rmse": rmse(y, samples.mean(0)),
        "crps": crps_samples(y, samples, rng=rng),
        "coverage": coverage(y, samples, alpha=alpha),
        "interval_score": interval_score(y, samples, alpha=alpha),
    }
