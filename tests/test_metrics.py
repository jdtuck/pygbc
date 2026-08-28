"""Tests for gbc.metrics, including parity with the upstream definitions."""

import numpy as np
import pytest
from scipy.stats import norm

from gbc import metrics


@pytest.fixture
def draws():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200)
    samples = rng.standard_normal((300, 200))
    return y, samples


# ── Parity with the upstream formulas ────────────────────────────────────────

def test_crps_gaussian_matches_reference():
    rng = np.random.default_rng(3)
    y, mu = rng.standard_normal(50), rng.standard_normal(50)
    sigma = rng.uniform(0.1, 2.0, 50)

    z = (y - mu) / np.maximum(sigma, 1e-12)
    ref = float(np.mean(
        sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z)
                 - 1.0 / np.sqrt(np.pi))))
    assert metrics.crps_gaussian(y, mu, sigma) == pytest.approx(ref)


def test_crps_gaussian_closed_form_sanity():
    """CRPS of N(0,1) against y=0 is 1/sqrt(pi) - 1/sqrt(2*pi)... ~0.2337."""
    val = metrics.crps_gaussian(np.zeros(1), np.zeros(1), np.ones(1))
    assert val == pytest.approx(2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi),
                                rel=1e-9)


def test_crps_samples_approaches_gaussian_closed_form():
    """Sample CRPS on N(0,1) draws should match the closed form."""
    rng = np.random.default_rng(1)
    y = np.zeros(400)
    samples = rng.standard_normal((4000, 400))
    approx = metrics.crps_samples(y, samples, rng=0)
    exact = metrics.crps_gaussian(y, np.zeros(400), np.ones(400))
    assert approx == pytest.approx(exact, abs=0.01)


def test_crps_is_zero_for_a_perfect_point_forecast():
    y = np.array([1.0, 2.0, 3.0])
    samples = np.tile(y, (50, 1))
    assert metrics.crps_samples(y, samples, rng=0) == pytest.approx(0.0)


def test_coverage_matches_reference(draws):
    y, samples = draws
    lo = np.quantile(samples, 0.05, axis=0)
    hi = np.quantile(samples, 0.95, axis=0)
    ref = float(np.mean((y >= lo) & (y <= hi)))
    assert metrics.coverage(y, samples, alpha=0.90) == ref


def test_coverage_is_calibrated_when_the_model_is_right():
    rng = np.random.default_rng(4)
    y = rng.standard_normal(3000)
    samples = rng.standard_normal((500, 3000))
    assert metrics.coverage(y, samples, 0.90) == pytest.approx(0.90, abs=0.02)


def test_rmse_and_ms():
    assert metrics.rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert metrics.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))

    mean, se = metrics.ms([1.0, 2.0, 3.0, np.nan])
    assert mean == pytest.approx(2.0)
    assert se == pytest.approx(np.std([1.0, 2.0, 3.0]) / np.sqrt(3))


# ── Additions ────────────────────────────────────────────────────────────────

def test_crps_samples_rng_is_reproducible(draws):
    y, samples = draws
    assert metrics.crps_samples(y, samples, rng=7) == \
        metrics.crps_samples(y, samples, rng=7)
    assert metrics.crps_samples(y, samples, rng=np.random.default_rng(7)) == \
        metrics.crps_samples(y, samples, rng=7)


def test_interval_score_penalizes_width_and_misses():
    y = np.array([0.0])
    tight = np.linspace(-1, 1, 101).reshape(-1, 1)
    wide = np.linspace(-10, 10, 101).reshape(-1, 1)
    assert metrics.interval_score(y, tight) < metrics.interval_score(y, wide)

    missed = np.linspace(5, 6, 101).reshape(-1, 1)
    assert metrics.interval_score(y, missed) > metrics.interval_score(y, tight)


def test_summarize_keys_and_values(draws):
    y, samples = draws
    out = metrics.summarize(y, samples, alpha=0.9, rng=0)
    assert set(out) == {"rmse", "crps", "coverage", "interval_score"}
    assert out["rmse"] == pytest.approx(metrics.rmse(y, samples.mean(0)))
    assert out["coverage"] == metrics.coverage(y, samples, 0.9)


def test_shape_validation(draws):
    y, samples = draws
    with pytest.raises(ValueError):
        metrics.crps_samples(y, samples[:, :10])
    with pytest.raises(ValueError):
        metrics.crps_samples(y, samples[0])


def test_metrics_work_on_a_real_fit():
    from gbc import GBCRegressor

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(200, 1))
    y = X[:, 0] ** 2 + 0.1 * rng.standard_normal(200)
    m = GBCRegressor(epochs=1500, hdim=32, nh=8, n_samples=101,
                     seed=0, device="cpu").fit(X, y)
    s = m.sample(X)
    out = metrics.summarize(y, s, rng=0)
    assert 0.0 < out["rmse"] < 0.3
    assert out["crps"] > 0
    assert 0.5 < out["coverage"] <= 1.0
