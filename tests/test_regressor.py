"""Behavioral tests for GBCRegressor."""

import numpy as np
import pytest
import torch

from gbc import GBCRegressor, NotFittedError

EPOCHS = 120
SMALL = dict(epochs=EPOCHS, hdim=32, nh=8, n_samples=41, device="cpu")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(150, 1))
    y = (X[:, 0] ** 2 + 0.05 * rng.standard_normal(150))
    Xte = np.linspace(-1, 1, 40).reshape(-1, 1)
    yte = Xte[:, 0] ** 2
    return X, y, Xte, yte


@pytest.fixture(scope="module")
def fitted(data):
    """Cheap fit — for shape/plumbing checks."""
    X, y, _, _ = data
    return GBCRegressor(seed=0, **SMALL).fit(X, y)


@pytest.fixture(scope="module")
def trained(data):
    """Properly converged fit — for statistical behavior checks."""
    X, y, _, _ = data
    return GBCRegressor(seed=0, **{**SMALL, "epochs": 3000}).fit(X, y)


def test_not_fitted_raises():
    m = GBCRegressor(**SMALL)
    assert not m.is_fitted
    for call in (lambda: m.predict(np.zeros((2, 1))),
                 lambda: m.sample(np.zeros((2, 1))),
                 lambda: m.predict_quantiles(np.zeros((2, 1))),
                 lambda: m.save("/tmp/x.pt")):
        with pytest.raises(NotFittedError):
            call()


def test_fit_returns_self_and_sets_attrs(fitted):
    assert fitted.is_fitted
    assert fitted.n_features_in_ == 1
    assert len(fitted.history_) == EPOCHS
    assert np.isfinite(fitted.history_).all()


def test_shapes(fitted, data):
    _, _, Xte, _ = data
    n = len(Xte)
    assert fitted.predict(Xte).shape == (n,)
    assert fitted.sample(Xte).shape == (41, n)
    assert fitted.sample(Xte, n_samples=7).shape == (7, n)
    assert fitted.predict_quantiles(Xte, (0.1, 0.5, 0.9)).shape == (n, 3)
    lo, hi = fitted.predict_interval(Xte, alpha=0.8)
    assert lo.shape == hi.shape == (n,)
    mu, sd = fitted.predict(Xte, return_std=True)
    assert mu.shape == sd.shape == (n,)
    assert (sd > 0).all()


def test_1d_input_accepted(data):
    X, y, _, _ = data
    m = GBCRegressor(seed=0, **SMALL).fit(X.ravel(), y)
    assert m.n_features_in_ == 1
    assert m.predict(np.linspace(-1, 1, 5)).shape == (5,)


def test_quantiles_are_ordered_when_trained(trained, data):
    """The monotonicity term should give a properly ordered quantile function."""
    _, _, Xte, _ = data
    q = trained.predict_quantiles(Xte, (0.05, 0.25, 0.5, 0.75, 0.95))
    ordered = np.mean(np.all(np.diff(q, axis=1) >= -1e-6, axis=1))
    assert ordered > 0.9, f"only {ordered:.2f} of test points had ordered quantiles"
    lo, hi = trained.predict_interval(Xte, alpha=0.9)
    assert np.all(lo <= hi)


def test_predict_learns_the_signal(trained, data):
    _, y, Xte, yte = data
    rmse = np.sqrt(np.mean((trained.predict(Xte) - yte) ** 2))
    baseline = np.sqrt(np.mean((y.mean() - yte) ** 2))
    assert rmse < 0.2 * baseline, f"rmse={rmse:.3f} baseline={baseline:.3f}"


def test_predictive_spread_covers_the_noise(trained, data):
    """90% interval should have roughly nominal coverage on held-out points."""
    _, _, Xte, yte = data
    rng = np.random.default_rng(7)
    y_noisy = yte + 0.05 * rng.standard_normal(len(yte))
    lo, hi = trained.predict_interval(Xte, alpha=0.90)
    cov = np.mean((y_noisy >= lo) & (y_noisy <= hi))
    assert 0.6 <= cov <= 1.0, f"coverage={cov:.2f}"


def test_score_is_negative_rmse(fitted, data):
    _, _, Xte, yte = data
    expected = -np.sqrt(np.mean((fitted.predict(Xte) - yte) ** 2))
    assert fitted.score(Xte, yte) == pytest.approx(expected)


def test_mean_head_method(fitted, data):
    _, _, Xte, _ = data
    mh = fitted.predict(Xte, method="mean_head")
    assert mh.shape == (len(Xte),)
    assert np.isfinite(mh).all()
    with pytest.raises(ValueError):
        fitted.predict(Xte, return_std=True, method="mean_head")
    with pytest.raises(ValueError):
        fitted.predict(Xte, method="nope")


def test_determinism(data):
    X, y, Xte, _ = data
    a = GBCRegressor(seed=5, **SMALL).fit(X, y).predict(Xte)
    b = GBCRegressor(seed=5, **SMALL).fit(X, y).predict(Xte)
    np.testing.assert_array_equal(a, b)
    c = GBCRegressor(seed=6, **SMALL).fit(X, y).predict(Xte)
    assert not np.array_equal(a, c)


def test_sample_rng_is_reproducible(fitted, data):
    _, _, Xte, _ = data
    a = fitted.sample(Xte, rng=0)
    np.testing.assert_array_equal(a, fitted.sample(Xte, rng=0))
    np.testing.assert_array_equal(
        a, fitted.sample(Xte, rng=np.random.default_rng(0)))
    assert not np.array_equal(a, fitted.sample(Xte, rng=1))


def test_default_sampling_stream_is_seeded_and_advances(data):
    """Two fits with the same seed sample identically; calls still differ."""
    X, y, Xte, _ = data
    m1 = GBCRegressor(seed=3, **SMALL).fit(X, y)
    m2 = GBCRegressor(seed=3, **SMALL).fit(X, y)
    first = m1.sample(Xte)
    np.testing.assert_array_equal(first, m2.sample(Xte))
    assert not np.array_equal(first, m1.sample(Xte))


def test_sample_ignores_the_global_numpy_rng(fitted, data):
    """Regression test: sample() used to consume np.random's global state."""
    _, _, Xte, _ = data
    np.random.seed(0)
    fitted.sample(Xte, rng=0)
    after = np.random.uniform()
    np.random.seed(0)
    assert after == np.random.uniform()


def test_save_load_roundtrip(fitted, data, tmp_path):
    _, _, Xte, _ = data
    p = tmp_path / "model.pt"
    fitted.save(p)
    loaded = GBCRegressor.load(p, device="cpu")
    np.testing.assert_array_equal(fitted.predict(Xte), loaded.predict(Xte))
    assert loaded.n_features_in_ == fitted.n_features_in_
    assert loaded.hdim == fitted.hdim and loaded.nh == fitted.nh


def test_get_set_params_roundtrip():
    m = GBCRegressor(epochs=10, loss_w=(0.1, 0.2, 0.7))
    p = m.get_params()
    assert p["epochs"] == 10 and p["loss_w"] == (0.1, 0.2, 0.7)
    m.set_params(epochs=20)
    assert m.epochs == 20
    with pytest.raises(ValueError):
        m.set_params(nonsense=1)
    assert "epochs=20" in repr(m)


def test_input_validation(fitted, data):
    X, y, _, _ = data
    with pytest.raises(ValueError):
        GBCRegressor(**SMALL).fit(X, y[:10])
    with pytest.raises(ValueError):
        fitted.predict(np.zeros((5, 3)))          # wrong n_features
    with pytest.raises(ValueError):
        fitted.predict_quantiles(np.zeros((5, 1)), quantiles=(0.0, 0.5))
    with pytest.raises(ValueError):
        fitted.predict_interval(np.zeros((5, 1)), alpha=1.5)


# ── opt-in training extras ───────────────────────────────────────────────────

def test_minibatch_runs_and_differs(data):
    X, y, Xte, _ = data
    base = GBCRegressor(seed=0, **SMALL).fit(X, y).predict(Xte)
    mb = GBCRegressor(seed=0, batch_size=32, **SMALL).fit(X, y).predict(Xte)
    assert np.isfinite(mb).all()
    assert not np.array_equal(base, mb)


def test_taus_per_step(data):
    X, y, Xte, _ = data
    m = GBCRegressor(seed=0, taus_per_step=8, **SMALL).fit(X, y)
    assert np.isfinite(m.predict(Xte)).all()


def test_early_stopping_stops_early(data):
    X, y, _, _ = data
    m = GBCRegressor(seed=0, validation_fraction=0.2, patience=5,
                     **{**SMALL, "epochs": 2000}).fit(X, y)
    assert len(m.history_) < 2000
    assert m.is_fitted


def test_explicit_validation_data(data):
    X, y, Xte, yte = data
    m = GBCRegressor(seed=0, patience=10, **SMALL)
    m.fit(X, y, validation_data=(Xte, yte))
    assert m.is_fitted


def test_sklearn_interop_if_available(data):
    """Optional: clone / GridSearchCV work when scikit-learn is installed."""
    sk = pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV

    X, y, _, _ = data
    m = GBCRegressor(seed=0, **{**SMALL, "epochs": 60})
    assert clone(m).get_params() == m.get_params()
    gs = GridSearchCV(m, {"loss_w": [(0.3, 0.3, 0.4), (0.1, 0.2, 0.7)]},
                      cv=2).fit(X, y)
    assert gs.best_params_["loss_w"] in [(0.3, 0.3, 0.4), (0.1, 0.2, 0.7)]


def test_bad_extras_raise(data):
    X, y, _, _ = data
    with pytest.raises(ValueError):
        GBCRegressor(taus_per_step=0, **SMALL).fit(X, y)
    with pytest.raises(ValueError):
        GBCRegressor(validation_fraction=1.5, patience=3, **SMALL).fit(X, y)


# ── device reporting ─────────────────────────────────────────────────────────

def test_device_property_before_and_after_fit(data):
    import torch
    X, y, _, _ = data
    m = GBCRegressor(seed=0, **SMALL)
    assert m.device_ == torch.device("cpu")      # what fit() would select
    m.fit(X, y)
    assert m.device_ == next(m.model_.parameters()).device


def test_device_property_reports_explicit_device(data):
    import torch
    X, y, _, _ = data
    m = GBCRegressor(seed=0, **{**SMALL, "device": "cpu"}).fit(X, y)
    assert m.device_ == torch.device("cpu")


def test_diagnose_reports_accelerators():
    import gbc
    a = gbc.diagnose(verbose=False)["accelerators"]
    assert set(a) >= {"cuda", "cuda_devices", "mps_built", "mps_available",
                      "resolved", "torch_threads"}
    assert a["resolved"] in ("cpu", "cuda", "mps")
    assert isinstance(a["cuda"], bool)
    assert isinstance(a["cuda_devices"], list)
    # a CUDA-capable report must name its devices
    assert bool(a["cuda_devices"]) == a["cuda"]
