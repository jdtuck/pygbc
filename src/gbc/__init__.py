"""GBC: Generative Bayesian Computation via Implicit Quantile Networks.

A scalable alternative to Gaussian-process surrogates. Reference:

    Polson & Sokolov (2026). "Generative Bayesian Computation as a Scalable
    Alternative to Gaussian Process Surrogates." *Technometrics*.

Distributed on PyPI as ``pygbc``; imported as ``gbc``.

Object-oriented entry points::

    from gbc import GBCRegressor, AugGBCRegressor

    model = GBCRegressor(epochs=3000, seed=0).fit(X_train, y_train)
    mu = model.predict(X_test)
    lo, hi = model.predict_interval(X_test, alpha=0.90)

    # jump / regime-switching surfaces
    aug = AugGBCRegressor(epochs=8000, loss_w=(0.10, 0.20, 0.70))
    aug.fit(X_train, y_train)

Evaluation lives in :mod:`gbc.metrics`::

    from gbc.metrics import crps_samples, coverage, rmse, summarize

The original functional API (``train_iqn`` / ``sample_iqn`` and the augiqn
helpers) is re-exported unchanged and produces identical numbers.
"""

from . import metrics
from .aug import AugGBCRegressor
from .augiqn import (ClassifierMLP, augment_features, cluster_y,
                     get_regime_prob, thread_limit, train_classifier)
from .diagnostics import diagnose
from .iqn import IQN, resolve_device, sample_iqn, train_iqn
from .metrics import (coverage, crps_gaussian, crps_samples, interval_score,
                      ms, rmse, summarize)
from .model import GBCRegressor, NotFittedError
from .gbc4mvBayes import gbc4mvBayes

__version__ = "0.2.2"

__all__ = [
    # classes
    "GBCRegressor",
    "AugGBCRegressor",
    "IQN",
    "ClassifierMLP",
    "gbc4mvBayes",
    "NotFittedError",
    # functional API
    "train_iqn",
    "sample_iqn",
    "cluster_y",
    "train_classifier",
    "get_regime_prob",
    "augment_features",
    "resolve_device",
    "thread_limit",
    "diagnose",
    # metrics
    "metrics",
    "crps_gaussian",
    "crps_samples",
    "coverage",
    "rmse",
    "ms",
    "interval_score",
    "summarize",
    "__version__",
]
