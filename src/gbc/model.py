"""Object-oriented interface to the GBC / IQN surrogate.

    >>> from gbc import GBCRegressor
    >>> model = GBCRegressor(epochs=3000, seed=0).fit(X_train, y_train)
    >>> mu = model.predict(X_test)
    >>> lo, hi = model.predict_interval(X_test, alpha=0.90)
    >>> draws = model.sample(X_test, n_samples=500)     # random draws, (500, n_test)
    >>> grid_predictions = model.sample(X_test, n_samples=500, method="grid")
"""

from __future__ import annotations

import inspect
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .iqn import (IQN, auto_chunk, resolve_device, sample_iqn,
                  train_iqn)

__all__ = ["GBCRegressor", "NotFittedError"]


class NotFittedError(RuntimeError):
    """Raised when a predict-time method is called before :meth:`fit`."""


# Inheriting scikit-learn's bases makes the estimator usable with clone(),
# GridSearchCV, cross_val_score and Pipeline (they need estimator tags that
# only BaseEstimator supplies). We still define get_params/set_params/score
# explicitly so the class is readable on its own terms.
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import RegressorMixin as _RegressorMixin


class GBCRegressor(_RegressorMixin, _BaseEstimator):
    """Generative Bayesian Computation surrogate (scikit-learn style).

    Fits an Implicit Quantile Network to ``(X, y)`` and exposes the full
    conditional predictive distribution of ``y | x`` — not just a point
    estimate. Inputs and the response are standardized internally; everything
    returned is on the original scale.

    Args:
        epochs: optimizer steps (full-batch) or epochs (with ``batch_size``).
        hdim: hidden width of the network.
        nh: number of cosine basis functions in the quantile embedding.
        lr: Adam learning rate, cosine-annealed to ``lr * 0.01``.
        weight_decay: Adam weight decay.
        loss_w: ``(mean_anchor, monotonicity, quantile)`` loss weights.
            The paper uses ``(0.3, 0.3, 0.4)`` by default and the
            quantile-dominant ``(0.1, 0.2, 0.7)`` for jump/discontinuous data.
        seed: torch manual seed set before initialization.
        device: ``"cpu"``, ``"cuda"``, a :class:`torch.device`, or ``None``
            to auto-select CUDA when available.
        n_samples: default number of quantile levels used by
            :meth:`sample`, :meth:`predict` and :meth:`predict_interval`.
        chunk: rows per forward pass at predict time. ``1000`` (default)
            matches the reference; ``"auto"`` sizes it for throughput
            (faster on large prediction sets, ~1e-7 relative change).
        batch_size: opt-in minibatch training. ``None`` (default) reproduces
            the paper's full-batch loop.
        taus_per_step: opt-in averaging over several quantile levels per
            optimizer step. ``1`` (default) reproduces the paper.
        validation_fraction: opt-in. Fraction of the training data held out
            for early stopping. Requires ``patience``.
        patience: opt-in. Stop after this many epochs without improvement on
            the held-out pinball loss; best weights are restored.
        verbose: if > 0, print the training loss every ``verbose`` epochs.
        foreach: multi-tensor Adam; see :func:`~gbc.iqn.train_iqn`.
            Identical weights, ~10% faster at small ``n``.
        track_history: set ``False`` to skip reading the loss back each
            epoch (saves a GPU sync per step; ``history_`` stays empty).

    Attributes:
        model_: the fitted :class:`~gbc.iqn.IQN`.
        n_features_in_: number of input features seen during :meth:`fit`.
        x_mean_, x_std_, y_mean_, y_std_: normalization statistics.
    """

    def __init__(
        self,
        epochs: int = 3000,
        hdim: int = 256,
        nh: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_w: Tuple[float, float, float] = (0.3, 0.3, 0.4),
        seed: int = 42,
        device=None,
        n_samples: int = 500,
        chunk=1000,
        batch_size: Optional[int] = None,
        taus_per_step: int = 1,
        validation_fraction: Optional[float] = None,
        patience: Optional[int] = None,
        verbose: int = 0,
        foreach: Optional[bool] = None,
        track_history: bool = True,
    ):
        self.epochs = epochs
        self.hdim = hdim
        self.nh = nh
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_w = loss_w
        self.seed = seed
        self.device = device
        self.n_samples = n_samples
        self.chunk = chunk
        self.batch_size = batch_size
        self.taus_per_step = taus_per_step
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.verbose = verbose
        self.foreach = foreach
        self.track_history = track_history

        self.model_: Optional[IQN] = None
        self.n_features_in_: Optional[int] = None
        self.x_mean_ = self.x_std_ = None
        self.y_mean_ = self.y_std_ = None
        self.history_: list = []
        # Lazily seeded from `seed` on first random sample(); not a
        # constructor parameter, so get_params()/clone() are unaffected.
        self._sample_rng: Optional[np.random.Generator] = None

    # ── scikit-learn style parameter plumbing ────────────────────────────

    @classmethod
    def _param_names(cls) -> list:
        sig = inspect.signature(cls.__init__)
        return [p for p in sig.parameters if p != "self"]

    def get_params(self, deep: bool = True) -> dict:
        """Return the constructor parameters (scikit-learn compatible)."""
        return {k: getattr(self, k) for k in self._param_names()}

    def set_params(self, **params) -> "GBCRegressor":
        """Set constructor parameters (scikit-learn compatible)."""
        valid = set(self._param_names())
        for k, v in params.items():
            if k not in valid:
                raise ValueError(
                    f"Invalid parameter {k!r} for {type(self).__name__}")
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        defaults = inspect.signature(type(self).__init__).parameters
        shown = ", ".join(
            f"{k}={getattr(self, k)!r}" for k in self._param_names()
            if getattr(self, k) != defaults[k].default)
        return f"{type(self).__name__}({shown})"

    # ── fitting ──────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self.model_ is not None

    @property
    def device_(self) -> torch.device:
        """The device the fitted network actually lives on.

        This is the ground truth for "did it use the GPU?" — ``device=None``
        auto-selects, and silently falls back to CPU when no supported
        accelerator is found. Before :meth:`fit`, reports what would be
        selected.
        """
        if self.model_ is None:
            return resolve_device(self.device)
        return next(self.model_.parameters()).device

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(
                f"This {type(self).__name__} is not fitted yet. "
                "Call fit() first.")

    def _check_X(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, d); got shape {X.shape}")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this "
                f"{type(self).__name__} was fitted with "
                f"{self.n_features_in_}")
        return X

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Map validated inputs to what the network consumes.

        Identity here; :class:`~gbc.AugGBCRegressor` overrides it to append
        the regime probability.
        """
        return X

    def _prepare(self, X) -> np.ndarray:
        return self._transform(self._check_X(X))

    def fit(self, X, y, validation_data=None) -> "GBCRegressor":
        """Fit the surrogate.

        Args:
            X: ``(n, d)`` array of inputs (a 1-D array is treated as ``d=1``).
            y: ``(n,)`` array of responses.
            validation_data: optional explicit ``(X_val, y_val)`` for early
                stopping. Overrides ``validation_fraction``.

        Returns:
            ``self``.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, d); got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch: {len(X)} vs {len(y)}")
        if len(X) < 2:
            raise ValueError("Need at least 2 training points")

        if validation_data is None and self.validation_fraction:
            rng = np.random.default_rng(self.seed)
            n_val = max(1, int(round(self.validation_fraction * len(X))))
            if n_val >= len(X):
                raise ValueError("validation_fraction leaves no training data")
            va = rng.choice(len(X), size=n_val, replace=False)
            tr = np.setdiff1d(np.arange(len(X)), va)
            validation_data = (X[va], y[va])
            X, y = X[tr], y[tr]

        self.n_features_in_ = X.shape[1]
        self.history_ = []
        self._sample_rng = None  # fit-then-sample is reproducible from `seed`

        # Subclass hook: fit any preprocessing and map X into network space.
        X_net = self._fit_transform(X, y)
        if validation_data is not None:
            validation_data = (
                self._transform(self._check_X(validation_data[0])),
                np.asarray(validation_data[1], dtype=float).ravel())

        def _cb(epoch: int, loss: float) -> None:
            self.history_.append(loss)
            if self.verbose and (epoch % self.verbose == 0
                                 or epoch == self.epochs - 1):
                print(f"epoch {epoch + 1}/{self.epochs}  loss={loss:.6f}")

        model, xm, xs, ym, ys = train_iqn(
            X_net, y,
            epochs=self.epochs,
            hdim=self.hdim,
            nh=self.nh,
            lr=self.lr,
            wd=self.weight_decay,
            seed=self.seed,
            loss_w=tuple(self.loss_w),
            device=self.device,
            batch_size=self.batch_size,
            taus_per_step=self.taus_per_step,
            validation_data=validation_data,
            patience=self.patience,
            callback=_cb,
            foreach=self.foreach,
            track_history=self.track_history,
        )
        self.model_ = model
        self.x_mean_, self.x_std_ = xm, xs
        self.y_mean_, self.y_std_ = ym, ys
        return self

    def _fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit any preprocessing and return the network's training inputs.

        Identity here; :class:`~gbc.AugGBCRegressor` overrides it.
        """
        return X

    # ── prediction ───────────────────────────────────────────────────────
    
    def _resolve_rng(self, rng) -> np.random.Generator:
        """Pick the generator for the random quantile levels.

        ``None`` uses the estimator's own stream, seeded from ``seed`` on
        first use. That stream *advances* across calls, so repeated
        ``sample()`` calls give fresh draws while the whole session stays
        reproducible from ``seed``. Pass an int or a
        :class:`numpy.random.Generator` to pin a single call.
        """
        if rng is None:
            if getattr(self, "_sample_rng", None) is None:
                self._sample_rng = np.random.default_rng(self.seed)
            return self._sample_rng
        if isinstance(rng, np.random.Generator):
            return rng
        return np.random.default_rng(rng)

    def sample(self, X, n_samples: Optional[int] = None,
               chunk: Optional[int] = None,
               method: str = "random", rng=None) -> np.ndarray:
        """Draw from the predictive distribution at each row of ``X``.

        Args:
            X: ``(n, d)`` inputs.
            n_samples: number of predictive draws / quantile evaluations
                (defaults to ``self.n_samples``).
            chunk: rows per forward pass (defaults to ``self.chunk``).
            method:
                - ``"random"`` (default): draw tau ~ Uniform(0, 1) and evaluate
                  the learned quantile function at those random taus.
                - ``"grid"``: evaluate the learned quantile function on an
                  equally spaced grid of quantile levels in ``[0.005, 0.995]``.
            rng: ``None`` (default) draws from the estimator's own stream,
                seeded from ``seed``; an int or
                :class:`numpy.random.Generator` pins this call. Ignored by
                ``method="grid"``, which is deterministic. The global
                ``numpy.random`` state is never used or disturbed.

        Returns:
            ``(n_samples, n)`` array on the original scale.
        """
        self._check_fitted()
        X = self._prepare(X)
        B = self.n_samples if n_samples is None else n_samples

        if method == "random":
            taus = self._resolve_rng(rng).uniform(low=0.0, high=1.0, size=B)
        elif method == "grid":
            # Leave the grid to sample_iqn, which builds it with
            # torch.linspace in float32 — the same construction the reference
            # implementation uses. Building it here with np.linspace in
            # float64 and letting torch round it to float32 moves 4 of 21
            # levels by one ulp and breaks bit-parity with the reference.
            taus = None
        else:
            raise ValueError("method must be 'random' or 'grid'")

        return sample_iqn(
            self.model_, X, self.x_mean_, self.x_std_,
            self.y_mean_, self.y_std_, B=B,
            chunk=self.chunk if chunk is None else chunk, taus=taus)

    def predict_quantiles(self, X, quantiles: Sequence[float] = (0.05, 0.5, 0.95),
                          chunk: Optional[int] = None) -> np.ndarray:
        """Predict specific quantiles of ``y | x``.

        Args:
            X: ``(n, d)`` inputs.
            quantiles: quantile levels in ``(0, 1)``.
            chunk: rows per forward pass.

        Returns:
            ``(n, len(quantiles))`` array on the original scale, columns in
            the order given.
        """
        self._check_fitted()
        X = self._prepare(X)
        q = np.asarray(quantiles, dtype=float).ravel()
        if np.any((q <= 0) | (q >= 1)):
            raise ValueError("quantiles must lie strictly inside (0, 1)")
        out = sample_iqn(
            self.model_, X, self.x_mean_, self.x_std_,
            self.y_mean_, self.y_std_,
            chunk=self.chunk if chunk is None else chunk, taus=q)
        return out.T

    def predict(self, X, return_std: bool = False,
                n_samples: Optional[int] = None, method: str = "samples"):
        """Predict the conditional mean of ``y | x``.

        Args:
            X: ``(n, d)`` inputs.
            return_std: also return the predictive standard deviation.
            n_samples: number of quantile levels used to average.
            method: ``"samples"`` (default) averages deterministic grid-based
                quantile evaluations for stable numerical integration of the
                conditional mean. ``"mean_head"`` reads the network's L1 mean
                head directly: one forward pass, but no ``return_std``.

        Returns:
            ``mu`` of shape ``(n,)``, or ``(mu, sd)`` when ``return_std``.
        """
        self._check_fitted()
        if method == "mean_head":
            if return_std:
                raise ValueError(
                    "return_std requires method='samples'")
            return self._mean_head(X)
        if method != "samples":
            raise ValueError("method must be 'samples' or 'mean_head'")
        s = self.sample(X, n_samples=n_samples, method="grid")
        mu = s.mean(0)
        return (mu, s.std(0)) if return_std else mu

    def _mean_head(self, X) -> np.ndarray:
        X = self._prepare(X)
        device = next(self.model_.parameters()).device
        Xt = torch.tensor((X - self.x_mean_) / self.x_std_,
                          dtype=torch.float32, device=device)
        chunk = self.chunk
        if chunk is None or chunk == "auto":
            chunk = auto_chunk(len(Xt), self.hdim)
        outs = []
        with torch.inference_mode():
            for i in range(0, len(Xt), chunk):
                outs.append(
                    self.model_(Xt[i:i + chunk], 0.5)[:, 0].cpu().numpy())
        return np.concatenate(outs) * self.y_std_ + self.y_mean_

    def predict_interval(self, X, alpha: float = 0.90,
                         n_samples: Optional[int] = None):
        """Predict a central ``alpha`` prediction interval.

        Args:
            X: ``(n, d)`` inputs.
            alpha: nominal coverage, e.g. ``0.90`` for a 90% interval.

        Returns:
            ``(lo, hi)``, each of shape ``(n,)``.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly inside (0, 1)")
        lo, hi = (1 - alpha) / 2, 1 - (1 - alpha) / 2
        q = self.predict_quantiles(X, quantiles=(lo, hi))
        return q[:, 0], q[:, 1]

    def score(self, X, y) -> float:
        """Negative RMSE of the conditional mean (higher is better).

        Sign follows the scikit-learn convention so that this can be used
        directly with ``GridSearchCV`` and friends.
        """
        y = np.asarray(y, dtype=float).ravel()
        return -float(np.sqrt(np.mean((self.predict(X) - y) ** 2)))

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path) -> None:
        """Save the fitted model and its hyperparameters to ``path``."""
        self._check_fitted()
        params = self.get_params()
        params["device"] = None  # resolved at load time
        payload = {
            "format": 1,
            "class": type(self).__name__,
            "params": params,
            "state_dict": {k: v.cpu()
                           for k, v in self.model_.state_dict().items()},
            "n_features_in": int(self.n_features_in_),
            "x_mean": torch.as_tensor(np.asarray(self.x_mean_, dtype=float)),
            "x_std": torch.as_tensor(np.asarray(self.x_std_, dtype=float)),
            "y_mean": float(self.y_mean_),
            "y_std": float(self.y_std_),
            "extra": self._extra_state(),
        }
        torch.save(payload, path)

    def _extra_state(self) -> dict:
        """Subclass hook: extra tensors/scalars to persist. Must be
        ``weights_only``-safe (tensors, numbers, strings, and containers)."""
        return {}

    def _load_extra_state(self, extra: dict, device) -> None:
        """Subclass hook: restore what :meth:`_extra_state` saved."""
        return None

    @classmethod
    def load(cls, path, device=None) -> "GBCRegressor":
        """Load a model saved with :meth:`save`."""
        device = resolve_device(device)
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
        except TypeError:  # torch < 1.13
            payload = torch.load(path, map_location=device)
        saved_cls = payload.get("class", cls.__name__)
        if saved_cls != cls.__name__:
            raise ValueError(
                f"{path} holds a {saved_cls}; load it with "
                f"{saved_cls}.load(...)")
        obj = cls(**payload["params"])
        obj.device = device
        obj.n_features_in_ = int(payload["n_features_in"])
        obj.x_mean_ = payload["x_mean"].cpu().numpy()
        obj.x_std_ = payload["x_std"].cpu().numpy()
        obj.y_mean_ = float(payload["y_mean"])
        obj.y_std_ = float(payload["y_std"])
        # The network's input width can exceed n_features_in_ (GBC-Aug adds a
        # feature), so take it from the saved weights.
        xdim = int(payload["state_dict"]["fc_x.0.weight"].shape[1])
        model = IQN(xdim, hdim=obj.hdim, nh=obj.nh)
        model.load_state_dict(payload["state_dict"])
        obj.model_ = model.to(device).eval()
        obj._load_extra_state(payload.get("extra", {}), device)
        return obj
