"""Boundary-augmented GBC surrogate (GBC-Aug), object-oriented interface.

    >>> from gbc import AugGBCRegressor
    >>> model = AugGBCRegressor(epochs=8000, loss_w=(0.10, 0.20, 0.70))
    >>> model.fit(X_train, y_train)
    >>> lo, hi = model.predict_interval(X_test)
    >>> model.classifier_accuracy_
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .augiqn import (ClassifierMLP, augment_features, cluster_y,
                     get_regime_prob, train_classifier)
from .model import GBCRegressor

__all__ = ["AugGBCRegressor"]


class AugGBCRegressor(GBCRegressor):
    """GBC surrogate with a learned regime feature, for jump surfaces.

    Same interface as :class:`~gbc.GBCRegressor` — ``fit``, ``predict``,
    ``predict_quantiles``, ``predict_interval``, ``sample``, ``save``,
    ``load`` — with one extra preprocessing stage inside ``fit``:

      1. EM-cluster the marginal ``y`` into ``n_components`` regimes.
      2. Train an MLP classifier ``X -> P(regime = 1 | X)``.
      3. Append that probability as an extra input feature.
      4. Fit the IQN on the augmented inputs.

    The augmentation is applied automatically at predict time, so callers
    always pass the original ``d``-column ``X``.

    On smooth surfaces this buys nothing over :class:`~gbc.GBCRegressor` and
    costs an extra network — it is aimed at responses with a discontinuity,
    where the paper pairs it with the quantile-dominant loss weighting
    ``loss_w=(0.10, 0.20, 0.70)``.

    Args:
        n_components: mixture components for the EM clustering of ``y``.
            Two (default) gives a single binary regime probability.
        clf_hdim: classifier hidden width.
        clf_layers: number of classifier hidden layers.
        clf_epochs: classifier training steps (full batch).
        clf_lr: classifier Adam learning rate.
        clf_seed: seed for the mixture and the classifier. ``None`` (default)
            reuses ``seed``. The paper's Table 3 uses the replicate index here
            and ``rep * 13 + 7`` for ``seed``.
        init_params: mixture initialization, passed to
            :func:`~gbc.augiqn.cluster_y`. ``"kmeans"`` (default) matches the
            paper; ``"random_from_data"`` avoids scikit-learn's k-means,
            which segfaults on some macOS installs (see the README's
            troubleshooting section).
        cluster_threads: native thread cap for the mixture fit. ``1``
            (default) makes the clustering reproducible across machines and
            avoids the macOS OpenMP crash; ``None`` restores the library
            default used by the upstream script.

    All other arguments are inherited from :class:`~gbc.GBCRegressor`.

    Attributes:
        classifier_: the fitted :class:`~gbc.augiqn.ClassifierMLP`.
        classifier_accuracy_: training accuracy of the regime classifier —
            worth checking. Near 0.5 means the regimes are not predictable
            from ``X`` and the extra feature is noise.
        regime_labels_: ``(n,)`` EM labels assigned to the training responses.
        clf_x_mean_, clf_x_std_: input normalization used by the classifier
            (computed on the raw ``X``, separate from the IQN's statistics
            over the augmented inputs).
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
        n_components: int = 2,
        clf_hdim: int = 256,
        clf_layers: int = 3,
        clf_epochs: int = 2000,
        clf_lr: float = 1e-3,
        clf_seed: Optional[int] = None,
        init_params: str = "kmeans",
        cluster_threads: Optional[int] = 1,
    ):
        super().__init__(
            epochs=epochs, hdim=hdim, nh=nh, lr=lr,
            weight_decay=weight_decay, loss_w=loss_w, seed=seed,
            device=device, n_samples=n_samples, chunk=chunk,
            batch_size=batch_size, taus_per_step=taus_per_step,
            validation_fraction=validation_fraction, patience=patience,
            verbose=verbose, foreach=foreach,
            track_history=track_history)
        self.n_components = n_components
        self.clf_hdim = clf_hdim
        self.clf_layers = clf_layers
        self.clf_epochs = clf_epochs
        self.clf_lr = clf_lr
        self.clf_seed = clf_seed
        self.init_params = init_params
        self.cluster_threads = cluster_threads

        self.classifier_: Optional[ClassifierMLP] = None
        self.classifier_accuracy_: Optional[float] = None
        self.regime_labels_: Optional[np.ndarray] = None
        self.clf_x_mean_ = self.clf_x_std_ = None
        self.clf_history_: list = []

    @property
    def _clf_seed(self) -> int:
        return self.seed if self.clf_seed is None else self.clf_seed

    # ── the extra stage ──────────────────────────────────────────────────

    def _fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.n_components < 2:
            raise ValueError("n_components must be >= 2")
        seed = self._clf_seed
        labels = cluster_y(y, n_components=self.n_components, seed=seed,
                           init_params=self.init_params,
                           threads=self.cluster_threads)
        if self.n_components > 2:
            # Collapse to "is this the top regime?" so the appended feature
            # stays a single probability, as in the paper.
            labels = (labels == self.n_components - 1).astype(int)

        self.regime_labels_ = labels
        self.clf_x_mean_ = X.mean(0)
        self.clf_x_std_ = X.std(0) + 1e-8
        self.clf_history_ = []

        def _cb(epoch: int, loss: float) -> None:
            self.clf_history_.append(loss)
            if self.verbose and (epoch % self.verbose == 0
                                 or epoch == self.clf_epochs - 1):
                print(f"[clf] epoch {epoch + 1}/{self.clf_epochs}  "
                      f"loss={loss:.6f}")

        clf, acc = train_classifier(
            X, labels, self.clf_x_mean_, self.clf_x_std_,
            hdim=self.clf_hdim, n_layers=self.clf_layers,
            epochs=self.clf_epochs, lr=self.clf_lr, seed=seed,
            device=self.device, callback=_cb)
        self.classifier_ = clf
        self.classifier_accuracy_ = acc
        if self.verbose:
            print(f"[clf] training accuracy = {acc:.4f}")
        return self._transform(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.classifier_ is None:
            return X
        return augment_features(X, self.classifier_,
                                self.clf_x_mean_, self.clf_x_std_)

    def regime_prob(self, X) -> np.ndarray:
        """Return ``P(regime = 1 | X)`` from the fitted classifier."""
        self._check_fitted()
        X = self._check_X(X)
        return get_regime_prob(self.classifier_, X,
                               self.clf_x_mean_, self.clf_x_std_)

    # ── persistence ──────────────────────────────────────────────────────

    def _extra_state(self) -> dict:
        return {
            "clf_state_dict": {k: v.cpu() for k, v
                               in self.classifier_.state_dict().items()},
            "clf_x_mean": torch.as_tensor(
                np.asarray(self.clf_x_mean_, dtype=float)),
            "clf_x_std": torch.as_tensor(
                np.asarray(self.clf_x_std_, dtype=float)),
            "classifier_accuracy": float(self.classifier_accuracy_),
        }

    def _load_extra_state(self, extra: dict, device) -> None:
        if not extra:
            raise ValueError("Saved file has no classifier state")
        self.clf_x_mean_ = extra["clf_x_mean"].cpu().numpy()
        self.clf_x_std_ = extra["clf_x_std"].cpu().numpy()
        self.classifier_accuracy_ = float(extra["classifier_accuracy"])
        clf = ClassifierMLP(self.n_features_in_, hdim=self.clf_hdim,
                            n_layers=self.clf_layers)
        clf.load_state_dict(extra["clf_state_dict"])
        self.classifier_ = clf.to(device).eval()
