"""Boundary-augmented IQN (Aug-IQN / GBC-Aug).

For response surfaces with a jump or regime change, a single smooth quantile
network has to bridge the discontinuity. GBC-Aug gives it a hint:

  1. EM-cluster the marginal ``y`` into two components (GaussianMixture).
  2. Train an MLP classifier ``X -> P(regime = 1 | X)``.
  3. Append that probability as an extra input feature: ``X_aug = [X, f(X)]``.
  4. Train a standard IQN on the augmented inputs.

Faithful port of the reference implementation released with Polson & Sokolov
(2026). With default arguments these functions reproduce it bit for bit.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .iqn import resolve_device

__all__ = ["cluster_y", "ClassifierMLP", "train_classifier",
           "get_regime_prob", "augment_features", "thread_limit"]


@contextlib.contextmanager
def thread_limit(threads: Optional[int]):
    """Temporarily cap the native thread pools (BLAS / OpenMP).

    ``threads=None`` is a no-op. This exists for two reasons:

    * **Crash avoidance.** On macOS, PyTorch and a conda/MKL scikit-learn
      often load two different OpenMP runtimes (``libomp`` and
      ``libiomp5``) into one process. scikit-learn's k-means — which
      :class:`~sklearn.mixture.GaussianMixture` uses to initialize — then
      segfaults inside its OpenMP parallel region. Running it
      single-threaded avoids that region.
    * **Determinism.** Thread count changes floating-point summation order,
      so an unpinned k-means can give bitwise-different cluster means on
      different machines. Pinning makes :func:`cluster_y` reproducible.
    """
    if threads is None:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with sklearn
        yield
        return
    with threadpool_limits(limits=threads):
        yield


def cluster_y(y, n_components: int = 2, seed: int = 0,
              init_params: str = "kmeans",
              threads: Optional[int] = 1) -> np.ndarray:
    """EM-cluster the marginal response into regimes.

    Args:
        y: ``(n,)`` responses.
        n_components: number of mixture components (2 in the paper).
        seed: ``random_state`` for the GaussianMixture.
        init_params: how the mixture is initialized. ``"kmeans"`` (default)
            matches the paper. ``"random_from_data"`` and ``"random"`` skip
            k-means entirely — use one of them if k-means segfaults in your
            environment and ``threads=1`` did not help.
        threads: native thread cap for the fit, via :func:`thread_limit`.
            ``1`` (default) is deterministic and dodges the macOS OpenMP
            crash; ``None`` uses whatever the libraries default to, which is
            what the upstream script does.

    Returns:
        ``(n,)`` integer labels, relabeled so that component 0 has the lower
        mean — the ordering the classifier and the paper assume.
    """
    from sklearn.mixture import GaussianMixture

    y = np.asarray(y, dtype=float).ravel()
    gm = GaussianMixture(n_components=n_components, random_state=seed,
                         init_params=init_params)
    with thread_limit(threads):
        gm.fit(y.reshape(-1, 1))
        labels = gm.predict(y.reshape(-1, 1))
    if n_components == 2:
        if gm.means_[0, 0] > gm.means_[1, 0]:
            labels = 1 - labels
    else:
        order = np.argsort(gm.means_[:, 0])
        remap = np.empty(n_components, dtype=int)
        remap[order] = np.arange(n_components)
        labels = remap[labels]
    return labels


class ClassifierMLP(nn.Module):
    """MLP binary classifier: ``X -> P(regime = 1 | X)``.

    Args:
        xdim: number of input features.
        hdim: hidden width.
        n_layers: number of hidden layers.
    """

    def __init__(self, xdim: int, hdim: int = 256, n_layers: int = 3):
        super().__init__()
        self.xdim = int(xdim)
        self.hdim = int(hdim)
        self.n_layers = int(n_layers)
        layers = [nn.Linear(xdim, hdim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hdim, hdim), nn.ReLU()]
        layers += [nn.Linear(hdim, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def train_classifier(X_np, labels, xm, xs, hdim: int = 256, n_layers: int = 3,
                     epochs: int = 2000, lr: float = 1e-3, seed: int = 0,
                     device=None, *, callback: Optional[Callable] = None):
    """Train the regime classifier with BCE + cosine annealing.

    Args:
        X_np: ``(n, d)`` inputs on the original scale.
        labels: ``(n,)`` binary regime labels from :func:`cluster_y`.
        xm, xs: input normalization statistics (mean and std of ``X_np``).
        hdim, n_layers: classifier architecture.
        epochs: full-batch optimizer steps.
        lr: Adam learning rate, cosine-annealed to ``lr * 0.01``.
        seed: torch manual seed set before initialization.
        device: torch device; defaults to CUDA when available.
        callback: optional ``callback(epoch, loss)`` after each step.

    Returns:
        ``(classifier, training_accuracy)``.
    """
    device = resolve_device(device)
    X_np = np.asarray(X_np, dtype=float)
    labels = np.asarray(labels, dtype=float).ravel()
    if len(X_np) != len(labels):
        raise ValueError(
            f"X and labels length mismatch: {len(X_np)} vs {len(labels)}")

    torch.manual_seed(seed)
    Xt = torch.tensor((X_np - xm) / xs, dtype=torch.float32, device=device)
    yt = torch.tensor(labels, dtype=torch.float32, device=device)

    clf = ClassifierMLP(X_np.shape[1], hdim=hdim, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.01)

    clf.train()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = clf(Xt)
        loss = nn.functional.binary_cross_entropy(pred, yt)
        loss.backward()
        opt.step()
        sched.step()
        if callback is not None:
            callback(epoch, float(loss.detach()))

    clf.eval()
    with torch.no_grad():
        acc = ((clf(Xt) > 0.5).float() == yt).float().mean().item()
    return clf, acc


def get_regime_prob(clf: ClassifierMLP, X_np, xm, xs, device=None) -> np.ndarray:
    """Evaluate ``P(regime = 1 | X)`` for a trained classifier."""
    if device is None:
        device = next(clf.parameters()).device
    device = resolve_device(device)
    X_np = np.asarray(X_np, dtype=float)
    Xt = torch.tensor((X_np - xm) / xs, dtype=torch.float32, device=device)
    was_training = clf.training
    clf.eval()
    with torch.no_grad():
        prob = clf(Xt).cpu().numpy()
    if was_training:
        clf.train()
    return prob


def augment_features(X_np, clf: ClassifierMLP, xm, xs, device=None) -> np.ndarray:
    """Append the regime probability as an extra feature: ``[X, f(X)]``."""
    X_np = np.asarray(X_np, dtype=float)
    fhat = get_regime_prob(clf, X_np, xm, xs, device=device)
    return np.column_stack([X_np, fhat])
