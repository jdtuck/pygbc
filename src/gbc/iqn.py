"""Implicit Quantile Network (IQN) — the GBC backbone.

Architecture: cosine quantile embedding (Dabney et al. 2018).
Loss: three-term (L1 mean anchor + monotonicity + quantile).
Optimizer: Adam + cosine annealing.

The network and the default training/sampling path are a faithful port of the
reference implementation released with Polson & Sokolov (2026). With default
arguments, :func:`train_iqn` and :func:`sample_iqn` reproduce the reference
results bit for bit. Everything beyond the defaults is opt-in.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = ["IQN", "train_iqn", "sample_iqn", "resolve_device"]


def resolve_device(device=None) -> torch.device:
    """Return ``device`` as a :class:`torch.device`, defaulting to CUDA if available."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class IQN(nn.Module):
    """Implicit Quantile Network.

    Cosine embedding: ``phi(tau) = [cos(0*pi*tau), ..., cos((nh-1)*pi*tau)]``.

    Args:
        xdim: number of input features.
        hdim: width of the hidden layers.
        nh: number of cosine basis functions in the quantile embedding.

    Forward returns a ``(batch, 2)`` tensor where column 0 is the conditional
    mean head and column 1 is the quantile head at level ``tau``.

    Note:
        ``tau`` is a *scalar* (python float or 0-d tensor): one quantile level
        is evaluated per forward pass, exactly as in the reference code.
    """

    def __init__(self, xdim: int, hdim: int = 256, nh: int = 32):
        super().__init__()
        self.xdim = int(xdim)
        self.hdim = int(hdim)
        self.nh = int(nh)
        self.fc_tau = nn.Sequential(nn.Linear(nh, hdim), nn.ReLU())
        self.fc_x = nn.Sequential(nn.Linear(xdim, hdim), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hdim, 64), nn.Tanh())
        self.fc_out = nn.Linear(64, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, tau) -> torch.Tensor:
        i = torch.arange(self.nh, dtype=torch.float32, device=x.device)
        h_tau = self.fc_tau(torch.cos(i * torch.pi * tau))
        h_x = self.fc_x(x)
        h = self.fc1(h_x * h_tau.unsqueeze(0))
        return self.fc_out(self.fc2(h))

    def loss_fn(self, x, y, w=(0.3, 0.3, 0.4), tau=None):
        """Three-term loss at a single quantile level.

        Args:
            x: ``(n, xdim)`` standardized inputs.
            y: ``(n,)`` standardized responses.
            w: ``(w_mean, w_mono, w_quantile)`` loss weights.
                ``w[0]`` L1 on the conditional mean (location anchor);
                ``w[1]`` monotonicity regularization (quantile ordering);
                ``w[2]`` standard quantile (pinball) loss.
            tau: quantile level. If ``None`` (default) one is drawn uniformly
                on the device, matching the reference implementation.
        """
        if tau is None:
            tau = torch.rand(1, device=x.device).item()
        tauind = tau < 0.5
        f = self(x, tau)
        e = y.view(-1, 1) - f
        loss = w[0] * torch.mean(torch.abs(e[:, 0]))
        mono = (tauind * torch.mean(torch.relu(-e[:, 1]))
                + (1 - tauind) * torch.mean(torch.relu(e[:, 1])))
        loss += w[1] * abs(tau - 0.5) * mono
        loss += w[2] * torch.mean(
            torch.maximum(tau * e[:, 1], (tau - 1) * e[:, 1]))
        return loss


def _pinball(model: IQN, Xt: torch.Tensor, yt: torch.Tensor,
             taus: Sequence[float]) -> float:
    """Mean pinball loss over a fixed grid of taus (used for early stopping)."""
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad():
        for tau in taus:
            e = yt.view(-1, 1) - model(Xt, float(tau))
            total += torch.mean(
                torch.maximum(tau * e[:, 1], (tau - 1) * e[:, 1])).item()
    if was_training:
        model.train()
    return total / len(taus)


def train_iqn(
    X_np,
    y_np,
    epochs: int = 3000,
    hdim: int = 256,
    nh: int = 32,
    lr: float = 1e-3,
    wd: float = 1e-4,
    seed: int = 42,
    loss_w: Tuple[float, float, float] = (0.3, 0.3, 0.4),
    device=None,
    *,
    batch_size: Optional[int] = None,
    taus_per_step: int = 1,
    validation_data=None,
    patience: Optional[int] = None,
    callback: Optional[Callable[[int, float], None]] = None,
    model: Optional[IQN] = None,
):
    """Train an IQN with Adam + cosine annealing.

    With default arguments this is the reference training loop: full-batch,
    one uniformly drawn ``tau`` per epoch, no validation. All keyword-only
    arguments below are opt-in extensions that change the numerics.

    Args:
        X_np: ``(n, d)`` array of inputs.
        y_np: ``(n,)`` array of responses.
        epochs: number of optimizer steps (full-batch) or epochs over the
            data (when ``batch_size`` is set).
        hdim, nh: network width and quantile-embedding size.
        lr: Adam learning rate; cosine-annealed to ``lr * 0.01``.
        wd: Adam weight decay.
        seed: torch manual seed, set before initialization.
        loss_w: three-term loss weights, see :meth:`IQN.loss_fn`.
        device: torch device; defaults to CUDA when available.

    Keyword-only (opt-in, off by default):
        batch_size: if set, use shuffled minibatches of this size instead of
            full-batch gradient steps.
        taus_per_step: average the loss over this many quantile levels per
            optimizer step (variance reduction). ``1`` = reference behavior.
        validation_data: ``(X_val, y_val)`` used for early stopping. Scored
            with mean pinball loss on a fixed 9-point tau grid.
        patience: stop after this many epochs without validation improvement.
            Requires ``validation_data``. The best weights are restored.
        callback: called as ``callback(epoch, loss)`` after each epoch.
        model: reuse/warm-start an existing :class:`IQN` instead of building
            a fresh one (normalization is still recomputed from ``X_np``).

    Returns:
        ``(model, xm, xs, ym, ys)`` — the trained model and the normalization
        statistics needed by :func:`sample_iqn`.
    """
    device = resolve_device(device)
    X_np = np.asarray(X_np, dtype=float)
    y_np = np.asarray(y_np, dtype=float).ravel()
    if X_np.ndim != 2:
        raise ValueError(f"X must be 2-D (n, d); got shape {X_np.shape}")
    if len(X_np) != len(y_np):
        raise ValueError(
            f"X and y length mismatch: {len(X_np)} vs {len(y_np)}")
    if taus_per_step < 1:
        raise ValueError("taus_per_step must be >= 1")
    if patience is not None and validation_data is None:
        raise ValueError("patience requires validation_data")

    torch.manual_seed(seed)
    xm, xs = X_np.mean(0), X_np.std(0) + 1e-8
    ym, ys = float(y_np.mean()), float(y_np.std()) + 1e-8
    Xt = torch.tensor((X_np - xm) / xs, dtype=torch.float32, device=device)
    yt = torch.tensor((y_np - ym) / ys, dtype=torch.float32, device=device)

    if model is None:
        model = IQN(X_np.shape[1], hdim=hdim, nh=nh).to(device)
    else:
        model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.01)

    val_taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if validation_data is not None:
        Xv_np = np.asarray(validation_data[0], dtype=float)
        yv_np = np.asarray(validation_data[1], dtype=float).ravel()
        Xv = torch.tensor((Xv_np - xm) / xs, dtype=torch.float32,
                          device=device)
        yv = torch.tensor((yv_np - ym) / ys, dtype=torch.float32,
                          device=device)
    best_score, best_state, bad_epochs = float("inf"), None, 0

    n = len(Xt)
    model.train()
    for epoch in range(epochs):
        if batch_size is None or batch_size >= n:
            batches: Iterable = [slice(None)]
        else:
            perm = torch.randperm(n, device=device)
            batches = [perm[i:i + batch_size]
                       for i in range(0, n, batch_size)]

        epoch_loss = 0.0
        for idx in batches:
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            if taus_per_step == 1:
                loss = model.loss_fn(xb, yb, w=loss_w)
            else:
                loss = sum(model.loss_fn(xb, yb, w=loss_w)
                           for _ in range(taus_per_step)) / taus_per_step
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
        sched.step()
        epoch_loss /= len(batches)

        if callback is not None:
            callback(epoch, epoch_loss)

        if patience is not None:
            score = _pinball(model, Xv, yv, val_taus)
            if score < best_score - 1e-9:
                best_score, bad_epochs = score, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, xm, xs, ym, ys


def sample_iqn(model: IQN, X_np, xm, xs, ym, ys, B: int = 500,
               chunk: int = 1000, device=None, taus=None) -> np.ndarray:
    """Evaluate a trained IQN on a grid of quantile levels.

    Args:
        model: trained :class:`IQN`.
        X_np: ``(n, d)`` inputs on the original scale.
        xm, xs, ym, ys: normalization statistics returned by :func:`train_iqn`.
        B: number of quantile levels, equally spaced on ``[0.005, 0.995]``.
        chunk: rows evaluated per forward pass (memory control).
        device: torch device; defaults to the model's device.
        taus: explicit sequence of quantile levels. Overrides ``B``.

    Returns:
        ``(B, n)`` array of quantile predictions on the original scale, with
        rows ordered by increasing tau.
    """
    if device is None:
        device = next(model.parameters()).device
    device = resolve_device(device)
    X_np = np.asarray(X_np, dtype=float)
    if X_np.ndim != 2:
        raise ValueError(f"X must be 2-D (n, d); got shape {X_np.shape}")

    Xt = torch.tensor((X_np - xm) / xs, dtype=torch.float32, device=device)
    if taus is None:
        tau_t = torch.linspace(0.005, 0.995, B, device=device)
    else:
        tau_t = torch.as_tensor(np.asarray(taus, dtype=float),
                                dtype=torch.float32, device=device)
    rows = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for tau in tau_t:
            q_list = []
            for i in range(0, len(Xt), chunk):
                q_list.append(
                    model(Xt[i:i + chunk], tau.item())[:, 1].cpu().numpy())
            rows.append(np.concatenate(q_list) * ys + ym)
    if was_training:
        model.train()
    return np.array(rows)
