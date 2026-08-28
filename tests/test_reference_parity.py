"""Bit-for-bit parity with the upstream reference implementation.

The reference `IQN`, `train_iqn` and `sample_iqn` from
https://github.com/VadimSokolov/gbc-surrogate (gbc/iqn.py) are reproduced
verbatim below. The package must produce *identical* numbers with default
arguments, both through the functional API and through `GBCRegressor`.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

import gbc


# ── Verbatim upstream reference ──────────────────────────────────────────────

class RefIQN(nn.Module):
    def __init__(self, xdim, hdim=256, nh=32):
        super().__init__()
        self.nh = nh
        self.fc_tau = nn.Sequential(nn.Linear(nh, hdim), nn.ReLU())
        self.fc_x = nn.Sequential(nn.Linear(xdim, hdim), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hdim, 64), nn.Tanh())
        self.fc_out = nn.Linear(64, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, tau):
        i = torch.arange(self.nh, dtype=torch.float32, device=x.device)
        h_tau = self.fc_tau(torch.cos(i * torch.pi * tau))
        h_x = self.fc_x(x)
        h = self.fc1(h_x * h_tau.unsqueeze(0))
        return self.fc_out(self.fc2(h))

    def loss_fn(self, x, y, w=(0.3, 0.3, 0.4)):
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


def ref_train_iqn(X_np, y_np, epochs=3000, hdim=256, nh=32, lr=1e-3, wd=1e-4,
                  seed=42, loss_w=(0.3, 0.3, 0.4), device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    xm, xs = X_np.mean(0), X_np.std(0) + 1e-8
    ym, ys = float(y_np.mean()), float(y_np.std()) + 1e-8
    Xt = torch.tensor((X_np - xm) / xs, dtype=torch.float32, device=device)
    yt = torch.tensor((y_np - ym) / ys, dtype=torch.float32, device=device)

    model = RefIQN(X_np.shape[1], hdim=hdim, nh=nh).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.01)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = model.loss_fn(Xt, yt, w=loss_w)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model, xm, xs, ym, ys


def ref_sample_iqn(model, X_np, xm, xs, ym, ys, B=500, chunk=1000, device=None):
    if device is None:
        device = next(model.parameters()).device
    Xt = torch.tensor((X_np - xm) / xs, dtype=torch.float32, device=device)
    taus = torch.linspace(0.005, 0.995, B, device=device)
    rows = []
    with torch.no_grad():
        for tau in taus:
            q_list = []
            for i in range(0, len(Xt), chunk):
                q_list.append(
                    model(Xt[i:i + chunk], tau.item())[:, 1].cpu().numpy())
            rows.append(np.concatenate(q_list) * ys + ym)
    return np.array(rows)


# ── Fixtures ─────────────────────────────────────────────────────────────────

EPOCHS = 60
HDIM = 32
NH = 8
B = 21


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(80, 2))
    y = np.sin(6 * X[:, 0]) + 0.3 * X[:, 1] + 0.1 * rng.standard_normal(80)
    Xte = rng.uniform(0, 1, size=(25, 2))
    return X, y, Xte


# ── Tests ────────────────────────────────────────────────────────────────────

def test_functional_api_matches_reference(data):
    X, y, Xte = data
    kw = dict(epochs=EPOCHS, hdim=HDIM, nh=NH, seed=7,
              device=torch.device("cpu"))

    ref_m, *ref_norm = ref_train_iqn(X, y, **kw)
    pkg_m, *pkg_norm = gbc.train_iqn(X, y, **kw)

    for a, b in zip(ref_norm, pkg_norm):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    ref_sd = ref_m.state_dict()
    for k, v in pkg_m.state_dict().items():
        assert torch.equal(v, ref_sd[k]), f"weights differ at {k}"

    ref_s = ref_sample_iqn(ref_m, Xte, *ref_norm, B=B)
    pkg_s = gbc.sample_iqn(pkg_m, Xte, *pkg_norm, B=B)
    np.testing.assert_array_equal(ref_s, pkg_s)


def test_regressor_matches_reference(data):
    """GBCRegressor with default extras == the paper's numbers."""
    X, y, Xte = data
    ref_m, *ref_norm = ref_train_iqn(
        X, y, epochs=EPOCHS, hdim=HDIM, nh=NH, seed=7,
        device=torch.device("cpu"))
    ref_mu = ref_sample_iqn(ref_m, Xte, *ref_norm, B=B).mean(0)

    model = gbc.GBCRegressor(epochs=EPOCHS, hdim=HDIM, nh=NH, seed=7,
                             n_samples=B, device="cpu").fit(X, y)
    np.testing.assert_array_equal(ref_mu, model.predict(Xte))


def test_loss_fn_matches_reference_at_fixed_tau(data):
    X, y, _ = data
    torch.manual_seed(3)
    ref = RefIQN(2, hdim=HDIM, nh=NH)
    torch.manual_seed(3)
    pkg = gbc.IQN(2, hdim=HDIM, nh=NH)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(11)
    a = ref.loss_fn(Xt, yt)
    torch.manual_seed(11)
    b = pkg.loss_fn(Xt, yt)
    assert torch.equal(a, b)
