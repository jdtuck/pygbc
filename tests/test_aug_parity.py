"""Bit-for-bit parity of the GBC-Aug pipeline with the upstream reference.

The upstream `cluster_y`, `ClassifierMLP`, `train_classifier`,
`get_regime_prob` and `augment_features` from gbc/augiqn.py are reproduced
verbatim below, wired together exactly as `experiments/tab3_flowers.py`
does it. `AugGBCRegressor` must produce identical numbers.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

import gbc
from gbc import AugGBCRegressor

from test_reference_parity import ref_sample_iqn, ref_train_iqn


# ── Verbatim upstream reference ──────────────────────────────────────────────

def ref_cluster_y(y, n_components=2, seed=0):
    gm = GaussianMixture(n_components=n_components, random_state=seed)
    gm.fit(y.reshape(-1, 1))
    labels = gm.predict(y.reshape(-1, 1))
    if gm.means_[0, 0] > gm.means_[1, 0]:
        labels = 1 - labels
    return labels


class RefClassifierMLP(nn.Module):
    def __init__(self, xdim, hdim=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(xdim, hdim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hdim, hdim), nn.ReLU()]
        layers += [nn.Linear(hdim, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def ref_train_classifier(X_np, labels, xm, xs, hdim=256, n_layers=3,
                         epochs=2000, lr=1e-3, seed=0, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xt = torch.tensor((X_np - xm) / xs, dtype=torch.float32, device=device)
    yt = torch.tensor(labels, dtype=torch.float32, device=device)

    clf = RefClassifierMLP(X_np.shape[1], hdim=hdim, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.01)

    clf.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = clf(Xt)
        loss = nn.functional.binary_cross_entropy(pred, yt)
        loss.backward()
        opt.step()
        sched.step()

    clf.eval()
    with torch.no_grad():
        acc = ((clf(Xt) > 0.5).float() == yt).float().mean().item()
    return clf, acc


def ref_get_regime_prob(clf, X_np, xm, xs, device=None):
    if device is None:
        device = next(clf.parameters()).device
    Xt = torch.tensor((X_np - xm) / xs, dtype=torch.float32, device=device)
    with torch.no_grad():
        prob = clf(Xt).cpu().numpy()
    return prob


def ref_augment_features(X_np, clf, xm, xs, device=None):
    fhat = ref_get_regime_prob(clf, X_np, xm, xs, device=device)
    return np.column_stack([X_np, fhat])


# ── Fixtures ─────────────────────────────────────────────────────────────────

CPU = torch.device("cpu")
IQN_EPOCHS = 60
CLF_EPOCHS = 40
HDIM = 32
NH = 8
CLF_HDIM = 16
CLF_LAYERS = 2
B = 21
SEED_IQN = 7 * 13 + 7
SEED_CLF = 7


@pytest.fixture(scope="module")
def jump_data():
    """A surface with a jump — the regime the augmentation is meant for."""
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, size=(120, 2))
    y = np.where(X[:, 0] < 0.5, 0.0, 3.0) + 0.5 * X[:, 1]
    y = y + 0.1 * rng.standard_normal(120)
    Xte = rng.uniform(0, 1, size=(30, 2))
    yte = np.where(Xte[:, 0] < 0.5, 0.0, 3.0) + 0.5 * Xte[:, 1]
    return X, y, Xte, yte


def _reference_pipeline(X, y, Xte):
    """Exactly experiments/tab3_flowers.py::run_augiqn for one replicate."""
    labels = ref_cluster_y(y, seed=SEED_CLF)
    xm, xs = X.mean(0), X.std(0) + 1e-8
    clf, acc = ref_train_classifier(
        X, labels, xm, xs, hdim=CLF_HDIM, n_layers=CLF_LAYERS,
        epochs=CLF_EPOCHS, seed=SEED_CLF, device=CPU)
    X_aug_tr = ref_augment_features(X, clf, xm, xs, device=CPU)
    X_aug_te = ref_augment_features(Xte, clf, xm, xs, device=CPU)
    mdl, *norm = ref_train_iqn(
        X_aug_tr, y, epochs=IQN_EPOCHS, hdim=HDIM, nh=NH, seed=SEED_IQN,
        loss_w=(0.10, 0.20, 0.70), device=CPU)
    samp = ref_sample_iqn(mdl, X_aug_te, *norm, B=B)
    return labels, acc, samp


def _package_model(**overrides):
    kw = dict(epochs=IQN_EPOCHS, hdim=HDIM, nh=NH, seed=SEED_IQN,
              loss_w=(0.10, 0.20, 0.70), device="cpu", n_samples=B,
              clf_hdim=CLF_HDIM, clf_layers=CLF_LAYERS,
              clf_epochs=CLF_EPOCHS, clf_seed=SEED_CLF)
    kw.update(overrides)
    return AugGBCRegressor(**kw)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_functional_augiqn_matches_reference(jump_data):
    X, y, Xte, _ = jump_data
    xm, xs = X.mean(0), X.std(0) + 1e-8

    np.testing.assert_array_equal(ref_cluster_y(y, seed=0),
                                  gbc.cluster_y(y, seed=0))

    ref_clf, ref_acc = ref_train_classifier(
        X, ref_cluster_y(y, seed=0), xm, xs, hdim=CLF_HDIM,
        n_layers=CLF_LAYERS, epochs=CLF_EPOCHS, seed=0, device=CPU)
    pkg_clf, pkg_acc = gbc.train_classifier(
        X, gbc.cluster_y(y, seed=0), xm, xs, hdim=CLF_HDIM,
        n_layers=CLF_LAYERS, epochs=CLF_EPOCHS, seed=0, device=CPU)

    assert ref_acc == pkg_acc
    ref_sd = ref_clf.state_dict()
    for k, v in pkg_clf.state_dict().items():
        assert torch.equal(v, ref_sd["net." + k.split("net.", 1)[-1]]), k

    np.testing.assert_array_equal(
        ref_augment_features(Xte, ref_clf, xm, xs, device=CPU),
        gbc.augment_features(Xte, pkg_clf, xm, xs, device=CPU))


def test_aug_regressor_matches_reference(jump_data):
    X, y, Xte, _ = jump_data
    ref_labels, ref_acc, ref_samp = _reference_pipeline(X, y, Xte)

    model = _package_model().fit(X, y)

    np.testing.assert_array_equal(ref_labels, model.regime_labels_)
    assert ref_acc == model.classifier_accuracy_
    np.testing.assert_array_equal(ref_samp, model.sample(Xte))
    np.testing.assert_array_equal(ref_samp.mean(0), model.predict(Xte))


# ── Behavior ─────────────────────────────────────────────────────────────────

def test_api_surface_matches_base(jump_data):
    X, y, Xte, _ = jump_data
    m = _package_model().fit(X, y)
    n = len(Xte)
    assert m.n_features_in_ == 2                    # raw, not augmented
    assert m.model_.xdim == 3                       # network sees the extra col
    assert m.predict(Xte).shape == (n,)
    assert m.sample(Xte).shape == (B, n)
    assert m.predict_quantiles(Xte, (0.1, 0.9)).shape == (n, 2)
    lo, hi = m.predict_interval(Xte)
    assert lo.shape == hi.shape == (n,)
    p = m.regime_prob(Xte)
    assert p.shape == (n,) and np.all((p >= 0) & (p <= 1))
    assert len(m.clf_history_) == CLF_EPOCHS


def test_classifier_finds_the_jump(jump_data):
    X, y, _, _ = jump_data
    m = _package_model(clf_epochs=800, epochs=200).fit(X, y)
    assert m.classifier_accuracy_ > 0.9
    # regime probability should track the x0 < 0.5 boundary
    grid = np.column_stack([np.linspace(0, 1, 40), np.full(40, 0.5)])
    p = m.regime_prob(grid)
    assert p[:10].mean() < 0.2 < 0.8 < p[-10:].mean()


def test_aug_helps_on_a_jump_surface(jump_data):
    """Not a guarantee in general, but it should not hurt on jump data."""
    X, y, Xte, yte = jump_data
    kw = dict(epochs=2000, hdim=HDIM, nh=NH, seed=0,
              loss_w=(0.10, 0.20, 0.70), device="cpu", n_samples=101)
    plain = gbc.GBCRegressor(**kw).fit(X, y)
    aug = AugGBCRegressor(clf_epochs=800, clf_hdim=CLF_HDIM,
                          clf_layers=CLF_LAYERS, clf_seed=0, **kw).fit(X, y)
    assert gbc.rmse(yte, aug.predict(Xte)) <= 1.15 * gbc.rmse(
        yte, plain.predict(Xte))


def test_save_load_roundtrip(jump_data, tmp_path):
    X, y, Xte, _ = jump_data
    m = _package_model().fit(X, y)
    p = tmp_path / "aug.pt"
    m.save(p)
    loaded = AugGBCRegressor.load(p, device="cpu")
    np.testing.assert_array_equal(m.predict(Xte), loaded.predict(Xte))
    np.testing.assert_array_equal(m.regime_prob(Xte), loaded.regime_prob(Xte))
    assert loaded.classifier_accuracy_ == m.classifier_accuracy_


def test_wrong_class_load_is_rejected(jump_data, tmp_path):
    X, y, _, _ = jump_data
    p = tmp_path / "aug.pt"
    _package_model().fit(X, y).save(p)
    with pytest.raises(ValueError, match="AugGBCRegressor"):
        gbc.GBCRegressor.load(p, device="cpu")


def test_sklearn_clone_keeps_aug_params():
    from sklearn.base import clone
    m = AugGBCRegressor(clf_epochs=123, n_components=3)
    c = clone(m)
    assert c.clf_epochs == 123 and c.n_components == 3
    assert "clf_epochs=123" in repr(m)


def test_more_components_collapse_to_one_probability(jump_data):
    X, y, Xte, _ = jump_data
    m = _package_model(n_components=3).fit(X, y)
    assert set(np.unique(m.regime_labels_)) <= {0, 1}
    assert m.model_.xdim == 3
    assert m.predict(Xte).shape == (len(Xte),)


def test_bad_n_components(jump_data):
    X, y, _, _ = jump_data
    with pytest.raises(ValueError):
        _package_model(n_components=1).fit(X, y)
