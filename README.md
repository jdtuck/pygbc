# pygbc

**Generative Bayesian Computation surrogates via Implicit Quantile Networks.**

Distributed as `pygbc`, imported as `gbc`.

A packaged, object-oriented version of the reference code for:

> Polson & Sokolov (2026). *Generative Bayesian Computation as a Scalable Alternative to Gaussian Process Surrogates.* Technometrics.
> Source: <https://github.com/VadimSokolov/gbc-surrogate>

Instead of a point prediction with an error bar bolted on, a GBC surrogate learns the **whole conditional distribution** `y | x` as an implicit quantile function. It scales to tens of thousands of points where a GP does not, and it handles heteroskedastic and discontinuous ("jump") response surfaces that a stationary GP smooths away.

## Install

```bash
pip install pygbc
```

From a checkout:

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+, PyTorch 2.0+, NumPy, SciPy, scikit-learn. A GPU is optional and used automatically when available.

## Quick start

```python
import numpy as np
from gbc import GBCRegressor

X = np.random.uniform(0, 1, size=(2000, 3))
y = np.sin(6 * X[:, 0]) + X[:, 1] ** 2 + 0.1 * np.random.randn(2000)

model = GBCRegressor(epochs=3000, seed=0).fit(X, y)

mu       = model.predict(X)                            # (n,) conditional mean
mu, sd   = model.predict(X, return_std=True)           # + predictive sd
lo, hi   = model.predict_interval(X, alpha=0.90)       # 90% interval
q        = model.predict_quantiles(X, [0.1, 0.5, 0.9]) # (n, 3)
draws    = model.sample(X, n_samples=500)              # (500, n)

model.save("surrogate.pt")
model = GBCRegressor.load("surrogate.pt")
```

`sample` returns the quantile function evaluated on an equally spaced grid of levels in `[0.005, 0.995]` — a deterministic stratified sample of the predictive distribution. Feed it straight to CRPS, coverage, or any downstream Monte Carlo.

## Evaluation

```python
from gbc.metrics import summarize, crps_samples, coverage, rmse, ms

draws = model.sample(X_test)
summarize(y_test, draws, alpha=0.90)
# {'rmse': 0.104, 'crps': 0.058, 'coverage': 0.902, 'interval_score': 0.41}
```

| Function | What it gives you |
|---|---|
| `crps_samples(y, samples)` | CRPS from predictive draws (energy-score estimator) |
| `crps_gaussian(y, mu, sigma)` | Closed-form CRPS, for comparing against a GP baseline |
| `coverage(y, samples, alpha)` | Empirical coverage of the central interval |
| `interval_score(y, samples, alpha)` | Winkler score — coverage *and* sharpness |
| `rmse(y_true, y_pred)` | Root mean squared error |
| `ms(values)` | `(mean, standard error)`, the paper's table format |
| `summarize(y, samples)` | All of the above for one set of draws |

`crps_samples` takes an optional `rng=` for a reproducible pairing permutation; the default matches the reference implementation.

## GBC-Aug: jump surfaces

When the response has a discontinuity, a single smooth quantile network has to bridge it. `AugGBCRegressor` gives the network a hint: EM-cluster `y` into regimes, train an MLP classifier `X -> P(regime | X)`, and append that probability as an extra input feature.

```python
from gbc import AugGBCRegressor

model = AugGBCRegressor(
    epochs=8000,
    loss_w=(0.10, 0.20, 0.70),   # quantile-dominant, as in the paper
    clf_epochs=3000,
).fit(X, y)

model.predict(X_test)            # same API — you still pass the raw d columns
model.classifier_accuracy_       # check this: near 0.5 means no usable regime
model.regime_prob(X_test)        # P(regime = 1 | x)
```

Everything from `GBCRegressor` carries over — the augmentation is applied automatically at predict time. Extra parameters: `n_components`, `clf_hdim`, `clf_layers`, `clf_epochs`, `clf_lr`, `clf_seed`.

On a smooth surface this buys nothing and costs an extra network. Use it where the paper does: Phantom and Star in Table 3.

## The classes

### `GBCRegressor(...)`

| Parameter | Default | What it does |
|---|---|---|
| `epochs` | `3000` | Optimizer steps (full-batch) or epochs (with `batch_size`) |
| `hdim` | `256` | Hidden width |
| `nh` | `32` | Cosine basis functions in the quantile embedding |
| `lr` | `1e-3` | Adam LR, cosine-annealed to `lr * 0.01` |
| `weight_decay` | `1e-4` | Adam weight decay |
| `loss_w` | `(0.3, 0.3, 0.4)` | `(mean anchor, monotonicity, quantile)` weights |
| `seed` | `42` | Torch seed set before initialization |
| `device` | `None` | Auto-selects CUDA when available |
| `n_samples` | `500` | Default number of quantile levels at predict time |
| `chunk` | `1000` | Rows per forward pass (memory control) |
| `verbose` | `0` | Print training loss every *n* epochs |

Opt-in extras, all off by default (see below): `batch_size`, `taus_per_step`, `validation_fraction`, `patience`.

### Methods

| Method | Returns |
|---|---|
| `fit(X, y, validation_data=None)` | `self` |
| `predict(X, return_std=False, method="samples")` | `(n,)` mean, or `(mu, sd)` |
| `predict_quantiles(X, quantiles)` | `(n, len(quantiles))` |
| `predict_interval(X, alpha=0.90)` | `(lo, hi)` |
| `sample(X, n_samples=None)` | `(n_samples, n)` |
| `score(X, y)` | negative RMSE (scikit-learn sign convention) |
| `save(path)` / `.load(path)` | — |
| `get_params()` / `set_params(**kw)` | scikit-learn compatible |

`predict(..., method="mean_head")` reads the network's L1 mean head in a single forward pass — cheaper than averaging the quantile grid, but no predictive spread.

Both classes inherit scikit-learn's `BaseEstimator`/`RegressorMixin`, so `clone`, `GridSearchCV`, `cross_val_score` and `Pipeline` work out of the box:

```python
from sklearn.model_selection import GridSearchCV
GridSearchCV(GBCRegressor(epochs=3000),
             {"loss_w": [(0.3, 0.3, 0.4), (0.1, 0.2, 0.7)]}, cv=5).fit(X, y)
```

Note `score` returns **negative RMSE**, not R² — higher is still better, so meta-estimators behave correctly.

## Reproducibility

With default arguments, this package is a **bit-for-bit** port of the paper's code. `tests/test_reference_parity.py` and `tests/test_aug_parity.py` embed the upstream `train_iqn`/`sample_iqn` and the whole `augiqn` pipeline verbatim and assert exact equality of weights, labels, classifier accuracy and predictions — through the functional API *and* through the classes.

The original functions are re-exported unchanged:

```python
from gbc import IQN, train_iqn, sample_iqn
from gbc import cluster_y, train_classifier, augment_features

model, xm, xs, ym, ys = train_iqn(X, y, epochs=3000, seed=42)
samples = sample_iqn(model, X_test, xm, xs, ym, ys, B=500)
```

Reproducing the paper's Table 3 replicate seeding:

```python
AugGBCRegressor(seed=rep * 13 + 7, clf_seed=rep, ...)
```

## Opt-in extras

These change the numerics and are therefore **off by default**, so paper reproductions stay exact.

```python
# Minibatch training instead of full-batch gradient steps
GBCRegressor(batch_size=256, epochs=200)

# Average the loss over several quantile levels per step (variance reduction)
GBCRegressor(taus_per_step=8)

# Early stopping on held-out pinball loss; best weights restored
GBCRegressor(epochs=20000, validation_fraction=0.15, patience=200)

# Or supply your own validation set
model.fit(X, y, validation_data=(X_val, y_val))

# Training curves
model.history_        # per-epoch IQN loss
model.clf_history_    # per-epoch classifier loss (AugGBCRegressor)
```

`train_iqn` takes the same extras as keyword-only arguments, plus `callback(epoch, loss)` and `model=` for warm starts.

## Method

The backbone is an Implicit Quantile Network (Dabney et al., 2018) with a cosine quantile embedding, `phi(tau) = [cos(0·pi·tau), ..., cos((nh-1)·pi·tau)]`, multiplied elementwise into a feature embedding of `x`. Two heads come out: a conditional mean and a quantile at level `tau`.

Training minimizes a three-term loss at a random `tau` each step:

1. **L1 anchor** on the conditional mean — a location regularizer,
2. **Monotonicity** penalty, weighted by `|tau − 0.5|`, enforcing quantile ordering,
3. **Pinball** (quantile) loss.

Adam with cosine annealing; inputs and response standardized internally.

## Module map

```
gbc/
  model.py    GBCRegressor
  aug.py      AugGBCRegressor
  iqn.py      IQN, train_iqn, sample_iqn
  augiqn.py   cluster_y, ClassifierMLP, train_classifier, augment_features
  metrics.py  crps_*, coverage, interval_score, rmse, ms, summarize
```

The upstream repo's `utils.py` (dataset loaders, split helpers, ledger logging) and the per-table experiment scripts are not packaged — they are experiment scaffolding rather than library surface.

## Examples

```bash
python examples/motorcycle.py     # heteroskedastic, GBCRegressor (Table 1 protocol)
python examples/jump_surface.py   # discontinuous, GBCRegressor vs AugGBCRegressor
```

## License

MIT, matching upstream.
