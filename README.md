[![Pipeline Status](https://github.com/jdtuck/pygbc/actions/workflows/Build.yml/badge.svg)](https://github.com/jdtuck/pygbc/actions/workflows/Build.yml)

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

Everything from `GBCRegressor` carries over — the augmentation is applied automatically at predict time. Extra parameters: `n_components`, `clf_hdim`, `clf_layers`, `clf_epochs`, `clf_lr`, `clf_seed`, `init_params`, `cluster_threads`.

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
| `device` | `None` | Auto-selects CUDA when available, else CPU (never MPS — see below) |
| `n_samples` | `500` | Default number of quantile levels at predict time |
| `chunk` | `1000` | Rows per forward pass; `"auto"` tunes for speed |
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
  model.py       GBCRegressor
  aug.py         AugGBCRegressor
  iqn.py         IQN, train_iqn, sample_iqn
  augiqn.py      cluster_y, ClassifierMLP, train_classifier, augment_features
  metrics.py     crps_*, coverage, interval_score, rmse, ms, summarize
  diagnostics.py diagnose()  —  also `python -m gbc`
```

The upstream repo's `utils.py` (dataset loaders, split helpers, ledger logging) and the per-table experiment scripts are not packaged — they are experiment scaffolding rather than library surface.

## Making it faster

Measured on 2 CPU cores, `d=4`, predicting on all `n` rows. Your absolute times will be lower; the ratios are what transfer.

### Prediction

| Change | n=10,000 | n=20,000 | Cost |
|---|---|---|---|
| baseline (`B=500`, `chunk=1000`) | 5.9 s | 11.3 s | — |
| `chunk="auto"` | 4.5x → **1.13x** | **1.12x** | ~1e-7 relative |
| `n_samples=100` + `chunk="auto"` | **4.9x** | **5.6x** | none measurable |
| `predict(method="mean_head")` | **~600x** | **~600x** | ~1% higher RMSE |

**Cutting `n_samples` is the big lever and it is nearly free.** Sampling cost is exactly linear in the number of quantile levels, and RMSE and coverage are flat from `B≈100`:

```
B=500: RMSE=0.1012  Cov90=0.910
B=100: RMSE=0.1012  Cov90=0.910
```

One caveat: the CRPS *estimator* is biased at small `B` (the energy score's pairing term), so keep `B >= 200` when you are reporting CRPS, even though the model itself is unchanged.

**For point predictions only, use the mean head.** `predict(method="mean_head")` is one forward pass instead of `B`, and reads the network's L1 mean output directly rather than averaging the quantile grid. Its accuracy converges as training proceeds — on a 4000-point fit, grid-vs-head disagreement fell from 0.068 at 300 epochs to 0.008 at 3000. At convergence it cost ~1% RMSE for a ~600x speedup. Use the grid when you need intervals or CRPS.

**`chunk="auto"`** targets ~1M hidden activations per forward pass (about 4000 rows at `hdim=256`). It is worth ~12%, and it is opt-in because chunk size is *not* numerically neutral: splitting the rows differently sends the matmuls down a different BLAS blocking path and moves results by ~1e-7 relative. Chunks at or above `n` stay bit-identical to the default. The optimum is cache-dependent — benchmark on your own hardware before trusting it.

### Fitting

| Option | Effect |
|---|---|
| `foreach=True` | ~1.08x at small/medium `n`, **weights identical** |
| `track_history=False` | skips a device sync per epoch (matters on GPU, not CPU) |
| `validation_fraction` + `patience` | the real win — stop when it has converged |
| `batch_size=...` | **slower**, do not use it for speed |

Two negative results worth recording, because both look like obvious wins and are not:

- **Minibatching does not help.** At `n=20,000`, `batch_size=1024` matched full-batch and `batch_size=256` was **3x slower** — same work per epoch, split into more and smaller kernel launches. `batch_size` is for changing the optimization, not for speed.
- **`torch.compile` is much worse here**, ~50x slower at small `n`. `loss_fn` draws `tau` with `.item()`, which is a graph break, and the resulting python float is then baked into the traced graph so Dynamo recompiles every single step until it hits the recompile limit. Keeping `tau` as a 0-d tensor would fix both that and the per-step device sync — but the tau arithmetic then runs in float32 instead of python float64 and the loss stops being bitwise equal to the reference, so this package does not do it. If you ever relax the parity requirement, that is the first thing to change.

Where the time actually goes, profiled: at `n=133` a step is ~2.4 ms split roughly evenly between backward (32%), the Adam step (29%) and the forward (26%) — all dispatch overhead, no real math, which is why `foreach` helps there and nothing else does. At `n>=10,000` it is BLAS-bound in `fc1`, so only fewer epochs, more threads, or a GPU will move it.

Also check `torch.set_num_threads()`: PyTorch does not always pick well, and on this 2-core box the difference between 1 and 2 threads was 1.6x on sampling.

## Is the GPU being used?

```python
model.device_          # the device the fitted network actually lives on
```

```bash
python -m gbc          # what this environment can reach, before you fit
```

`python -m gbc` now prints a `Compute` block:

```
Compute
  CUDA available   False
  MPS (Apple GPU)  built=True available=True
  torch threads    8
  device=None ->   cpu   <-- GPU NOT in use
```

`device=None` (the default) selects **CUDA if available, else CPU** — it does not try Apple's MPS backend, so on Apple silicon it silently resolves to CPU. Nothing warns you; the run is just slower. `diagnose()` flags that case explicitly when it sees an unused Apple GPU.

To force a device:

```python
GBCRegressor(device="cuda")   # or "cuda:1"
GBCRegressor(device="mps")    # Apple GPU — opt-in, see caveats below
GBCRegressor(device="cpu")
```

On CUDA, `nvidia-smi -l 1` during a fit is the external confirmation: the python process should be listed with non-zero utilization.

**Whether the GPU is worth it depends on `n`.** Training is full-batch, so each epoch is one forward/backward over the entire dataset. At `n` in the hundreds (motorcycle, Table 1) the kernel-launch overhead of 3,000 sequential tiny steps dominates and CPU usually wins. At `n` in the tens of thousands (Table 3's Michalewicz is 90,000 x 4) the per-epoch matmuls are large enough that the GPU pays off — which is why the paper recommends one for Table 3.

MPS caveats: results will not match CPU bit-for-bit (different kernels and reduction orders), so the reproducibility guarantees above hold per-device, not across devices. Op coverage on MPS is also still incomplete. Treat `device="mps"` as an experiment to benchmark, not a default.

## Troubleshooting: segfault in scikit-learn's k-means

A crash like this — most often on macOS with conda, and only when `AugGBCRegressor` or `cluster_y` runs:

```
Fatal Python error: Segmentation fault
  File ".../sklearn/cluster/_kmeans.py", line 756 in _kmeans_single_lloyd
  File ".../sklearn/mixture/_base.py", line 125 in _initialize_parameters
```

is not a bug in this package. PyTorch and a conda/MKL scikit-learn load **two different OpenMP runtimes** (`libomp` and `libiomp5`) into one process, and scikit-learn's k-means — which `GaussianMixture` uses to initialize — dies inside its parallel region. It is a long-standing interaction between the two projects ([pytorch#132372](https://github.com/pytorch/pytorch/issues/132372), [scikit-learn#21302](https://github.com/scikit-learn/scikit-learn/issues/21302), [scikit-learn#23574](https://github.com/scikit-learn/scikit-learn/issues/23574)).

Confirm it:

```bash
python -m gbc
```

That prints your versions and every OpenMP runtime loaded. Two different ones is the smoking gun.

Fixes, cheapest first:

1. **Already the default.** `cluster_y(threads=1)` / `AugGBCRegressor(cluster_threads=1)` pin the fit to one thread, which avoids the parallel region. If you are on an older copy of this package, upgrade.
2. **Pin the whole process** before Python starts — this also covers the test suite:
   ```bash
   OMP_NUM_THREADS=1 pytest
   ```
   (`tests/conftest.py` sets this for you when running pytest from the repo.)
3. **Skip k-means entirely**:
   ```python
   AugGBCRegressor(init_params="random_from_data")
   ```
   `"random_from_data"`, `"random"` and `"k-means++"` all initialize the mixture without the crashing code path. On well-separated regimes they find the same clusters; on the package's jump-surface fixtures all four agree to >95%.
4. **Fix the environment** so only one OpenMP runtime is present — `conda install nomkl`, or install `numpy`/`scipy`/`scikit-learn` from PyPI wheels rather than the MKL conda builds.

Do **not** reach for `KMP_DUPLICATE_LIB_OK=TRUE`. It suppresses the runtime's own guard rather than the conflict, and can turn a clean abort into silent corruption.

### A note on determinism

scikit-learn's k-means changes its floating-point summation order with the thread count, so an unpinned `GaussianMixture` can produce bitwise-different cluster means on different machines — true of the upstream script too. Pinning to one thread by default makes `cluster_y` reproducible. Pass `threads=None` for the unpinned upstream behavior.

## Examples

```bash
python examples/motorcycle.py     # heteroskedastic, GBCRegressor (Table 1 protocol)
python examples/jump_surface.py   # discontinuous, GBCRegressor vs AugGBCRegressor
```

## License

MIT, matching upstream.
