"""GBC surrogate on the MASS::mcycle benchmark (n=133, heteroskedastic).

Mirrors the Table 1 protocol from Polson & Sokolov (2026): 80/20 split,
CRPS / RMSE / 90% coverage on the held-out points.

    python examples/motorcycle.py
"""

import numpy as np

from gbc import GBCRegressor

TIMES = np.array([
    2.4, 2.6, 3.2, 3.6, 4, 6.2, 6.6, 6.8, 7.8, 8.2, 8.8, 8.8, 9.6, 10, 10.2,
    10.6, 11, 11.4, 13.2, 13.6, 13.8, 14.6, 14.6, 14.6, 14.6, 14.6, 14.6,
    14.8, 15.4, 15.4, 15.4, 15.4, 15.6, 15.6, 15.8, 15.8, 16, 16, 16.2, 16.2,
    16.2, 16.4, 16.4, 16.6, 16.8, 16.8, 16.8, 17.6, 17.6, 17.6, 17.6, 17.8,
    17.8, 18.6, 18.6, 19.2, 19.4, 19.4, 19.6, 20.2, 20.4, 21.2, 21.4, 21.8,
    22, 23.2, 23.4, 24, 24.2, 24.2, 24.6, 25, 25, 25.4, 25.4, 25.6, 26, 26.2,
    26.2, 26.4, 27, 27.2, 27.2, 27.2, 27.6, 28.2, 28.4, 28.4, 28.6, 29.4,
    30.2, 31, 31.2, 32, 32, 32.8, 33.4, 33.8, 34.4, 34.8, 35.2, 35.2, 35.4,
    35.6, 35.6, 36.2, 36.2, 38, 38, 39.2, 39.4, 40, 40.4, 41.6, 41.6, 42.4,
    42.8, 42.8, 43, 44, 44.4, 45, 46.6, 47.8, 47.8, 48.8, 50.6, 52, 53.2, 55,
    55, 55.4, 57.6])
ACCEL = np.array([
    0, -1.3, -2.7, 0, -2.7, -2.7, -2.7, -1.3, -2.7, -2.7, -1.3, -2.7, -2.7,
    -2.7, -5.4, -2.7, -5.4, 0, -2.7, -2.7, 0, -13.3, -5.4, -5.4, -9.3, -16,
    -22.8, -2.7, -22.8, -32.1, -53.5, -54.9, -40.2, -21.5, -21.5, -50.8,
    -42.9, -26.8, -21.5, -50.8, -61.7, -5.4, -80.4, -59, -71, -91.1, -77.7,
    -37.5, -85.6, -123.1, -101.9, -99.1, -104.4, -112.5, -50.8, -123.1,
    -85.6, -72.3, -127.2, -123.1, -117.9, -134, -101.9, -108.4, -123.1,
    -123.1, -128.5, -112.5, -95.1, -81.8, -53.5, -64.4, -57.6, -72.3, -44.3,
    -26.8, -5.4, -107.1, -21.5, -65.6, -16, -45.6, -24.2, 9.5, 4, 12, -21.5,
    37.5, 46.9, -17.4, 36.2, 75, 8.1, 54.9, 48.2, 46.9, 16, 45.6, 1.3, 75,
    -16, -54.9, 69.6, 34.8, 32.1, -37.5, 22.8, 46.9, 10.7, 5.4, -1.3, -21.5,
    -13.3, 30.8, -10.7, 29.4, 0, -10.7, 14.7, -1.3, 0, 10.7, 10.7, -26.8,
    -14.7, -13.3, 0, 10.7, -14.7, -2.7, 10.7, -2.7, 10.7])


def crps_samples(y, samples):
    """Energy-score estimator: CRPS(F, y) = E|Y-y| - 0.5 E|Y-Y'|."""
    term1 = np.mean(np.abs(samples - y[None, :]), axis=0)
    idx = np.random.permutation(samples.shape[0])
    term2 = 0.5 * np.mean(np.abs(samples - samples[idx, :]), axis=0)
    return float(np.mean(term1 - term2))


def main() -> None:
    n = min(len(TIMES), len(ACCEL))
    X, y = TIMES[:n].reshape(-1, 1), ACCEL[:n]

    rng = np.random.default_rng(300)
    te = rng.choice(n, size=int(0.2 * n), replace=False)
    tr = np.setdiff1d(np.arange(n), te)

    model = GBCRegressor(epochs=3000, seed=0, verbose=500).fit(X[tr], y[tr])

    draws = model.sample(X[te], n_samples=500)
    mu = draws.mean(0)
    lo, hi = model.predict_interval(X[te], alpha=0.90)

    print(f"\nRMSE      {np.sqrt(np.mean((mu - y[te]) ** 2)):8.3f}")
    print(f"CRPS      {crps_samples(y[te], draws):8.3f}")
    print(f"Coverage  {np.mean((y[te] >= lo) & (y[te] <= hi)):8.3f}  (nominal 0.90)")


if __name__ == "__main__":
    main()
