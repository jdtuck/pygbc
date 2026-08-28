"""GBC vs GBC-Aug on a discontinuous response surface.

A 2-D surface with a jump at x0 = 0.5 — the setting AugGBCRegressor is for.
Reports the paper's metric set for both models.

    python examples/jump_surface.py
"""

import numpy as np

from gbc import AugGBCRegressor, GBCRegressor
from gbc.metrics import summarize


def surface(X, rng=None):
    f = np.where(X[:, 0] < 0.5, 0.2 * X[:, 1], 1.0 + 0.2 * X[:, 1])
    if rng is not None:
        f = f + 0.05 * rng.standard_normal(len(X))
    return f


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(1500, 2))
    y = surface(X, rng)
    Xte = rng.uniform(0, 1, size=(400, 2))
    yte = surface(Xte, rng)

    common = dict(epochs=4000, loss_w=(0.10, 0.20, 0.70), seed=0,
                  n_samples=500)

    print("Fitting GBC ...")
    plain = GBCRegressor(**common).fit(X, y)

    print("Fitting GBC-Aug ...")
    aug = AugGBCRegressor(clf_epochs=2000, clf_seed=0, **common).fit(X, y)
    print(f"  regime classifier accuracy: {aug.classifier_accuracy_:.4f}")

    rows = [("GBC", summarize(yte, plain.sample(Xte), rng=0)),
            ("GBC-Aug", summarize(yte, aug.sample(Xte), rng=0))]

    print(f"\n{'model':<10}{'RMSE':>10}{'CRPS':>10}{'Cov90':>10}{'IntScore':>11}")
    for name, m in rows:
        print(f"{name:<10}{m['rmse']:>10.4f}{m['crps']:>10.4f}"
              f"{m['coverage']:>10.3f}{m['interval_score']:>11.4f}")


if __name__ == "__main__":
    main()
