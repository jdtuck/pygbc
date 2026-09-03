"""
mvBayes-compatible wrapper for gbc.GBCRegressor.

This module is intended to live inside the gbc package.

Behavior
--------
- Fits a GBCRegressor model to a univariate response y
- Uses GBCRegressor.sample(...) as the posterior-like predictive draws
- Returns predictions in shape (n_samples, n_obs), compatible with mvBayes
- Sets .samples.residSD = 0 for all samples, so no extra residual noise is added
  downstream by mvBayes-based tooling unless explicitly desired elsewhere

Intended use
------------
Pass `gbc_mvbayes_model` as the `bayesModel` argument to mvBayes(...).
"""

import numpy as np

from .model import GBCRegressor


class _MvBayesGBCSamples:
    """Simple container for mvBayes-compatible posterior sample attributes."""
    pass


class MvBayesGBCWrapper:
    """
    mvBayes-compatible wrapper around GBCRegressor for univariate coefficient models.

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix.
    y : np.ndarray
        Univariate response.
    nSamples : int, default=1000
        Number of predictive draws returned by GBCRegressor.sample(...).
    **kwargs
        Additional keyword arguments passed directly to GBCRegressor(...).
    """

    def __init__(self, X, y, nSamples=1000, **kwargs):
        y = np.asarray(y)
        if y.ndim != 1:
            y = np.squeeze(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array or coercible to 1D.")

        if not isinstance(nSamples, int) or nSamples <= 0:
            raise ValueError("nSamples must be a positive integer.")

        self.nSamples = nSamples

        self.model = GBCRegressor(**kwargs)
        self.model.fit(X, y)

        # mvBayes-compatible samples object
        self.samples = _MvBayesGBCSamples()

        # By design, use GBC sample() draws as the full predictive uncertainty
        # and do not add any extra residual error via mvBayes downstream logic.
        self.samples.residSD = np.zeros(self.nSamples)

    def predict(self, Xtest, idxSamples=None):
        """
        Return predictive draws from the fitted GBC model.

        Parameters
        ----------
        Xtest : array-like
            Test predictors.
        idxSamples : None or array-like of int
            If None, return self.nSamples draws.
            If provided, generate only len(idxSamples) draws and return them.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples_selected, n_obs), compatible with mvBayes.
        """
        if idxSamples is None:
            n_draws = self.nSamples
        else:
            idxSamples = np.asarray(idxSamples)
            if idxSamples.ndim == 0:
                idxSamples = idxSamples.reshape(1)
            n_draws = len(idxSamples)

        draws = self.model.sample(Xtest, n_samples=n_draws)
        draws = np.asarray(draws)

        # GBCRegressor.sample is documented to return shape (n_samples, n_obs)
        if draws.ndim != 2:
            raise ValueError(
                f"Expected predictive draws with 2 dimensions, got shape {draws.shape}."
            )

        return draws


def gbc4mvBayes(X, y, **kwargs):
    """
    Factory function for use as mvBayes(..., bayesModel=...).

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix.
    y : np.ndarray
        Univariate response.
    **kwargs
        Additional keyword arguments passed to MvBayesGBCWrapper.
        These include:
          - nSamples
          - any valid GBCRegressor constructor kwargs

    Returns
    -------
    MvBayesGBCWrapper
        mvBayes-compatible fitted model object.
    """
    return MvBayesGBCWrapper(X, y, **kwargs)
