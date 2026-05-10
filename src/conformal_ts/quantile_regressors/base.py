"""Base abstraction for quantile regressors used by SPCI."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class QuantileRegressor(ABC):
    """
    Quantile regression model used by SPCI.

    Trained on autoregressive residual data: given the last ``w`` residuals,
    predict the conditional quantile of the next residual.

    Subclasses can wrap any quantile-regression library (QRF, LightGBM
    quantile, neural quantile networks, transformers, etc.). The interface
    is intentionally minimal: a single ``fit`` followed by repeated
    ``predict_quantile(X, q)`` calls.
    """

    @abstractmethod
    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        """
        Fit the regressor.

        Parameters
        ----------
        X : NDArray, shape (n_samples, n_features)
            Training features. For SPCI, ``n_features`` is the window size
            ``w`` (lag count).
        y : NDArray, shape (n_samples,)
            Training targets (the next residual).
        """
        ...

    @abstractmethod
    def predict_quantile(
        self,
        X: NDArray[np.floating],
        q: float,
    ) -> NDArray[np.floating]:
        """
        Predict the ``q``-quantile of the conditional distribution.

        Parameters
        ----------
        X : NDArray, shape (n_samples, n_features)
        q : float in (0, 1)
            Quantile level.

        Returns
        -------
        NDArray, shape (n_samples,)
        """
        ...
