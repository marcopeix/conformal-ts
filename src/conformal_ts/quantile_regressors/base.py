"""Abstract base class for quantile regressors used by SPCI."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class QuantileRegressor(ABC):
    """
    Quantile regression model used by SPCI.

    Trained on autoregressive residual data: given the last ``w`` residuals,
    predict the quantile distribution of the next residual. Subclasses can
    wrap any quantile-regression library (QRF, LightGBM quantile, neural
    quantile networks, etc.).
    """

    @abstractmethod
    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> None:
        """
        Fit the regressor.

        Parameters
        ----------
        X : NDArray, shape (n_samples, w)
            Training features. Each row is a window of ``w`` past residuals.
        y : NDArray, shape (n_samples,)
            Training targets (the next residual after each window).
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
        X : NDArray, shape (n_samples, w)
        q : float in (0, 1)

        Returns
        -------
        NDArray, shape (n_samples,)
        """
        ...
