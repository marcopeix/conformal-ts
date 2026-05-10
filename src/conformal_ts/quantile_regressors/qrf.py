"""Quantile Random Forest regressor — the default for SPCI."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import QuantileRegressor

try:
    from quantile_forest import RandomForestQuantileRegressor  # type: ignore[import-untyped]
except ImportError as _err:  # pragma: no cover - import guard
    raise ImportError(
        "QRFQuantileRegressor requires the 'quantile-forest' package. "
        "Install it with: pip install conformal-ts[spci]"
    ) from _err


class QRFQuantileRegressor(QuantileRegressor):
    """
    Quantile Random Forest regressor (Meinshausen 2006).

    Default quantile estimator used in SPCI per Xu & Xie (2023). Wraps
    ``quantile_forest.RandomForestQuantileRegressor``.

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees in the forest.
    min_samples_leaf : int, default 5
        Minimum number of samples per leaf.
    random_state : int or None, default None
        Random seed forwarded to the underlying estimator.
    **kwargs : Any
        Additional keyword arguments forwarded to
        ``RandomForestQuantileRegressor``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_leaf: int = 5,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.kwargs = kwargs
        self._model: RandomForestQuantileRegressor | None = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        """
        Fit the underlying quantile forest.

        Parameters
        ----------
        X : NDArray, shape (n_samples, n_features)
        y : NDArray, shape (n_samples,)
        """
        self._model = RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            **self.kwargs,
        )
        self._model.fit(X, y)

    def predict_quantile(
        self,
        X: NDArray[np.floating],
        q: float,
    ) -> NDArray[np.floating]:
        """
        Predict the ``q``-quantile from the fitted forest.

        Parameters
        ----------
        X : NDArray, shape (n_samples, n_features)
        q : float in (0, 1)

        Returns
        -------
        NDArray, shape (n_samples,)

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._model is None:
            raise RuntimeError("QRFQuantileRegressor must be fit before predict_quantile.")
        # quantile-forest accepts a list of quantiles and returns
        # shape (n_samples, n_quantiles).
        pred = self._model.predict(X, quantiles=[q])
        return np.asarray(pred, dtype=np.float64).flatten()
