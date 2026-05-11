"""Quantile Random Forest implementation of :class:`QuantileRegressor`."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import QuantileRegressor

try:
    from quantile_forest import (  # type: ignore[import-untyped]
        RandomForestQuantileRegressor,
    )

    _IMPORT_ERROR: ImportError | None = None
except ImportError as _err:
    RandomForestQuantileRegressor = None  # type: ignore[assignment,misc]
    _IMPORT_ERROR = _err


class QRFQuantileRegressor(QuantileRegressor):
    """
    Quantile Random Forest regressor (Meinshausen 2006).

    Default quantile estimator for SPCI per Xu & Xie (2023). Wraps the
    ``quantile-forest`` package, exposing a clean two-method interface.

    Parameters
    ----------
    n_estimators : int, default 100
    min_samples_leaf : int, default 5
    random_state : int or None, default None
    **kwargs : forwarded to ``RandomForestQuantileRegressor``.

    Notes
    -----
    The ``quantile-forest`` package is an optional dependency. The import is
    deferred to construction time so the surrounding ``quantile_regressors``
    package can be imported without it (e.g. by users who supply a custom
    :class:`QuantileRegressor` to SPCI). Instantiating this class without
    ``quantile-forest`` installed raises :class:`ImportError`.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_leaf: int = 5,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "QRFQuantileRegressor requires the 'quantile-forest' package. "
                "Install it with: pip install conformal-ts[spci]"
            ) from _IMPORT_ERROR

        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.kwargs = kwargs
        self._model: Any | None = None

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> None:
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
        if self._model is None:
            raise RuntimeError("QRFQuantileRegressor must be fit before predict_quantile.")
        pred = self._model.predict(X, quantiles=[q])
        return np.asarray(pred, dtype=np.float64).reshape(-1)
