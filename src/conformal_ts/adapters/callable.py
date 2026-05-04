"""Adapter that wraps a plain callable into a ForecasterAdapter."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ..base import Forecast, ForecasterAdapter, Series


class CallableAdapter(ForecasterAdapter):
    """
    Wrap a user-supplied function into a :class:`ForecasterAdapter`.

    The callable must accept a history array and return a point forecast.
    No capability mixins are provided — the callable has no concept of
    refit, quantiles, or bootstrap.

    Parameters
    ----------
    predict_fn : callable
        ``fn(history) -> forecast`` where *history* has shape
        ``(n_series, T)`` and the return value has shape
        ``(n_series, horizon)``.
    horizon : int
        Number of steps to forecast.
    n_series : int
        Number of series in the panel.
    """

    def __init__(
        self,
        predict_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        horizon: int,
        n_series: int,
    ) -> None:
        super().__init__(horizon=horizon, n_series=n_series)

        if not callable(predict_fn):
            raise ValueError("predict_fn must be callable")
        self._predict_fn = predict_fn

    def predict(self, history: Series) -> Forecast:
        """
        Produce a point forecast by calling the wrapped function.

        Parameters
        ----------
        history : Series, shape (n_series, T)

        Returns
        -------
        Forecast, shape (n_series, 1, horizon)

        Raises
        ------
        ValueError
            If ``history`` is not 2-D, has the wrong leading axis, contains
            NaN, or if the callable returns an unexpected shape.
        """
        history = self._validate_history(history)

        raw = self._predict_fn(history)
        raw = np.asarray(raw, dtype=np.float64)

        if raw.shape != (self.n_series, self.horizon):
            raise ValueError(
                f"predict_fn must return shape ({self.n_series}, {self.horizon}), got {raw.shape}"
            )

        # (n_series, horizon) -> (n_series, 1, horizon)
        return raw[:, np.newaxis, :]
