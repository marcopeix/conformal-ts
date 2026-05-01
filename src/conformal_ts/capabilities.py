"""
Capability mixins for forecaster adapters.

Each mixin declares one optional capability via an abstract method. Adapters
inherit from the mixins they implement; conformal methods declare which mixins
they require via `REQUIRED_CAPABILITIES`.

Inheriting from a mixin without overriding its method is a programming error
caught at adapter import time (the abstract method makes the class
uninstantiable).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .base import Forecast, Series


class SupportsRefit(ABC):
    """Adapter can retrain its underlying model on new data."""

    @abstractmethod
    def refit(self, history: Series) -> None:
        """
        Refit the underlying model on `history`.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        """
        ...


class SupportsQuantiles(ABC):
    """Adapter can produce quantile forecasts directly (needed for CQR)."""

    @abstractmethod
    def predict_quantiles(
        self,
        history: Series,
        quantiles: NDArray[np.floating],
    ) -> Forecast:
        """
        Predict requested quantiles.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        quantiles : NDArray, shape (n_quantiles,)
            Values in (0, 1).

        Returns
        -------
        Forecast, shape (n_series, n_quantiles, horizon)
            Quantile forecasts. Note the middle axis indexes quantiles, not samples.
        """
        ...


class SupportsBootstrap(ABC):
    """Adapter can produce bootstrap ensemble forecasts (needed for EnbPI)."""

    @abstractmethod
    def bootstrap_predict(
        self,
        history: Series,
        n_bootstraps: int,
        seed: int | None = None,
    ) -> Forecast:
        """
        Produce `n_bootstraps` predictions from bootstrap resamples of training data.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        n_bootstraps : int
        seed : int or None

        Returns
        -------
        Forecast, shape (n_series, n_bootstraps, horizon)
        """
        ...
