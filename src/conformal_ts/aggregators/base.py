"""Base abstraction for online expert aggregators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class OnlineAggregator(ABC):
    """
    Online expert aggregator for AgACI and similar methods.

    Maintains per-expert cumulative losses and exposes normalized weights.
    Subclasses differ in how they convert losses to weights.

    Parameters
    ----------
    n_experts : int
        Number of experts being aggregated.
    n_series : int
        Number of series in the panel.
    horizon : int
        Forecast horizon length.
    """

    def __init__(self, n_experts: int, n_series: int, horizon: int) -> None:
        if n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {n_experts}")
        if n_series < 1:
            raise ValueError(f"n_series must be >= 1, got {n_series}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.n_experts = n_experts
        self.n_series = n_series
        self.horizon = horizon
        self.cumulative_losses_: NDArray[np.floating] = np.zeros(
            (n_experts, n_series, horizon), dtype=np.float64
        )

    @abstractmethod
    def weights(self) -> NDArray[np.floating]:
        """
        Return current normalized weights.

        Returns
        -------
        NDArray, shape (n_experts, n_series, horizon)
            Sum along axis 0 equals 1 for every (series, horizon) cell.
        """
        ...

    def update(self, losses: NDArray[np.floating]) -> None:
        """
        Add per-expert losses to cumulative losses.

        Parameters
        ----------
        losses : NDArray, shape (n_experts, n_series, horizon)

        Raises
        ------
        ValueError
            If ``losses`` does not have the expected shape.
        """
        expected = (self.n_experts, self.n_series, self.horizon)
        if losses.shape != expected:
            raise ValueError(f"losses must have shape {expected}, got {losses.shape}")
        self.cumulative_losses_ = self.cumulative_losses_ + losses
