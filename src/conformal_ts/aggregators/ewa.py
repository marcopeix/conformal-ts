"""Exponentially Weighted Average aggregator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import OnlineAggregator


class EWA(OnlineAggregator):
    """
    Exponentially Weighted Average aggregator (Vovk 1990).

    Weights are proportional to ``exp(-eta * cumulative_loss)``. Larger ``eta``
    converges faster to the best expert but is noisier.

    Parameters
    ----------
    n_experts : int
        Number of experts being aggregated.
    n_series : int
        Number of series in the panel.
    horizon : int
        Forecast horizon length.
    eta : float, default 1.0
        Learning rate. Must be positive.
    """

    def __init__(
        self,
        n_experts: int,
        n_series: int,
        horizon: int,
        eta: float = 1.0,
    ) -> None:
        super().__init__(n_experts, n_series, horizon)
        if eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        self.eta = float(eta)

    def weights(self) -> NDArray[np.floating]:
        """
        Return current softmax weights over experts.

        Returns
        -------
        NDArray, shape (n_experts, n_series, horizon)
            Numerically stable softmax along axis 0. Sums to 1 along axis 0.
        """
        neg_loss = -self.eta * self.cumulative_losses_
        neg_loss_max = neg_loss.max(axis=0, keepdims=True)
        exp_w = np.exp(neg_loss - neg_loss_max)
        denom = exp_w.sum(axis=0, keepdims=True)
        return exp_w / denom
