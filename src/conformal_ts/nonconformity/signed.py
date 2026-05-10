"""Signed residual nonconformity score."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..base import Forecast, Interval, ScoreFunction


class SignedResidual(ScoreFunction):
    """
    Signed-residual nonconformity score: ``ε = y - ŷ``.

    Unlike :class:`AbsoluteResidual`, this score preserves direction. Positive
    scores mean truth was above prediction; negative means below. SPCI uses
    signed residuals because its asymmetric intervals depend on the
    directional distribution.

    Intervals built from signed scores are *asymmetric*: the lower and upper
    offsets from the point forecast are independent and need not be opposite.

    Notes
    -----
    The :meth:`invert` method takes a 3-D score-threshold array whose last
    axis carries the ``(lower_offset, upper_offset)`` pair, and applies it
    to the point forecast by per-cell broadcasting along the sample axis.
    """

    def score(self, prediction: Forecast, truth: Forecast) -> NDArray[np.floating]:
        """
        Compute signed residual scores.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, n_samples, horizon)
        truth : Forecast, shape (n_series, n_samples, horizon)

        Returns
        -------
        scores : NDArray, shape (n_series, n_samples, horizon)
            ``truth - prediction``. Sign preserved.
        """
        return np.asarray(truth, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)

    def invert(
        self,
        prediction: Forecast,
        score_threshold: NDArray[np.floating],
    ) -> Interval:
        """
        Produce an asymmetric interval from per-cell ``(lower, upper)`` offsets.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, n_samples, horizon)
        score_threshold : NDArray, shape (n_series, horizon, 2)
            Last axis is ``(lower_offset, upper_offset)``. The interval at
            ``(series=s, sample=k, horizon=h)`` is
            ``[prediction[s, k, h] + lower_offset[s, h],
               prediction[s, k, h] + upper_offset[s, h]]``.
            ``lower_offset`` is typically negative for sensible intervals.

        Returns
        -------
        Interval, shape (n_series, n_samples, horizon, 2)
            Last axis is ``(lower, upper)``.
        """
        pred = np.asarray(prediction, dtype=np.float64)
        thr = np.asarray(score_threshold, dtype=np.float64)
        if thr.ndim != 3 or thr.shape[-1] != 2:
            raise ValueError(
                f"score_threshold must have shape (n_series, horizon, 2), got {thr.shape}."
            )

        # thr[..., 0]: (n_series, horizon) -> (n_series, 1, horizon)
        lower_offset = thr[:, np.newaxis, :, 0]
        upper_offset = thr[:, np.newaxis, :, 1]

        lower = pred + lower_offset
        upper = pred + upper_offset
        return np.stack([lower, upper], axis=-1)
