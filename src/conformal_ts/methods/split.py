"""Split conformal prediction for time series."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..base import (
    CalibrationError,
    CalibrationResult,
    ConformalMethod,
    Forecast,
    Interval,
    PredictionResult,
    ScoreFunction,
    Series,
)
from ..nonconformity.absolute import AbsoluteResidual


class SplitConformal(ConformalMethod):
    """
    Classic split conformal prediction adapted for panel time series.

    Algorithm
    ---------
    1. **calibrate**: for each calibration history, produce a forecast and
       compute nonconformity scores against the ground truth. Store the
       empirical ``ceil((1 - alpha)(1 + n)) / n`` quantile of scores **per
       (series, horizon) pair**.
    2. **predict**: produce a point forecast and apply
       ``score_fn.invert`` using the stored quantile to get the interval.

    Parameters
    ----------
    forecaster : ForecasterAdapter
        Any adapter — no special capabilities required.
    alpha : float
        Miscoverage level in (0, 1). The target coverage is ``1 - alpha``.
    score : ScoreFunction or None
        Nonconformity score. Defaults to :class:`AbsoluteResidual`.
    """

    REQUIRED_CAPABILITIES: tuple[type, ...] = ()
    IS_ONLINE: bool = False

    def _default_score(self) -> ScoreFunction:
        return AbsoluteResidual()

    def calibrate(
        self,
        histories: list[Series],
        truths: Forecast,
    ) -> CalibrationResult:
        """
        Fit the conformal correction on a calibration set.

        Parameters
        ----------
        histories : list of Series, each shape (n_series, T)
            History windows available at each calibration timestep.
        truths : Forecast, shape (n_series, len(histories), horizon)
            Ground truth values for each calibration history and horizon.

        Returns
        -------
        CalibrationResult

        Raises
        ------
        CalibrationError
            If the number of calibration samples is too small for the
            requested ``alpha``. At least ``ceil(1 / alpha)`` samples are
            required for the quantile to be well-defined.
        """
        n_cal = len(histories)
        min_samples = math.ceil(1.0 / self.alpha)
        if n_cal < min_samples:
            raise CalibrationError(
                f"Need at least ceil(1/alpha) = {min_samples} calibration "
                f"samples for alpha={self.alpha}, got {n_cal}."
            )

        # Produce forecasts for every calibration history: (n_series, n_cal, horizon)
        predictions = self.forecaster.predict_batch(histories)

        # Let the score function fit any internal parameters (no-op for AbsoluteResidual)
        self.score_fn.fit(predictions, truths)

        # Nonconformity scores: (n_series, n_cal, horizon)
        scores = self.score_fn.score(predictions, truths)

        # Empirical quantile level with finite-sample correction
        quantile_level = min((math.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal), 1.0)

        # Quantile per (series, horizon): (n_series, horizon)
        self._score_quantile: NDArray[np.floating] = np.quantile(scores, quantile_level, axis=1)

        self._is_calibrated = True

        return CalibrationResult(
            n_calibration_samples=n_cal,
            score_quantile=self._score_quantile,
            diagnostics={"quantile_level": quantile_level},
        )

    def predict(self, history: Series) -> PredictionResult:
        """
        Produce a calibrated point forecast and prediction interval.

        Parameters
        ----------
        history : Series, shape (n_series, T)

        Returns
        -------
        PredictionResult
            ``point`` has shape ``(n_series, 1, horizon)``.
            ``interval`` has shape ``(n_series, 1, horizon, 2)``.

        Raises
        ------
        CalibrationError
            If called before :meth:`calibrate`.
        """
        if not self._is_calibrated:
            raise CalibrationError("predict() called before calibrate(). Call calibrate() first.")

        point: Forecast = self.forecaster.predict(history)
        interval: Interval = self.score_fn.invert(point, self._score_quantile)

        return PredictionResult(
            point=point,
            interval=interval,
            alpha=self.alpha,
        )
