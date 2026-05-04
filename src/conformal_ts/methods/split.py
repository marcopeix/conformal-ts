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
    UnsupportedCapability,
)
from ..capabilities import SupportsCrossValidation
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
        histories: list[Series] | None = None,
        truths: Forecast | None = None,
        *,
        n_windows: int | None = None,
        step_size: int = 1,
        refit: bool | int = False,
    ) -> CalibrationResult:
        """
        Fit the conformal correction on a calibration set.

        Two calling conventions:

        * Pass ``histories`` and ``truths`` (works with any adapter).
        * Pass ``n_windows`` (and optionally ``step_size`` / ``refit``) to
          delegate to ``forecaster.cross_validate``. Requires
          :class:`SupportsCrossValidation`. This is the fast path for
          libraries with native CV (StatsForecast, MLForecast, …) — one
          library-native call instead of N separate forecast calls.

        Parameters
        ----------
        histories : list of Series, optional
            Each shape ``(n_series, T)``. Required if ``n_windows`` is None.
        truths : Forecast, optional
            Shape ``(n_series, len(histories), horizon)``. Required if
            ``n_windows`` is None.
        n_windows : int, optional
            Number of CV windows.
        step_size : int
            Step size between CV windows.
        refit : bool or int
            Whether to refit between CV windows.

        Returns
        -------
        CalibrationResult

        Raises
        ------
        CalibrationError
            If the number of calibration samples is too small for the
            requested ``alpha``. At least ``ceil(1 / alpha)`` samples are
            required for the quantile to be well-defined.
        UnsupportedCapability
            If ``n_windows`` is provided but the adapter does not implement
            :class:`SupportsCrossValidation`.
        ValueError
            If neither ``(histories, truths)`` nor ``n_windows`` is provided,
            or if both are.
        """
        predictions, truths = self._collect_calibration_data(
            histories=histories,
            truths=truths,
            n_windows=n_windows,
            step_size=step_size,
            refit=refit,
        )
        n_cal = predictions.shape[1]

        min_samples = math.ceil(1.0 / self.alpha)
        if n_cal < min_samples:
            raise CalibrationError(
                f"Need at least ceil(1/alpha) = {min_samples} calibration "
                f"samples for alpha={self.alpha}, got {n_cal}."
            )

        # Let the score function fit any internal parameters (no-op for AbsoluteResidual)
        self.score_fn.fit(predictions, truths)

        # Nonconformity scores: (n_series, n_cal, horizon)
        scores = self.score_fn.score(predictions, truths)

        # Empirical quantile level with finite-sample correction
        quantile_level = min((math.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal), 1.0)

        # Fitted state lives on self (sklearn convention: trailing underscore).
        self.score_quantile_: NDArray[np.floating] = np.quantile(scores, quantile_level, axis=1)
        self.n_calibration_samples_: int = n_cal
        self.is_calibrated_ = True

        # CalibrationResult is a snapshot — defensively copy mutable arrays so the
        # caller's record can't be mutated by future update() calls.
        return CalibrationResult(
            n_calibration_samples=n_cal,
            score_quantile=self.score_quantile_.copy(),
            diagnostics={"quantile_level": quantile_level},
        )

    def _collect_calibration_data(
        self,
        histories: list[Series] | None,
        truths: Forecast | None,
        n_windows: int | None,
        step_size: int,
        refit: bool | int,
    ) -> tuple[Forecast, Forecast]:
        """Resolve calibration predictions and truths from either input form."""
        if n_windows is not None:
            if histories is not None or truths is not None:
                raise ValueError("Provide either (histories, truths) or n_windows, not both.")
            if not isinstance(self.forecaster, SupportsCrossValidation):
                raise UnsupportedCapability(
                    f"calibrate(n_windows=...) requires an adapter implementing "
                    f"SupportsCrossValidation. Got {type(self.forecaster).__name__}. "
                    "Pass explicit (histories, truths) instead."
                )
            return self.forecaster.cross_validate(
                n_windows=n_windows, step_size=step_size, refit=refit
            )

        if histories is None or truths is None:
            raise ValueError("Must provide either (histories, truths) or n_windows.")
        # Loop fallback: one forecast per history.
        predictions = self.forecaster.predict_batch(histories)
        return predictions, truths

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
        if not self.is_calibrated_:
            raise CalibrationError("predict() called before calibrate(). Call calibrate() first.")

        point: Forecast = self.forecaster.predict(history)
        interval: Interval = self.score_fn.invert(point, self.score_quantile_)

        return PredictionResult(
            point=point,
            interval=interval,
            alpha=self.alpha,
        )
