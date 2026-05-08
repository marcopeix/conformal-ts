"""Conformalized Quantile Regression for time series."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..base import (
    CalibrationError,
    CalibrationResult,
    ConformalMethod,
    Forecast,
    ForecasterAdapter,
    Interval,
    PredictionResult,
    ScoreFunction,
    Series,
)
from ..capabilities import SupportsCrossValidationQuantiles, SupportsQuantiles
from ..nonconformity.quantile import QuantileScore


class ConformalizedQuantileRegression(ConformalMethod):
    """
    Conformalized Quantile Regression (Romano, Patterson, Candès 2019)
    adapted for panel time series.

    Wraps a quantile-producing forecaster and calibrates its quantile
    interval so that empirical coverage matches the target ``1 - alpha``.

    Algorithm
    ---------
    1. **calibrate**: for each calibration history, query the forecaster
       for the lower and upper quantile predictions and compute CQR
       nonconformity scores against the truth (see :class:`QuantileScore`).
       Store the empirical ``ceil((1 - alpha)(1 + n)) / n`` quantile of
       scores per (series, horizon) pair.
    2. **predict**: query the forecaster's quantile predictions and apply
       :meth:`QuantileScore.invert` using the stored quantile to widen the
       interval.

    Parameters
    ----------
    forecaster : ForecasterAdapter
        Must implement :class:`SupportsQuantiles`.
    alpha : float
        Target miscoverage in ``(0, 1)``. The target coverage is
        ``1 - alpha``.
    symmetric : bool, default True
        If True, the underlying quantile pair is fixed at
        ``(alpha/2, 1 - alpha/2)`` and ``alpha_lo`` / ``alpha_hi`` are
        ignored. If False, the user must supply both ``alpha_lo`` and
        ``alpha_hi`` explicitly with combined miscoverage equal to
        ``alpha``.
    alpha_lo : float, optional
        Lower-quantile level (probability mass below the lower bound).
        Required when ``symmetric=False``.
    alpha_hi : float, optional
        Upper-quantile level (probability mass below the upper bound).
        Required when ``symmetric=False``.
    score : ScoreFunction or None
        Defaults to :class:`QuantileScore`.

    Notes
    -----
    Calibration supports two calling conventions:

    * **Loop path** — pass ``histories`` and ``truths``. The method calls
      ``forecaster.predict_quantiles`` once per calibration window. Works
      with any :class:`SupportsQuantiles` adapter (including library-agnostic
      ``QuantileCallableAdapter``-style test adapters).
    * **Cross-validation path** — pass ``n_windows`` (and optionally
      ``step_size`` / ``refit``). Delegates to
      ``forecaster.cross_validate_quantiles`` for a single library-native
      call instead of N separate quantile predictions. Requires the adapter
      to implement :class:`SupportsCrossValidationQuantiles`. For libraries
      with a real CV implementation (StatsForecast, NeuralForecast, …),
      this is dramatically faster than the loop path.

    Dispatch happens in :meth:`calibrate` based on whether ``n_windows`` is
    provided. Both paths produce numerically equivalent ``score_quantile_``
    values when fed equivalent calibration data (within floating-point
    ordering tolerance) and are reported in
    ``CalibrationResult.diagnostics["path"]``.
    """

    REQUIRED_CAPABILITIES: tuple[type, ...] = (SupportsQuantiles,)
    IS_ONLINE: bool = False

    def __init__(
        self,
        forecaster: ForecasterAdapter,
        alpha: float,
        symmetric: bool = True,
        alpha_lo: float | None = None,
        alpha_hi: float | None = None,
        score: ScoreFunction | None = None,
    ) -> None:
        super().__init__(forecaster=forecaster, alpha=alpha, score=score)

        if symmetric:
            if alpha_lo is not None or alpha_hi is not None:
                raise ValueError(
                    "alpha_lo and alpha_hi cannot be specified when symmetric=True. "
                    "Either remove them or set symmetric=False."
                )
            alpha_lo = alpha / 2.0
            alpha_hi = 1.0 - alpha / 2.0
        else:
            if alpha_lo is None or alpha_hi is None:
                raise ValueError(
                    "alpha_lo and alpha_hi must both be provided when symmetric=False."
                )
            if not (0.0 < alpha_lo < 0.5 < alpha_hi < 1.0):
                raise ValueError(
                    "alpha_lo and alpha_hi must satisfy "
                    f"0 < alpha_lo < 0.5 < alpha_hi < 1; got alpha_lo={alpha_lo}, "
                    f"alpha_hi={alpha_hi}."
                )
            combined = alpha_lo + (1.0 - alpha_hi)
            if not np.isclose(combined, alpha):
                raise ValueError(
                    "alpha_lo + (1 - alpha_hi) must equal alpha. "
                    f"Got alpha_lo={alpha_lo}, alpha_hi={alpha_hi}, "
                    f"sum={combined}, alpha={alpha}."
                )

        self.symmetric: bool = symmetric
        self.alpha_lo: float = alpha_lo
        self.alpha_hi: float = alpha_hi
        self.quantiles_: NDArray[np.floating] = np.array([alpha_lo, alpha_hi], dtype=np.float64)

    def _default_score(self) -> ScoreFunction:
        return QuantileScore()

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
        Fit the CQR conformal correction on a calibration set.

        Two calling conventions are supported:

        * Pass ``histories`` and ``truths`` (loop path). Works with any
          :class:`SupportsQuantiles` adapter.
        * Pass ``n_windows`` (and optionally ``step_size`` / ``refit``) to
          delegate to ``forecaster.cross_validate_quantiles``. Requires the
          adapter to implement :class:`SupportsCrossValidationQuantiles`.

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
            Step size between CV windows. Only used with ``n_windows``.
        refit : bool or int
            Whether to refit between CV windows. Only used with ``n_windows``.

        Returns
        -------
        CalibrationResult
            ``diagnostics["path"]`` is ``"loop"`` for the explicit
            ``(histories, truths)`` form and ``"cross_validation"`` for the
            ``n_windows`` form.

        Raises
        ------
        CalibrationError
            If fewer than ``ceil(1 / alpha)`` calibration samples are
            available.
        ValueError
            If neither calling convention is provided, both are provided, or
            ``n_windows`` is requested on an adapter without
            :class:`SupportsCrossValidationQuantiles`.
        UnsupportedCapability
            If ``n_windows`` is requested on an adapter that declares
            :class:`SupportsCrossValidationQuantiles` but cannot deliver it
            at runtime (e.g. NeuralForecast fit with a non-probabilistic
            loss).
        """
        if n_windows is not None:
            if histories is not None or truths is not None:
                raise ValueError("Provide either (histories, truths) or n_windows, not both.")
            if not isinstance(self.forecaster, SupportsCrossValidationQuantiles):
                raise ValueError(
                    "calibrate(n_windows=...) requires an adapter implementing "
                    f"SupportsCrossValidationQuantiles. Got "
                    f"{type(self.forecaster).__name__}. Pass explicit "
                    "(histories, truths) instead."
                )
            return self._calibrate_via_cv(n_windows=n_windows, step_size=step_size, refit=refit)

        if histories is None or truths is None:
            raise ValueError("Must provide either (histories, truths) or n_windows.")
        return self._calibrate_via_loop(histories=histories, truths=truths)

    def _calibrate_via_loop(
        self,
        histories: list[Series],
        truths: Forecast,
    ) -> CalibrationResult:
        n_cal = len(histories)
        min_samples = math.ceil(1.0 / self.alpha)
        if n_cal < min_samples:
            raise CalibrationError(
                f"Need at least ceil(1/alpha) = {min_samples} calibration "
                f"samples for alpha={self.alpha}, got {n_cal}."
            )

        # Each predict_quantiles call returns shape (n_series, 2, horizon).
        # Stack across calibration windows -> (n_cal, n_series, 2, horizon),
        # then transpose so quantiles end up on the last axis to match
        # QuantileScore's expected shape (n_series, n_cal, horizon, 2).
        forecaster = self.forecaster
        assert isinstance(forecaster, SupportsQuantiles)  # checked in __init__
        raw = np.stack(
            [forecaster.predict_quantiles(h, self.quantiles_) for h in histories],
            axis=0,
        )  # (n_cal, n_series, 2, horizon)
        quantile_predictions = np.transpose(raw, (1, 0, 3, 2))
        # (n_series, n_cal, horizon, 2)

        truths_arr = np.asarray(truths, dtype=np.float64)

        self.score_fn.fit(quantile_predictions, truths_arr)
        scores = self.score_fn.score(quantile_predictions, truths_arr)
        # (n_series, n_cal, horizon)

        quantile_level = min(math.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal, 1.0)

        self.score_quantile_: NDArray[np.floating] = np.quantile(scores, quantile_level, axis=1)
        self.n_calibration_samples_: int = n_cal
        self.is_calibrated_ = True

        return CalibrationResult(
            n_calibration_samples=n_cal,
            score_quantile=self.score_quantile_.copy(),
            diagnostics={
                "quantile_level": quantile_level,
                "quantiles_used": self.quantiles_.tolist(),
                "symmetric": self.symmetric,
                "path": "loop",
            },
        )

    def _calibrate_via_cv(
        self,
        n_windows: int,
        step_size: int,
        refit: bool | int,
    ) -> CalibrationResult:
        forecaster = self.forecaster
        assert isinstance(forecaster, SupportsCrossValidationQuantiles)  # checked above

        min_samples = math.ceil(1.0 / self.alpha)
        if n_windows < min_samples:
            raise CalibrationError(
                f"Need at least ceil(1/alpha) = {min_samples} calibration "
                f"samples for alpha={self.alpha}, got n_windows={n_windows}."
            )

        # cross_validate_quantiles returns quantiles on the LAST axis already,
        # so the shape matches QuantileScore's expected input directly — no
        # moveaxis needed (unlike the loop path, which has to transpose from
        # the predict_quantiles convention).
        quantile_predictions, truths = forecaster.cross_validate_quantiles(
            n_windows=n_windows,
            step_size=step_size,
            quantiles=self.quantiles_,
            refit=refit,
        )
        # quantile_predictions: (n_series, n_windows, horizon, 2)
        # truths: (n_series, n_windows, horizon)

        self.score_fn.fit(quantile_predictions, truths)
        scores = self.score_fn.score(quantile_predictions, truths)
        # (n_series, n_windows, horizon)

        quantile_level = min(math.ceil((1 - self.alpha) * (n_windows + 1)) / n_windows, 1.0)

        self.score_quantile_ = np.quantile(scores, quantile_level, axis=1)
        self.n_calibration_samples_ = n_windows
        self.is_calibrated_ = True

        return CalibrationResult(
            n_calibration_samples=n_windows,
            score_quantile=self.score_quantile_.copy(),
            diagnostics={
                "quantile_level": quantile_level,
                "quantiles_used": self.quantiles_.tolist(),
                "symmetric": self.symmetric,
                "path": "cross_validation",
            },
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
            ``point`` has shape ``(n_series, 1, horizon)`` and is the
            **midpoint of the predicted quantile interval**, not a
            model-native point forecast.
            ``interval`` has shape ``(n_series, 1, horizon, 2)``.

        Raises
        ------
        CalibrationError
            If called before :meth:`calibrate`.
        """
        if not self.is_calibrated_:
            raise CalibrationError("predict() called before calibrate(). Call calibrate() first.")

        forecaster = self.forecaster
        assert isinstance(forecaster, SupportsQuantiles)  # checked in __init__
        q_pred = forecaster.predict_quantiles(history, self.quantiles_)
        # (n_series, 2, horizon) -> (n_series, horizon, 2) -> (n_series, 1, horizon, 2)
        q_pred_reshaped = np.transpose(q_pred, (0, 2, 1))[:, np.newaxis, :, :]

        # Point forecast: midpoint of the predicted quantile interval.
        point: Forecast = (q_pred_reshaped[..., 0] + q_pred_reshaped[..., 1]) / 2.0

        interval: Interval = self.score_fn.invert(q_pred_reshaped, self.score_quantile_)

        return PredictionResult(point=point, interval=interval, alpha=self.alpha)
