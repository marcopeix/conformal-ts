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
from ..capabilities import SupportsQuantiles
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
    **v0.1 limitation**: CQR currently calibrates by calling
    ``forecaster.predict_quantiles`` once per calibration history. A
    ``SupportsCrossValidationQuantiles`` fast path that mirrors
    :class:`SplitConformal`'s ``cross_validate`` dispatch is tracked as
    future work.
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

        Only the explicit ``(histories, truths)`` calling convention is
        supported in v0.1. The cross-validation fast path requires a
        ``SupportsCrossValidationQuantiles`` capability that does not yet
        exist; passing ``n_windows`` raises :class:`NotImplementedError`.

        Parameters
        ----------
        histories : list of Series
            Each shape ``(n_series, T)``.
        truths : Forecast
            Shape ``(n_series, len(histories), horizon)``.
        n_windows : int, optional
            Not supported. Raises :class:`NotImplementedError` if set.
        step_size : int
            Unused in v0.1.
        refit : bool or int
            Unused in v0.1.

        Returns
        -------
        CalibrationResult

        Raises
        ------
        CalibrationError
            If fewer than ``ceil(1 / alpha)`` calibration samples are
            provided.
        NotImplementedError
            If ``n_windows`` is provided (CV fast path is future work).
        ValueError
            If ``histories`` or ``truths`` is missing.
        """
        if n_windows is not None:
            raise NotImplementedError(
                "CQR cross-validation calibration is not supported in v0.1. "
                "A SupportsCrossValidationQuantiles fast path is tracked as "
                "future work. Pass explicit (histories, truths) instead."
            )
        if histories is None or truths is None:
            raise ValueError("Must provide both histories and truths.")

        # step_size and refit are accepted for signature consistency with
        # SplitConformal.calibrate but unused in the loop path.
        del step_size, refit

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
