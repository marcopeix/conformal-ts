"""Adaptive Conformal Inference (ACI) for time series."""

from __future__ import annotations

import math
import warnings

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
    UnsupportedCapability,
)
from ..capabilities import SupportsCrossValidation
from ..nonconformity.absolute import AbsoluteResidual


class AdaptiveConformalInference(ConformalMethod):
    """
    Adaptive Conformal Inference (Gibbs & Candès 2021).

    An online conformal method that maintains a time-varying miscoverage rate
    ``alpha_t`` per ``(series, horizon)`` cell. After each new observation, the
    rate is nudged toward the target ``alpha`` by

        alpha_{t+1} = alpha_t + gamma * (alpha - 1{Y_t outside C_t})

    where ``C_t`` is the interval at step ``t``. The fixed-point property of
    this recursion guarantees long-run coverage of ``1 - alpha`` *without* any
    exchangeability assumption, which makes ACI well-suited to non-stationary
    time series.

    The implementation uses the same conformity score
    (:class:`AbsoluteResidual` by default) as split CP. The difference is that
    instead of a fixed ``1 - alpha`` quantile of calibration scores, ACI uses
    a time-varying ``1 - alpha_t`` quantile that is updated whenever
    :meth:`update` is called with a new observation.

    Parameters
    ----------
    forecaster : ForecasterAdapter
        Any adapter — no special capabilities required.
    alpha : float
        Target miscoverage in ``(0, 1)``. The target coverage is ``1 - alpha``.
    gamma : float, default 0.05
        Learning rate for ``alpha_t`` updates. Larger values adapt faster but
        track noisier; smaller values are smoother but slower. Gibbs & Candès
        recommend values in ``[0.005, 0.05]``.
    score : ScoreFunction or None
        Defaults to :class:`AbsoluteResidual`.

    Notes
    -----
    Calibration mirrors :class:`SplitConformal`'s two-path dispatch. The fast
    path delegates to ``forecaster.cross_validate`` (requires
    :class:`SupportsCrossValidation`); the loop path uses ``predict_batch`` over
    a user-supplied calibration set.

    Online lifecycle
    ----------------
    After ``calibrate``, callers alternate ``predict(history)`` and
    ``update(prediction, truth)``. ``update`` recomputes the nonconformity
    score from the supplied ``(prediction, truth)`` pair, appends it to the
    score history, and rolls ``alpha_t`` forward. The caller passes the same
    ``point`` they got back from :class:`PredictionResult`, so there is no
    hidden state coupling between ``predict`` and ``update``.
    """

    REQUIRED_CAPABILITIES: tuple[type, ...] = ()
    IS_ONLINE: bool = True

    def __init__(
        self,
        forecaster: ForecasterAdapter,
        alpha: float,
        gamma: float = 0.05,
        score: ScoreFunction | None = None,
    ) -> None:
        super().__init__(forecaster=forecaster, alpha=alpha, score=score)

        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if gamma > 0.1:
            warnings.warn(
                f"gamma={gamma} is unusually large; Gibbs & Candès recommend [0.005, 0.05]. "
                "Updates will be very noisy.",
                stacklevel=2,
            )

        self.gamma: float = float(gamma)

    def _default_score(self) -> ScoreFunction:
        return AbsoluteResidual()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

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
        Fit the ACI online state on a calibration set.

        Two calling conventions:

        * Pass ``histories`` and ``truths`` (loop path). Works with any adapter.
        * Pass ``n_windows`` (and optionally ``step_size`` / ``refit``) to
          delegate to ``forecaster.cross_validate``. Requires
          :class:`SupportsCrossValidation`.

        Both paths run the same online ACI loop over the calibration samples
        in chronological order, producing identical fitted state when fed
        equivalent calibration data (within floating-point ordering).

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
            ``diagnostics["path"]`` is ``"loop"`` or ``"cross_validation"``.

        Raises
        ------
        CalibrationError
            If fewer than ``ceil(1 / alpha)`` calibration samples are
            available.
        ValueError
            If neither calling convention is provided, or both are.
        UnsupportedCapability
            If ``n_windows`` is requested on an adapter without
            :class:`SupportsCrossValidation`.
        """
        if n_windows is not None:
            if histories is not None or truths is not None:
                raise ValueError("Provide either (histories, truths) or n_windows, not both.")
            return self._calibrate_via_cv(n_windows=n_windows, step_size=step_size, refit=refit)

        if histories is None or truths is None:
            raise ValueError("Must provide either (histories, truths) or n_windows.")
        return self._calibrate_via_loop(histories=histories, truths=truths)

    def _calibrate_via_cv(
        self,
        n_windows: int,
        step_size: int,
        refit: bool | int,
    ) -> CalibrationResult:
        if not isinstance(self.forecaster, SupportsCrossValidation):
            raise UnsupportedCapability(
                "calibrate(n_windows=...) requires an adapter implementing "
                f"SupportsCrossValidation. Got {type(self.forecaster).__name__}. "
                "Pass explicit (histories, truths) instead."
            )

        min_samples = math.ceil(1.0 / self.alpha)
        if n_windows < min_samples:
            raise CalibrationError(
                f"Need at least ceil(1/alpha) = {min_samples} calibration "
                f"samples for alpha={self.alpha}, got n_windows={n_windows}."
            )

        predictions, truths = self.forecaster.cross_validate(
            n_windows=n_windows, step_size=step_size, refit=refit
        )
        self._run_aci_loop(predictions, truths)

        return CalibrationResult(
            n_calibration_samples=self.n_calibration_samples_,
            score_quantile=self._current_threshold().copy(),
            diagnostics={
                "gamma": self.gamma,
                "alpha_t": self.alpha_t_.copy(),
                "path": "cross_validation",
            },
        )

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

        predictions = self.forecaster.predict_batch(histories)
        truths_arr = np.asarray(truths, dtype=np.float64)
        self._run_aci_loop(predictions, truths_arr)

        return CalibrationResult(
            n_calibration_samples=self.n_calibration_samples_,
            score_quantile=self._current_threshold().copy(),
            diagnostics={
                "gamma": self.gamma,
                "alpha_t": self.alpha_t_.copy(),
                "path": "loop",
            },
        )

    def _run_aci_loop(
        self,
        predictions: Forecast,
        truths: Forecast,
    ) -> None:
        """
        Run the online ACI loop over calibration samples.

        Updates fitted state in place: ``alpha_t_``, ``scores_``,
        ``n_observations_``, ``n_calibration_samples_``, ``is_calibrated_``.
        """
        # Let the score function fit any internal parameters (no-op for
        # AbsoluteResidual). Done before scoring per the base contract.
        self.score_fn.fit(predictions, truths)

        # (n_series, n_cal, horizon)
        all_scores = self.score_fn.score(predictions, truths)
        n_series, n_cal, horizon = all_scores.shape

        alpha_t = np.full((n_series, horizon), self.alpha, dtype=np.float64)

        for t in range(n_cal):
            current_score = all_scores[:, t, :]  # (n_series, horizon)

            if t > 0:
                past_scores = all_scores[:, :t, :]  # (n_series, t, horizon)
                quantile_levels = np.clip(1.0 - alpha_t, 0.0, 1.0)
                threshold = self._per_cell_quantile(past_scores, quantile_levels)
                missed = (current_score > threshold).astype(np.float64)
            else:
                # No past scores at t=0. Following Gibbs & Candès' online
                # formulation, treat the first step as a miss to encourage
                # initial widening rather than producing a degenerate
                # zero-width interval.
                missed = np.ones((n_series, horizon), dtype=np.float64)

            alpha_t = alpha_t + self.gamma * (self.alpha - missed)

        self.alpha_t_: NDArray[np.floating] = alpha_t
        self.scores_: NDArray[np.floating] = all_scores
        self.n_calibration_samples_: int = n_cal
        self.n_observations_: int = n_cal
        self.is_calibrated_ = True

    # ------------------------------------------------------------------
    # Per-cell quantile helper
    # ------------------------------------------------------------------

    def _per_cell_quantile(
        self,
        scores: NDArray[np.floating],
        quantile_levels: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Per-(series, horizon) quantile of an observed-score panel.

        Parameters
        ----------
        scores : NDArray, shape (n_series, t, horizon)
            Time-axis is axis 1.
        quantile_levels : NDArray, shape (n_series, horizon)
            Quantile level per cell, in ``[0, 1]``.

        Returns
        -------
        NDArray, shape (n_series, horizon)
            ``out[s, h] = np.quantile(scores[s, :, h], quantile_levels[s, h])``
            with saturation at the level boundaries.

        Notes
        -----
        The nested-loop implementation is O(n_series * horizon * t log t).
        Vectorising via ``np.sort`` + ``searchsorted`` on a per-cell quantile
        index is a known optimisation opportunity, deferred until profiling
        flags it.
        """
        n_series, _, horizon = scores.shape
        out = np.empty((n_series, horizon), dtype=np.float64)
        for s in range(n_series):
            for h in range(horizon):
                ql = float(quantile_levels[s, h])
                if ql >= 1.0:
                    out[s, h] = np.finfo(np.float64).max
                elif ql <= 0.0:
                    out[s, h] = -np.finfo(np.float64).max
                else:
                    out[s, h] = float(np.quantile(scores[s, :, h], ql))
        return out

    def _current_threshold(self) -> NDArray[np.floating]:
        """Score threshold per (series, horizon) at the current ``alpha_t_``."""
        quantile_levels = np.clip(1.0 - self.alpha_t_, 0.0, 1.0)
        return self._per_cell_quantile(self.scores_, quantile_levels)

    # ------------------------------------------------------------------
    # Predict / update
    # ------------------------------------------------------------------

    def predict(self, history: Series) -> PredictionResult:
        """
        Produce an ACI-calibrated point forecast and prediction interval.

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

        Notes
        -----
        Unlike split CP, repeated ``predict`` calls with the same ``history``
        may return different intervals if :meth:`update` was called in
        between, because ``alpha_t_`` and ``scores_`` evolve over time.
        """
        if not self.is_calibrated_:
            raise CalibrationError("predict() called before calibrate(). Call calibrate() first.")

        point: Forecast = self.forecaster.predict(history)
        threshold = self._current_threshold()  # (n_series, horizon)
        interval: Interval = self.score_fn.invert(point, threshold)

        return PredictionResult(point=point, interval=interval, alpha=self.alpha)

    def update(self, prediction: Forecast, truth: Forecast) -> None:
        """
        Roll ACI's online state forward with a new ``(prediction, truth)`` pair.

        Pass the ``point`` field of the :class:`PredictionResult` returned by
        the corresponding :meth:`predict` call as ``prediction``, and the
        realized values for the same horizon as ``truth``.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, 1, horizon)
        truth : Forecast, shape (n_series, 1, horizon)

        Raises
        ------
        CalibrationError
            If :meth:`calibrate` has not been called.
        ValueError
            If ``prediction`` and ``truth`` do not share the expected shape
            ``(n_series, 1, horizon)``.
        """
        if not self.is_calibrated_:
            raise CalibrationError("update() called before calibrate(). Call calibrate() first.")

        prediction_arr = np.asarray(prediction, dtype=np.float64)
        truth_arr = np.asarray(truth, dtype=np.float64)

        n_series, _, horizon = self.scores_.shape
        expected_shape = (n_series, 1, horizon)
        if prediction_arr.shape != expected_shape:
            raise ValueError(
                f"prediction must have shape {expected_shape}, got {prediction_arr.shape}."
            )
        if truth_arr.shape != expected_shape:
            raise ValueError(f"truth must have shape {expected_shape}, got {truth_arr.shape}.")

        # Score the realized residual.
        new_score = self.score_fn.score(prediction_arr, truth_arr)
        # (n_series, 1, horizon)

        # Coverage indicator under the alpha_t_ used to construct the interval
        # the user just observed. Use the score history *before* appending
        # this new observation, matching the calibration loop's convention.
        quantile_levels = np.clip(1.0 - self.alpha_t_, 0.0, 1.0)
        threshold = self._per_cell_quantile(self.scores_, quantile_levels)
        # (n_series, horizon)

        missed = (new_score[:, 0, :] > threshold).astype(np.float64)
        # (n_series, horizon)

        # Append the new score and roll alpha_t forward.
        self.scores_ = np.concatenate([self.scores_, new_score], axis=1)
        self.n_observations_ += 1
        self.alpha_t_ = self.alpha_t_ + self.gamma * (self.alpha - missed)
