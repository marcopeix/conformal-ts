"""Aggregated Adaptive Conformal Inference (AgACI)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..aggregators import EWA, OnlineAggregator
from ..base import (
    CalibrationError,
    CalibrationResult,
    ConformalMethod,
    Forecast,
    ForecasterAdapter,
    PredictionResult,
    ScoreFunction,
    Series,
    UnsupportedCapability,
)
from ..capabilities import SupportsCrossValidation
from ..nonconformity.absolute import AbsoluteResidual
from ._online_helpers import _per_cell_quantile, _validate_online_shapes


def _default_aggregator_factory(n_experts: int, n_series: int, horizon: int) -> OnlineAggregator:
    return EWA(n_experts=n_experts, n_series=n_series, horizon=horizon)


class AggregatedAdaptiveConformalInference(ConformalMethod):
    """
    Aggregated Adaptive Conformal Inference (Zaffran et al. 2022).

    Runs ``K = len(gammas)`` parallel ACI experts with different learning rates
    and aggregates their interval bounds via online expert aggregation. The
    aggregation is independent for the lower and upper bounds, with each
    bound's weights driven by per-expert pinball loss at level
    ``beta_lower = alpha / 2`` and ``beta_upper = 1 - alpha / 2`` respectively.

    Parameters
    ----------
    forecaster : ForecasterAdapter
        Any adapter — no special capabilities required.
    alpha : float
        Target miscoverage level in ``(0, 1)``. Coverage target is ``1 - alpha``.
    gammas : Sequence[float], default ``(0.001, 0.005, 0.01, 0.05, 0.1, 0.2)``
        Grid of ACI learning rates. Must be non-empty; each value must be > 0.
    aggregator_factory : Callable[[int, int, int], OnlineAggregator], optional
        Factory ``fn(K, n_series, horizon) -> OnlineAggregator`` constructing
        the lower- and upper-bound aggregators. Two aggregators are built per
        :meth:`calibrate` call, so the factory is invoked twice. Defaults to
        :class:`~conformal_ts.aggregators.EWA` with ``eta=1``.
    interval_clip_lower : float or None, default None
        Manual clip threshold for lower bounds. If ``None``, computed at
        calibration time as ``min(truths) - 5 * std(truths)``.
    interval_clip_upper : float or None, default None
        Manual clip threshold for upper bounds. If ``None``, computed at
        calibration time as ``max(truths) + 5 * std(truths)``.
    score : ScoreFunction or None, optional
        Defaults to :class:`AbsoluteResidual`.

    Notes
    -----
    Per the paper, AgACI uses BOA (Bernstein Online Aggregation) with the
    gradient trick. v0.1 ships :class:`EWA` as the only aggregator and exposes
    ``aggregator_factory`` to make the choice pluggable. BOA and ML-Poly are
    tracked as future work.
    """

    REQUIRED_CAPABILITIES: tuple[type, ...] = ()
    IS_ONLINE: bool = True

    def __init__(
        self,
        forecaster: ForecasterAdapter,
        alpha: float,
        gammas: Sequence[float] = (0.001, 0.005, 0.01, 0.05, 0.1, 0.2),
        aggregator_factory: Callable[[int, int, int], OnlineAggregator] | None = None,
        interval_clip_lower: float | None = None,
        interval_clip_upper: float | None = None,
        score: ScoreFunction | None = None,
    ) -> None:
        super().__init__(forecaster=forecaster, alpha=alpha, score=score)

        gammas_tuple = tuple(float(g) for g in gammas)
        if len(gammas_tuple) == 0:
            raise ValueError("gammas must contain at least one learning rate.")
        if any(g <= 0 for g in gammas_tuple):
            raise ValueError(f"all gammas must be > 0, got {gammas_tuple}.")
        self.gammas: tuple[float, ...] = gammas_tuple
        self.n_experts: int = len(gammas_tuple)

        # Clip thresholds: both None or both not None.
        if (interval_clip_lower is None) != (interval_clip_upper is None):
            raise ValueError(
                "interval_clip_lower and interval_clip_upper must both be provided or both be None."
            )
        if interval_clip_lower is not None and interval_clip_upper is not None:
            if interval_clip_lower >= interval_clip_upper:
                raise ValueError(
                    f"interval_clip_lower ({interval_clip_lower}) must be strictly less "
                    f"than interval_clip_upper ({interval_clip_upper})."
                )
        self.interval_clip_lower: float | None = (
            float(interval_clip_lower) if interval_clip_lower is not None else None
        )
        self.interval_clip_upper: float | None = (
            float(interval_clip_upper) if interval_clip_upper is not None else None
        )

        self.aggregator_factory: Callable[[int, int, int], OnlineAggregator] = (
            aggregator_factory if aggregator_factory is not None else _default_aggregator_factory
        )

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
        Fit the AgACI online state on a calibration set.

        Two calling conventions:

        * Pass ``histories`` and ``truths`` (loop path). Works with any adapter.
        * Pass ``n_windows`` (and optionally ``step_size`` / ``refit``) to
          delegate to ``forecaster.cross_validate``. Requires
          :class:`SupportsCrossValidation`.

        Both paths run the same online AgACI loop over the calibration samples
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
            If fewer than ``ceil(1 / alpha)`` calibration samples are available.
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
        self._run_agaci_loop(predictions, truths)

        return CalibrationResult(
            n_calibration_samples=self.n_calibration_samples_,
            score_quantile=self._per_expert_thresholds(),
            diagnostics={
                "gammas": self.gammas,
                "alpha_t_per_expert": self.alpha_t_per_expert_.copy(),
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
        self._run_agaci_loop(predictions, truths_arr)

        return CalibrationResult(
            n_calibration_samples=self.n_calibration_samples_,
            score_quantile=self._per_expert_thresholds(),
            diagnostics={
                "gammas": self.gammas,
                "alpha_t_per_expert": self.alpha_t_per_expert_.copy(),
                "path": "loop",
            },
        )

    def _run_agaci_loop(self, predictions: Forecast, truths: Forecast) -> None:
        """
        Run the online AgACI loop over calibration samples.

        Updates fitted state in place: ``alpha_t_per_expert_``, ``scores_``,
        ``aggregator_lower_``, ``aggregator_upper_``, ``interval_clip_lower_``,
        ``interval_clip_upper_``, ``n_observations_``, ``n_calibration_samples_``,
        ``is_calibrated_``.
        """
        # Score function is shared across experts (depends only on the score
        # function, not the per-expert alpha_t).
        self.score_fn.fit(predictions, truths)
        all_scores = self.score_fn.score(predictions, truths)
        # all_scores: (n_series, n_cal, horizon)
        n_series, n_cal, horizon = predictions.shape
        K = self.n_experts
        gammas_arr = np.array(self.gammas, dtype=np.float64).reshape(K, 1, 1)

        # Auto-compute clip thresholds if not provided.
        if self.interval_clip_lower is None:
            truth_std = float(np.std(truths))
            self.interval_clip_lower_: float = float(np.min(truths) - 5 * truth_std)
            self.interval_clip_upper_: float = float(np.max(truths) + 5 * truth_std)
        else:
            assert self.interval_clip_upper is not None  # validated in __init__
            self.interval_clip_lower_ = self.interval_clip_lower
            self.interval_clip_upper_ = self.interval_clip_upper

        alpha_t_per_expert = np.full((K, n_series, horizon), self.alpha, dtype=np.float64)

        self.aggregator_lower_: OnlineAggregator = self.aggregator_factory(K, n_series, horizon)
        self.aggregator_upper_: OnlineAggregator = self.aggregator_factory(K, n_series, horizon)

        beta_lower = self.alpha / 2.0
        beta_upper = 1.0 - self.alpha / 2.0

        for t in range(n_cal):
            point_t = predictions[:, t, :]  # (n_series, horizon)
            truth_t = truths[:, t, :]  # (n_series, horizon)

            if t > 0:
                past_scores = all_scores[:, :t, :]
                quantile_levels = np.clip(1.0 - alpha_t_per_expert, 0.0, 1.0)
                # (K, n_series, horizon)
                thresholds = np.empty((K, n_series, horizon), dtype=np.float64)
                for k in range(K):
                    thresholds[k] = _per_cell_quantile(past_scores, quantile_levels[k])
                lower_bounds = point_t[None, :, :] - thresholds
                upper_bounds = point_t[None, :, :] + thresholds
            else:
                # First step: no past scores. Use the point forecast itself
                # as a degenerate interval (zero width); the clip step below
                # makes this finite.
                lower_bounds = np.broadcast_to(point_t[None, :, :], (K, n_series, horizon)).astype(
                    np.float64, copy=True
                )
                upper_bounds = lower_bounds.copy()

            lower_bounds = np.maximum(lower_bounds, self.interval_clip_lower_)
            upper_bounds = np.minimum(upper_bounds, self.interval_clip_upper_)

            losses_lower = self._pinball_loss(truth_t[None, :, :], lower_bounds, beta_lower)
            losses_upper = self._pinball_loss(truth_t[None, :, :], upper_bounds, beta_upper)

            self.aggregator_lower_.update(losses_lower)
            self.aggregator_upper_.update(losses_upper)

            missed = (
                (truth_t[None, :, :] < lower_bounds) | (truth_t[None, :, :] > upper_bounds)
            ).astype(np.float64)
            alpha_t_per_expert = alpha_t_per_expert + gammas_arr * (self.alpha - missed)

        self.alpha_t_per_expert_: NDArray[np.floating] = alpha_t_per_expert
        self.scores_: NDArray[np.floating] = all_scores
        self.n_calibration_samples_: int = n_cal
        self.n_observations_: int = n_cal
        self.is_calibrated_ = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pinball_loss(
        truth: NDArray[np.floating],
        prediction: NDArray[np.floating],
        beta: float,
    ) -> NDArray[np.floating]:
        """Pinball loss ``rho_beta(y, yhat) = max(beta * (y - yhat), (beta - 1) * (y - yhat))``."""
        diff = truth - prediction
        return np.maximum(beta * diff, (beta - 1.0) * diff)

    def _per_expert_thresholds(self) -> NDArray[np.floating]:
        """
        Return per-expert score thresholds at the current ``alpha_t_per_expert_``.

        Used for diagnostic reporting in :class:`CalibrationResult`. Shape
        ``(n_experts, n_series, horizon)``. Saturated cells contain
        ``np.finfo(np.float64).max``; callers should not naively average
        across the expert axis.
        """
        K = self.n_experts
        n_series = self.scores_.shape[0]
        horizon = self.scores_.shape[2]
        quantile_levels = np.clip(1.0 - self.alpha_t_per_expert_, 0.0, 1.0)
        thresholds = np.empty((K, n_series, horizon), dtype=np.float64)
        for k in range(K):
            thresholds[k] = _per_cell_quantile(self.scores_, quantile_levels[k])
        return thresholds

    # ------------------------------------------------------------------
    # Predict / update
    # ------------------------------------------------------------------

    def _per_expert_bounds(
        self,
        point_squeezed: NDArray[np.floating],
        scores: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return ``(lower_bounds, upper_bounds)``, both shape ``(K, n_series, horizon)``."""
        K = self.n_experts
        n_series, horizon = point_squeezed.shape
        quantile_levels = np.clip(1.0 - self.alpha_t_per_expert_, 0.0, 1.0)
        thresholds = np.empty((K, n_series, horizon), dtype=np.float64)
        for k in range(K):
            thresholds[k] = _per_cell_quantile(scores, quantile_levels[k])
        lower_bounds = point_squeezed[None, :, :] - thresholds
        upper_bounds = point_squeezed[None, :, :] + thresholds
        lower_bounds = np.maximum(lower_bounds, self.interval_clip_lower_)
        upper_bounds = np.minimum(upper_bounds, self.interval_clip_upper_)
        return lower_bounds, upper_bounds

    def predict(self, history: Series) -> PredictionResult:
        """
        Produce a calibrated point forecast and aggregated prediction interval.

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
        The interval is the per-cell weighted average of ``K`` per-expert
        bounds, with weights given by the current aggregator state. Lower and
        upper bounds use independent aggregators, so the resulting interval
        is generally not symmetric around the point forecast.
        """
        if not self.is_calibrated_:
            raise CalibrationError("predict() called before calibrate(). Call calibrate() first.")

        point: Forecast = self.forecaster.predict(history)
        point_squeezed = point[:, 0, :]  # (n_series, horizon)
        lower_bounds, upper_bounds = self._per_expert_bounds(point_squeezed, self.scores_)

        w_lower = self.aggregator_lower_.weights()
        w_upper = self.aggregator_upper_.weights()
        aggregated_lower = (w_lower * lower_bounds).sum(axis=0)
        aggregated_upper = (w_upper * upper_bounds).sum(axis=0)

        interval = np.stack(
            [aggregated_lower[:, None, :], aggregated_upper[:, None, :]],
            axis=-1,
        )
        return PredictionResult(point=point, interval=interval, alpha=self.alpha)

    def update(self, prediction: Forecast, truth: Forecast) -> None:
        """
        Roll AgACI's online state forward with a new ``(prediction, truth)`` pair.

        Updates each expert's ``alpha_t`` based on whether *that expert's*
        individual interval (before aggregation) covered ``truth``, and updates
        both bound aggregators with the per-expert pinball losses on the new
        observation. Finally appends the new score to the score history.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, 1, horizon)
            The point forecast produced for this step (the ``point`` field of
            the corresponding :class:`PredictionResult`).
        truth : Forecast, shape (n_series, 1, horizon)
            Realized values for the same horizon.

        Raises
        ------
        CalibrationError
            If :meth:`calibrate` has not been called.
        ValueError
            If ``prediction`` and ``truth`` do not share shape ``(n_series, 1, horizon)``.
        """
        if not self.is_calibrated_:
            raise CalibrationError("update() called before calibrate(). Call calibrate() first.")

        n_series, _, horizon = self.scores_.shape
        prediction_arr, truth_arr = _validate_online_shapes(prediction, truth, n_series, horizon)

        # Recompute the per-expert bounds the user observed: these used the
        # alpha_t before this update and the score history before this step.
        point_squeezed = prediction_arr[:, 0, :]
        lower_bounds, upper_bounds = self._per_expert_bounds(point_squeezed, self.scores_)
        # (K, n_series, horizon)

        truth_squeezed = truth_arr[:, 0, :]
        beta_lower = self.alpha / 2.0
        beta_upper = 1.0 - self.alpha / 2.0
        losses_lower = self._pinball_loss(truth_squeezed[None, :, :], lower_bounds, beta_lower)
        losses_upper = self._pinball_loss(truth_squeezed[None, :, :], upper_bounds, beta_upper)
        self.aggregator_lower_.update(losses_lower)
        self.aggregator_upper_.update(losses_upper)

        # Per-expert miss indicator: did this expert's individual interval miss?
        missed = (
            (truth_squeezed[None, :, :] < lower_bounds)
            | (truth_squeezed[None, :, :] > upper_bounds)
        ).astype(np.float64)
        gammas_arr = np.array(self.gammas, dtype=np.float64).reshape(self.n_experts, 1, 1)
        self.alpha_t_per_expert_ = self.alpha_t_per_expert_ + gammas_arr * (self.alpha - missed)

        # Score the realized residual and append.
        new_score = self.score_fn.score(prediction_arr, truth_arr)
        self.scores_ = np.concatenate([self.scores_, new_score], axis=1)
        self.n_observations_ += 1
