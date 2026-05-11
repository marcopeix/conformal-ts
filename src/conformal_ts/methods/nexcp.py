"""Nonexchangeable Conformal Prediction (NexCP) for time series."""

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
    PredictionResult,
    ScoreFunction,
    Series,
    UnsupportedCapability,
)
from ..capabilities import SupportsCrossValidation
from ..nonconformity.absolute import AbsoluteResidual
from ._online_helpers import _validate_online_shapes


class NonexchangeableConformalPrediction(ConformalMethod):
    """
    Nonexchangeable Conformal Prediction (Barber et al. 2022).

    Generalizes split conformal prediction by assigning weights to calibration
    samples, with recent samples weighted more heavily. Provides valid
    coverage guarantees under non-exchangeability — particularly suited to
    non-stationary time series.

    Algorithm
    ---------
    Calibration scores ``S_1, ..., S_n`` (chronological, oldest first) are
    assigned exponential weights ``w_i = ρ^(n - i)`` so that the most recent
    sample gets weight ``1`` and the oldest gets weight ``ρ^(n - 1)``. The
    score threshold is the smallest score whose cumulative weight reaches
    ``(1 - α) (W + 1) / W`` where ``W = Σ w_i``. The ``(W + 1) / W`` factor
    accounts for the unobserved test point; as ``W → ∞`` it recovers the
    split-CP threshold ``1 - α``. For finite ``W`` the target is slightly
    above ``1 - α``, yielding a marginally more conservative interval.

    Online use
    ----------
    Each call to :meth:`update` appends the freshly observed score to the
    history. The newest sample receives weight ``1`` and all previous samples
    decay by a factor of ``ρ`` (this is the implicit consequence of indexing
    weights from the new ``n``). Weights are recomputed from
    ``n_observations_`` each time, so they are not stored as state.

    Parameters
    ----------
    forecaster : ForecasterAdapter
        Any adapter — no special capabilities required.
    alpha : float
        Miscoverage level in ``(0, 1)``. The target coverage is ``1 - alpha``.
    rho : float, default 0.99
        Exponential decay factor in ``(0, 1]``. Sample ``i`` (out of ``n``
        total, oldest first) receives weight ``ρ^(n - i)``. ``ρ = 1``
        recovers split conformal (uniform weights). Smaller ``ρ`` weights
        recent samples more heavily.
    score : ScoreFunction or None
        Defaults to :class:`AbsoluteResidual`.
    """

    REQUIRED_CAPABILITIES: tuple[type, ...] = ()
    IS_ONLINE: bool = True

    def __init__(
        self,
        forecaster: ForecasterAdapter,
        alpha: float,
        rho: float = 0.99,
        score: ScoreFunction | None = None,
    ) -> None:
        super().__init__(forecaster=forecaster, alpha=alpha, score=score)

        if not 0 < rho <= 1:
            raise ValueError(f"rho must be in (0, 1], got {rho}")

        self.rho: float = float(rho)

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
        Fit the weighted-quantile threshold on a calibration set.

        Two calling conventions:

        * Pass ``histories`` and ``truths`` (loop path). Works with any adapter.
        * Pass ``n_windows`` (and optionally ``step_size`` / ``refit``) to
          delegate to ``forecaster.cross_validate``. Requires
          :class:`SupportsCrossValidation`.

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
            ``diagnostics`` includes ``rho``, ``effective_sample_size``, and
            ``path`` (``"loop"`` or ``"cross_validation"``).

        Raises
        ------
        CalibrationError
            If the effective sample size of the weighted calibration scores
            is below ``ceil(1 / alpha)``.
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

        ess = self._effective_sample_size(n_windows)
        self._check_ess(ess)

        predictions, truths = self.forecaster.cross_validate(
            n_windows=n_windows, step_size=step_size, refit=refit
        )
        self._fit_state(predictions, truths)

        return CalibrationResult(
            n_calibration_samples=self.n_calibration_samples_,
            score_quantile=self.score_quantile_.copy(),
            diagnostics={
                "rho": self.rho,
                "effective_sample_size": ess,
                "path": "cross_validation",
            },
        )

    def _calibrate_via_loop(
        self,
        histories: list[Series],
        truths: Forecast,
    ) -> CalibrationResult:
        n_cal = len(histories)
        ess = self._effective_sample_size(n_cal)
        self._check_ess(ess)

        predictions = self.forecaster.predict_batch(histories)
        truths_arr = np.asarray(truths, dtype=np.float64)
        self._fit_state(predictions, truths_arr)

        return CalibrationResult(
            n_calibration_samples=self.n_calibration_samples_,
            score_quantile=self.score_quantile_.copy(),
            diagnostics={
                "rho": self.rho,
                "effective_sample_size": ess,
                "path": "loop",
            },
        )

    def _fit_state(self, predictions: Forecast, truths: Forecast) -> None:
        """Run score-function fit, compute scores, set fitted state."""
        self.score_fn.fit(predictions, truths)
        scores = self.score_fn.score(predictions, truths)
        # (n_series, n_cal, horizon)
        self.scores_: NDArray[np.floating] = scores
        self.n_calibration_samples_: int = scores.shape[1]
        self.n_observations_: int = scores.shape[1]
        self.score_quantile_: NDArray[np.floating] = self._compute_weighted_quantile(
            self.scores_, self.rho, self.alpha
        )
        self.is_calibrated_ = True

    def _effective_sample_size(self, n: int) -> float:
        """
        Effective sample size of the weighted calibration set.

        ``ESS = (Σ w_i)^2 / Σ w_i^2`` where ``w_i = ρ^(n - i)`` for
        ``i = 1, ..., n``. ``ESS = n`` when ``ρ = 1`` (all weights equal),
        and approaches ``(1 + ρ) / (1 - ρ)`` as ``n → ∞`` for ``ρ < 1``.
        """
        if n <= 0:
            return 0.0
        if self.rho >= 1.0:
            return float(n)
        # Closed-form: weights are ρ^0, ρ^1, ..., ρ^(n-1) (any ordering yields
        # the same sum-of-powers).
        sum_w = (1.0 - self.rho**n) / (1.0 - self.rho)
        sum_w_sq = (1.0 - self.rho ** (2 * n)) / (1.0 - self.rho**2)
        return float(sum_w * sum_w / sum_w_sq)

    def _check_ess(self, ess: float) -> None:
        min_ess = math.ceil(1.0 / self.alpha)
        if ess < min_ess:
            raise CalibrationError(
                f"Effective sample size {ess:.1f} is below required {min_ess} "
                f"for alpha={self.alpha} with rho={self.rho}. "
                "Increase n_windows, increase rho, or increase alpha."
            )

    # ------------------------------------------------------------------
    # Weighted quantile helper
    # ------------------------------------------------------------------

    def _compute_weighted_quantile(
        self,
        scores: NDArray[np.floating],
        rho: float,
        alpha: float,
    ) -> NDArray[np.floating]:
        """
        Per-(series, horizon) weighted quantile of an observed-score panel.

        Parameters
        ----------
        scores : NDArray, shape (n_series, n_obs, horizon)
            Time-axis is axis 1, oldest first.
        rho : float
            Exponential decay factor.
        alpha : float
            Miscoverage level.

        Returns
        -------
        NDArray, shape (n_series, horizon)

        Notes
        -----
        Weights are chronological exponential: position ``i`` (0-indexed)
        gets ``ρ^(n_obs - 1 - i)``. The newest sample (position
        ``n_obs - 1``) receives weight ``1``; the oldest (position ``0``)
        receives ``ρ^(n_obs - 1)``.

        The threshold target ``(1 - α) (W + 1) / W`` is clipped to
        ``[0, 1]``. For very small ``W`` (or large ``α``) the unclipped
        target may exceed ``1`` (degenerate case) and we saturate at the
        largest observed score. As ``W → ∞`` the target tends to ``1 - α``
        (the split-CP threshold); for finite ``W`` it is slightly above
        ``1 - α``, matching the standard split-CP rank
        ``⌈(1 - α)(n + 1)⌉`` when ``ρ = 1``.

        The nested Python loop mirrors :meth:`AdaptiveConformalInference._per_cell_quantile`.
        Vectorisation via ``np.argsort`` along axis 1 plus broadcast
        ``cumsum``/``searchsorted`` is a known optimisation, deferred until
        profiling flags it.
        """
        n_series, n_obs, horizon = scores.shape

        i = np.arange(n_obs)
        weights = rho ** (n_obs - 1 - i)
        # (n_obs,)
        W = float(weights.sum())
        target = (1.0 - alpha) * (W + 1.0) / W
        target = max(min(target, 1.0), 0.0)

        out = np.empty((n_series, horizon), dtype=np.float64)
        for s in range(n_series):
            for h in range(horizon):
                cell_scores = scores[s, :, h]
                order = np.argsort(cell_scores)
                sorted_scores = cell_scores[order]
                sorted_weights = weights[order]
                cumw = np.cumsum(sorted_weights) / W
                idx = int(np.searchsorted(cumw, target, side="left"))
                if idx >= n_obs:
                    out[s, h] = float(sorted_scores[-1])
                else:
                    out[s, h] = float(sorted_scores[idx])
        return out

    # ------------------------------------------------------------------
    # Predict / update
    # ------------------------------------------------------------------

    def predict(self, history: Series) -> PredictionResult:
        """
        Produce a NexCP-calibrated point forecast and prediction interval.

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
        interval = self.score_fn.invert(point, self.score_quantile_)
        return PredictionResult(point=point, interval=interval, alpha=self.alpha)

    def update(self, prediction: Forecast, truth: Forecast) -> None:
        """
        Append the realized score and recompute the weighted-quantile threshold.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, 1, horizon)
        truth : Forecast, shape (n_series, 1, horizon)

        Raises
        ------
        CalibrationError
            If :meth:`calibrate` has not been called.
        ValueError
            If ``prediction`` or ``truth`` does not have the expected shape.
        """
        if not self.is_calibrated_:
            raise CalibrationError("update() called before calibrate(). Call calibrate() first.")

        n_series, _, horizon = self.scores_.shape
        prediction_arr, truth_arr = _validate_online_shapes(prediction, truth, n_series, horizon)

        new_score = self.score_fn.score(prediction_arr, truth_arr)
        self.scores_ = np.concatenate([self.scores_, new_score], axis=1)
        self.n_observations_ += 1
        self.score_quantile_ = self._compute_weighted_quantile(self.scores_, self.rho, self.alpha)
