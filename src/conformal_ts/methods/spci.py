"""Sequential Predictive Conformal Inference (SPCI)."""

from __future__ import annotations

import math
from collections.abc import Callable

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
from ..nonconformity.signed import SignedResidual
from ..quantile_regressors.base import QuantileRegressor


def _default_regressor_factory() -> QuantileRegressor:
    """Return a fresh :class:`QRFQuantileRegressor`.

    The import is deferred to call-time so importing :mod:`conformal_ts`
    does not require ``quantile-forest`` to be installed.
    """
    from ..quantile_regressors.qrf import QRFQuantileRegressor

    return QRFQuantileRegressor()


class SequentialPredictiveConformalInference(ConformalMethod):
    """
    Sequential Predictive Conformal Inference (Xu & Xie 2023).

    Replaces the empirical quantile of conformity scores with a learned
    conditional quantile estimator (e.g. Quantile Random Forest) trained
    autoregressively on signed residuals. Given the last ``w`` residuals,
    the regressor predicts the conditional quantile distribution of the next
    residual. The interval is *asymmetric* and chosen to minimise width:

    .. math::

        [\\hat f(X_t) + \\hat Q_t(\\hat\\beta),
         \\hat f(X_t) + \\hat Q_t(1 - \\alpha + \\hat\\beta)]

    where :math:`\\hat\\beta = \\arg\\min_{\\beta \\in [0, \\alpha]}
    (\\hat Q_t(1 - \\alpha + \\beta) - \\hat Q_t(\\beta))`.

    Parameters
    ----------
    forecaster : ForecasterAdapter
        Any adapter — no special capabilities required.
    alpha : float
        Target miscoverage in ``(0, 1)``. Coverage target is ``1 - alpha``.
    window_size : int, default 100
        Number of past residuals used as features for the quantile regressor
        (``w`` in the paper).
    regressor_factory : Callable[[], QuantileRegressor], optional
        Factory returning a fresh :class:`QuantileRegressor` for each
        ``(series, horizon)`` cell. Must be callable with no arguments.
        Defaults to a lazy :class:`QRFQuantileRegressor` factory (requires
        the ``spci`` extra).
    beta_grid_size : int, default 21
        Number of points in the ``β`` grid for the width-minimising
        optimisation. Larger gives a finer search at the cost of
        ``beta_grid_size`` additional quantile-regressor predictions per
        cell per :meth:`predict` call.
    refit_every : int, default 1
        Number of :meth:`update` calls between regressor refits. Setting
        ``> 1`` amortises the refit cost; the regressor state goes stale in
        between, but the residual buffer is still updated and used as the
        feature vector at :meth:`predict` time.
    score : ScoreFunction, optional
        Must be :class:`SignedResidual` (the only score supported in v0.1).
        Defaults to a fresh :class:`SignedResidual`.

    Notes
    -----
    The full residual buffer is kept and used to refit each regressor; the
    paper's "sliding window of most recent T residuals" is a future
    enhancement (tracked as a ``max_residual_history`` parameter idea).

    The per-cell ``β`` loop in :meth:`predict` makes
    ``2 * beta_grid_size * n_series * horizon`` quantile-regressor calls per
    prediction. Each individual call is cheap, but Python loop overhead
    dominates on small panels; vectorising the ``β`` search is a tracked
    optimisation.
    """

    REQUIRED_CAPABILITIES: tuple[type, ...] = ()
    IS_ONLINE: bool = True

    def __init__(
        self,
        forecaster: ForecasterAdapter,
        alpha: float,
        window_size: int = 100,
        regressor_factory: Callable[[], QuantileRegressor] | None = None,
        beta_grid_size: int = 21,
        refit_every: int = 1,
        score: ScoreFunction | None = None,
    ) -> None:
        if score is not None and not isinstance(score, SignedResidual):
            # The interval-construction logic depends on signed scores.
            raise TypeError(
                f"SPCI requires score to be a SignedResidual instance. Got {type(score).__name__}."
            )

        super().__init__(forecaster=forecaster, alpha=alpha, score=score)

        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if beta_grid_size < 2:
            raise ValueError(f"beta_grid_size must be >= 2, got {beta_grid_size}")
        if refit_every < 1:
            raise ValueError(f"refit_every must be >= 1, got {refit_every}")

        factory = regressor_factory if regressor_factory is not None else _default_regressor_factory
        if not callable(factory):
            raise ValueError("regressor_factory must be callable.")
        probe = factory()
        if not isinstance(probe, QuantileRegressor):
            raise TypeError(
                "regressor_factory must return a QuantileRegressor instance. "
                f"Got {type(probe).__name__}."
            )

        self.window_size: int = int(window_size)
        self.regressor_factory: Callable[[], QuantileRegressor] = factory
        self.beta_grid_size: int = int(beta_grid_size)
        self.refit_every: int = int(refit_every)
        self._regressor_class_name: str = type(probe).__name__

    def _default_score(self) -> ScoreFunction:
        return SignedResidual()

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
        Fit the SPCI quantile regressors on a calibration set.

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
            Whether to refit the underlying forecaster between CV windows.

        Returns
        -------
        CalibrationResult
            ``diagnostics["path"]`` is ``"loop"`` or ``"cross_validation"``.
            ``diagnostics["window_size"]`` and
            ``diagnostics["regressor_class"]`` are included for traceability.

        Raises
        ------
        CalibrationError
            If fewer than ``2 * window_size + ceil(1 / alpha)`` calibration
            samples are available.
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

    def _min_required_samples(self) -> int:
        return 2 * self.window_size + math.ceil(1.0 / self.alpha)

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

        min_samples = self._min_required_samples()
        if n_windows < min_samples:
            raise CalibrationError(
                f"Need at least 2 * window_size + ceil(1/alpha) = "
                f"2 * {self.window_size} + {math.ceil(1.0 / self.alpha)} = {min_samples} "
                f"calibration samples for window_size={self.window_size} and "
                f"alpha={self.alpha}, got n_windows={n_windows}."
            )

        predictions, truths = self.forecaster.cross_validate(
            n_windows=n_windows, step_size=step_size, refit=refit
        )
        self._fit_initial_state(predictions, truths)

        return CalibrationResult(
            n_calibration_samples=self.n_calibration_samples_,
            score_quantile=self._compute_score_thresholds(),
            diagnostics={
                "window_size": self.window_size,
                "regressor_class": self._regressor_class_name,
                "beta_grid_size": self.beta_grid_size,
                "refit_every": self.refit_every,
                "path": "cross_validation",
            },
        )

    def _calibrate_via_loop(
        self,
        histories: list[Series],
        truths: Forecast,
    ) -> CalibrationResult:
        n_cal = len(histories)
        min_samples = self._min_required_samples()
        if n_cal < min_samples:
            raise CalibrationError(
                f"Need at least 2 * window_size + ceil(1/alpha) = "
                f"2 * {self.window_size} + {math.ceil(1.0 / self.alpha)} = {min_samples} "
                f"calibration samples for window_size={self.window_size} and "
                f"alpha={self.alpha}, got {n_cal}."
            )

        predictions = self.forecaster.predict_batch(histories)
        truths_arr = np.asarray(truths, dtype=np.float64)
        self._fit_initial_state(predictions, truths_arr)

        return CalibrationResult(
            n_calibration_samples=self.n_calibration_samples_,
            score_quantile=self._compute_score_thresholds(),
            diagnostics={
                "window_size": self.window_size,
                "regressor_class": self._regressor_class_name,
                "beta_grid_size": self.beta_grid_size,
                "refit_every": self.refit_every,
                "path": "loop",
            },
        )

    def _fit_initial_state(self, predictions: Forecast, truths: Forecast) -> None:
        """Compute the calibration residual buffer and fit all per-cell regressors."""
        self.score_fn.fit(predictions, truths)
        residuals = self.score_fn.score(predictions, truths)
        # residuals: (n_series, n_cal, horizon)
        self.residuals_: NDArray[np.floating] = np.asarray(residuals, dtype=np.float64)
        self.n_calibration_samples_: int = int(residuals.shape[1])
        self.n_observations_: int = int(residuals.shape[1])
        self._steps_since_refit_: int = 0
        self._fit_all_regressors()
        self.is_calibrated_ = True

    # ------------------------------------------------------------------
    # Quantile-regressor fitting
    # ------------------------------------------------------------------

    def _fit_all_regressors(self) -> None:
        """Fit one :class:`QuantileRegressor` per ``(series, horizon)`` cell."""
        n_series, n_obs, horizon = self.residuals_.shape
        w = self.window_size
        if n_obs <= w:
            raise CalibrationError(
                f"Need n_observations > window_size to build training pairs, "
                f"got n_observations={n_obs}, window_size={w}."
            )

        regressors: dict[tuple[int, int], QuantileRegressor] = {}
        for s in range(n_series):
            for h in range(horizon):
                cell_residuals = self.residuals_[s, :, h]  # (n_obs,)
                # Build autoregressive training set via stride tricks.
                X_train = np.lib.stride_tricks.sliding_window_view(
                    cell_residuals[:-1], window_shape=w
                )
                # sliding_window_view returns a view; make a contiguous copy
                # so the underlying regressor can store/forward it safely.
                X_train = np.ascontiguousarray(X_train, dtype=np.float64)
                y_train = np.ascontiguousarray(cell_residuals[w:], dtype=np.float64)

                reg = self.regressor_factory()
                reg.fit(X_train, y_train)
                regressors[(s, h)] = reg

        self.quantile_regressors_: dict[tuple[int, int], QuantileRegressor] = regressors

    # ------------------------------------------------------------------
    # Predict / update
    # ------------------------------------------------------------------

    def _optimize_beta(
        self,
        regressor: QuantileRegressor,
        X_query: NDArray[np.floating],
    ) -> tuple[float, float, float]:
        """Return ``(β̂, lower_offset, upper_offset)`` minimising interval width.

        Parameters
        ----------
        regressor : QuantileRegressor
            Fitted per-cell quantile regressor.
        X_query : NDArray, shape (1, window_size)
            The feature vector for the next step (most recent ``w`` residuals).

        Returns
        -------
        beta_hat : float
        lower_offset : float
            ``Q̂(β̂)`` — typically negative.
        upper_offset : float
            ``Q̂(1 - α + β̂)``.
        """
        beta_grid = np.linspace(0.0, self.alpha, self.beta_grid_size)
        best_width = np.inf
        best_lower = 0.0
        best_upper = 0.0
        best_beta = 0.0
        eps = 1e-6
        for beta in beta_grid:
            q_lower_level = max(float(beta), eps)
            q_upper_level = min(1.0 - self.alpha + float(beta), 1.0 - eps)
            lower = float(regressor.predict_quantile(X_query, q_lower_level)[0])
            upper = float(regressor.predict_quantile(X_query, q_upper_level)[0])
            width = upper - lower
            if width < best_width:
                best_width = width
                best_lower = lower
                best_upper = upper
                best_beta = float(beta)
        return best_beta, best_lower, best_upper

    def _compute_score_thresholds(self) -> NDArray[np.floating]:
        """Return ``(n_series, horizon, 2)`` array of optimal ``(lower, upper)`` offsets."""
        n_series, _, horizon = self.residuals_.shape
        w = self.window_size
        thresholds = np.empty((n_series, horizon, 2), dtype=np.float64)
        for s in range(n_series):
            for h in range(horizon):
                cell_residuals = self.residuals_[s, :, h]
                X_query = cell_residuals[-w:].reshape(1, w)
                _, lower, upper = self._optimize_beta(self.quantile_regressors_[(s, h)], X_query)
                thresholds[s, h, 0] = lower
                thresholds[s, h, 1] = upper
        return thresholds

    def predict(self, history: Series) -> PredictionResult:
        """
        Produce an SPCI-calibrated point forecast and asymmetric interval.

        Parameters
        ----------
        history : Series, shape (n_series, T)

        Returns
        -------
        PredictionResult
            ``point`` has shape ``(n_series, 1, horizon)``;
            ``interval`` has shape ``(n_series, 1, horizon, 2)``.

        Raises
        ------
        CalibrationError
            If called before :meth:`calibrate`.
        """
        if not self.is_calibrated_:
            raise CalibrationError("predict() called before calibrate(). Call calibrate() first.")

        point: Forecast = self.forecaster.predict(history)
        score_thresholds = self._compute_score_thresholds()
        interval: Interval = self.score_fn.invert(point, score_thresholds)

        return PredictionResult(point=point, interval=interval, alpha=self.alpha)

    def update(self, prediction: Forecast, truth: Forecast) -> None:
        """
        Append the new residual and, every ``refit_every`` calls, refit
        the per-cell quantile regressors.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, 1, horizon)
            The point forecast for this step.
        truth : Forecast, shape (n_series, 1, horizon)
            Realized values for the same horizon.

        Raises
        ------
        CalibrationError
            If :meth:`calibrate` has not been called.
        ValueError
            If ``prediction`` and ``truth`` do not share shape
            ``(n_series, 1, horizon)``.
        """
        if not self.is_calibrated_:
            raise CalibrationError("update() called before calibrate(). Call calibrate() first.")

        prediction_arr = np.asarray(prediction, dtype=np.float64)
        truth_arr = np.asarray(truth, dtype=np.float64)

        n_series, _, horizon = self.residuals_.shape
        expected_shape = (n_series, 1, horizon)
        if prediction_arr.shape != expected_shape:
            raise ValueError(
                f"prediction must have shape {expected_shape}, got {prediction_arr.shape}."
            )
        if truth_arr.shape != expected_shape:
            raise ValueError(f"truth must have shape {expected_shape}, got {truth_arr.shape}.")

        new_residual = self.score_fn.score(prediction_arr, truth_arr)
        # shape: (n_series, 1, horizon)
        self.residuals_ = np.concatenate([self.residuals_, new_residual], axis=1)
        self.n_observations_ += 1
        self._steps_since_refit_ += 1

        if self._steps_since_refit_ >= self.refit_every:
            self._fit_all_regressors()
            self._steps_since_refit_ = 0
