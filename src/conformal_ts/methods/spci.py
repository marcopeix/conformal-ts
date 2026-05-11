"""Sequential Predictive Conformal Inference (Xu & Xie 2023)."""

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
from ..quantile_regressors.qrf import QRFQuantileRegressor
from ._online_helpers import _validate_online_shapes


def _default_regressor_factory() -> QuantileRegressor:
    return QRFQuantileRegressor()


class SequentialPredictiveConformalInference(ConformalMethod):
    """
    Sequential Predictive Conformal Inference (Xu & Xie 2023).

    Replaces the empirical quantile of conformity scores with a learned
    conditional quantile estimator (Quantile Random Forest by default)
    trained autoregressively on residuals. Intervals are asymmetric:
    produced by a 1-D optimization over a width-minimizing offset ``beta``.

    Algorithm
    ---------
    Given a sequence of signed residuals ``r_1, ..., r_n`` from a fitted point
    predictor, a quantile regressor is trained to predict the conditional
    distribution of ``r_t`` from the previous ``w`` residuals
    ``(r_{t-w}, ..., r_{t-1})``. At prediction time, the interval becomes

        [point + Q(beta_hat), point + Q(1 - alpha + beta_hat)]

    where ``Q`` is the trained regressor's quantile prediction at the query
    feature ``(r_{n-w+1}, ..., r_n)`` and ``beta_hat`` minimizes the interval
    width ``Q(1 - alpha + beta) - Q(beta)`` over ``beta in [0, alpha]``.

    Because the offsets are asymmetric, SPCI uses :class:`SignedResidual` as
    its score function (not :class:`AbsoluteResidual`).

    Parameters
    ----------
    forecaster : ForecasterAdapter
        Any adapter — no special capabilities required.
    alpha : float
        Target miscoverage level in ``(0, 1)``. The target coverage is
        ``1 - alpha``.
    window_size : int, default 100
        Number of past residuals used as features for the quantile regressor
        (``w`` in the paper).
    regressor_factory : Callable[[], QuantileRegressor], optional
        Factory function returning a fresh :class:`QuantileRegressor` for
        each ``(series, horizon)`` cell. Defaults to a factory producing
        :class:`QRFQuantileRegressor` instances. The factory is called once
        per cell at fit time; each cell gets an independent regressor
        instance.
    beta_grid_size : int, default 21
        Number of points in the ``beta`` grid for the width-minimizing
        optimization. Larger gives finer optimization at the cost of more
        quantile predictions per :meth:`predict` call.
    refit_every : int, default 1
        Number of :meth:`update` calls between regressor refits. Setting
        ``> 1`` amortizes refit cost; the regressor state goes stale between
        refits but the residual buffer is still updated.
    score : ScoreFunction or None
        Must be :class:`SignedResidual` (default). SPCI's asymmetric interval
        logic depends on signed scores; other scores are rejected at
        construction.

    Notes
    -----
    The residual buffer keeps the full history rather than a sliding window
    of the last ``T`` residuals — the paper's sliding-window pruning could be
    added later via a ``max_residual_history`` parameter if memory or
    staleness becomes a concern. With the full history the regressor always
    has the maximum training data available.

    Online lifecycle
    ----------------
    After ``calibrate``, callers alternate :meth:`predict` and
    :meth:`update`. Each :meth:`update` appends a new signed residual to the
    per-cell buffer and refits the regressor when ``refit_every`` steps have
    elapsed since the last fit. The caller passes the same ``point`` they
    got back from :class:`PredictionResult`, so there is no hidden state
    coupling between :meth:`predict` and :meth:`update`.
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
        super().__init__(forecaster=forecaster, alpha=alpha, score=score)

        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if beta_grid_size < 2:
            raise ValueError(f"beta_grid_size must be >= 2, got {beta_grid_size}")
        if refit_every < 1:
            raise ValueError(f"refit_every must be >= 1, got {refit_every}")

        # SPCI's asymmetric-offset construction is built on top of
        # SignedResidual; reject any other score function at construction
        # time so the failure mode is loud and explicit, not a runtime
        # shape mismatch deep inside ``invert``.
        if not isinstance(self.score_fn, SignedResidual):
            raise TypeError(
                f"SequentialPredictiveConformalInference requires SignedResidual "
                f"as the score function; got {type(self.score_fn).__name__}."
            )

        factory: Callable[[], QuantileRegressor] = (
            regressor_factory if regressor_factory is not None else _default_regressor_factory
        )
        if not callable(factory):
            raise TypeError(f"regressor_factory must be callable, got {type(factory).__name__}.")
        # Probe the factory to catch misconfiguration early. The probe
        # instance is discarded; each cell gets a fresh regressor at fit
        # time.
        test_instance = factory()
        if not isinstance(test_instance, QuantileRegressor):
            raise TypeError(
                f"regressor_factory must return a QuantileRegressor, "
                f"got {type(test_instance).__name__}."
            )

        self.window_size: int = int(window_size)
        self.beta_grid_size: int = int(beta_grid_size)
        self.refit_every: int = int(refit_every)
        self.regressor_factory: Callable[[], QuantileRegressor] = factory

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
        Fit the SPCI residual buffer and per-cell quantile regressors.

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
            ``diagnostics["path"]`` is ``"loop"`` or ``"cross_validation"``.
            ``score_quantile`` is empty for SPCI — the method does not have a
            scalar threshold; quantile predictions are made on demand by the
            per-cell regressors at :meth:`predict` time.

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
        # Need at least ``window_size`` residuals to form a single training
        # pair, and ``ceil(1 / alpha)`` to make a quantile at level alpha
        # meaningful. Using ``2 * w`` as a practical training-stability floor.
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

        min_required = self._min_required_samples()
        if n_windows < min_required:
            raise CalibrationError(
                f"n_windows={n_windows} is below the minimum {min_required} required "
                f"for window_size={self.window_size} and alpha={self.alpha}. "
                f"SPCI needs at least 2*w + ceil(1/alpha) calibration windows."
            )

        predictions, truths_arr = self.forecaster.cross_validate(
            n_windows=n_windows, step_size=step_size, refit=refit
        )
        self._fit_state(predictions, truths_arr)

        return CalibrationResult(
            n_calibration_samples=self.n_calibration_samples_,
            score_quantile=np.array([], dtype=np.float64),
            diagnostics={
                "window_size": self.window_size,
                "regressor_class": type(self.regressor_factory()).__name__,
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
        min_required = self._min_required_samples()
        if n_cal < min_required:
            raise CalibrationError(
                f"n_windows={n_cal} is below the minimum {min_required} required "
                f"for window_size={self.window_size} and alpha={self.alpha}. "
                f"SPCI needs at least 2*w + ceil(1/alpha) calibration windows."
            )

        predictions = self.forecaster.predict_batch(histories)
        truths_arr = np.asarray(truths, dtype=np.float64)
        self._fit_state(predictions, truths_arr)

        return CalibrationResult(
            n_calibration_samples=self.n_calibration_samples_,
            score_quantile=np.array([], dtype=np.float64),
            diagnostics={
                "window_size": self.window_size,
                "regressor_class": type(self.regressor_factory()).__name__,
                "beta_grid_size": self.beta_grid_size,
                "refit_every": self.refit_every,
                "path": "loop",
            },
        )

    def _fit_state(self, predictions: Forecast, truths: Forecast) -> None:
        """Populate residual buffer and fit per-cell regressors."""
        self.score_fn.fit(predictions, truths)
        self.residuals_: NDArray[np.floating] = self.score_fn.score(predictions, truths)
        # residuals_: (n_series, n_observations, horizon)
        self._fit_all_regressors()

        self.n_calibration_samples_: int = self.residuals_.shape[1]
        self.n_observations_: int = self.residuals_.shape[1]
        self._steps_since_refit_: int = 0
        self.is_calibrated_ = True

    def _fit_all_regressors(self) -> None:
        """Fit one :class:`QuantileRegressor` per ``(series, horizon)`` cell."""
        n_series, n_obs, horizon = self.residuals_.shape
        w = self.window_size
        n_train = n_obs - w

        if n_train < 1:
            raise CalibrationError(
                f"Insufficient residuals after windowing: have {n_obs} residuals, "
                f"need at least {w + 1} for window_size={w}."
            )

        self.quantile_regressors_: dict[tuple[int, int], QuantileRegressor] = {}
        for s in range(n_series):
            for h in range(horizon):
                cell_residuals = self.residuals_[s, :, h]
                # Vectorized AR feature construction. ``cell_residuals[:-1]``
                # has shape (n_obs - 1,); sliding_window_view with window=w
                # produces (n_train, w). ``cell_residuals[w:]`` of shape
                # (n_train,) is the aligned target.
                X_train = np.lib.stride_tricks.sliding_window_view(
                    cell_residuals[:-1], window_shape=w
                )
                y_train = cell_residuals[w:]

                reg = self.regressor_factory()
                reg.fit(X_train, y_train)
                self.quantile_regressors_[(s, h)] = reg

    # ------------------------------------------------------------------
    # Predict / update
    # ------------------------------------------------------------------

    def predict(self, history: Series) -> PredictionResult:
        """
        Produce an SPCI-calibrated point forecast and asymmetric interval.

        Parameters
        ----------
        history : Series, shape (n_series, T)

        Returns
        -------
        PredictionResult
            ``point`` has shape ``(n_series, 1, horizon)``.
            ``interval`` has shape ``(n_series, 1, horizon, 2)``. The lower
            and upper offsets from ``point`` are not symmetric in general.

        Raises
        ------
        CalibrationError
            If called before :meth:`calibrate`.
        """
        if not self.is_calibrated_:
            raise CalibrationError("predict() called before calibrate(). Call calibrate() first.")

        point: Forecast = self.forecaster.predict(history)
        w = self.window_size
        n_series, _, horizon = self.residuals_.shape

        score_thresholds = np.empty((n_series, horizon, 2), dtype=np.float64)
        for s in range(n_series):
            for h in range(horizon):
                X_query = self.residuals_[s, -w:, h].reshape(1, w)
                regressor = self.quantile_regressors_[(s, h)]
                lower_offset, upper_offset = self._optimize_beta(regressor, X_query)
                score_thresholds[s, h, 0] = lower_offset
                score_thresholds[s, h, 1] = upper_offset

        interval: Interval = self.score_fn.invert(point, score_thresholds)
        return PredictionResult(point=point, interval=interval, alpha=self.alpha)

    def _optimize_beta(
        self,
        regressor: QuantileRegressor,
        X_query: NDArray[np.floating],
    ) -> tuple[float, float]:
        """
        Find ``beta`` minimizing interval width over the grid ``[0, alpha]``.

        Returns
        -------
        (lower_offset, upper_offset) : tuple of floats
            The quantile predictions at the optimal ``beta``:
            ``lower_offset = Q(beta_hat)``, ``upper_offset = Q(1 - alpha + beta_hat)``.
        """
        beta_grid = np.linspace(0.0, self.alpha, self.beta_grid_size)
        # Clip endpoints away from exactly 0 or 1 to avoid quantile edge
        # cases in regressors that mishandle them.
        eps = 1e-6
        best_width = np.inf
        best_lower = 0.0
        best_upper = 0.0
        for beta in beta_grid:
            q_lo_level = max(float(beta), eps)
            q_hi_level = min(float(1.0 - self.alpha + beta), 1.0 - eps)
            lower = float(regressor.predict_quantile(X_query, q_lo_level)[0])
            upper = float(regressor.predict_quantile(X_query, q_hi_level)[0])
            width = upper - lower
            if width < best_width:
                best_width = width
                best_lower = lower
                best_upper = upper
        return best_lower, best_upper

    def update(self, prediction: Forecast, truth: Forecast) -> None:
        """
        Append a new residual and (every ``refit_every`` steps) refit regressors.

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
            If ``prediction`` or ``truth`` does not have shape
            ``(n_series, 1, horizon)``.

        Notes
        -----
        Each update appends a new signed residual. The quantile regressors
        are refit every ``refit_every`` calls; setting ``refit_every > 1``
        amortizes refit cost at the price of using a stale regressor between
        refits.
        """
        if not self.is_calibrated_:
            raise CalibrationError("update() called before calibrate(). Call calibrate() first.")

        n_series, _, horizon = self.residuals_.shape
        prediction_arr, truth_arr = _validate_online_shapes(prediction, truth, n_series, horizon)

        new_residual = self.score_fn.score(prediction_arr, truth_arr)
        # (n_series, 1, horizon)
        self.residuals_ = np.concatenate([self.residuals_, new_residual], axis=1)
        self.n_observations_ += 1
        self._steps_since_refit_ += 1

        if self._steps_since_refit_ >= self.refit_every:
            self._fit_all_regressors()
            self._steps_since_refit_ = 0
