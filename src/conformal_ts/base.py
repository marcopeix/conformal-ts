"""
Base abstractions for conformal-ts.

Three primary contracts define the package:

- ForecasterAdapter: wraps any forecasting library into a uniform interface
- ScoreFunction: maps (prediction, truth) pairs to nonconformity scores
- ConformalMethod: the user-facing object that calibrates and produces intervals

Capability flags are expressed as mixins (see `capabilities.py`). Methods declare
what mixins their adapter must inherit from; this is checked at construction time.

All array shapes are panel-aware: `n_series` is the leading axis everywhere.
Single-series users pass `n_series=1`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# -----------------------------------------------------------------------------
# Type aliases. These are the only shapes that cross module boundaries.
# -----------------------------------------------------------------------------

# A panel of time series. Shape: (n_series, T).
# Series may have different effective lengths via NaN-padding on the left.
Series = NDArray[np.floating]

# Predictions or truths over multiple horizons across a panel.
# Shape: (n_series, n_samples, horizon). All three axes always present.
Forecast = NDArray[np.floating]

# Prediction intervals. Shape: (n_series, n_samples, horizon, 2).
# Last axis is (lower, upper).
Interval = NDArray[np.floating]


# -----------------------------------------------------------------------------
# ForecasterAdapter — abstract base
# -----------------------------------------------------------------------------


class ForecasterAdapter(ABC):
    """
    Uniform interface over any forecasting library.

    Concrete adapters live in `adapters/`. They convert framework-specific
    objects to/from numpy arrays so that ConformalMethod implementations
    never see framework types.

    Capabilities are declared via mixins in `capabilities.py`. An adapter that
    can produce quantiles inherits `SupportsQuantiles`. An adapter that can
    refit on demand inherits `SupportsRefit`. Etc.

    The minimum contract is `predict()`. Everything else is opt-in via mixins.
    """

    def __init__(self, horizon: int, n_series: int) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if n_series < 1:
            raise ValueError(f"n_series must be >= 1, got {n_series}")
        self.horizon = horizon
        self.n_series = n_series

    @abstractmethod
    def predict(self, history: Series) -> Forecast:
        """
        Produce a point forecast for the next `self.horizon` steps for every series.

        Parameters
        ----------
        history : Series, shape (n_series, T)
            Past observations for each series, ending immediately before the
            forecast window. T may be any value >= the model's required minimum.

        Returns
        -------
        Forecast, shape (n_series, 1, horizon)
            Point forecast over the horizon for each series. The middle axis
            is 1 because this call produces one forecast per series.
        """
        ...

    def predict_batch(self, histories: list[Series]) -> Forecast:
        """
        Vectorized prediction over multiple history windows.

        Parameters
        ----------
        histories : list of Series, each shape (n_series, T)

        Returns
        -------
        Forecast, shape (n_series, len(histories), horizon)

        Default implementation calls `predict` in a loop and stacks results.
        Adapters where the underlying library supports batch inference should
        override this for performance.
        """
        out = [self.predict(h) for h in histories]  # each (n_series, 1, horizon)
        return np.concatenate(out, axis=1)


# -----------------------------------------------------------------------------
# ScoreFunction
# -----------------------------------------------------------------------------


class ScoreFunction(ABC):
    """
    Maps (predictions, truths) to nonconformity scores.

    Stateless except for parameters fitted in `fit`. All operations are
    panel-aware: the leading axis is `n_series`.
    """

    @abstractmethod
    def score(self, prediction: Forecast, truth: Forecast) -> NDArray[np.floating]:
        """
        Compute nonconformity scores.

        Parameters
        ----------
        prediction, truth : Forecast, shape (n_series, n_samples, horizon)

        Returns
        -------
        scores : NDArray, shape (n_series, n_samples, horizon)
        """
        ...

    @abstractmethod
    def invert(
        self,
        prediction: Forecast,
        score_threshold: NDArray[np.floating],
    ) -> Interval:
        """
        Given a point forecast and a score threshold, return the prediction interval.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, n_samples, horizon)
        score_threshold : NDArray
            Either scalar, or shape (n_series,), or shape (n_series, horizon).
            Will be broadcast against `prediction` along its last two axes.

        Returns
        -------
        Interval, shape (n_series, n_samples, horizon, 2)
        """
        ...

    def fit(self, prediction: Forecast, truth: Forecast) -> None:
        """
        Fit any internal parameters of the score function (e.g. scale
        estimates for normalised residuals). Default: no-op.

        Called by :meth:`ConformalMethod.calibrate` **before** the first
        call to :meth:`score` on the calibration data, so implementations
        can assume that ``prediction`` and ``truth`` are the full
        calibration arrays.

        Parameters
        ----------
        prediction : Forecast, shape (n_series, n_samples, horizon)
        truth : Forecast, shape (n_series, n_samples, horizon)
        """
        return None


# -----------------------------------------------------------------------------
# ConformalMethod
# -----------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Returned by ``ConformalMethod.calibrate``.

    Attributes
    ----------
    n_calibration_samples : int
        Number of calibration windows used.
    score_quantile : NDArray[np.floating]
        Nonconformity-score quantile(s) used for interval construction.
        Shape is method-dependent:

        * Split CP: ``(n_series, horizon)`` — one quantile per
          (series, horizon step) pair.
        * Online methods (ACI, AgACI, …): may be a scalar or
          ``(n_series,)`` that evolves via ``update()``.
    diagnostics : dict
        Method-specific diagnostics (e.g. quantile level, learning rate).
    """

    n_calibration_samples: int
    score_quantile: NDArray[np.floating]
    diagnostics: dict


@dataclass
class PredictionResult:
    """Returned by `ConformalMethod.predict`."""

    point: Forecast  # shape (n_series, n_samples, horizon)
    interval: Interval  # shape (n_series, n_samples, horizon, 2)
    alpha: float


class ConformalMethod(ABC):
    """
    A conformal prediction wrapper around a forecaster.

    Lifecycle:
        1. Construct with a forecaster adapter and target alpha.
        2. Call `calibrate(histories, truths)` once on a calibration window.
        3. Call `predict(history)` repeatedly to get intervals.
        4. For online methods, call `update(truth)` after each step.

    Subclasses declare adapter capability requirements via the
    `REQUIRED_CAPABILITIES` class attribute. The constructor verifies that
    the supplied adapter inherits from each required mixin.

    Invariants
    ----------
    - `alpha` is in (0, 1). Coverage target is 1 - alpha.
    - `calibrate` must be called before `predict`. Raises `CalibrationError` otherwise.
    - `predict` is deterministic given fitted state and input history.
    - `update` is a no-op for offline methods. Online methods (IS_ONLINE = True)
      override it to update internal state.
    - All shape conventions are panel-aware (n_series leading).
    """

    # Subclasses override these.
    REQUIRED_CAPABILITIES: tuple[type, ...] = ()
    IS_ONLINE: bool = False

    def __init__(
        self,
        forecaster: ForecasterAdapter,
        alpha: float,
        score: ScoreFunction | None = None,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self._check_capabilities(forecaster)

        self.forecaster = forecaster
        self.alpha = alpha
        self.score_fn = score if score is not None else self._default_score()
        self._is_calibrated = False

    def _check_capabilities(self, forecaster: ForecasterAdapter) -> None:
        for cap in self.REQUIRED_CAPABILITIES:
            if not isinstance(forecaster, cap):
                raise UnsupportedCapability(
                    f"{type(self).__name__} requires an adapter implementing "
                    f"{cap.__name__}. Got {type(forecaster).__name__}."
                )

    @abstractmethod
    def _default_score(self) -> ScoreFunction:
        """Each method declares its default score function."""
        ...

    @abstractmethod
    def calibrate(
        self,
        histories: list[Series],
        truths: Forecast,
    ) -> CalibrationResult:
        """
        Fit the conformal correction on a calibration set.

        Implementations **must** call ``self.score_fn.fit(predictions, truths)``
        before ``self.score_fn.score(...)`` so that score functions which need
        fitted parameters (e.g. normalised residuals) are initialised.

        Parameters
        ----------
        histories : list of Series, each shape (n_series, T)
            History available at each calibration timestep.
        truths : Forecast, shape (n_series, len(histories), horizon)
            Ground truth values for each calibration history and horizon.

        Returns
        -------
        CalibrationResult
        """
        ...

    @abstractmethod
    def predict(self, history: Series) -> PredictionResult:
        """
        Produce a calibrated point forecast and prediction interval.

        Parameters
        ----------
        history : Series, shape (n_series, T)

        Returns
        -------
        PredictionResult
        """
        ...

    def update(self, truth: Forecast) -> None:
        """
        Online update step. No-op for offline methods.

        Parameters
        ----------
        truth : Forecast, shape (n_series, 1, horizon)
        """
        return None


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class ConformalTSError(Exception):
    """Base exception for the package."""


class CalibrationError(ConformalTSError):
    """Raised when predict() is called before calibrate(), or calibration data is invalid."""


class UnsupportedCapability(ConformalTSError):
    """Raised when a method requires an adapter capability that isn't supported."""
