"""End-to-end holdout evaluation: ``evaluate`` and the ``Report`` dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..base import CalibrationError, ConformalMethod, Forecast, Series
from .coverage import (
    coverage_by_horizon,
    coverage_by_series,
    coverage_per_cell,
    marginal_coverage,
)
from .method_diagnostics import method_state
from .scoring import mean_interval_width, winkler_score


@dataclass
class Report:
    """
    Output of :func:`evaluate`.

    Carries Layer 1 (generic) coverage and scoring metrics plus Layer 2
    (method-specific) state, along with the raw intervals/truths/points for
    callers who want to run further analysis.
    """

    # Identification
    method_name: str
    alpha: float
    n_holdout_samples: int

    # Layer 1: coverage
    marginal_coverage: float
    coverage_by_horizon: NDArray[np.floating]  # shape (horizon,)
    coverage_by_series: NDArray[np.floating]  # shape (n_series,)
    coverage_per_cell: NDArray[np.floating]  # shape (n_series, horizon)

    # Layer 1: scoring
    mean_interval_width: float
    mean_winkler_score: float

    # Layer 2: method-specific
    method_state: dict[str, Any] = field(default_factory=dict)

    # Raw arrays (for users wanting custom analysis)
    intervals: NDArray[np.floating] = field(
        default_factory=lambda: np.empty((0, 0, 0, 2), dtype=np.float64)
    )
    truths: NDArray[np.floating] = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=np.float64)
    )
    points: NDArray[np.floating] = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=np.float64)
    )

    def summary(self) -> str:
        """Human-readable, multi-line summary for notebooks."""
        target = 1.0 - self.alpha
        cov_h = ", ".join(f"{v:.3f}" for v in self.coverage_by_horizon)

        lines = [
            f"Report — {self.method_name}",
            f"  target coverage: {target:.3f}",
            f"  marginal coverage: {self.marginal_coverage:.3f}",
            f"  coverage by horizon: [{cov_h}]",
            f"  mean width: {self.mean_interval_width:.3f}",
            f"  mean Winkler score: {self.mean_winkler_score:.3f}",
            f"  n_holdout_samples: {self.n_holdout_samples}",
        ]

        if "n_observations" in self.method_state:
            lines.append(f"  n_observations: {self.method_state['n_observations']}")
        if "alpha_t_minus_alpha" in self.method_state:
            drift = self.method_state["alpha_t_minus_alpha"]
            lines.append(
                f"  alpha_t drift: max {float(drift.max()):.3f}, min {float(drift.min()):.3f}"
            )
        if "effective_sample_size" in self.method_state:
            ess = self.method_state["effective_sample_size"]
            lines.append(f"  effective sample size: {ess:.1f}")
        if "n_regressors" in self.method_state:
            lines.append(
                f"  n_regressors: {self.method_state['n_regressors']} "
                f"({self.method_state.get('regressor_class', '?')})"
            )

        return "\n".join(lines)


def evaluate(
    method: ConformalMethod,
    holdout_histories: list[Series],
    holdout_truths: Forecast,
    update_online: bool = True,
) -> Report:
    """
    Run a conformal method on a holdout set and compute Mode 2 diagnostics.

    For each ``(history, truth)`` pair, calls ``method.predict(history)``. For
    online methods (``IS_ONLINE = True``), also calls
    ``method.update(prediction, truth)`` between iterations unless
    ``update_online=False``.

    Parameters
    ----------
    method : ConformalMethod
        Must be calibrated.
    holdout_histories : list of Series
        Each ``Series`` has shape ``(n_series, T_i)``. The list length
        determines ``n_holdout_samples``.
    holdout_truths : Forecast, shape (n_series, n_holdout_samples, horizon)
        Ground truth for each history's forecast horizon.
    update_online : bool, default True
        If ``method.IS_ONLINE``, whether to call ``method.update()`` after
        each prediction. Default ``True`` simulates real online deployment;
        set ``False`` to evaluate the calibration-time snapshot of the
        method.

    Returns
    -------
    Report

    Raises
    ------
    CalibrationError
        If ``method`` has not been calibrated.
    ValueError
        If ``holdout_truths`` is not 3-D, or if the number of samples does
        not match ``len(holdout_histories)``.
    """
    if not getattr(method, "is_calibrated_", False):
        raise CalibrationError(
            "evaluate() called on an uncalibrated method. Call calibrate() first."
        )

    truths_arr = np.asarray(holdout_truths, dtype=np.float64)
    if truths_arr.ndim != 3:
        raise ValueError(
            "holdout_truths must have shape (n_series, n_holdout_samples, horizon); "
            f"got {truths_arr.shape}."
        )
    n_holdout = len(holdout_histories)
    if truths_arr.shape[1] != n_holdout:
        raise ValueError(
            "len(holdout_histories) must equal holdout_truths.shape[1]; "
            f"got {n_holdout} histories and holdout_truths {truths_arr.shape}."
        )

    point_list: list[NDArray[np.floating]] = []
    interval_list: list[NDArray[np.floating]] = []

    is_online = bool(getattr(method, "IS_ONLINE", False))

    for i, history in enumerate(holdout_histories):
        result = method.predict(history)
        point_list.append(np.asarray(result.point, dtype=np.float64))
        interval_list.append(np.asarray(result.interval, dtype=np.float64))

        if is_online and update_online:
            truth_i = truths_arr[:, i : i + 1, :]
            method.update(result.point, truth_i)

    # Concatenate along the sample axis. Each predict() returns
    # (n_series, 1, horizon) and (n_series, 1, horizon, 2).
    points_arr = np.concatenate(point_list, axis=1)
    intervals_arr = np.concatenate(interval_list, axis=1)

    cov_marginal = marginal_coverage(intervals_arr, truths_arr)
    cov_h = coverage_by_horizon(intervals_arr, truths_arr)
    cov_s = coverage_by_series(intervals_arr, truths_arr)
    cov_cell = coverage_per_cell(intervals_arr, truths_arr)

    width_per_cell = mean_interval_width(intervals_arr)
    mean_width = float(width_per_cell.mean())
    mean_winkler = float(winkler_score(intervals_arr, truths_arr, method.alpha).mean())

    state = method_state(method)

    return Report(
        method_name=type(method).__name__,
        alpha=float(method.alpha),
        n_holdout_samples=n_holdout,
        marginal_coverage=cov_marginal,
        coverage_by_horizon=cov_h,
        coverage_by_series=cov_s,
        coverage_per_cell=cov_cell,
        mean_interval_width=mean_width,
        mean_winkler_score=mean_winkler,
        method_state=state,
        intervals=intervals_arr,
        truths=truths_arr,
        points=points_arr,
    )
