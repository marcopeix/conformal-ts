"""Coverage utilities for prediction intervals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..base import Forecast, Interval


def _coverage_indicator(interval: Interval, truth: Forecast) -> NDArray[np.bool_]:
    """
    Cell-wise coverage indicator.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)

    Returns
    -------
    NDArray[bool], shape (n_series, n_samples, horizon)
        True where lower <= truth <= upper.

    Raises
    ------
    ValueError
        If ``interval`` is not 4-D with last axis 2, or if
        ``truth.shape != interval.shape[:-1]``.
    """
    if interval.ndim != 4 or interval.shape[-1] != 2:
        raise ValueError(
            f"interval must have shape (n_series, n_samples, horizon, 2); got {interval.shape}."
        )
    if truth.shape != interval.shape[:-1]:
        raise ValueError(
            "truth.shape must equal interval.shape[:-1]; "
            f"got truth {truth.shape}, interval {interval.shape}."
        )
    lower = interval[..., 0]
    upper = interval[..., 1]
    return (lower <= truth) & (truth <= upper)


def marginal_coverage(interval: Interval, truth: Forecast) -> float:
    """
    Overall fraction of cells covered by their interval.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)

    Returns
    -------
    float
        Mean of the cell-wise coverage indicator over all axes.

    Raises
    ------
    ValueError
        If shapes are inconsistent (see :func:`_coverage_indicator`).
    """
    indicator = _coverage_indicator(interval, truth)
    return float(indicator.mean())


def coverage_by_horizon(interval: Interval, truth: Forecast) -> NDArray[np.floating]:
    """
    Coverage averaged across series and samples, per horizon step.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)

    Returns
    -------
    NDArray, shape (horizon,)
        Coverage per horizon step.

    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    indicator = _coverage_indicator(interval, truth)
    return np.asarray(indicator.mean(axis=(0, 1)), dtype=np.float64)


def coverage_by_series(interval: Interval, truth: Forecast) -> NDArray[np.floating]:
    """
    Coverage averaged across samples and horizon, per series.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)

    Returns
    -------
    NDArray, shape (n_series,)
        Coverage per series.

    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    indicator = _coverage_indicator(interval, truth)
    return np.asarray(indicator.mean(axis=(1, 2)), dtype=np.float64)


def coverage_per_cell(interval: Interval, truth: Forecast) -> NDArray[np.floating]:
    """
    Coverage averaged across samples, per (series, horizon).

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)

    Returns
    -------
    NDArray, shape (n_series, horizon)
        Coverage per (series, horizon) cell.

    Raises
    ------
    ValueError
        If shapes are inconsistent.
    """
    indicator = _coverage_indicator(interval, truth)
    return np.asarray(indicator.mean(axis=1), dtype=np.float64)


def rolling_coverage(
    interval: Interval,
    truth: Forecast,
    window: int,
) -> NDArray[np.floating]:
    """
    Rolling coverage averaged across the samples axis.

    Parameters
    ----------
    interval : Interval, shape (n_series, n_samples, horizon, 2)
    truth : Forecast, shape (n_series, n_samples, horizon)
    window : int
        Rolling window size in samples. Must satisfy
        ``1 <= window <= n_samples``.

    Returns
    -------
    NDArray, shape (n_series, n_samples - window + 1, horizon)
        Rolling-mean coverage. Output index ``t`` corresponds to coverage
        over samples ``[t, t + window)`` along the sample axis.

    Raises
    ------
    ValueError
        If shapes are inconsistent, or if ``window`` is out of range.
    """
    indicator = _coverage_indicator(interval, truth)
    n_samples = indicator.shape[1]

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    if window > n_samples:
        raise ValueError(f"window must be <= n_samples ({n_samples}), got {window}.")

    # sliding_window_view on the sample axis (axis=1) produces shape
    # (n_series, n_samples - window + 1, horizon, window); average over the
    # last axis.
    windows = np.lib.stride_tricks.sliding_window_view(
        indicator.astype(np.float64), window_shape=window, axis=1
    )
    return np.asarray(windows.mean(axis=-1), dtype=np.float64)
