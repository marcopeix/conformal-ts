"""Shared helpers for online conformal methods.

Centralises two recurring snippets that previously appeared verbatim across
``aci.py``, ``agaci.py``, ``nexcp.py``, and ``spci.py``:

* :func:`_per_cell_quantile` — per-(series, horizon) empirical quantile of a
  score panel (used by ACI's and AgACI's online loops).
* :func:`_validate_online_shapes` — shape validation for the
  ``(n_series, 1, horizon)`` ``(prediction, truth)`` pair passed to
  :meth:`ConformalMethod.update`.

Both are private to the methods package — call sites import them through the
single-underscore module name.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..base import Forecast


def _per_cell_quantile(
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
        with saturation at the level boundaries (``ql >= 1`` → ``+max float``;
        ``ql <= 0`` → ``-max float``).

    Notes
    -----
    The nested-loop implementation is ``O(n_series * horizon * t log t)``.
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


def _validate_online_shapes(
    prediction: Forecast,
    truth: Forecast,
    n_series: int,
    horizon: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Coerce ``(prediction, truth)`` to ``float64`` and verify the online shape.

    Online methods always feed the ``predict`` → ``update`` loop with a
    single-sample ``(n_series, 1, horizon)`` pair. This helper centralises
    the dtype coercion and shape check.

    Parameters
    ----------
    prediction, truth : Forecast
        Expected shape ``(n_series, 1, horizon)``.
    n_series : int
    horizon : int

    Returns
    -------
    (prediction_arr, truth_arr) : tuple of NDArray
        Float64 views of the inputs.

    Raises
    ------
    ValueError
        If either array does not have shape ``(n_series, 1, horizon)``.
    """
    prediction_arr = np.asarray(prediction, dtype=np.float64)
    truth_arr = np.asarray(truth, dtype=np.float64)
    expected_shape = (n_series, 1, horizon)
    if prediction_arr.shape != expected_shape:
        raise ValueError(
            f"prediction must have shape {expected_shape}, got {prediction_arr.shape}."
        )
    if truth_arr.shape != expected_shape:
        raise ValueError(f"truth must have shape {expected_shape}, got {truth_arr.shape}.")
    return prediction_arr, truth_arr
