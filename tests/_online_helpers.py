"""Shared test fixtures for the online conformal-method test suite.

Previously each of ``test_aci.py``, ``test_agaci.py``, ``test_nexcp.py``, and
``test_spci.py`` carried its own near-identical copies of these helpers.
Centralising them here keeps every test exercising the same fixture
behaviour and shrinks the per-method test file by ~100 lines.

The module is test-only — it lives under ``tests/`` rather than the
installed package and is imported as ``tests._online_helpers``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from conformal_ts.adapters.callable import CallableAdapter
from conformal_ts.base import Forecast, Series
from conformal_ts.capabilities import SupportsCrossValidation


def _zero_predict_fn(n_series: int, horizon: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable that emits ``(n_series, horizon)`` zeros for any history."""

    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.zeros((n_series, horizon))

    return predict_fn


def _last_value_predict_fn(horizon: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable that repeats the last column of ``history`` ``horizon`` times."""

    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.repeat(history[:, -1:], horizon, axis=1)

    return predict_fn


class CVCallableAdapter(CallableAdapter, SupportsCrossValidation):
    """
    Test-only :class:`CallableAdapter` that owns a training panel and exposes
    rolling-origin cross-validation by replaying it.

    The CV semantics match :class:`StatsForecastAdapter`: the *last* window
    ends at column ``T``; window ``i`` ends at column
    ``T - (n_windows - 1 - i) * step_size``.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        training_panel: np.ndarray,
        horizon: int,
    ) -> None:
        super().__init__(
            predict_fn=predict_fn,
            horizon=horizon,
            n_series=training_panel.shape[0],
        )
        self._training_panel = np.asarray(training_panel, dtype=np.float64)

    def cross_validate(
        self,
        n_windows: int,
        step_size: int = 1,
        refit: bool | int = False,
    ) -> tuple[Forecast, Forecast]:
        T = self._training_panel.shape[1]
        preds: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        for i in range(n_windows):
            cutoff = T - self.horizon - (n_windows - 1 - i) * step_size
            if cutoff < 1:
                raise ValueError(
                    f"training panel too short: cutoff={cutoff} for window {i}, "
                    f"n_windows={n_windows}, step_size={step_size}, "
                    f"horizon={self.horizon}, T={T}."
                )
            history = self._training_panel[:, :cutoff]
            truth = self._training_panel[:, cutoff : cutoff + self.horizon]
            pred = self.predict(history)  # (n_series, 1, horizon)
            preds.append(pred)
            truths.append(truth[:, np.newaxis, :])
        return np.concatenate(preds, axis=1), np.concatenate(truths, axis=1)

    def cv_window_history(self, n_windows: int, step_size: int, window_index: int) -> np.ndarray:
        """Reconstruct the history slice used for window ``window_index``."""
        T = self._training_panel.shape[1]
        cutoff = T - self.horizon - (n_windows - 1 - window_index) * step_size
        return self._training_panel[:, :cutoff]


def _make_iid_dataset(
    rng: np.random.Generator,
    n_series: int,
    horizon: int,
    n_samples: int,
    noise_std: float,
    history_len: int = 30,
) -> tuple[list[Series], np.ndarray, list[np.ndarray]]:
    """
    Build an iid Gaussian ``(histories, truths, truth_panels)`` triple.

    ``predict_fn`` is typically the zero function for callers, so the scores
    are effectively ``|truth|``.

    Returns
    -------
    histories : list of (n_series, history_len) arrays
    truths : (n_series, n_samples, horizon) array
    truth_panels : list of (n_series, horizon) arrays — same data, per-sample.
    """
    histories: list[Series] = []
    truth_panels: list[np.ndarray] = []
    for _ in range(n_samples):
        h = rng.normal(0.0, noise_std, (n_series, history_len))
        t = rng.normal(0.0, noise_std, (n_series, horizon))
        histories.append(h)
        truth_panels.append(t)
    truths = np.stack(truth_panels, axis=1)
    return histories, truths, truth_panels


def _run_online_cycle(
    method,
    holdout_histories: list[Series],
    holdout_truth_panels: list[np.ndarray],
    do_update: bool = True,
) -> list[np.ndarray]:
    """
    Run ``predict`` (and optionally ``update``) for each holdout sample.

    Parameters
    ----------
    method
        Any object with ``predict(history) -> PredictionResult`` and
        ``update(prediction, truth)``.
    holdout_histories : list of Series
    holdout_truth_panels : list of (n_series, horizon) arrays
    do_update : bool, default True
        Whether to call :meth:`update` between predictions. Set ``False`` for
        purely offline evaluation (e.g. when the score function does not
        condition on the residual history).

    Returns
    -------
    list of (n_series, 1, horizon, 2) intervals — one per holdout sample.
    """
    intervals: list[np.ndarray] = []
    for h, t_panel in zip(holdout_histories, holdout_truth_panels, strict=True):
        result = method.predict(h)
        intervals.append(result.interval.copy())
        if do_update:
            method.update(result.point, t_panel[:, np.newaxis, :])
    return intervals


def _empirical_coverage(intervals: list[np.ndarray], truth_panels: list[np.ndarray]) -> float:
    """Element-wise coverage averaged over ``(n_series, 1, horizon)``."""
    covered = 0
    total = 0
    for interval, truth in zip(intervals, truth_panels, strict=True):
        truth_3d = truth[:, np.newaxis, :]
        in_interval = (truth_3d >= interval[..., 0]) & (truth_3d <= interval[..., 1])
        covered += int(in_interval.sum())
        total += int(in_interval.size)
    return covered / total
