"""Method-specific introspection (Layer 2 diagnostics)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import ConformalMethod
from ..methods.aci import AdaptiveConformalInference
from ..methods.agaci import AggregatedAdaptiveConformalInference
from ..methods.nexcp import NonexchangeableConformalPrediction
from ..methods.spci import SequentialPredictiveConformalInference


def _require_calibrated(method: ConformalMethod) -> None:
    if not getattr(method, "is_calibrated_", False):
        raise RuntimeError(f"{type(method).__name__} is not calibrated. Call calibrate() first.")


def aci_state(method: AdaptiveConformalInference) -> dict[str, Any]:
    """
    Snapshot of ACI's adaptive state.

    Parameters
    ----------
    method : AdaptiveConformalInference
        Must be calibrated.

    Returns
    -------
    dict with keys:
        ``"alpha_t"`` : NDArray, shape (n_series, horizon)
            Current adaptive miscoverage rate per cell.
        ``"alpha_t_minus_alpha"`` : NDArray, shape (n_series, horizon)
            How far the adaptive rate has drifted from the target.
            Magnitude indicates how much adaptation occurred.
        ``"n_observations"`` : int
            Total observations (calibration + updates).
        ``"gamma"`` : float
            Learning rate.

    Raises
    ------
    TypeError
        If ``method`` is not an :class:`AdaptiveConformalInference`.
    RuntimeError
        If ``method`` is not calibrated.
    """
    if not isinstance(method, AdaptiveConformalInference):
        raise TypeError(
            "aci_state expects an AdaptiveConformalInference instance; "
            f"got {type(method).__name__}."
        )
    _require_calibrated(method)

    alpha_t = method.alpha_t_.copy()
    return {
        "alpha_t": alpha_t,
        "alpha_t_minus_alpha": alpha_t - method.alpha,
        "n_observations": int(method.n_observations_),
        "gamma": float(method.gamma),
    }


def agaci_state(method: AggregatedAdaptiveConformalInference) -> dict[str, Any]:
    """
    Snapshot of AgACI's per-expert state and aggregator weights.

    Parameters
    ----------
    method : AggregatedAdaptiveConformalInference
        Must be calibrated.

    Returns
    -------
    dict with keys:
        ``"alpha_t_per_expert"`` : NDArray, shape (n_experts, n_series, horizon)
        ``"gammas"`` : tuple of float
        ``"weights_lower"`` : NDArray, shape (n_experts, n_series, horizon)
            Current normalized weights of the lower-bound aggregator.
        ``"weights_upper"`` : NDArray, shape (n_experts, n_series, horizon)
        ``"best_expert_per_cell_lower"`` : NDArray of int, shape (n_series, horizon)
            Index of the expert with highest weight per cell, for the
            lower-bound aggregator.
        ``"best_expert_per_cell_upper"`` : NDArray of int, shape (n_series, horizon)
        ``"n_observations"`` : int

    Raises
    ------
    TypeError
        If ``method`` is not an :class:`AggregatedAdaptiveConformalInference`.
    RuntimeError
        If ``method`` is not calibrated.
    """
    if not isinstance(method, AggregatedAdaptiveConformalInference):
        raise TypeError(
            "agaci_state expects an AggregatedAdaptiveConformalInference instance; "
            f"got {type(method).__name__}."
        )
    _require_calibrated(method)

    weights_lower = method.aggregator_lower_.weights()
    weights_upper = method.aggregator_upper_.weights()
    best_lower = np.argmax(weights_lower, axis=0).astype(np.int64)
    best_upper = np.argmax(weights_upper, axis=0).astype(np.int64)

    return {
        "alpha_t_per_expert": method.alpha_t_per_expert_.copy(),
        "gammas": method.gammas,
        "weights_lower": weights_lower,
        "weights_upper": weights_upper,
        "best_expert_per_cell_lower": best_lower,
        "best_expert_per_cell_upper": best_upper,
        "n_observations": int(method.n_observations_),
    }


def nexcp_state(method: NonexchangeableConformalPrediction) -> dict[str, Any]:
    """
    Snapshot of NexCP's weighting state.

    Parameters
    ----------
    method : NonexchangeableConformalPrediction
        Must be calibrated.

    Returns
    -------
    dict with keys:
        ``"rho"`` : float
        ``"n_observations"`` : int
        ``"effective_sample_size"`` : float
            ``ESS = (sum w)^2 / sum(w^2)``. With ``rho = 1`` this equals
            ``n_observations``; with smaller ``rho`` it's less.
        ``"weights_minimum"`` : float
            Smallest weight in the buffer (the oldest sample's weight).
        ``"weights_maximum"`` : float
            Largest weight (the newest sample's weight = 1).
        ``"score_quantile"`` : NDArray, shape (n_series, horizon)
            Current weighted quantile threshold.

    Raises
    ------
    TypeError
        If ``method`` is not a :class:`NonexchangeableConformalPrediction`.
    RuntimeError
        If ``method`` is not calibrated.
    """
    if not isinstance(method, NonexchangeableConformalPrediction):
        raise TypeError(
            "nexcp_state expects a NonexchangeableConformalPrediction instance; "
            f"got {type(method).__name__}."
        )
    _require_calibrated(method)

    n_obs = int(method.n_observations_)
    rho = float(method.rho)
    # Weights for the current buffer: position i (0-indexed, oldest first)
    # gets rho^(n_obs - 1 - i). Newest sample is 1, oldest is rho^(n_obs - 1).
    i = np.arange(n_obs)
    weights = rho ** (n_obs - 1 - i)
    sum_w = float(weights.sum())
    sum_w_sq = float((weights * weights).sum())
    ess = (sum_w * sum_w / sum_w_sq) if sum_w_sq > 0 else 0.0

    return {
        "rho": rho,
        "n_observations": n_obs,
        "effective_sample_size": ess,
        "weights_minimum": float(weights.min()),
        "weights_maximum": float(weights.max()),
        "score_quantile": method.score_quantile_.copy(),
    }


def spci_state(method: SequentialPredictiveConformalInference) -> dict[str, Any]:
    """
    Snapshot of SPCI's regressor inventory.

    Parameters
    ----------
    method : SequentialPredictiveConformalInference
        Must be calibrated.

    Returns
    -------
    dict with keys:
        ``"window_size"`` : int
        ``"n_observations"`` : int
        ``"n_regressors"`` : int
            Should equal ``n_series * horizon``.
        ``"regressor_class"`` : str
        ``"refit_every"`` : int
        ``"steps_since_refit"`` : int
            Where in the refit cycle we are.
        ``"residuals_shape"`` : tuple
            Shape of the residual buffer
            ``(n_series, n_observations, horizon)``.

    Raises
    ------
    TypeError
        If ``method`` is not a :class:`SequentialPredictiveConformalInference`.
    RuntimeError
        If ``method`` is not calibrated.
    """
    if not isinstance(method, SequentialPredictiveConformalInference):
        raise TypeError(
            "spci_state expects a SequentialPredictiveConformalInference instance; "
            f"got {type(method).__name__}."
        )
    _require_calibrated(method)

    regressors = method.quantile_regressors_
    # All regressor instances are produced by the same factory; sample one
    # for its class name.
    sample = next(iter(regressors.values()))
    return {
        "window_size": int(method.window_size),
        "n_observations": int(method.n_observations_),
        "n_regressors": len(regressors),
        "regressor_class": type(sample).__name__,
        "refit_every": int(method.refit_every),
        "steps_since_refit": int(method._steps_since_refit_),
        "residuals_shape": tuple(method.residuals_.shape),
    }


def method_state(method: ConformalMethod) -> dict[str, Any]:
    """
    Dispatch to the appropriate method-specific state function.

    Parameters
    ----------
    method : ConformalMethod
        Must be calibrated if the dispatcher recognises it.

    Returns
    -------
    dict
        Result of :func:`aci_state` / :func:`agaci_state` / :func:`nexcp_state`
        / :func:`spci_state`, or an empty dict for methods without Layer 2
        diagnostics (e.g. :class:`SplitConformal`,
        :class:`ConformalizedQuantileRegression`).
    """
    if isinstance(method, AdaptiveConformalInference):
        return aci_state(method)
    if isinstance(method, AggregatedAdaptiveConformalInference):
        return agaci_state(method)
    if isinstance(method, NonexchangeableConformalPrediction):
        return nexcp_state(method)
    if isinstance(method, SequentialPredictiveConformalInference):
        return spci_state(method)
    return {}
