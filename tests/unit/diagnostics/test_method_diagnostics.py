"""Tests for diagnostics.method_diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from conformal_ts.adapters.callable import CallableAdapter
from conformal_ts.diagnostics.method_diagnostics import (
    aci_state,
    agaci_state,
    method_state,
    nexcp_state,
    spci_state,
)
from conformal_ts.methods.aci import AdaptiveConformalInference
from conformal_ts.methods.agaci import AggregatedAdaptiveConformalInference
from conformal_ts.methods.nexcp import NonexchangeableConformalPrediction
from conformal_ts.methods.spci import SequentialPredictiveConformalInference
from conformal_ts.methods.split import SplitConformal
from conformal_ts.quantile_regressors.base import QuantileRegressor


def _zero_predict_fn(n_series: int, horizon: int):
    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.zeros((n_series, horizon))

    return predict_fn


def _make_iid(rng: np.random.Generator, n_series: int, horizon: int, n: int):
    histories = [rng.normal(size=(n_series, 20)) for _ in range(n)]
    truths = rng.normal(size=(n_series, n, horizon))
    return histories, truths


class _MockQuantileRegressor(QuantileRegressor):
    """Mock regressor returning the empirical quantile of training y."""

    def __init__(self) -> None:
        self._y: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        self._y = np.asarray(y, dtype=np.float64)

    def predict_quantile(self, X: NDArray[np.floating], q: float) -> NDArray[np.floating]:
        assert self._y is not None
        return np.full(X.shape[0], float(np.quantile(self._y, q)))


def _mock_factory() -> QuantileRegressor:
    return _MockQuantileRegressor()


# ===========================================================================
# aci_state
# ===========================================================================


class TestAciState:
    def _calibrated(self, gamma: float = 0.05) -> AdaptiveConformalInference:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=gamma)
        histories, truths = _make_iid(rng, n_series, horizon, 60)
        method.calibrate(histories, truths)
        return method

    def test_keys_and_shapes(self) -> None:
        method = self._calibrated()
        state = aci_state(method)
        assert set(state.keys()) == {
            "alpha_t",
            "alpha_t_minus_alpha",
            "n_observations",
            "gamma",
        }
        assert state["alpha_t"].shape == (2, 3)
        assert state["alpha_t_minus_alpha"].shape == (2, 3)
        assert isinstance(state["n_observations"], int)
        assert state["gamma"] == pytest.approx(0.05)

    def test_drift_changes_after_updates(self) -> None:
        method = self._calibrated()
        before = aci_state(method)["alpha_t_minus_alpha"].copy()
        # Force misses by passing huge truths.
        for _ in range(20):
            result = method.predict(np.zeros((2, 20)))
            method.update(result.point, np.full((2, 1, 3), 1e6, dtype=np.float64))
        after = aci_state(method)["alpha_t_minus_alpha"]
        assert not np.allclose(before, after)

    def test_wrong_type_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        split = SplitConformal(adapter, alpha=0.1)
        with pytest.raises(TypeError, match="AdaptiveConformalInference"):
            aci_state(split)  # type: ignore[arg-type]

    def test_uncalibrated_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        with pytest.raises(RuntimeError, match="calibrated"):
            aci_state(method)


# ===========================================================================
# agaci_state
# ===========================================================================


class TestAgaciState:
    def _calibrated(self) -> AggregatedAdaptiveConformalInference:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AggregatedAdaptiveConformalInference(
            adapter,
            alpha=0.1,
            gammas=(0.01, 0.05, 0.1),
        )
        histories, truths = _make_iid(rng, n_series, horizon, 60)
        method.calibrate(histories, truths)
        return method

    def test_keys_and_shapes(self) -> None:
        method = self._calibrated()
        state = agaci_state(method)
        expected_keys = {
            "alpha_t_per_expert",
            "gammas",
            "weights_lower",
            "weights_upper",
            "best_expert_per_cell_lower",
            "best_expert_per_cell_upper",
            "n_observations",
        }
        assert set(state.keys()) == expected_keys
        n_experts, n_series, horizon = 3, 2, 3
        assert state["alpha_t_per_expert"].shape == (n_experts, n_series, horizon)
        assert state["weights_lower"].shape == (n_experts, n_series, horizon)
        assert state["weights_upper"].shape == (n_experts, n_series, horizon)
        assert state["best_expert_per_cell_lower"].shape == (n_series, horizon)
        assert state["best_expert_per_cell_upper"].shape == (n_series, horizon)
        assert state["gammas"] == (0.01, 0.05, 0.1)

    def test_weights_sum_to_one_per_cell(self) -> None:
        method = self._calibrated()
        state = agaci_state(method)
        np.testing.assert_allclose(state["weights_lower"].sum(axis=0), 1.0, atol=1e-12)
        np.testing.assert_allclose(state["weights_upper"].sum(axis=0), 1.0, atol=1e-12)

    def test_wrong_type_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        split = SplitConformal(adapter, alpha=0.1)
        with pytest.raises(TypeError, match="AggregatedAdaptiveConformalInference"):
            agaci_state(split)  # type: ignore[arg-type]


# ===========================================================================
# nexcp_state
# ===========================================================================


class TestNexcpState:
    def _calibrated(self, rho: float = 0.99) -> NonexchangeableConformalPrediction:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=rho)
        histories, truths = _make_iid(rng, n_series, horizon, 100)
        method.calibrate(histories, truths)
        return method

    def test_ess_equals_n_when_rho_one(self) -> None:
        method = self._calibrated(rho=1.0)
        state = nexcp_state(method)
        assert state["effective_sample_size"] == pytest.approx(state["n_observations"])

    def test_ess_lt_n_when_rho_lt_one(self) -> None:
        method = self._calibrated(rho=0.99)
        state = nexcp_state(method)
        n = state["n_observations"]
        assert 50 < state["effective_sample_size"] < n

    def test_keys_and_weight_bounds(self) -> None:
        method = self._calibrated(rho=0.99)
        state = nexcp_state(method)
        expected_keys = {
            "rho",
            "n_observations",
            "effective_sample_size",
            "weights_minimum",
            "weights_maximum",
            "score_quantile",
        }
        assert set(state.keys()) == expected_keys
        assert state["weights_maximum"] == pytest.approx(1.0)
        assert state["weights_minimum"] < state["weights_maximum"]
        assert state["score_quantile"].shape == (1, 2)

    def test_wrong_type_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        split = SplitConformal(adapter, alpha=0.1)
        with pytest.raises(TypeError, match="NonexchangeableConformalPrediction"):
            nexcp_state(split)  # type: ignore[arg-type]


# ===========================================================================
# spci_state
# ===========================================================================


class TestSpciState:
    def _calibrated(self) -> SequentialPredictiveConformalInference:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=10,
            regressor_factory=_mock_factory,
            beta_grid_size=5,
            refit_every=2,
        )
        histories, truths = _make_iid(rng, n_series, horizon, 80)
        method.calibrate(histories, truths)
        return method

    def test_keys_and_values(self) -> None:
        method = self._calibrated()
        state = spci_state(method)
        expected_keys = {
            "window_size",
            "n_observations",
            "n_regressors",
            "regressor_class",
            "refit_every",
            "steps_since_refit",
            "residuals_shape",
        }
        assert set(state.keys()) == expected_keys
        assert state["n_regressors"] == 2 * 3
        assert state["regressor_class"] == "_MockQuantileRegressor"
        assert state["window_size"] == 10
        assert state["refit_every"] == 2
        assert state["residuals_shape"] == (2, 80, 3)

    def test_wrong_type_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        split = SplitConformal(adapter, alpha=0.1)
        with pytest.raises(TypeError, match="SequentialPredictiveConformalInference"):
            spci_state(split)  # type: ignore[arg-type]


# ===========================================================================
# method_state dispatcher
# ===========================================================================


class TestMethodStateDispatch:
    def test_aci(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        histories, truths = _make_iid(rng, 1, 1, 50)
        method.calibrate(histories, truths)
        state = method_state(method)
        assert "alpha_t" in state

    def test_split_returns_empty_dict(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(adapter, alpha=0.1)
        histories, truths = _make_iid(rng, 1, 1, 50)
        method.calibrate(histories, truths)
        state = method_state(method)
        assert state == {}
