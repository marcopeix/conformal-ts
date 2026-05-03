"""Tests for CallableAdapter."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_ts.adapters.callable import CallableAdapter


def _naive_forecast(history: np.ndarray) -> np.ndarray:
    """Repeat the last value for each series across the horizon."""
    # history: (n_series, T) -> (n_series, horizon)
    # We'll use a closure to pass horizon; tests create lambdas instead.
    return np.repeat(history[:, -1:], history.shape[0], axis=1)


class TestConstruction:
    """CallableAdapter construction validation."""

    def test_valid_construction(self) -> None:
        adapter = CallableAdapter(
            predict_fn=lambda h: h[:, -1:],
            horizon=1,
            n_series=1,
        )
        assert adapter.horizon == 1
        assert adapter.n_series == 1

    def test_non_callable_raises(self) -> None:
        with pytest.raises(ValueError, match="callable"):
            CallableAdapter(predict_fn="not a function", horizon=1, n_series=1)  # type: ignore[arg-type]

    def test_invalid_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            CallableAdapter(predict_fn=lambda h: h, horizon=0, n_series=1)

    def test_invalid_n_series_raises(self) -> None:
        with pytest.raises(ValueError, match="n_series"):
            CallableAdapter(predict_fn=lambda h: h, horizon=1, n_series=0)


class TestPredict:
    """CallableAdapter.predict shape and validation."""

    @pytest.mark.parametrize(
        "n_series,horizon",
        [(1, 1), (1, 12), (5, 1), (5, 12)],
    )
    def test_output_shape(self, n_series: int, horizon: int) -> None:
        adapter = CallableAdapter(
            predict_fn=lambda h, hz=horizon: np.ones((h.shape[0], hz)),
            horizon=horizon,
            n_series=n_series,
        )
        history = np.random.default_rng(0).standard_normal((n_series, 50))
        result = adapter.predict(history)
        assert result.shape == (n_series, 1, horizon)

    def test_wrong_history_ndim_raises(self) -> None:
        adapter = CallableAdapter(
            predict_fn=lambda h: h[:, -1:],
            horizon=1,
            n_series=1,
        )
        with pytest.raises(ValueError, match="2-D"):
            adapter.predict(np.array([1.0, 2.0, 3.0]))

    def test_wrong_history_n_series_raises(self) -> None:
        adapter = CallableAdapter(
            predict_fn=lambda h: h[:, -1:],
            horizon=1,
            n_series=2,
        )
        with pytest.raises(ValueError, match="leading axis"):
            adapter.predict(np.ones((3, 10)))

    def test_bad_predict_fn_output_shape_raises(self) -> None:
        adapter = CallableAdapter(
            predict_fn=lambda h: h,  # returns (n_series, T), wrong horizon
            horizon=5,
            n_series=1,
        )
        with pytest.raises(ValueError, match="predict_fn must return shape"):
            adapter.predict(np.ones((1, 10)))


class TestPredictBatch:
    """CallableAdapter.predict_batch stacks correctly."""

    @pytest.mark.parametrize("n_series", [1, 5])
    def test_output_shape(self, n_series: int) -> None:
        horizon = 6
        adapter = CallableAdapter(
            predict_fn=lambda h, hz=horizon: np.ones((h.shape[0], hz)),
            horizon=horizon,
            n_series=n_series,
        )
        rng = np.random.default_rng(0)
        histories = [rng.standard_normal((n_series, 50)) for _ in range(10)]
        result = adapter.predict_batch(histories)
        assert result.shape == (n_series, 10, horizon)
