"""Tests for diagnostics.reports.evaluate_calibration (Mode 1)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import norm

from conformal_ts.adapters.callable import CallableAdapter
from conformal_ts.base import CalibrationError, Forecast, ForecasterAdapter, Series
from conformal_ts.capabilities import SupportsQuantiles
from conformal_ts.diagnostics import Report, evaluate, evaluate_calibration
from conformal_ts.methods.aci import AdaptiveConformalInference
from conformal_ts.methods.agaci import AggregatedAdaptiveConformalInference
from conformal_ts.methods.cqr import ConformalizedQuantileRegression
from conformal_ts.methods.nexcp import NonexchangeableConformalPrediction
from conformal_ts.methods.spci import SequentialPredictiveConformalInference
from conformal_ts.methods.split import SplitConformal
from conformal_ts.quantile_regressors.base import QuantileRegressor


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _zero_predict_fn(n_series: int, horizon: int):
    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.zeros((n_series, horizon))

    return predict_fn


def _make_iid(rng: np.random.Generator, n_series: int, horizon: int, n: int):
    histories = [rng.normal(size=(n_series, 20)) for _ in range(n)]
    truths = rng.normal(size=(n_series, n, horizon))
    return histories, truths


class _MockQuantileRegressor(QuantileRegressor):
    """Empirical-quantile mock regressor — keeps SPCI tests fast."""

    def __init__(self) -> None:
        self._y: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        self._y = np.asarray(y, dtype=np.float64)

    def predict_quantile(self, X: NDArray[np.floating], q: float) -> NDArray[np.floating]:
        assert self._y is not None
        return np.full(X.shape[0], float(np.quantile(self._y, q)))


def _mock_factory() -> QuantileRegressor:
    return _MockQuantileRegressor()


class _QuantileCallableAdapter(ForecasterAdapter, SupportsQuantiles):
    """Quantile adapter mirroring the const-Gaussian style used in CQR tests."""

    def __init__(self, n_series: int, horizon: int) -> None:
        super().__init__(horizon=horizon, n_series=n_series)

    def predict(self, history: Series) -> Forecast:
        history = self._validate_history(history)
        return np.zeros((self.n_series, 1, self.horizon))

    def predict_quantiles(
        self,
        history: Series,
        quantiles: NDArray[np.floating],
    ) -> Forecast:
        history = self._validate_history(history)
        z = norm.ppf(quantiles)
        # Centered at 0; quantile = z (broadcast over series & horizon).
        out = np.broadcast_to(
            z[np.newaxis, :, np.newaxis], (self.n_series, quantiles.shape[0], self.horizon)
        )
        return np.asarray(out, dtype=np.float64).copy()


# ===========================================================================
# Per-method end-to-end
# ===========================================================================


class TestEvaluateCalibrationSplit:
    def test_basic(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SplitConformal(adapter, alpha=0.1)
        histories, truths = _make_iid(rng, n_series, horizon, 200)
        method.calibrate(histories, truths)

        report = evaluate_calibration(method)
        assert isinstance(report, Report)
        assert report.method_name == "SplitConformal (calibration)"
        assert report.alpha == 0.1
        assert report.n_holdout_samples == 200
        assert report.intervals.shape == (n_series, 200, horizon, 2)
        assert report.truths.shape == (n_series, 200, horizon)
        assert report.points.shape == (n_series, 200, horizon)
        # Split CP achieves close to nominal in-sample (no time-varying state).
        assert abs(report.marginal_coverage - 0.9) < 0.05
        # SplitConformal exposes no Layer 2 state.
        assert report.method_state == {}


class TestEvaluateCalibrationCQR:
    def test_basic(self) -> None:
        n_series, horizon = 1, 2
        adapter = _QuantileCallableAdapter(n_series=n_series, horizon=horizon)
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)
        rng = np.random.default_rng(0)
        histories = [rng.standard_normal((n_series, 20)) for _ in range(200)]
        truths = rng.standard_normal((n_series, 200, horizon))
        method.calibrate(histories, truths)

        report = evaluate_calibration(method)
        assert report.method_name == "ConformalizedQuantileRegression (calibration)"
        assert report.intervals.shape == (n_series, 200, horizon, 2)
        assert abs(report.marginal_coverage - 0.9) < 0.05


class TestEvaluateCalibrationACI:
    def test_basic_and_state_populated(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        histories, truths = _make_iid(rng, n_series, horizon, 200)
        method.calibrate(histories, truths)

        report = evaluate_calibration(method)
        assert report.method_name == "AdaptiveConformalInference (calibration)"
        assert "alpha_t" in report.method_state
        assert "gamma" in report.method_state
        # In-sample coverage uses the post-calibration alpha_t_ applied to all
        # samples; should land within a loose band of nominal.
        assert 0.5 < report.marginal_coverage < 1.0


class TestEvaluateCalibrationAgACI:
    def test_basic_and_weights_in_state(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1, gammas=(0.01, 0.05, 0.1))
        histories, truths = _make_iid(rng, n_series, horizon, 100)
        method.calibrate(histories, truths)

        report = evaluate_calibration(method)
        assert report.method_name == "AggregatedAdaptiveConformalInference (calibration)"
        assert "weights_lower" in report.method_state
        assert "weights_upper" in report.method_state
        assert report.intervals.shape == (n_series, 100, horizon, 2)


class TestEvaluateCalibrationNexCP:
    def test_basic_and_ess_in_state(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        histories, truths = _make_iid(rng, n_series, horizon, 200)
        method.calibrate(histories, truths)

        report = evaluate_calibration(method)
        assert report.method_name == "NonexchangeableConformalPrediction (calibration)"
        assert "effective_sample_size" in report.method_state
        assert abs(report.marginal_coverage - 0.9) < 0.06


class TestEvaluateCalibrationSPCI:
    def test_basic(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 2
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
        )
        histories, truths = _make_iid(rng, n_series, horizon, 60)
        method.calibrate(histories, truths)

        report = evaluate_calibration(method)
        assert report.method_name == "SequentialPredictiveConformalInference (calibration)"
        assert "n_regressors" in report.method_state
        assert report.intervals.shape == (n_series, 60, horizon, 2)


# ===========================================================================
# Error paths
# ===========================================================================


class TestEvaluateCalibrationErrors:
    def test_uncalibrated_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(adapter, alpha=0.1)
        with pytest.raises(CalibrationError, match="calibrate"):
            evaluate_calibration(method)

    def test_missing_predictions_calibration_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(adapter, alpha=0.1)
        histories, truths = _make_iid(rng, 1, 1, 50)
        method.calibrate(histories, truths)
        del method.predictions_calibration_
        with pytest.raises(RuntimeError, match="predictions_calibration_"):
            evaluate_calibration(method)

    def test_missing_truths_calibration_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(adapter, alpha=0.1)
        histories, truths = _make_iid(rng, 1, 1, 50)
        method.calibrate(histories, truths)
        del method.truths_calibration_
        with pytest.raises(RuntimeError, match="truths_calibration_"):
            evaluate_calibration(method)


# ===========================================================================
# Cross-mode consistency
# ===========================================================================


class TestCrossModeConsistency:
    """Mode 1 and Mode 2 produce the same Report structure."""

    def test_report_fields_match(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SplitConformal(adapter, alpha=0.1)
        histories, truths = _make_iid(rng, n_series, horizon, 200)
        method.calibrate(histories, truths)
        ho_h, ho_t = _make_iid(rng, n_series, horizon, 100)

        report_cal = evaluate_calibration(method)
        report_ho = evaluate(method, ho_h, ho_t)

        # Same field names with the same shapes.
        assert report_cal.coverage_by_horizon.shape == report_ho.coverage_by_horizon.shape
        assert report_cal.coverage_by_series.shape == report_ho.coverage_by_series.shape
        # method_name differs only by the " (calibration)" suffix.
        assert report_cal.method_name.endswith("(calibration)")
        assert report_ho.method_name == "SplitConformal"
