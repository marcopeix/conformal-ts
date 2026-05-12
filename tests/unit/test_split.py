"""Tests for SplitConformal method."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_ts.adapters.callable import CallableAdapter
from conformal_ts.base import CalibrationError, UnsupportedCapability
from conformal_ts.methods.split import SplitConformal


def _make_adapter(n_series: int, horizon: int) -> tuple[CallableAdapter, np.random.Generator]:
    """Create a naive last-value-repeat adapter."""
    rng = np.random.default_rng(0)

    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.repeat(history[:, -1:], horizon, axis=1)

    adapter = CallableAdapter(predict_fn=predict_fn, horizon=horizon, n_series=n_series)
    return adapter, rng


class TestCoverageGuarantee:
    """Empirical coverage matches 1 - alpha on synthetic iid Gaussian data."""

    @pytest.mark.parametrize(
        "n_series,horizon,alpha",
        [
            (1, 1, 0.1),
            (1, 6, 0.1),
            (3, 1, 0.1),
            (3, 6, 0.1),
            (1, 1, 0.2),
            (3, 6, 0.2),
        ],
    )
    def test_marginal_coverage(self, n_series: int, horizon: int, alpha: float) -> None:
        rng = np.random.default_rng(0)
        noise_std = 1.0
        n_cal = 500
        n_test = 500
        T = 50

        # Synthetic data: constant signal + iid noise
        signal = rng.standard_normal((n_series, 1)) * 10  # per-series level

        def predict_fn(history: np.ndarray) -> np.ndarray:
            # "Oracle" that returns the true signal level
            return np.repeat(
                history[:, -1:] - noise_std * rng.standard_normal((n_series, 1)) + signal * 0,
                horizon,
                axis=1,
            )

        def last_value_fn(history: np.ndarray) -> np.ndarray:
            return np.repeat(history[:, -1:], horizon, axis=1)

        adapter = CallableAdapter(predict_fn=last_value_fn, horizon=horizon, n_series=n_series)

        # Generate calibration data
        cal_histories = []
        cal_truths_list = []
        for _ in range(n_cal):
            noise_hist = rng.normal(0, noise_std, (n_series, T))
            history = signal + noise_hist
            noise_future = rng.normal(0, noise_std, (n_series, horizon))
            truth = signal + noise_future
            cal_histories.append(history)
            cal_truths_list.append(truth)

        # truths: (n_series, n_cal, horizon)
        cal_truths = np.stack(cal_truths_list, axis=1)

        method = SplitConformal(adapter, alpha=alpha)
        method.calibrate(cal_histories, cal_truths)

        # Evaluate on test data
        covered = 0
        total = 0
        for _ in range(n_test):
            noise_hist = rng.normal(0, noise_std, (n_series, T))
            history = signal + noise_hist
            noise_future = rng.normal(0, noise_std, (n_series, horizon))
            truth = signal + noise_future

            result = method.predict(history)
            lower = result.interval[..., 0]  # (n_series, 1, horizon)
            upper = result.interval[..., 1]

            # Check if truth is inside the interval for each (series, horizon)
            truth_3d = truth[:, np.newaxis, :]  # (n_series, 1, horizon)
            in_interval = (truth_3d >= lower) & (truth_3d <= upper)
            covered += in_interval.sum()
            total += in_interval.size

        empirical_coverage = covered / total
        assert abs(empirical_coverage - (1 - alpha)) < 0.03, (
            f"Empirical coverage {empirical_coverage:.4f} "
            f"deviates from target {1 - alpha:.2f} by more than 0.03"
        )


class TestShape:
    """predict() returns documented shapes."""

    @pytest.mark.parametrize(
        "n_series,horizon",
        [(1, 1), (1, 12), (5, 1), (5, 12)],
    )
    def test_output_shapes(self, n_series: int, horizon: int) -> None:
        adapter, rng = _make_adapter(n_series, horizon)
        method = SplitConformal(adapter, alpha=0.1)

        n_cal = 50
        histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]
        truths = rng.standard_normal((n_series, n_cal, horizon))
        method.calibrate(histories, truths)

        history = rng.standard_normal((n_series, 30))
        result = method.predict(history)

        assert result.point.shape == (n_series, 1, horizon)
        assert result.interval.shape == (n_series, 1, horizon, 2)
        assert result.alpha == 0.1


class TestDeterminism:
    """Same input + same calibration -> identical outputs."""

    def test_deterministic_intervals(self) -> None:
        n_series, horizon = 3, 6
        rng = np.random.default_rng(42)
        n_cal = 100

        histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]
        truths = rng.standard_normal((n_series, n_cal, horizon))
        test_history = rng.standard_normal((n_series, 30))

        def run_once() -> tuple[np.ndarray, np.ndarray]:
            adapter = CallableAdapter(
                predict_fn=lambda h: np.repeat(h[:, -1:], horizon, axis=1),
                horizon=horizon,
                n_series=n_series,
            )
            method = SplitConformal(adapter, alpha=0.1)
            method.calibrate(histories, truths)
            result = method.predict(test_history)
            return result.point, result.interval

        point1, interval1 = run_once()
        point2, interval2 = run_once()

        np.testing.assert_array_equal(point1, point2)
        np.testing.assert_array_equal(interval1, interval2)


class TestCalibrationErrors:
    """CalibrationError raised in the expected situations."""

    def test_predict_before_calibrate_raises(self) -> None:
        adapter, _ = _make_adapter(1, 1)
        method = SplitConformal(adapter, alpha=0.1)
        with pytest.raises(CalibrationError, match="calibrate"):
            method.predict(np.ones((1, 10)))

    @pytest.mark.parametrize("alpha", [0.1, 0.2, 0.5])
    def test_too_few_calibration_samples_raises(self, alpha: float) -> None:
        import math

        n_series, horizon = 1, 1
        adapter, rng = _make_adapter(n_series, horizon)
        method = SplitConformal(adapter, alpha=alpha)

        min_needed = math.ceil(1.0 / alpha)
        too_few = min_needed - 1

        histories = [rng.standard_normal((n_series, 10)) for _ in range(too_few)]
        truths = rng.standard_normal((n_series, too_few, horizon))

        with pytest.raises(CalibrationError, match="calibration samples"):
            method.calibrate(histories, truths)


class TestInvalidAlpha:
    """ValueError for alpha outside (0, 1)."""

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0, 1.5, 2.0])
    def test_invalid_alpha_raises(self, alpha: float) -> None:
        adapter, _ = _make_adapter(1, 1)
        with pytest.raises(ValueError, match="alpha"):
            SplitConformal(adapter, alpha=alpha)


class TestCalibrateDispatch:
    """calibrate() input validation across the two calling conventions."""

    def test_no_args_raises(self) -> None:
        adapter, _ = _make_adapter(1, 1)
        method = SplitConformal(adapter, alpha=0.1)
        with pytest.raises(ValueError, match="histories"):
            method.calibrate()

    def test_both_paths_raises(self) -> None:
        adapter, rng = _make_adapter(1, 1)
        method = SplitConformal(adapter, alpha=0.1)
        histories = [rng.standard_normal((1, 10)) for _ in range(50)]
        truths = rng.standard_normal((1, 50, 1))
        with pytest.raises(ValueError, match="not both"):
            method.calibrate(histories, truths, n_windows=10)

    def test_n_windows_without_cv_support_raises(self) -> None:
        # CallableAdapter does not implement SupportsCrossValidation.
        adapter, _ = _make_adapter(1, 1)
        method = SplitConformal(adapter, alpha=0.1)
        with pytest.raises(UnsupportedCapability, match="SupportsCrossValidation"):
            method.calibrate(n_windows=20)


class TestFittedState:
    """sklearn-style fitted state on the instance is canonical."""

    def test_fitted_attributes_set_after_calibrate(self) -> None:
        adapter, rng = _make_adapter(2, 3)
        method = SplitConformal(adapter, alpha=0.1)

        assert method.is_calibrated_ is False

        histories = [rng.standard_normal((2, 30)) for _ in range(50)]
        truths = rng.standard_normal((2, 50, 3))
        method.calibrate(histories, truths)

        assert method.is_calibrated_ is True
        assert method.score_quantile_.shape == (2, 3)
        assert method.n_calibration_samples_ == 50

    def test_calibration_result_is_a_snapshot(self) -> None:
        """Mutating the returned CalibrationResult does not affect instance state."""
        adapter, rng = _make_adapter(1, 2)
        method = SplitConformal(adapter, alpha=0.1)

        histories = [rng.standard_normal((1, 30)) for _ in range(50)]
        truths = rng.standard_normal((1, 50, 2))
        cal = method.calibrate(histories, truths)

        original = method.score_quantile_.copy()
        cal.score_quantile[...] = 999.0  # mutate the snapshot

        np.testing.assert_array_equal(method.score_quantile_, original)

    def test_stores_calibration_data(self) -> None:
        adapter, rng = _make_adapter(2, 3)
        method = SplitConformal(adapter, alpha=0.1)
        histories = [rng.standard_normal((2, 30)) for _ in range(50)]
        truths = rng.standard_normal((2, 50, 3))
        method.calibrate(histories, truths)

        assert method.predictions_calibration_.shape == (2, 50, 3)
        assert method.truths_calibration_.shape == (2, 50, 3)
        np.testing.assert_allclose(method.truths_calibration_, truths)

    def test_calibration_data_is_defensive_copy(self) -> None:
        adapter, rng = _make_adapter(1, 1)
        method = SplitConformal(adapter, alpha=0.1)
        histories = [rng.standard_normal((1, 30)) for _ in range(50)]
        truths = rng.standard_normal((1, 50, 1))
        method.calibrate(histories, truths)

        score_quantile_before = method.score_quantile_.copy()
        method.predictions_calibration_[...] = 0.0
        method.truths_calibration_[...] = 0.0
        np.testing.assert_array_equal(method.score_quantile_, score_quantile_before)


class TestIntervalsFromPredictions:
    def test_matches_invert(self) -> None:
        adapter, rng = _make_adapter(2, 3)
        method = SplitConformal(adapter, alpha=0.1)
        histories = [rng.standard_normal((2, 30)) for _ in range(50)]
        truths = rng.standard_normal((2, 50, 3))
        method.calibrate(histories, truths)

        intervals = method._intervals_from_predictions(method.predictions_calibration_)
        expected = method.score_fn.invert(method.predictions_calibration_, method.score_quantile_)
        np.testing.assert_allclose(intervals, expected)
        assert intervals.shape == (2, 50, 3, 2)
