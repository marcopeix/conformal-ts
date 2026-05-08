"""Tests for ConformalizedQuantileRegression."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import norm

from conformal_ts.adapters.callable import CallableAdapter
from conformal_ts.base import (
    CalibrationError,
    Forecast,
    ForecasterAdapter,
    Series,
    UnsupportedCapability,
)
from conformal_ts.capabilities import (
    SupportsCrossValidationQuantiles,
    SupportsQuantiles,
)
from conformal_ts.methods.cqr import ConformalizedQuantileRegression
from conformal_ts.methods.split import SplitConformal


# ---------------------------------------------------------------------------
# Test-local adapter that supports quantiles. Production CallableAdapter is
# point-only by design — we build a quantile-capable variant here without
# touching production code.
# ---------------------------------------------------------------------------


class QuantileCallableAdapter(ForecasterAdapter, SupportsQuantiles):
    """Quantile-capable callable adapter for testing CQR."""

    def __init__(
        self,
        predict_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        predict_quantiles_fn: Callable[
            [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
        ],
        horizon: int,
        n_series: int,
    ) -> None:
        super().__init__(horizon=horizon, n_series=n_series)
        self._predict_fn = predict_fn
        self._predict_quantiles_fn = predict_quantiles_fn

    def predict(self, history: Series) -> Forecast:
        history = self._validate_history(history)
        raw = np.asarray(self._predict_fn(history), dtype=np.float64)
        if raw.shape != (self.n_series, self.horizon):
            raise ValueError(
                f"predict_fn must return shape ({self.n_series}, {self.horizon}), got {raw.shape}"
            )
        return raw[:, np.newaxis, :]

    def predict_quantiles(
        self,
        history: Series,
        quantiles: NDArray[np.floating],
    ) -> Forecast:
        history = self._validate_history(history)
        q_arr = np.asarray(quantiles, dtype=np.float64)
        raw = np.asarray(self._predict_quantiles_fn(history, q_arr), dtype=np.float64)
        n_q = q_arr.shape[0]
        if raw.shape != (self.n_series, n_q, self.horizon):
            raise ValueError(
                f"predict_quantiles_fn must return shape "
                f"({self.n_series}, {n_q}, {self.horizon}), got {raw.shape}"
            )
        return raw


class QuantileCVCallableAdapter(QuantileCallableAdapter, SupportsCrossValidationQuantiles):
    """Variant that also implements SupportsCrossValidationQuantiles.

    The CV implementation simply replays a pre-baked list of calibration
    histories through ``predict_quantiles_fn`` and zips the results with a
    pre-baked truths panel. This lets the CV path of CQR be tested against
    the loop path with identical inputs.
    """

    def __init__(
        self,
        predict_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
        predict_quantiles_fn: Callable[
            [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
        ],
        horizon: int,
        n_series: int,
        cv_histories: list[NDArray[np.floating]],
        cv_truths: NDArray[np.floating],
    ) -> None:
        super().__init__(
            predict_fn=predict_fn,
            predict_quantiles_fn=predict_quantiles_fn,
            horizon=horizon,
            n_series=n_series,
        )
        self._cv_histories = cv_histories
        self._cv_truths = np.asarray(cv_truths, dtype=np.float64)

    def cross_validate_quantiles(
        self,
        n_windows: int,
        step_size: int,
        quantiles: NDArray[np.floating],
        refit: bool | int = False,
    ) -> tuple[Forecast, Forecast]:
        if n_windows > len(self._cv_histories):
            raise ValueError(
                f"n_windows={n_windows} exceeds bound histories ({len(self._cv_histories)})"
            )
        # Each call: (n_series, n_q, horizon).
        raw = np.stack(
            [self.predict_quantiles(h, quantiles) for h in self._cv_histories[:n_windows]],
            axis=0,
        )  # (n_windows, n_series, n_q, horizon)
        # -> (n_series, n_windows, horizon, n_q): quantiles on the LAST axis.
        preds = np.transpose(raw, (1, 0, 3, 2))
        truths = self._cv_truths[:, :n_windows, :]
        return preds, truths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_oracle_adapter(
    n_series: int,
    horizon: int,
    signal: np.ndarray,
    noise_std: float,
) -> QuantileCallableAdapter:
    """Adapter with mean = signal level and Gaussian quantiles at fixed sigma.

    Both the point predictor and the quantile predictor know the true
    underlying signal level (broadcast to the requested horizon).
    """

    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.broadcast_to(signal, (n_series, horizon)).copy()

    def predict_quantiles_fn(history: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        # quantiles: (n_q,)
        z = norm.ppf(quantiles)  # (n_q,)
        # broadcast to (n_series, n_q, horizon)
        out = signal[:, np.newaxis, np.newaxis] + z[np.newaxis, :, np.newaxis] * noise_std
        return np.broadcast_to(out, (n_series, quantiles.shape[0], horizon)).copy()

    return QuantileCallableAdapter(
        predict_fn=predict_fn,
        predict_quantiles_fn=predict_quantiles_fn,
        horizon=horizon,
        n_series=n_series,
    )


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------


class TestCoverageGuarantee:
    """Empirical coverage matches 1 - alpha on synthetic data."""

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
    def test_marginal_coverage_symmetric(self, n_series: int, horizon: int, alpha: float) -> None:
        rng = np.random.default_rng(0)
        noise_std = 1.0
        n_cal = 500
        n_test = 500
        T = 30

        signal = rng.standard_normal(n_series) * 5.0  # per-series level
        adapter = _gaussian_oracle_adapter(
            n_series=n_series,
            horizon=horizon,
            signal=signal,
            noise_std=noise_std,
        )

        # Build calibration data
        cal_histories: list[np.ndarray] = []
        cal_truths_list: list[np.ndarray] = []
        for _ in range(n_cal):
            history = signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, T))
            truth = signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, horizon))
            cal_histories.append(history)
            cal_truths_list.append(truth)
        cal_truths = np.stack(cal_truths_list, axis=1)  # (n_series, n_cal, horizon)

        method = ConformalizedQuantileRegression(adapter, alpha=alpha)
        method.calibrate(cal_histories, cal_truths)

        covered = 0
        total = 0
        for _ in range(n_test):
            history = signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, T))
            truth = signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, horizon))
            result = method.predict(history)
            lower = result.interval[..., 0]
            upper = result.interval[..., 1]
            truth_3d = truth[:, np.newaxis, :]
            in_interval = (truth_3d >= lower) & (truth_3d <= upper)
            covered += in_interval.sum()
            total += in_interval.size

        empirical_coverage = covered / total
        assert abs(empirical_coverage - (1 - alpha)) < 0.03, (
            f"Empirical coverage {empirical_coverage:.4f} deviates from "
            f"target {1 - alpha:.2f} by more than 0.03"
        )

    def test_marginal_coverage_asymmetric(self) -> None:
        """Asymmetric quantile bounds still produce target coverage."""
        rng = np.random.default_rng(1)
        n_series, horizon = 3, 6
        alpha = 0.2
        alpha_lo, alpha_hi = 0.05, 0.85
        noise_std = 1.0
        n_cal = 500
        n_test = 500
        T = 30

        signal = rng.standard_normal(n_series) * 5.0
        adapter = _gaussian_oracle_adapter(
            n_series=n_series,
            horizon=horizon,
            signal=signal,
            noise_std=noise_std,
        )

        cal_histories = [
            signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, T)) for _ in range(n_cal)
        ]
        cal_truths = np.stack(
            [
                signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, horizon))
                for _ in range(n_cal)
            ],
            axis=1,
        )

        method = ConformalizedQuantileRegression(
            adapter,
            alpha=alpha,
            symmetric=False,
            alpha_lo=alpha_lo,
            alpha_hi=alpha_hi,
        )
        method.calibrate(cal_histories, cal_truths)

        covered = 0
        total = 0
        for _ in range(n_test):
            history = signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, T))
            truth = signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, horizon))
            result = method.predict(history)
            lower = result.interval[..., 0]
            upper = result.interval[..., 1]
            truth_3d = truth[:, np.newaxis, :]
            covered += ((truth_3d >= lower) & (truth_3d <= upper)).sum()
            total += truth_3d.size

        empirical_coverage = covered / total
        assert abs(empirical_coverage - (1 - alpha)) < 0.03, (
            f"Asymmetric empirical coverage {empirical_coverage:.4f} "
            f"deviates from target {1 - alpha:.2f} by more than 0.03"
        )


# ---------------------------------------------------------------------------
# Adaptivity test: CQR vs Split CP under heteroscedastic noise
# ---------------------------------------------------------------------------


class TestAdaptivity:
    """Under heteroscedastic noise, CQR's interval width adapts; Split's doesn't."""

    def test_cqr_widths_more_variable_than_split(self) -> None:
        """
        Heteroscedastic synthetic data: noise_std depends on the last value of
        the history. CQR sees this through its quantile model and produces
        adaptive interval widths; Split CP uses absolute residuals on a single
        mean predictor and produces near-constant widths.

        Assertion: std-of-widths across test points is meaningfully larger for
        CQR than for SplitConformal.
        """
        rng = np.random.default_rng(7)
        n_series, horizon = 1, 1
        alpha = 0.1
        n_cal = 500
        n_test = 500
        T = 30

        def sigma_of(last_val: np.ndarray) -> np.ndarray:
            # noise_std = 0.5 + |last value of history| ranges roughly in [0.5, 5].
            return 0.5 + np.abs(last_val)

        def mean_predict(history: np.ndarray) -> np.ndarray:
            # Mean = last value of history, repeated over horizon.
            return np.repeat(history[:, -1:], horizon, axis=1)

        def quantile_predict(history: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
            # Conditional quantiles: mean ± z * sigma(history[:, -1]).
            z = norm.ppf(quantiles)
            mean = history[:, -1]  # (n_series,)
            sigma = sigma_of(mean)  # (n_series,)
            # Build (n_series, n_q, horizon)
            out = (
                mean[:, np.newaxis, np.newaxis]
                + z[np.newaxis, :, np.newaxis] * sigma[:, np.newaxis, np.newaxis]
            )
            return np.broadcast_to(out, (n_series, quantiles.shape[0], horizon)).copy()

        cqr_adapter = QuantileCallableAdapter(
            predict_fn=mean_predict,
            predict_quantiles_fn=quantile_predict,
            horizon=horizon,
            n_series=n_series,
        )
        split_adapter = CallableAdapter(predict_fn=mean_predict, horizon=horizon, n_series=n_series)

        def make_history_and_truth() -> tuple[np.ndarray, np.ndarray]:
            # Random walk-ish histories: start at random, then add noise so
            # the last value is informative and varies in magnitude.
            base = rng.uniform(-4.0, 4.0, size=(n_series, 1))
            steps = rng.standard_normal((n_series, T - 1)) * 0.5
            history = np.concatenate([base, base + np.cumsum(steps, axis=1)], axis=1)
            mean_future = history[:, -1:]  # last value
            sigma = sigma_of(history[:, -1])
            truth = mean_future + rng.standard_normal((n_series, horizon)) * sigma[:, np.newaxis]
            return history, truth

        cal_histories: list[np.ndarray] = []
        cal_truths_list: list[np.ndarray] = []
        for _ in range(n_cal):
            h, t = make_history_and_truth()
            cal_histories.append(h)
            cal_truths_list.append(t)
        cal_truths = np.stack(cal_truths_list, axis=1)

        cqr = ConformalizedQuantileRegression(cqr_adapter, alpha=alpha)
        cqr.calibrate(cal_histories, cal_truths)

        split = SplitConformal(split_adapter, alpha=alpha)
        split.calibrate(cal_histories, cal_truths)

        cqr_widths = []
        split_widths = []
        for _ in range(n_test):
            h, _ = make_history_and_truth()
            cqr_int = cqr.predict(h).interval
            split_int = split.predict(h).interval
            cqr_widths.append(cqr_int[..., 1] - cqr_int[..., 0])
            split_widths.append(split_int[..., 1] - split_int[..., 0])

        cqr_widths_arr = np.concatenate(cqr_widths, axis=0).ravel()
        split_widths_arr = np.concatenate(split_widths, axis=0).ravel()

        # Split CP gives constant widths (std ≈ 0); CQR adapts (std clearly > 0).
        assert cqr_widths_arr.std() > 5 * split_widths_arr.std(), (
            f"CQR width-std={cqr_widths_arr.std():.4f} not meaningfully larger "
            f"than Split width-std={split_widths_arr.std():.4f}"
        )


# ---------------------------------------------------------------------------
# Shape and determinism
# ---------------------------------------------------------------------------


def _make_const_quantile_adapter(n_series: int, horizon: int) -> QuantileCallableAdapter:
    """Adapter whose predictions are deterministic functions of history."""

    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.repeat(history[:, -1:], horizon, axis=1)

    def predict_quantiles_fn(history: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        z = norm.ppf(quantiles)
        last = history[:, -1]
        out = last[:, np.newaxis, np.newaxis] + z[np.newaxis, :, np.newaxis]
        return np.broadcast_to(out, (n_series, quantiles.shape[0], horizon)).copy()

    return QuantileCallableAdapter(
        predict_fn=predict_fn,
        predict_quantiles_fn=predict_quantiles_fn,
        horizon=horizon,
        n_series=n_series,
    )


class TestShape:
    @pytest.mark.parametrize(
        "n_series,horizon",
        [(1, 1), (1, 6), (3, 1), (3, 6)],
    )
    def test_output_shapes(self, n_series: int, horizon: int) -> None:
        rng = np.random.default_rng(0)
        adapter = _make_const_quantile_adapter(n_series, horizon)
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)

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
    def test_deterministic_intervals(self) -> None:
        n_series, horizon = 3, 6
        rng = np.random.default_rng(42)
        n_cal = 100

        histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]
        truths = rng.standard_normal((n_series, n_cal, horizon))
        test_history = rng.standard_normal((n_series, 30))

        def run_once() -> tuple[np.ndarray, np.ndarray]:
            adapter = _make_const_quantile_adapter(n_series, horizon)
            method = ConformalizedQuantileRegression(adapter, alpha=0.1)
            method.calibrate(histories, truths)
            result = method.predict(test_history)
            return result.point, result.interval

        p1, i1 = run_once()
        p2, i2 = run_once()
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(i1, i2)


# ---------------------------------------------------------------------------
# Calibration errors
# ---------------------------------------------------------------------------


class TestCalibrationErrors:
    def test_predict_before_calibrate_raises(self) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)
        with pytest.raises(CalibrationError, match="calibrate"):
            method.predict(np.ones((1, 10)))

    @pytest.mark.parametrize("alpha", [0.1, 0.2, 0.5])
    def test_too_few_calibration_samples_raises(self, alpha: float) -> None:
        import math

        n_series, horizon = 1, 1
        adapter = _make_const_quantile_adapter(n_series, horizon)
        method = ConformalizedQuantileRegression(adapter, alpha=alpha)

        rng = np.random.default_rng(0)
        min_needed = math.ceil(1.0 / alpha)
        too_few = min_needed - 1

        histories = [rng.standard_normal((n_series, 10)) for _ in range(too_few)]
        truths = rng.standard_normal((n_series, too_few, horizon))

        with pytest.raises(CalibrationError, match="calibration samples"):
            method.calibrate(histories, truths)

    def test_n_windows_on_non_cv_adapter_raises_value_error(self) -> None:
        """Adapter has SupportsQuantiles but not SupportsCrossValidationQuantiles."""
        adapter = _make_const_quantile_adapter(1, 1)
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)
        with pytest.raises(ValueError, match="SupportsCrossValidationQuantiles"):
            method.calibrate(n_windows=10)

    def test_missing_inputs_raises_value_error(self) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)
        with pytest.raises(ValueError, match="histories, truths"):
            method.calibrate()


# ---------------------------------------------------------------------------
# Capability error
# ---------------------------------------------------------------------------


class TestCapability:
    def test_point_only_adapter_raises(self) -> None:
        # Standard CallableAdapter is point-only (no SupportsQuantiles).
        adapter = CallableAdapter(
            predict_fn=lambda h: np.repeat(h[:, -1:], 1, axis=1),
            horizon=1,
            n_series=1,
        )
        with pytest.raises(UnsupportedCapability, match="SupportsQuantiles"):
            ConformalizedQuantileRegression(adapter, alpha=0.1)


# ---------------------------------------------------------------------------
# Constructor: symmetric / asymmetric ergonomics
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_symmetric_sets_quantiles(self) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        method = ConformalizedQuantileRegression(adapter, alpha=0.2)
        assert method.symmetric is True
        assert method.alpha_lo == 0.1
        assert method.alpha_hi == 0.9
        np.testing.assert_array_almost_equal(method.quantiles_, [0.1, 0.9])

    def test_symmetric_with_alpha_lo_raises(self) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        with pytest.raises(ValueError, match="symmetric=True"):
            ConformalizedQuantileRegression(adapter, alpha=0.2, symmetric=True, alpha_lo=0.1)

    def test_symmetric_with_alpha_hi_raises(self) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        with pytest.raises(ValueError, match="symmetric=True"):
            ConformalizedQuantileRegression(adapter, alpha=0.2, symmetric=True, alpha_hi=0.9)

    def test_asymmetric_missing_lo_raises(self) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        with pytest.raises(ValueError, match="symmetric=False"):
            ConformalizedQuantileRegression(adapter, alpha=0.2, symmetric=False, alpha_hi=0.85)

    def test_asymmetric_missing_hi_raises(self) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        with pytest.raises(ValueError, match="symmetric=False"):
            ConformalizedQuantileRegression(adapter, alpha=0.2, symmetric=False, alpha_lo=0.05)

    def test_asymmetric_valid(self) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        method = ConformalizedQuantileRegression(
            adapter,
            alpha=0.2,
            symmetric=False,
            alpha_lo=0.05,
            alpha_hi=0.85,
        )
        assert method.symmetric is False
        assert method.alpha_lo == 0.05
        assert method.alpha_hi == 0.85
        np.testing.assert_array_almost_equal(method.quantiles_, [0.05, 0.85])

    def test_asymmetric_combined_miscoverage_mismatch_raises(self) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        # 0.1 + (1 - 0.85) = 0.25 != 0.1
        with pytest.raises(ValueError, match="must equal alpha"):
            ConformalizedQuantileRegression(
                adapter,
                alpha=0.1,
                symmetric=False,
                alpha_lo=0.1,
                alpha_hi=0.85,
            )

    def test_asymmetric_alpha_lo_at_boundary_raises(self) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        # alpha_lo = 0.5 violates 0 < alpha_lo < 0.5
        with pytest.raises(ValueError, match="alpha_lo"):
            ConformalizedQuantileRegression(
                adapter,
                alpha=0.5,
                symmetric=False,
                alpha_lo=0.5,
                alpha_hi=1.0,
            )

    def test_asymmetric_alpha_hi_below_half_raises(self) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        with pytest.raises(ValueError, match="alpha_lo"):
            ConformalizedQuantileRegression(
                adapter,
                alpha=0.7,
                symmetric=False,
                alpha_lo=0.1,
                alpha_hi=0.4,
            )


class TestInvalidAlpha:
    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0, 1.5])
    def test_invalid_alpha_raises(self, alpha: float) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        with pytest.raises(ValueError, match="alpha"):
            ConformalizedQuantileRegression(adapter, alpha=alpha)


# ---------------------------------------------------------------------------
# Fitted state
# ---------------------------------------------------------------------------


class TestFittedState:
    def test_fitted_attributes_set_after_calibrate(self) -> None:
        n_series, horizon = 2, 3
        adapter = _make_const_quantile_adapter(n_series, horizon)
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)

        assert method.is_calibrated_ is False

        rng = np.random.default_rng(0)
        histories = [rng.standard_normal((n_series, 30)) for _ in range(50)]
        truths = rng.standard_normal((n_series, 50, horizon))
        method.calibrate(histories, truths)

        assert method.is_calibrated_ is True
        assert method.score_quantile_.shape == (n_series, horizon)
        assert method.n_calibration_samples_ == 50

    def test_calibration_result_is_a_snapshot(self) -> None:
        adapter = _make_const_quantile_adapter(1, 2)
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)

        rng = np.random.default_rng(0)
        histories = [rng.standard_normal((1, 30)) for _ in range(50)]
        truths = rng.standard_normal((1, 50, 2))
        cal = method.calibrate(histories, truths)

        original = method.score_quantile_.copy()
        cal.score_quantile[...] = 999.0  # mutate snapshot
        np.testing.assert_array_equal(method.score_quantile_, original)

    def test_diagnostics_includes_path(self) -> None:
        adapter = _make_const_quantile_adapter(1, 1)
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)
        rng = np.random.default_rng(0)
        histories = [rng.standard_normal((1, 30)) for _ in range(50)]
        truths = rng.standard_normal((1, 50, 1))
        cal = method.calibrate(histories, truths)
        assert cal.diagnostics["path"] == "loop"
        assert cal.diagnostics["symmetric"] is True
        assert cal.diagnostics["quantiles_used"] == [0.05, 0.95]


# ---------------------------------------------------------------------------
# CV fast path
# ---------------------------------------------------------------------------


def _make_cv_adapter(
    n_series: int,
    horizon: int,
    cv_histories: list[np.ndarray],
    cv_truths: np.ndarray,
) -> QuantileCVCallableAdapter:
    """CV-capable adapter using the same const quantile shape as the loop tests."""

    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.repeat(history[:, -1:], horizon, axis=1)

    def predict_quantiles_fn(history: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        z = norm.ppf(quantiles)
        last = history[:, -1]
        out = last[:, np.newaxis, np.newaxis] + z[np.newaxis, :, np.newaxis]
        return np.broadcast_to(out, (n_series, quantiles.shape[0], horizon)).copy()

    return QuantileCVCallableAdapter(
        predict_fn=predict_fn,
        predict_quantiles_fn=predict_quantiles_fn,
        horizon=horizon,
        n_series=n_series,
        cv_histories=cv_histories,
        cv_truths=cv_truths,
    )


class TestCrossValidationPath:
    """The CV fast path produces the same calibrated state as the loop path."""

    def test_coverage_via_cv_path(self) -> None:
        """End-to-end coverage check using the CV fast path."""
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        n_cal = 500
        n_test = 500
        T = 30
        noise_std = 1.0

        signal = rng.standard_normal(n_series) * 5.0

        def predict_fn(history: np.ndarray) -> np.ndarray:
            return np.broadcast_to(signal, (n_series, horizon)).copy()

        def predict_quantiles_fn(history: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
            z = norm.ppf(quantiles)
            out = signal[:, np.newaxis, np.newaxis] + z[np.newaxis, :, np.newaxis] * noise_std
            return np.broadcast_to(out, (n_series, quantiles.shape[0], horizon)).copy()

        cal_histories: list[np.ndarray] = []
        cal_truths_list: list[np.ndarray] = []
        for _ in range(n_cal):
            history = signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, T))
            truth = signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, horizon))
            cal_histories.append(history)
            cal_truths_list.append(truth)
        cal_truths = np.stack(cal_truths_list, axis=1)

        adapter = QuantileCVCallableAdapter(
            predict_fn=predict_fn,
            predict_quantiles_fn=predict_quantiles_fn,
            horizon=horizon,
            n_series=n_series,
            cv_histories=cal_histories,
            cv_truths=cal_truths,
        )

        method = ConformalizedQuantileRegression(adapter, alpha=alpha)
        cal = method.calibrate(n_windows=n_cal, step_size=1)
        assert cal.diagnostics["path"] == "cross_validation"
        assert method.is_calibrated_

        covered = 0
        total = 0
        for _ in range(n_test):
            history = signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, T))
            truth = signal[:, np.newaxis] + rng.normal(0, noise_std, (n_series, horizon))
            result = method.predict(history)
            lower = result.interval[..., 0]
            upper = result.interval[..., 1]
            truth_3d = truth[:, np.newaxis, :]
            in_interval = (truth_3d >= lower) & (truth_3d <= upper)
            covered += in_interval.sum()
            total += in_interval.size
        empirical_coverage = covered / total
        assert abs(empirical_coverage - (1 - alpha)) < 0.03

    def test_cv_path_matches_loop_path(self) -> None:
        """CV and loop paths produce numerically equivalent score_quantile_.

        The two paths perform the same arithmetic (predict_quantiles → score →
        empirical quantile), but the CV path stacks via cross_validate_quantiles
        whereas the loop path stacks via predict_quantiles + transpose. Because
        the two call orders are different, floating-point ordering can produce
        bit-level differences. We compare with ``np.allclose``'s default
        tolerance, which is ample.
        """
        n_series, horizon = 2, 3
        rng = np.random.default_rng(7)
        n_cal = 100

        cal_histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]
        cal_truths = rng.standard_normal((n_series, n_cal, horizon))

        adapter = _make_cv_adapter(n_series, horizon, cal_histories, cal_truths)

        method_loop = ConformalizedQuantileRegression(adapter, alpha=0.1)
        cal_loop = method_loop.calibrate(cal_histories, cal_truths)
        assert cal_loop.diagnostics["path"] == "loop"

        method_cv = ConformalizedQuantileRegression(adapter, alpha=0.1)
        cal_cv = method_cv.calibrate(n_windows=n_cal, step_size=1)
        assert cal_cv.diagnostics["path"] == "cross_validation"

        assert np.allclose(cal_loop.score_quantile, cal_cv.score_quantile)
        assert cal_loop.n_calibration_samples == cal_cv.n_calibration_samples


class TestCalibrateDispatch:
    """Routing between loop and CV paths based on input arguments."""

    def _cv_adapter(self) -> QuantileCVCallableAdapter:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        n_cal = 50
        cal_histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]
        cal_truths = rng.standard_normal((n_series, n_cal, horizon))
        return _make_cv_adapter(n_series, horizon, cal_histories, cal_truths)

    def test_n_windows_dispatches_to_cv(self) -> None:
        adapter = self._cv_adapter()
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)
        cal = method.calibrate(n_windows=50, step_size=1)
        assert cal.diagnostics["path"] == "cross_validation"

    def test_histories_dispatches_to_loop(self) -> None:
        adapter = self._cv_adapter()  # has CV capability, but we choose loop path
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)
        cal = method.calibrate(adapter._cv_histories, adapter._cv_truths)
        assert cal.diagnostics["path"] == "loop"

    def test_both_calling_conventions_raise_value_error(self) -> None:
        adapter = self._cv_adapter()
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)
        with pytest.raises(ValueError, match="not both"):
            method.calibrate(
                adapter._cv_histories,
                adapter._cv_truths,
                n_windows=10,
            )

    def test_neither_calling_convention_raises_value_error(self) -> None:
        adapter = self._cv_adapter()
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)
        with pytest.raises(ValueError, match="histories, truths"):
            method.calibrate()

    def test_cv_path_on_non_cv_adapter_raises_value_error(self) -> None:
        """Loop-only adapter + n_windows → ValueError naming the missing capability."""
        loop_only_adapter = _make_const_quantile_adapter(1, 1)
        method = ConformalizedQuantileRegression(loop_only_adapter, alpha=0.1)
        with pytest.raises(ValueError, match="SupportsCrossValidationQuantiles"):
            method.calibrate(n_windows=20)

    def test_cv_path_on_runtime_gated_adapter_raises_unsupported(self) -> None:
        """An adapter that declares the mixin but flips a runtime flag off
        (mimicking NeuralForecast with a point loss) raises UnsupportedCapability
        when the underlying ``cross_validate_quantiles`` is called."""

        class GatedAdapter(QuantileCVCallableAdapter):
            def cross_validate_quantiles(
                self,
                n_windows: int,
                step_size: int,
                quantiles: NDArray[np.floating],
                refit: bool | int = False,
            ) -> tuple[Forecast, Forecast]:
                raise UnsupportedCapability(
                    "Mock NeuralForecast-style adapter: non-probabilistic loss."
                )

        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        cal_histories = [rng.standard_normal((n_series, 30)) for _ in range(20)]
        cal_truths = rng.standard_normal((n_series, 20, horizon))

        def predict_fn(history: np.ndarray) -> np.ndarray:
            return np.repeat(history[:, -1:], horizon, axis=1)

        def predict_quantiles_fn(history: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
            z = norm.ppf(quantiles)
            last = history[:, -1]
            out = last[:, np.newaxis, np.newaxis] + z[np.newaxis, :, np.newaxis]
            return np.broadcast_to(out, (n_series, quantiles.shape[0], horizon)).copy()

        adapter = GatedAdapter(
            predict_fn=predict_fn,
            predict_quantiles_fn=predict_quantiles_fn,
            horizon=horizon,
            n_series=n_series,
            cv_histories=cal_histories,
            cv_truths=cal_truths,
        )
        method = ConformalizedQuantileRegression(adapter, alpha=0.1)
        with pytest.raises(UnsupportedCapability, match="non-probabilistic loss"):
            method.calibrate(n_windows=20)
