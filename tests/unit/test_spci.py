"""Tests for SequentialPredictiveConformalInference (SPCI)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from conformal_ts.adapters.callable import CallableAdapter
from conformal_ts.base import (
    CalibrationError,
    UnsupportedCapability,
)
from conformal_ts.methods.spci import SequentialPredictiveConformalInference
from conformal_ts.methods.split import SplitConformal
from conformal_ts.nonconformity.absolute import AbsoluteResidual
from conformal_ts.nonconformity.quantile import QuantileScore
from conformal_ts.quantile_regressors.base import QuantileRegressor
from tests._online_helpers import (
    CVCallableAdapter,
    _empirical_coverage,
    _last_value_predict_fn,
    _make_iid_dataset,
    _run_online_cycle,
    _zero_predict_fn,
)

try:
    import quantile_forest  # type: ignore[import-untyped]  # noqa: F401

    _HAS_QF = True
except ImportError:
    _HAS_QF = False

qf_required = pytest.mark.skipif(not _HAS_QF, reason="quantile-forest is not installed")


# ---------------------------------------------------------------------------
# Test-only quantile regressor — returns empirical quantile of y unconditionally.
# Lets us exercise SPCI's full pipeline without quantile-forest.
# ---------------------------------------------------------------------------


class MockQuantileRegressor(QuantileRegressor):
    def __init__(self) -> None:
        self._y: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        self._y = np.asarray(y, dtype=np.float64)

    def predict_quantile(self, X: NDArray[np.floating], q: float) -> NDArray[np.floating]:
        if self._y is None:
            raise RuntimeError("MockQuantileRegressor must be fit first.")
        return np.full(X.shape[0], float(np.quantile(self._y, q)), dtype=np.float64)


def mock_factory() -> QuantileRegressor:
    return MockQuantileRegressor()


# ===========================================================================
# Coverage with mock regressor (no quantile-forest needed)
# ===========================================================================


class TestCoverageLoopPath:
    """Mock regressor returns the unconditional empirical quantile of residuals,
    so SPCI is effectively split-conformal with signed scores. Coverage should
    land near nominal on iid Gaussian data.
    """

    @pytest.mark.parametrize(
        "n_series,horizon,alpha",
        [
            (1, 1, 0.1),
            (1, 6, 0.1),
            (3, 1, 0.1),
            (3, 6, 0.2),
        ],
    )
    def test_marginal_coverage_loop(self, n_series: int, horizon: int, alpha: float) -> None:
        rng = np.random.default_rng(0)
        noise_std = 1.0
        n_cal = 200
        n_holdout = 200

        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        cal_histories, cal_truths, _ = _make_iid_dataset(rng, n_series, horizon, n_cal, noise_std)

        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=alpha,
            window_size=20,
            regressor_factory=mock_factory,
            beta_grid_size=11,
        )
        method.calibrate(cal_histories, cal_truths)

        ho_histories, _, ho_truth_panels = _make_iid_dataset(
            rng, n_series, horizon, n_holdout, noise_std
        )
        # Do NOT update during holdout — the mock returns the empirical
        # quantile of stored residuals, which doesn't depend on the query
        # feature. Updates would still grow the buffer but not change
        # interval width meaningfully under iid data.
        intervals = _run_online_cycle(method, ho_histories, ho_truth_panels, do_update=False)

        coverage = _empirical_coverage(intervals, ho_truth_panels)
        assert abs(coverage - (1 - alpha)) < 0.05, (
            f"Empirical coverage {coverage:.4f} deviates from target "
            f"{1 - alpha:.2f} by more than 0.05."
        )


class TestCoverageCVPath:
    def test_marginal_coverage_cv(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        n_cal = 200
        n_holdout = 200

        # Need enough room in the training panel for n_cal CV windows.
        T = n_cal + horizon + 5
        training_panel = rng.normal(0.0, 1.0, (n_series, T))
        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            training_panel=training_panel,
            horizon=horizon,
        )

        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=alpha,
            window_size=20,
            regressor_factory=mock_factory,
            beta_grid_size=11,
        )
        result = method.calibrate(n_windows=n_cal, step_size=1)
        assert result.diagnostics["path"] == "cross_validation"

        ho_histories, _, ho_truth_panels = _make_iid_dataset(rng, n_series, horizon, n_holdout, 1.0)
        intervals = _run_online_cycle(method, ho_histories, ho_truth_panels, do_update=False)
        coverage = _empirical_coverage(intervals, ho_truth_panels)
        assert abs(coverage - (1 - alpha)) < 0.05


# ===========================================================================
# Coverage with QRF (gated)
# ===========================================================================


@qf_required
class TestCoverageQRF:
    """SPCI with the default QRF regressor achieves nominal coverage."""

    def test_marginal_coverage_qrf(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        n_cal = 400
        n_holdout = 200

        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        cal_histories, cal_truths, _ = _make_iid_dataset(rng, n_series, horizon, n_cal, 1.0)
        # ``beta_grid_size=3`` keeps β optimization mild so selection-bias
        # under-coverage stays small on this finite calibration set.
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=alpha,
            window_size=30,
            beta_grid_size=3,
            refit_every=20,
        )
        method.calibrate(cal_histories, cal_truths)

        ho_histories, _, ho_truth_panels = _make_iid_dataset(rng, n_series, horizon, n_holdout, 1.0)
        intervals = _run_online_cycle(method, ho_histories, ho_truth_panels)
        coverage = _empirical_coverage(intervals, ho_truth_panels)
        # Tolerance absorbs QRF finite-sample variance and randomness as well
        # as residual β-optimization selection bias on iid data.
        assert abs(coverage - (1 - alpha)) < 0.08, (
            f"QRF coverage {coverage:.4f} deviates from {1 - alpha:.2f} by more than 0.08."
        )


# ===========================================================================
# β optimization narrows skewed intervals
# ===========================================================================


class TestBetaOptimization:
    """Width-minimizing β should produce narrower intervals on skewed data
    than forced-symmetric β = α/2.

    We build a residual distribution with a heavy right tail by construction:
    residuals = N(0, 1) + 5 * Exponential(1). The (α/2, 1 - α/2) symmetric
    interval over-covers the left tail and under-covers the right tail; the
    optimized β shifts the interval rightward and is narrower.
    """

    def test_optimized_width_smaller_than_symmetric(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.2  # widen the search space so β has room to optimize
        n_cal = 300

        # Custom regressor: returns the empirical quantile of fitted y so
        # the comparison is reproducible and independent of QRF.
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        # Right-skewed residual distribution: heavy upper tail.
        skewed = rng.exponential(scale=2.0, size=(n_series, n_cal, horizon))
        # Histories don't influence the score for the zero predictor.
        cal_histories = [np.zeros((n_series, 30)) for _ in range(n_cal)]
        cal_truths = skewed  # since prediction = 0, residual = truth.

        # Optimized β (β_grid_size large enough to find the right shift).
        method_opt = SequentialPredictiveConformalInference(
            adapter,
            alpha=alpha,
            window_size=20,
            regressor_factory=mock_factory,
            beta_grid_size=21,
        )
        method_opt.calibrate(cal_histories, cal_truths)

        # Symmetric β = α/2 only.
        method_sym = SequentialPredictiveConformalInference(
            adapter,
            alpha=alpha,
            window_size=20,
            regressor_factory=mock_factory,
            beta_grid_size=2,
        )
        method_sym.calibrate(cal_histories, cal_truths)
        # beta_grid_size=2 gives {0, alpha}; we want a single mid-grid point.
        # Workaround: hard-set the grid via override beta_grid_size after
        # construction is not supported. Instead compare to a manual
        # symmetric β computation.

        # Predict on a fixed history.
        history = np.zeros((n_series, 30))
        opt_interval = method_opt.predict(history).interval

        # Reference: width at symmetric β = α/2.
        # Lower quantile level = α/2; upper quantile level = 1 - α/2.
        y = skewed[0, :, 0]
        q_lo_sym = float(np.quantile(y, alpha / 2.0))
        q_hi_sym = float(np.quantile(y, 1.0 - alpha / 2.0))
        sym_width = q_hi_sym - q_lo_sym

        opt_width = float(opt_interval[0, 0, 0, 1] - opt_interval[0, 0, 0, 0])
        # Skewness should let the optimized width beat the symmetric one
        # by a non-trivial margin. Allow a small numerical buffer.
        assert opt_width <= sym_width + 1e-9, (
            f"Optimized width {opt_width:.4f} should not exceed symmetric width {sym_width:.4f}."
        )
        # And we expect a meaningful reduction on this distribution.
        assert opt_width < sym_width - 0.1, (
            f"Optimized width {opt_width:.4f} should be visibly narrower than "
            f"symmetric width {sym_width:.4f} on skewed data."
        )


# ===========================================================================
# Distribution shift: SPCI stays competitive
# ===========================================================================


class TestDistributionShift:
    """SPCI's online residual updates + refits keep coverage competitive under
    a noise-scale shift. Compared against SplitConformal (which doesn't
    adapt), SPCI should land closer to nominal in the post-shift second half.
    """

    def test_spci_outperforms_split_under_shift(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        n_cal = 200
        n_holdout = 200
        noise_low = 1.0
        noise_high = 3.0

        cal_histories, cal_truths, _ = _make_iid_dataset(rng, n_series, horizon, n_cal, noise_low)

        adapter_split = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        adapter_spci = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        method_split = SplitConformal(adapter_split, alpha=alpha)
        method_split.calibrate(cal_histories, cal_truths)

        method_spci = SequentialPredictiveConformalInference(
            adapter_spci,
            alpha=alpha,
            window_size=20,
            regressor_factory=mock_factory,
            beta_grid_size=11,
            refit_every=5,
        )
        method_spci.calibrate(cal_histories, cal_truths)

        # Holdout: high-noise regime.
        ho_histories, _, ho_truth_panels = _make_iid_dataset(
            rng, n_series, horizon, n_holdout, noise_high
        )

        split_intervals: list[np.ndarray] = []
        spci_intervals: list[np.ndarray] = []
        for h, t_panel in zip(ho_histories, ho_truth_panels, strict=True):
            split_intervals.append(method_split.predict(h).interval.copy())
            r = method_spci.predict(h)
            spci_intervals.append(r.interval.copy())
            method_spci.update(r.point, t_panel[:, np.newaxis, :])

        half = n_holdout // 2
        cov_split_late = _empirical_coverage(split_intervals[half:], ho_truth_panels[half:])
        cov_spci_late = _empirical_coverage(spci_intervals[half:], ho_truth_panels[half:])
        # SPCI should be closer to nominal than Split CP after the shift.
        assert abs(cov_spci_late - (1 - alpha)) < abs(cov_split_late - (1 - alpha)), (
            f"SPCI late coverage {cov_spci_late:.3f} should beat split CP "
            f"{cov_split_late:.3f} (target {1 - alpha:.2f})."
        )


# ===========================================================================
# refit_every accounting
# ===========================================================================


class TestRefitCadence:
    """_fit_all_regressors is called once per calibrate, then every
    refit_every update() calls.
    """

    def test_refit_count_matches_refit_every(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=15,
            regressor_factory=mock_factory,
            beta_grid_size=5,
            refit_every=5,
        )
        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 60, 1.0)
        method.calibrate(histories, truths)

        call_count = {"n": 0}
        original = method._fit_all_regressors

        def counting_fit() -> None:
            call_count["n"] += 1
            original()

        method._fit_all_regressors = counting_fit  # type: ignore[method-assign]

        # 25 update cycles. With refit_every=5 we expect exactly 5 refits.
        for _ in range(25):
            pred = method.predict(np.zeros((n_series, 30)))
            t = rng.normal(size=(n_series, 1, horizon))
            method.update(pred.point, t)

        assert call_count["n"] == 5, (
            f"Expected 5 refits in 25 updates with refit_every=5, got {call_count['n']}."
        )

    def test_refit_every_one_refits_every_update(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=10,
            regressor_factory=mock_factory,
            beta_grid_size=5,
            refit_every=1,
        )
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)

        call_count = {"n": 0}
        original = method._fit_all_regressors

        def counting_fit() -> None:
            call_count["n"] += 1
            original()

        method._fit_all_regressors = counting_fit  # type: ignore[method-assign]

        for _ in range(7):
            pred = method.predict(np.zeros((1, 30)))
            method.update(pred.point, np.zeros((1, 1, 1)))

        assert call_count["n"] == 7


# ===========================================================================
# Pluggable regressor
# ===========================================================================


class TestPluggableRegressor:
    def test_custom_factory_is_used(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=10,
            regressor_factory=mock_factory,
        )
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 40, 1.0)
        method.calibrate(histories, truths)

        # Every fitted regressor should be a MockQuantileRegressor.
        for reg in method.quantile_regressors_.values():
            assert isinstance(reg, MockQuantileRegressor)

    def test_factory_returning_non_regressor_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)

        def bad_factory():
            return object()

        with pytest.raises(TypeError, match="QuantileRegressor"):
            SequentialPredictiveConformalInference(
                adapter,
                alpha=0.1,
                regressor_factory=bad_factory,
            )


# ===========================================================================
# Score function enforcement
# ===========================================================================


class TestScoreEnforcement:
    """SPCI requires SignedResidual — other scores are rejected at construction."""

    def test_absolute_residual_rejected(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(TypeError, match="SignedResidual"):
            SequentialPredictiveConformalInference(
                adapter,
                alpha=0.1,
                regressor_factory=mock_factory,
                score=AbsoluteResidual(),
            )

    def test_quantile_score_rejected(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(TypeError, match="SignedResidual"):
            SequentialPredictiveConformalInference(
                adapter,
                alpha=0.1,
                regressor_factory=mock_factory,
                score=QuantileScore(),
            )


# ===========================================================================
# Shape tests
# ===========================================================================


class TestShape:
    @pytest.mark.parametrize(
        "n_series,horizon",
        [(1, 1), (1, 6), (3, 1), (3, 6)],
    )
    def test_output_shapes(self, n_series: int, horizon: int) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(
            predict_fn=_last_value_predict_fn(horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=10,
            regressor_factory=mock_factory,
            beta_grid_size=5,
        )
        n_cal = 40
        histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]
        truths = rng.standard_normal((n_series, n_cal, horizon))
        method.calibrate(histories, truths)

        history = rng.standard_normal((n_series, 30))
        result = method.predict(history)

        assert result.point.shape == (n_series, 1, horizon)
        assert result.interval.shape == (n_series, 1, horizon, 2)
        assert result.alpha == 0.1
        assert method.residuals_.shape == (n_series, n_cal, horizon)


# ===========================================================================
# Asymmetric intervals
# ===========================================================================


class TestAsymmetricIntervals:
    """Under skewed residuals, at least one cell should have visibly
    asymmetric offsets from the point forecast.
    """

    def test_asymmetric_offsets_present(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.2
        n_cal = 200

        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        # Right-skewed truths.
        skewed = rng.exponential(scale=2.0, size=(n_series, n_cal, horizon))
        cal_histories = [np.zeros((n_series, 30)) for _ in range(n_cal)]

        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=alpha,
            window_size=20,
            regressor_factory=mock_factory,
            beta_grid_size=21,
        )
        method.calibrate(cal_histories, skewed)

        history = np.zeros((n_series, 30))
        result = method.predict(history)
        point = result.point
        interval = result.interval

        asym_found = False
        for s in range(n_series):
            for h in range(horizon):
                lower_offset = float(interval[s, 0, h, 0] - point[s, 0, h])
                upper_offset = float(interval[s, 0, h, 1] - point[s, 0, h])
                if not np.isclose(-lower_offset, upper_offset, atol=1e-3):
                    asym_found = True
        assert asym_found, (
            "Expected at least one cell to have asymmetric offsets under skewed data."
        )


# ===========================================================================
# Online updates extend residual buffer
# ===========================================================================


class TestOnlineUpdates:
    def test_residual_buffer_grows(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=10,
            regressor_factory=mock_factory,
        )
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 40, 1.0)
        method.calibrate(histories, truths)
        assert method.n_observations_ == 40

        for _ in range(6):
            pred = method.predict(np.zeros((1, 30)))
            method.update(pred.point, np.zeros((1, 1, 1)))

        assert method.n_observations_ == 46
        assert method.residuals_.shape[1] == 46
        assert method.n_calibration_samples_ == 40  # unchanged


# ===========================================================================
# Lifecycle errors
# ===========================================================================


class TestLifecycleErrors:
    def _make_calibrated(self) -> SequentialPredictiveConformalInference:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter, alpha=0.1, window_size=10, regressor_factory=mock_factory
        )
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 40, 1.0)
        method.calibrate(histories, truths)
        return method

    def test_predict_before_calibrate_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter, alpha=0.1, window_size=10, regressor_factory=mock_factory
        )
        with pytest.raises(CalibrationError, match="calibrate"):
            method.predict(np.zeros((1, 10)))

    def test_update_before_calibrate_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter, alpha=0.1, window_size=10, regressor_factory=mock_factory
        )
        with pytest.raises(CalibrationError, match="calibrate"):
            method.update(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))

    def test_update_with_wrong_truth_shape_raises(self) -> None:
        method = self._make_calibrated()
        prediction = np.zeros((1, 1, 1))
        with pytest.raises(ValueError, match="truth must have shape"):
            method.update(prediction, np.zeros((1, 1)))

    def test_update_with_wrong_prediction_shape_raises(self) -> None:
        method = self._make_calibrated()
        truth = np.zeros((1, 1, 1))
        with pytest.raises(ValueError, match="prediction must have shape"):
            method.update(np.zeros((1, 1)), truth)


# ===========================================================================
# Invalid parameters
# ===========================================================================


class TestInvalidParameters:
    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0, 1.5])
    def test_invalid_alpha_raises(self, alpha: float) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="alpha"):
            SequentialPredictiveConformalInference(
                adapter, alpha=alpha, regressor_factory=mock_factory
            )

    def test_invalid_window_size_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="window_size"):
            SequentialPredictiveConformalInference(
                adapter, alpha=0.1, window_size=0, regressor_factory=mock_factory
            )

    def test_invalid_beta_grid_size_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="beta_grid_size"):
            SequentialPredictiveConformalInference(
                adapter,
                alpha=0.1,
                window_size=10,
                beta_grid_size=1,
                regressor_factory=mock_factory,
            )

    def test_invalid_refit_every_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="refit_every"):
            SequentialPredictiveConformalInference(
                adapter,
                alpha=0.1,
                window_size=10,
                refit_every=0,
                regressor_factory=mock_factory,
            )


# ===========================================================================
# Window-size validation in calibrate
# ===========================================================================


class TestCalibrationMinSamples:
    def test_too_few_calibration_samples_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=10,
            regressor_factory=mock_factory,
        )
        rng = np.random.default_rng(0)
        min_required = 2 * 10 + math.ceil(1 / 0.1)
        too_few = min_required - 1
        histories = [rng.standard_normal((1, 10)) for _ in range(too_few)]
        truths = rng.standard_normal((1, too_few, 1))
        with pytest.raises(CalibrationError, match=str(min_required)):
            method.calibrate(histories, truths)


# ===========================================================================
# Capability requirements
# ===========================================================================


class TestCapabilityRequirements:
    def test_constructs_with_callable_adapter(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter, alpha=0.1, regressor_factory=mock_factory
        )
        assert isinstance(method.forecaster, CallableAdapter)


# ===========================================================================
# Calibrate dispatch
# ===========================================================================


class TestCalibrateDispatch:
    def test_no_args_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter, alpha=0.1, regressor_factory=mock_factory
        )
        with pytest.raises(ValueError, match="histories"):
            method.calibrate()

    def test_both_paths_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(1, 1),
            training_panel=rng.normal(0, 1.0, (1, 80)),
            horizon=1,
        )
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=10,
            regressor_factory=mock_factory,
        )
        histories = [rng.standard_normal((1, 10)) for _ in range(40)]
        truths = rng.standard_normal((1, 40, 1))
        with pytest.raises(ValueError, match="not both"):
            method.calibrate(histories, truths, n_windows=40)

    def test_n_windows_without_cv_support_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter, alpha=0.1, regressor_factory=mock_factory
        )
        with pytest.raises(UnsupportedCapability, match="SupportsCrossValidation"):
            method.calibrate(n_windows=200)

    def test_loop_path_diagnostics(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=10,
            regressor_factory=mock_factory,
        )
        rng = np.random.default_rng(0)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 40, 1.0)
        result = method.calibrate(histories, truths)
        assert result.diagnostics["path"] == "loop"
        assert result.diagnostics["window_size"] == 10
        assert result.diagnostics["regressor_class"] == "MockQuantileRegressor"

    def test_cv_path_diagnostics(self) -> None:
        rng = np.random.default_rng(0)
        T = 200
        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(1, 1),
            training_panel=rng.normal(0, 1.0, (1, T)),
            horizon=1,
        )
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=10,
            regressor_factory=mock_factory,
        )
        result = method.calibrate(n_windows=100, step_size=1)
        assert result.diagnostics["path"] == "cross_validation"


# ===========================================================================
# Fitted state convention
# ===========================================================================


class TestFittedState:
    def test_fitted_attributes_after_calibrate(self) -> None:
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
            regressor_factory=mock_factory,
        )
        assert method.is_calibrated_ is False

        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 50, 1.0)
        method.calibrate(histories, truths)

        assert method.is_calibrated_ is True
        assert method.residuals_.shape == (n_series, 50, horizon)
        assert method.n_calibration_samples_ == 50
        assert method.n_observations_ == 50
        # one regressor per (series, horizon) cell
        assert len(method.quantile_regressors_) == n_series * horizon

    def test_stores_calibration_data(self) -> None:
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
            regressor_factory=mock_factory,
        )
        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 50, 1.0)
        method.calibrate(histories, truths)

        assert method.predictions_calibration_.shape == (n_series, 50, horizon)
        assert method.truths_calibration_.shape == (n_series, 50, horizon)
        np.testing.assert_allclose(method.truths_calibration_, truths)

    def test_calibration_data_is_defensive_copy(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=10,
            regressor_factory=mock_factory,
        )
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)

        residuals_before = method.residuals_.copy()
        method.predictions_calibration_[...] = 0.0
        method.truths_calibration_[...] = 0.0
        np.testing.assert_array_equal(method.residuals_, residuals_before)


class TestIntervalsFromPredictions:
    def test_shape_matches_panel(self) -> None:
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
            regressor_factory=mock_factory,
            beta_grid_size=5,
        )
        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 50, 1.0)
        method.calibrate(histories, truths)

        intervals = method._intervals_from_predictions(method.predictions_calibration_)
        assert intervals.shape == (n_series, 50, horizon, 2)
        # Lower <= upper for every cell.
        assert (intervals[..., 0] <= intervals[..., 1]).all()
