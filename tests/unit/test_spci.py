"""Tests for SequentialPredictiveConformalInference (SPCI)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conformal_ts.adapters.callable import CallableAdapter
from conformal_ts.base import (
    CalibrationError,
    Forecast,
    Series,
    UnsupportedCapability,
)
from conformal_ts.capabilities import SupportsCrossValidation
from conformal_ts.methods.spci import SequentialPredictiveConformalInference
from conformal_ts.methods.split import SplitConformal
from conformal_ts.nonconformity.absolute import AbsoluteResidual
from conformal_ts.nonconformity.signed import SignedResidual
from conformal_ts.quantile_regressors.base import QuantileRegressor


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _zero_predict_fn(n_series: int, horizon: int):
    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.zeros((n_series, horizon))

    return predict_fn


def _last_value_predict_fn(horizon: int):
    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.repeat(history[:, -1:], horizon, axis=1)

    return predict_fn


class CVCallableAdapter(CallableAdapter, SupportsCrossValidation):
    """Test-only adapter exposing rolling-origin CV from an owned training panel."""

    def __init__(
        self,
        predict_fn,
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
                raise ValueError(f"training panel too short: cutoff={cutoff} for window {i}.")
            history = self._training_panel[:, :cutoff]
            truth = self._training_panel[:, cutoff : cutoff + self.horizon]
            pred = self.predict(history)
            preds.append(pred)
            truths.append(truth[:, np.newaxis, :])
        return np.concatenate(preds, axis=1), np.concatenate(truths, axis=1)


class MockQuantileRegressor(QuantileRegressor):
    """Returns ``np.quantile(y_train, q)`` regardless of X (unconditional)."""

    n_fits = 0  # class-level counter — tests inspect this

    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._y = np.asarray(y, dtype=np.float64)
        type(self).n_fits += 1

    def predict_quantile(self, X: np.ndarray, q: float) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("MockQuantileRegressor must be fit before predict_quantile.")
        return np.full(X.shape[0], float(np.quantile(self._y, q)))


def _mock_factory() -> QuantileRegressor:
    return MockQuantileRegressor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_iid_dataset(
    rng: np.random.Generator,
    n_series: int,
    horizon: int,
    n_samples: int,
    noise_std: float,
    history_len: int = 30,
) -> tuple[list[Series], np.ndarray, list[np.ndarray]]:
    histories: list[Series] = []
    truth_panels: list[np.ndarray] = []
    for _ in range(n_samples):
        h = rng.normal(0.0, noise_std, (n_series, history_len))
        t = rng.normal(0.0, noise_std, (n_series, horizon))
        histories.append(h)
        truth_panels.append(t)
    truths = np.stack(truth_panels, axis=1)
    return histories, truths, truth_panels


def _empirical_coverage(
    intervals: list[np.ndarray],
    truth_panels: list[np.ndarray],
) -> float:
    covered = 0
    total = 0
    for interval, truth in zip(intervals, truth_panels, strict=True):
        truth_3d = truth[:, np.newaxis, :]
        in_interval = (truth_3d >= interval[..., 0]) & (truth_3d <= interval[..., 1])
        covered += int(in_interval.sum())
        total += int(in_interval.size)
    return covered / total


# ===========================================================================
# Coverage tests (loop path) with MockQuantileRegressor — no QRF dependency
# ===========================================================================


class TestCoverageLoopPath:
    """Empirical coverage with mock regressor matches 1 - alpha within tolerance."""

    @pytest.mark.parametrize(
        "n_series,horizon,alpha",
        [
            (1, 1, 0.1),
            (1, 6, 0.1),
            (3, 1, 0.2),
            (3, 6, 0.2),
        ],
    )
    def test_marginal_coverage(self, n_series: int, horizon: int, alpha: float) -> None:
        rng = np.random.default_rng(0)
        noise_std = 1.0
        window_size = 30
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
            window_size=window_size,
            regressor_factory=_mock_factory,
            beta_grid_size=21,
        )
        method.calibrate(cal_histories, cal_truths)

        ho_histories, _, ho_truth_panels = _make_iid_dataset(
            rng, n_series, horizon, n_holdout, noise_std
        )

        intervals: list[np.ndarray] = []
        for h in ho_histories:
            result = method.predict(h)
            intervals.append(result.interval.copy())

        coverage = _empirical_coverage(intervals, ho_truth_panels)
        assert abs(coverage - (1 - alpha)) < 0.05, (
            f"Empirical coverage {coverage:.4f} deviates from target "
            f"{1 - alpha:.2f} by more than 0.05."
        )


# ===========================================================================
# Coverage tests (CV path) with MockQuantileRegressor
# ===========================================================================


class TestCoverageCVPath:
    def test_marginal_coverage_cv(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        window_size = 30
        n_cal_windows = 200
        n_holdout = 200

        T = n_cal_windows + horizon + 5
        training_panel = rng.normal(0.0, 1.0, (n_series, T))

        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            training_panel=training_panel,
            horizon=horizon,
        )

        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=alpha,
            window_size=window_size,
            regressor_factory=_mock_factory,
            beta_grid_size=21,
        )
        result = method.calibrate(n_windows=n_cal_windows, step_size=1)
        assert result.diagnostics["path"] == "cross_validation"
        assert result.diagnostics["window_size"] == window_size

        ho_histories, _, ho_truth_panels = _make_iid_dataset(rng, n_series, horizon, n_holdout, 1.0)
        intervals: list[np.ndarray] = []
        for h in ho_histories:
            res = method.predict(h)
            intervals.append(res.interval.copy())

        coverage = _empirical_coverage(intervals, ho_truth_panels)
        assert abs(coverage - (1 - alpha)) < 0.05, (
            f"CV-path empirical coverage {coverage:.4f} deviates from target "
            f"{1 - alpha:.2f} by more than 0.05."
        )


# ===========================================================================
# Coverage test with QRF (gated)
# ===========================================================================


class TestCoverageQRF:
    def test_marginal_coverage_qrf(self) -> None:
        pytest.importorskip("quantile_forest")
        from conformal_ts.quantile_regressors.qrf import QRFQuantileRegressor

        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.2  # tail quantile is easier to estimate well with few trees
        window_size = 30
        n_cal = 400
        n_holdout = 200

        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        cal_histories, cal_truths, _ = _make_iid_dataset(rng, n_series, horizon, n_cal, 1.0)

        def factory() -> QuantileRegressor:
            return QRFQuantileRegressor(n_estimators=200, min_samples_leaf=5, random_state=0)

        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=alpha,
            window_size=window_size,
            regressor_factory=factory,
            beta_grid_size=11,
        )
        method.calibrate(cal_histories, cal_truths)

        ho_histories, _, ho_truth_panels = _make_iid_dataset(rng, n_series, horizon, n_holdout, 1.0)
        intervals: list[np.ndarray] = []
        for h in ho_histories:
            res = method.predict(h)
            intervals.append(res.interval.copy())

        coverage = _empirical_coverage(intervals, ho_truth_panels)
        # QRF is non-deterministic and has finite-sample variance.
        assert abs(coverage - (1 - alpha)) < 0.07, (
            f"QRF-based empirical coverage {coverage:.4f} deviates from target "
            f"{1 - alpha:.2f} by more than 0.07."
        )


# ===========================================================================
# β optimization narrows asymmetric intervals
# ===========================================================================


class TestBetaOptimization:
    """On skewed residuals, optimised β produces narrower intervals than β=α/2."""

    def test_optimized_beta_narrower_than_symmetric(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.2
        window_size = 30
        n_cal = 400

        # Strongly right-skewed noise: exponential. Residuals are y - 0 = y.
        cal_histories: list[np.ndarray] = []
        truth_panels: list[np.ndarray] = []
        for _ in range(n_cal):
            h = rng.standard_normal((n_series, 30))
            t = rng.exponential(scale=1.0, size=(n_series, horizon))
            cal_histories.append(h)
            truth_panels.append(t)
        cal_truths = np.stack(truth_panels, axis=1)

        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        # Optimized SPCI (fine grid).
        spci_opt = SequentialPredictiveConformalInference(
            adapter,
            alpha=alpha,
            window_size=window_size,
            regressor_factory=_mock_factory,
            beta_grid_size=41,
        )
        spci_opt.calibrate(cal_histories, cal_truths)

        # "Symmetric" SPCI: a beta_grid_size of 1 is invalid, but a 3-point
        # grid forces a coarse search. To compare directly with the symmetric
        # choice, we evaluate the symmetric-β interval manually below.
        cell_residuals = spci_opt.residuals_[0, :, 0]
        regressor = MockQuantileRegressor()
        regressor.fit(np.zeros((cell_residuals.size, 1)), cell_residuals)
        symmetric_lower = float(regressor.predict_quantile(np.zeros((1, 1)), alpha / 2.0)[0])
        symmetric_upper = float(regressor.predict_quantile(np.zeros((1, 1)), 1.0 - alpha / 2.0)[0])
        symmetric_width = symmetric_upper - symmetric_lower

        history = rng.standard_normal((n_series, 30))
        result_opt = spci_opt.predict(history)
        opt_lower = result_opt.interval[0, 0, 0, 0] - result_opt.point[0, 0, 0]
        opt_upper = result_opt.interval[0, 0, 0, 1] - result_opt.point[0, 0, 0]
        optimized_width = opt_upper - opt_lower

        assert optimized_width < symmetric_width, (
            f"Optimised width {optimized_width:.4f} should be strictly below "
            f"symmetric width {symmetric_width:.4f} on a right-skewed distribution."
        )

    def test_intervals_can_be_asymmetric(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.2
        window_size = 30
        n_cal = 400

        truth_panels = [rng.exponential(scale=1.0, size=(n_series, horizon)) for _ in range(n_cal)]
        cal_truths = np.stack(truth_panels, axis=1)
        cal_histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]

        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=alpha,
            window_size=window_size,
            regressor_factory=_mock_factory,
            beta_grid_size=41,
        )
        method.calibrate(cal_histories, cal_truths)

        result = method.predict(rng.standard_normal((n_series, 30)))
        lower_offset = result.interval[0, 0, 0, 0] - result.point[0, 0, 0]
        upper_offset = result.interval[0, 0, 0, 1] - result.point[0, 0, 0]
        # Asymmetric means |lower_offset| != |upper_offset|.
        assert not math.isclose(abs(lower_offset), abs(upper_offset), rel_tol=0.05), (
            f"Skewed residuals should yield asymmetric offsets, got "
            f"lower={lower_offset:.4f}, upper={upper_offset:.4f}."
        )


# ===========================================================================
# Distribution shift — SPCI adapts as new residuals arrive
# ===========================================================================


class TestDistributionShift:
    def test_spci_outperforms_split_under_shift(self) -> None:
        pytest.importorskip("quantile_forest")
        from conformal_ts.quantile_regressors.qrf import QRFQuantileRegressor

        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        window_size = 30
        n_cal = 300
        n_holdout = 400
        noise_low = 1.0
        noise_high = 3.0

        cal_histories, cal_truths, _ = _make_iid_dataset(rng, n_series, horizon, n_cal, noise_low)

        adapter_split = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method_split = SplitConformal(adapter_split, alpha=alpha)
        method_split.calibrate(cal_histories, cal_truths)

        adapter_spci = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        def factory() -> QuantileRegressor:
            return QRFQuantileRegressor(n_estimators=50, min_samples_leaf=5, random_state=0)

        method_spci = SequentialPredictiveConformalInference(
            adapter_spci,
            alpha=alpha,
            window_size=window_size,
            regressor_factory=factory,
            beta_grid_size=11,
            refit_every=10,
        )
        method_spci.calibrate(cal_histories, cal_truths)

        ho_histories, _, ho_truth_panels = _make_iid_dataset(
            rng, n_series, horizon, n_holdout, noise_high
        )

        split_intervals: list[np.ndarray] = []
        spci_intervals: list[np.ndarray] = []
        for h, t_panel in zip(ho_histories, ho_truth_panels, strict=True):
            split_intervals.append(method_split.predict(h).interval.copy())
            spci_res = method_spci.predict(h)
            spci_intervals.append(spci_res.interval.copy())
            method_spci.update(spci_res.point, t_panel[:, np.newaxis, :])

        half = n_holdout // 2
        cov_split_late = _empirical_coverage(split_intervals[half:], ho_truth_panels[half:])
        cov_spci_late = _empirical_coverage(spci_intervals[half:], ho_truth_panels[half:])

        # Split CP under-covers under shift; SPCI does meaningfully better.
        assert cov_spci_late > cov_split_late + 0.05, (
            f"Expected SPCI ({cov_spci_late:.3f}) to beat split CP "
            f"({cov_split_late:.3f}) by at least 0.05 under shift."
        )


# ===========================================================================
# refit_every accounting
# ===========================================================================


class TestRefitEvery:
    def test_refit_count(self) -> None:
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
            window_size=20,
            regressor_factory=_mock_factory,
            beta_grid_size=5,
            refit_every=5,
        )

        histories, truths, truth_panels = _make_iid_dataset(rng, n_series, horizon, 80, 1.0)
        # Reset class-level counter just before calibration to track refits cleanly.
        MockQuantileRegressor.n_fits = 0
        method.calibrate(histories, truths)
        # Calibration does one _fit_all_regressors call → 1 mock fit (n_series * horizon = 1).
        fits_after_calibrate = MockQuantileRegressor.n_fits

        ho_histories, _, ho_truth_panels = _make_iid_dataset(rng, n_series, horizon, 25, 1.0)
        for h, t_panel in zip(ho_histories, ho_truth_panels, strict=True):
            res = method.predict(h)
            method.update(res.point, t_panel[:, np.newaxis, :])

        # 25 updates with refit_every=5 → 5 refits → 5 additional fits per cell.
        fits_after_updates = MockQuantileRegressor.n_fits - fits_after_calibrate
        n_cells = n_series * horizon
        assert fits_after_updates == 5 * n_cells, (
            f"Expected 5 refits × {n_cells} cells = {5 * n_cells} mock fits during "
            f"updates, got {fits_after_updates}."
        )


# ===========================================================================
# Pluggable regressor
# ===========================================================================


class TestPluggableRegressor:
    def test_custom_factory_used(self) -> None:
        rng = np.random.default_rng(0)

        used = {"count": 0}

        class CountingMock(MockQuantileRegressor):
            def fit(self, X: np.ndarray, y: np.ndarray) -> None:
                used["count"] += 1
                super().fit(X, y)

        def factory() -> QuantileRegressor:
            return CountingMock()

        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=factory,
            beta_grid_size=5,
        )
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)

        # factory is called twice in __init__ (probe + class-name lookup)? No, just once.
        # Then once per (series, horizon) cell at calibration. count >= 1 cell fit.
        assert used["count"] >= 1
        assert isinstance(method.quantile_regressors_[(0, 0)], CountingMock)


# ===========================================================================
# Score enforcement
# ===========================================================================


class TestScoreEnforcement:
    def test_non_signed_score_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(TypeError, match="SignedResidual"):
            SequentialPredictiveConformalInference(
                adapter,
                alpha=0.1,
                window_size=20,
                regressor_factory=_mock_factory,
                score=AbsoluteResidual(),
            )

    def test_explicit_signed_residual_accepted(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
            score=SignedResidual(),
        )
        assert isinstance(method.score_fn, SignedResidual)


# ===========================================================================
# Calibration sample-size validation
# ===========================================================================


class TestSampleSizeValidation:
    def test_too_few_loop_raises_with_both_numbers(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
        )
        # 2 * 20 + ceil(1/0.1) = 50 required; pass 49.
        n_cal = 2 * 20  # insufficient by ceil(1/alpha)
        histories = [rng.standard_normal((1, 10)) for _ in range(n_cal)]
        truths = rng.standard_normal((1, n_cal, 1))
        with pytest.raises(CalibrationError) as exc_info:
            method.calibrate(histories, truths)
        msg = str(exc_info.value)
        assert "window_size" in msg
        assert "20" in msg
        assert "50" in msg  # the required total
        assert str(n_cal) in msg  # what we got

    def test_too_few_cv_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(1, 1),
            training_panel=rng.normal(0, 1.0, (1, 200)),
            horizon=1,
        )
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
        )
        with pytest.raises(CalibrationError, match="window_size"):
            method.calibrate(n_windows=2 * 20)  # under threshold


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
            window_size=20,
            regressor_factory=_mock_factory,
            beta_grid_size=5,
        )

        n_cal = 50
        histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]
        truths = rng.standard_normal((n_series, n_cal, horizon))
        method.calibrate(histories, truths)

        history = rng.standard_normal((n_series, 30))
        result = method.predict(history)
        assert result.point.shape == (n_series, 1, horizon)
        assert result.interval.shape == (n_series, 1, horizon, 2)
        assert result.alpha == 0.1


# ===========================================================================
# Online updates evolve the residual buffer
# ===========================================================================


class TestOnlineUpdates:
    def test_residual_buffer_grows(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
            beta_grid_size=5,
            refit_every=100,  # avoid refit cost during updates
        )
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)
        assert method.n_observations_ == 50
        assert method.residuals_.shape[1] == 50

        for _ in range(7):
            result = method.predict(np.zeros((1, 10)))
            method.update(result.point, np.zeros((1, 1, 1)))

        assert method.n_observations_ == 57
        assert method.residuals_.shape[1] == 57
        assert method.n_calibration_samples_ == 50  # unchanged


# ===========================================================================
# Lifecycle errors
# ===========================================================================


class TestLifecycleErrors:
    def test_predict_before_calibrate_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
        )
        with pytest.raises(CalibrationError, match="calibrate"):
            method.predict(np.zeros((1, 10)))

    def test_update_before_calibrate_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
        )
        with pytest.raises(CalibrationError, match="calibrate"):
            method.update(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))

    def _make_calibrated(self) -> SequentialPredictiveConformalInference:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
            beta_grid_size=5,
        )
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)
        return method

    def test_update_wrong_truth_shape_raises(self) -> None:
        method = self._make_calibrated()
        with pytest.raises(ValueError, match="truth"):
            method.update(np.zeros((1, 1, 1)), np.zeros((1, 1)))

    def test_update_wrong_prediction_shape_raises(self) -> None:
        method = self._make_calibrated()
        with pytest.raises(ValueError, match="prediction"):
            method.update(np.zeros((1, 1)), np.zeros((1, 1, 1)))


# ===========================================================================
# Calibrate dispatch
# ===========================================================================


class TestCalibrateDispatch:
    def test_no_args_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
        )
        with pytest.raises(ValueError, match="histories"):
            method.calibrate()

    def test_both_paths_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(1, 1),
            training_panel=rng.normal(0, 1.0, (1, 200)),
            horizon=1,
        )
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
        )
        histories = [rng.standard_normal((1, 10)) for _ in range(60)]
        truths = rng.standard_normal((1, 60, 1))
        with pytest.raises(ValueError, match="not both"):
            method.calibrate(histories, truths, n_windows=60)

    def test_n_windows_without_cv_support_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
        )
        with pytest.raises(UnsupportedCapability, match="SupportsCrossValidation"):
            method.calibrate(n_windows=80)

    def test_diagnostics_path_loop(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
        )
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 60, 1.0)
        result = method.calibrate(histories, truths)
        assert result.diagnostics["path"] == "loop"
        assert result.diagnostics["window_size"] == 20
        assert result.diagnostics["regressor_class"] == "MockQuantileRegressor"


# ===========================================================================
# Invalid hyperparameters
# ===========================================================================


class TestInvalidHyperparameters:
    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0, 1.5])
    def test_invalid_alpha_raises(self, alpha: float) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="alpha"):
            SequentialPredictiveConformalInference(
                adapter,
                alpha=alpha,
                window_size=20,
                regressor_factory=_mock_factory,
            )

    @pytest.mark.parametrize("window_size", [0, -1])
    def test_invalid_window_size_raises(self, window_size: int) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="window_size"):
            SequentialPredictiveConformalInference(
                adapter,
                alpha=0.1,
                window_size=window_size,
                regressor_factory=_mock_factory,
            )

    @pytest.mark.parametrize("beta_grid_size", [0, 1])
    def test_invalid_beta_grid_size_raises(self, beta_grid_size: int) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="beta_grid_size"):
            SequentialPredictiveConformalInference(
                adapter,
                alpha=0.1,
                window_size=20,
                beta_grid_size=beta_grid_size,
                regressor_factory=_mock_factory,
            )

    @pytest.mark.parametrize("refit_every", [0, -1])
    def test_invalid_refit_every_raises(self, refit_every: int) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="refit_every"):
            SequentialPredictiveConformalInference(
                adapter,
                alpha=0.1,
                window_size=20,
                refit_every=refit_every,
                regressor_factory=_mock_factory,
            )

    def test_bad_factory_return_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)

        def bad_factory():
            return "not a regressor"

        with pytest.raises(TypeError, match="QuantileRegressor"):
            SequentialPredictiveConformalInference(
                adapter,
                alpha=0.1,
                window_size=20,
                regressor_factory=bad_factory,
            )


# ===========================================================================
# Capability requirements
# ===========================================================================


class TestCapabilityRequirements:
    def test_constructs_with_callable_adapter(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
        )
        assert isinstance(method.forecaster, CallableAdapter)


# ===========================================================================
# Refit timing: not during predict
# ===========================================================================


class TestRefitTiming:
    def test_predict_does_not_refit(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=0.1,
            window_size=20,
            regressor_factory=_mock_factory,
            beta_grid_size=5,
            refit_every=10_000,
        )
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        MockQuantileRegressor.n_fits = 0
        method.calibrate(histories, truths)
        fits_after_calibrate = MockQuantileRegressor.n_fits

        for _ in range(5):
            method.predict(np.zeros((1, 10)))
        assert MockQuantileRegressor.n_fits == fits_after_calibrate, (
            "predict() must not trigger any quantile-regressor refits."
        )
