"""Tests for AggregatedAdaptiveConformalInference (AgACI)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conformal_ts.adapters.callable import CallableAdapter
from conformal_ts.aggregators import EWA, OnlineAggregator
from conformal_ts.base import (
    CalibrationError,
    Forecast,
    Series,
    UnsupportedCapability,
)
from conformal_ts.capabilities import SupportsCrossValidation
from conformal_ts.methods.aci import AdaptiveConformalInference
from conformal_ts.methods.agaci import AggregatedAdaptiveConformalInference


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
    """Test-only callable adapter that exposes rolling-origin CV via a panel."""

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
                raise ValueError(
                    f"training panel too short: cutoff={cutoff} for window {i}, "
                    f"n_windows={n_windows}, step_size={step_size}, horizon={self.horizon}, T={T}."
                )
            history = self._training_panel[:, :cutoff]
            truth = self._training_panel[:, cutoff : cutoff + self.horizon]
            pred = self.predict(history)
            preds.append(pred)
            truths.append(truth[:, np.newaxis, :])
        return np.concatenate(preds, axis=1), np.concatenate(truths, axis=1)

    def cv_window_history(self, n_windows: int, step_size: int, window_index: int) -> np.ndarray:
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
    method: AggregatedAdaptiveConformalInference,
    holdout_histories: list[Series],
    holdout_truth_panels: list[np.ndarray],
) -> list[np.ndarray]:
    intervals: list[np.ndarray] = []
    for h, t_panel in zip(holdout_histories, holdout_truth_panels, strict=True):
        result = method.predict(h)
        intervals.append(result.interval.copy())
        method.update(result.point, t_panel[:, np.newaxis, :])
    return intervals


def _empirical_coverage(intervals: list[np.ndarray], truth_panels: list[np.ndarray]) -> float:
    covered = 0
    total = 0
    for interval, truth in zip(intervals, truth_panels, strict=True):
        truth_3d = truth[:, np.newaxis, :]
        in_interval = (truth_3d >= interval[..., 0]) & (truth_3d <= interval[..., 1])
        covered += int(in_interval.sum())
        total += int(in_interval.size)
    return covered / total


# ===========================================================================
# Coverage — loop calibration path
# ===========================================================================


class TestCoverageLoopPath:
    """AgACI achieves nominal coverage on iid Gaussian noise."""

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
        n_cal = 200
        n_holdout = 300

        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        cal_histories, cal_truths, _ = _make_iid_dataset(rng, n_series, horizon, n_cal, noise_std)

        method = AggregatedAdaptiveConformalInference(adapter, alpha=alpha)
        method.calibrate(cal_histories, cal_truths)

        ho_histories, _, ho_truth_panels = _make_iid_dataset(
            rng, n_series, horizon, n_holdout, noise_std
        )
        intervals = _run_online_cycle(method, ho_histories, ho_truth_panels)

        coverage = _empirical_coverage(intervals, ho_truth_panels)
        assert abs(coverage - (1 - alpha)) < 0.04, (
            f"Empirical coverage {coverage:.4f} deviates from target "
            f"{1 - alpha:.2f} by more than 0.04."
        )


# ===========================================================================
# Coverage — CV calibration path
# ===========================================================================


class TestCoverageCVPath:
    """AgACI achieves nominal coverage via the CV calibration path."""

    def test_marginal_coverage_cv(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        noise_std = 1.0
        n_cal_windows = 200
        n_holdout = 300

        T = n_cal_windows + horizon + 5
        training_panel = rng.normal(0.0, noise_std, (n_series, T))

        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            training_panel=training_panel,
            horizon=horizon,
        )

        method = AggregatedAdaptiveConformalInference(adapter, alpha=alpha)
        result = method.calibrate(n_windows=n_cal_windows, step_size=1)
        assert result.diagnostics["path"] == "cross_validation"

        ho_histories, _, ho_truth_panels = _make_iid_dataset(
            rng, n_series, horizon, n_holdout, noise_std
        )
        intervals = _run_online_cycle(method, ho_histories, ho_truth_panels)

        coverage = _empirical_coverage(intervals, ho_truth_panels)
        assert abs(coverage - (1 - alpha)) < 0.04, (
            f"Empirical coverage {coverage:.4f} deviates from target "
            f"{1 - alpha:.2f} by more than 0.04."
        )


# ===========================================================================
# Single-expert reduces to ACI
# ===========================================================================


class TestSingleExpertReducesToACI:
    """With K=1 and identical setup, AgACI's intervals match ACI's bounds."""

    def test_single_expert_matches_aci(self) -> None:
        # Small gamma + many samples keeps alpha_t safely inside (0, 1)
        # throughout calibration, so neither method hits the saturation regime
        # where ACI's max_float thresholds and AgACI's clipping diverge.
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        alpha = 0.1
        gamma = 0.005
        n_cal = 100

        cal_histories, cal_truths, _ = _make_iid_dataset(rng, n_series, horizon, n_cal, 1.0)

        adapter_aci = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        adapter_agaci = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        method_aci = AdaptiveConformalInference(adapter_aci, alpha=alpha, gamma=gamma)
        method_aci.calibrate(cal_histories, cal_truths)

        # A clip wider than any plausible bound (finite to avoid inf-pinball)
        # is a no-op here since alpha_t stays bounded.
        method_agaci = AggregatedAdaptiveConformalInference(
            adapter_agaci,
            alpha=alpha,
            gammas=(gamma,),
            interval_clip_lower=-1e10,
            interval_clip_upper=1e10,
        )
        method_agaci.calibrate(cal_histories, cal_truths)

        # Sanity: alpha_t stays well inside (0, 1) for this setup.
        assert method_aci.alpha_t_.min() > 0.01
        assert method_aci.alpha_t_.max() < 0.99

        # alpha_t (ACI) and alpha_t_per_expert_[0] (AgACI) should match.
        np.testing.assert_allclose(
            method_agaci.alpha_t_per_expert_[0],
            method_aci.alpha_t_,
            rtol=1e-12,
        )

        # Interval equality on a fresh history.
        history = rng.normal(0.0, 1.0, (n_series, 30))
        agaci_interval = method_agaci.predict(history).interval
        aci_interval = method_aci.predict(history).interval

        np.testing.assert_allclose(agaci_interval, aci_interval, rtol=1e-10, atol=1e-10)


# ===========================================================================
# Distribution shift
# ===========================================================================


class TestDistributionShift:
    """AgACI adapts to a noise-scale shift, matching ACI's behaviour."""

    def test_agaci_recovers_under_shift(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        n_cal = 300
        n_holdout = 400
        noise_low = 1.0
        noise_high = 3.0

        cal_histories, cal_truths, _ = _make_iid_dataset(rng, n_series, horizon, n_cal, noise_low)

        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AggregatedAdaptiveConformalInference(adapter, alpha=alpha)
        method.calibrate(cal_histories, cal_truths)

        ho_histories, _, ho_truth_panels = _make_iid_dataset(
            rng, n_series, horizon, n_holdout, noise_high
        )
        intervals = _run_online_cycle(method, ho_histories, ho_truth_panels)

        half = n_holdout // 2
        cov_late = _empirical_coverage(intervals[half:], ho_truth_panels[half:])
        # AgACI should land within 0.05 of nominal once it adapts. The grid
        # contains gamma values up to 0.2, which adapt fastest under shift.
        assert abs(cov_late - (1 - alpha)) < 0.05, (
            f"AgACI late coverage {cov_late:.3f} differs from nominal "
            f"{1 - alpha:.2f} by more than 0.05."
        )


# ===========================================================================
# Path equivalence: CV vs loop produce same fitted state
# ===========================================================================


class TestPathEquivalence:
    """alpha_t_per_expert_ and aggregator state match between CV and loop paths."""

    def test_state_close_between_paths(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        alpha = 0.1
        n_cal_windows = 60
        T = n_cal_windows + horizon + 5
        training_panel = rng.normal(0.0, 1.0, (n_series, T))

        cv_adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            training_panel=training_panel,
            horizon=horizon,
        )
        method_cv = AggregatedAdaptiveConformalInference(cv_adapter, alpha=alpha)
        method_cv.calibrate(n_windows=n_cal_windows, step_size=1)

        cv_preds, cv_truths = cv_adapter.cross_validate(n_windows=n_cal_windows, step_size=1)
        cal_histories = [
            cv_adapter.cv_window_history(n_cal_windows, 1, w) for w in range(n_cal_windows)
        ]
        loop_adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method_loop = AggregatedAdaptiveConformalInference(loop_adapter, alpha=alpha)
        method_loop.calibrate(cal_histories, cv_truths)

        np.testing.assert_allclose(
            method_cv.alpha_t_per_expert_, method_loop.alpha_t_per_expert_, rtol=1e-12
        )
        np.testing.assert_allclose(method_cv.scores_, method_loop.scores_, rtol=1e-12)
        np.testing.assert_allclose(
            method_cv.aggregator_lower_.cumulative_losses_,
            method_loop.aggregator_lower_.cumulative_losses_,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            method_cv.aggregator_upper_.cumulative_losses_,
            method_loop.aggregator_upper_.cumulative_losses_,
            rtol=1e-12,
        )


# ===========================================================================
# Per-expert alpha_t evolution
# ===========================================================================


class TestPerExpertAlphaT:
    """The K rows of alpha_t_per_expert_ diverge as expected."""

    def test_per_expert_alpha_t_diverges(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1, gammas=(0.001, 0.1, 0.2))
        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 100, 1.0)
        method.calibrate(histories, truths)

        # The K rows should not all be identical: differing gammas drive
        # distinct alpha_t trajectories. We deliberately don't assert that
        # higher gamma is *farther* from the initial alpha at the end of
        # calibration — high-gamma experts random-walk and can return near
        # their starting value by chance.
        rows = method.alpha_t_per_expert_  # (3, n_series, horizon)
        assert not np.allclose(rows[0], rows[1]), (
            "Per-expert alpha_t should diverge across distinct gammas."
        )
        assert not np.allclose(rows[0], rows[2]), (
            "Per-expert alpha_t should diverge across distinct gammas."
        )


# ===========================================================================
# Aggregator weight evolution
# ===========================================================================


class TestAggregatorWeights:
    """Aggregator weights evolve away from uniform as cumulative losses accumulate."""

    def test_weights_become_non_uniform(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 200, 1.0)
        method.calibrate(histories, truths)

        K = method.n_experts
        uniform = np.full((K, n_series, horizon), 1.0 / K)
        w_lower = method.aggregator_lower_.weights()
        w_upper = method.aggregator_upper_.weights()

        assert not np.allclose(w_lower, uniform, atol=1e-3), (
            "aggregator_lower_ weights should drift away from uniform after calibration."
        )
        assert not np.allclose(w_upper, uniform, atol=1e-3), (
            "aggregator_upper_ weights should drift away from uniform after calibration."
        )


# ===========================================================================
# Lower vs upper aggregators differ under asymmetric noise
# ===========================================================================


class TestAsymmetricNoise:
    """Lower- and upper-bound aggregators converge to different weights under skewed noise.

    Setup: truths have right-skewed Gamma noise added to a zero baseline. The
    point forecaster predicts the mean. Upper bounds need to widen against
    heavy upper-tail outliers; lower bounds barely need to move because the
    Gamma distribution has a hard floor near zero. This asymmetry should drive
    the two aggregators to different weight distributions.
    """

    def test_lower_and_upper_weights_differ(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        n_cal = 400

        # Right-skewed Gamma noise (mean = shape * scale = 1.0, heavy upper tail).
        shape, scale = 1.5, 1.0
        noise_mean = shape * scale

        # Histories don't matter — the predictor returns zero (the residual mean
        # absorbs the gamma mean below).
        histories = [np.zeros((n_series, 30)) for _ in range(n_cal)]
        truth_panels = [
            rng.gamma(shape, scale, size=(n_series, horizon)) - noise_mean for _ in range(n_cal)
        ]
        truths = np.stack(truth_panels, axis=1)

        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AggregatedAdaptiveConformalInference(adapter, alpha=alpha)
        method.calibrate(histories, truths)

        w_lower = method.aggregator_lower_.weights()
        w_upper = method.aggregator_upper_.weights()
        assert not np.allclose(w_lower, w_upper, atol=1e-3), (
            "Lower and upper aggregator weights should differ under asymmetric noise. "
            f"max |w_lower - w_upper| = {float(np.abs(w_lower - w_upper).max()):.4f}."
        )


# ===========================================================================
# Clipping prevents infinite bounds
# ===========================================================================


class TestClipping:
    """Even if alpha_t is driven outside [0, 1], reported bounds remain finite."""

    def test_clipped_bounds_under_extreme_alpha_t(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AggregatedAdaptiveConformalInference(
            adapter,
            alpha=0.1,
            interval_clip_lower=-100.0,
            interval_clip_upper=100.0,
        )
        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 50, 1.0)
        method.calibrate(histories, truths)

        # Force one expert's alpha_t below 0 to saturate threshold to max float
        # without explicit clipping.
        method.alpha_t_per_expert_[:] = -0.5

        result = method.predict(np.zeros((n_series, 10)))
        lower = result.interval[..., 0]
        upper = result.interval[..., 1]

        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))
        # All bounds clamped to [-100, 100].
        assert np.all(lower >= -100.0 - 1e-9)
        assert np.all(upper <= 100.0 + 1e-9)

    def test_auto_clip_brackets_data(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        _, truths, _ = _make_iid_dataset(rng, n_series, horizon, 50, 1.0)
        histories = [rng.normal(0.0, 1.0, (n_series, 30)) for _ in range(50)]
        method.calibrate(histories, truths)

        truth_min = float(np.min(truths))
        truth_max = float(np.max(truths))
        assert method.interval_clip_lower_ < truth_min
        assert method.interval_clip_upper_ > truth_max

    def test_manual_clip_honored(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AggregatedAdaptiveConformalInference(
            adapter,
            alpha=0.1,
            interval_clip_lower=-100.0,
            interval_clip_upper=100.0,
        )
        rng = np.random.default_rng(0)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)
        assert method.interval_clip_lower_ == -100.0
        assert method.interval_clip_upper_ == 100.0


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
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)

        n_cal = 50
        histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]
        truths = rng.standard_normal((n_series, n_cal, horizon))
        method.calibrate(histories, truths)

        history = rng.standard_normal((n_series, 30))
        result = method.predict(history)

        assert result.point.shape == (n_series, 1, horizon)
        assert result.interval.shape == (n_series, 1, horizon, 2)
        assert result.alpha == 0.1
        assert method.alpha_t_per_expert_.shape == (
            method.n_experts,
            n_series,
            horizon,
        )


# ===========================================================================
# Lifecycle errors
# ===========================================================================


class TestLifecycleErrors:
    def _make_calibrated(self) -> AggregatedAdaptiveConformalInference:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)
        return method

    def test_predict_before_calibrate_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        with pytest.raises(CalibrationError, match="calibrate"):
            method.predict(np.zeros((1, 10)))

    def test_update_before_calibrate_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
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
            AggregatedAdaptiveConformalInference(adapter, alpha=alpha)

    @pytest.mark.parametrize("gammas", [[], (0.0,), (-0.01, 0.05), [0.05, 0.0, 0.1]])
    def test_invalid_gammas_raises(self, gammas) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="gammas"):
            AggregatedAdaptiveConformalInference(adapter, alpha=0.1, gammas=gammas)

    def test_only_one_clip_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="interval_clip"):
            AggregatedAdaptiveConformalInference(adapter, alpha=0.1, interval_clip_lower=10.0)
        with pytest.raises(ValueError, match="interval_clip"):
            AggregatedAdaptiveConformalInference(adapter, alpha=0.1, interval_clip_upper=10.0)

    def test_clip_lower_ge_upper_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="interval_clip_lower"):
            AggregatedAdaptiveConformalInference(
                adapter,
                alpha=0.1,
                interval_clip_lower=10.0,
                interval_clip_upper=10.0,
            )
        with pytest.raises(ValueError, match="interval_clip_lower"):
            AggregatedAdaptiveConformalInference(
                adapter,
                alpha=0.1,
                interval_clip_lower=20.0,
                interval_clip_upper=10.0,
            )


# ===========================================================================
# Capability requirements
# ===========================================================================


class TestCapabilityRequirements:
    def test_constructs_with_callable_adapter(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        assert isinstance(method.forecaster, CallableAdapter)


# ===========================================================================
# Calibrate dispatch
# ===========================================================================


class TestCalibrateDispatch:
    def test_no_args_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        with pytest.raises(ValueError, match="histories"):
            method.calibrate()

    def test_both_paths_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(1, 1),
            training_panel=rng.normal(0, 1.0, (1, 50)),
            horizon=1,
        )
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        histories = [rng.standard_normal((1, 10)) for _ in range(20)]
        truths = rng.standard_normal((1, 20, 1))
        with pytest.raises(ValueError, match="not both"):
            method.calibrate(histories, truths, n_windows=10)

    def test_n_windows_without_cv_support_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        with pytest.raises(UnsupportedCapability, match="SupportsCrossValidation"):
            method.calibrate(n_windows=20)

    def test_too_few_calibration_samples_loop_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        rng = np.random.default_rng(0)
        too_few = math.ceil(1.0 / 0.1) - 1
        histories = [rng.standard_normal((1, 10)) for _ in range(too_few)]
        truths = rng.standard_normal((1, too_few, 1))
        with pytest.raises(CalibrationError, match="calibration samples"):
            method.calibrate(histories, truths)

    def test_too_few_calibration_samples_cv_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(1, 1),
            training_panel=rng.normal(0, 1.0, (1, 50)),
            horizon=1,
        )
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        too_few = math.ceil(1.0 / 0.1) - 1
        with pytest.raises(CalibrationError, match="calibration samples"):
            method.calibrate(n_windows=too_few)

    def test_loop_path_diagnostics(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        rng = np.random.default_rng(0)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        result = method.calibrate(histories, truths)
        assert result.diagnostics["path"] == "loop"
        assert result.diagnostics["gammas"] == method.gammas


# ===========================================================================
# Custom aggregator: pluggable design
# ===========================================================================


class TestCustomAggregator:
    """A user can plug in their own OnlineAggregator subclass."""

    def test_custom_aggregator_is_used(self) -> None:
        class UniformAggregator(OnlineAggregator):
            def weights(self) -> np.ndarray:
                return np.full(
                    (self.n_experts, self.n_series, self.horizon),
                    1.0 / self.n_experts,
                    dtype=np.float64,
                )

        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        def factory(K: int, S: int, H: int) -> OnlineAggregator:
            return UniformAggregator(K, S, H)

        method = AggregatedAdaptiveConformalInference(
            adapter,
            alpha=0.1,
            aggregator_factory=factory,
        )
        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 100, 1.0)
        method.calibrate(histories, truths)

        assert isinstance(method.aggregator_lower_, UniformAggregator)
        assert isinstance(method.aggregator_upper_, UniformAggregator)

        K = method.n_experts
        w_lower = method.aggregator_lower_.weights()
        w_upper = method.aggregator_upper_.weights()
        np.testing.assert_allclose(w_lower, np.full_like(w_lower, 1.0 / K), rtol=1e-12)
        np.testing.assert_allclose(w_upper, np.full_like(w_upper, 1.0 / K), rtol=1e-12)

        # The interval should equal the unweighted mean of per-expert bounds.
        history = rng.standard_normal((n_series, 30))
        result = method.predict(history)
        # Re-derive the per-expert bounds and compare the simple mean.
        point = method.forecaster.predict(history)
        point_squeezed = point[:, 0, :]
        lower_bounds, upper_bounds = method._per_expert_bounds(point_squeezed, method.scores_)
        expected_lower = lower_bounds.mean(axis=0)
        expected_upper = upper_bounds.mean(axis=0)
        np.testing.assert_allclose(result.interval[..., 0][:, 0, :], expected_lower, rtol=1e-12)
        np.testing.assert_allclose(result.interval[..., 1][:, 0, :], expected_upper, rtol=1e-12)

    def test_default_factory_is_ewa(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        rng = np.random.default_rng(0)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)
        assert isinstance(method.aggregator_lower_, EWA)
        assert isinstance(method.aggregator_upper_, EWA)


# ===========================================================================
# Fitted state
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
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        assert method.is_calibrated_ is False

        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 50, 1.0)
        method.calibrate(histories, truths)

        assert method.is_calibrated_ is True
        assert method.alpha_t_per_expert_.shape == (
            method.n_experts,
            n_series,
            horizon,
        )
        assert method.scores_.shape == (n_series, 50, horizon)
        assert method.n_calibration_samples_ == 50
        assert method.n_observations_ == 50

    def test_n_observations_grows_with_updates(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AggregatedAdaptiveConformalInference(adapter, alpha=0.1)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)
        assert method.n_observations_ == 50

        for _ in range(5):
            result = method.predict(np.zeros((1, 10)))
            method.update(result.point, np.zeros((1, 1, 1)))
        assert method.n_observations_ == 55
        assert method.scores_.shape[1] == 55
        assert method.n_calibration_samples_ == 50
