"""Tests for AdaptiveConformalInference (ACI)."""

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
from conformal_ts.methods.aci import AdaptiveConformalInference
from conformal_ts.methods.split import SplitConformal


# ---------------------------------------------------------------------------
# Test fixtures: callable adapters with and without cross-validation support
# ---------------------------------------------------------------------------


def _last_value_predict_fn(horizon: int):
    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.repeat(history[:, -1:], horizon, axis=1)

    return predict_fn


def _zero_predict_fn(n_series: int, horizon: int):
    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.zeros((n_series, horizon))

    return predict_fn


class CVCallableAdapter(CallableAdapter, SupportsCrossValidation):
    """
    Test-only :class:`CallableAdapter` that owns a training panel and exposes
    rolling-origin cross-validation by replaying the panel.

    The CV semantics match :class:`StatsForecastAdapter`: the *last* window
    ends at column ``T``; window ``i`` ends at column
    ``T - (n_windows - 1 - i) * step_size``.
    """

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
            pred = self.predict(history)  # (n_series, 1, horizon)
            preds.append(pred)
            truths.append(truth[:, np.newaxis, :])
        return np.concatenate(preds, axis=1), np.concatenate(truths, axis=1)

    def cv_window_history(self, n_windows: int, step_size: int, window_index: int) -> np.ndarray:
        """Reconstruct the history slice used for window ``window_index``."""
        T = self._training_panel.shape[1]
        cutoff = T - self.horizon - (n_windows - 1 - window_index) * step_size
        return self._training_panel[:, :cutoff]


# ---------------------------------------------------------------------------
# Helpers for tests
# ---------------------------------------------------------------------------


def _make_iid_dataset(
    rng: np.random.Generator,
    n_series: int,
    horizon: int,
    n_samples: int,
    noise_std: float,
    history_len: int = 30,
) -> tuple[list[Series], np.ndarray, list[np.ndarray]]:
    """
    iid Gaussian noise. ``predict_fn`` is the zero function, so scores are |truth|.

    Returns
    -------
    histories : list of (n_series, history_len) arrays
    truths : (n_series, n_samples, horizon) array
    truth_panels : list of (n_series, horizon) — same data, per-sample.
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
    method: AdaptiveConformalInference,
    holdout_histories: list[Series],
    holdout_truth_panels: list[np.ndarray],
) -> list[np.ndarray]:
    """Predict-update for each holdout sample. Return list of (n_series, 1, horizon, 2) intervals."""
    intervals: list[np.ndarray] = []
    for h, t_panel in zip(holdout_histories, holdout_truth_panels, strict=True):
        result = method.predict(h)
        intervals.append(result.interval.copy())
        method.update(result.point, t_panel[:, np.newaxis, :])
    return intervals


def _empirical_coverage(intervals: list[np.ndarray], truth_panels: list[np.ndarray]) -> float:
    """Element-wise coverage averaged over (n_series, 1, horizon)."""
    covered = 0
    total = 0
    for interval, truth in zip(intervals, truth_panels, strict=True):
        truth_3d = truth[:, np.newaxis, :]  # (n_series, 1, horizon)
        in_interval = (truth_3d >= interval[..., 0]) & (truth_3d <= interval[..., 1])
        covered += int(in_interval.sum())
        total += int(in_interval.size)
    return covered / total


# ===========================================================================
# Coverage — loop calibration path
# ===========================================================================


class TestCoverageLoopPath:
    """Empirical coverage matches 1 - alpha with loop calibration + online updates."""

    @pytest.mark.parametrize(
        "n_series,horizon,alpha",
        [
            (1, 1, 0.1),
            (1, 6, 0.1),
            (3, 1, 0.1),
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

        method = AdaptiveConformalInference(adapter, alpha=alpha, gamma=0.05)
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
    """Empirical coverage with CV-based calibration matches the loop path within tolerance."""

    def test_marginal_coverage_cv(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        noise_std = 1.0
        n_cal_windows = 200
        n_holdout = 300

        # Training panel for CV: enough room for all windows.
        # Window 0 cutoff = T - horizon - (n_windows - 1) * step_size, must be >= 1.
        T = n_cal_windows + horizon + 5
        training_panel = rng.normal(0.0, noise_std, (n_series, T))

        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            training_panel=training_panel,
            horizon=horizon,
        )

        method = AdaptiveConformalInference(adapter, alpha=alpha, gamma=0.05)
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
# Distribution shift — ACI vs split CP
# ===========================================================================


class TestDistributionShift:
    """ACI maintains nominal coverage under a noise-scale shift; split CP doesn't."""

    def test_aci_outperforms_split_under_shift(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        n_cal = 300
        n_holdout = 400
        noise_low = 1.0
        noise_high = 3.0

        # Both methods see the same calibration (low-noise) data.
        cal_histories, cal_truths, _ = _make_iid_dataset(rng, n_series, horizon, n_cal, noise_low)

        adapter_split = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        adapter_aci = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        method_split = SplitConformal(adapter_split, alpha=alpha)
        method_split.calibrate(cal_histories, cal_truths)

        method_aci = AdaptiveConformalInference(adapter_aci, alpha=alpha, gamma=0.05)
        method_aci.calibrate(cal_histories, cal_truths)

        # Holdout: high noise.
        ho_histories, _, ho_truth_panels = _make_iid_dataset(
            rng, n_series, horizon, n_holdout, noise_high
        )

        split_intervals: list[np.ndarray] = []
        aci_intervals: list[np.ndarray] = []
        for h, t_panel in zip(ho_histories, ho_truth_panels, strict=True):
            split_intervals.append(method_split.predict(h).interval.copy())
            aci_result = method_aci.predict(h)
            aci_intervals.append(aci_result.interval.copy())
            method_aci.update(aci_result.point, t_panel[:, np.newaxis, :])

        # Evaluate coverage on the second half — give ACI time to adapt.
        half = n_holdout // 2
        cov_split_late = _empirical_coverage(split_intervals[half:], ho_truth_panels[half:])
        cov_aci_late = _empirical_coverage(aci_intervals[half:], ho_truth_panels[half:])

        # Split CP under-covers badly under shift.
        assert cov_split_late < (1 - alpha) - 0.10, (
            f"Split CP coverage {cov_split_late:.3f} should be at least 0.10 below "
            f"nominal {1 - alpha:.2f} under noise shift."
        )
        # ACI adapts to the new regime.
        assert abs(cov_aci_late - (1 - alpha)) < 0.05, (
            f"ACI coverage {cov_aci_late:.3f} should be within 0.05 of nominal "
            f"{1 - alpha:.2f} after adaptation."
        )


# ===========================================================================
# CV vs loop equivalence
# ===========================================================================


class TestPathEquivalence:
    """alpha_t_ matches between the CV and loop paths on equivalent data."""

    def test_alpha_t_close_between_paths(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        alpha = 0.1
        n_cal_windows = 60
        T = n_cal_windows + horizon + 5
        training_panel = rng.normal(0.0, 1.0, (n_series, T))

        # CV path
        cv_adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            training_panel=training_panel,
            horizon=horizon,
        )
        method_cv = AdaptiveConformalInference(cv_adapter, alpha=alpha, gamma=0.05)
        method_cv.calibrate(n_windows=n_cal_windows, step_size=1)

        # Loop path with the *same* histories and truths the CV adapter would
        # produce internally.
        cv_preds, cv_truths = cv_adapter.cross_validate(n_windows=n_cal_windows, step_size=1)
        cal_histories = [
            cv_adapter.cv_window_history(n_cal_windows, 1, w) for w in range(n_cal_windows)
        ]
        loop_adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method_loop = AdaptiveConformalInference(loop_adapter, alpha=alpha, gamma=0.05)
        method_loop.calibrate(cal_histories, cv_truths)

        # The two paths should produce identical predictions on the same
        # histories (the predict callable is stateless), so alpha_t_ matches.
        np.testing.assert_allclose(method_cv.alpha_t_, method_loop.alpha_t_, rtol=1e-12)
        np.testing.assert_allclose(method_cv.scores_, method_loop.scores_, rtol=1e-12)
        np.testing.assert_allclose(cv_preds, loop_adapter.predict_batch(cal_histories), rtol=1e-12)


# ===========================================================================
# Shape tests
# ===========================================================================


class TestShape:
    """predict() returns documented shapes for several (n_series, horizon)."""

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
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)

        n_cal = 50
        histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]
        truths = rng.standard_normal((n_series, n_cal, horizon))
        method.calibrate(histories, truths)

        history = rng.standard_normal((n_series, 30))
        result = method.predict(history)

        assert result.point.shape == (n_series, 1, horizon)
        assert result.interval.shape == (n_series, 1, horizon, 2)
        assert result.alpha == 0.1
        assert method.alpha_t_.shape == (n_series, horizon)


# ===========================================================================
# Online adaptation
# ===========================================================================


class TestOnlineAdaptation:
    """update() actually evolves alpha_t_ and intervals."""

    def test_alpha_t_changes_with_updates(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        histories, truths, truth_panels = _make_iid_dataset(rng, n_series, horizon, 50, 1.0)
        method.calibrate(histories, truths)

        alpha_t_initial = method.alpha_t_.copy()

        # 10 update cycles with mixed-noise data.
        for _ in range(10):
            h = rng.normal(0.0, 1.0, (n_series, 30))
            t = rng.normal(0.0, 1.0, (n_series, horizon))
            result = method.predict(h)
            method.update(result.point, t[:, np.newaxis, :])

        assert not np.allclose(alpha_t_initial, method.alpha_t_), (
            "alpha_t_ should evolve after 10 update cycles."
        )

    def test_predict_widens_after_miss(self) -> None:
        """Same input → wider interval after an update() that missed."""
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 50, 1.0)
        method.calibrate(histories, truths)

        history = rng.normal(0.0, 1.0, (n_series, 30))

        first = method.predict(history)
        # Force a miss by passing a truth far outside the interval.
        huge_truth = np.array([[[1e6]]], dtype=np.float64)
        method.update(first.point, huge_truth)

        second = method.predict(history)

        first_width = first.interval[..., 1] - first.interval[..., 0]
        second_width = second.interval[..., 1] - second.interval[..., 0]
        assert (second_width >= first_width).all()
        assert (second_width > first_width).any(), (
            "Interval width should strictly increase after a miss-driven update."
        )


# ===========================================================================
# Lifecycle errors
# ===========================================================================


class TestLifecycleErrors:
    """predict/update precondition checks raise clear errors."""

    def _make_calibrated(self) -> AdaptiveConformalInference:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(1, 1),
            horizon=1,
            n_series=1,
        )
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)
        return method

    def test_predict_before_calibrate_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        with pytest.raises(CalibrationError, match="calibrate"):
            method.predict(np.zeros((1, 10)))

    def test_update_before_calibrate_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        with pytest.raises(CalibrationError, match="calibrate"):
            method.update(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))

    def test_update_with_wrong_truth_shape_raises(self) -> None:
        method = self._make_calibrated()
        prediction = np.zeros((1, 1, 1))
        with pytest.raises(ValueError, match="truth must have shape"):
            method.update(prediction, np.zeros((1, 1)))  # missing horizon axis

    def test_update_with_wrong_prediction_shape_raises(self) -> None:
        method = self._make_calibrated()
        truth = np.zeros((1, 1, 1))
        with pytest.raises(ValueError, match="prediction must have shape"):
            method.update(np.zeros((1, 1)), truth)  # missing horizon axis


# ===========================================================================
# Saturation
# ===========================================================================


class TestSaturation:
    """Negative alpha_t_ -> very wide but finite intervals."""

    def test_negative_alpha_t_yields_wide_finite_intervals(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)

        # Force saturation: 1 - alpha_t > 1 -> clipped to 1 -> threshold = max float.
        method.alpha_t_ = np.full_like(method.alpha_t_, -0.5)

        result = method.predict(np.zeros((1, 10)))
        lower = result.interval[..., 0]
        upper = result.interval[..., 1]

        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))
        assert not np.any(np.isnan(lower))
        assert not np.any(np.isnan(upper))
        # Each bound is bounded by max float, and the interval is very wide on
        # both sides of the prediction (we deliberately avoid computing
        # ``upper - lower`` because ``2 * max_float`` overflows to inf).
        assert np.all(upper >= 1e10)
        assert np.all(lower <= -1e10)


# ===========================================================================
# Invalid alpha / gamma
# ===========================================================================


class TestInvalidParameters:
    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0, 1.5])
    def test_invalid_alpha_raises(self, alpha: float) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="alpha"):
            AdaptiveConformalInference(adapter, alpha=alpha, gamma=0.05)

    @pytest.mark.parametrize("gamma", [0.0, -0.01])
    def test_invalid_gamma_raises(self, gamma: float) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="gamma"):
            AdaptiveConformalInference(adapter, alpha=0.1, gamma=gamma)

    def test_large_gamma_warns_but_constructs(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.warns(UserWarning, match="gamma"):
            method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.5)
        assert method.gamma == 0.5


# ===========================================================================
# Capability requirements
# ===========================================================================


class TestCapabilityRequirements:
    """ACI has no required capabilities — constructs with any adapter."""

    def test_constructs_with_callable_adapter(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        assert isinstance(method.forecaster, CallableAdapter)

    def test_constructs_with_cv_callable_adapter(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(1, 1),
            training_panel=rng.normal(0, 1.0, (1, 50)),
            horizon=1,
        )
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        assert isinstance(method.forecaster, CVCallableAdapter)


# ===========================================================================
# Calibrate dispatch
# ===========================================================================


class TestCalibrateDispatch:
    """calibrate() input validation across the two calling conventions."""

    def test_no_args_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        with pytest.raises(ValueError, match="histories"):
            method.calibrate()

    def test_both_paths_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(1, 1),
            training_panel=rng.normal(0, 1.0, (1, 50)),
            horizon=1,
        )
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        histories = [rng.standard_normal((1, 10)) for _ in range(20)]
        truths = rng.standard_normal((1, 20, 1))
        with pytest.raises(ValueError, match="not both"):
            method.calibrate(histories, truths, n_windows=10)

    def test_n_windows_without_cv_support_raises(self) -> None:
        # CallableAdapter does not implement SupportsCrossValidation.
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        with pytest.raises(UnsupportedCapability, match="SupportsCrossValidation"):
            method.calibrate(n_windows=20)

    def test_too_few_calibration_samples_loop_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
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
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        too_few = math.ceil(1.0 / 0.1) - 1
        with pytest.raises(CalibrationError, match="calibration samples"):
            method.calibrate(n_windows=too_few)


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
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        assert method.is_calibrated_ is False

        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 50, 1.0)
        method.calibrate(histories, truths)

        assert method.is_calibrated_ is True
        assert method.alpha_t_.shape == (n_series, horizon)
        assert method.scores_.shape == (n_series, 50, horizon)
        assert method.n_calibration_samples_ == 50
        assert method.n_observations_ == 50

    def test_n_observations_grows_with_updates(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)
        assert method.n_observations_ == 50

        for _ in range(7):
            result = method.predict(np.zeros((1, 10)))
            method.update(result.point, np.zeros((1, 1, 1)))
        assert method.n_observations_ == 57
        assert method.scores_.shape[1] == 57
        assert method.n_calibration_samples_ == 50  # unchanged

    def test_calibration_result_is_a_snapshot(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 2), horizon=2, n_series=1)
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        histories, truths, _ = _make_iid_dataset(rng, 1, 2, 50, 1.0)
        cal = method.calibrate(histories, truths)

        original_alpha_t = method.alpha_t_.copy()
        cal.diagnostics["alpha_t"][...] = 999.0
        cal.score_quantile[...] = 999.0

        np.testing.assert_array_equal(method.alpha_t_, original_alpha_t)
