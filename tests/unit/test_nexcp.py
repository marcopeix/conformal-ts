"""Tests for NonexchangeableConformalPrediction (NexCP)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conformal_ts.adapters.callable import CallableAdapter
from conformal_ts.base import (
    CalibrationError,
    UnsupportedCapability,
)
from conformal_ts.methods.nexcp import NonexchangeableConformalPrediction
from conformal_ts.methods.split import SplitConformal
from tests._online_helpers import (
    CVCallableAdapter,
    _empirical_coverage,
    _last_value_predict_fn,
    _make_iid_dataset,
    _zero_predict_fn,
)


# ===========================================================================
# Coverage — loop calibration path (offline, no updates)
# ===========================================================================


class TestCoverageLoopPath:
    """Empirical coverage matches 1 - alpha for offline NexCP on iid data."""

    @pytest.mark.parametrize(
        "n_series,horizon,alpha,rho",
        [
            (1, 1, 0.1, 1.0),
            (1, 1, 0.1, 0.99),
            (3, 6, 0.1, 0.99),
            (1, 1, 0.2, 0.99),
            (3, 6, 0.2, 1.0),
        ],
    )
    def test_marginal_coverage(self, n_series: int, horizon: int, alpha: float, rho: float) -> None:
        rng = np.random.default_rng(0)
        noise_std = 1.0
        n_cal = 200
        n_test = 200

        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        cal_histories, cal_truths, _ = _make_iid_dataset(rng, n_series, horizon, n_cal, noise_std)

        method = NonexchangeableConformalPrediction(adapter, alpha=alpha, rho=rho)
        method.calibrate(cal_histories, cal_truths)

        ho_histories, _, ho_truth_panels = _make_iid_dataset(
            rng, n_series, horizon, n_test, noise_std
        )

        intervals: list[np.ndarray] = []
        for h in ho_histories:
            intervals.append(method.predict(h).interval.copy())

        coverage = _empirical_coverage(intervals, ho_truth_panels)
        assert abs(coverage - (1 - alpha)) < 0.04, (
            f"Empirical coverage {coverage:.4f} deviates from target "
            f"{1 - alpha:.2f} by more than 0.04."
        )


# ===========================================================================
# Coverage — CV calibration path
# ===========================================================================


class TestCoverageCVPath:
    """CV calibration produces expected coverage and diagnostics."""

    def test_marginal_coverage_cv(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        rho = 0.99
        noise_std = 1.0
        n_cal_windows = 200
        n_test = 200

        T = n_cal_windows + horizon + 5
        training_panel = rng.normal(0.0, noise_std, (n_series, T))

        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            training_panel=training_panel,
            horizon=horizon,
        )

        method = NonexchangeableConformalPrediction(adapter, alpha=alpha, rho=rho)
        result = method.calibrate(n_windows=n_cal_windows, step_size=1)
        assert result.diagnostics["path"] == "cross_validation"
        assert result.diagnostics["rho"] == rho
        assert "effective_sample_size" in result.diagnostics

        ho_histories, _, ho_truth_panels = _make_iid_dataset(
            rng, n_series, horizon, n_test, noise_std
        )
        intervals = [method.predict(h).interval.copy() for h in ho_histories]

        coverage = _empirical_coverage(intervals, ho_truth_panels)
        assert abs(coverage - (1 - alpha)) < 0.04, (
            f"Empirical coverage {coverage:.4f} deviates from target "
            f"{1 - alpha:.2f} by more than 0.04."
        )


# ===========================================================================
# rho=1 reduces to split CP (within float tolerance)
# ===========================================================================


class TestSplitCPEquivalence:
    """With rho=1.0 NexCP recovers the textbook split-CP order statistic."""

    def test_rho_one_matches_split_cp_rank(self) -> None:
        """NexCP at rho=1 picks the rank-``ceil((1 - alpha)(n + 1))`` score.

        This is the canonical split-CP threshold. ``SplitConformal`` produces
        a value within one inter-order-statistic gap of this (it uses
        ``np.quantile`` with linear interpolation), so we also verify
        ``np.allclose`` between the two with a tolerance that admits a
        one-position interpolation difference.
        """
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        alpha = 0.1
        n_cal = 100

        histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]
        truths = rng.standard_normal((n_series, n_cal, horizon))

        adapter_split = CallableAdapter(
            predict_fn=_last_value_predict_fn(horizon),
            horizon=horizon,
            n_series=n_series,
        )
        adapter_nex = CallableAdapter(
            predict_fn=_last_value_predict_fn(horizon),
            horizon=horizon,
            n_series=n_series,
        )

        split = SplitConformal(adapter_split, alpha=alpha)
        split.calibrate(histories, truths)

        nex = NonexchangeableConformalPrediction(adapter_nex, alpha=alpha, rho=1.0)
        nex.calibrate(histories, truths)

        # Exact textbook split-CP rank.
        preds = adapter_split.predict_batch(histories)
        truths_arr = np.asarray(truths, dtype=np.float64)
        scores = np.abs(truths_arr - preds)
        sorted_scores = np.sort(scores, axis=1)
        rank = math.ceil((1 - alpha) * (n_cal + 1)) - 1
        expected = sorted_scores[:, rank, :]

        np.testing.assert_allclose(nex.score_quantile_, expected)

        # SplitConformal interpolates, so it sits between adjacent order stats.
        gaps = sorted_scores[:, rank + 1, :] - sorted_scores[:, rank, :]
        atol = float(gaps.max()) + 1e-9
        np.testing.assert_allclose(nex.score_quantile_, split.score_quantile_, atol=atol)


# ===========================================================================
# Distribution shift: NexCP beats split CP on post-shift coverage
# ===========================================================================


class TestDistributionShift:
    """NexCP with rho<1 adapts to a noise-scale shift; split CP does not."""

    def test_nexcp_outperforms_split_under_shift(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        alpha = 0.1
        rho = 0.9
        n_cal = 200
        n_holdout = 400
        noise_low = 1.0
        noise_high = 3.0

        cal_histories, cal_truths, _ = _make_iid_dataset(rng, n_series, horizon, n_cal, noise_low)

        adapter_split = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        adapter_nex = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )

        split = SplitConformal(adapter_split, alpha=alpha)
        split.calibrate(cal_histories, cal_truths)

        nex = NonexchangeableConformalPrediction(adapter_nex, alpha=alpha, rho=rho)
        nex.calibrate(cal_histories, cal_truths)

        ho_histories, _, ho_truth_panels = _make_iid_dataset(
            rng, n_series, horizon, n_holdout, noise_high
        )

        split_intervals: list[np.ndarray] = []
        nex_intervals: list[np.ndarray] = []
        for h, t_panel in zip(ho_histories, ho_truth_panels, strict=True):
            split_intervals.append(split.predict(h).interval.copy())
            nex_result = nex.predict(h)
            nex_intervals.append(nex_result.interval.copy())
            nex.update(nex_result.point, t_panel[:, np.newaxis, :])

        half = n_holdout // 2
        cov_split_late = _empirical_coverage(split_intervals[half:], ho_truth_panels[half:])
        cov_nex_late = _empirical_coverage(nex_intervals[half:], ho_truth_panels[half:])

        # Split CP under-covers badly under the shift.
        assert cov_split_late < (1 - alpha) - 0.10, (
            f"Split CP coverage {cov_split_late:.3f} should be at least 0.10 below "
            f"nominal {1 - alpha:.2f} under noise shift."
        )
        # NexCP visibly improves on split CP after a few hundred updates.
        assert cov_nex_late > cov_split_late + 0.05, (
            f"NexCP late coverage {cov_nex_late:.3f} should exceed split CP "
            f"{cov_split_late:.3f} by at least 0.05."
        )


# ===========================================================================
# Online updates change the threshold
# ===========================================================================


class TestOnlineAdaptation:
    """update() actually moves score_quantile_."""

    def test_score_quantile_changes_with_updates(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.95)
        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 100, 1.0)
        method.calibrate(histories, truths)

        initial_quantile = method.score_quantile_.copy()

        for _ in range(10):
            h = rng.normal(0.0, 1.0, (n_series, 30))
            t = rng.normal(0.0, 5.0, (n_series, horizon))
            result = method.predict(h)
            method.update(result.point, t[:, np.newaxis, :])

        assert not np.allclose(initial_quantile, method.score_quantile_), (
            "score_quantile_ should evolve after 10 update cycles."
        )
        assert method.n_observations_ == 110
        assert method.n_calibration_samples_ == 100


# ===========================================================================
# Decay weights newer samples
# ===========================================================================


class TestDecayWeighting:
    """_compute_weighted_quantile favours recent scores when rho is small."""

    def test_recent_dominate_with_small_rho(self) -> None:
        # First half of scores is large, second half is small. With rho=0.9
        # the recent (small) scores dominate the weighted quantile; with
        # rho=1.0 they don't.
        n_obs = 100
        scores = np.empty((1, n_obs, 1), dtype=np.float64)
        scores[0, : n_obs // 2, 0] = 10.0
        scores[0, n_obs // 2 :, 0] = 1.0

        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(1, 1),
            horizon=1,
            n_series=1,
        )
        # rho=0.9 keeps target = (1 - alpha)(W + 1)/W < 1 for n=100, alpha=0.1
        # (W ≈ 10, target ≈ 0.99), so the search resolves below the saturated max.
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.9)

        q_small = method._compute_weighted_quantile(scores, rho=0.9, alpha=0.1)
        q_large = method._compute_weighted_quantile(scores, rho=1.0, alpha=0.1)

        # Small rho: tail is dominated by recent (small) values.
        assert q_small[0, 0] < 5.0, (
            f"With rho=0.9, recent small scores should pull the quantile below 5.0, "
            f"got {q_small[0, 0]:.4f}."
        )
        # Uniform weights: both regimes contribute, so the quantile sits in the
        # large-score tail.
        assert q_large[0, 0] > 5.0, (
            f"With rho=1.0, the quantile should be in the large-score tail, "
            f"got {q_large[0, 0]:.4f}."
        )


# ===========================================================================
# Effective sample size validation
# ===========================================================================


class TestEffectiveSampleSize:
    """Calibration raises when ESS is below ceil(1 / alpha)."""

    def test_low_ess_raises(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        # rho=0.5 with n=20 gives ESS ≈ (1 + rho) / (1 - rho) ≈ 3, well below
        # ceil(1/0.05) = 20.
        n_cal = 20
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]
        truths = rng.standard_normal((n_series, n_cal, horizon))

        method = NonexchangeableConformalPrediction(adapter, alpha=0.05, rho=0.5)
        with pytest.raises(CalibrationError, match=r"Effective sample size .* below required"):
            method.calibrate(histories, truths)


# ===========================================================================
# CV vs loop equivalence
# ===========================================================================


class TestPathEquivalence:
    """score_quantile_ matches between CV and loop paths on equivalent data."""

    def test_score_quantile_close_between_paths(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        alpha = 0.1
        rho = 0.99
        n_cal_windows = 60
        T = n_cal_windows + horizon + 5
        training_panel = rng.normal(0.0, 1.0, (n_series, T))

        cv_adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            training_panel=training_panel,
            horizon=horizon,
        )
        method_cv = NonexchangeableConformalPrediction(cv_adapter, alpha=alpha, rho=rho)
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
        method_loop = NonexchangeableConformalPrediction(loop_adapter, alpha=alpha, rho=rho)
        method_loop.calibrate(cal_histories, cv_truths)

        np.testing.assert_allclose(method_cv.score_quantile_, method_loop.score_quantile_)
        np.testing.assert_allclose(method_cv.scores_, method_loop.scores_)


# ===========================================================================
# Shape tests
# ===========================================================================


class TestShape:
    """predict() returns documented shapes."""

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
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)

        n_cal = 50
        histories = [rng.standard_normal((n_series, 30)) for _ in range(n_cal)]
        truths = rng.standard_normal((n_series, n_cal, horizon))
        method.calibrate(histories, truths)

        history = rng.standard_normal((n_series, 30))
        result = method.predict(history)

        assert result.point.shape == (n_series, 1, horizon)
        assert result.interval.shape == (n_series, 1, horizon, 2)
        assert result.alpha == 0.1
        assert method.score_quantile_.shape == (n_series, horizon)


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    """Same calibration + same updates produce identical fitted state."""

    def test_deterministic_score_quantile(self) -> None:
        n_series, horizon = 2, 3
        rng = np.random.default_rng(42)

        cal_histories = [rng.standard_normal((n_series, 30)) for _ in range(80)]
        cal_truths = rng.standard_normal((n_series, 80, horizon))
        update_pairs = [
            (
                rng.standard_normal((n_series, 1, horizon)),
                rng.standard_normal((n_series, 1, horizon)),
            )
            for _ in range(5)
        ]

        def run() -> np.ndarray:
            adapter = CallableAdapter(
                predict_fn=_last_value_predict_fn(horizon),
                horizon=horizon,
                n_series=n_series,
            )
            method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.95)
            method.calibrate(cal_histories, cal_truths)
            for pred, truth in update_pairs:
                method.update(pred, truth)
            return method.score_quantile_.copy()

        np.testing.assert_array_equal(run(), run())


# ===========================================================================
# Lifecycle errors
# ===========================================================================


class TestLifecycleErrors:
    """predict/update precondition checks raise clear errors."""

    def _make_calibrated(self) -> NonexchangeableConformalPrediction:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(1, 1),
            horizon=1,
            n_series=1,
        )
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)
        return method

    def test_predict_before_calibrate_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        with pytest.raises(CalibrationError, match="calibrate"):
            method.predict(np.zeros((1, 10)))

    def test_update_before_calibrate_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
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
# Invalid alpha / rho
# ===========================================================================


class TestInvalidParameters:
    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0, 1.5])
    def test_invalid_alpha_raises(self, alpha: float) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="alpha"):
            NonexchangeableConformalPrediction(adapter, alpha=alpha, rho=0.99)

    @pytest.mark.parametrize("rho", [0.0, -0.1, 1.5])
    def test_invalid_rho_raises(self, rho: float) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        with pytest.raises(ValueError, match="rho"):
            NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=rho)

    def test_rho_one_is_valid(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=1.0)
        assert method.rho == 1.0


# ===========================================================================
# Capability requirements
# ===========================================================================


class TestCapabilityRequirements:
    """NexCP has no required capabilities — constructs with any adapter."""

    def test_constructs_with_callable_adapter(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        assert isinstance(method.forecaster, CallableAdapter)


# ===========================================================================
# Calibrate dispatch
# ===========================================================================


class TestCalibrateDispatch:
    """calibrate() input validation across the two calling conventions."""

    def test_no_args_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        with pytest.raises(ValueError, match="histories"):
            method.calibrate()

    def test_both_paths_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CVCallableAdapter(
            predict_fn=_zero_predict_fn(1, 1),
            training_panel=rng.normal(0, 1.0, (1, 50)),
            horizon=1,
        )
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        histories = [rng.standard_normal((1, 10)) for _ in range(20)]
        truths = rng.standard_normal((1, 20, 1))
        with pytest.raises(ValueError, match="not both"):
            method.calibrate(histories, truths, n_windows=10)

    def test_n_windows_without_cv_support_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        with pytest.raises(UnsupportedCapability, match="SupportsCrossValidation"):
            method.calibrate(n_windows=20)

    def test_loop_path_diagnostics(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        result = method.calibrate(histories, truths)
        assert result.diagnostics["path"] == "loop"
        assert result.diagnostics["rho"] == 0.99
        assert "effective_sample_size" in result.diagnostics


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
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        assert method.is_calibrated_ is False

        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 50, 1.0)
        method.calibrate(histories, truths)

        assert method.is_calibrated_ is True
        assert method.score_quantile_.shape == (n_series, horizon)
        assert method.scores_.shape == (n_series, 50, horizon)
        assert method.n_calibration_samples_ == 50
        assert method.n_observations_ == 50

    def test_n_observations_grows_with_updates(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)

        for _ in range(7):
            result = method.predict(np.zeros((1, 10)))
            method.update(result.point, np.zeros((1, 1, 1)))
        assert method.n_observations_ == 57
        assert method.scores_.shape[1] == 57
        assert method.n_calibration_samples_ == 50

    def test_calibration_result_is_a_snapshot(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 2), horizon=2, n_series=1)
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        histories, truths, _ = _make_iid_dataset(rng, 1, 2, 50, 1.0)
        cal = method.calibrate(histories, truths)

        original = method.score_quantile_.copy()
        cal.score_quantile[...] = 999.0

        np.testing.assert_array_equal(method.score_quantile_, original)

    def test_stores_calibration_data(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 50, 1.0)
        method.calibrate(histories, truths)

        assert method.predictions_calibration_.shape == (n_series, 50, horizon)
        assert method.truths_calibration_.shape == (n_series, 50, horizon)
        np.testing.assert_allclose(method.truths_calibration_, truths)

    def test_calibration_data_is_defensive_copy(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        histories, truths, _ = _make_iid_dataset(rng, 1, 1, 50, 1.0)
        method.calibrate(histories, truths)

        scores_before = method.scores_.copy()
        method.predictions_calibration_[...] = 0.0
        method.truths_calibration_[...] = 0.0
        np.testing.assert_array_equal(method.scores_, scores_before)


class TestIntervalsFromPredictions:
    def test_matches_invert(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 2, 3
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = NonexchangeableConformalPrediction(adapter, alpha=0.1, rho=0.99)
        histories, truths, _ = _make_iid_dataset(rng, n_series, horizon, 50, 1.0)
        method.calibrate(histories, truths)

        intervals = method._intervals_from_predictions(method.predictions_calibration_)
        expected = method.score_fn.invert(method.predictions_calibration_, method.score_quantile_)
        np.testing.assert_allclose(intervals, expected)
        assert intervals.shape == (n_series, 50, horizon, 2)
