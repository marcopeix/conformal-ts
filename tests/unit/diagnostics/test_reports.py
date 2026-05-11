"""Tests for diagnostics.reports."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_ts.adapters.callable import CallableAdapter
from conformal_ts.base import CalibrationError
from conformal_ts.diagnostics.reports import Report, evaluate
from conformal_ts.methods.aci import AdaptiveConformalInference
from conformal_ts.methods.split import SplitConformal


def _zero_predict_fn(n_series: int, horizon: int):
    def predict_fn(history: np.ndarray) -> np.ndarray:
        return np.zeros((n_series, horizon))

    return predict_fn


def _make_iid(rng: np.random.Generator, n_series: int, horizon: int, n: int):
    histories = [rng.normal(size=(n_series, 20)) for _ in range(n)]
    truths = rng.normal(size=(n_series, n, horizon))
    return histories, truths


class TestEvaluateSplit:
    def test_basic_split_conformal(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 2
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = SplitConformal(adapter, alpha=0.1)
        cal_h, cal_t = _make_iid(rng, n_series, horizon, 200)
        method.calibrate(cal_h, cal_t)

        ho_h, ho_t = _make_iid(rng, n_series, horizon, 200)
        report = evaluate(method, ho_h, ho_t)

        assert isinstance(report, Report)
        assert report.method_name == "SplitConformal"
        assert report.alpha == 0.1
        assert report.n_holdout_samples == 200
        # Marginal coverage should be near 1 - alpha on iid data.
        assert abs(report.marginal_coverage - 0.9) < 0.06
        assert report.coverage_by_horizon.shape == (horizon,)
        assert report.coverage_by_series.shape == (n_series,)
        assert report.coverage_per_cell.shape == (n_series, horizon)
        assert report.intervals.shape == (n_series, 200, horizon, 2)
        assert report.truths.shape == (n_series, 200, horizon)
        assert report.points.shape == (n_series, 200, horizon)
        # Layer 2 is empty for SplitConformal.
        assert report.method_state == {}

    def test_evaluate_aci_updates_state(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        cal_h, cal_t = _make_iid(rng, n_series, horizon, 100)
        method.calibrate(cal_h, cal_t)

        n_obs_before = method.n_observations_
        ho_h, ho_t = _make_iid(rng, n_series, horizon, 50)
        report = evaluate(method, ho_h, ho_t)

        assert method.n_observations_ == n_obs_before + 50
        # method_state should be populated for ACI.
        assert "alpha_t" in report.method_state
        assert "gamma" in report.method_state

    def test_update_online_false_does_not_call_update(self) -> None:
        rng = np.random.default_rng(0)
        n_series, horizon = 1, 1
        adapter = CallableAdapter(
            predict_fn=_zero_predict_fn(n_series, horizon),
            horizon=horizon,
            n_series=n_series,
        )
        method = AdaptiveConformalInference(adapter, alpha=0.1, gamma=0.05)
        cal_h, cal_t = _make_iid(rng, n_series, horizon, 100)
        method.calibrate(cal_h, cal_t)
        n_obs_before = method.n_observations_

        ho_h, ho_t = _make_iid(rng, n_series, horizon, 50)
        evaluate(method, ho_h, ho_t, update_online=False)
        assert method.n_observations_ == n_obs_before


class TestEvaluateErrors:
    def test_uncalibrated_raises(self) -> None:
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(adapter, alpha=0.1)
        ho_h = [np.zeros((1, 5))]
        ho_t = np.zeros((1, 1, 1))
        with pytest.raises(CalibrationError, match="calibrate"):
            evaluate(method, ho_h, ho_t)

    def test_shape_mismatch_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(adapter, alpha=0.1)
        cal_h, cal_t = _make_iid(rng, 1, 1, 50)
        method.calibrate(cal_h, cal_t)

        # 3 histories but holdout_truths claims 5 samples.
        ho_h = [np.zeros((1, 5)) for _ in range(3)]
        ho_t = np.zeros((1, 5, 1))
        with pytest.raises(ValueError, match="holdout_histories"):
            evaluate(method, ho_h, ho_t)

    def test_holdout_truths_wrong_ndim_raises(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(adapter, alpha=0.1)
        cal_h, cal_t = _make_iid(rng, 1, 1, 50)
        method.calibrate(cal_h, cal_t)
        ho_h = [np.zeros((1, 5))]
        bad_truths = np.zeros((1, 1))  # 2-D
        with pytest.raises(ValueError, match="holdout_truths"):
            evaluate(method, ho_h, bad_truths)


class TestReportSummary:
    def test_summary_contains_key_info(self) -> None:
        rng = np.random.default_rng(0)
        adapter = CallableAdapter(predict_fn=_zero_predict_fn(1, 1), horizon=1, n_series=1)
        method = SplitConformal(adapter, alpha=0.1)
        cal_h, cal_t = _make_iid(rng, 1, 1, 100)
        method.calibrate(cal_h, cal_t)
        ho_h, ho_t = _make_iid(rng, 1, 1, 50)
        report = evaluate(method, ho_h, ho_t)

        text = report.summary()
        assert isinstance(text, str)
        assert len(text) > 0
        assert "SplitConformal" in text
        assert f"{report.marginal_coverage:.3f}" in text
