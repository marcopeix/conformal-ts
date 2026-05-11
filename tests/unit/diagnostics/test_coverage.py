"""Tests for diagnostics.coverage."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_ts.diagnostics.coverage import (
    _coverage_indicator,
    coverage_by_horizon,
    coverage_by_series,
    coverage_per_cell,
    marginal_coverage,
    rolling_coverage,
)


def _make_interval(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.stack([lower, upper], axis=-1)


class TestCoverageIndicator:
    def test_hand_checked(self) -> None:
        # Two cells: (1) covered, (2) above upper, (3) below lower.
        interval = np.array(
            [[[[0.0, 2.0], [0.0, 1.0], [-1.0, 0.0]]]],
            dtype=np.float64,
        )  # shape (1, 1, 3, 2)
        truth = np.array([[[1.0, 5.0, -2.0]]], dtype=np.float64)
        ind = _coverage_indicator(interval, truth)
        assert ind.shape == (1, 1, 3)
        np.testing.assert_array_equal(ind, np.array([[[True, False, False]]]))

    def test_edge_inclusive(self) -> None:
        # truth exactly equal to lower / upper should count as covered.
        interval = np.array([[[[0.0, 1.0], [0.0, 1.0]]]], dtype=np.float64)
        truth = np.array([[[0.0, 1.0]]], dtype=np.float64)
        ind = _coverage_indicator(interval, truth)
        np.testing.assert_array_equal(ind, np.array([[[True, True]]]))


class TestMarginalCoverage:
    def test_exact_fraction(self) -> None:
        # Construct 10 cells with exactly 9 covered.
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(1, 10, 1))
        interval = np.zeros((1, 10, 1, 2))
        interval[..., 0] = truth - 1.0
        interval[..., 1] = truth + 1.0
        # Force one cell to miss.
        interval[0, 0, 0, :] = [100.0, 200.0]
        cov = marginal_coverage(interval, truth)
        assert cov == pytest.approx(0.9)


class TestShapes:
    @pytest.mark.parametrize(
        "n_series,n_samples,horizon",
        [(1, 5, 1), (1, 10, 4), (3, 5, 1), (3, 20, 6)],
    )
    def test_coverage_by_horizon_shape(self, n_series: int, n_samples: int, horizon: int) -> None:
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(n_series, n_samples, horizon))
        interval = _make_interval(truth - 1.0, truth + 1.0)
        out = coverage_by_horizon(interval, truth)
        assert out.shape == (horizon,)
        np.testing.assert_allclose(out, np.ones(horizon))

    @pytest.mark.parametrize(
        "n_series,n_samples,horizon",
        [(1, 5, 1), (1, 10, 4), (3, 5, 1), (3, 20, 6)],
    )
    def test_coverage_by_series_shape(self, n_series: int, n_samples: int, horizon: int) -> None:
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(n_series, n_samples, horizon))
        interval = _make_interval(truth - 1.0, truth + 1.0)
        out = coverage_by_series(interval, truth)
        assert out.shape == (n_series,)
        np.testing.assert_allclose(out, np.ones(n_series))

    @pytest.mark.parametrize(
        "n_series,n_samples,horizon",
        [(1, 5, 1), (1, 10, 4), (3, 5, 1), (3, 20, 6)],
    )
    def test_coverage_per_cell_shape(self, n_series: int, n_samples: int, horizon: int) -> None:
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(n_series, n_samples, horizon))
        interval = _make_interval(truth - 1.0, truth + 1.0)
        out = coverage_per_cell(interval, truth)
        assert out.shape == (n_series, horizon)


class TestRollingCoverage:
    def test_shape(self) -> None:
        rng = np.random.default_rng(0)
        n_series, n_samples, horizon = 2, 12, 3
        truth = rng.normal(size=(n_series, n_samples, horizon))
        interval = _make_interval(truth - 1.0, truth + 1.0)
        out = rolling_coverage(interval, truth, window=5)
        assert out.shape == (n_series, n_samples - 5 + 1, horizon)

    def test_window_1_matches_per_sample_indicator(self) -> None:
        rng = np.random.default_rng(1)
        n_series, n_samples, horizon = 1, 6, 2
        truth = rng.normal(size=(n_series, n_samples, horizon))
        # Mix of covered/uncovered.
        interval = np.zeros((n_series, n_samples, horizon, 2))
        interval[..., 0] = truth - 0.5
        interval[..., 1] = truth + 0.5
        interval[0, 0, 0, :] = [100.0, 200.0]  # force a miss
        out = rolling_coverage(interval, truth, window=1)
        expected = _coverage_indicator(interval, truth).astype(np.float64)
        np.testing.assert_allclose(out, expected)

    def test_window_equals_n_samples(self) -> None:
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(1, 4, 1))
        interval = _make_interval(truth - 1.0, truth + 1.0)
        out = rolling_coverage(interval, truth, window=4)
        assert out.shape == (1, 1, 1)
        np.testing.assert_allclose(out, 1.0)

    def test_invalid_window_zero_raises(self) -> None:
        truth = np.zeros((1, 4, 1))
        interval = _make_interval(truth - 1.0, truth + 1.0)
        with pytest.raises(ValueError, match="window"):
            rolling_coverage(interval, truth, window=0)

    def test_invalid_window_too_large_raises(self) -> None:
        truth = np.zeros((1, 4, 1))
        interval = _make_interval(truth - 1.0, truth + 1.0)
        with pytest.raises(ValueError, match="window"):
            rolling_coverage(interval, truth, window=5)


class TestShapeValidation:
    def test_wrong_interval_dim_raises(self) -> None:
        bad_interval = np.zeros((1, 4, 1))  # 3-D
        truth = np.zeros((1, 4, 1))
        with pytest.raises(ValueError, match="interval"):
            marginal_coverage(bad_interval, truth)

    def test_wrong_last_axis_raises(self) -> None:
        bad_interval = np.zeros((1, 4, 1, 3))  # last axis 3 instead of 2
        truth = np.zeros((1, 4, 1))
        with pytest.raises(ValueError, match="interval"):
            coverage_by_horizon(bad_interval, truth)

    def test_truth_interval_mismatch_raises(self) -> None:
        interval = np.zeros((1, 4, 1, 2))
        truth = np.zeros((1, 5, 1))  # wrong sample count
        with pytest.raises(ValueError, match="truth"):
            coverage_by_series(interval, truth)
