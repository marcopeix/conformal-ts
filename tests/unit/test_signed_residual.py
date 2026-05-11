"""Tests for SignedResidual score function."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_ts.nonconformity.signed import SignedResidual


class TestScore:
    """SignedResidual.score returns correct values and shapes."""

    def test_hand_checked_values(self) -> None:
        score_fn = SignedResidual()
        prediction = np.array([[[1.0, 2.0, 3.0]]])  # (1, 1, 3)
        truth = np.array([[[1.5, 1.5, 4.5]]])
        scores = score_fn.score(prediction, truth)
        expected = np.array([[[0.5, -0.5, 1.5]]])
        np.testing.assert_array_almost_equal(scores, expected)

    def test_sign_truth_above_prediction(self) -> None:
        score_fn = SignedResidual()
        prediction = np.array([[[1.0]]])
        truth = np.array([[[2.5]]])
        assert score_fn.score(prediction, truth).item() > 0

    def test_sign_truth_below_prediction(self) -> None:
        score_fn = SignedResidual()
        prediction = np.array([[[5.0]]])
        truth = np.array([[[2.0]]])
        assert score_fn.score(prediction, truth).item() < 0

    def test_sign_equal_is_zero(self) -> None:
        score_fn = SignedResidual()
        prediction = np.array([[[3.14]]])
        truth = np.array([[[3.14]]])
        assert score_fn.score(prediction, truth).item() == 0.0

    @pytest.mark.parametrize(
        "n_series,n_samples,horizon",
        [(1, 1, 1), (1, 50, 1), (3, 1, 6), (3, 50, 6)],
    )
    def test_shape_preserved(self, n_series: int, n_samples: int, horizon: int) -> None:
        score_fn = SignedResidual()
        rng = np.random.default_rng(0)
        pred = rng.standard_normal((n_series, n_samples, horizon))
        truth = rng.standard_normal((n_series, n_samples, horizon))
        scores = score_fn.score(pred, truth)
        assert scores.shape == (n_series, n_samples, horizon)


class TestInvert:
    """SignedResidual.invert produces asymmetric intervals."""

    def test_hand_checked_interval(self) -> None:
        score_fn = SignedResidual()
        prediction = np.array([[[10.0]]])  # (1, 1, 1)
        threshold = np.array([[[-2.0, 3.0]]])  # (1, 1, 2) — n_series=1, horizon=1
        interval = score_fn.invert(prediction, threshold)
        expected = np.array([[[[8.0, 13.0]]]])  # (1, 1, 1, 2)
        np.testing.assert_array_almost_equal(interval, expected)

    @pytest.mark.parametrize(
        "n_series,n_samples,horizon",
        [(1, 1, 1), (1, 50, 1), (3, 1, 6), (3, 50, 6)],
    )
    def test_output_shape(self, n_series: int, n_samples: int, horizon: int) -> None:
        score_fn = SignedResidual()
        rng = np.random.default_rng(42)
        pred = rng.standard_normal((n_series, n_samples, horizon))
        lower_offset = -rng.random((n_series, horizon)) - 0.1
        upper_offset = rng.random((n_series, horizon)) + 0.1
        threshold = np.stack([lower_offset, upper_offset], axis=-1)
        interval = score_fn.invert(pred, threshold)
        assert interval.shape == (n_series, n_samples, horizon, 2)

    def test_asymmetric_interval_preserved(self) -> None:
        """When |lower_offset| != |upper_offset|, interval is asymmetric."""
        score_fn = SignedResidual()
        prediction = np.array([[[100.0]]])  # (1, 1, 1)
        threshold = np.array([[[-1.0, 4.0]]])  # asymmetric
        interval = score_fn.invert(prediction, threshold)
        upper_gap = interval[..., 1] - prediction
        lower_gap = prediction - interval[..., 0]
        assert not np.allclose(upper_gap, lower_gap)
        np.testing.assert_array_almost_equal(upper_gap, np.array([[[4.0]]]))
        np.testing.assert_array_almost_equal(lower_gap, np.array([[[1.0]]]))

    def test_2d_threshold_raises(self) -> None:
        score_fn = SignedResidual()
        prediction = np.zeros((1, 1, 3))
        threshold = np.zeros((1, 3))  # missing last axis
        with pytest.raises(ValueError, match=r"\(n_series, horizon, 2\)"):
            score_fn.invert(prediction, threshold)

    def test_trailing_axis_size_3_raises(self) -> None:
        score_fn = SignedResidual()
        prediction = np.zeros((1, 1, 3))
        threshold = np.zeros((1, 3, 3))
        with pytest.raises(ValueError, match=r"\(n_series, horizon, 2\)"):
            score_fn.invert(prediction, threshold)

    def test_scalar_threshold_raises(self) -> None:
        score_fn = SignedResidual()
        prediction = np.zeros((1, 1, 3))
        with pytest.raises(ValueError, match=r"\(n_series, horizon, 2\)"):
            score_fn.invert(prediction, np.float64(1.0))

    def test_broadcasts_across_samples(self) -> None:
        """Same threshold applied across n_samples produces same offsets."""
        score_fn = SignedResidual()
        rng = np.random.default_rng(7)
        prediction = rng.standard_normal((1, 10, 1))  # 10 samples
        threshold = np.array([[[-0.5, 1.5]]])  # (1, 1, 2)
        interval = score_fn.invert(prediction, threshold)
        assert interval.shape == (1, 10, 1, 2)
        # The offset from prediction should be identical across all samples.
        lower_offsets = interval[..., 0] - prediction
        upper_offsets = interval[..., 1] - prediction
        np.testing.assert_array_almost_equal(lower_offsets, np.full((1, 10, 1), -0.5))
        np.testing.assert_array_almost_equal(upper_offsets, np.full((1, 10, 1), 1.5))
