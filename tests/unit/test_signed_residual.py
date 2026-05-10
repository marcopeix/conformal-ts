"""Tests for SignedResidual score function."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_ts.nonconformity.signed import SignedResidual


class TestScore:
    """SignedResidual.score returns ``truth - prediction``."""

    def test_hand_checked_values(self) -> None:
        score_fn = SignedResidual()
        # (n_series=1, n_samples=3, horizon=1)
        prediction = np.array([[[1.0], [2.0], [3.0]]])
        truth = np.array([[[1.5], [1.5], [4.5]]])
        scores = score_fn.score(prediction, truth)
        expected = np.array([[[0.5], [-0.5], [1.5]]])
        np.testing.assert_array_almost_equal(scores, expected)

    def test_sign_correctness(self) -> None:
        score_fn = SignedResidual()
        prediction = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        truth_above = np.array([[[1.5, 2.5], [3.5, 4.5]]])
        truth_below = np.array([[[0.5, 1.5], [2.5, 3.5]]])

        scores_above = score_fn.score(prediction, truth_above)
        scores_below = score_fn.score(prediction, truth_below)

        assert np.all(scores_above > 0)
        assert np.all(scores_below < 0)

    def test_shape_preserved(self) -> None:
        score_fn = SignedResidual()
        rng = np.random.default_rng(0)
        for n_series, n_samples, horizon in [(1, 10, 1), (5, 20, 12)]:
            pred = rng.standard_normal((n_series, n_samples, horizon))
            truth = rng.standard_normal((n_series, n_samples, horizon))
            scores = score_fn.score(pred, truth)
            assert scores.shape == (n_series, n_samples, horizon)


class TestInvert:
    """SignedResidual.invert produces asymmetric intervals from per-cell offsets."""

    def test_hand_checked_interval(self) -> None:
        score_fn = SignedResidual()
        # (n_series=1, n_samples=1, horizon=1)
        prediction = np.array([[[10.0]]])
        # (n_series=1, horizon=1, 2) — (lower_offset, upper_offset)
        threshold = np.array([[[-2.0, 3.0]]])

        interval = score_fn.invert(prediction, threshold)

        assert interval.shape == (1, 1, 1, 2)
        np.testing.assert_array_almost_equal(interval[..., 0], [[[8.0]]])
        np.testing.assert_array_almost_equal(interval[..., 1], [[[13.0]]])

    def test_asymmetric_offsets_broadcast_across_samples(self) -> None:
        score_fn = SignedResidual()
        # (n_series=1, n_samples=2, horizon=2)
        prediction = np.array([[[10.0, 20.0], [30.0, 40.0]]])
        # (n_series=1, horizon=2, 2)
        threshold = np.array([[[-1.0, 2.0], [-3.0, 5.0]]])

        interval = score_fn.invert(prediction, threshold)

        assert interval.shape == (1, 2, 2, 2)
        # series 0, sample 0
        np.testing.assert_array_almost_equal(interval[0, 0, :, 0], [9.0, 17.0])
        np.testing.assert_array_almost_equal(interval[0, 0, :, 1], [12.0, 25.0])
        # series 0, sample 1 — same offsets applied to a different point
        np.testing.assert_array_almost_equal(interval[0, 1, :, 0], [29.0, 37.0])
        np.testing.assert_array_almost_equal(interval[0, 1, :, 1], [32.0, 45.0])

    @pytest.mark.parametrize(
        "n_series,horizon",
        [(1, 1), (1, 12), (5, 1), (5, 12)],
    )
    def test_output_shape(self, n_series: int, horizon: int) -> None:
        score_fn = SignedResidual()
        rng = np.random.default_rng(42)
        n_samples = 3
        pred = rng.standard_normal((n_series, n_samples, horizon))
        # Build (n_series, horizon, 2) with negative lower, positive upper.
        lower = -rng.random((n_series, horizon))
        upper = rng.random((n_series, horizon))
        threshold = np.stack([lower, upper], axis=-1)

        interval = score_fn.invert(pred, threshold)
        assert interval.shape == (n_series, n_samples, horizon, 2)
        # Verify ordering: interval[..., 0] uses lower_offset, [..., 1] uses upper.
        np.testing.assert_array_almost_equal(interval[..., 0], pred + lower[:, np.newaxis, :])
        np.testing.assert_array_almost_equal(interval[..., 1], pred + upper[:, np.newaxis, :])

    def test_wrong_threshold_shape_raises(self) -> None:
        score_fn = SignedResidual()
        prediction = np.zeros((1, 1, 1))
        with pytest.raises(ValueError, match="score_threshold"):
            score_fn.invert(prediction, np.zeros((1, 1)))  # missing trailing 2-axis
        with pytest.raises(ValueError, match="score_threshold"):
            score_fn.invert(prediction, np.zeros((1, 1, 3)))  # wrong last dim
