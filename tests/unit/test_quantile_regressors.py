"""Tests for the QuantileRegressor ABC and QRFQuantileRegressor."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from conformal_ts.quantile_regressors.base import QuantileRegressor
from conformal_ts.quantile_regressors.qrf import QRFQuantileRegressor

try:
    import quantile_forest  # type: ignore[import-untyped]  # noqa: F401

    _HAS_QF = True
except ImportError:
    _HAS_QF = False

qf_required = pytest.mark.skipif(not _HAS_QF, reason="quantile-forest is not installed")


class MockQuantileRegressor(QuantileRegressor):
    """Returns ``np.quantile(y, q)`` unconditionally on the fitted target.

    Useful as the test workhorse for the SPCI pipeline: it exercises the
    full :class:`QuantileRegressor` ABC without pulling in the
    quantile-forest dependency, and behaves like an empirical quantile of
    calibration residuals (so SPCI's coverage degrades to split-conformal
    semantics on iid data).
    """

    def __init__(self) -> None:
        self._y: NDArray[np.floating] | None = None

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        self._y = np.asarray(y, dtype=np.float64)

    def predict_quantile(self, X: NDArray[np.floating], q: float) -> NDArray[np.floating]:
        if self._y is None:
            raise RuntimeError("MockQuantileRegressor must be fit first.")
        return np.full(X.shape[0], float(np.quantile(self._y, q)), dtype=np.float64)


# ===========================================================================
# Mock regressor (ABC sanity)
# ===========================================================================


class TestMockRegressor:
    """The mock exercises the ABC contract without external dependencies."""

    def test_fit_and_predict(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        y = rng.normal(size=50)

        reg = MockQuantileRegressor()
        reg.fit(X, y)
        out_lo = reg.predict_quantile(X[:5], 0.1)
        out_hi = reg.predict_quantile(X[:5], 0.9)

        assert out_lo.shape == (5,)
        assert out_hi.shape == (5,)
        # Mock returns the empirical quantile of y broadcast across rows.
        np.testing.assert_allclose(out_lo, np.full(5, float(np.quantile(y, 0.1))))
        np.testing.assert_allclose(out_hi, np.full(5, float(np.quantile(y, 0.9))))

    def test_predict_before_fit_raises(self) -> None:
        reg = MockQuantileRegressor()
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict_quantile(np.zeros((1, 3)), 0.5)


# ===========================================================================
# QRFQuantileRegressor (gated on quantile-forest install)
# ===========================================================================


@qf_required
class TestQRFRegressor:
    """QRF fits and predicts on synthetic data, respects quantile ordering."""

    def test_fit_and_predict_shape(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        y = rng.normal(size=200)

        reg = QRFQuantileRegressor(n_estimators=20, random_state=0)
        reg.fit(X, y)

        out = reg.predict_quantile(X[:7], 0.5)
        assert out.shape == (7,)
        assert out.dtype == np.float64

    def test_quantile_ordering_on_monotone_data(self) -> None:
        """Lower-q predictions should not exceed higher-q predictions."""
        rng = np.random.default_rng(0)
        n = 300
        X = rng.normal(size=(n, 2))
        # Target depends on X[:, 0] plus heteroskedastic noise so quantiles
        # vary across rows.
        y = X[:, 0] + rng.normal(size=n) * (1.0 + np.abs(X[:, 1]))

        reg = QRFQuantileRegressor(n_estimators=50, random_state=0)
        reg.fit(X, y)

        # Predict on a fresh sample of rows so the test isn't trivially
        # satisfied by training-set memorization.
        X_test = rng.normal(size=(30, 2))
        q_lo = reg.predict_quantile(X_test, 0.1)
        q_hi = reg.predict_quantile(X_test, 0.9)
        assert (q_lo <= q_hi + 1e-9).all()

    def test_predict_before_fit_raises(self) -> None:
        reg = QRFQuantileRegressor(n_estimators=10, random_state=0)
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict_quantile(np.zeros((2, 3)), 0.5)
