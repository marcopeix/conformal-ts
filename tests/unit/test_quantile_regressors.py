"""Tests for the QuantileRegressor ABC and its implementations."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_ts.quantile_regressors.base import QuantileRegressor


class MockQuantileRegressor(QuantileRegressor):
    """Test-only regressor: returns ``np.quantile(y_train, q)`` regardless of X."""

    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._y = np.asarray(y, dtype=np.float64)

    def predict_quantile(self, X: np.ndarray, q: float) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("MockQuantileRegressor must be fit before predict_quantile.")
        return np.full(X.shape[0], float(np.quantile(self._y, q)))


class TestABCPluggability:
    """The QuantileRegressor ABC is genuinely pluggable."""

    def test_mock_implements_abc(self) -> None:
        reg = MockQuantileRegressor()
        assert isinstance(reg, QuantileRegressor)

    def test_mock_returns_empirical_quantiles(self) -> None:
        reg = MockQuantileRegressor()
        y = np.linspace(0.0, 1.0, 101)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((101, 3))
        reg.fit(X, y)

        X_query = rng.standard_normal((5, 3))
        # 0.5-quantile of linspace(0, 1, 101) is 0.5.
        out_50 = reg.predict_quantile(X_query, 0.5)
        np.testing.assert_allclose(out_50, np.full(5, 0.5), atol=1e-12)

        # 0.1- and 0.9-quantiles are monotone.
        out_10 = reg.predict_quantile(X_query, 0.1)
        out_90 = reg.predict_quantile(X_query, 0.9)
        assert np.all(out_10 < out_50)
        assert np.all(out_90 > out_50)


@pytest.mark.skipif(
    pytest.importorskip("quantile_forest", reason="quantile-forest not installed") is None,
    reason="quantile-forest not installed",
)
class TestQRFQuantileRegressor:
    """QRFQuantileRegressor fits and predicts on synthetic data."""

    def test_fit_then_predict_quantile(self) -> None:
        from conformal_ts.quantile_regressors.qrf import QRFQuantileRegressor

        rng = np.random.default_rng(0)
        n = 500
        X = rng.standard_normal((n, 4))
        # Linear-ish target with Gaussian noise.
        y = X.sum(axis=1) + rng.normal(0.0, 0.5, n)

        reg = QRFQuantileRegressor(n_estimators=50, random_state=0)
        reg.fit(X, y)

        X_query = rng.standard_normal((20, 4))
        q10 = reg.predict_quantile(X_query, 0.1)
        q90 = reg.predict_quantile(X_query, 0.9)

        assert q10.shape == (20,)
        assert q90.shape == (20,)
        # Lower quantile is below upper quantile on synthetic monotone data.
        assert np.all(q10 < q90)

    def test_predict_quantile_before_fit_raises(self) -> None:
        from conformal_ts.quantile_regressors.qrf import QRFQuantileRegressor

        reg = QRFQuantileRegressor()
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict_quantile(np.zeros((1, 1)), 0.5)
