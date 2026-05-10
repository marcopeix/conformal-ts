"""Integration test: SPCI over StatsForecastAdapter with the QRF default regressor.

SPCI's online holdout loop refits the adapter at each step (so the prediction
anchor advances with new observations). To keep the runtime tractable we use
AutoARIMA's lightweight cousin ``AutoARIMA`` with restricted search, and a
high ``refit_every`` so that the QRF refit cost is amortised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sf_mod = pytest.importorskip("statsforecast")
pytest.importorskip("quantile_forest")
StatsForecast = sf_mod.StatsForecast

from statsforecast.models import AutoARIMA  # noqa: E402

from conformal_ts.adapters.statsforecast import StatsForecastAdapter  # noqa: E402
from conformal_ts.methods.spci import SequentialPredictiveConformalInference  # noqa: E402
from conformal_ts.quantile_regressors.qrf import QRFQuantileRegressor  # noqa: E402


def _make_data(n_series: int, t_len: int, freq: str = "D", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=t_len, freq=freq)
    rows: list[pd.DataFrame] = []
    for i in range(n_series):
        rows.append(
            pd.DataFrame(
                {
                    "unique_id": f"series_{i}",
                    "ds": dates,
                    "y": rng.standard_normal(t_len).cumsum(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


class TestSPCIStatsForecast:
    """End-to-end SPCI on a real AutoARIMA model via the CV calibration path."""

    def test_cv_calibration_then_online_cycle(self) -> None:
        n_series = 3
        train_len = 400
        n_cal = 300
        n_holdout = 50
        horizon = 1
        alpha = 0.1
        window_size = 50
        freq = "D"
        t_len = train_len + n_holdout + horizon

        df = _make_data(n_series=n_series, t_len=t_len, freq=freq, seed=0)

        df_train = df.groupby("unique_id").head(train_len).reset_index(drop=True)
        sf = StatsForecast(models=[AutoARIMA(max_p=1, max_q=1, max_d=1)], freq=freq)
        sf.fit(df_train)

        adapter = StatsForecastAdapter(
            sf=sf,
            train_df=df_train,
            horizon=horizon,
            freq=freq,
            model_name="AutoARIMA",
        )

        def factory() -> QRFQuantileRegressor:
            return QRFQuantileRegressor(n_estimators=50, min_samples_leaf=5, random_state=0)

        method = SequentialPredictiveConformalInference(
            adapter,
            alpha=alpha,
            window_size=window_size,
            regressor_factory=factory,
            beta_grid_size=11,
            refit_every=5,
        )
        cal = method.calibrate(n_windows=n_cal, step_size=1, refit=False)

        assert cal.diagnostics["path"] == "cross_validation"
        assert cal.diagnostics["regressor_class"] == "QRFQuantileRegressor"
        assert cal.diagnostics["window_size"] == window_size
        assert cal.n_calibration_samples == n_cal

        full_panel = adapter._df_to_panel(df, "y")

        covered = 0
        total = 0
        asymmetry_seen = False
        for w in range(n_holdout):
            new_history = full_panel[:, : train_len + w]
            adapter.refit(new_history)

            truth = full_panel[:, train_len + w : train_len + w + horizon]
            result = method.predict(new_history)
            assert result.point.shape == (n_series, 1, horizon)
            assert result.interval.shape == (n_series, 1, horizon, 2)

            # Asymmetry: |lower offset| != |upper offset| somewhere.
            lower_offsets = result.interval[..., 0] - result.point
            upper_offsets = result.interval[..., 1] - result.point
            if np.any(np.abs(np.abs(lower_offsets) - np.abs(upper_offsets)) > 1e-3):
                asymmetry_seen = True

            truth_3d = truth[:, np.newaxis, :]
            in_interval = (truth_3d >= result.interval[..., 0]) & (
                truth_3d <= result.interval[..., 1]
            )
            covered += int(in_interval.sum())
            total += int(in_interval.size)

            method.update(result.point, truth_3d)

        coverage = covered / total
        assert abs(coverage - (1 - alpha)) < 0.07, (
            f"Empirical coverage {coverage:.4f} deviates from target "
            f"{1 - alpha:.2f} by more than 0.07."
        )
        assert asymmetry_seen, (
            "SPCI intervals should be asymmetric on at least one cell over the holdout."
        )
