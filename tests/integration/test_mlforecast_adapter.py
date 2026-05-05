"""Integration tests for MLForecastAdapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

mlf_mod = pytest.importorskip("mlforecast")
MLForecast = mlf_mod.MLForecast

from mlforecast.lag_transforms import ExpandingMean  # noqa: E402
from mlforecast.target_transforms import Differences  # noqa: E402
from mlforecast.utils import PredictionIntervals  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from conformal_ts.adapters.mlforecast import MLForecastAdapter  # noqa: E402
from conformal_ts.capabilities import (  # noqa: E402
    SupportsBootstrap,
    SupportsCrossValidation,
    SupportsQuantiles,
    SupportsRefit,
)
from conformal_ts.methods.split import SplitConformal  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(
    n_series: int = 1,
    t_len: int = 100,
    freq: str = "D",
    start: str = "2020-01-01",
    seed: int = 0,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """Create a synthetic long-format DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=t_len, freq=freq)
    rows: list[pd.DataFrame] = []
    for i in range(n_series):
        sid = f"series_{i}"
        rows.append(
            pd.DataFrame(
                {
                    id_col: sid,
                    time_col: dates,
                    target_col: rng.standard_normal(t_len).cumsum(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _make_adapter(
    n_series: int = 1,
    t_len: int = 100,
    horizon: int = 3,
    freq: str = "D",
    model_name: str = "Ridge",
    **mlf_kwargs: object,
) -> tuple[MLForecastAdapter, pd.DataFrame]:
    """Fit and return an adapter plus its training DataFrame."""
    df = _make_df(n_series=n_series, t_len=t_len, freq=freq)
    base_kwargs: dict[str, object] = {"lags": [1, 2, 3]}
    base_kwargs.update(mlf_kwargs)
    mlf = MLForecast(models={model_name: Ridge()}, freq=freq, **base_kwargs)
    mlf.fit(df)
    adapter = MLForecastAdapter(
        mlf=mlf, train_df=df, horizon=horizon, freq=freq, model_name=model_name
    )
    return adapter, df


# ===========================================================================
# Construction tests
# ===========================================================================


class TestConstruction:
    """Successful construction and validation failures."""

    def test_single_model(self) -> None:
        adapter, _ = _make_adapter(n_series=1, horizon=3)
        assert adapter.n_series == 1
        assert adapter.horizon == 3

    def test_multi_model_mlf(self) -> None:
        """Construct from a multi-model MLForecast, selecting one model."""
        df = _make_df(n_series=2, t_len=100)
        mlf = MLForecast(
            models={"Ridge": Ridge(), "Ridge2": Ridge(alpha=2.0)},
            freq="D",
            lags=[1, 2, 3],
        )
        mlf.fit(df)
        adapter = MLForecastAdapter(mlf=mlf, train_df=df, horizon=5, freq="D", model_name="Ridge")
        assert adapter.n_series == 2
        assert adapter.model_name == "Ridge"

    # --- validation failure tests ---

    def test_not_fitted_raises(self) -> None:
        df = _make_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        with pytest.raises(ValueError, match="not fitted"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")

    def test_bad_model_name_raises(self) -> None:
        df = _make_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df)
        with pytest.raises(ValueError, match="model_name 'NoSuchModel'"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="NoSuchModel")

    def test_missing_columns_raises(self) -> None:
        df = _make_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df)
        bad_df = df.rename(columns={"y": "target"})
        with pytest.raises(ValueError, match="missing required columns"):
            MLForecastAdapter(mlf=mlf, train_df=bad_df, horizon=3, freq="D", model_name="Ridge")

    def test_invalid_freq_raises(self) -> None:
        df = _make_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df)
        with pytest.raises(ValueError, match="Invalid frequency"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="BOGUS", model_name="Ridge")

    def test_non_contiguous_raises(self) -> None:
        df = _make_df(t_len=100)
        df = df.drop(index=50).reset_index(drop=True)
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        # MLForecast itself complains about missing dates; silence by using validate_data=False
        mlf.fit(df, validate_data=False)
        with pytest.raises(ValueError, match="Non-contiguous"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")

    def test_nan_target_raises(self) -> None:
        clean_df = _make_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(clean_df)
        bad_df = clean_df.copy()
        bad_df.loc[10, "y"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            MLForecastAdapter(mlf=mlf, train_df=bad_df, horizon=3, freq="D", model_name="Ridge")

    def test_insufficient_common_range_raises(self) -> None:
        """Two series with barely overlapping ranges fail the 2*horizon check."""
        dates1 = pd.date_range("2020-01-01", periods=50, freq="D")
        dates2 = pd.date_range("2020-02-15", periods=50, freq="D")
        rng = np.random.default_rng(0)
        df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "unique_id": "a",
                        "ds": dates1,
                        "y": rng.standard_normal(50).cumsum(),
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": "b",
                        "ds": dates2,
                        "y": rng.standard_normal(50).cumsum(),
                    }
                ),
            ],
            ignore_index=True,
        )
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df)
        with pytest.raises(ValueError, match="steps are needed"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=20, freq="D", model_name="Ridge")

    def test_not_mlforecast_instance_raises(self) -> None:
        df = _make_df()
        with pytest.raises(ValueError, match="MLForecast instance"):
            MLForecastAdapter(
                mlf="not_mlf",  # type: ignore[arg-type]
                train_df=df,
                horizon=3,
                freq="D",
                model_name="Ridge",
            )

    def test_polars_df_rejected(self) -> None:
        df = _make_df()
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df)

        class FakePolarsDF:
            """Stand-in that isn't a pandas.DataFrame."""

        with pytest.raises(ValueError, match="polars DataFrames are not supported"):
            MLForecastAdapter(
                mlf=mlf,
                train_df=FakePolarsDF(),  # type: ignore[arg-type]
                horizon=3,
                freq="D",
                model_name="Ridge",
            )

    def test_prediction_intervals_rejected(self) -> None:
        """A fit with prediction_intervals must be rejected at construction."""
        df = _make_df(n_series=1, t_len=100)
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(
            df,
            prediction_intervals=PredictionIntervals(
                n_windows=2, h=3, method="conformal_distribution"
            ),
        )
        # Confirm the test setup exercises the rejection path.
        assert mlf._cs_df is not None
        with pytest.raises(ValueError, match="produces its own intervals"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")

    def test_max_horizon_too_small_rejected(self) -> None:
        df = _make_df(n_series=1, t_len=100)
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df, max_horizon=5)
        with pytest.raises(ValueError, match="max_horizon=5"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=10, freq="D", model_name="Ridge")

    def test_horizons_missing_steps_rejected(self) -> None:
        df = _make_df(n_series=1, t_len=100)
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df, horizons=[1, 3, 5])
        with pytest.raises(ValueError, match="missing required steps"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=5, freq="D", model_name="Ridge")

    def test_static_features_rejected(self) -> None:
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        rows = []
        for sid in ["a", "b"]:
            rows.append(
                pd.DataFrame(
                    {
                        "unique_id": sid,
                        "ds": dates,
                        "y": rng.standard_normal(100).cumsum(),
                        # constant feature so MLForecast accepts it as static
                        "feat_x": float(rng.standard_normal()),
                    }
                )
            )
        df = pd.concat(rows, ignore_index=True)
        mlf = MLForecast(models={"Ridge": Ridge()}, freq="D", lags=[1, 2, 3])
        mlf.fit(df, static_features=["feat_x"])
        with pytest.raises(ValueError, match="Static features are not supported"):
            MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")

    def test_lag_and_target_transforms_pass_through(self) -> None:
        """Differences + ExpandingMean should construct cleanly and predict."""
        df = _make_df(n_series=2, t_len=100)
        mlf = MLForecast(
            models={"Ridge": Ridge()},
            freq="D",
            lags=[1, 2, 3],
            target_transforms=[Differences([1])],
            lag_transforms={1: [ExpandingMean()]},
        )
        mlf.fit(df)
        adapter = MLForecastAdapter(mlf=mlf, train_df=df, horizon=3, freq="D", model_name="Ridge")
        history = adapter._df_to_panel(df, "y")
        out = adapter.predict(history)
        assert out.shape == (2, 1, 3)
        assert not np.isnan(out).any()


# ===========================================================================
# predict tests
# ===========================================================================


class TestPredict:
    """predict() shape, determinism, and anchor semantics."""

    @pytest.mark.parametrize(
        "n_series,horizon",
        [(1, 1), (1, 12), (5, 1), (5, 12)],
    )
    def test_output_shape(self, n_series: int, horizon: int) -> None:
        adapter, df = _make_adapter(n_series=n_series, horizon=horizon, t_len=100)
        history = adapter._df_to_panel(df, "y")
        result = adapter.predict(history)
        assert result.shape == (n_series, 1, horizon)

    def test_determinism(self) -> None:
        adapter, df = _make_adapter(n_series=2, horizon=5)
        history = adapter._df_to_panel(df, "y")
        r1 = adapter.predict(history)
        r2 = adapter.predict(history)
        np.testing.assert_array_equal(r1, r2)

    def test_different_histories_produce_different_forecasts(self) -> None:
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=100)
        history_full = adapter._df_to_panel(df, "y")
        history_short = history_full[:, :50]
        r1 = adapter.predict(history_full)
        r2 = adapter.predict(history_short)
        assert not np.array_equal(r1, r2)


# ===========================================================================
# refit tests
# ===========================================================================


class TestRefit:
    """refit() updates the model and anchoring."""

    def test_refit_with_longer_panel_shifts_end(self) -> None:
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=80)
        old_end = adapter._common_end

        new_df = _make_df(n_series=1, t_len=100, seed=1)
        new_panel = adapter._df_to_panel(new_df, "y")

        adapter.refit(new_panel)

        assert adapter._common_end == old_end + pd.Timedelta(days=20)
        result = adapter.predict(new_panel)
        assert result.shape == (1, 1, 3)

    def test_refit_with_different_series_raises(self) -> None:
        adapter, _ = _make_adapter(n_series=2, horizon=3)
        bad_panel = np.random.default_rng(0).standard_normal((3, 100))
        with pytest.raises(ValueError, match="leading axis"):
            adapter.refit(bad_panel)


# ===========================================================================
# cross_validate tests
# ===========================================================================


class TestCrossValidate:
    """cross_validate() shapes and validation."""

    @pytest.mark.parametrize("n_series", [1, 3])
    def test_output_shape(self, n_series: int) -> None:
        horizon = 3
        n_windows = 4
        step_size = 2
        adapter, _ = _make_adapter(n_series=n_series, horizon=horizon, t_len=100)
        preds, truths = adapter.cross_validate(n_windows=n_windows, step_size=step_size)
        assert preds.shape == (n_series, n_windows, horizon)
        assert truths.shape == (n_series, n_windows, horizon)

    def test_refit_true_vs_false_same_shape(self) -> None:
        adapter, _ = _make_adapter(n_series=1, horizon=3, t_len=100)
        p1, t1 = adapter.cross_validate(n_windows=3, step_size=1, refit=False)
        p2, t2 = adapter.cross_validate(n_windows=3, step_size=1, refit=True)
        assert p1.shape == p2.shape
        assert t1.shape == t2.shape

    def test_invalid_n_windows_raises(self) -> None:
        adapter, _ = _make_adapter()
        with pytest.raises(ValueError, match="n_windows"):
            adapter.cross_validate(n_windows=0, step_size=1)

    def test_invalid_step_size_raises(self) -> None:
        adapter, _ = _make_adapter()
        with pytest.raises(ValueError, match="step_size"):
            adapter.cross_validate(n_windows=2, step_size=0)


# ===========================================================================
# Capability declarations
# ===========================================================================


class TestCapabilities:
    """isinstance checks for capability mixins."""

    def test_supports_refit(self) -> None:
        adapter, _ = _make_adapter()
        assert isinstance(adapter, SupportsRefit)

    def test_supports_cross_validation(self) -> None:
        adapter, _ = _make_adapter()
        assert isinstance(adapter, SupportsCrossValidation)

    def test_not_supports_quantiles(self) -> None:
        adapter, _ = _make_adapter()
        assert not isinstance(adapter, SupportsQuantiles)

    def test_not_supports_bootstrap(self) -> None:
        adapter, _ = _make_adapter()
        assert not isinstance(adapter, SupportsBootstrap)


# ===========================================================================
# SplitConformal calibration via cross_validate
# ===========================================================================


class TestSplitConformalCV:
    """SplitConformal.calibrate(n_windows=...) dispatches to forecaster.cross_validate."""

    def test_cv_path_matches_explicit_path(self) -> None:
        """CV-based and predict_batch-based calibration produce the same quantile.

        MLForecast's ``cross_validation(refit=False)`` trains the model exactly
        once on the data up to the first cutoff (``T - h - (n_windows-1)*step_size``
        steps). The explicit path therefore needs to refit on the same prefix
        before producing batch predictions for the equivalence to hold.
        """
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=100)

        method_cv = SplitConformal(adapter, alpha=0.1)
        cal_cv = method_cv.calibrate(n_windows=10, step_size=1, refit=False)

        cv_preds, cv_truths = adapter.cross_validate(n_windows=10, step_size=1, refit=False)
        full = adapter._df_to_panel(df, "y")
        T = full.shape[1]
        cal_histories = [full[:, : T - 3 - (10 - 1 - w)] for w in range(10)]

        # Match the training data MLForecast's CV used internally.
        adapter.refit(full[:, : T - 3 - (10 - 1)])

        method_explicit = SplitConformal(adapter, alpha=0.1)
        cal_explicit = method_explicit.calibrate(cal_histories, cv_truths)

        np.testing.assert_array_almost_equal(cal_cv.score_quantile, cal_explicit.score_quantile)
        assert cal_cv.n_calibration_samples == cal_explicit.n_calibration_samples

    def test_cv_calibration_then_predict(self) -> None:
        """End-to-end: calibrate via CV, then predict produces a valid interval."""
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=100)
        method = SplitConformal(adapter, alpha=0.1)
        method.calibrate(n_windows=10, step_size=1)

        history = adapter._df_to_panel(df, "y")
        result = method.predict(history)

        assert result.point.shape == (1, 1, 3)
        assert result.interval.shape == (1, 1, 3, 2)
        assert np.all(result.interval[..., 0] <= result.point)
        assert np.all(result.point <= result.interval[..., 1])
