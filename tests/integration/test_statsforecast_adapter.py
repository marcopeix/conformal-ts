"""Integration tests for StatsForecastAdapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sf_mod = pytest.importorskip("statsforecast")
StatsForecast = sf_mod.StatsForecast

from statsforecast.models import AutoETS, Naive  # noqa: E402

from conformal_ts.adapters.statsforecast import StatsForecastAdapter  # noqa: E402
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
    model_cls: type = Naive,
    **model_kwargs: object,
) -> tuple[StatsForecastAdapter, pd.DataFrame]:
    """Fit and return an adapter plus its training DataFrame."""
    df = _make_df(n_series=n_series, t_len=t_len, freq=freq)
    model = model_cls(**model_kwargs) if model_kwargs else model_cls()
    sf = StatsForecast(models=[model], freq=freq)
    sf.fit(df)
    model_name = model.alias if hasattr(model, "alias") else type(model).__name__
    adapter = StatsForecastAdapter(
        sf=sf, train_df=df, horizon=horizon, freq=freq, model_name=model_name
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

    def test_multi_model_sf(self) -> None:
        """Construct from a multi-model SF, selecting one model."""
        df = _make_df(n_series=2, t_len=100)
        sf = StatsForecast(models=[Naive(), AutoETS(season_length=1)], freq="D")
        sf.fit(df)
        adapter = StatsForecastAdapter(sf=sf, train_df=df, horizon=5, freq="D", model_name="Naive")
        assert adapter.n_series == 2
        assert adapter.model_name == "Naive"

    # --- validation failure tests ---

    def test_not_fitted_raises(self) -> None:
        df = _make_df()
        sf = StatsForecast(models=[Naive()], freq="D")
        with pytest.raises(ValueError, match="not fitted"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="D", model_name="Naive")

    def test_bad_model_name_raises(self) -> None:
        df = _make_df()
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(df)
        with pytest.raises(ValueError, match="model_name 'NoSuchModel'"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="D", model_name="NoSuchModel")

    def test_missing_columns_raises(self) -> None:
        df = _make_df()
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(df)
        bad_df = df.rename(columns={"y": "target"})
        with pytest.raises(ValueError, match="missing required columns"):
            StatsForecastAdapter(sf=sf, train_df=bad_df, horizon=3, freq="D", model_name="Naive")

    def test_invalid_freq_raises(self) -> None:
        df = _make_df()
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(df)
        with pytest.raises(ValueError, match="Invalid frequency"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="BOGUS", model_name="Naive")

    def test_non_contiguous_raises(self) -> None:
        df = _make_df(t_len=100)
        # Drop a row to create a gap
        df = df.drop(index=50).reset_index(drop=True)
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(df)
        with pytest.raises(ValueError, match="Non-contiguous"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="D", model_name="Naive")

    def test_nan_target_raises(self) -> None:
        df = _make_df()
        df.loc[10, "y"] = np.nan
        sf = StatsForecast(models=[Naive()], freq="D")
        # Fit on clean data
        clean_df = _make_df()
        sf.fit(clean_df)
        with pytest.raises(ValueError, match="NaN"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=3, freq="D", model_name="Naive")

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
                        "y": rng.standard_normal(50),
                    }
                ),
                pd.DataFrame(
                    {
                        "unique_id": "b",
                        "ds": dates2,
                        "y": rng.standard_normal(50),
                    }
                ),
            ],
            ignore_index=True,
        )
        sf = StatsForecast(models=[Naive()], freq="D")
        sf.fit(df)
        with pytest.raises(ValueError, match="steps are needed"):
            StatsForecastAdapter(sf=sf, train_df=df, horizon=20, freq="D", model_name="Naive")

    def test_not_statsforecast_instance_raises(self) -> None:
        df = _make_df()
        with pytest.raises(ValueError, match="StatsForecast instance"):
            StatsForecastAdapter(
                sf="not_sf",  # type: ignore[arg-type]
                train_df=df,
                horizon=3,
                freq="D",
                model_name="Naive",
            )


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
        # Use different subset of history (different last values)
        history_short = history_full[:, :50]
        r1 = adapter.predict(history_full)
        r2 = adapter.predict(history_short)
        # Naive model repeats last value, so different histories → different forecasts
        assert not np.array_equal(r1, r2)


# ===========================================================================
# refit tests
# ===========================================================================


class TestRefit:
    """refit() updates the model and anchoring."""

    def test_refit_with_longer_panel_shifts_end(self) -> None:
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=80)
        old_end = adapter._common_end

        # Create a longer panel (20 extra steps = 20 new days)
        new_df = _make_df(n_series=1, t_len=100, seed=1)
        new_panel = adapter._df_to_panel(new_df, "y")

        adapter.refit(new_panel)

        # End timestamp should have shifted forward by 20 days
        assert adapter._common_end == old_end + pd.Timedelta(days=20)
        # Predict still works
        result = adapter.predict(new_panel)
        assert result.shape == (1, 1, 3)

    def test_refit_with_different_series_raises(self) -> None:
        adapter, _ = _make_adapter(n_series=2, horizon=3)
        # Create panel with 3 series instead of 2
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
        """CV-based and predict_batch-based calibration produce the same quantile."""
        adapter, df = _make_adapter(n_series=1, horizon=3, t_len=100)

        # Fast path: delegate to cross_validate
        method_cv = SplitConformal(adapter, alpha=0.1)
        cal_cv = method_cv.calibrate(n_windows=10, step_size=1, refit=False)

        # Slow path: replicate cross_validate manually via predict_batch
        cv_preds, cv_truths = adapter.cross_validate(n_windows=10, step_size=1, refit=False)
        # Build histories matching the CV cutoffs.
        full = adapter._df_to_panel(df, "y")
        T = full.shape[1]
        cal_histories = [full[:, : T - 3 - (10 - 1 - w)] for w in range(10)]

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
        # Interval should bracket the point forecast.
        assert np.all(result.interval[..., 0] <= result.point)
        assert np.all(result.point <= result.interval[..., 1])
