"""MLForecast adapter for conformal-ts.

Wraps a fitted ``mlforecast.MLForecast`` instance and exposes it through the
conformal-ts adapter contract.

Limitations (v0.1)
------------------
- Quantile prediction is not supported (``SupportsQuantiles`` is not inherited).
  conformal-ts produces its own intervals from point forecasts; quantile-based
  methods (CQR family) require an adapter that exposes native quantile output.
- Bootstrap prediction is not supported (``SupportsBootstrap`` is not inherited).
- Static features, sample weights, and natively conformal predictions
  (``prediction_intervals=...``) are rejected at construction time.
- Business-day frequencies (``'B'``, ``'BM'``, …) are not tested.
- Only ``refit(history)`` is provided; no incremental update.
- ``predict`` anchors forecasts at the adapter's known end timestamp. For
  forecasting from arbitrary recent data, call ``refit`` with a newer panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..base import Forecast, ForecasterAdapter, Series
from ..capabilities import SupportsCrossValidation, SupportsRefit

try:
    import pandas as pd  # type: ignore[import-untyped]
    from mlforecast import MLForecast  # type: ignore[import-untyped]
except ImportError as _err:
    raise ImportError(
        "MLForecastAdapter requires the 'mlforecast' package. "
        "Install it with: pip install conformal-ts[nixtla]"
    ) from _err

if TYPE_CHECKING:
    pass


class MLForecastAdapter(
    ForecasterAdapter,
    SupportsRefit,
    SupportsCrossValidation,
):
    """
    Adapter for a fitted :class:`mlforecast.MLForecast` instance.

    The adapter wraps **point** forecasters only. Any MLForecast configuration
    that produces intervals natively (``prediction_intervals``) is rejected at
    construction; conformal-ts produces its own intervals from point forecasts.

    Parameters
    ----------
    mlf : MLForecast
        An already-fitted MLForecast instance with point models only.
    train_df : pd.DataFrame
        The long-format pandas DataFrame used to fit ``mlf``. Polars DataFrames
        are not supported in v0.1.
    horizon : int
        Forecast horizon in time steps.
    freq : str
        Pandas frequency string (e.g. ``'D'``, ``'h'``, ``'MS'``, ``'W'``).
    model_name : str
        Which model column to read from MLForecast output. Must be an alias
        present in ``mlf.models_``.
    id_col : str
        Column identifying individual series (default ``'unique_id'``).
    time_col : str
        Timestamp column (default ``'ds'``).
    target_col : str
        Target value column (default ``'y'``).
    """

    def __init__(
        self,
        mlf: MLForecast,
        train_df: pd.DataFrame,
        horizon: int,
        freq: str,
        model_name: str,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        self.freq = freq
        self.model_name = model_name
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col

        # --- validation ---
        self._validate_pandas(train_df)
        self._validate_fitted(mlf)
        self._validate_model_name(mlf, model_name)
        self._validate_no_prediction_intervals(mlf)
        self._validate_no_static_features(mlf)
        self._validate_no_weight_col(mlf)
        self._validate_columns(train_df)
        self._offset = self._validate_freq(freq)
        self._validate_contiguity(train_df)
        self._validate_no_nan(train_df)
        self._series_ids, common_start, common_end = self._compute_panel_bounds(train_df, horizon)
        self._validate_horizon_compatibility(mlf, horizon)

        n_series = len(self._series_ids)
        super().__init__(horizon=horizon, n_series=n_series)

        self._mlf = mlf
        self._last_train_df = train_df.copy()
        self._common_start: pd.Timestamp = common_start
        self._common_end: pd.Timestamp = common_end

    # ------------------------------------------------------------------
    # Construction-time validation helpers
    # ------------------------------------------------------------------

    def _validate_pandas(self, df: Any) -> None:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                "polars DataFrames are not supported in v0.1. "
                "Convert via train_df.to_pandas() before constructing the adapter."
            )

    def _validate_fitted(self, mlf: Any) -> None:
        if not isinstance(mlf, MLForecast):
            raise ValueError(f"Expected an MLForecast instance, got {type(mlf).__name__}.")
        if not (hasattr(mlf, "models_") and bool(mlf.models_)):
            raise ValueError("MLForecast instance is not fitted. Call mlf.fit(df) first.")

    def _validate_model_name(self, mlf: MLForecast, model_name: str) -> None:
        if model_name not in mlf.models_:
            raise ValueError(
                f"model_name '{model_name}' not found among fitted models. "
                f"Available: {list(mlf.models_.keys())}"
            )

    def _validate_no_prediction_intervals(self, mlf: MLForecast) -> None:
        # MLForecast.fit assigns ``_cs_df = None`` and only writes a DataFrame to
        # it when prediction_intervals is passed. Use getattr so the check
        # survives older mlforecast versions where the attribute may be absent.
        if getattr(mlf, "_cs_df", None) is not None:
            raise ValueError(
                "MLForecast was fit with prediction_intervals. conformal-ts "
                "produces its own intervals from point forecasts — refit with "
                "mlf.fit(df) without the prediction_intervals argument."
            )

    def _validate_no_static_features(self, mlf: MLForecast) -> None:
        # Different mlforecast versions stash this either on ``mlf`` directly
        # or on the inner ``mlf.ts`` core. Check both.
        static = getattr(mlf, "static_features_", None)
        if not static:
            ts = getattr(mlf, "ts", None)
            static = getattr(ts, "static_features", None) if ts is not None else None
        if static:
            raise ValueError(
                "Static features are not supported in conformal-ts v0.1. "
                "Refit MLForecast without static_features."
            )

    def _validate_no_weight_col(self, mlf: MLForecast) -> None:
        weight = getattr(mlf, "weight_col", None)
        if weight is None:
            ts = getattr(mlf, "ts", None)
            weight = getattr(ts, "weight_col", None) if ts is not None else None
        if weight is not None:
            raise ValueError("Sample weights are not supported in conformal-ts v0.1.")

    def _validate_horizon_compatibility(self, mlf: MLForecast, horizon: int) -> None:
        max_horizon = getattr(mlf, "max_horizon", None)
        if max_horizon is None:
            ts = getattr(mlf, "ts", None)
            max_horizon = getattr(ts, "max_horizon", None) if ts is not None else None
        if max_horizon is not None and max_horizon < horizon:
            raise ValueError(
                f"MLForecast was fit with max_horizon={max_horizon}, cannot "
                f"forecast {horizon} steps. Refit with max_horizon >= {horizon} "
                "or with the default recursive strategy."
            )

        horizons_ = getattr(mlf, "horizons_", None)
        if horizons_ is None:
            ts = getattr(mlf, "ts", None)
            internal = getattr(ts, "_horizons", None) if ts is not None else None
            # Internal storage in mlforecast is 0-indexed; the public contract
            # documented in fit() is 1-indexed. Convert before comparing.
            if internal is not None:
                horizons_ = [int(h) + 1 for h in internal]
        if horizons_ is not None:
            required = set(range(1, horizon + 1))
            present = set(int(h) for h in horizons_)
            missing = sorted(required - present)
            if missing:
                raise ValueError(
                    f"MLForecast was fit with horizons={sorted(present)}, missing "
                    f"required steps {missing} for horizon={horizon}. Refit with "
                    f"horizons covering 1..{horizon} or with the default recursive strategy."
                )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = {self.id_col, self.time_col, self.target_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"train_df is missing required columns: {sorted(missing)}")

    def _validate_freq(self, freq: str) -> pd.DateOffset:
        """Validate that *freq* is recognised by pandas.

        Returns the offset object for downstream date arithmetic.
        """
        try:
            offset = pd.tseries.frequencies.to_offset(freq)
        except ValueError as exc:
            raise ValueError(
                f"Invalid frequency '{freq}': {exc}. "
                "Use a pandas frequency string such as 'D', 'h', 'MS', or 'W'."
            ) from exc
        if offset is None:
            raise ValueError(f"Invalid frequency '{freq}'.")
        return offset

    def _validate_contiguity(self, df: pd.DataFrame) -> None:
        grouped = df.sort_values(self.time_col).groupby(self.id_col, sort=False)[self.time_col]
        for series_id, timestamps in grouped:
            ts_sorted = timestamps.reset_index(drop=True)
            expected = pd.date_range(
                ts_sorted.iloc[0], ts_sorted.iloc[len(ts_sorted) - 1], freq=self.freq
            )
            if len(ts_sorted) != len(expected) or not (ts_sorted.values == expected.values).all():
                for i in range(min(len(ts_sorted), len(expected))):
                    if ts_sorted.iloc[i] != expected[i]:
                        prev = ts_sorted.iloc[i - 1] if i > 0 else "N/A"
                        raise ValueError(
                            f"Non-contiguous timestamps in series '{series_id}': "
                            f"unexpected {ts_sorted.iloc[i]} at position {i} "
                            f"(previous: {prev}, expected: {expected[i]})."
                        )
                raise ValueError(
                    f"Non-contiguous timestamps in series '{series_id}': "
                    f"expected {len(expected)} timestamps at freq='{self.freq}', "
                    f"got {len(ts_sorted)}."
                )

    def _validate_no_nan(self, df: pd.DataFrame) -> None:
        nan_mask = df[self.target_col].isna()
        if nan_mask.any():
            offending = df.loc[nan_mask, self.id_col].unique().tolist()
            raise ValueError(f"NaN values found in '{self.target_col}' for series: {offending}")

    def _compute_panel_bounds(
        self, df: pd.DataFrame, horizon: int
    ) -> tuple[tuple[str, ...], pd.Timestamp, pd.Timestamp]:
        starts = df.groupby(self.id_col)[self.time_col].min()
        ends = df.groupby(self.id_col)[self.time_col].max()
        common_start: pd.Timestamp = starts.max()
        common_end: pd.Timestamp = ends.min()

        span_steps = len(pd.date_range(common_start, common_end, freq=self.freq)) - 1
        if span_steps < 2 * horizon:
            raise ValueError(
                f"Common date range ({common_start} to {common_end}) spans "
                f"{span_steps:.0f} steps, but at least 2 * horizon = "
                f"{2 * horizon} steps are needed for calibration."
            )

        series_ids = tuple(sorted(df[self.id_col].unique()))
        return series_ids, common_start, common_end

    # ------------------------------------------------------------------
    # Panel ↔ DataFrame conversion
    # ------------------------------------------------------------------

    def _df_to_panel(self, df: pd.DataFrame, value_col: str) -> NDArray[np.floating]:
        """Pivot a long DataFrame to shape ``(n_series, T)``.

        Series order follows ``self._series_ids``. Time order is ascending.
        """
        pivot = df.pivot(index=self.id_col, columns=self.time_col, values=value_col)
        pivot = pivot.loc[list(self._series_ids)]
        pivot = pivot.sort_index(axis=1)
        return pivot.to_numpy(dtype=np.float64)

    def _panel_to_df(
        self,
        panel: NDArray[np.floating],
        end_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """Convert a ``(n_series, T)`` panel to a long DataFrame.

        Reconstructs timestamps by stepping backward from *end_timestamp* at
        ``self.freq`` for *T* steps.
        """
        n_s, t_len = panel.shape
        timestamps = pd.date_range(end=end_timestamp, periods=t_len, freq=self.freq)
        rows: list[pd.DataFrame] = []
        for i, sid in enumerate(self._series_ids):
            rows.append(
                pd.DataFrame(
                    {
                        self.id_col: sid,
                        self.time_col: timestamps,
                        self.target_col: panel[i],
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------
    # ForecasterAdapter contract
    # ------------------------------------------------------------------

    def predict(self, history: Series) -> Forecast:
        """
        Produce a point forecast anchored at the adapter's known end timestamp.

        Parameters
        ----------
        history : Series, shape (n_series, T)

        Returns
        -------
        Forecast, shape (n_series, 1, horizon)
        """
        history = self._validate_history(np.asarray(history, dtype=np.float64))

        history_df = self._panel_to_df(history, self._common_end)
        result_df = self._mlf.predict(h=self.horizon, new_df=history_df)

        forecast_panel = self._df_to_panel(result_df, self.model_name)
        return forecast_panel[:, np.newaxis, :]

    # ------------------------------------------------------------------
    # SupportsRefit
    # ------------------------------------------------------------------

    def refit(self, history: Series) -> None:
        """
        Refit the underlying MLForecast model on a new history panel.

        The set of series must be identical to the original training data.
        If the new panel is longer than the previous one, the end timestamp
        shifts forward proportionally.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        """
        history = self._validate_history(np.asarray(history, dtype=np.float64))

        old_t_len = self._last_train_df.groupby(self.id_col)[self.time_col].count().iloc[0]
        delta = int(history.shape[1] - old_t_len)
        new_end = self._common_end + delta * self._offset
        history_df = self._panel_to_df(history, new_end)
        new_ids = tuple(sorted(history_df[self.id_col].unique()))
        if new_ids != self._series_ids:
            raise ValueError(
                f"Series IDs changed during refit. "
                f"Expected {list(self._series_ids)}, got {list(new_ids)}."
            )

        self._mlf.fit(history_df)
        self._last_train_df = history_df.copy()

        starts = history_df.groupby(self.id_col)[self.time_col].min()
        ends = history_df.groupby(self.id_col)[self.time_col].max()
        self._common_start = starts.max()
        self._common_end = ends.min()

    # ------------------------------------------------------------------
    # SupportsCrossValidation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        n_windows: int,
        step_size: int,
        refit: bool | int = False,
    ) -> tuple[Forecast, Forecast]:
        """
        Run rolling-origin cross-validation.

        Parameters
        ----------
        n_windows : int
            Number of evaluation windows (>= 1).
        step_size : int
            Steps between successive windows (>= 1).
        refit : bool or int
            Whether (or how often) to refit between windows.

        Returns
        -------
        predictions : Forecast, shape (n_series, n_windows, horizon)
        truths : Forecast, shape (n_series, n_windows, horizon)
        """
        if n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {n_windows}")
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")

        cv_df = self._mlf.cross_validation(
            df=self._last_train_df,
            h=self.horizon,
            n_windows=n_windows,
            step_size=step_size,
            refit=refit,
        )
        # MLForecast.cross_validation drops ``models_`` on the underlying
        # instance and instructs callers to refit. Restore the fitted state on
        # the stored training data so subsequent ``predict`` calls work.
        self._mlf.fit(self._last_train_df)

        predictions = self._reshape_cv(cv_df, self.model_name)
        truths = self._reshape_cv(cv_df, self.target_col)

        if np.isnan(predictions).any() or np.isnan(truths).any():
            raise RuntimeError(
                "cross_validation produced NaN values. This indicates an "
                "internal error in the MLForecast cross-validation output."
            )

        return predictions, truths

    def _reshape_cv(self, cv_df: pd.DataFrame, value_col: str) -> NDArray[np.floating]:
        """Reshape cross-validation output to ``(n_series, n_windows, horizon)``.

        The cutoff column determines the window axis; ``ds`` within a cutoff
        group determines the horizon axis.
        """
        cutoffs = sorted(cv_df["cutoff"].unique())
        n_windows = len(cutoffs)

        result = np.empty((self.n_series, n_windows, self.horizon), dtype=np.float64)

        for w_idx, cutoff in enumerate(cutoffs):
            window_df = cv_df[cv_df["cutoff"] == cutoff]
            panel = self._df_to_panel(window_df, value_col)
            result[:, w_idx, :] = panel

        return result
