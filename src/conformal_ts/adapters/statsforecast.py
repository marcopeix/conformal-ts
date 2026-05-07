"""StatsForecast adapter for conformal-ts.

Wraps a fitted ``statsforecast.StatsForecast`` instance and exposes it through
the conformal-ts adapter contract.  This is the reference adapter for the
Nixtla family of forecasting libraries.

Scope
-----
The adapter only exposes the operations conformal methods consume internally:
panel-shaped point forecasts, refit, and cross-validation. Quantile forecasts,
bootstrap ensembles, and other library-native uncertainty outputs are out of
scope — conformal-ts derives intervals from point forecasts. Methods that
need quantile or bootstrap inputs (e.g. CQR family, EnbPI) accept those
arrays directly from the user; generate them with StatsForecast and pass
them to the method.

Limitations (v0.1)
------------------
- Covariates (static, historical exog, future exog) are not supported.
- Business-day frequencies (``'B'``, ``'BM'``, …) are not tested.
- Only ``refit(history)`` is provided; no ``update_horizon`` or ``add_series``.
- ``predict`` anchors forecasts at the adapter's known end timestamp. For
  forecasting from arbitrary recent data, call ``refit`` with a newer panel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..base import Forecast, ForecasterAdapter, Series
from ..capabilities import SupportsCrossValidation, SupportsRefit
from . import _nixtla_common as _nx

try:
    import pandas as pd  # type: ignore[import-untyped]
    from statsforecast import StatsForecast  # type: ignore[import-untyped]
except ImportError as _err:
    raise ImportError(
        "StatsForecastAdapter requires the 'statsforecast' package. "
        "Install it with: pip install conformal-ts[nixtla]"
    ) from _err

if TYPE_CHECKING:
    pass


class StatsForecastAdapter(
    ForecasterAdapter,
    SupportsRefit,
    SupportsCrossValidation,
):
    """
    Adapter for a fitted :class:`statsforecast.StatsForecast` instance.

    Parameters
    ----------
    sf : StatsForecast
        An already-fitted StatsForecast instance.
    train_df : pd.DataFrame
        The long-format DataFrame used to fit ``sf``.
    horizon : int
        Forecast horizon in time steps.
    freq : str
        Pandas frequency string (e.g. ``'D'``, ``'h'``, ``'MS'``, ``'W'``).
    model_name : str
        Which model column to read from StatsForecast output. Must match one
        of the fitted model aliases.
    id_col : str
        Column identifying individual series (default ``'unique_id'``).
    time_col : str
        Timestamp column (default ``'ds'``).
    target_col : str
        Target value column (default ``'y'``).
    """

    def __init__(
        self,
        sf: StatsForecast,
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

        # --- validation (steps 1-8) ---
        self._validate_fitted(sf)
        self._validate_model_name(sf, model_name)
        _nx.validate_columns(train_df, id_col=id_col, time_col=time_col, target_col=target_col)
        self._offset = _nx.validate_freq(freq)
        _nx.validate_contiguity(train_df, id_col=id_col, time_col=time_col, freq=freq)
        _nx.validate_no_nan(train_df, id_col=id_col, target_col=target_col)
        self._series_ids, common_start, common_end = _nx.compute_panel_bounds(
            train_df, horizon, id_col=id_col, time_col=time_col, freq=freq
        )

        n_series = len(self._series_ids)
        super().__init__(horizon=horizon, n_series=n_series)

        self._sf = sf
        self._last_train_df = train_df.copy()
        self._common_start: pd.Timestamp = common_start
        self._common_end: pd.Timestamp = common_end

    # ------------------------------------------------------------------
    # Construction-time validation helpers
    # ------------------------------------------------------------------

    def _validate_fitted(self, sf: Any) -> None:
        if not isinstance(sf, StatsForecast):
            raise ValueError(f"Expected a StatsForecast instance, got {type(sf).__name__}.")
        if not (hasattr(sf, "fitted_") and sf.fitted_ is not None):
            raise ValueError("StatsForecast instance is not fitted. Call sf.fit(df) first.")

    def _validate_model_name(self, sf: StatsForecast, model_name: str) -> None:
        available = [m.alias if hasattr(m, "alias") else type(m).__name__ for m in sf.models]
        if model_name not in available:
            raise ValueError(
                f"model_name '{model_name}' not found among fitted models. Available: {available}"
            )

    # ------------------------------------------------------------------
    # Panel ↔ DataFrame conversion (thin shims over _nixtla_common)
    # ------------------------------------------------------------------

    def _df_to_panel(self, df: pd.DataFrame, value_col: str) -> NDArray[np.floating]:
        return _nx.df_to_panel(
            df,
            value_col,
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
        )

    def _panel_to_df(
        self,
        panel: NDArray[np.floating],
        end_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        return _nx.panel_to_df(
            panel,
            end_timestamp,
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            freq=self.freq,
        )

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
        history = self._validate_history(history)

        history_df = self._panel_to_df(history, self._common_end)
        result_df = self._sf.forecast(h=self.horizon, df=history_df)

        # Extract point forecasts and reshape
        forecast_panel = self._df_to_panel(result_df, self.model_name)
        # (n_series, horizon) -> (n_series, 1, horizon)
        return forecast_panel[:, np.newaxis, :]

    # ------------------------------------------------------------------
    # SupportsRefit
    # ------------------------------------------------------------------

    def refit(self, history: Series) -> None:
        """
        Refit the underlying StatsForecast model on a new history panel.

        The set of series must be identical to the original training data.
        If the new panel is longer than the previous one, the end timestamp
        shifts forward proportionally (i.e. extra length = new data appended
        at the end).

        Parameters
        ----------
        history : Series, shape (n_series, T)
        """
        history = self._validate_history(history)

        # Infer new end: extra steps = new data at the end of the panel.
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

        self._sf.fit(history_df)
        self._last_train_df = history_df.copy()

        # Recompute bounds
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

        cv_df = self._sf.cross_validation(
            df=self._last_train_df,
            h=self.horizon,
            n_windows=n_windows,
            step_size=step_size,
            refit=refit,
        )

        # Reshape into (n_series, n_windows, horizon)
        predictions = self._reshape_cv(cv_df, self.model_name)
        truths = self._reshape_cv(cv_df, self.target_col)

        if np.isnan(predictions).any() or np.isnan(truths).any():
            raise RuntimeError(
                "cross_validation produced NaN values. This indicates an "
                "internal error in the StatsForecast cross-validation output."
            )

        return predictions, truths

    def _reshape_cv(self, cv_df: pd.DataFrame, value_col: str) -> NDArray[np.floating]:
        return _nx.reshape_cv(
            cv_df,
            value_col,
            series_ids=self._series_ids,
            id_col=self.id_col,
            time_col=self.time_col,
            n_series=self.n_series,
            horizon=self.horizon,
        )
