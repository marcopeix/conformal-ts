"""NeuralForecast adapter for conformal-ts.

Wraps a fitted ``neuralforecast.NeuralForecast`` instance and exposes it
through the conformal-ts adapter contract.

Scope
-----
The adapter only exposes the operations conformal methods consume internally:
panel-shaped point forecasts, refit, and cross-validation. Quantile forecasts,
bootstrap ensembles, and other library-native uncertainty outputs are out of
scope — conformal-ts derives intervals from point forecasts. NeuralForecast
does support quantile-loss outputs (``MQLoss``, ``IQLoss``, …) and
``nf.predict(quantiles=...)``, but exposing those via ``predict_quantiles``
would be a thin passthrough that adds no value. Users wanting CQR with
NeuralForecast should feed pre-computed quantile predictions through a
future ``PrecomputedQuantileAdapter``.

Limitations (v0.1)
------------------
- Static, historical, and future exogenous features are rejected at
  construction.
- Polars DataFrames are not supported. Convert with ``df.to_pandas()`` first.
- All models in the underlying ``NeuralForecast`` must share the same horizon.
- ``refit`` retrains the neural network on the new history. This is
  significantly slower than the StatsForecast / MLForecast equivalents; for
  online-style usage, batch refits.
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
    from neuralforecast import NeuralForecast  # type: ignore[import-untyped]
except ImportError as _err:
    raise ImportError(
        "NeuralForecastAdapter requires the 'neuralforecast' package. "
        "Install it with: pip install conformal-ts[nixtla]"
    ) from _err

if TYPE_CHECKING:
    pass


def _model_alias(model: Any) -> str:
    """Resolve a NeuralForecast model's effective alias.

    NF models carry an ``alias`` attribute that defaults to ``None``; in that
    case the column name in predict output falls back to the class name.
    """
    alias = getattr(model, "alias", None)
    return alias if alias is not None else type(model).__name__


class NeuralForecastAdapter(
    ForecasterAdapter,
    SupportsRefit,
    SupportsCrossValidation,
):
    """
    Adapter for a fitted :class:`neuralforecast.NeuralForecast` instance.

    The adapter wraps **point** forecasters only. Any NeuralForecast
    configuration that produces intervals natively (``prediction_intervals``)
    is rejected at construction; conformal-ts produces its own intervals
    from point forecasts.

    The horizon is read from the underlying models' ``h`` attribute. All
    models in ``nf.models`` must share the same horizon.

    Parameters
    ----------
    nf : NeuralForecast
        An already-fitted NeuralForecast instance with point models only.
    train_df : pd.DataFrame
        The long-format pandas DataFrame used to fit ``nf``. Polars DataFrames
        are not supported in v0.1.
    freq : str
        Pandas frequency string (e.g. ``'D'``, ``'h'``, ``'MS'``, ``'W'``).
        Must match the frequency the NeuralForecast instance was constructed
        with.
    model_name : str
        Which model column to read from NeuralForecast output. Must be the
        alias of a model in ``nf.models`` (or its class name if no alias was
        set explicitly).
    id_col : str
        Column identifying individual series (default ``'unique_id'``).
    time_col : str
        Timestamp column (default ``'ds'``).
    target_col : str
        Target value column (default ``'y'``).
    """

    def __init__(
        self,
        nf: NeuralForecast,
        train_df: pd.DataFrame,
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
        _nx.validate_pandas(train_df)
        self._validate_fitted(nf)
        self._validate_model_name(nf, model_name)
        self._validate_no_prediction_intervals(nf)
        horizon = self._validate_horizon_consistency(nf)
        self._validate_no_exogenous(nf, model_name)
        _nx.validate_columns(train_df, id_col=id_col, time_col=time_col, target_col=target_col)
        self._offset = _nx.validate_freq(freq)
        _nx.validate_contiguity(train_df, id_col=id_col, time_col=time_col, freq=freq)
        _nx.validate_no_nan(train_df, id_col=id_col, target_col=target_col)
        self._series_ids, common_start, common_end = _nx.compute_panel_bounds(
            train_df, horizon, id_col=id_col, time_col=time_col, freq=freq
        )

        n_series = len(self._series_ids)
        super().__init__(horizon=horizon, n_series=n_series)

        self._nf = nf
        self._last_train_df = train_df.copy()
        self._common_start: pd.Timestamp = common_start
        self._common_end: pd.Timestamp = common_end

    # ------------------------------------------------------------------
    # Construction-time validation helpers
    # ------------------------------------------------------------------

    def _validate_fitted(self, nf: Any) -> None:
        if not isinstance(nf, NeuralForecast):
            raise ValueError(f"Expected a NeuralForecast instance, got {type(nf).__name__}.")
        if not getattr(nf, "_fitted", False):
            raise ValueError("NeuralForecast instance is not fitted. Call nf.fit(df) first.")

    def _validate_model_name(self, nf: NeuralForecast, model_name: str) -> None:
        aliases = [_model_alias(m) for m in nf.models]
        if model_name not in aliases:
            raise ValueError(
                f"model_name '{model_name}' not found among fitted models. Available: {aliases}"
            )

    def _validate_no_prediction_intervals(self, nf: NeuralForecast) -> None:
        # NeuralForecast.fit assigns ``_cs_df`` to a calibration DataFrame only
        # when prediction_intervals is passed; otherwise it stays None (or
        # absent on instances that have never gone through fit).
        if getattr(nf, "_cs_df", None) is not None:
            raise ValueError(
                "NeuralForecast was fit with prediction_intervals. conformal-ts "
                "produces its own intervals from point forecasts — refit with "
                "nf.fit(df) without the prediction_intervals argument."
            )

    def _validate_horizon_consistency(self, nf: NeuralForecast) -> int:
        horizons = {_model_alias(m): int(m.h) for m in nf.models}
        unique_h = set(horizons.values())
        if len(unique_h) != 1:
            raise ValueError(
                f"All models in NeuralForecast must share the same horizon h. Got: {horizons}."
            )
        return int(nf.models[0].h)

    def _validate_no_exogenous(self, nf: NeuralForecast, model_name: str) -> None:
        target = next(m for m in nf.models if _model_alias(m) == model_name)
        categories = {
            "future exogenous": "futr_exog_list",
            "historical exogenous": "hist_exog_list",
            "static": "stat_exog_list",
        }
        for label, attr in categories.items():
            features = getattr(target, attr, None)
            if features:
                raise ValueError(
                    f"Model '{model_name}' uses {label} features ({list(features)}). "
                    "Exogenous and static features are not supported in conformal-ts v0.1."
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
        history = self._validate_history(np.asarray(history, dtype=np.float64))

        history_df = self._panel_to_df(history, self._common_end)
        result_df = self._nf.predict(df=history_df)

        forecast_panel = self._df_to_panel(result_df, self.model_name)
        return forecast_panel[:, np.newaxis, :]

    # ------------------------------------------------------------------
    # SupportsRefit
    # ------------------------------------------------------------------

    def refit(self, history: Series) -> None:
        """
        Refit the underlying NeuralForecast model on a new history panel.

        The set of series must be identical to the original training data.
        If the new panel is longer than the previous one, the end timestamp
        shifts forward proportionally.

        Note that NeuralForecast's ``fit`` retrains the neural networks from
        scratch and may take significantly longer than the StatsForecast or
        MLForecast equivalents. For online-style usage, batch the refits.

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

        self._nf.fit(history_df)
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

        cv_df = self._nf.cross_validation(
            df=self._last_train_df,
            n_windows=n_windows,
            step_size=step_size,
            refit=refit,
        )

        predictions = self._reshape_cv(cv_df, self.model_name)
        truths = self._reshape_cv(cv_df, self.target_col)

        if np.isnan(predictions).any() or np.isnan(truths).any():
            raise RuntimeError(
                "cross_validation produced NaN values. This indicates an "
                "internal error in the NeuralForecast cross-validation output."
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
