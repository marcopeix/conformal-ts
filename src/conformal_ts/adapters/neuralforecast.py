"""NeuralForecast adapter for conformal-ts.

Wraps a fitted ``neuralforecast.NeuralForecast`` instance and exposes it
through the conformal-ts adapter contract.

Scope
-----
The adapter exposes panel-shaped point forecasts, quantile forecasts, refit,
and cross-validation. Quantile output is gated at call time on whether the
underlying model was fit with a probabilistic loss (any of the quantile-loss
classes — ``QuantileLoss`` / ``MQLoss`` / ``IQLoss`` and their Huber variants
— or any loss that exposes ``is_distribution_output = True`` such as
``DistributionLoss``, ``PMM``, ``GMM``).

The adapter always declares the :class:`SupportsQuantiles` and
:class:`SupportsCrossValidationQuantiles` mixins. ``isinstance`` checks
therefore return True regardless of the loss; the runtime gate is the
:attr:`_supports_quantiles_runtime` flag, which is False for point-loss
models. Calling ``predict_quantiles`` or ``cross_validate_quantiles`` on a
point-loss adapter raises :class:`UnsupportedCapability` at call time. This
"declare always, gate at runtime" pattern is necessary because the capability
depends on a constructor argument (the loss), not the adapter class itself.

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

from ..base import Forecast, ForecasterAdapter, Series, UnsupportedCapability
from ..capabilities import (
    SupportsCrossValidation,
    SupportsCrossValidationQuantiles,
    SupportsQuantiles,
    SupportsRefit,
)
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


# NeuralForecast losses that produce quantile output deterministically.
# Distribution losses (DistributionLoss / PMM / GMM / ISQF) advertise themselves
# via the ``is_distribution_output`` attribute and are detected separately.
QUANTILE_LOSS_CLASS_NAMES = frozenset(
    {
        "QuantileLoss",
        "MQLoss",
        "IQLoss",
        "HuberQLoss",
        "HuberMQLoss",
        "HuberIQLoss",
    }
)


def _supports_quantiles(loss: Any) -> bool:
    """Detect whether a NeuralForecast loss can produce quantile output.

    Two categories qualify:

    * Distribution losses (``DistributionLoss``, ``PMM``, ``GMM``, ``ISQF``)
      which set ``is_distribution_output = True`` and produce quantiles via
      sampling.
    * Direct quantile losses (``QuantileLoss``, ``MQLoss``, ``IQLoss``, and
      their ``Huber*`` variants) which produce quantiles deterministically.
    """
    if getattr(loss, "is_distribution_output", False):
        return True
    return type(loss).__name__ in QUANTILE_LOSS_CLASS_NAMES


def _resolve_quantile_column(model_name: str, columns: list[str], q: float) -> str:
    """Find the NeuralForecast output column for a given quantile.

    NeuralForecast's column naming convention varies across loss types and
    versions. As of ``neuralforecast==3.1.x``:

    * ``DistributionLoss`` produces ``<model>-lo-<level>`` / ``<model>-hi-<level>``
      (no decimal suffix on the level).
    * ``MQLoss`` and friends produce ``<model>-lo-<level>.0`` / ``<model>-hi-<level>.0``
      (with a ``.0`` decimal suffix).

    We try both conventions and return the first match. If neither column is
    present, raise ``ValueError`` listing the candidates considered.
    """
    side = "lo" if q < 0.5 else "hi"
    level = _nx.quantile_level(q)
    candidates = [
        f"{model_name}-{side}-{level}",
        f"{model_name}-{side}-{level}.0",
    ]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"NeuralForecast did not return quantile column for q={q}. "
        f"Tried {candidates}. Available columns: {columns}. For MQLoss, "
        "the requested quantile must match a symmetric pair the loss was "
        "trained with."
    )


class NeuralForecastAdapter(
    ForecasterAdapter,
    SupportsRefit,
    SupportsCrossValidation,
    SupportsQuantiles,
    SupportsCrossValidationQuantiles,
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

        # Runtime quantile capability is loss-driven, not class-driven. We
        # always declare SupportsQuantiles / SupportsCrossValidationQuantiles
        # at the class level (so isinstance is True and CQR's REQUIRED_CAPABILITIES
        # check passes), then gate the actual methods at call time.
        selected_model = next(m for m in nf.models if _model_alias(m) == model_name)
        self._loss_class_name: str = type(selected_model.loss).__name__
        self._supports_quantiles_runtime: bool = _supports_quantiles(selected_model.loss)

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

    # ------------------------------------------------------------------
    # SupportsQuantiles (runtime-gated)
    # ------------------------------------------------------------------

    def _require_runtime_quantile_support(self) -> None:
        if not self._supports_quantiles_runtime:
            raise UnsupportedCapability(
                f"NeuralForecast model '{self.model_name}' was fit with "
                f"non-probabilistic loss '{self._loss_class_name}'. Use a "
                "probabilistic loss: any quantile loss (QuantileLoss, MQLoss, "
                "IQLoss, HuberQLoss, HuberMQLoss, HuberIQLoss) or any "
                "DistributionLoss / PMM / GMM."
            )

    def predict_quantiles(
        self,
        history: Series,
        quantiles: NDArray[np.floating],
    ) -> Forecast:
        """
        Produce quantile forecasts via NeuralForecast's level-based API.

        Parameters
        ----------
        history : Series, shape (n_series, T)
        quantiles : NDArray, shape (n_quantiles,)
            Values in ``(0, 1)``. Must come in symmetric pairs around 0.5;
            ``q == 0.5`` is rejected. NeuralForecast's runtime behaviour for
            ``MQLoss`` ignores arbitrary quantile requests in favour of the
            quantiles the loss was trained with — this adapter therefore
            uses ``level=`` (the underlying API both losses honour) and
            requires symmetric inputs.

        Returns
        -------
        Forecast, shape (n_series, n_quantiles, horizon)
            Quantiles are on the middle axis, in the order requested.

        Raises
        ------
        UnsupportedCapability
            If the underlying model was fit with a non-probabilistic loss.
        ValueError
            If the symmetric-pair / range checks fail, or if the model did
            not return a requested quantile column. For ``MQLoss``, the
            requested quantiles must match a symmetric pair the loss was
            trained with.
        """
        self._require_runtime_quantile_support()
        history = self._validate_history(np.asarray(history, dtype=np.float64))
        q_arr = np.asarray(quantiles, dtype=np.float64)
        levels = _nx.quantiles_to_levels(q_arr)

        history_df = self._panel_to_df(history, self._common_end)
        result_df = self._nf.predict(df=history_df, level=levels)

        columns = list(result_df.columns)
        panels: list[NDArray[np.floating]] = []
        for q in q_arr:
            col = _resolve_quantile_column(self.model_name, columns, float(q))
            panels.append(self._df_to_panel(result_df, col))  # (n_series, horizon)

        return np.stack(panels, axis=1)  # (n_series, n_quantiles, horizon)

    # ------------------------------------------------------------------
    # SupportsCrossValidationQuantiles (runtime-gated)
    # ------------------------------------------------------------------

    def cross_validate_quantiles(
        self,
        n_windows: int,
        step_size: int,
        quantiles: NDArray[np.floating],
        refit: bool | int = False,
    ) -> tuple[Forecast, Forecast]:
        """
        Run rolling-origin cross-validation producing quantile forecasts.

        Parameters
        ----------
        n_windows : int
            Number of evaluation windows (>= 1).
        step_size : int
            Steps between successive windows (>= 1).
        quantiles : NDArray, shape (n_quantiles,)
            Values in ``(0, 1)``. Symmetric-pair / range rules of
            :func:`_nixtla_common.quantiles_to_levels` apply.
        refit : bool or int
            Whether (or how often) to refit between windows.

        Returns
        -------
        quantile_predictions : Forecast, shape (n_series, n_windows, horizon, n_quantiles)
            Quantiles are on the **last axis**, in the order requested.
        truths : Forecast, shape (n_series, n_windows, horizon)

        Raises
        ------
        UnsupportedCapability
            If the underlying model was fit with a non-probabilistic loss.
        ValueError
            If ``n_windows`` / ``step_size`` are invalid, the quantile
            validation fails, or the model did not return a requested
            quantile column.
        RuntimeError
            If the cross-validation output contains NaN.
        """
        self._require_runtime_quantile_support()
        if n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {n_windows}")
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")
        q_arr = np.asarray(quantiles, dtype=np.float64)
        levels = _nx.quantiles_to_levels(q_arr)

        cv_df = self._nf.cross_validation(
            df=self._last_train_df,
            n_windows=n_windows,
            step_size=step_size,
            refit=refit,
            level=levels,
        )

        truths = self._reshape_cv(cv_df, self.target_col)

        columns = list(cv_df.columns)
        panels: list[NDArray[np.floating]] = []
        for q in q_arr:
            col = _resolve_quantile_column(self.model_name, columns, float(q))
            panels.append(self._reshape_cv(cv_df, col))

        # (n_series, n_windows, horizon, n_quantiles)
        quantile_predictions = np.stack(panels, axis=-1)

        if np.isnan(quantile_predictions).any() or np.isnan(truths).any():
            raise RuntimeError(
                "cross_validation produced NaN values. This indicates an "
                "internal error in the NeuralForecast cross-validation output."
            )

        return quantile_predictions, truths
