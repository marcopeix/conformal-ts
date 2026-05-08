"""Shared helpers for Nixtla-family adapters.

The ``StatsForecastAdapter``, ``MLForecastAdapter``, and
``NeuralForecastAdapter`` all consume the same long-format Nixtla DataFrame
(``unique_id`` / ``ds`` / ``y``) and produce the same panel-shaped numpy
output. This module centralises the validation, panel↔DataFrame conversion,
and cross-validation reshaping logic so each adapter only contains the bits
that are specific to its underlying library.

The functions are pure and parameterised on column names; adapters keep
thin instance-method shims that forward to them, which preserves the
``adapter._df_to_panel(...)`` API the integration tests use.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

# A library-specific resolver that maps (model_name, available_columns, q) to
# the column name that holds the quantile-q forecast. Each Nixtla library
# names columns differently (StatsForecast vs NeuralForecast vs MLForecast);
# this signature lets ``reshape_cv_quantiles`` and ``stack_quantile_panels``
# stay library-agnostic.
ColumnResolver = Callable[[str, list[str], float], str]


def quantiles_to_levels(quantiles: NDArray[np.floating]) -> list[int]:
    """Convert quantiles in (0, 1) to a sorted list of symmetric Nixtla level integers.

    The Nixtla family (StatsForecast, NeuralForecast, MLForecast) all expose
    intervals via integer ``level`` values: ``level=80`` →
    ``<model>-lo-80`` / ``<model>-hi-80`` for the 10th and 90th percentiles.
    The conformal-ts public API speaks in raw quantiles; this helper converts
    a user-supplied quantile array to the level representation Nixtla
    libraries actually consume.

    Parameters
    ----------
    quantiles : NDArray, shape (n_quantiles,)
        Values in ``(0, 1)``. Must come in symmetric pairs around 0.5
        (e.g. ``[0.05, 0.95]``); ``q == 0.5`` is rejected because the
        level API has no median column.

    Returns
    -------
    list of int
        Sorted unique levels.

    Raises
    ------
    ValueError
        If any quantile lies outside ``(0, 1)``, equals 0.5, or lacks its
        symmetric counterpart.
    """
    arr = np.asarray(quantiles, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"quantiles must be 1-D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("quantiles must contain at least one value.")
    if not np.all((arr > 0.0) & (arr < 1.0)):
        raise ValueError(f"All quantiles must satisfy 0 < q < 1, got {arr.tolist()}.")
    if np.any(np.isclose(arr, 0.5)):
        raise ValueError(
            "Quantile 0.5 (median) is not supported by the level-based API. "
            "Pass strict (0, 0.5) ∪ (0.5, 1) values."
        )
    for q in arr:
        mirror = 1.0 - float(q)
        if not bool(np.any(np.isclose(arr, mirror))):
            raise ValueError(
                f"Quantile {float(q)} requires its symmetric pair {mirror} to "
                f"also be present in quantiles. Got {arr.tolist()}."
            )

    levels: set[int] = set()
    for q in arr:
        q_lo = float(min(float(q), 1.0 - float(q)))
        levels.add(int(round((1.0 - 2.0 * q_lo) * 100.0)))
    return sorted(levels)


def quantile_level(q: float) -> int:
    """Integer level for one quantile (e.g. 0.05 → 90)."""
    return int(round((1.0 - 2.0 * min(q, 1.0 - q)) * 100.0))


def sf_resolve_quantile_column(model_name: str, columns: list[str], q: float) -> str:
    """StatsForecast quantile column resolver.

    StatsForecast uses a deterministic naming convention:
    ``<model>-lo-<level>`` for ``q < 0.5`` and ``<model>-hi-<level>`` for
    ``q > 0.5``. If the expected column is missing, raise with a
    StatsForecast-specific message recommending interval-supporting models.
    """
    side = "lo" if q < 0.5 else "hi"
    col = f"{model_name}-{side}-{quantile_level(q)}"
    if col not in columns:
        raise ValueError(
            f"StatsForecast did not return quantile column '{col}'. "
            f"Model '{model_name}' may not support quantile prediction. "
            "Use a model with native interval support (AutoARIMA, AutoETS, etc.)."
        )
    return col


def nf_resolve_quantile_column(model_name: str, columns: list[str], q: float) -> str:
    """NeuralForecast quantile column resolver.

    NeuralForecast's column naming differs across loss types and versions.
    As of ``neuralforecast==3.1.x``:

    * ``DistributionLoss`` produces ``<model>-lo-<level>`` / ``<model>-hi-<level>``
      (no decimal suffix on the level).
    * ``MQLoss`` and friends produce ``<model>-lo-<level>.0`` /
      ``<model>-hi-<level>.0`` (with a ``.0`` decimal suffix).

    We try both conventions and return the first match. If neither column
    exists, raise ``ValueError`` listing the candidates considered.
    """
    side = "lo" if q < 0.5 else "hi"
    level = quantile_level(q)
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


def validate_pandas(df: Any) -> None:
    """Reject anything that isn't a ``pandas.DataFrame``.

    Polars input would otherwise reach the Nixtla libraries and produce
    confusing downstream errors. v0.1 only supports pandas.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "polars DataFrames are not supported in v0.1. "
            "Convert via train_df.to_pandas() before constructing the adapter."
        )


def validate_columns(df: pd.DataFrame, *, id_col: str, time_col: str, target_col: str) -> None:
    required = {id_col, time_col, target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train_df is missing required columns: {sorted(missing)}")


def validate_freq(freq: str) -> pd.DateOffset:
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


def validate_contiguity(df: pd.DataFrame, *, id_col: str, time_col: str, freq: str) -> None:
    """Verify each series has timestamps on a regular ``freq`` grid with no gaps."""
    grouped = df.sort_values(time_col).groupby(id_col, sort=False)[time_col]
    for series_id, timestamps in grouped:
        ts_sorted = timestamps.reset_index(drop=True)
        expected = pd.date_range(ts_sorted.iloc[0], ts_sorted.iloc[len(ts_sorted) - 1], freq=freq)
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
                f"expected {len(expected)} timestamps at freq='{freq}', "
                f"got {len(ts_sorted)}."
            )


def validate_no_nan(df: pd.DataFrame, *, id_col: str, target_col: str) -> None:
    nan_mask = df[target_col].isna()
    if nan_mask.any():
        offending = df.loc[nan_mask, id_col].unique().tolist()
        raise ValueError(f"NaN values found in '{target_col}' for series: {offending}")


def compute_panel_bounds(
    df: pd.DataFrame,
    horizon: int,
    *,
    id_col: str,
    time_col: str,
    freq: str,
) -> tuple[tuple[str, ...], pd.Timestamp, pd.Timestamp]:
    """Return ``(series_ids, common_start, common_end)`` for a panel DataFrame.

    Raises if the common date range is too short to support a calibration
    set of size ``2 * horizon``.
    """
    starts = df.groupby(id_col)[time_col].min()
    ends = df.groupby(id_col)[time_col].max()
    common_start: pd.Timestamp = starts.max()
    common_end: pd.Timestamp = ends.min()

    span_steps = len(pd.date_range(common_start, common_end, freq=freq)) - 1
    if span_steps < 2 * horizon:
        raise ValueError(
            f"Common date range ({common_start} to {common_end}) spans "
            f"{span_steps:.0f} steps, but at least 2 * horizon = "
            f"{2 * horizon} steps are needed for calibration."
        )

    series_ids = tuple(sorted(df[id_col].unique()))
    return series_ids, common_start, common_end


def df_to_panel(
    df: pd.DataFrame,
    value_col: str,
    *,
    series_ids: tuple[str, ...],
    id_col: str,
    time_col: str,
) -> NDArray[np.floating]:
    """Pivot a long DataFrame to shape ``(n_series, T)``.

    Series order follows ``series_ids``. Time order is ascending.
    """
    pivot = df.pivot(index=id_col, columns=time_col, values=value_col)
    pivot = pivot.loc[list(series_ids)]
    pivot = pivot.sort_index(axis=1)
    return pivot.to_numpy(dtype=np.float64)


def panel_to_df(
    panel: NDArray[np.floating],
    end_timestamp: pd.Timestamp,
    *,
    series_ids: tuple[str, ...],
    id_col: str,
    time_col: str,
    target_col: str,
    freq: str,
) -> pd.DataFrame:
    """Convert a ``(n_series, T)`` panel to a Nixtla-format long DataFrame.

    Reconstructs timestamps by stepping backward from *end_timestamp* at
    *freq* for *T* steps.
    """
    t_len = panel.shape[1]
    timestamps = pd.date_range(end=end_timestamp, periods=t_len, freq=freq)
    rows: list[pd.DataFrame] = []
    for i, sid in enumerate(series_ids):
        rows.append(
            pd.DataFrame(
                {
                    id_col: sid,
                    time_col: timestamps,
                    target_col: panel[i],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def reshape_cv(
    cv_df: pd.DataFrame,
    value_col: str,
    *,
    series_ids: tuple[str, ...],
    id_col: str,
    time_col: str,
    n_series: int,
    horizon: int,
) -> NDArray[np.floating]:
    """Reshape Nixtla-format CV output to ``(n_series, n_windows, horizon)``.

    The ``cutoff`` column determines the window axis; ``time_col`` within a
    cutoff group determines the horizon axis.
    """
    cutoffs = sorted(cv_df["cutoff"].unique())
    n_windows = len(cutoffs)

    result = np.empty((n_series, n_windows, horizon), dtype=np.float64)

    for w_idx, cutoff in enumerate(cutoffs):
        window_df = cv_df[cv_df["cutoff"] == cutoff]
        panel = df_to_panel(
            window_df,
            value_col,
            series_ids=series_ids,
            id_col=id_col,
            time_col=time_col,
        )
        result[:, w_idx, :] = panel

    return result


def reshape_cv_quantiles(
    cv_df: pd.DataFrame,
    *,
    model_name: str,
    target_col: str,
    q_arr: NDArray[np.floating],
    resolver: ColumnResolver,
    series_ids: tuple[str, ...],
    id_col: str,
    time_col: str,
    n_series: int,
    horizon: int,
    library_name: str,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Reshape a Nixtla-format quantile CV DataFrame.

    Resolves the quantile column for each entry in ``q_arr`` via ``resolver``
    (which encapsulates each library's column-name conventions and "missing
    column" error message), reshapes truths and per-quantile panels via
    :func:`reshape_cv`, and stacks the panels with quantiles on the **last
    axis**. Raises ``RuntimeError`` if any output cell is NaN.

    Returns
    -------
    quantile_predictions : NDArray, shape ``(n_series, n_windows, horizon, n_quantiles)``
        Quantiles on the last axis, in the order of ``q_arr``.
    truths : NDArray, shape ``(n_series, n_windows, horizon)``
    """
    columns = list(cv_df.columns)
    truths = reshape_cv(
        cv_df,
        target_col,
        series_ids=series_ids,
        id_col=id_col,
        time_col=time_col,
        n_series=n_series,
        horizon=horizon,
    )

    panels: list[NDArray[np.floating]] = []
    for q in q_arr:
        col = resolver(model_name, columns, float(q))
        panels.append(
            reshape_cv(
                cv_df,
                col,
                series_ids=series_ids,
                id_col=id_col,
                time_col=time_col,
                n_series=n_series,
                horizon=horizon,
            )
        )
    quantile_predictions = np.stack(panels, axis=-1)

    if np.isnan(quantile_predictions).any() or np.isnan(truths).any():
        raise RuntimeError(
            f"{library_name} cross_validation produced NaN values. This "
            f"indicates an internal error in the {library_name} "
            "cross-validation output."
        )

    return quantile_predictions, truths


def stack_quantile_panels(
    result_df: pd.DataFrame,
    *,
    model_name: str,
    q_arr: NDArray[np.floating],
    resolver: ColumnResolver,
    series_ids: tuple[str, ...],
    id_col: str,
    time_col: str,
) -> NDArray[np.floating]:
    """Read each requested quantile column from ``result_df`` and stack.

    The forecast-side counterpart to :func:`reshape_cv_quantiles`. Used by
    ``predict_quantiles`` on each Nixtla adapter: the upstream library has
    just produced a wide DataFrame indexed by (id, ds) with one column per
    quantile, and this helper pivots each requested quantile to a panel and
    stacks them with quantiles on the **middle axis**.

    Returns
    -------
    NDArray, shape ``(n_series, n_quantiles, horizon)``
    """
    columns = list(result_df.columns)
    panels: list[NDArray[np.floating]] = []
    for q in q_arr:
        col = resolver(model_name, columns, float(q))
        panels.append(
            df_to_panel(
                result_df,
                col,
                series_ids=series_ids,
                id_col=id_col,
                time_col=time_col,
            )
        )
    return np.stack(panels, axis=1)
