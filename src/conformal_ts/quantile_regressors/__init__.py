"""Quantile-regression models used by SPCI and related conformal methods."""

from __future__ import annotations

from .base import QuantileRegressor

try:
    from .qrf import QRFQuantileRegressor

    __all__ = ["QRFQuantileRegressor", "QuantileRegressor"]
except ImportError:
    # quantile-forest is an optional dependency installed via the ``spci`` extra.
    __all__ = ["QuantileRegressor"]
