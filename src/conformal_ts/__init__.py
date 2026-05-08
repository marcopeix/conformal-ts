"""Conformal prediction for time series forecasting."""

from __future__ import annotations

from .adapters import CallableAdapter
from .base import CalibrationError, ConformalTSError, UnsupportedCapability
from .methods import (
    AdaptiveConformalInference,
    ConformalizedQuantileRegression,
    SplitConformal,
)
from .nonconformity import AbsoluteResidual, QuantileScore

__version__ = "0.1.0"

__all__ = [
    "AdaptiveConformalInference",
    "CallableAdapter",
    "ConformalizedQuantileRegression",
    "SplitConformal",
    "AbsoluteResidual",
    "QuantileScore",
    "CalibrationError",
    "ConformalTSError",
    "UnsupportedCapability",
]
