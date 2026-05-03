"""Conformal prediction for time series forecasting."""

from __future__ import annotations

from .adapters import CallableAdapter
from .base import CalibrationError, ConformalTSError, UnsupportedCapability
from .methods import SplitConformal
from .nonconformity import AbsoluteResidual

__version__ = "0.1.0"

__all__ = [
    "CallableAdapter",
    "SplitConformal",
    "AbsoluteResidual",
    "CalibrationError",
    "ConformalTSError",
    "UnsupportedCapability",
]
