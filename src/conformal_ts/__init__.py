"""Conformal prediction for time series forecasting."""

from __future__ import annotations

from .adapters import CallableAdapter
from .aggregators import EWA, OnlineAggregator
from .base import CalibrationError, ConformalTSError, UnsupportedCapability
from .methods import (
    AdaptiveConformalInference,
    AggregatedAdaptiveConformalInference,
    ConformalizedQuantileRegression,
    NonexchangeableConformalPrediction,
    SplitConformal,
)
from .nonconformity import AbsoluteResidual, QuantileScore, SignedResidual

__version__ = "0.1.0"

__all__ = [
    "AbsoluteResidual",
    "AdaptiveConformalInference",
    "AggregatedAdaptiveConformalInference",
    "CalibrationError",
    "CallableAdapter",
    "ConformalTSError",
    "ConformalizedQuantileRegression",
    "EWA",
    "NonexchangeableConformalPrediction",
    "OnlineAggregator",
    "QuantileScore",
    "SignedResidual",
    "SplitConformal",
    "UnsupportedCapability",
]
