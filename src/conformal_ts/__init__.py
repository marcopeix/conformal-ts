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
    SequentialPredictiveConformalInference,
    SplitConformal,
)
from .nonconformity import AbsoluteResidual, QuantileScore, SignedResidual
from .quantile_regressors import QRFQuantileRegressor, QuantileRegressor

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
    "QRFQuantileRegressor",
    "QuantileRegressor",
    "QuantileScore",
    "SequentialPredictiveConformalInference",
    "SignedResidual",
    "SplitConformal",
    "UnsupportedCapability",
]
