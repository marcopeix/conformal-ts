"""Nonconformity score functions."""

from __future__ import annotations

from .absolute import AbsoluteResidual
from .quantile import QuantileScore

__all__ = ["AbsoluteResidual", "QuantileScore"]
