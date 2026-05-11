"""Diagnostics for conformal prediction methods."""

from __future__ import annotations

from .conditional import coverage_by_group, coverage_by_magnitude_bin
from .coverage import (
    coverage_by_horizon,
    coverage_by_series,
    coverage_per_cell,
    marginal_coverage,
    rolling_coverage,
)
from .method_diagnostics import (
    aci_state,
    agaci_state,
    method_state,
    nexcp_state,
    spci_state,
)
from .reports import Report, evaluate
from .scoring import (
    coverage_width_summary,
    mean_interval_width,
    pinball_loss,
    winkler_score,
)

__all__ = [
    # Layer 1 — coverage
    "marginal_coverage",
    "coverage_by_horizon",
    "coverage_by_series",
    "coverage_per_cell",
    "rolling_coverage",
    # Layer 1 — scoring
    "winkler_score",
    "mean_interval_width",
    "pinball_loss",
    "coverage_width_summary",
    # Layer 1 — conditional
    "coverage_by_magnitude_bin",
    "coverage_by_group",
    # Layer 2 — method-specific
    "aci_state",
    "agaci_state",
    "nexcp_state",
    "spci_state",
    "method_state",
    # Integration
    "Report",
    "evaluate",
]
