"""SNQI (Social Navigation Quality Index) utilities.

This package centralizes computation and helpers used by weight recomputation,
optimization, and sensitivity analysis scripts. Keeping a single canonical
implementation prevents drift if the scoring formula changes.
"""

from .compute import WEIGHT_NAMES, compute_snqi, normalize_metric  # noqa: F401
