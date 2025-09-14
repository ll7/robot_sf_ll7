"""SNQI (Social Navigation Quality Index) utilities.

This package centralizes computation and helpers used by weight recomputation,
optimization, and sensitivity analysis scripts. Keeping a single canonical
implementation prevents drift if the scoring formula changes.
"""

from .compute import WEIGHT_NAMES, compute_snqi, normalize_metric  # noqa: F401
from .exit_codes import (  # noqa: F401
    EXIT_INPUT_ERROR,
    EXIT_MISSING_METRIC_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    EXIT_VALIDATION_ERROR,
)
from .schema import EXPECTED_SCHEMA_VERSION, assert_all_finite, validate_snqi  # noqa: F401
from .weights_validation import validate_weights_mapping  # noqa: F401

__all__ = [
    # Core compute
    "compute_snqi",
    "normalize_metric",
    "WEIGHT_NAMES",
    # Validation helpers
    "validate_snqi",
    "assert_all_finite",
    "EXPECTED_SCHEMA_VERSION",
    "validate_weights_mapping",
    # Exit codes
    "EXIT_SUCCESS",
    "EXIT_INPUT_ERROR",
    "EXIT_VALIDATION_ERROR",
    "EXIT_RUNTIME_ERROR",
    "EXIT_MISSING_METRIC_ERROR",
]
