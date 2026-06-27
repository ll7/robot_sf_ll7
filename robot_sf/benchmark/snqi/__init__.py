"""SNQI (Social Navigation Quality Index) utilities.

This package centralizes computation and helpers used by weight recomputation,
optimization, and sensitivity analysis scripts. Keeping a single canonical
implementation prevents drift if the scoring formula changes.
"""

from .compute import WEIGHT_NAMES, compute_snqi, normalize_metric
from .exit_codes import (
    EXIT_INPUT_ERROR,
    EXIT_MISSING_METRIC_ERROR,
    EXIT_OPTIONAL_DEPS_MISSING,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    EXIT_VALIDATION_ERROR,
)
from .schema import EXPECTED_SCHEMA_VERSION, assert_all_finite, validate_snqi
from .weights_inventory import (
    SNQIWeightProvenanceError,
    WeightInventoryReport,
    build_inventory_report,
    detect_conflicts,
    inventory_weight_sets,
    preflight_snqi_weight_sets,
)
from .weights_validation import validate_weights_mapping

__all__ = [
    "EXIT_INPUT_ERROR",
    "EXIT_MISSING_METRIC_ERROR",
    "EXIT_OPTIONAL_DEPS_MISSING",
    "EXIT_RUNTIME_ERROR",
    # Exit codes
    "EXIT_SUCCESS",
    "EXIT_VALIDATION_ERROR",
    "EXPECTED_SCHEMA_VERSION",
    "WEIGHT_NAMES",
    # Weight-set provenance inventory
    "SNQIWeightProvenanceError",
    "WeightInventoryReport",
    "assert_all_finite",
    "build_inventory_report",
    # Core compute
    "compute_snqi",
    "detect_conflicts",
    "inventory_weight_sets",
    "normalize_metric",
    "preflight_snqi_weight_sets",
    # Validation helpers
    "validate_snqi",
    "validate_weights_mapping",
]
