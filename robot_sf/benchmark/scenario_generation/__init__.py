"""Distill replay-pending catalog entries from critical episode trace windows.

Entries produced here are generated scenario hypotheses, never benchmark evidence.
"""

from robot_sf.benchmark.scenario_generation.catalog_schema import (
    CATALOG_ENTRY_SCHEMA_VERSION,
    GeneratedScenarioCatalogValidationError,
    validate_catalog_entry,
)
from robot_sf.benchmark.scenario_generation.segment_extraction import extract_critical_segment

__all__ = [
    "CATALOG_ENTRY_SCHEMA_VERSION",
    "GeneratedScenarioCatalogValidationError",
    "extract_critical_segment",
    "validate_catalog_entry",
]
