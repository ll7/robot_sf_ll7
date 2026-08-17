"""Compatibility import for the canonical scenario-schema validation helpers."""

from __future__ import annotations

import sys

from robot_sf.benchmark.scenario import scenario_schema as _canonical
from robot_sf.benchmark.scenario.scenario_schema import (
    SCENARIO_MATRIX_SCHEMA_VERSION,
    SCHEMA_FILE,
    load_scenario_schema,
    validate_scenario_list,
    validate_scenario_matrix_metadata,
)

__all__ = (
    "SCENARIO_MATRIX_SCHEMA_VERSION",
    "SCHEMA_FILE",
    "load_scenario_schema",
    "validate_scenario_list",
    "validate_scenario_matrix_metadata",
)

sys.modules[__name__] = _canonical
