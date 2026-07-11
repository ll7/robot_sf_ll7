"""Distill replay-pending catalog entries from critical episode trace windows.

Entries produced here are generated scenario hypotheses, never benchmark evidence.
"""

from robot_sf.benchmark.scenario_generation.adaptive_selector import (
    AdaptiveSelectionSpec,
    GeneratedScenarioAdaptiveSelectionError,
    run_adaptive_selection,
    select_generated_proposals,
)
from robot_sf.benchmark.scenario_generation.archive_sampler import (
    ArchiveSamplingSpec,
    GeneratedScenarioArchiveSamplingError,
    run_archive_sampling,
    sample_generated_archive,
)
from robot_sf.benchmark.scenario_generation.catalog_schema import (
    CATALOG_ENTRY_SCHEMA_VERSION,
    GeneratedScenarioCatalogValidationError,
    validate_catalog_entry,
)
from robot_sf.benchmark.scenario_generation.replay_adapter import (
    GeneratedScenarioMaterialization,
    apply_generated_replay_runtime,
    dump_generated_scenario_yaml,
    generated_replay_status_entry,
    materialize_generated_scenario,
)
from robot_sf.benchmark.scenario_generation.segment_extraction import extract_critical_segment

__all__ = [
    "CATALOG_ENTRY_SCHEMA_VERSION",
    "AdaptiveSelectionSpec",
    "ArchiveSamplingSpec",
    "GeneratedScenarioAdaptiveSelectionError",
    "GeneratedScenarioArchiveSamplingError",
    "GeneratedScenarioCatalogValidationError",
    "GeneratedScenarioMaterialization",
    "apply_generated_replay_runtime",
    "dump_generated_scenario_yaml",
    "extract_critical_segment",
    "generated_replay_status_entry",
    "materialize_generated_scenario",
    "run_adaptive_selection",
    "run_archive_sampling",
    "sample_generated_archive",
    "select_generated_proposals",
    "validate_catalog_entry",
]
