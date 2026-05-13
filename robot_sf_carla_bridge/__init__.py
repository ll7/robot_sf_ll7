"""Import-safe CARLA bridge surfaces for Robot-SF transfer experiments.

The package must remain usable without CARLA installed. Runtime replay integrations should call
the availability helpers before importing or invoking CARLA-specific APIs.
"""

from robot_sf_carla_bridge.availability import (
    AVAILABILITY_SCHEMA_VERSION,
    CarlaUnavailableError,
    check_carla_availability,
    load_availability_schema,
    require_carla,
)
from robot_sf_carla_bridge.export import (
    BATCH_VALIDATION_SUMMARY_SCHEMA_VERSION,
    EXPORT_MANIFEST_SCHEMA_VERSION,
    EXPORT_SCHEMA_VERSION,
    CertificateRef,
    PedestrianReplaySpec,
    Pose2D,
    RobotReplaySpec,
    ScenarioReplayRef,
    SimulationSpec,
    Waypoint2D,
    build_export_payload,
    build_export_payload_from_map_definition,
    build_export_payload_from_scenario_entry,
    build_export_payloads_from_scenario_file,
    load_batch_validation_summary_schema,
    load_export_manifest_payloads,
    load_export_manifest_schema,
    load_export_schema,
    read_export_manifest,
    read_export_payload,
    resolve_export_manifest_payload_paths,
    validate_export_payload,
    write_export_payload,
    write_export_records,
)
from robot_sf_carla_bridge.parity import (
    DEFAULT_PARITY_METRICS,
    MetricParityRow,
    compare_oracle_replay_metrics,
)
from robot_sf_carla_bridge.replay_smoke import (
    T1_ORACLE_REPLAY_SMOKE_SCHEMA_VERSION,
    build_t1_oracle_replay_smoke_setup,
    select_t0_export_payload,
    validate_t1_replay_catalog_payload,
)
from robot_sf_carla_bridge.schema_catalog import (
    SCHEMA_CATALOG_VERSION,
    list_carla_bridge_schema_catalog,
    load_schema_catalog_schema,
)

__all__ = [
    "AVAILABILITY_SCHEMA_VERSION",
    "BATCH_VALIDATION_SUMMARY_SCHEMA_VERSION",
    "DEFAULT_PARITY_METRICS",
    "EXPORT_MANIFEST_SCHEMA_VERSION",
    "EXPORT_SCHEMA_VERSION",
    "SCHEMA_CATALOG_VERSION",
    "T1_ORACLE_REPLAY_SMOKE_SCHEMA_VERSION",
    "CarlaUnavailableError",
    "CertificateRef",
    "MetricParityRow",
    "PedestrianReplaySpec",
    "Pose2D",
    "RobotReplaySpec",
    "ScenarioReplayRef",
    "SimulationSpec",
    "Waypoint2D",
    "build_export_payload",
    "build_export_payload_from_map_definition",
    "build_export_payload_from_scenario_entry",
    "build_export_payloads_from_scenario_file",
    "build_t1_oracle_replay_smoke_setup",
    "check_carla_availability",
    "compare_oracle_replay_metrics",
    "list_carla_bridge_schema_catalog",
    "load_availability_schema",
    "load_batch_validation_summary_schema",
    "load_export_manifest_payloads",
    "load_export_manifest_schema",
    "load_export_schema",
    "load_schema_catalog_schema",
    "read_export_manifest",
    "read_export_payload",
    "require_carla",
    "resolve_export_manifest_payload_paths",
    "select_t0_export_payload",
    "validate_export_payload",
    "validate_t1_replay_catalog_payload",
    "write_export_payload",
    "write_export_records",
]
