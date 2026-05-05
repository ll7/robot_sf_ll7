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

__all__ = [
    "AVAILABILITY_SCHEMA_VERSION",
    "EXPORT_MANIFEST_SCHEMA_VERSION",
    "EXPORT_SCHEMA_VERSION",
    "CarlaUnavailableError",
    "CertificateRef",
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
    "check_carla_availability",
    "load_availability_schema",
    "load_export_manifest_payloads",
    "load_export_manifest_schema",
    "load_export_schema",
    "read_export_manifest",
    "read_export_payload",
    "require_carla",
    "resolve_export_manifest_payload_paths",
    "validate_export_payload",
    "write_export_payload",
    "write_export_records",
]
