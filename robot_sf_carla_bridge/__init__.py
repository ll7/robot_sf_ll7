"""Import-safe CARLA bridge surfaces for Robot-SF transfer experiments.

The package must remain usable without CARLA installed. Runtime replay integrations should call
the availability helpers before importing or invoking CARLA-specific APIs.
"""

from robot_sf_carla_bridge.availability import check_carla_availability
from robot_sf_carla_bridge.export import (
    EXPORT_SCHEMA_VERSION,
    CertificateRef,
    PedestrianReplaySpec,
    Pose2D,
    RobotReplaySpec,
    ScenarioReplayRef,
    SimulationSpec,
    build_export_payload,
    load_export_schema,
    read_export_payload,
    validate_export_payload,
    write_export_payload,
)

__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "CertificateRef",
    "PedestrianReplaySpec",
    "Pose2D",
    "RobotReplaySpec",
    "ScenarioReplayRef",
    "SimulationSpec",
    "build_export_payload",
    "check_carla_availability",
    "load_export_schema",
    "read_export_payload",
    "validate_export_payload",
    "write_export_payload",
]
