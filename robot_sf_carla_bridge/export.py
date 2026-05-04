"""T0 neutral replay export helpers for future CARLA oracle replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import jsonschema

EXPORT_SCHEMA_VERSION = "carla-replay-export.v1"
_SCHEMA_RESOURCE = "schemas/carla_replay_export.v1.json"


@dataclass(frozen=True)
class Pose2D:
    """Planar pose or waypoint for neutral replay export."""

    x: float
    y: float
    theta: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Return the schema payload for this pose.

        Returns:
            JSON-ready pose dictionary.
        """

        payload: dict[str, float | None] = {"x": float(self.x), "y": float(self.y)}
        if self.theta is not None:
            payload["theta"] = float(self.theta)
        return payload


@dataclass(frozen=True)
class CertificateRef:
    """Reference to the scenario certification input used for export."""

    status: str
    source: str | None = None
    schema_version: str = "scenario_cert.v1"

    def to_dict(self) -> dict[str, Any]:
        """Return the schema payload for this certificate reference.

        Returns:
            JSON-ready certificate dictionary.
        """

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": self.status,
        }
        if self.source is not None:
            payload["source"] = self.source
        return payload


@dataclass(frozen=True)
class ScenarioReplayRef:
    """Scenario identity fields for a T0 neutral replay export."""

    scenario_id: str
    source_config: str
    map_id: str
    certificate: CertificateRef

    def to_dict(self) -> dict[str, Any]:
        """Return the schema payload for this scenario reference.

        Returns:
            JSON-ready scenario dictionary.
        """

        return {
            "id": self.scenario_id,
            "source_config": self.source_config,
            "map_id": self.map_id,
            "certificate": self.certificate.to_dict(),
        }


@dataclass(frozen=True)
class RobotReplaySpec:
    """Robot replay contract for a T0 neutral export."""

    start: Pose2D
    goal: Pose2D
    radius_m: float
    kinematics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the schema payload for this robot replay spec.

        Returns:
            JSON-ready robot dictionary.
        """

        return {
            "start": self.start.to_dict(),
            "goal": self.goal.to_dict(),
            "footprint": {"radius_m": float(self.radius_m)},
            "kinematics": dict(self.kinematics),
        }


@dataclass(frozen=True)
class PedestrianReplaySpec:
    """Scripted pedestrian replay contract for a T0 neutral export."""

    ped_id: str
    start: Pose2D
    route: list[Pose2D]
    timing: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the schema payload for this pedestrian replay spec.

        Returns:
            JSON-ready pedestrian dictionary.
        """

        payload: dict[str, Any] = {
            "id": self.ped_id,
            "start": self.start.to_dict(),
            "route": [waypoint.to_dict() for waypoint in self.route],
        }
        if self.timing is not None:
            payload["timing"] = dict(self.timing)
        return payload


@dataclass(frozen=True)
class SimulationSpec:
    """Simulation timing and termination contract for neutral replay export."""

    dt_s: float
    horizon_s: float
    termination: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the schema payload for this simulation spec.

        Returns:
            JSON-ready simulation dictionary.
        """

        return {
            "dt_s": float(self.dt_s),
            "horizon_s": float(self.horizon_s),
            "termination": list(self.termination or ["success", "collision", "timeout"]),
        }


def load_export_schema() -> dict[str, Any]:
    """Load the versioned T0 neutral export JSON schema.

    Returns:
        Parsed JSON schema dictionary.
    """

    schema_path = files("robot_sf_carla_bridge").joinpath(_SCHEMA_RESOURCE)
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_export_payload(payload: dict[str, Any]) -> None:
    """Validate one T0 neutral export payload.

    Raises:
        jsonschema.ValidationError: if ``payload`` does not satisfy the export schema.
    """

    jsonschema.validate(instance=payload, schema=load_export_schema())


def build_export_payload(
    *,
    scenario: ScenarioReplayRef,
    robot: RobotReplaySpec,
    pedestrians: list[PedestrianReplaySpec],
    static_geometry: dict[str, Any],
    simulation: SimulationSpec,
    trajectory_fields: list[str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build and validate a T0 neutral export payload from typed sections.

    Returns:
        Schema-valid export payload dictionary.
    """

    payload = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "mode": "neutral-export",
        "scenario": scenario.to_dict(),
        "robot": robot.to_dict(),
        "pedestrians": [pedestrian.to_dict() for pedestrian in pedestrians],
        "static_geometry": dict(static_geometry),
        "simulation": simulation.to_dict(),
        "metrics": {"trajectory_fields": list(trajectory_fields)},
        "provenance": dict(provenance),
    }
    validate_export_payload(payload)
    return payload


def write_export_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Validate and write a T0 export payload as stable UTF-8 JSON.

    Returns:
        The output path that was written.
    """

    validate_export_payload(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_export_payload(input_path: str | Path) -> dict[str, Any]:
    """Read and validate one T0 export payload from UTF-8 JSON.

    Streams the file via ``json.load`` instead of loading its full text first,
    so very large exports do not need to be held in memory twice.

    Returns:
        Schema-valid export payload dictionary.

    Raises:
        json.JSONDecodeError: if the file does not contain valid JSON.
        jsonschema.ValidationError: if the parsed payload does not satisfy the export schema.
    """

    with Path(input_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_export_payload(payload)
    return cast("dict[str, Any]", payload)
