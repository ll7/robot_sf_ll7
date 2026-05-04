"""T0 neutral replay export helpers for future CARLA oracle replay."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jsonschema

if TYPE_CHECKING:
    from collections.abc import Mapping

EXPORT_SCHEMA_VERSION = "carla-replay-export.v1"
_SCHEMA_RESOURCE = "schemas/carla_replay_export.v1.json"
EXPORT_MANIFEST_SCHEMA_VERSION = "carla-replay-export-manifest.v1"
_EXPORTABLE_CERT_STATUSES = {"passed", "valid", "hard_but_solvable", "knife_edge"}
_DEFAULT_TRAJECTORY_FIELDS = [
    "success",
    "collision",
    "min_distance",
    "ttc",
    "comfort",
    "jerk",
    "curvature",
    "intervention_rate",
]


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


def build_export_payload_from_map_definition(  # noqa: PLR0913
    *,
    map_def: Any,
    certificate: Mapping[str, Any],
    scenario_id: str,
    source_config: str | Path,
    map_id: str,
    robot_radius_m: float,
    robot_kinematics: Mapping[str, Any],
    dt_s: float,
    horizon_s: float,
    provenance: Mapping[str, Any],
    route_index: int = 0,
    trajectory_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Build a T0 neutral export payload from a certified Robot-SF map definition.

    The adapter serializes Robot-SF world coordinates and scripted geometry only. CARLA world
    coordinate conversion and runtime replay remain separate T1 work.

    Returns:
        Schema-valid export payload dictionary.

    Raises:
        ValueError: if the certificate is excluded or the selected route is unavailable.
    """

    certificate_ref = _certificate_ref_from_payload(certificate)
    route = _select_robot_route(map_def, route_index=route_index)
    route_points = _route_waypoints(route)
    if len(route_points) < 2:
        raise ValueError("Selected robot route must contain at least a start and goal waypoint")

    return build_export_payload(
        scenario=ScenarioReplayRef(
            scenario_id=scenario_id,
            source_config=Path(source_config).as_posix(),
            map_id=map_id,
            certificate=certificate_ref,
        ),
        robot=RobotReplaySpec(
            start=Pose2D(*route_points[0]),
            goal=Pose2D(*route_points[-1]),
            radius_m=robot_radius_m,
            kinematics=dict(robot_kinematics),
        ),
        pedestrians=_pedestrians_from_map_definition(map_def),
        static_geometry=_static_geometry_from_map_definition(map_def),
        simulation=SimulationSpec(dt_s=dt_s, horizon_s=horizon_s),
        trajectory_fields=list(trajectory_fields or _DEFAULT_TRAJECTORY_FIELDS),
        provenance=dict(provenance),
    )


def build_export_payload_from_scenario_entry(
    scenario: Mapping[str, Any],
    *,
    scenario_path: str | Path,
    provenance: Mapping[str, Any],
    route_index: int = 0,
    trajectory_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Build a T0 neutral export payload from one scenario-loader entry.

    The helper uses the existing Robot-SF scenario loader and ``scenario_cert.v1`` certification
    path, then delegates geometry serialization to
    :func:`build_export_payload_from_map_definition`.

    Returns:
        Schema-valid export payload dictionary.
    """

    path = Path(scenario_path)
    config = _build_robot_config_for_scenario_entry(scenario, path)
    map_id, map_def = _single_loaded_map(config)
    certificate = _certificate_payload_for_scenario_entry(scenario, path)
    return build_export_payload_from_map_definition(
        map_def=map_def,
        certificate=certificate,
        scenario_id=_scenario_identifier(scenario),
        source_config=path,
        map_id=str(getattr(config, "map_id", None) or map_id),
        robot_radius_m=float(config.robot_config.radius),
        robot_kinematics=_robot_kinematics_payload(config.robot_config),
        dt_s=float(config.sim_config.time_per_step_in_secs),
        horizon_s=float(config.sim_config.sim_time_in_secs),
        provenance=provenance,
        route_index=route_index,
        trajectory_fields=trajectory_fields,
    )


def build_export_payloads_from_scenario_file(
    scenario_path: str | Path,
    *,
    provenance: Mapping[str, Any],
    route_index: int = 0,
    trajectory_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build T0 neutral export payload records for every scenario in a manifest file.

    Returns:
        JSON-safe records with ``scenario_id`` and validated export ``payload`` fields.
    """

    path = Path(scenario_path)
    records: list[dict[str, Any]] = []
    for scenario in _load_scenario_entries(path):
        payload = build_export_payload_from_scenario_entry(
            scenario,
            scenario_path=path,
            provenance=provenance,
            route_index=route_index,
            trajectory_fields=trajectory_fields,
        )
        records.append({"scenario_id": _scenario_identifier(scenario), "payload": payload})
    return records


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


def write_export_records(
    records: list[Mapping[str, Any]], output_dir: str | Path
) -> dict[str, Any]:
    """Write ordered T0 export records to an output directory.

    Returns:
        JSON-safe manifest describing the files written.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
        "exports": [],
    }
    used_names: set[str] = set()
    for index, record in enumerate(records):
        scenario_id = str(record.get("scenario_id") or "").strip()
        if not scenario_id:
            raise ValueError(f"export record {index} missing scenario_id")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"export record {index} payload must be a mapping")
        file_name = _unique_export_file_name(scenario_id, used_names)
        write_export_payload(payload, root / file_name)
        manifest["exports"].append({"scenario_id": scenario_id, "path": file_name})
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_export_payload(input_path: str | Path) -> dict[str, Any]:
    """Read and validate one T0 export payload from UTF-8 JSON.

    Returns:
        Schema-valid export payload dictionary.

    Raises:
        json.JSONDecodeError: if the file does not contain valid JSON.
        jsonschema.ValidationError: if the parsed payload does not satisfy the export schema.
    """

    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    validate_export_payload(payload)
    return cast("dict[str, Any]", payload)


def _build_robot_config_for_scenario_entry(scenario: Mapping[str, Any], scenario_path: Path) -> Any:
    """Build Robot-SF runtime config for a scenario entry via the existing loader.

    Returns:
        Robot-SF simulation config for the supplied scenario entry.
    """

    from robot_sf.training.scenario_loader import build_robot_config_from_scenario  # noqa: PLC0415

    return build_robot_config_from_scenario(scenario, scenario_path=scenario_path)


def _load_scenario_entries(scenario_path: Path) -> list[Mapping[str, Any]]:
    """Load scenario entries via the existing scenario loader.

    Returns:
        Scenario-loader entries in manifest order.
    """

    from robot_sf.training.scenario_loader import load_scenarios  # noqa: PLC0415

    return list(load_scenarios(scenario_path))


def _unique_export_file_name(scenario_id: str, used_names: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario_id).strip("._")
    if not stem:
        stem = "scenario"
    candidate = f"{stem}.json"
    suffix = 2
    while candidate in used_names:
        candidate = f"{stem}_{suffix}.json"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _certificate_payload_for_scenario_entry(
    scenario: Mapping[str, Any],
    scenario_path: Path,
) -> dict[str, Any]:
    """Return JSON-safe ``scenario_cert.v1`` payload for one scenario entry."""

    from robot_sf.scenario_certification import (  # noqa: PLC0415
        certificate_to_dict,
        certify_scenario,
    )

    return certificate_to_dict(certify_scenario(scenario, scenario_path=scenario_path))


def _single_loaded_map(config: Any) -> tuple[str, Any]:
    """Return the selected loaded map from a Robot-SF simulation config."""

    map_defs = getattr(getattr(config, "map_pool", None), "map_defs", None)
    if not map_defs:
        raise ValueError("Scenario entry did not load a map definition")
    selected = getattr(config, "map_id", None)
    if selected in map_defs:
        return str(selected), map_defs[selected]
    map_id, map_def = next(iter(map_defs.items()))
    return str(map_id), map_def


def _scenario_identifier(scenario: Mapping[str, Any]) -> str:
    """Return the scenario id field used by scenario-loader manifests."""

    for key in ("name", "scenario_id", "id"):
        value = scenario.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("Scenario entry must define name, scenario_id, or id")


def _robot_kinematics_payload(robot_config: Any) -> dict[str, Any]:
    """Return neutral kinematics metadata for supported Robot-SF robot settings."""

    if hasattr(robot_config, "max_linear_speed"):
        return {
            "model": "differential_drive",
            "max_speed_mps": float(robot_config.max_linear_speed),
            "max_angular_speed_radps": float(robot_config.max_angular_speed),
        }
    if hasattr(robot_config, "max_velocity"):
        return {
            "model": "bicycle_drive",
            "max_speed_mps": float(robot_config.max_velocity),
            "max_steer_rad": float(robot_config.max_steer),
        }
    if hasattr(robot_config, "max_speed"):
        return {
            "model": "holonomic",
            "max_speed_mps": float(robot_config.max_speed),
            "max_angular_speed_radps": float(robot_config.max_angular_speed),
            "command_mode": str(robot_config.command_mode),
        }
    return {"model": type(robot_config).__name__}


def _certificate_ref_from_payload(certificate: Mapping[str, Any]) -> CertificateRef:
    """Return an exportable certificate reference or fail closed for excluded certificates."""

    eligibility = str(certificate.get("benchmark_eligibility") or "").strip().lower()
    if eligibility == "excluded":
        raise ValueError("Cannot export CARLA T0 payload for excluded scenario certificate")

    status = str(certificate.get("status") or certificate.get("classification") or "").strip()
    if status not in _EXPORTABLE_CERT_STATUSES:
        raise ValueError(f"Cannot export CARLA T0 payload for certificate status '{status}'")
    return CertificateRef(
        status=status,
        source=str(certificate["source"]) if certificate.get("source") is not None else None,
    )


def _select_robot_route(map_def: Any, *, route_index: int) -> Any:
    routes = list(getattr(map_def, "robot_routes", []) or [])
    if route_index < 0 or route_index >= len(routes):
        raise ValueError(f"robot route index {route_index} is not available")
    return routes[route_index]


def _route_waypoints(route: Any) -> list[tuple[float, float]]:
    return [_point_tuple(point) for point in getattr(route, "waypoints", [])]


def _point_tuple(point: Any) -> tuple[float, float]:
    if not isinstance(point, list | tuple) or len(point) < 2:
        raise ValueError(f"Expected 2D point, got {point!r}")
    return (float(point[0]), float(point[1]))


def _pedestrians_from_map_definition(map_def: Any) -> list[PedestrianReplaySpec]:
    pedestrians: list[PedestrianReplaySpec] = []
    for ped in getattr(map_def, "single_pedestrians", []) or []:
        timing: dict[str, Any] = {}
        speed = getattr(ped, "speed_m_s", None)
        if speed is not None:
            timing["speed_m_s"] = float(speed)
        waits = getattr(ped, "wait_at", None)
        if waits:
            timing["wait_at"] = [
                {
                    "waypoint_index": int(rule.waypoint_index),
                    "wait_s": float(rule.wait_s),
                    **({"note": rule.note} if rule.note else {}),
                }
                for rule in waits
            ]
        pedestrians.append(
            PedestrianReplaySpec(
                ped_id=str(ped.id),
                start=Pose2D(*_point_tuple(ped.start)),
                route=[Pose2D(*point) for point in _pedestrian_route_points(ped)],
                timing=timing or None,
            )
        )
    return pedestrians


def _pedestrian_route_points(ped: Any) -> list[tuple[float, float]]:
    trajectory = getattr(ped, "trajectory", None)
    if trajectory:
        return [_point_tuple(point) for point in trajectory]
    goal = getattr(ped, "goal", None)
    if goal is not None:
        return [_point_tuple(goal)]
    return [_point_tuple(ped.start)]


def _static_geometry_from_map_definition(map_def: Any) -> dict[str, Any]:
    return {
        "obstacles": _obstacles_from_map_definition(map_def),
        "map_bounds": {
            "width_m": float(map_def.width),
            "height_m": float(map_def.height),
        },
    }


def _obstacles_from_map_definition(map_def: Any) -> list[dict[str, Any]]:
    obstacles: list[dict[str, Any]] = []
    for index, obstacle in enumerate(getattr(map_def, "obstacles", []) or []):
        polygons = getattr(obstacle, "iter_polygons", lambda: [])()
        if polygons:
            for poly_index, polygon in enumerate(polygons):
                coords = [list(_point_tuple(point)) for point in list(polygon.exterior.coords)[:-1]]
                obstacles.append(
                    {
                        "id": f"obstacle_{index}_{poly_index}",
                        "type": "polygon",
                        "vertices": coords,
                    }
                )
            continue
        vertices = [list(_point_tuple(point)) for point in getattr(obstacle, "vertices", [])]
        obstacles.append({"id": f"obstacle_{index}", "type": "polygon", "vertices": vertices})
    return obstacles
