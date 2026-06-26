"""T1 live CARLA oracle replay execution for one T0 export payload."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from robot_sf_carla_bridge.availability import require_carla
from robot_sf_carla_bridge.replay_smoke import (
    select_t0_export_payload,
    validate_t1_replay_catalog_payload,
)

T1_ORACLE_LIVE_REPLAY_SCHEMA_VERSION = "carla-t1-oracle-live-replay.v1"
CARLA_DEFAULT_HOST = "127.0.0.1"
CARLA_DEFAULT_RPC_PORT = 2000
_DEFAULT_MAX_STEPS = 200
_DEFAULT_ACTOR_Z = 0.1
_VEHICLE_ACTOR_Z = 0.6


class _Actor(Protocol):
    """Subset of the CARLA actor API used by the live replay runner."""

    def set_transform(self, transform: Any) -> None:
        """Move the actor to an oracle transform."""

    def destroy(self) -> None:
        """Remove the actor from the live CARLA world."""


def robot_sf_planar_vector_to_carla(vector: dict[str, Any]) -> dict[str, float]:
    """Mirror a Robot-SF planar vector into CARLA's left-handed x/y frame.

    Returns:
        Converted ``{"x", "y"}`` vector in CARLA coordinates.
    """

    return {"x": float(vector["x"]), "y": -float(vector["y"])}


def carla_planar_vector_to_robot_sf(vector: dict[str, Any]) -> dict[str, float]:
    """Mirror a CARLA planar vector back into Robot-SF's right-handed x/y frame.

    Returns:
        Converted ``{"x", "y"}`` vector in Robot-SF coordinates.
    """

    return {"x": float(vector["x"]), "y": -float(vector["y"])}


def robot_sf_yaw_to_carla_degrees(theta_rad: float) -> float:
    """Convert Robot-SF counter-clockwise yaw radians to CARLA yaw degrees.

    Returns:
        CARLA yaw angle in degrees.
    """

    return -math.degrees(float(theta_rad))


def carla_yaw_degrees_to_robot_sf(yaw_degrees: float) -> float:
    """Convert CARLA yaw degrees back to Robot-SF counter-clockwise yaw radians.

    Returns:
        Robot-SF yaw angle in radians.
    """

    return -math.radians(float(yaw_degrees))


def robot_sf_angular_velocity_to_carla_radps(omega_radps: float) -> float:
    """Convert Robot-SF yaw rate to CARLA yaw rate across handedness boundary.

    Returns:
        CARLA yaw rate in radians per second.
    """

    return -float(omega_radps)


def carla_angular_velocity_to_robot_sf_radps(omega_radps: float) -> float:
    """Convert CARLA yaw rate back to Robot-SF yaw rate across handedness boundary.

    Returns:
        Robot-SF yaw rate in radians per second.
    """

    return -float(omega_radps)


def robot_sf_pose_to_carla_transform(
    carla_module: Any,
    pose: dict[str, Any],
    *,
    z: float = _DEFAULT_ACTOR_Z,
) -> Any:
    """Convert a Robot-SF planar pose to a CARLA transform.

    Robot-SF exports use a right-handed metric x/y plane. CARLA/Unreal uses x-forward with the
    lateral axis mirrored for this first bridge boundary, so y and yaw are negated here instead of
    mutating T0 exports.

    Returns:
        CARLA ``Transform`` object built from the converted pose.
    """

    theta = float(pose.get("theta") or 0.0)
    location = robot_sf_planar_vector_to_carla(pose)
    return carla_module.Transform(
        carla_module.Location(x=location["x"], y=location["y"], z=float(z)),
        carla_module.Rotation(yaw=robot_sf_yaw_to_carla_degrees(theta)),
    )


def run_t1_oracle_live_replay_against_server(
    manifest_path: str | Path,
    *,
    scenario_id: str | None = None,
    carla_module: Any | None = None,
    host: str = CARLA_DEFAULT_HOST,
    port: int = CARLA_DEFAULT_RPC_PORT,
    timeout_s: float = 10.0,
    max_steps: int | None = _DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    """Replay one representable T0 payload against an already-running CARLA server.

    The runner is deliberately conservative: it validates the T0 payload, replays only static
    obstacle shapes that have a bounded CARLA proxy, spawns explicit robot/pedestrian actors, moves
    them by oracle transforms, and records that no metric parity claim is made.

    Returns:
        JSON-safe live replay status summary.
    """

    manifest = Path(manifest_path)
    record = select_t0_export_payload(manifest, scenario_id=scenario_id)
    payload = cast("dict[str, Any]", record["payload"])
    catalog_metadata = validate_t1_replay_catalog_payload(payload)
    base = _base_summary(manifest, record, payload, catalog_metadata, host=host, port=port)

    unsupported = _unsupported_payload_reasons(payload)
    if unsupported:
        return {
            **base,
            "status": "failed",
            "mode": "failed",
            "reason": unsupported["reason"],
            "unsupported": unsupported,
        }

    carla_api = carla_module if carla_module is not None else require_carla()
    actors: list[_Actor] = []
    started_at_utc = _utc_now_iso()
    try:
        client = carla_api.Client(host, port)
        client.set_timeout(timeout_s)
        world = client.get_world()
        carla_map = world.get_map()
        carla_map_name = getattr(carla_map, "name", None)
        carla_server_version = client.get_server_version()
        actor_summary = _spawn_replay_actors(carla_api, world, payload)
        actors.extend(actor_summary.pop("_actors"))
        trajectory = _replay_oracle_transforms(
            carla_api,
            world,
            actor_summary=actor_summary,
            payload=payload,
            max_steps=max_steps,
        )
        adaptations = cast("list[dict[str, Any]]", actor_summary.get("adaptations", []))
        coordinate_alignment = _coordinate_alignment_metadata(
            adaptations,
            payload=payload,
            carla_map_name=carla_map_name,
            carla_server_version=carla_server_version,
            started_at_utc=started_at_utc,
            ended_at_utc=_utc_now_iso(),
        )
        mode = "oracle-replay-adapted" if adaptations else "oracle-replay"
        reason = (
            "Oracle transforms were replayed against a live CARLA world with explicit spawn adaptation"
            if adaptations
            else "Oracle transforms were replayed against a live CARLA world"
        )
        return {
            **base,
            "status": "oracle-replay",
            "mode": mode,
            "reason": reason,
            "carla": {
                "dependency": "carla",
                "host": host,
                "port": port,
                "client_version": client.get_client_version(),
                "server_version": carla_server_version,
                "map": carla_map_name,
            },
            "actors": _actor_counts(payload),
            "adaptations": adaptations,
            "coordinate_alignment": coordinate_alignment,
            "replay_metadata": _replay_metadata(
                adaptations,
                coordinate_alignment=coordinate_alignment,
            ),
            "trajectory": trajectory,
            "metrics": trajectory["metrics"],
            "boundary": _live_replay_boundary(),
        }
    except Exception as exc:  # noqa: BLE001 - CARLA raises simulator-specific exceptions.
        return {
            **base,
            "status": "failed",
            "mode": "failed",
            "reason": str(exc),
            "boundary": _live_replay_boundary(),
        }
    finally:
        _destroy_actors(actors)


def _base_summary(
    manifest: Path,
    record: dict[str, Any],
    payload: dict[str, Any],
    catalog_metadata: dict[str, Any],
    *,
    host: str,
    port: int,
) -> dict[str, Any]:
    scenario = cast("dict[str, Any]", payload["scenario"])
    return {
        "schema_version": T1_ORACLE_LIVE_REPLAY_SCHEMA_VERSION,
        "stage": "live-replay",
        "manifest": manifest.as_posix(),
        "selected_payload": {
            "scenario_id": str(record["scenario_id"]),
            "path": Path(record["path"]).as_posix(),
            "payload_index": int(record["payload_index"]),
        },
        "catalog": catalog_metadata,
        "scenario": {
            "id": scenario["id"],
            "map_id": scenario["map_id"],
            "source_config": scenario["source_config"],
            "certificate_status": scenario["certificate"]["status"],
        },
        "connection": {"host": host, "port": port},
    }


def _unsupported_payload_reasons(payload: dict[str, Any]) -> dict[str, Any] | None:
    static_geometry = cast("dict[str, Any]", payload["static_geometry"])
    obstacles = cast("list[dict[str, Any]]", static_geometry.get("obstacles", []))
    unsupported = [
        obstacle for obstacle in obstacles if _axis_aligned_rectangle_bounds(obstacle) is None
    ]
    if unsupported:
        return {
            "reason": "T0 payload contains unsupported static obstacle geometry",
            "static_obstacle_count": len(obstacles),
            "unsupported_static_obstacle_count": len(unsupported),
            "supported_static_obstacle_count": len(obstacles) - len(unsupported),
            "boundary": _live_replay_boundary(),
        }
    return None


def _spawn_replay_actors(
    carla_module: Any,
    world: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    blueprints = world.get_blueprint_library()
    robot = cast("dict[str, Any]", payload["robot"])
    pedestrians = cast("list[dict[str, Any]]", payload["pedestrians"])
    static_geometry = cast("dict[str, Any]", payload["static_geometry"])
    obstacles = cast("list[dict[str, Any]]", static_geometry.get("obstacles", []))

    static_actors = _spawn_static_obstacle_actors(carla_module, world, blueprints, obstacles)

    robot_actor: _Actor | None = None
    pedestrian_actors: list[_Actor] = []
    try:
        robot_blueprint = _blueprint(
            blueprints,
            "vehicle.tesla.model3",
            "vehicle.*",
            role_name="robot_sf_robot",
        )
        robot_transform = robot_sf_pose_to_carla_transform(
            carla_module,
            cast("dict[str, Any]", robot["start"]),
            z=_VEHICLE_ACTOR_Z,
        )
        robot_actor, adaptations = _spawn_robot_actor(
            carla_module,
            world,
            robot_blueprint,
            robot_transform,
        )
        for index, pedestrian in enumerate(pedestrians):
            pedestrian_actors.append(
                _spawn_actor(
                    world,
                    _blueprint(
                        blueprints,
                        "walker.pedestrian.0001",
                        "walker.pedestrian.*",
                        role_name=f"robot_sf_pedestrian_{index}",
                    ),
                    robot_sf_pose_to_carla_transform(
                        carla_module,
                        cast("dict[str, Any]", pedestrian["start"]),
                    ),
                    label=f"pedestrian {pedestrian['id']}",
                )
            )
    except Exception:
        _destroy_actors([actor for actor in [robot_actor, *pedestrian_actors] if actor is not None])
        _destroy_actors(static_actors)
        raise

    if robot_actor is None:
        raise RuntimeError("CARLA failed to initialize robot actor")
    return {
        "_actors": [*static_actors, robot_actor, *pedestrian_actors],
        "static_obstacle_actors": static_actors,
        "robot_actor": robot_actor,
        "pedestrian_actors": pedestrian_actors,
        "adaptations": adaptations,
    }


def _actor_counts(payload: dict[str, Any]) -> dict[str, int]:
    static_geometry = cast("dict[str, Any]", payload["static_geometry"])
    obstacles = cast("list[dict[str, Any]]", static_geometry.get("obstacles", []))
    pedestrians = cast("list[dict[str, Any]]", payload["pedestrians"])
    return {
        "static_obstacles": len(obstacles),
        "robot": 1,
        "pedestrians": len(pedestrians),
    }


def _spawn_static_obstacle_actors(
    carla_module: Any,
    world: Any,
    blueprints: Any,
    obstacles: list[dict[str, Any]],
) -> list[_Actor]:
    static_actors: list[_Actor] = []
    try:
        for index, obstacle in enumerate(obstacles):
            bounds = _axis_aligned_rectangle_bounds(obstacle)
            if bounds is None:
                raise RuntimeError(f"Unsupported static obstacle geometry: {obstacle.get('id')}")
            static_actors.append(
                _spawn_actor(
                    world,
                    _blueprint(
                        blueprints,
                        "static.prop.box01",
                        "static.prop.*",
                        role_name=f"robot_sf_static_obstacle_{index}",
                    ),
                    _static_obstacle_transform(carla_module, bounds),
                    label=f"static obstacle {obstacle.get('id', index)}",
                )
            )
    except Exception:
        _destroy_actors(static_actors)
        raise
    return static_actors


def _static_obstacle_transform(carla_module: Any, bounds: dict[str, float]) -> Any:
    center_x = (bounds["min_x"] + bounds["max_x"]) / 2.0
    center_y = (bounds["min_y"] + bounds["max_y"]) / 2.0
    center = robot_sf_planar_vector_to_carla({"x": center_x, "y": center_y})
    return carla_module.Transform(
        carla_module.Location(x=center["x"], y=center["y"], z=_DEFAULT_ACTOR_Z),
        carla_module.Rotation(yaw=0.0),
    )


def _axis_aligned_rectangle_bounds(obstacle: dict[str, Any]) -> dict[str, float] | None:
    if obstacle.get("type") != "polygon":
        return None
    vertices = obstacle.get("vertices")
    if not isinstance(vertices, list) or len(vertices) != 4:
        return None
    try:
        points = [(float(vertex[0]), float(vertex[1])) for vertex in vertices]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    xs = {point[0] for point in points}
    ys = {point[1] for point in points}
    if len(xs) != 2 or len(ys) != 2:
        return None
    expected = {(x, y) for x in xs for y in ys}
    if set(points) != expected:
        return None
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if min_x == max_x or min_y == max_y:
        return None
    return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}


def _blueprint(
    library: Any,
    preferred_id: str,
    fallback_pattern: str,
    *,
    role_name: str,
) -> Any:
    try:
        blueprint = library.find(preferred_id)
    except (AttributeError, KeyError, RuntimeError, ValueError):
        blueprint = None
    if blueprint is None:
        matches = list(library.filter(fallback_pattern))
        if not matches:
            raise RuntimeError(f"CARLA blueprint not found: {preferred_id}") from None
        blueprint = matches[0]
    if hasattr(blueprint, "set_attribute"):
        blueprint.set_attribute("role_name", role_name)
    return blueprint


def _spawn_actor(world: Any, blueprint: Any, transform: Any, *, label: str) -> _Actor:
    actor = _try_spawn_actor(world, blueprint, transform)
    if actor is None:
        raise RuntimeError(_spawn_failure_message(world, blueprint, transform, label=label))
    return actor


def _spawn_robot_actor(
    carla_module: Any,
    world: Any,
    blueprint: Any,
    transform: Any,
) -> tuple[_Actor, list[dict[str, Any]]]:
    actor = _try_spawn_actor(world, blueprint, transform)
    if actor is not None:
        return actor, []

    exact_failure = _spawn_failure_message(world, blueprint, transform, label="robot")
    try:
        projection = _project_vehicle_spawn_transform(carla_module, world, transform)
    except RuntimeError as exc:
        raise RuntimeError(f"{exact_failure}; {exc}") from None

    projected_actor = _try_spawn_actor(world, blueprint, projection["transform"])
    if projected_actor is None:
        projected_summary = _transform_location_summary(
            projection["transform"],
            spawn_name=_spawn_api_name(world),
        )
        raise RuntimeError(
            f"{exact_failure}; CARLA map projection fallback via {projection['method']} "
            f"also failed for robot with blueprint {getattr(blueprint, 'id', '<unknown>')}"
            f"{projected_summary}"
        )

    return projected_actor, [
        {
            "actor": "robot",
            "type": "carla-map-spawn-projection",
            "reason": exact_failure,
            "method": projection["method"],
            "distance_m": projection["distance_m"],
            "original_transform": _transform_summary(transform),
            "projected_transform": _transform_summary(projection["transform"]),
            "parity_caveat": (
                "robot actor spawn was projected onto a CARLA map surface; "
                "this replay is not native metric-parity evidence"
            ),
        }
    ]


def _spawn_api_name(world: Any) -> str:
    return (
        "try_spawn_actor" if getattr(world, "try_spawn_actor", None) is not None else "spawn_actor"
    )


def _try_spawn_actor(world: Any, blueprint: Any, transform: Any) -> _Actor | None:
    spawn = getattr(world, "try_spawn_actor", None)
    if spawn is None:
        spawn = world.spawn_actor
    try:
        actor = spawn(blueprint, transform)
    except Exception:  # noqa: BLE001 - CARLA spawn adapters raise simulator/runtime-specific errors.
        return None
    return cast("_Actor | None", actor)


def _spawn_failure_message(world: Any, blueprint: Any, transform: Any, *, label: str) -> str:
    return (
        f"CARLA failed to spawn {label} with blueprint {getattr(blueprint, 'id', '<unknown>')}"
        f"{_transform_location_summary(transform, spawn_name=_spawn_api_name(world))}"
    )


def _transform_location_summary(transform: Any, *, spawn_name: str) -> str:
    location = getattr(transform, "location", None)
    rotation = getattr(transform, "rotation", None)
    if location is not None:
        location_summary = (
            f" via {spawn_name} at x={float(getattr(location, 'x', 0.0)):.3f}, "
            f"y={float(getattr(location, 'y', 0.0)):.3f}, "
            f"z={float(getattr(location, 'z', 0.0)):.3f}"
        )
    else:
        location_summary = f" via {spawn_name}"
    if rotation is not None:
        rotation_summary = f", yaw={float(getattr(rotation, 'yaw', 0.0)):.3f}"
    else:
        rotation_summary = ""
    return f"{location_summary}{rotation_summary}"


def _project_vehicle_spawn_transform(
    carla_module: Any,
    world: Any,
    transform: Any,
) -> dict[str, Any]:
    try:
        carla_map = world.get_map()
    except (AttributeError, RuntimeError):
        raise RuntimeError(
            "CARLA map projection fallback unavailable: world.get_map() API not available"
        ) from None

    original_location = getattr(transform, "location", None)
    if original_location is None:
        raise RuntimeError(
            "CARLA map projection fallback unavailable: requested transform has no location"
        )

    get_waypoint = getattr(carla_map, "get_waypoint", None)
    if get_waypoint is None:
        raise RuntimeError(
            "CARLA map projection fallback unavailable: map.get_waypoint API not available"
        )
    kwargs: dict[str, Any] = {"project_to_road": True}
    lane_type = getattr(getattr(carla_module, "LaneType", None), "Driving", None)
    if lane_type is not None:
        kwargs["lane_type"] = lane_type
    try:
        waypoint = get_waypoint(original_location, **kwargs)
    except TypeError:
        waypoint = get_waypoint(original_location, project_to_road=True)
    except RuntimeError as exc:
        raise RuntimeError(f"CARLA map projection fallback failed: {exc}") from None

    waypoint_transform = getattr(waypoint, "transform", None) if waypoint is not None else None
    if waypoint_transform is None:
        raise RuntimeError(
            "CARLA map projection fallback unavailable: map.get_waypoint returned no waypoint transform"
        )

    spawn_transform = _vehicle_spawn_transform_from_map_transform(
        carla_module,
        waypoint_transform,
        z_offset=_VEHICLE_ACTOR_Z,
    )
    return {
        "method": "world.get_map().get_waypoint(project_to_road=True)",
        "transform": spawn_transform,
        "distance_m": _location_distance_m(original_location, spawn_transform.location),
    }


def _vehicle_spawn_transform_from_map_transform(
    carla_module: Any,
    transform: Any,
    *,
    z_offset: float,
) -> Any:
    location = transform.location
    rotation = transform.rotation
    return carla_module.Transform(
        carla_module.Location(
            x=float(getattr(location, "x", 0.0)),
            y=float(getattr(location, "y", 0.0)),
            z=float(getattr(location, "z", 0.0)) + z_offset,
        ),
        carla_module.Rotation(yaw=float(getattr(rotation, "yaw", 0.0))),
    )


def _location_distance_m(a: Any, b: Any) -> float:
    dx = float(getattr(a, "x", 0.0)) - float(getattr(b, "x", 0.0))
    dy = float(getattr(a, "y", 0.0)) - float(getattr(b, "y", 0.0))
    dz = float(getattr(a, "z", 0.0)) - float(getattr(b, "z", 0.0))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _transform_summary(transform: Any) -> dict[str, float]:
    location = getattr(transform, "location", None)
    rotation = getattr(transform, "rotation", None)
    if location is None:
        return {}
    return {
        "x": float(getattr(location, "x", 0.0)),
        "y": float(getattr(location, "y", 0.0)),
        "z": float(getattr(location, "z", 0.0)),
        "yaw": float(getattr(rotation, "yaw", 0.0)) if rotation is not None else 0.0,
    }


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp for replay provenance."""

    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _coordinate_alignment_metadata(
    adaptations: list[dict[str, Any]],
    *,
    payload: dict[str, Any],
    carla_map_name: str | None,
    carla_server_version: str,
    started_at_utc: str,
    ended_at_utc: str,
) -> dict[str, Any]:
    """Return #1444-style coordinate-alignment metadata for one live replay."""

    robot_spawn = next(
        (adaptation for adaptation in adaptations if adaptation.get("actor") == "robot"),
        None,
    )
    provenance = cast("dict[str, Any]", payload.get("provenance", {}))
    scenario = cast("dict[str, Any]", payload["scenario"])
    if robot_spawn is None:
        return {
            "replay_mode": "native",
            "projection_meters": 0.0,
            "projection_rationale": "none",
            "carla_map_name": carla_map_name,
            "carla_server_version": carla_server_version,
            "robot_sf_commit": provenance.get("robot_sf_commit"),
            "scenario_cert_id": scenario["id"],
            "scenario_certificate_source": cast("dict[str, Any]", scenario["certificate"]).get(
                "source"
            ),
            "start": started_at_utc,
            "end": ended_at_utc,
            "bridge_version": T1_ORACLE_LIVE_REPLAY_SCHEMA_VERSION,
            "eligible_for_metric_parity": True,
        }

    return {
        "replay_mode": "adapted",
        "projection_meters": float(robot_spawn.get("distance_m", 0.0)),
        "projection_rationale": str(robot_spawn.get("method") or "carla map spawn projection"),
        "carla_map_name": carla_map_name,
        "carla_server_version": carla_server_version,
        "robot_sf_commit": provenance.get("robot_sf_commit"),
        "scenario_cert_id": scenario["id"],
        "scenario_certificate_source": cast("dict[str, Any]", scenario["certificate"]).get(
            "source"
        ),
        "start": started_at_utc,
        "end": ended_at_utc,
        "bridge_version": T1_ORACLE_LIVE_REPLAY_SCHEMA_VERSION,
        "eligible_for_metric_parity": False,
        "exclusion_reason": (
            "robot actor spawn used CARLA map projection; adapted replay is not native/aligned "
            "metric-parity evidence"
        ),
    }


def _replay_metadata(
    adaptations: list[dict[str, Any]],
    *,
    coordinate_alignment: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "coordinate_alignment": coordinate_alignment,
        "robot_spawn": {"strategy": "exact", "adapted": False},
    }
    robot_spawn = next(
        (adaptation for adaptation in adaptations if adaptation.get("actor") == "robot"),
        None,
    )
    if robot_spawn is not None:
        metadata["robot_spawn"] = {
            "strategy": "carla-map-projection",
            "adapted": True,
            "projection_source": robot_spawn["method"],
            "requested_transform": robot_spawn["original_transform"],
            "spawn_transform": robot_spawn["projected_transform"],
            "parity_caveat": robot_spawn["parity_caveat"],
        }
    return metadata


def _destroy_actors(actors: list[_Actor]) -> None:
    """Best-effort actor cleanup for partially initialized CARLA replay state."""
    for actor in reversed(actors):
        try:
            actor.destroy()
        except Exception:  # noqa: BLE001 - cleanup must tolerate CARLA-specific failures.
            pass


def _replay_oracle_transforms(
    carla_module: Any,
    world: Any,
    *,
    actor_summary: dict[str, Any],
    payload: dict[str, Any],
    max_steps: int | None,
) -> dict[str, Any]:
    simulation = cast("dict[str, Any]", payload["simulation"])
    dt_s = float(simulation["dt_s"])
    horizon_s = float(simulation["horizon_s"])
    requested_steps = max(1, math.ceil(horizon_s / dt_s) + 1)
    step_count = requested_steps if max_steps is None else min(requested_steps, max(1, max_steps))

    robot = cast("dict[str, Any]", payload["robot"])
    pedestrians = cast("list[dict[str, Any]]", payload["pedestrians"])
    static_geometry = cast("dict[str, Any]", payload["static_geometry"])
    robot_actor = cast("_Actor", actor_summary["robot_actor"])
    pedestrian_actors = cast("list[_Actor]", actor_summary["pedestrian_actors"])
    robot_samples: list[dict[str, float]] = []
    pedestrian_samples: list[list[dict[str, float]]] = []

    for step in range(step_count):
        alpha = 0.0 if step_count == 1 else step / (step_count - 1)
        robot_pose = _interpolate_pose(
            cast("dict[str, Any]", robot["start"]),
            cast("dict[str, Any]", robot["goal"]),
            alpha,
        )
        robot_samples.append(robot_pose)
        step_pedestrians: list[dict[str, float]] = []
        robot_actor.set_transform(
            robot_sf_pose_to_carla_transform(
                carla_module,
                robot_pose,
                z=_VEHICLE_ACTOR_Z,
            )
        )
        for pedestrian_actor, pedestrian in zip(pedestrian_actors, pedestrians, strict=True):
            pedestrian_pose = _path_pose(
                cast("dict[str, Any]", pedestrian["start"]),
                cast("list[dict[str, Any]]", pedestrian["route"]),
                alpha,
            )
            step_pedestrians.append(pedestrian_pose)
            pedestrian_actor.set_transform(
                robot_sf_pose_to_carla_transform(
                    carla_module,
                    pedestrian_pose,
                )
            )
        pedestrian_samples.append(step_pedestrians)
        _tick_world(world)

    metrics = _oracle_replay_metrics(
        robot=robot,
        pedestrians=pedestrians,
        static_geometry=static_geometry,
        robot_samples=robot_samples,
        pedestrian_samples=pedestrian_samples,
        truncated=step_count < requested_steps,
    )
    return {
        "steps": step_count,
        "requested_steps": requested_steps,
        "dt_s": dt_s,
        "horizon_s": horizon_s,
        "truncated_by_max_steps": step_count < requested_steps,
        "metrics": metrics,
        "robot": {
            "start": robot["start"],
            "goal": robot["goal"],
        },
        "pedestrian_ids": [str(pedestrian["id"]) for pedestrian in pedestrians],
    }


def _interpolate_pose(
    start: dict[str, Any],
    end: dict[str, Any],
    alpha: float,
) -> dict[str, float]:
    x0 = float(start["x"])
    y0 = float(start["y"])
    x1 = float(end["x"])
    y1 = float(end["y"])
    theta = start.get("theta")
    if theta is None:
        theta = math.atan2(y1 - y0, x1 - x0)
    return {
        "x": x0 + (x1 - x0) * alpha,
        "y": y0 + (y1 - y0) * alpha,
        "theta": float(theta),
    }


def _path_pose(
    start: dict[str, Any], route: list[dict[str, Any]], alpha: float
) -> dict[str, float]:
    points = [start, *route]
    if len(points) == 1:
        return {
            "x": float(start["x"]),
            "y": float(start["y"]),
            "theta": float(start.get("theta") or 0),
        }
    scaled = min(alpha * (len(points) - 1), len(points) - 1)
    segment = min(math.floor(scaled), len(points) - 2)
    local_alpha = scaled - segment
    return _interpolate_pose(points[segment], points[segment + 1], local_alpha)


def _tick_world(world: Any) -> None:
    if hasattr(world, "tick"):
        world.tick()
    elif hasattr(world, "wait_for_tick"):
        world.wait_for_tick()


def _oracle_replay_metrics(
    *,
    robot: dict[str, Any],
    pedestrians: list[dict[str, Any]],
    static_geometry: dict[str, Any],
    robot_samples: list[dict[str, float]],
    pedestrian_samples: list[list[dict[str, float]]],
    truncated: bool,
) -> dict[str, bool | float]:
    """Compute conservative metrics from the scripted oracle replay trajectory.

    Returns:
        dict[str, bool | float]: Metric fields that the CARLA parity adapter can compare.
    """
    robot_radius = float(cast("dict[str, Any]", robot.get("footprint", {})).get("radius_m", 0.0))
    pedestrian_radii = [
        float(cast("dict[str, Any]", ped.get("footprint", {})).get("radius_m", 0.0))
        for ped in pedestrians
    ]
    obstacle_bounds: list[dict[str, float]] = []
    for obstacle in cast("list[dict[str, Any]]", static_geometry.get("obstacles", [])):
        bounds = _axis_aligned_rectangle_bounds(obstacle)
        if bounds is not None:
            obstacle_bounds.append(bounds)

    min_clearance = math.inf
    collision = False
    for robot_pose, step_pedestrians in zip(robot_samples, pedestrian_samples, strict=True):
        for pedestrian_pose, pedestrian_radius in zip(
            step_pedestrians, pedestrian_radii, strict=True
        ):
            clearance = (
                _planar_distance(robot_pose, pedestrian_pose) - robot_radius - pedestrian_radius
            )
            min_clearance = min(min_clearance, clearance)
            collision = collision or clearance <= 0.0
        for bounds in obstacle_bounds:
            clearance = _point_rectangle_clearance(robot_pose, bounds) - robot_radius
            min_clearance = min(min_clearance, clearance)
            collision = collision or clearance <= 0.0

    final_goal_distance = _planar_distance(robot_samples[-1], cast("dict[str, Any]", robot["goal"]))
    metrics: dict[str, bool | float] = {
        "success": (not truncated) and final_goal_distance <= max(robot_radius, 1e-6),
        "collision": collision,
        "intervention_rate": 0.0,
    }
    if math.isfinite(min_clearance):
        metrics["min_distance_m"] = min_clearance
    return metrics


def _planar_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    dx = float(a["x"]) - float(b["x"])
    dy = float(a["y"]) - float(b["y"])
    return math.hypot(dx, dy)


def _point_rectangle_clearance(point: dict[str, Any], bounds: dict[str, float]) -> float:
    x = float(point["x"])
    y = float(point["y"])
    dx = max(bounds["min_x"] - x, 0.0, x - bounds["max_x"])
    dy = max(bounds["min_y"] - y, 0.0, y - bounds["max_y"])
    if dx > 0.0 or dy > 0.0:
        return math.hypot(dx, dy)
    return -min(
        x - bounds["min_x"],
        bounds["max_x"] - x,
        y - bounds["min_y"],
        bounds["max_y"] - y,
    )


def _live_replay_boundary() -> dict[str, bool | str]:
    return {
        "oracle_transform_replay": True,
        "scripted_pedestrian_playback": True,
        "static_geometry_replay": True,
        "full_metrics_parity": False,
        "sensor_perception": False,
        "long_running_benchmark": False,
        "note": (
            "live CARLA actor replay attempt with axis-aligned static obstacle proxies; "
            "no Robot-SF/CARLA parity claim"
        ),
    }
