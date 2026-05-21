"""T1 live CARLA oracle replay execution for one T0 export payload."""

from __future__ import annotations

import math
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
    return carla_module.Transform(
        carla_module.Location(x=float(pose["x"]), y=-float(pose["y"]), z=float(z)),
        carla_module.Rotation(yaw=-math.degrees(theta)),
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
    try:
        client = carla_api.Client(host, port)
        client.set_timeout(timeout_s)
        world = client.get_world()
        carla_map = world.get_map()
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
                "server_version": client.get_server_version(),
                "map": getattr(carla_map, "name", None),
            },
            "actors": _actor_counts(payload),
            "adaptations": adaptations,
            "replay_metadata": _replay_metadata(adaptations),
            "trajectory": trajectory,
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
    return carla_module.Transform(
        carla_module.Location(x=center_x, y=-center_y, z=_DEFAULT_ACTOR_Z),
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


def _replay_metadata(adaptations: list[dict[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"robot_spawn": {"strategy": "exact", "adapted": False}}
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
    robot_actor = cast("_Actor", actor_summary["robot_actor"])
    pedestrian_actors = cast("list[_Actor]", actor_summary["pedestrian_actors"])

    for step in range(step_count):
        alpha = 0.0 if step_count == 1 else step / (step_count - 1)
        robot_actor.set_transform(
            robot_sf_pose_to_carla_transform(
                carla_module,
                _interpolate_pose(
                    cast("dict[str, Any]", robot["start"]),
                    cast("dict[str, Any]", robot["goal"]),
                    alpha,
                ),
                z=_VEHICLE_ACTOR_Z,
            )
        )
        for pedestrian_actor, pedestrian in zip(pedestrian_actors, pedestrians, strict=True):
            pedestrian_actor.set_transform(
                robot_sf_pose_to_carla_transform(
                    carla_module,
                    _path_pose(
                        cast("dict[str, Any]", pedestrian["start"]),
                        cast("list[dict[str, Any]]", pedestrian["route"]),
                        alpha,
                    ),
                )
            )
        _tick_world(world)

    return {
        "steps": step_count,
        "requested_steps": requested_steps,
        "dt_s": dt_s,
        "horizon_s": horizon_s,
        "truncated_by_max_steps": step_count < requested_steps,
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
