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

    The runner is deliberately conservative: it validates the T0 payload, rejects static geometry
    that cannot yet be represented in CARLA, spawns only explicit robot/pedestrian actors, moves
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
        return {
            **base,
            "status": "oracle-replay",
            "mode": "oracle-replay",
            "reason": "Oracle transforms were replayed against a live CARLA world",
            "carla": {
                "dependency": "carla",
                "host": host,
                "port": port,
                "client_version": client.get_client_version(),
                "server_version": client.get_server_version(),
                "map": getattr(carla_map, "name", None),
            },
            "actors": {
                "robot": 1,
                "pedestrians": len(cast("list[dict[str, Any]]", payload["pedestrians"])),
            },
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
        for actor in reversed(actors):
            try:
                actor.destroy()
            except Exception:  # noqa: BLE001 - cleanup must tolerate CARLA-specific failures.
                pass


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
    if obstacles:
        return {
            "reason": "T0 payload static obstacle replay is not implemented",
            "static_obstacle_count": len(obstacles),
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

    robot_actor = _spawn_actor(
        world,
        _blueprint(blueprints, "vehicle.tesla.model3", "vehicle.*", role_name="robot_sf_robot"),
        robot_sf_pose_to_carla_transform(carla_module, cast("dict[str, Any]", robot["start"])),
        label="robot",
    )

    pedestrian_actors: list[_Actor] = []
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

    return {
        "_actors": [robot_actor, *pedestrian_actors],
        "robot_actor": robot_actor,
        "pedestrian_actors": pedestrian_actors,
    }


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
        matches = list(library.filter(fallback_pattern))
        if not matches:
            raise RuntimeError(f"CARLA blueprint not found: {preferred_id}") from None
        blueprint = matches[0]
    if hasattr(blueprint, "set_attribute"):
        blueprint.set_attribute("role_name", role_name)
    return blueprint


def _spawn_actor(world: Any, blueprint: Any, transform: Any, *, label: str) -> _Actor:
    spawn = getattr(world, "try_spawn_actor", None) or world.spawn_actor
    actor = spawn(blueprint, transform)
    if actor is None:
        raise RuntimeError(f"CARLA failed to spawn {label}")
    return cast("_Actor", actor)


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
        "static_geometry_replay": False,
        "full_metrics_parity": False,
        "sensor_perception": False,
        "long_running_benchmark": False,
        "note": "live CARLA actor replay attempt; no Robot-SF/CARLA parity claim",
    }
