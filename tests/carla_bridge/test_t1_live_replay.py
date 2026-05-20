"""CARLA T1 live oracle replay tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tests.carla_bridge.test_t1_replay_smoke import _minimal_t0_payload, _write_manifest

if TYPE_CHECKING:
    from pathlib import Path


class _FakeLocation:
    def __init__(self, *, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _FakeRotation:
    def __init__(self, *, yaw: float) -> None:
        self.yaw = yaw


class _FakeTransform:
    def __init__(self, location: _FakeLocation, rotation: _FakeRotation) -> None:
        self.location = location
        self.rotation = rotation


class _FakeBlueprint:
    def __init__(self, blueprint_id: str) -> None:
        self.id = blueprint_id
        self.attributes: dict[str, str] = {}

    def set_attribute(self, name: str, value: str) -> None:
        self.attributes[name] = value


class _FakeBlueprintLibrary:
    def find(self, blueprint_id: str) -> _FakeBlueprint:
        return _FakeBlueprint(blueprint_id)

    def filter(self, pattern: str) -> list[_FakeBlueprint]:
        return [_FakeBlueprint(pattern.replace("*", "0001"))]


class _FakeActor:
    def __init__(self, blueprint: _FakeBlueprint, transform: _FakeTransform) -> None:
        self.blueprint = blueprint
        self.transforms = [transform]
        self.destroyed = False

    def set_transform(self, transform: _FakeTransform) -> None:
        self.transforms.append(transform)

    def destroy(self) -> None:
        self.destroyed = True


class _FakeWorld:
    def __init__(self) -> None:
        self.actors: list[_FakeActor] = []
        self.ticks = 0

    def get_map(self):
        return SimpleNamespace(name="Town01")

    def get_blueprint_library(self) -> _FakeBlueprintLibrary:
        return _FakeBlueprintLibrary()

    def try_spawn_actor(self, blueprint: _FakeBlueprint, transform: _FakeTransform) -> _FakeActor:
        actor = _FakeActor(blueprint, transform)
        self.actors.append(actor)
        return actor

    def tick(self) -> int:
        self.ticks += 1
        return self.ticks


class _PedestrianSpawnFailureWorld(_FakeWorld):
    def try_spawn_actor(
        self, blueprint: _FakeBlueprint, transform: _FakeTransform
    ) -> _FakeActor | None:
        if blueprint.id.startswith("walker."):
            return None
        return super().try_spawn_actor(blueprint, transform)


def _fake_carla_module(world: _FakeWorld):
    class FakeClient:
        def __init__(self, host: str, port: int) -> None:
            self.host = host
            self.port = port
            self.timeout = None

        def set_timeout(self, timeout_s: float) -> None:
            self.timeout = timeout_s

        def get_world(self) -> _FakeWorld:
            return world

        def get_client_version(self) -> str:
            return "0.9.16"

        def get_server_version(self) -> str:
            return "0.9.16"

    return SimpleNamespace(
        Client=FakeClient,
        Location=_FakeLocation,
        Rotation=_FakeRotation,
        Transform=_FakeTransform,
    )


def test_live_replay_spawns_static_geometry_and_cleans_up(tmp_path: Path) -> None:
    """Rectangular T0 static obstacles should be represented by CARLA proxy actors."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

    world = _FakeWorld()
    payload = _minimal_t0_payload()
    payload["static_geometry"]["obstacles"] = [
        {"id": "wall", "type": "polygon", "vertices": [[0, 0], [2, 0], [2, 1], [0, 1]]}
    ]
    manifest_path = _write_manifest(
        tmp_path, [{"scenario_id": "unit_crossing", "payload": payload}]
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
        max_steps=2,
    )

    assert summary["status"] == "oracle-replay"
    assert summary["actors"] == {"static_obstacles": 1, "robot": 1, "pedestrians": 1}
    assert summary["boundary"]["static_geometry_replay"] is True
    assert len(world.actors) == 3
    static_actor = world.actors[0]
    assert static_actor.blueprint.id == "static.prop.box01"
    assert static_actor.blueprint.attributes["role_name"] == "robot_sf_static_obstacle_0"
    assert static_actor.transforms[0].location.x == pytest.approx(1.0)
    assert static_actor.transforms[0].location.y == pytest.approx(-0.5)
    assert all(actor.destroyed for actor in world.actors)


def test_live_replay_fails_closed_for_unsupported_static_geometry(tmp_path: Path) -> None:
    """Unsupported T0 static obstacle shapes should fail closed before CARLA replay."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

    world = _FakeWorld()
    payload = _minimal_t0_payload()
    payload["static_geometry"]["obstacles"] = [
        {"id": "triangle", "type": "polygon", "vertices": [[0, 0], [1, 0], [0, 1]]}
    ]
    manifest_path = _write_manifest(
        tmp_path, [{"scenario_id": "unit_crossing", "payload": payload}]
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
    )

    assert summary["status"] == "failed"
    assert summary["mode"] == "failed"
    assert summary["stage"] == "live-replay"
    assert summary["reason"] == "T0 payload contains unsupported static obstacle geometry"
    assert summary["unsupported"]["static_obstacle_count"] == 1
    assert summary["unsupported"]["unsupported_static_obstacle_count"] == 1
    assert world.actors == []


def test_live_replay_fails_closed_for_malformed_static_geometry(tmp_path: Path) -> None:
    """Malformed T0 static obstacle vertices should fail closed before CARLA replay."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

    world = _FakeWorld()
    payload = _minimal_t0_payload()
    payload["static_geometry"]["obstacles"] = [
        {"id": "dict_vertices", "type": "polygon", "vertices": [{"x": 0, "y": 0}] * 4}
    ]
    manifest_path = _write_manifest(
        tmp_path, [{"scenario_id": "unit_crossing", "payload": payload}]
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
    )

    assert summary["status"] == "failed"
    assert summary["reason"] == "T0 payload contains unsupported static obstacle geometry"
    assert summary["unsupported"]["static_obstacle_count"] == 1
    assert summary["unsupported"]["unsupported_static_obstacle_count"] == 1
    assert world.actors == []


def test_live_replay_spawns_replays_and_cleans_up_actors(tmp_path: Path) -> None:
    """Representable payloads should run an oracle transform replay against a live world."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

    world = _FakeWorld()
    manifest_path = _write_manifest(
        tmp_path,
        [{"scenario_id": "unit_crossing", "payload": _minimal_t0_payload()}],
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
        max_steps=3,
    )

    assert summary["status"] == "oracle-replay"
    assert summary["mode"] == "oracle-replay"
    assert summary["stage"] == "live-replay"
    assert summary["selected_payload"]["scenario_id"] == "unit_crossing"
    assert summary["carla"]["client_version"] == "0.9.16"
    assert summary["carla"]["server_version"] == "0.9.16"
    assert summary["carla"]["map"] == "Town01"
    assert summary["actors"] == {"static_obstacles": 0, "robot": 1, "pedestrians": 1}
    assert summary["trajectory"]["steps"] == 3
    assert summary["trajectory"]["truncated_by_max_steps"] is True
    assert world.ticks == 3
    assert len(world.actors) == 2
    assert all(actor.destroyed for actor in world.actors)
    assert len(world.actors[0].transforms) == 4


def test_live_replay_cleans_up_robot_when_pedestrian_spawn_fails(tmp_path: Path) -> None:
    """Partial CARLA actor spawn failures should not leave earlier actors live."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

    world = _PedestrianSpawnFailureWorld()
    manifest_path = _write_manifest(
        tmp_path,
        [{"scenario_id": "unit_crossing", "payload": _minimal_t0_payload()}],
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
        max_steps=3,
    )

    assert summary["status"] == "failed"
    assert summary["mode"] == "failed"
    assert summary["reason"] == "CARLA failed to spawn pedestrian ped_0"
    assert len(world.actors) == 1
    assert world.actors[0].destroyed is True
    assert world.ticks == 0


def test_live_replay_uses_carla_coordinate_boundary() -> None:
    """Robot-SF y/theta should be converted at the CARLA boundary, not in T0 export."""
    from robot_sf_carla_bridge.live_replay import robot_sf_pose_to_carla_transform

    transform = robot_sf_pose_to_carla_transform(
        _fake_carla_module(_FakeWorld()),
        {"x": 1.0, "y": 2.0, "theta": 1.57079632679},
    )

    assert transform.location.x == pytest.approx(1.0)
    assert transform.location.y == pytest.approx(-2.0)
    assert transform.location.z == pytest.approx(0.1)
    assert transform.rotation.yaw == pytest.approx(-90.0)
