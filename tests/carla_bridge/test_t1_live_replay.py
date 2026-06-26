"""CARLA T1 live oracle replay tests."""

from __future__ import annotations

import math
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


class _FakeWaypoint:
    def __init__(self, transform: _FakeTransform) -> None:
        self.transform = transform


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


class _NoneReturningRobotBlueprintLibrary(_FakeBlueprintLibrary):
    def find(self, blueprint_id: str) -> _FakeBlueprint | None:
        if blueprint_id == "vehicle.tesla.model3":
            return None
        return super().find(blueprint_id)


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
        return _MapWithoutProjectionApi()

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


class _RobotSpawnFailureWorld(_FakeWorld):
    def try_spawn_actor(
        self, blueprint: _FakeBlueprint, transform: _FakeTransform
    ) -> _FakeActor | None:
        if blueprint.id.startswith("vehicle."):
            return None
        return super().try_spawn_actor(blueprint, transform)


class _MapWithoutProjectionApi:
    name = "Town01"


class _ProjectionFallbackMap(_MapWithoutProjectionApi):
    def __init__(self, waypoint_transform: _FakeTransform) -> None:
        self.calls: list[dict[str, float | bool | None]] = []
        self._waypoint = _FakeWaypoint(waypoint_transform)

    def get_waypoint(
        self,
        location: _FakeLocation,
        *,
        project_to_road: bool = False,
        lane_type=None,
    ) -> _FakeWaypoint:
        self.calls.append(
            {
                "x": location.x,
                "y": location.y,
                "z": location.z,
                "project_to_road": project_to_road,
                "lane_type": lane_type,
            }
        )
        return self._waypoint


class _RobotProjectionFallbackWorld(_FakeWorld):
    def __init__(self, carla_map: _ProjectionFallbackMap) -> None:
        super().__init__()
        self._map = carla_map

    def get_map(self) -> _ProjectionFallbackMap:
        return self._map

    def try_spawn_actor(
        self, blueprint: _FakeBlueprint, transform: _FakeTransform
    ) -> _FakeActor | None:
        if (
            blueprint.id.startswith("vehicle.")
            and transform.location.x == pytest.approx(0.0)
            and transform.location.y == pytest.approx(-0.0)
            and transform.location.z == pytest.approx(0.6)
        ):
            return None
        return super().try_spawn_actor(blueprint, transform)


class _RobotProjectionRejectedWorld(_RobotProjectionFallbackWorld):
    def try_spawn_actor(
        self, blueprint: _FakeBlueprint, transform: _FakeTransform
    ) -> _FakeActor | None:
        if blueprint.id.startswith("vehicle."):
            return None
        return super().try_spawn_actor(blueprint, transform)


class _SpawnActorRaisesProjectionWorld:
    def __init__(self, carla_map: _ProjectionFallbackMap) -> None:
        self.actors: list[_FakeActor] = []
        self.ticks = 0
        self._map = carla_map

    def get_map(self) -> _ProjectionFallbackMap:
        return self._map

    def get_blueprint_library(self) -> _FakeBlueprintLibrary:
        return _FakeBlueprintLibrary()

    def spawn_actor(self, blueprint: _FakeBlueprint, transform: _FakeTransform) -> _FakeActor:
        if (
            blueprint.id.startswith("vehicle.")
            and transform.location.x == pytest.approx(0.0)
            and transform.location.y == pytest.approx(-0.0)
            and transform.location.z == pytest.approx(0.6)
        ):
            raise RuntimeError("spawn blocked")
        actor = _FakeActor(blueprint, transform)
        self.actors.append(actor)
        return actor

    def tick(self) -> int:
        self.ticks += 1
        return self.ticks


class _RobotBlueprintFallbackWorld(_FakeWorld):
    def get_blueprint_library(self) -> _NoneReturningRobotBlueprintLibrary:
        return _NoneReturningRobotBlueprintLibrary()


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


def test_live_replay_projects_robot_when_spawn_actor_raises(tmp_path: Path) -> None:
    """Fallback spawn exceptions should become a retryable miss so projection can continue."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

    projected_transform = _FakeTransform(
        _FakeLocation(x=4.0, y=1.5, z=0.0),
        _FakeRotation(yaw=15.0),
    )
    carla_map = _ProjectionFallbackMap(projected_transform)
    world = _SpawnActorRaisesProjectionWorld(carla_map)
    manifest_path = _write_manifest(
        tmp_path, [{"scenario_id": "unit_crossing", "payload": _minimal_t0_payload()}]
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
        max_steps=2,
    )

    assert summary["status"] == "oracle-replay"
    assert summary["mode"] == "oracle-replay-adapted"
    assert summary["replay_metadata"]["robot_spawn"]["strategy"] == "carla-map-projection"
    assert summary["adaptations"][0]["reason"].startswith("CARLA failed to spawn robot")
    assert summary["adaptations"][0]["projected_transform"] == {
        "x": pytest.approx(4.0),
        "y": pytest.approx(1.5),
        "z": pytest.approx(0.6),
        "yaw": pytest.approx(15.0),
    }
    assert len(carla_map.calls) == 1
    assert all(actor.destroyed for actor in world.actors)


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
    assert summary["coordinate_alignment"] == {
        "bridge_version": "carla-t1-oracle-live-replay.v1",
        "carla_map_name": "Town01",
        "carla_server_version": "0.9.16",
        "eligible_for_metric_parity": True,
        "end": summary["coordinate_alignment"]["end"],
        "projection_meters": 0.0,
        "projection_rationale": "none",
        "replay_mode": "native",
        "robot_sf_commit": "abc123",
        "scenario_cert_id": "unit_crossing",
        "scenario_certificate_source": "output/cert.json",
        "start": summary["coordinate_alignment"]["start"],
    }
    assert summary["replay_metadata"]["coordinate_alignment"] == summary["coordinate_alignment"]
    assert summary["actors"] == {"static_obstacles": 0, "robot": 1, "pedestrians": 1}
    assert summary["trajectory"]["steps"] == 3
    assert summary["trajectory"]["truncated_by_max_steps"] is True
    assert summary["metrics"] == summary["trajectory"]["metrics"]
    assert summary["metrics"] == {
        "collision": True,
        "intervention_rate": 0.0,
        "min_distance_m": pytest.approx(-0.3),
        "success": False,
    }
    assert world.ticks == 3
    assert len(world.actors) == 2
    assert all(actor.destroyed for actor in world.actors)
    assert len(world.actors[0].transforms) == 4


def test_live_replay_metrics_are_parity_comparable(tmp_path: Path) -> None:
    """Native live replay metrics should be consumable by the parity adapter."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server
    from robot_sf_carla_bridge.parity import compare_oracle_replay_metrics

    world = _FakeWorld()
    payload = _minimal_t0_payload()
    payload["pedestrians"] = []
    payload["simulation"]["horizon_s"] = 0.2
    manifest_path = _write_manifest(
        tmp_path,
        [{"scenario_id": "unit_crossing", "payload": payload}],
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
        max_steps=None,
    )
    report = compare_oracle_replay_metrics(
        {"metrics": {"success": True, "collision": False}},
        summary,
        metric_names=("success", "collision"),
    )

    assert report["status"] == "comparable"
    assert [row["status"] for row in report["metrics"]] == ["match", "match"]


def test_live_replay_metrics_report_success_for_complete_oracle_replay(tmp_path: Path) -> None:
    """Untruncated native oracle replay should emit comparable success metrics."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

    world = _FakeWorld()
    payload = _minimal_t0_payload()
    payload["pedestrians"] = []
    payload["simulation"]["horizon_s"] = 0.2
    manifest_path = _write_manifest(
        tmp_path,
        [{"scenario_id": "unit_crossing", "payload": payload}],
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
        max_steps=None,
    )

    assert summary["mode"] == "oracle-replay"
    assert summary["trajectory"]["truncated_by_max_steps"] is False
    assert summary["metrics"] == {
        "collision": False,
        "intervention_rate": 0.0,
        "success": True,
    }


def test_live_replay_metrics_detect_static_obstacle_collision(tmp_path: Path) -> None:
    """Replay metrics should expose collisions against replayed static geometry."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

    world = _FakeWorld()
    payload = _minimal_t0_payload()
    payload["pedestrians"] = []
    payload["robot"]["footprint"]["radius_m"] = 0.3
    payload["static_geometry"]["obstacles"] = [
        {"id": "start_overlap", "type": "polygon", "vertices": [[0, -1], [1, -1], [1, 1], [0, 1]]}
    ]
    manifest_path = _write_manifest(
        tmp_path,
        [{"scenario_id": "unit_crossing", "payload": payload}],
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
        max_steps=1,
    )

    assert summary["metrics"]["collision"] is True
    assert summary["metrics"]["min_distance_m"] == pytest.approx(-0.3)


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
    assert summary["reason"] == (
        "CARLA failed to spawn pedestrian ped_0 with blueprint walker.pedestrian.0001 "
        "via try_spawn_actor at x=2.000, y=1.000, z=0.100, yaw=-89.954"
    )
    assert len(world.actors) == 1
    assert world.actors[0].destroyed is True
    assert world.ticks == 0


def test_live_replay_falls_back_when_robot_blueprint_find_returns_none(tmp_path: Path) -> None:
    """CARLA blueprint libraries may return None instead of raising for missing preferred IDs."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

    world = _RobotBlueprintFallbackWorld()
    manifest_path = _write_manifest(
        tmp_path,
        [{"scenario_id": "unit_crossing", "payload": _minimal_t0_payload()}],
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
        max_steps=1,
    )

    assert summary["status"] == "oracle-replay"
    assert world.actors[0].blueprint.id == "vehicle.0001"
    assert world.actors[0].blueprint.attributes["role_name"] == "robot_sf_robot"


def test_live_replay_reports_robot_spawn_blueprint_and_location(tmp_path: Path) -> None:
    """Robot spawn failures should fail closed when map projection fallback is unavailable."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

    world = _RobotSpawnFailureWorld()
    manifest_path = _write_manifest(
        tmp_path,
        [{"scenario_id": "unit_crossing", "payload": _minimal_t0_payload()}],
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
        max_steps=1,
    )

    assert summary["status"] == "failed"
    assert summary["reason"] == (
        "CARLA failed to spawn robot with blueprint vehicle.tesla.model3 "
        "via try_spawn_actor at x=0.000, y=-0.000, z=0.600, yaw=-0.000; "
        "CARLA map projection fallback unavailable: map.get_waypoint API not available"
    )
    assert world.actors == []


def test_live_replay_records_robot_spawn_projection_fallback_metadata(tmp_path: Path) -> None:
    """Robot spawn should record explicit map-projection adaptation when exact CARLA placement fails."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

    projected_transform = _FakeTransform(
        _FakeLocation(x=0.25, y=-0.5, z=0.05),
        _FakeRotation(yaw=-12.0),
    )
    carla_map = _ProjectionFallbackMap(projected_transform)
    world = _RobotProjectionFallbackWorld(carla_map)
    manifest_path = _write_manifest(
        tmp_path,
        [{"scenario_id": "unit_crossing", "payload": _minimal_t0_payload()}],
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
        max_steps=1,
    )

    assert summary["status"] == "oracle-replay"
    assert summary["replay_metadata"]["robot_spawn"]["strategy"] == "carla-map-projection"
    assert summary["replay_metadata"]["robot_spawn"]["adapted"] is True
    assert summary["coordinate_alignment"]["replay_mode"] == "adapted"
    assert summary["coordinate_alignment"]["eligible_for_metric_parity"] is False
    assert summary["coordinate_alignment"]["projection_meters"] == pytest.approx(
        (0.25**2 + 0.5**2 + 0.05**2) ** 0.5
    )
    assert "exclusion_reason" in summary["coordinate_alignment"]
    assert summary["replay_metadata"]["robot_spawn"]["projection_source"] == (
        "world.get_map().get_waypoint(project_to_road=True)"
    )
    assert summary["replay_metadata"]["robot_spawn"]["requested_transform"] == {
        "x": 0.0,
        "y": -0.0,
        "z": 0.6,
        "yaw": -0.0,
    }
    assert summary["replay_metadata"]["robot_spawn"]["spawn_transform"] == {
        "x": 0.25,
        "y": -0.5,
        "z": 0.65,
        "yaw": -12.0,
    }
    assert carla_map.calls == [
        {"x": 0.0, "y": -0.0, "z": 0.6, "project_to_road": True, "lane_type": None}
    ]
    assert world.actors[0].transforms[0].location.x == pytest.approx(0.25)
    assert world.actors[0].transforms[0].location.y == pytest.approx(-0.5)
    assert world.actors[0].transforms[0].location.z == pytest.approx(0.65)
    assert world.actors[0].transforms[1].location.x == pytest.approx(0.0)
    assert world.actors[0].transforms[1].location.z == pytest.approx(0.6)


def test_live_replay_fails_closed_when_projected_robot_spawn_is_rejected(tmp_path: Path) -> None:
    """Projected robot spawn rejections should keep the adapter fail-closed with both attempts in diagnostics."""
    from robot_sf_carla_bridge.live_replay import run_t1_oracle_live_replay_against_server

    carla_map = _ProjectionFallbackMap(
        _FakeTransform(
            _FakeLocation(x=0.25, y=-0.5, z=0.05),
            _FakeRotation(yaw=-12.0),
        )
    )
    world = _RobotProjectionRejectedWorld(carla_map)
    manifest_path = _write_manifest(
        tmp_path,
        [{"scenario_id": "unit_crossing", "payload": _minimal_t0_payload()}],
    )

    summary = run_t1_oracle_live_replay_against_server(
        manifest_path,
        carla_module=_fake_carla_module(world),
        max_steps=1,
    )

    assert summary["status"] == "failed"
    assert summary["reason"] == (
        "CARLA failed to spawn robot with blueprint vehicle.tesla.model3 "
        "via try_spawn_actor at x=0.000, y=-0.000, z=0.600, yaw=-0.000; "
        "CARLA map projection fallback via world.get_map().get_waypoint(project_to_road=True) "
        "also failed for robot with blueprint vehicle.tesla.model3 via try_spawn_actor "
        "at x=0.250, y=-0.500, z=0.650, yaw=-12.000"
    )
    assert world.actors == []


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


def test_live_replay_coordinate_helpers_pin_handedness_contract() -> None:
    """CARLA bridge helpers should mirror vector, yaw, and yaw-rate signs."""
    from robot_sf_carla_bridge.live_replay import (
        carla_angular_velocity_to_robot_sf_radps,
        carla_planar_vector_to_robot_sf,
        carla_yaw_degrees_to_robot_sf,
        robot_sf_angular_velocity_to_carla_radps,
        robot_sf_planar_vector_to_carla,
        robot_sf_yaw_to_carla_degrees,
    )

    carla_vector = robot_sf_planar_vector_to_carla({"x": -3.0, "y": 4.5})
    assert carla_vector == {"x": pytest.approx(-3.0), "y": pytest.approx(-4.5)}
    assert carla_planar_vector_to_robot_sf(carla_vector) == {
        "x": pytest.approx(-3.0),
        "y": pytest.approx(4.5),
    }

    carla_yaw = robot_sf_yaw_to_carla_degrees(-0.25)
    assert carla_yaw == pytest.approx(math.degrees(0.25))
    assert carla_yaw_degrees_to_robot_sf(carla_yaw) == pytest.approx(-0.25)

    carla_omega = robot_sf_angular_velocity_to_carla_radps(0.75)
    assert carla_omega == pytest.approx(-0.75)
    assert carla_angular_velocity_to_robot_sf_radps(carla_omega) == pytest.approx(0.75)
