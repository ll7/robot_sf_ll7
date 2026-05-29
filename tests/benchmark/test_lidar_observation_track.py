"""Tests for the issue #1613 LiDAR-observation benchmark track packet."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.map_runner import _run_map_episode
from robot_sf.benchmark.planner_command_contract import (
    PlannerContractValidationError,
    validate_planner_contract,
)
from robot_sf.common.types import Rect
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKET_PATH = _REPO_ROOT / "configs/benchmarks/lidar/observation_track_smoke_issue_1613.yaml"


def _load_packet() -> dict[str, Any]:
    payload = yaml.safe_load(_PACKET_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _minimal_map_def() -> MapDefinition:
    width = 5.0
    height = 4.0
    spawn_zone: Rect = ((0.5, 0.5), (1.0, 0.5), (0.5, 1.0))
    goal_zone: Rect = ((3.5, 2.5), (4.0, 2.5), (3.5, 3.0))
    bounds = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(0.75, 0.75), (3.75, 2.75)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    return MapDefinition(
        width=width,
        height=height,
        obstacles=[],
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[spawn_zone],
        robot_goal_zones=[goal_zone],
        bounds=bounds,
        robot_routes=[route],
        ped_goal_zones=[goal_zone],
        ped_crowded_zones=[],
        ped_routes=[route],
        single_pedestrians=[],
    )


def test_lidar_track_packet_declares_sensor_fusion_runtime_boundary() -> None:
    """The launch packet should define LiDAR plus goal inputs and exclude privileged state."""
    packet = _load_packet()
    track = packet["observation_track"]

    assert packet["scenario_matrix"] == "configs/scenarios/sanity_v1.yaml"
    assert (_REPO_ROOT / packet["scenario_matrix"]).exists()
    assert track["benchmark_observation_level"] == "lidar_2d"
    assert track["active_observation_mode"] == "sensor_fusion_state"
    assert track["environment_observation_keys"] == ["drive_state", "rays"]
    assert "occupancy_grid" in track["excluded_runtime_inputs"]
    assert "socnav_struct_pedestrian_positions" in track["excluded_runtime_inputs"]


def test_lidar_track_contract_accepts_only_sensor_fusion_learned_smoke() -> None:
    """Current benchmark contracts should not turn structured planners into LiDAR rows."""
    contract = validate_planner_contract(
        algo="ppo",
        robot_kinematics="differential_drive",
        algo_config={},
        observation_mode="sensor_fusion_state",
        observation_level="lidar_2d",
    )

    assert contract["observation_contract"]["observation_level"] == "lidar_2d"
    assert contract["observation_contract"]["active_mode"] == "sensor_fusion_state"
    assert "lidar_rays" in contract["observation_contract"]["required_inputs"]

    for planner in ("goal", "grid_route", "orca"):
        with pytest.raises(PlannerContractValidationError):
            validate_planner_contract(
                algo=planner,
                robot_kinematics="differential_drive",
                algo_config={},
                observation_level="lidar_2d",
            )


def test_stubbed_lidar_map_episode_records_track_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A minimal map episode should emit LiDAR track metadata and sensor-fusion keys."""
    seen_observations: list[dict[str, Any]] = []

    class _DummySim:
        def __init__(self, map_def: MapDefinition) -> None:
            self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
            self.ped_pos = np.zeros((0, 2), dtype=float)
            self.goal_pos = [np.array([1.0, 1.0], dtype=float)]
            self.map_def = map_def
            self.last_ped_forces = np.zeros((0, 2), dtype=float)

    class _DummyEnv:
        def __init__(self, map_def: MapDefinition) -> None:
            self.simulator = _DummySim(map_def)

        def reset(self, seed: int | None = None):
            del seed
            return _sensor_fusion_obs(), {}

        def step(self, action):
            del action
            return (
                _sensor_fusion_obs(),
                0.0,
                True,
                False,
                {"success": True, "meta": {"is_route_complete": True}},
            )

        def close(self) -> None:
            return None

    def _sensor_fusion_obs() -> dict[str, Any]:
        return {
            "drive_state": np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
            "rays": np.ones((1, 8), dtype=np.float32),
        }

    def _dummy_policy(obs: dict[str, Any]) -> tuple[float, float]:
        seen_observations.append(dict(obs))
        return (0.0, 0.0)

    map_def = _minimal_map_def()
    dummy_config = SimpleNamespace(
        sim_config=SimpleNamespace(time_per_step_in_secs=0.1, ped_radius=0.4),
        robot_config=None,
    )

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.make_robot_env",
        lambda config, seed, debug: _DummyEnv(map_def),
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_policy",
        lambda *args, **kwargs: (_dummy_policy, {"status": "ok"}),
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_shortest_path_length",
        lambda *args: 1.0,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_all_metrics",
        lambda *args, **kwargs: {"success": 1.0, "collisions": 0.0},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )

    record = _run_map_episode(
        {"name": "lidar_smoke", "simulation_config": {"max_episode_steps": 1}},
        seed=1613,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="ppo",
        algo_config={},
        scenario_path=Path("configs/scenarios/sanity_v1.yaml"),
        observation_mode="sensor_fusion_state",
        observation_level="lidar_2d",
    )

    assert seen_observations
    assert set(seen_observations[0]) == {"drive_state", "rays"}
    assert record["observation_level"] == "lidar_2d"
    assert record["observation_mode"] == "sensor_fusion_state"
    assert record["scenario_params"]["observation_level"] == "lidar_2d"
    assert record["scenario_params"]["observation_mode"] == "sensor_fusion_state"
    metadata = record["algorithm_metadata"]
    assert metadata["observation_level"]["key"] == "lidar_2d"
    assert metadata["observation_spec"]["active_mode"] == "sensor_fusion_state"
    assert metadata["planner_contract"]["observation_contract"]["required_inputs"] == [
        "robot_state",
        "goal",
        "lidar_rays",
        "history",
    ]
