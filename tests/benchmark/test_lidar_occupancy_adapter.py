"""Benchmark wiring tests for LiDAR-derived occupancy adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.map_runner import _build_policy, _run_map_episode
from robot_sf.gym_env.observation_mode import ObservationMode
from tests.benchmark.test_map_runner_utils import _minimal_map_def


def test_build_policy_wraps_safety_barrier_with_lidar_occupancy_adapter() -> None:
    """Explicit config should route safety_barrier through the LiDAR occupancy adapter."""
    policy, meta = _build_policy(
        "safety_barrier",
        {
            "lidar_occupancy_adapter": {
                "lidar_grid_resolution": 0.5,
                "lidar_grid_width": 4.0,
                "lidar_grid_height": 4.0,
                "lidar_max_range": 4.0,
                "lidar_angle_min": 0.0,
                "lidar_angle_max": 0.0,
                "lidar_obstacle_inflation_cells": 0,
                "lidar_rays_normalized": False,
                "drive_state_normalized": False,
            },
            "stop_distance": 0.6,
            "safe_distance": 1.0,
        },
        robot_kinematics="differential_drive",
        robot_command_mode="unicycle",
    )

    linear, angular = policy(
        {
            "rays": np.asarray([0.4], dtype=np.float32),
            "drive_state": np.asarray([0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32),
        }
    )
    stats = policy._planner_stats()

    assert meta["planner_kinematics"]["adapter_name"] == "LidarOccupancySafetyBarrierAdapter"
    assert meta["lidar_occupancy_adapter"]["status"] == "enabled"
    assert linear == 0.0
    assert abs(angular) > 0.0
    assert stats["lidar_occupancy_adapter"]["converted_observations"] == 1


def test_lidar_occupancy_map_episode_uses_sensor_fusion_observation(
    monkeypatch,
) -> None:
    """A LiDAR-level episode should hand sensor-fusion rays to the adapter."""

    class _DummySim:
        """Simulator stub for LiDAR adapter map-runner tests."""

        def __init__(self) -> None:
            self.robot_pos = [np.array([0.0, 0.0], dtype=float)]
            self.ped_pos = np.zeros((0, 2), dtype=float)
            self.goal_pos = [np.array([2.0, 0.0], dtype=float)]
            self.map_def = _minimal_map_def()
            self.last_ped_forces = np.zeros((0, 2), dtype=float)

    class _DummyEnv:
        """Environment stub that exposes DEFAULT_GYM-style LiDAR observations."""

        def __init__(self) -> None:
            self.simulator = _DummySim()

        def reset(self, seed: int | None = None):
            """Return one sensor-fusion observation."""
            _ = seed
            return {
                "rays": np.asarray([[0.1]], dtype=np.float32),
                "drive_state": np.asarray([[0.0, 0.0, 0.2, 0.0, 0.0]], dtype=np.float32),
            }, {}

        def step(self, action):
            """Finish after one action."""
            _ = action
            obs, _info = self.reset(seed=None)
            return obs, 0.0, True, False, {"success": False}

        def close(self) -> None:
            """Accept map-runner cleanup."""
            return None

    dummy_config = type(
        "Cfg",
        (),
        {"sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})()},
    )()
    captured_config = {}

    def _make_env(config, seed, debug):
        """Capture the environment config used by map-runner."""
        _ = seed, debug
        captured_config["observation_mode"] = config.observation_mode
        captured_config["use_occupancy_grid"] = config.use_occupancy_grid
        return _DummyEnv()

    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )
    monkeypatch.setattr("robot_sf.benchmark.map_runner.make_robot_env", _make_env)
    monkeypatch.setattr("robot_sf.benchmark.map_runner.sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_shortest_path_length",
        lambda *args: 1.0,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.compute_all_metrics",
        lambda *args, **kwargs: {"success": 0.0, "collisions": 0.0},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner.post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )

    record = _run_map_episode(
        {"name": "lidar-smoke", "simulation_config": {"max_episode_steps": 1}},
        seed=1,
        horizon=1,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="safety_barrier",
        algo_config={
            "lidar_occupancy_adapter": {
                "lidar_grid_resolution": 0.5,
                "lidar_grid_width": 4.0,
                "lidar_grid_height": 4.0,
                "lidar_max_range": 4.0,
            },
        },
        scenario_path=Path("."),
        observation_level="lidar_2d",
    )

    runtime = record["algorithm_metadata"]["planner_runtime"]["lidar_occupancy_adapter"]
    assert captured_config == {
        "observation_mode": ObservationMode.DEFAULT_GYM,
        "use_occupancy_grid": False,
    }
    assert record["observation_mode"] == "sensor_fusion_state"
    assert record["observation_level"] == "lidar_2d"
    assert runtime["converted_observations"] == 1
    assert runtime["unavailable_observations"] == 0


def test_lidar_safety_barrier_requires_explicit_occupancy_adapter(monkeypatch) -> None:
    """LiDAR-level safety_barrier runs should fail without the explicit adapter config."""
    dummy_config = type(
        "Cfg",
        (),
        {"sim_config": type("SC", (), {"time_per_step_in_secs": 0.1})()},
    )()
    monkeypatch.setattr(
        "robot_sf.benchmark.map_runner._build_env_config",
        lambda scenario, scenario_path: dummy_config,
    )

    with pytest.raises(ValueError, match="requires .*lidar_occupancy_adapter"):
        _run_map_episode(
            {"name": "lidar-smoke", "simulation_config": {"max_episode_steps": 1}},
            seed=1,
            horizon=1,
            dt=0.1,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo="safety_barrier",
            algo_config={},
            scenario_path=Path("."),
            observation_level="lidar_2d",
        )
