"""Every planner-facing pedestrian channel consumes one perceived snapshot."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Polygon

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.nav.occupancy_grid import GridChannel, OccupancyGrid
from robot_sf.sensor.pedestrian_perception import (
    PedestrianPerception,
    PedestrianPerceptionSpec,
    position_noise_offsets,
)
from tests.metamorphic.planner_arms import SAMPLED_PED_DENSITY, robot_env_config, sampled_scene

STEPS = 5


def _env(seed: int = 7):
    config = robot_env_config(sampled_scene(), max_steps=STEPS + 2, ped_density=SAMPLED_PED_DENSITY)
    return make_robot_env(config=config, seed=seed)


def _ped_layer(env, positions: np.ndarray) -> np.ndarray:
    """Rasterize ``positions`` with the env's own grid config and static geometry."""
    grid = OccupancyGrid(config=env.occupancy_grid.config)
    obstacles, polygons = env._get_static_grid_obstacles()
    radii = np.full(len(positions), 0.35)
    if getattr(env.simulator, "ped_radii", None) is not None:
        radii = np.asarray(env.simulator.ped_radii, dtype=float)[: len(positions)]
    grid.generate(
        obstacles=obstacles,
        pedestrians=(np.asarray(positions, dtype=float).reshape(-1, 2), radii),
        robot_pose=env.simulator.robot_poses[0],
        ego_frame=False,
        obstacle_polygons=polygons,
    )
    index = list(env.occupancy_grid.config.channels).index(GridChannel.PEDESTRIANS)
    return grid.to_observation()[index]


def _obs_ped_layer(env, obs) -> np.ndarray:
    index = list(env.occupancy_grid.config.channels).index(GridChannel.PEDESTRIANS)
    return np.asarray(obs["occupancy_grid"])[index]


def test_occupancy_layer_is_not_truth_under_position_noise() -> None:
    """Fails if the PPO grid layer is still rasterized from simulator truth."""
    env = _env()
    spec = PedestrianPerceptionSpec(condition="position_noise", position_noise_std_m=1.5)
    perception = PedestrianPerception(spec=spec, scenario_id="unit", seed=7)
    env.set_pedestrian_perception(perception)
    obs, _ = env.reset(seed=7)
    checked = 0
    for _ in range(STEPS):
        true_pos = np.asarray(env.simulator.ped_pos, dtype=float).reshape(-1, 2)
        perceived = perception.current.positions
        assert true_pos.shape[0] > 0
        assert not np.allclose(perceived, true_pos)
        layer = _obs_ped_layer(env, obs)
        np.testing.assert_array_equal(layer, _ped_layer(env, perceived))
        assert not np.array_equal(layer, _ped_layer(env, true_pos)), (
            "occupancy pedestrian layer still shows true positions under noise"
        )
        count = int(np.asarray(obs["pedestrians_count"]).reshape(-1)[0])
        rows = np.asarray(obs["pedestrians_positions"], dtype=float)[:count]
        cap = np.array([env.map_def.width, env.map_def.height], dtype=float)
        expected = np.clip(perceived, 0.0, None).astype(np.float32)
        for row in rows:
            assert np.min(np.linalg.norm(expected - row, axis=1)) < 1e-4 or np.any(
                row >= cap - 1e-3
            )
        checked += 1
        obs, *_ = env.step(np.zeros(2))
    summary = perception.summary()
    assert checked == STEPS
    assert summary["all_channels_consumed_every_snapshot"]
    assert set(summary["expected_channels"]) == {
        "socnav_pedestrian_rows",
        "occupancy_grid_pedestrians",
    }


def test_oracle_condition_matches_unperturbed_env_exactly() -> None:
    """The control routes through the snapshot but must equal the release observation."""
    plain, oracle = _env(), _env()
    perception = PedestrianPerception(
        spec=PedestrianPerceptionSpec(condition="oracle"), scenario_id="unit", seed=7
    )
    oracle.set_pedestrian_perception(perception)
    obs_a, _ = plain.reset(seed=7)
    obs_b, _ = oracle.reset(seed=7)
    for _ in range(STEPS):
        assert obs_a.keys() == obs_b.keys()
        for key in obs_a:
            np.testing.assert_array_equal(np.asarray(obs_a[key]), np.asarray(obs_b[key]), key)
        action = np.array([0.5, 0.1])
        obs_a, *_ = plain.step(action)
        obs_b, *_ = oracle.step(action)
    assert perception.summary()["stats"]["steps_perceived_differs"] == 0


def _fake_sim(ped_pos, ped_vel=None, robot=(0.0, 0.0), obstacles=()):
    ped_pos = np.asarray(ped_pos, dtype=float).reshape(-1, 2)
    return SimpleNamespace(
        ped_pos=ped_pos,
        ped_vel=np.zeros_like(ped_pos) if ped_vel is None else np.asarray(ped_vel, dtype=float),
        robots=[SimpleNamespace(pose=(np.asarray(robot, dtype=float), 0.0))],
        map_def=SimpleNamespace(obstacles=[SimpleNamespace(geometry=g) for g in obstacles]),
    )


def test_delay_replays_truth_from_two_steps_earlier() -> None:
    perception = PedestrianPerception(
        spec=PedestrianPerceptionSpec(condition="delay", delay_steps=2), scenario_id="u", seed=1
    )
    seen = []
    for step in range(6):
        snap = perception.observe(_fake_sim([[float(step), 0.0]], [[float(step), 1.0]]))
        seen.append((snap.positions[0, 0], snap.velocities[0, 0]))
    assert seen == [(0, 0), (0, 0), (0, 0), (1, 1), (2, 2), (3, 3)]
    assert perception.summary()["stats"]["delay_held_initial_steps"] == 2


def test_static_occlusion_removes_pedestrians_behind_walls_only() -> None:
    wall = Polygon([(2.0, -1.0), (3.0, -1.0), (3.0, 1.0), (2.0, 1.0)])
    perception = PedestrianPerception(
        spec=PedestrianPerceptionSpec(condition="static_occlusion"), scenario_id="u", seed=1
    )
    snap = perception.observe(_fake_sim([[5.0, 0.0], [0.0, 5.0]], obstacles=[wall]))
    assert snap.source_slots.tolist() == [1]
    np.testing.assert_array_equal(snap.positions, [[0.0, 5.0]])
    assert perception.summary()["stats"]["rows_occluded"] == 1


def test_noise_is_indexed_by_episode_step_and_pedestrian() -> None:
    a = position_noise_offsets(123, 4, np.arange(5), 0.05)
    b = position_noise_offsets(123, 4, np.array([3]), 0.05)
    np.testing.assert_array_equal(a[3], b[0])
    assert not np.array_equal(a, position_noise_offsets(123, 5, np.arange(5), 0.05))
    assert not np.array_equal(a, position_noise_offsets(124, 4, np.arange(5), 0.05))
    big = position_noise_offsets(9, 0, np.arange(20000), 0.05)
    assert abs(big.std() - 0.05) < 0.002


@pytest.mark.parametrize(
    "raw",
    [
        {"condition": "oracle", "position_noise_std_m": 0.05},
        {"condition": "position_noise"},
        {"condition": "delay", "delay_steps": 0},
        {"condition": "oracle", "delay_steps": 2},
        {"condition": "occlusion"},
        {"condition": "oracle", "extra": 1},
    ],
)
def test_spec_rejects_mixed_or_missing_effects(raw) -> None:
    with pytest.raises(ValueError):
        PedestrianPerceptionSpec.from_mapping(raw)
