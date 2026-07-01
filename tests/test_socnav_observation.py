"""Tests for SocNav observation bounds and clipping."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.gym_env.unified_config import ObservationVisibilitySettings, RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.sensor.socnav_observation import SocNavObservationFusion, socnav_observation_space


def _build_map_def(width: float, height: float) -> MapDefinition:
    """Build a map definition sized for SocNav observation bound tests."""
    obstacles = [Obstacle([(0, 0), (width, 0), (width, 1), (0, 1)])]
    robot_spawn_zones = [((1, 1), (2, 1), (2, 2))]
    ped_spawn_zones = [((3, 3), (4, 3), (4, 4))]
    robot_goal_zones = [((width - 2, height - 2), (width - 1, height - 2), (width - 1, height - 1))]
    bounds = [
        (0, width, 0, 0),
        (0, width, height, height),
        (0, 0, 0, height),
        (width, width, 0, height),
    ]
    ped_goal_zones = [((6, 6), (7, 6), (7, 7))]
    ped_crowded_zones: list = []
    robot_routes: list = []
    ped_routes: list = []
    single_pedestrians: list = []
    return MapDefinition(
        width,
        height,
        obstacles,
        robot_spawn_zones,
        ped_spawn_zones,
        robot_goal_zones,
        bounds,
        robot_routes,
        ped_goal_zones,
        ped_crowded_zones,
        ped_routes,
        single_pedestrians,
    )


def test_socnav_observation_space_uses_map_aware_cap() -> None:
    """SocNav observation space should preserve coordinates on maps larger than 50 m."""
    map_def = _build_map_def(120.0, 80.0)
    env_config = RobotSimulationConfig()
    space = socnav_observation_space(map_def, env_config, max_pedestrians=4)

    robot_pos_high = space["robot"]["position"].high
    map_size_high = space["map"]["size"].high

    assert robot_pos_high[0] == pytest.approx(120.0)
    assert robot_pos_high[1] == pytest.approx(80.0)
    assert map_size_high[0] == pytest.approx(120.0)
    assert map_size_high[1] == pytest.approx(80.0)


def test_socnav_observation_clips_positions_to_map_aware_cap() -> None:
    """SocNav observations should not clip valid large-map coordinates at 50 m."""
    env_config = RobotSimulationConfig()
    simulator = SimpleNamespace(
        ped_pos=np.array([[100.0, 100.0]], dtype=np.float32),
        ped_vel=np.array([[1.0, 0.0]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((100.0, 100.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([120.0, 80.0], dtype=np.float32)],
        next_goal_pos=[np.array([70.0, 70.0], dtype=np.float32)],
        map_def=SimpleNamespace(width=120.0, height=80.0),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )

    fusion = SocNavObservationFusion(simulator=simulator, env_config=env_config, max_pedestrians=4)
    obs = fusion.next_obs()

    assert obs["robot"]["position"].tolist() == pytest.approx([100.0, 80.0])
    assert obs["goal"]["current"].tolist() == pytest.approx([120.0, 80.0])
    assert obs["goal"]["next"].tolist() == pytest.approx([70.0, 70.0])
    assert obs["pedestrians"]["positions"][0].tolist() == pytest.approx([100.0, 80.0])
    assert obs["map"]["size"].tolist() == pytest.approx([120.0, 80.0])


def test_socnav_observation_rotates_pedestrian_velocities_to_ego_frame() -> None:
    """Pedestrian velocities should be rotated into the robot heading frame."""
    env_config = RobotSimulationConfig()
    simulator = SimpleNamespace(
        ped_pos=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ped_vel=np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), np.pi / 2.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )

    obs = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    ).next_obs()

    np.testing.assert_allclose(
        obs["pedestrians"]["velocities"][:2],
        np.array([[0.0, -1.0], [2.0, 0.0]], dtype=np.float32),
        atol=1e-6,
    )


def test_socnav_observation_visibility_filters_fov_without_mutating_ground_truth() -> None:
    """Planner-facing pedestrian observations should honor opt-in FOV settings only."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=90.0,
    )
    simulator = SimpleNamespace(
        ped_pos=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ped_vel=np.zeros((2, 2), dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )

    obs = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    ).next_obs()

    assert obs["pedestrians"]["count"][0] == pytest.approx(1.0)
    assert obs["pedestrians"]["positions"][0].tolist() == pytest.approx([3.0, 0.0])
    assert simulator.ped_pos.shape == (2, 2)


def test_socnav_observation_visibility_360_keeps_current_pedestrian_contract() -> None:
    """Full-FOV visibility should reproduce the unfiltered pedestrian observation."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=360.0,
        max_range_m=None,
    )
    simulator = SimpleNamespace(
        ped_pos=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ped_vel=np.zeros((2, 2), dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )

    obs = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    ).next_obs()

    assert obs["pedestrians"]["count"][0] == pytest.approx(2.0)
    np.testing.assert_allclose(
        obs["pedestrians"]["positions"][:2],
        np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
    )


def test_socnav_lost_pedestrian_memory_predicts_then_drops_after_horizon() -> None:
    """Lost pedestrians should persist with constant velocity only within the configured horizon."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=90.0,
        memory_for_lost_pedestrians=True,
        lost_pedestrian_memory_horizon_s=1.0,
    )
    simulator = SimpleNamespace(
        ped_pos=np.array([[2.0, 0.0]], dtype=np.float32),
        ped_vel=np.array([[1.0, 0.0]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.5),
    )
    fusion = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    first = fusion.next_obs()
    assert first["pedestrians"]["count"][0] == pytest.approx(1.0)

    simulator.ped_pos = np.array([[0.0, 2.0]], dtype=np.float32)
    remembered = fusion.next_obs()
    assert remembered["pedestrians"]["count"][0] == pytest.approx(1.0)
    np.testing.assert_allclose(
        remembered["pedestrians"]["positions"][0],
        np.array([2.5, 0.0], dtype=np.float32),
    )

    remembered_again = fusion.next_obs()
    assert remembered_again["pedestrians"]["count"][0] == pytest.approx(1.0)
    np.testing.assert_allclose(
        remembered_again["pedestrians"]["positions"][0],
        np.array([3.0, 0.0], dtype=np.float32),
    )

    dropped = fusion.next_obs()
    assert dropped["pedestrians"]["count"][0] == pytest.approx(0.0)


def test_socnav_observation_visibility_filters_static_occluded_pedestrian() -> None:
    """Static obstacle geometry should hide pedestrians behind an occluding polygon."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        static_occlusion=True,
    )
    obstacle = Obstacle([(1.0, -0.5), (2.0, -0.5), (2.0, 0.5), (1.0, 0.5)])
    simulator = SimpleNamespace(
        ped_pos=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ped_vel=np.zeros((2, 2), dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0, obstacles=[obstacle]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )

    obs = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    ).next_obs()

    assert obs["pedestrians"]["count"][0] == pytest.approx(1.0)
    assert obs["pedestrians"]["positions"][0].tolist() == pytest.approx([0.0, 3.0])
    np.testing.assert_allclose(
        simulator.ped_pos,
        np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
    )


def test_socnav_observation_visibility_filters_dynamic_occluded_pedestrian() -> None:
    """Opt-in dynamic occlusion should hide pedestrians blocked by nearer pedestrians."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        dynamic_occlusion=True,
    )
    simulator = SimpleNamespace(
        ped_pos=np.array([[2.0, 0.0], [4.0, 0.0], [4.0, 1.0]], dtype=np.float32),
        ped_vel=np.zeros((3, 2), dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )

    obs = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    ).next_obs()

    assert obs["pedestrians"]["count"][0] == pytest.approx(2.0)
    assert obs["pedestrians"]["positions"][0].tolist() == pytest.approx([2.0, 0.0])
    assert obs["pedestrians"]["positions"][1].tolist() == pytest.approx([4.0, 1.0])


def test_socnav_observation_no_stale_pedestrian_data_on_shrink() -> None:
    """Stale padded values from prior next_obs() calls must not leak into unfilled
    buffer rows when simulated pedestrian count shrinks."""
    env_config = RobotSimulationConfig()
    simulator = SimpleNamespace(
        ped_pos=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        ped_vel=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    fusion = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    obs1 = fusion.next_obs()
    assert obs1["pedestrians"]["count"][0] == pytest.approx(3.0)
    pos1_row2 = obs1["pedestrians"]["positions"][2].copy()
    vel1_row2 = obs1["pedestrians"]["velocities"][2].copy()

    # Shrink from 3 pedestrians to 1 for the second call
    simulator.ped_pos = np.array([[1.0, 1.0]], dtype=np.float32)
    simulator.ped_vel = np.array([[1.0, 0.0]], dtype=np.float32)
    obs2 = fusion.next_obs()

    assert obs2["pedestrians"]["count"][0] == pytest.approx(1.0)
    np.testing.assert_array_equal(
        obs2["pedestrians"]["positions"][0],
        np.array([1.0, 1.0], dtype=np.float32),
    )
    for i in range(1, 4):
        np.testing.assert_array_equal(
            obs2["pedestrians"]["positions"][i],
            np.zeros(2, dtype=np.float32),
            err_msg=f"Stale position leaked at padded row {i}",
        )
        np.testing.assert_array_equal(
            obs2["pedestrians"]["velocities"][i],
            np.zeros(2, dtype=np.float32),
            err_msg=f"Stale velocity leaked at padded row {i}",
        )

    # Verify snapshot semantics: first observation arrays must not be mutated
    np.testing.assert_array_equal(
        obs1["pedestrians"]["positions"][2],
        pos1_row2,
        err_msg="Snapshot semantics violated: first observation position mutated",
    )
    np.testing.assert_array_equal(
        obs1["pedestrians"]["velocities"][2],
        vel1_row2,
        err_msg="Snapshot semantics violated: first observation velocity mutated",
    )


def test_static_occlusion_cache_refresh_on_map_def_change() -> None:
    """Static-occlusion cache should rebuild when simulator.map_def identity changes."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        static_occlusion=True,
    )
    obstacle = Obstacle([(1.0, -0.5), (2.0, -0.5), (2.0, 0.5), (1.0, 0.5)])
    simulator = SimpleNamespace(
        ped_pos=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ped_vel=np.zeros((2, 2), dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0, obstacles=[obstacle]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    fusion = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    obs1 = fusion.next_obs()
    assert obs1["pedestrians"]["count"][0] == pytest.approx(1.0)

    # Replace map_def with a new object containing different obstacles
    new_obstacle = Obstacle([(3.5, -0.5), (4.5, -0.5), (4.5, 0.5), (3.5, 0.5)])
    simulator.map_def = SimpleNamespace(width=10.0, height=10.0, obstacles=[new_obstacle])

    obs2 = fusion.next_obs()
    assert obs2["pedestrians"]["count"][0] == pytest.approx(2.0)


def test_static_occlusion_cache_refresh_on_obstacles_change() -> None:
    """Static-occlusion cache should rebuild when obstacles list identity changes on same map_def."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        static_occlusion=True,
    )
    obstacle = Obstacle([(1.0, -0.5), (2.0, -0.5), (2.0, 0.5), (1.0, 0.5)])
    map_def = SimpleNamespace(width=10.0, height=10.0, obstacles=[obstacle])
    simulator = SimpleNamespace(
        ped_pos=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ped_vel=np.zeros((2, 2), dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=map_def,
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    fusion = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    obs1 = fusion.next_obs()
    assert obs1["pedestrians"]["count"][0] == pytest.approx(1.0)

    # Replace obstacles on the same map_def object
    map_def.obstacles = []

    obs2 = fusion.next_obs()
    assert obs2["pedestrians"]["count"][0] == pytest.approx(2.0)


def test_static_occlusion_edge_touch_not_occluded() -> None:
    """A line segment that only touches the obstacle boundary should not occlude."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        static_occlusion=True,
    )
    obstacle = Obstacle([(1.0, -0.5), (2.0, -0.5), (2.0, 0.5), (1.0, 0.5)])
    simulator = SimpleNamespace(
        ped_pos=np.array([[3.0, 0.5], [3.0, 0.0]], dtype=np.float32),
        ped_vel=np.zeros((2, 2), dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.5), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([5.0, 0.5], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=10.0, obstacles=[obstacle]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    fusion = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    obs = fusion.next_obs()

    # Robot at (0, 0.5)
    # ped at (3, 0.5): line at y=0.5 only touches obstacle top edge -> visible
    # ped at (3, 0.0): line goes through obstacle interior -> occluded
    assert obs["pedestrians"]["count"][0] == pytest.approx(1.0)
    np.testing.assert_allclose(obs["pedestrians"]["positions"][0], [3.0, 0.5], atol=1e-6)


def test_position_cap_cache_refreshes_on_map_def_change() -> None:
    """Position cap should refresh when map_def identity or width/height changes."""
    env_config = RobotSimulationConfig()
    simulator = SimpleNamespace(
        ped_pos=np.zeros((1, 2), dtype=np.float32),
        ped_vel=np.zeros((1, 2), dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=1.0),
            )
        ],
        goal_pos=[np.array([0.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=120.0, height=80.0),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    fusion = SocNavObservationFusion(
        simulator=simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    cap1 = fusion._position_cap()
    cap1_id = id(cap1)
    assert cap1.tolist() == pytest.approx([120.0, 80.0])

    cap1_again = fusion._position_cap()
    assert id(cap1_again) == cap1_id, "Same map_def should reuse cached array"

    # Mutate width on same map_def object — should refresh
    simulator.map_def.width = 200.0
    cap2 = fusion._position_cap()
    assert cap2.tolist() == pytest.approx([200.0, 80.0])
    assert id(cap2) != cap1_id, "Width change should produce new cached array"

    # Replace map_def entirely — should refresh
    simulator.map_def = SimpleNamespace(width=60.0, height=60.0)
    cap3 = fusion._position_cap()
    assert cap3.tolist() == pytest.approx([60.0, 60.0])
    assert id(cap3) != id(cap2), "Map def replacement should produce new cached array"

    # reset_cache should clear position cap too
    fusion.reset_cache()
    cap4 = fusion._position_cap()
    assert cap4.tolist() == pytest.approx([60.0, 60.0])

    # Verify next_obs() also respects refreshed caps
    cap_via_obs = fusion.next_obs()["map"]["size"]
    np.testing.assert_array_equal(cap_via_obs, np.array([60.0, 60.0], dtype=np.float32))
