"""Mirror symmetry for deterministic planner-driven robot episodes."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.planner.visibility_planner import PlannerConfig, VisibilityPlanner
from tests.metamorphic.test_pedestrian_removal import (
    _MAP_SIZE,
    _run_social_force_episode,
    _short_navigation_scene,
)

_SEED = 8244
_MAX_STEPS = 60
_TRACE_ATOL = 1e-5


def _visibility_map(
    *, mirrored: bool
) -> tuple[MapDefinition, tuple[float, float], tuple[float, float]]:
    """Build a continuous path-planning fixture with all scene points reflected together."""

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        return (point[0], _MAP_SIZE - point[1]) if mirrored else point

    start = transform((2.0, 7.0))
    goal = transform((18.0, 7.0))
    spawn = tuple(transform(point) for point in ((2.0, 7.0), (2.5, 7.0), (2.0, 7.5)))
    goal_zone = tuple(transform(point) for point in ((17.5, 7.0), (18.0, 7.0), (17.5, 7.5)))
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[start, goal],
        spawn_zone=spawn,  # type: ignore[arg-type]
        goal_zone=goal_zone,  # type: ignore[arg-type]
    )
    obstacle_vertices = [(8.0, 6.0), (10.0, 6.0), (10.0, 9.0), (8.0, 9.0)]
    pedestrian = SinglePedestrianDefinition(
        id="reflected",
        start=transform((4.0, 4.0)),
        goal=transform((5.0, 4.0)),
        speed_m_s=1.0,
    )
    map_def = MapDefinition(
        width=_MAP_SIZE,
        height=_MAP_SIZE,
        obstacles=[Obstacle([transform(point) for point in obstacle_vertices])],
        robot_spawn_zones=[spawn],
        ped_spawn_zones=[],
        robot_goal_zones=[goal_zone],
        bounds=[
            (transform((0.0, 0.0)), transform((_MAP_SIZE, 0.0))),
            (transform((_MAP_SIZE, 0.0)), transform((_MAP_SIZE, _MAP_SIZE))),
            (transform((_MAP_SIZE, _MAP_SIZE)), transform((0.0, _MAP_SIZE))),
            (transform((0.0, _MAP_SIZE)), transform((0.0, 0.0))),
        ],
        robot_routes=[route],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=[pedestrian],
    )
    return map_def, start, goal


def _assert_mirrored_trajectory(
    base: tuple[tuple[float, float, float], ...],
    mirrored: tuple[tuple[float, float, float], ...],
) -> None:
    """Compare position and heading traces under horizontal reflection."""
    base_trace = np.asarray(base, dtype=float)
    mirrored_trace = np.asarray(mirrored, dtype=float)
    assert base_trace.shape == mirrored_trace.shape
    expected = base_trace.copy()
    expected[:, 1] = _MAP_SIZE - expected[:, 1]
    np.testing.assert_allclose(mirrored_trace[:, :2], expected[:, :2], rtol=0.0, atol=_TRACE_ATOL)
    heading_delta = np.arctan2(
        np.sin(mirrored_trace[:, 2] + base_trace[:, 2]),
        np.cos(mirrored_trace[:, 2] + base_trace[:, 2]),
    )
    np.testing.assert_allclose(heading_delta, 0.0, rtol=0.0, atol=_TRACE_ATOL)


def test_social_force_episode_mirrors_map_route_and_robot_trace() -> None:
    """The actual deterministic planner preserves outcome and trace after reflection."""
    base_map = _short_navigation_scene(pedestrian_present=True)
    mirrored_map = _short_navigation_scene(pedestrian_present=True, mirrored=True)
    base = _run_social_force_episode(base_map, seed=_SEED, max_steps=_MAX_STEPS)
    mirrored = _run_social_force_episode(mirrored_map, seed=_SEED, max_steps=_MAX_STEPS)
    no_pedestrian = _run_social_force_episode(
        _short_navigation_scene(pedestrian_present=False),
        seed=_SEED,
        max_steps=_MAX_STEPS,
    )
    no_obstacle = _run_social_force_episode(
        _short_navigation_scene(pedestrian_present=True, obstacle_present=False),
        seed=_SEED,
        max_steps=_MAX_STEPS,
    )

    assert base.pedestrian_count == mirrored.pedestrian_count == 1
    assert not base.fallback and base.fallback_count == 0, base.fallback_reason
    assert not mirrored.fallback and mirrored.fallback_count == 0, mirrored.fallback_reason
    assert (base.success, base.collision, base.truncated) == (
        mirrored.success,
        mirrored.collision,
        mirrored.truncated,
    )
    assert base.success and not base.collision and not base.truncated
    _assert_mirrored_trajectory(base.positions, mirrored.positions)

    base_commands = np.asarray(base.commands, dtype=float)
    mirrored_commands = np.asarray(mirrored.commands, dtype=float)
    assert base_commands.shape == mirrored_commands.shape
    expected_commands = base_commands.copy()
    expected_commands[:, 1] *= -1.0
    np.testing.assert_allclose(
        mirrored_commands,
        expected_commands,
        rtol=0.0,
        atol=_TRACE_ATOL,
    )

    base_actions = np.asarray(base.actions, dtype=float)
    mirrored_actions = np.asarray(mirrored.actions, dtype=float)
    assert base_actions.shape == mirrored_actions.shape
    expected_actions = base_actions.copy()
    expected_actions[:, 1] *= -1.0
    np.testing.assert_allclose(
        mirrored_actions,
        expected_actions,
        rtol=0.0,
        atol=_TRACE_ATOL,
    )

    assert np.linalg.norm(np.subtract(base.commands[0], no_pedestrian.commands[0])) > 1e-6, (
        "the nearby pedestrian should exert a measurable planner interaction"
    )
    assert np.linalg.norm(np.subtract(base.commands[0], no_obstacle.commands[0])) > 1e-6, (
        "the nearby obstacle should exert a measurable planner interaction"
    )


def test_visibility_planner_path_mirrors_obstacle_map_and_route() -> None:
    """A second deterministic planner returns the reflected continuous path."""
    base_map, base_start, base_goal = _visibility_map(mirrored=False)
    mirrored_map, mirrored_start, mirrored_goal = _visibility_map(mirrored=True)
    config = PlannerConfig(
        robot_radius=0.2,
        min_safe_clearance=0.2,
        enable_smoothing=False,
        fallback_on_failure=False,
    )
    base_path = VisibilityPlanner(base_map, config).plan(base_start, base_goal)
    mirrored_path = VisibilityPlanner(mirrored_map, config).plan(mirrored_start, mirrored_goal)

    assert len(base_path) == len(mirrored_path) > 2
    expected = np.asarray([(x, _MAP_SIZE - y) for x, y in base_path], dtype=float)
    np.testing.assert_allclose(
        np.asarray(mirrored_path, dtype=float),
        expected,
        rtol=0.0,
        atol=_TRACE_ATOL,
    )
    base_length = float(np.sum(np.linalg.norm(np.diff(base_path, axis=0), axis=1)))
    mirrored_length = float(np.sum(np.linalg.norm(np.diff(mirrored_path, axis=0), axis=1)))
    assert mirrored_length == pytest.approx(base_length, rel=0.0, abs=_TRACE_ATOL)
