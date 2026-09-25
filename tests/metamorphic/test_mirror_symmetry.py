"""Mirror and rotation symmetry for deterministic planner-driven robot episodes.

Release arms (``social_force``, ``orca``, hybrid v3) run through the map-runner
policy builder. Exact trace relations apply only to deterministic planners.
Stochastic planners (sampling, MPPI, stochastic learned policies) draw different
samples in a transformed scene even with the same seed, so they are excluded from
exact mirror relations; their contract is seeded replay (``test_replay_determinism``)
and, in the cluster tier, distribution-level agreement over seeds.

A sign error that is itself a reflection (for example a flipped pedestrian ``vy``)
commutes with every mirror, so the 90-degree rotation relation is required to see it.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.planner import socnav
from robot_sf.planner.visibility_planner import PlannerConfig, VisibilityPlanner
from tests.metamorphic.planner_arms import (
    HYBRID_V3_ARM,
    ArmEpisode,
    interaction_scene,
    mirror_x,
    mirror_y,
    rotate_90,
    run_arm_episode,
)
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
    assert (base.success, base.collision, base.step_limit_reached) == (
        mirrored.success,
        mirrored.collision,
        mirrored.step_limit_reached,
    )
    assert base.success and not base.collision and not base.step_limit_reached
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


_ARM_STEPS = 60
_ARM_ATOL = 1e-4
# name -> (point map, heading map, angular-command sign)
_TRANSFORMS = {
    "mirror_y": (mirror_y, lambda heading: -heading, -1.0),
    "mirror_x": (mirror_x, lambda heading: np.pi - heading, -1.0),
    "rotate_90": (rotate_90, lambda heading: heading + np.pi / 2.0, 1.0),
}


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _assert_arm_equivariant(base: ArmEpisode, other: ArmEpisode, name: str) -> None:
    """Compare a transformed release-arm episode with the transformed base episode."""
    point_map, heading_map, angular_sign = _TRANSFORMS[name]
    assert (base.success, base.collision, base.step_limit_reached) == (
        other.success,
        other.collision,
        other.step_limit_reached,
    ), name
    base_poses = np.asarray(base.poses, dtype=float)
    other_poses = np.asarray(other.poses, dtype=float)
    assert base_poses.shape == other_poses.shape, name
    expected_xy = np.asarray([point_map(tuple(xy)) for xy in base_poses[:, :2]])
    np.testing.assert_allclose(
        other_poses[:, :2], expected_xy, rtol=0.0, atol=_ARM_ATOL, err_msg=f"{name} positions"
    )
    np.testing.assert_allclose(
        _wrap(other_poses[:, 2] - heading_map(base_poses[:, 2])),
        0.0,
        rtol=0.0,
        atol=_ARM_ATOL,
        err_msg=f"{name} headings",
    )
    expected_commands = np.asarray(base.commands, dtype=float) * np.asarray([1.0, angular_sign])
    np.testing.assert_allclose(
        np.asarray(other.commands, dtype=float),
        expected_commands,
        rtol=0.0,
        atol=_ARM_ATOL,
        err_msg=f"{name} planner commands",
    )


def _requires_rvo2(arm: str) -> None:
    if arm == "orca" and socnav.rvo2 is None:
        pytest.skip("rvo2 is required for the native ORCA release arm")


@pytest.mark.parametrize("arm", ["social_force", "orca"])
def test_release_arm_trace_is_mirror_and_rotation_equivariant(arm: str) -> None:
    """Release arms return the transformed trace for y-mirror, x-mirror and a 90-degree turn."""
    _requires_rvo2(arm)
    base = run_arm_episode(arm, interaction_scene(), seed=_SEED, max_steps=_ARM_STEPS)
    assert base.status == "ok"
    assert len(base.commands) >= 20
    without_pedestrian = run_arm_episode(
        arm, interaction_scene(pedestrian=False), seed=_SEED, max_steps=_ARM_STEPS
    )
    without_obstacle = run_arm_episode(
        arm, interaction_scene(obstacle=False), seed=_SEED, max_steps=_ARM_STEPS
    )
    shared = min(len(base.commands), len(without_pedestrian.commands))
    assert (
        np.max(np.abs(np.subtract(base.commands[:shared], without_pedestrian.commands[:shared])))
        > 1e-3
    ), "the crossing pedestrian must change the planner's commands"
    shared = min(len(base.commands), len(without_obstacle.commands))
    assert (
        np.max(np.abs(np.subtract(base.commands[:shared], without_obstacle.commands[:shared])))
        > 1e-3
    ), "the side obstacle must change the planner's commands"
    velocities = np.diff(np.asarray(base.pedestrian_positions, dtype=float)[:, 0, :], axis=0)
    assert np.all(np.abs(velocities[:10]).min(axis=0) > 1e-3), (
        "the pedestrian must move along both axes so each mirror flips a velocity component"
    )

    for name, (point_map, _heading_map, _sign) in _TRANSFORMS.items():
        transformed = run_arm_episode(
            arm, interaction_scene(point_map), seed=_SEED, max_steps=_ARM_STEPS
        )
        _assert_arm_equivariant(base, transformed, name)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "#9733 finding: hybrid v3 picks between near-tied discrete candidates, so a "
        "reflected float32 scene selects a different turn rate (0.0 vs 0.3 rad/s) "
        "within ten steps; tracked as a follow-up to #9733"
    ),
)
def test_release_hybrid_v3_trace_is_mirror_equivariant() -> None:
    """The hybrid v3 arm should return the reflected trace like the continuous arms."""
    base = run_arm_episode(HYBRID_V3_ARM, interaction_scene(), seed=_SEED, max_steps=_ARM_STEPS)
    for name in ("mirror_y", "mirror_x"):
        point_map = _TRANSFORMS[name][0]
        transformed = run_arm_episode(
            HYBRID_V3_ARM, interaction_scene(point_map), seed=_SEED, max_steps=_ARM_STEPS
        )
        _assert_arm_equivariant(base, transformed, name)


def test_release_hybrid_v3_outcome_is_mirror_and_rotation_invariant() -> None:
    """Whatever the trace does, reflection or rotation must not change the outcome."""
    outcomes = {}
    for name, point_map in (("base", None), *((n, t[0]) for n, t in _TRANSFORMS.items())):
        scene = interaction_scene() if point_map is None else interaction_scene(point_map)
        episode = run_arm_episode(HYBRID_V3_ARM, scene, seed=_SEED, max_steps=150)
        assert episode.status == "ok"
        outcomes[name] = (episode.success, episode.collision, episode.step_limit_reached)
    assert set(outcomes.values()) == {(True, False, False)}, outcomes
