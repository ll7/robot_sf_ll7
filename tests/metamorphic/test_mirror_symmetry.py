"""Mirror and rotation symmetry for deterministic planner-driven robot episodes.

Release arms (``social_force``, ``orca``, hybrid v3, and ``risk_dwa``) run
through the map-runner policy builder. Guarded-PPO also has a direct
flat-observation frame relation because its world-frame safety rollout is
deterministic, while the learned primary policy is not an exact scene-trace
oracle. Exact trace relations apply only to deterministic planners.
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
from pysocialforce.config import (
    SOCIAL_FORCE_KERNEL_LEGACY_UNWRAPPED_V1,
    SOCIAL_FORCE_KERNEL_WRAPPED_V2,
)

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.planner import socnav
from robot_sf.planner.guarded_ppo import GuardedPPOAdapter
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter
from robot_sf.planner.socnav_base import SocNavPlannerConfig
from robot_sf.planner.visibility_planner import PlannerConfig, VisibilityPlanner
from robot_sf.sim.pedestrian_model_variants import _pairwise_social_force_kernel
from tests.metamorphic.planner_arms import (
    HYBRID_V3_ARM,
    HYBRID_V4_DIAGNOSTIC_ARM,
    ArmEpisode,
    interaction_scene,
    mirror_x,
    mirror_y,
    release_arm,
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


def _flat_frame_observation(
    *,
    robot: tuple[float, float],
    heading: float,
    goal: tuple[float, float],
    pedestrian: tuple[float, float],
    pedestrian_velocity_ego: tuple[float, float],
) -> dict[str, object]:
    """Build a flat observation with the producer's ego velocity contract."""
    return {
        "robot_position": np.asarray(robot, dtype=float),
        "robot_heading": np.asarray([heading], dtype=float),
        "robot_speed": np.asarray([0.0], dtype=float),
        "robot_radius": np.asarray([0.25], dtype=float),
        "goal_current": np.asarray(goal, dtype=float),
        "goal_next": np.asarray(goal, dtype=float),
        "pedestrians_positions": np.asarray([pedestrian], dtype=float),
        "pedestrians_velocities": np.asarray([pedestrian_velocity_ego], dtype=float),
        "pedestrians_count": np.asarray([1.0], dtype=float),
        "pedestrians_radius": np.asarray([0.25], dtype=float),
        "sim_timestep": np.asarray([0.1], dtype=float),
    }


def _extracted_pedestrian_velocity(arm: str, observation: dict[str, object]) -> np.ndarray:
    """Extract one adapter's world-frame pedestrian velocity from a flat scene."""
    if arm == "risk_dwa":
        return RiskDWAPlannerAdapter()._extract_robot_goal_ped(observation)[-1]
    if arm == "guarded_ppo":
        return GuardedPPOAdapter()._extract_state(observation)[-1]
    raise AssertionError(f"unsupported frame-test arm: {arm}")


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


@pytest.mark.parametrize(
    "arm",
    ["social_force", "orca"],
)
def test_release_arm_trace_is_mirror_and_rotation_equivariant(arm: str) -> None:
    """The corrected social-force successor and ORCA preserve transformed traces."""
    _requires_rvo2(arm)
    kernel_version = SOCIAL_FORCE_KERNEL_WRAPPED_V2 if arm == "social_force" else None
    base = run_arm_episode(
        arm,
        interaction_scene(),
        seed=_SEED,
        max_steps=_ARM_STEPS,
        social_force_kernel_version=kernel_version,
    )
    assert base.status == "ok"
    assert base.flat_socnav_observation
    assert len(base.commands) >= 20
    without_pedestrian = run_arm_episode(
        arm,
        interaction_scene(pedestrian=False),
        seed=_SEED,
        max_steps=_ARM_STEPS,
        social_force_kernel_version=kernel_version,
    )
    without_obstacle = run_arm_episode(
        arm,
        interaction_scene(obstacle=False),
        seed=_SEED,
        max_steps=_ARM_STEPS,
        social_force_kernel_version=kernel_version,
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
            arm,
            interaction_scene(point_map),
            seed=_SEED,
            max_steps=_ARM_STEPS,
            social_force_kernel_version=kernel_version,
        )
        assert transformed.flat_socnav_observation
        _assert_arm_equivariant(base, transformed, name)


@pytest.mark.parametrize(
    "arm",
    [
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield",
        "scenario_adaptive_hybrid_orca_v2_collision_guard",
    ],
)
def test_release_scenario_orca_override_trace_is_rotation_equivariant(arm: str) -> None:
    """The two leave-group overrides resolve to ORCA and rotate through the episode path."""
    _requires_rvo2("orca")
    scenario = "francis2023_leave_group"
    effective_algo, effective_config = release_arm(arm, scenario=scenario)
    assert effective_algo == "orca"
    assert effective_config["orca_symmetry_bias"] == pytest.approx(0.22)
    assert effective_config["orca_head_on_bias"] == pytest.approx(0.30)

    base = run_arm_episode(
        arm, interaction_scene(), seed=1001, max_steps=_ARM_STEPS, scenario=scenario
    )
    rotated = run_arm_episode(
        arm,
        interaction_scene(rotate_90),
        seed=1001,
        max_steps=_ARM_STEPS,
        scenario=scenario,
    )

    assert base.status == rotated.status == "ok"
    assert base.flat_socnav_observation
    assert rotated.flat_socnav_observation
    _assert_arm_equivariant(base, rotated, "rotate_90")


def test_risk_dwa_flat_release_trace_is_rotation_equivariant() -> None:
    """The deterministic Risk-DWA release arm rotates its flat scene and trace."""
    # Use development seed 1001 to keep this episode check within the 1001-1030 dev range.
    base = run_arm_episode("risk_dwa", interaction_scene(), seed=1001, max_steps=_ARM_STEPS)
    rotated = run_arm_episode(
        "risk_dwa", interaction_scene(rotate_90), seed=1001, max_steps=_ARM_STEPS
    )

    assert base.status == rotated.status == "ok"
    assert base.flat_socnav_observation
    assert rotated.flat_socnav_observation
    _assert_arm_equivariant(base, rotated, "rotate_90")


@pytest.mark.parametrize("arm", ["risk_dwa", "guarded_ppo"])
def test_flat_velocity_rotation_equivariance_for_world_frame_rollouts(arm: str) -> None:
    """World-frame rollouts agree for a 90-degree transformed flat scene."""
    heading = 0.37
    ego_velocity = np.asarray([0.8, -0.25], dtype=float)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    base_world_velocity = np.asarray(
        [
            cos_h * ego_velocity[0] - sin_h * ego_velocity[1],
            sin_h * ego_velocity[0] + cos_h * ego_velocity[1],
        ],
        dtype=float,
    )
    rotate = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    rotated_heading = heading + np.pi / 2.0
    rotated_world_velocity = rotate @ base_world_velocity
    rotated_ego_velocity = np.asarray(
        [
            np.cos(rotated_heading) * rotated_world_velocity[0]
            + np.sin(rotated_heading) * rotated_world_velocity[1],
            -np.sin(rotated_heading) * rotated_world_velocity[0]
            + np.cos(rotated_heading) * rotated_world_velocity[1],
        ],
        dtype=float,
    )
    base = _flat_frame_observation(
        robot=(3.0, 4.0),
        heading=heading,
        goal=(12.0, 4.0),
        pedestrian=(6.0, 5.0),
        pedestrian_velocity_ego=tuple(ego_velocity),
    )
    transformed = _flat_frame_observation(
        robot=tuple(rotate @ np.asarray([3.0, 4.0])),
        heading=rotated_heading,
        goal=tuple(rotate @ np.asarray([12.0, 4.0])),
        pedestrian=tuple(rotate @ np.asarray([6.0, 5.0])),
        pedestrian_velocity_ego=tuple(rotated_ego_velocity),
    )

    base_velocity = _extracted_pedestrian_velocity(arm, base)
    transformed_velocity = _extracted_pedestrian_velocity(arm, transformed)
    np.testing.assert_allclose(
        transformed_velocity,
        base_velocity @ rotate.T,
        rtol=0.0,
        atol=1e-12,
        err_msg=arm,
    )

    command = (0.45, 0.12)
    if arm == "risk_dwa":
        planner = RiskDWAPlannerAdapter()
        base_state = planner._extract_robot_goal_ped(base)
        transformed_state = planner._extract_robot_goal_ped(transformed)
        base_score = planner._rollout_score(
            robot_pos=base_state[0],
            heading=base_state[1],
            goal=base_state[2],
            command=command,
            ped_pos=base_state[3],
            ped_vel=base_state[4],
            observation=base,
            current_speed=0.0,
        )
        transformed_score = planner._rollout_score(
            robot_pos=transformed_state[0],
            heading=transformed_state[1],
            goal=transformed_state[2],
            command=command,
            ped_pos=transformed_state[3],
            ped_vel=transformed_state[4],
            observation=transformed,
            current_speed=0.0,
        )
        assert transformed_score == pytest.approx(base_score, rel=0.0, abs=1e-12)
    else:
        planner = GuardedPPOAdapter()
        base_evaluation = planner._evaluate_command(base, command)
        transformed_evaluation = planner._evaluate_command(transformed, command)
        assert transformed_evaluation["safe"] == base_evaluation["safe"]
        for key in ("progress", "min_ped_clear", "first_ped_clear", "min_obs_clear", "min_ttc"):
            assert transformed_evaluation[key] == pytest.approx(
                base_evaluation[key], rel=0.0, abs=1e-12
            )


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


@pytest.mark.parametrize(
    ("scenario", "escape_speed"),
    [("metamorphic", 0.3), ("francis2023_perpendicular_traffic", 0.6)],
)
def test_diagnostic_hybrid_v4_preserves_historical_controls(
    scenario: str, escape_speed: float
) -> None:
    """Diagnostic controls stay historical while scenario overrides still apply."""
    algo, config = release_arm(HYBRID_V4_DIAGNOSTIC_ARM, scenario=scenario)
    assert algo == "hybrid_rule_local_planner"
    assert config.get("physical_static_exclusion_enabled") is False
    assert config.get("goal_next_validity_enabled") is False
    assert config["static_clearance_escape_max_speed"] == pytest.approx(escape_speed)


def test_diagnostic_hybrid_v4_trace_is_mirror_equivariant() -> None:
    """The explicitly bound v4 diagnostic arm preserves reflected traces."""
    base = run_arm_episode(
        HYBRID_V4_DIAGNOSTIC_ARM, interaction_scene(), seed=_SEED, max_steps=_ARM_STEPS
    )
    assert base.status == "ok"
    for name in ("mirror_y", "mirror_x"):
        point_map = _TRANSFORMS[name][0]
        transformed = run_arm_episode(
            HYBRID_V4_DIAGNOSTIC_ARM,
            interaction_scene(point_map),
            seed=_SEED,
            max_steps=_ARM_STEPS,
        )
        _assert_arm_equivariant(base, transformed, name)


def test_social_force_pair_kernel_is_rotation_equivariant() -> None:
    """Rotating a robot-pedestrian pair rotates its force (release v2 kernel parameters)."""
    config = SocNavPlannerConfig()
    parameters = {
        "n": int(config.social_force_n),
        "n_prime": int(config.social_force_n_prime),
        "lambda_importance": float(config.social_force_lambda_importance),
        "gamma": float(config.social_force_gamma),
    }
    # A pedestrian 2 m straight ahead of a robot driving along +x, crossing it.
    position_difference = np.asarray([[-2.0, -0.05]])
    velocity_difference = np.asarray([[-1.2, 0.9]])
    legacy_implicit = _pairwise_social_force_kernel(
        position_difference,
        velocity_difference,
        **parameters,
    )
    legacy_explicit = _pairwise_social_force_kernel(
        position_difference,
        velocity_difference,
        kernel_version=SOCIAL_FORCE_KERNEL_LEGACY_UNWRAPPED_V1,
        **parameters,
    )
    np.testing.assert_array_equal(legacy_implicit, legacy_explicit)
    reference = None
    for angle in (0.0, 0.3, np.pi / 2.0, np.pi, -np.pi / 2.0):
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        force = _pairwise_social_force_kernel(
            position_difference @ rotation.T,
            velocity_difference @ rotation.T,
            kernel_version=SOCIAL_FORCE_KERNEL_WRAPPED_V2,
            **parameters,
        )
        unrotated = force @ rotation
        if reference is None:
            reference = unrotated
        np.testing.assert_allclose(
            unrotated, reference, rtol=0.0, atol=1e-9, err_msg=f"rotation {angle:.3f} rad"
        )
    assert np.linalg.norm(reference) > 1e-3, "the crossing pedestrian must exert a force"
