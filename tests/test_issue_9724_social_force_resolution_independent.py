"""Regression coverage for the social-force planner obstacle term and speed limits (#9724).

The ``resolution_independent_v2`` planner version replaces the per-occupied-cell
obstacle sum with one repulsion per visible obstacle surface patch and caps
commands at the social-force desired speed.  The historical ``grid_cell_sum_v1``
path stays the default.
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.map_runner.map_runner import _run_map_episode
from robot_sf.planner.socnav import (
    SOCIAL_FORCE_PLANNER_LEGACY_V1,
    SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2,
    SocialForcePlannerAdapter,
    SocNavPlannerConfig,
)
from robot_sf.planner.socnav_base import resolve_social_force_planner_version
from robot_sf.training.scenario_loader import load_scenarios

V2 = SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2
ROBOT_RADIUS = 1.0
GRID_EXTENT = 16.0  # metres, ego grid spans [-8, 8] x [-8, 8]


def _observation(
    *,
    heading: float = 0.0,
    speed: float = 0.0,
    goal: tuple[float, float] = (20.0, 0.0),
    peds: np.ndarray | None = None,
    ped_vel: np.ndarray | None = None,
    dt: float = 0.1,
) -> dict:
    """Build a SocNav observation with the robot at the origin."""
    peds = np.zeros((0, 2), dtype=np.float32) if peds is None else peds
    ped_vel = np.zeros_like(peds) if ped_vel is None else ped_vel
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([heading], dtype=np.float32),
            "speed": np.array([speed, 0.0], dtype=np.float32),
            "radius": np.array([ROBOT_RADIUS], dtype=np.float32),
        },
        "goal": {
            "current": np.asarray(goal, dtype=np.float32),
            "next": np.asarray(goal, dtype=np.float32) + 1.0,
        },
        "pedestrians": {
            "positions": np.asarray(peds, dtype=np.float32),
            "velocities": np.asarray(ped_vel, dtype=np.float32),
            "count": np.array([float(len(peds))], dtype=np.float32),
        },
        "sim": {"timestep": np.array([dt], dtype=np.float32)},
    }


def _with_grid(observation: dict, resolution: float, occupied) -> dict:
    """Attach an ego-frame occupancy grid whose cells satisfy ``occupied(x, y)``."""
    cells = round(GRID_EXTENT / resolution)
    origin = -0.5 * GRID_EXTENT
    coords = origin + (np.arange(cells) + 0.5) * resolution
    xs, ys = np.meshgrid(coords, coords)  # rows index y, columns index x
    grid = np.zeros((3, cells, cells), dtype=np.float32)
    grid[0] = occupied(xs, ys).astype(np.float32)
    observation["occupancy_grid"] = grid
    observation["occupancy_grid_meta_origin"] = np.array([origin, origin], dtype=np.float32)
    observation["occupancy_grid_meta_resolution"] = np.array([resolution], dtype=np.float32)
    observation["occupancy_grid_meta_size"] = np.array([GRID_EXTENT, GRID_EXTENT], np.float32)
    observation["occupancy_grid_meta_use_ego_frame"] = np.array([1.0], dtype=np.float32)
    observation["occupancy_grid_meta_channel_indices"] = np.array([0, 1, 2], dtype=np.float32)
    return observation


def _wall_above(y0: float):
    """Return a predicate for a thick wall whose lower face is at ``y = y0``."""
    return lambda _xs, ys: ys >= y0


def _obstacle_force(version: str, resolution: float, occupied, **obs_kwargs) -> np.ndarray:
    adapter = SocialForcePlannerAdapter(SocNavPlannerConfig(social_force_planner_version=version))
    observation = _with_grid(_observation(**obs_kwargs), resolution, occupied)
    robot_state = observation["robot"]
    return adapter._compute_obstacle_force(
        observation, np.zeros(2), float(robot_state["heading"][0]), np.zeros(2), robot_state
    )


def test_planner_version_defaults_to_legacy_and_rejects_unknown() -> None:
    """The corrected planner is opt-in; malformed selectors fail closed."""
    assert SocNavPlannerConfig().social_force_planner_version == SOCIAL_FORCE_PLANNER_LEGACY_V1
    assert resolve_social_force_planner_version("") == SOCIAL_FORCE_PLANNER_LEGACY_V1
    assert resolve_social_force_planner_version(f" {V2} ") == V2
    with pytest.raises(ValueError, match="unsupported social-force planner version"):
        resolve_social_force_planner_version("grid_cell_sum_v9")
    with pytest.raises(TypeError):
        resolve_social_force_planner_version(2)


@pytest.mark.parametrize("surface_distance", [0.5, 1.0, 2.0])
def test_straight_wall_yields_one_exponential_term(surface_distance: float) -> None:
    """A straight wall contributes exactly one term A*exp(-d/B) along its normal."""
    config = SocNavPlannerConfig()
    force = _obstacle_force(V2, 0.1, _wall_above(ROBOT_RADIUS + surface_distance))
    expected = config.social_force_obstacle_v2_strength * np.exp(
        -surface_distance / config.social_force_obstacle_v2_length
    )
    assert force[1] < 0.0
    assert abs(force[0]) < 1e-6
    assert abs(force[1]) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "occupied",
    [
        _wall_above(2.0),
        # Wall rotated by 30 degrees, 2.1 m from the robot centre.
        lambda xs, ys: (-np.sin(np.pi / 6) * xs + np.cos(np.pi / 6) * ys) >= 2.1,
        # Disc obstacle ahead-left.
        lambda xs, ys: (xs - 2.5) ** 2 + (ys - 1.5) ** 2 <= 0.8**2,
    ],
)
def test_obstacle_force_is_invariant_when_grid_resolution_is_halved(occupied) -> None:
    """Halving the grid cell size leaves the v2 force unchanged within tolerance.

    Rasterisation alone places an oblique wall up to half a cell (0.1 m at 0.2 m
    cells) away from its true position; with the 0.6 m decay length that is an
    18 % magnitude change, so the tolerance is 25 %.  The v1 per-cell sum
    changes by far more (see the legacy test below).
    """
    coarse = _obstacle_force(V2, 0.2, occupied)
    fine = _obstacle_force(V2, 0.1, occupied)
    assert np.linalg.norm(coarse) > 0.05
    assert np.linalg.norm(fine - coarse) <= 0.25 * np.linalg.norm(fine)


def test_legacy_per_cell_sum_depends_on_resolution() -> None:
    """Document the #9724 defect: the v1 per-cell sum grows as cells shrink."""
    wall = _wall_above(3.0)
    coarse = _obstacle_force(SOCIAL_FORCE_PLANNER_LEGACY_V1, 0.2, wall)
    fine = _obstacle_force(SOCIAL_FORCE_PLANNER_LEGACY_V1, 0.1, wall)
    v2 = _obstacle_force(V2, 0.1, wall)
    assert np.linalg.norm(fine) > 1.3 * np.linalg.norm(coarse)
    assert np.linalg.norm(coarse) > 10.0 * np.linalg.norm(v2)


def test_corridor_centre_forces_cancel_and_off_centre_pushes_back() -> None:
    """Both corridor walls contribute, so the lateral force restores the centre line."""
    corridor = lambda _xs, ys: np.abs(ys) >= 2.0  # noqa: E731
    centre = _obstacle_force(V2, 0.1, corridor)
    assert np.linalg.norm(centre) < 1e-6

    shifted = lambda _xs, ys: np.abs(ys + 0.3) >= 2.0  # noqa: E731  robot 0.3 m above centre
    force = _obstacle_force(V2, 0.1, shifted)
    assert force[1] < 0.0


def test_obstacle_force_is_negligible_beyond_a_few_metres() -> None:
    """The v2 term decays to a negligible level a few metres from the surface."""
    far = _obstacle_force(V2, 0.1, _wall_above(ROBOT_RADIUS + 4.0))
    config = SocNavPlannerConfig()
    goal_force_scale = config.social_force_desired_speed / config.social_force_tau
    assert np.linalg.norm(far) * config.social_force_repulsion_weight < 0.01 * goal_force_scale


def test_v2_velocity_respects_desired_speed() -> None:
    """Commands never exceed v_des, even with a large force along the goal."""
    config = SocNavPlannerConfig(social_force_planner_version=V2)
    adapter = SocialForcePlannerAdapter(config)
    adapter._compute_social_force = lambda *_args: np.array([90.0, 0.0])
    observation = _observation(speed=0.9)
    velocity = adapter.plan_velocity_world(observation)
    assert np.linalg.norm(velocity) <= config.social_force_desired_speed + 1e-6
    linear, angular = adapter.plan(observation)
    assert 0.0 < linear <= config.social_force_desired_speed + 1e-6
    assert abs(angular) <= config.max_angular_speed

    legacy = SocialForcePlannerAdapter(SocNavPlannerConfig())
    legacy._compute_social_force = lambda *_args: np.array([90.0, 0.0])
    legacy_linear, _ = legacy.plan(_observation(speed=0.9))
    assert legacy_linear > 2.0  # historical path could command ~3 m/s for v_des 1 m/s


def test_v2_turns_in_place_when_net_force_points_backwards() -> None:
    """A backward net force turns in place (goal behind) or holds (goal ahead), never orbits."""
    config = SocNavPlannerConfig(social_force_planner_version=V2)
    adapter = SocialForcePlannerAdapter(config)
    adapter._compute_social_force = lambda *_args: np.array([-50.0, 0.5])
    linear, angular = adapter.plan(_observation(speed=0.2, goal=(-20.0, 1.0)))
    assert linear == pytest.approx(0.0, abs=1e-9)
    assert abs(angular) == pytest.approx(config.max_angular_speed)
    assert adapter.plan(_observation(speed=0.2)) == (0.0, 0.0)


def test_v2_pedestrian_repulsion_acts_on_an_approaching_pedestrian() -> None:
    """A pedestrian approaching on the goal line slows and deflects the robot.

    The v1 path passes ``v_robot - v_ped`` where the kernel expects
    ``v_ped - v_robot``; for this head-on case its force is essentially zero.
    """
    config = SocNavPlannerConfig(social_force_planner_version=V2)
    free = SocialForcePlannerAdapter(config).plan_velocity_world(_observation(speed=1.0))
    ped = np.array([[1.8, 0.1]], dtype=np.float32)
    ped_vel = np.array([[-1.0, 0.0]], dtype=np.float32)
    yielded = SocialForcePlannerAdapter(config).plan_velocity_world(
        _observation(speed=1.0, peds=ped, ped_vel=ped_vel)
    )
    assert yielded[0] < free[0] - 0.1
    assert yielded[1] < 0.0  # steers away from the pedestrian's side

    legacy = SocialForcePlannerAdapter(SocNavPlannerConfig())
    legacy_force = legacy._compute_social_force(
        np.zeros(2),
        np.array([1.0, 0.0]),
        _observation(peds=ped, ped_vel=ped_vel)["pedestrians"],
        0.0,
    )
    assert np.linalg.norm(legacy_force) < 0.01


def test_v2_metadata_names_the_geometry_convention() -> None:
    """Runtime metadata records the v2 version and its parameters."""
    adapter = SocialForcePlannerAdapter(SocNavPlannerConfig(social_force_planner_version=V2))
    adapter.plan(_with_grid(_observation(), 0.2, _wall_above(2.5)))
    diagnostics = adapter.diagnostics()
    assert diagnostics["planner_version"] == V2
    law = diagnostics["obstacle_force_law"]
    assert law["geometry_convention"] == "occupancy_visible_nearest_points"
    assert law["parameters"]["planner_version"] == V2
    assert law["parameters"]["visible_obstacle_terms"] == 1
    assert law["applied"] is True


def _disc(cx: float, cy: float, radius: float):
    return lambda xs, ys: (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2


def _terms(resolution: float, occupied, heading: float = 0.0):
    adapter = SocialForcePlannerAdapter(SocNavPlannerConfig(social_force_planner_version=V2))
    observation = _with_grid(_observation(heading=heading), resolution, occupied)
    return adapter._visible_obstacle_points(observation, np.zeros(2), heading)


@pytest.mark.parametrize("resolution", [0.2, 0.1])
def test_separate_post_in_front_of_wall_keeps_its_own_term(resolution: float) -> None:
    """A post near a wall is a separate obstacle and is never merged into the wall."""
    wall = _wall_above(2.4)
    post = _disc(1.0, 1.3, 0.3)
    points, _normals, distances = _terms(resolution, lambda xs, ys: wall(xs, ys) | post(xs, ys))
    assert points.shape[0] == 2
    # The wall term is the wall's foot point straight ahead, not a point off to the side.
    wall_term = points[np.argmax(distances)]
    assert wall_term[0] == pytest.approx(0.0, abs=resolution)
    assert wall_term[1] == pytest.approx(2.4, abs=resolution)


@pytest.mark.parametrize("resolution", [0.2, 0.1])
def test_row_of_columns_gives_one_term_per_column(resolution: float) -> None:
    """Each column of a row is its own obstacle; the row is not collapsed to one term."""
    columns = (-3.0, -1.5, 0.0, 1.5, 3.0)
    row = lambda xs, ys: np.any([_disc(x0, 1.8, 0.25)(xs, ys) for x0 in columns], axis=0)  # noqa: E731
    points, _normals, _distances = _terms(resolution, row)
    assert points.shape[0] == len(columns)
    assert sorted(np.round(points[:, 0] / 1.5).astype(int).tolist()) == [-2, -1, 0, 1, 2]


def test_connected_wall_is_one_patch_but_merges_are_bounded() -> None:
    """Merging happens only inside one connected region (documented behaviour)."""
    points, _normals, _distances = _terms(0.1, _wall_above(2.0))
    assert points.shape[0] == 1


@pytest.mark.parametrize("heading", [0.0, 0.7, 2.3])
def test_world_force_is_invariant_under_ego_frame_rotation(heading: float) -> None:
    """The same world geometry gives the same world force whatever the robot heading."""

    def world_scene(xw, yw):
        return (yw >= 2.2) | _disc(2.0, -1.0, 0.4)(xw, yw)

    def ego_scene(xe, ye):
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        return world_scene(cos_h * xe - sin_h * ye, sin_h * xe + cos_h * ye)

    reference = _obstacle_force(V2, 0.1, world_scene)
    rotated = _obstacle_force(V2, 0.1, ego_scene, heading=heading)
    assert np.linalg.norm(rotated - reference) <= 0.25 * np.linalg.norm(reference)


def test_v2_holds_only_when_pushed_back_with_the_goal_ahead() -> None:
    """A backward desired velocity holds only while the goal is in front."""
    config = SocNavPlannerConfig(social_force_planner_version=V2)
    adapter = SocialForcePlannerAdapter(config)
    assert adapter._unicycle_command_v2(0.05, 2.5, goal_ahead=True) == (0.0, 0.0)
    assert adapter._unicycle_command_v2(0.9, -2.5, goal_ahead=True) == (0.0, 0.0)
    linear, angular = adapter._unicycle_command_v2(0.05, 2.5, goal_ahead=False)
    assert linear == 0.0
    assert angular == pytest.approx(config.max_angular_speed)


def test_v2_turn_direction_has_hysteresis_behind_the_robot() -> None:
    """Beyond 150 deg the previous turn direction is kept, so turns do not flip each step."""
    config = SocNavPlannerConfig(social_force_planner_version=V2)
    adapter = SocialForcePlannerAdapter(config)
    _linear, first = adapter._unicycle_command_v2(0.5, 2.9, goal_ahead=False)
    _linear, second = adapter._unicycle_command_v2(0.5, -2.9, goal_ahead=False)
    assert first > 0.0
    assert second > 0.0  # kept the previous (positive) direction
    _linear, third = adapter._unicycle_command_v2(0.5, -1.0, goal_ahead=False)
    assert third < 0.0  # outside the hysteresis band the error decides again
    adapter.reset()
    _linear, after_reset = adapter._unicycle_command_v2(0.5, -2.9, goal_ahead=False)
    assert after_reset < 0.0


def _closed_loop(dt: float, world_occupied, steps: int, goal=(-10.0, 0.0)):
    """Integrate unicycle commands from rest, heading +x, with a world-fixed scene."""
    adapter = SocialForcePlannerAdapter(SocNavPlannerConfig(social_force_planner_version=V2))
    pos, heading, speed = np.zeros(2), 0.0, 0.0
    held = 0
    for _ in range(steps):
        cos_h, sin_h = np.cos(heading), np.sin(heading)

        def ego_scene(xe, ye, pos=pos, cos_h=cos_h, sin_h=sin_h):
            return world_occupied(
                pos[0] + cos_h * xe - sin_h * ye, pos[1] + sin_h * xe + cos_h * ye
            )

        observation = _with_grid(
            _observation(heading=heading, speed=speed, goal=tuple(np.asarray(goal) - pos), dt=dt),
            0.1,
            ego_scene,
        )
        linear, angular = adapter.plan(observation)
        held += int(linear == 0.0 and angular == 0.0)
        heading += angular * dt
        pos = pos + linear * dt * np.array([np.cos(heading), np.sin(heading)])
        speed = linear
    return pos, held


@pytest.mark.parametrize("dt", [0.05, 0.1])
def test_robot_at_rest_with_goal_behind_turns_and_drives(dt: float) -> None:
    """Open space, goal 10 m behind: the robot turns around and drives, at any dt."""
    pos, held = _closed_loop(dt, lambda xs, _ys: np.zeros_like(xs, dtype=bool), round(12.0 / dt))
    assert held == 0
    assert pos[0] < -5.0


def test_post_behind_the_robot_does_not_freeze_it() -> None:
    """A post between the robot and a goal behind it is passed, not a permanent hold."""
    post = _disc(-2.0, 0.3, 0.4)
    pos, _held = _closed_loop(0.1, post, 400)
    assert pos[0] < -6.0


V2_CONFIG_PATH = Path("configs/algos/social_force_resolution_independent_v2.yaml")
V2_CONFIG = yaml.safe_load(V2_CONFIG_PATH.read_text(encoding="utf-8"))
RELEASE_MATRIX = Path("configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml")


def _run(
    scenario_path: Path,
    scenario_id: str,
    seed: int,
    positions: list | None = None,
    commands: list | None = None,
) -> dict:
    scenario = next(
        dict(row) for row in load_scenarios(scenario_path) if row.get("name") == scenario_id
    )
    original = SocialForcePlannerAdapter.plan_velocity_world
    original_plan = SocialForcePlannerAdapter.plan

    def _record_plan(self, observation):
        command = original_plan(self, observation)
        if commands is not None:
            commands.append(command)
        return command

    def _record(self, observation):
        if positions is not None:
            robot_state, _goal, _peds = self._socnav_fields(observation)
            positions.append(np.asarray(robot_state["position"], dtype=float)[:2].copy())
        return original(self, observation)

    SocialForcePlannerAdapter.plan_velocity_world = _record
    SocialForcePlannerAdapter.plan = _record_plan
    try:
        return _run_map_episode(
            scenario,
            seed,
            horizon=500,
            dt=0.1,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo="social_force",
            scenario_path=scenario_path,
            algo_config=dict(V2_CONFIG),
            observation_level="tracked_agents_no_noise",
            record_simulation_step_trace=False,
        )
    finally:
        SocialForcePlannerAdapter.plan_velocity_world = original
        SocialForcePlannerAdapter.plan = original_plan


@pytest.mark.slow
def test_bottleneck_low_without_pedestrians_reaches_goal() -> None:
    """The pedestrian-free bottleneck is completed (v1 orbited for the whole horizon)."""
    record = _run(
        Path("configs/scenarios/archetypes/classic_bottleneck.yaml"),
        "classic_bottleneck_low",
        111,
    )
    assert record["outcome"] == {
        "route_complete": True,
        "collision_event": False,
        "timeout_event": False,
    }


@pytest.mark.slow
def test_group_crossing_seed_22_makes_monotone_progress_and_reaches_goal() -> None:
    """Group crossing seed 22 progresses every second of the first 10 s and completes."""
    positions: list[np.ndarray] = []
    record = _run(
        Path("configs/scenarios/archetypes/classic_group_crossing.yaml"),
        "classic_group_crossing_medium",
        22,
        positions,
    )
    assert record["outcome"]["route_complete"] is True
    assert record["outcome"]["collision_event"] is False
    final_position = positions[-1]
    distances = [float(np.linalg.norm(positions[i] - final_position)) for i in range(0, 101, 10)]
    assert all(later < earlier for earlier, later in pairwise(distances))


@pytest.mark.slow
def test_release_matrix_bottleneck_low_seed_112_enters_goal_zone() -> None:
    """The shipped v2 config completes under goal_zone_entry_v1.

    With terminal_goal_v1 enabled the robot stopped 1.75 m before the final
    waypoint, outside the goal zone, and timed out on this seed.
    """
    assert V2_CONFIG == {"social_force_planner_version": V2}
    record = _run(RELEASE_MATRIX, "classic_bottleneck_low", 112)
    assert record["outcome"]["route_complete"] is True


@pytest.mark.slow
def test_narrow_doorway_seed_111_does_not_spin_in_place() -> None:
    """A blocked robot holds instead of spinning with a turn sign flip every step.

    Before the hold/hysteresis rule, 234 of 400 commands were (0, +-max turn)
    with 55 sign flips and mean curvature about 330.
    """
    commands: list[tuple[float, float]] = []
    record = _run(RELEASE_MATRIX, "francis2023_narrow_doorway", 111, commands=commands)
    turns = [angular for _linear, angular in commands]
    flips = sum(1 for a, b in pairwise(turns) if a * b < 0.0 and abs(a) > 0.5 and abs(b) > 0.5)
    spinning = sum(1 for linear, angular in commands if linear < 0.05 and abs(angular) > 0.9)
    assert flips == 0
    assert spinning <= 10
    assert record["metrics"]["curvature_mean"] < 1.0
