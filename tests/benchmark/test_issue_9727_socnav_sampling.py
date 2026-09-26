"""Regressions for the opt-in ``bounded_v2`` sampling heuristic (issues #9727, #9746).

Every test that asserts ``bounded_v2`` behaviour fails on the historical planner: either
the selector does not exist there, or the legacy command shows the defect (see the
``legacy`` assertions that document it).
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.planner.socnav import SamplingPlannerAdapter, SocNavPlannerConfig
from robot_sf.planner.socnav_base import (
    SOCNAV_SAMPLING_BOUNDED_V2,
    SOCNAV_SAMPLING_LEGACY_V1,
    resolve_socnav_sampling_version,
)
from robot_sf.planner.socnav_sampling_v2 import (
    GoalPathField,
    _ObstacleClearance,
    _rollout,
    braking_speed_limit,
    plan_bounded_v2,
    stopping_distance,
)

V2_CONFIG = Path("configs/algos/socnav_sampling_bounded_v2.yaml")
ROBOT_RADIUS = 1.0
PED_RADIUS = 0.4
DRIVE = {"max_linear_speed": 2.0, "max_linear_decel": 1.0, "max_linear_accel": 1.0}
GRID_RES = 0.2
GRID_CELLS = 160


def _adapter(version: str | None = None, **overrides) -> SamplingPlannerAdapter:
    config = SocNavPlannerConfig(socnav_sampling_version=version, **overrides)
    return SamplingPlannerAdapter(config)


def _bound(adapter: SamplingPlannerAdapter) -> SamplingPlannerAdapter:
    robot_config = SimpleNamespace(radius=ROBOT_RADIUS, **DRIVE)
    adapter.bind_env(SimpleNamespace(env_config=SimpleNamespace(robot_config=robot_config)))
    return adapter


def _observation(
    peds=(),
    goal=(10.0, 0.0),
    position=(0.0, 0.0),
    heading=0.0,
    wall=None,
    speed=0.0,
) -> dict:
    peds = np.asarray(peds, dtype=float).reshape(-1, 2)
    obs = {
        "robot": {
            "position": np.asarray(position, dtype=float),
            "heading": np.array([heading]),
            "radius": np.array([ROBOT_RADIUS]),
            "speed": np.array([speed]),
        },
        "goal": {"current": np.asarray(goal, dtype=float)},
        "pedestrians": {
            "positions": peds,
            "count": np.array([len(peds)]),
            "radius": np.array([PED_RADIUS]),
        },
        "sim": {"timestep": np.array([0.1])},
    }
    if wall is not None:
        origin = np.array([-16.0, -16.0])
        centres = -16.0 + (np.arange(GRID_CELLS) + 0.5) * GRID_RES
        xs, ys = np.meshgrid(centres, centres)  # rows are y, columns are x
        occupied = wall(xs, ys).astype(float)
        grid = np.stack([occupied, np.zeros_like(occupied), occupied])
        obs["occupancy_grid"] = grid
        obs["occupancy_grid_meta"] = {
            "origin": origin,
            "resolution": [GRID_RES],
            "size": [GRID_CELLS * GRID_RES, GRID_CELLS * GRID_RES],
            "use_ego_frame": [0.0],
            "center_on_robot": [0.0],
            "channel_indices": [0, 1, -1, 2],
            "robot_pose": [position[0], position[1], heading],
        }
    return obs


def _drive(heading, speed0, target_heading, target_speed, seconds=2.0, dt=0.05):
    """Hold one (speed, heading) command under the release drive limits.

    Speed changes at most 1 m/s^2 and is capped at 2 m/s; the heading controller turns at
    ``clip(error, -1, 1)`` rad/s, as the heuristic's angular gain of 1 does.
    """
    x = y = 0.0
    theta, v = float(heading), float(speed0)
    target_speed = min(float(target_speed), DRIVE["max_linear_speed"])
    points = []
    for _ in range(int(seconds / dt)):
        v += max(-dt, min(dt, target_speed - v))
        err = math.atan2(math.sin(target_heading - theta), math.cos(target_heading - theta))
        theta += max(-1.0, min(1.0, err)) * dt
        x += v * dt * math.cos(theta)
        y += v * dt * math.sin(theta)
        points.append((x, y))
    return np.asarray(points)


def test_default_stays_legacy_and_release_config_opts_in() -> None:
    """No selector keeps the historical planner; the versioned config selects v2."""
    assert SocNavPlannerConfig().socnav_sampling_version == SOCNAV_SAMPLING_LEGACY_V1
    assert resolve_socnav_sampling_version(" ") == SOCNAV_SAMPLING_LEGACY_V1
    with pytest.raises(ValueError, match="unsupported socnav sampling version"):
        resolve_socnav_sampling_version("bounded_v3")
    payload = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))
    assert payload["socnav_sampling_version"] == SOCNAV_SAMPLING_BOUNDED_V2
    assert payload["sampling_braking_envelope"] is False
    assert payload["sampling_pedestrian_prediction"] is False
    assert SocNavPlannerConfig().sampling_pedestrian_prediction is False
    assert "socnav_sampling_version" not in _adapter().diagnostics()
    assert _adapter("bounded_v2").diagnostics()["socnav_sampling_version"] == "bounded_v2"


def test_distant_crowd_keeps_heading_near_goal() -> None:
    """Issue #9727: a crowd about 5 m away must not reverse the route heading."""
    pedestrians = [[10.0 + i * 0.1, 10.0 + i * 0.05] for i in range(24)]
    obs = _observation(pedestrians, goal=(7.0, 10.0), position=(4.8, 10.6))
    goal_bearing = math.atan2(10.0 - 10.6, 7.0 - 4.8)

    adapter = _adapter("bounded_v2")
    linear, _angular = adapter.plan(obs)

    desired = adapter._last_sampling_v2["desired_heading"]
    error = abs(math.atan2(math.sin(desired - goal_bearing), math.cos(desired - goal_bearing)))
    assert math.degrees(error) <= 50.0
    assert linear > 1.0
    # Legacy creeps at 0.07 m/s while turning toward a heading about 174 deg off the goal.
    legacy_linear, _ = _adapter().plan(obs)
    assert legacy_linear < 0.5


FAR_CROWD = [[-6.0 - 0.2 * i, 4.0 + 0.1 * i] for i in range(24)]


@pytest.mark.parametrize(
    ("label", "peds"),
    [
        ("0.5 m dead ahead", [[0.5, 0.05]]),
        ("0.4 m at -10 deg", [[0.4 * math.cos(-0.17), 0.4 * math.sin(-0.17)]]),
        ("0.7 m at +20 deg", [[0.7 * math.cos(0.35), 0.7 * math.sin(0.35)]]),
        (
            "far crowd plus 0.6 m at +15 deg",
            [*FAR_CROWD, [0.6 * math.cos(0.26), 0.6 * math.sin(0.26)]],
        ),
    ],
)
def test_close_pedestrian_avoidance_not_weaker_than_legacy(label: str, peds) -> None:
    """Near-field repulsion stays uncapped: turn at least as hard, drive no faster."""
    obs = _observation(peds)
    legacy_v, legacy_w = _adapter().plan(obs)
    v2_v, v2_w = _adapter("bounded_v2").plan(obs)

    assert v2_v <= legacy_v + 1e-9, label
    assert math.copysign(1.0, v2_w) == math.copysign(1.0, legacy_w), label
    assert abs(v2_w) >= abs(legacy_w) - 1e-9, label


def test_footprint_sweep_keeps_the_driven_arc_off_the_wall() -> None:
    """Issue #9746: the swept footprint along the driven arc, not a centre line, must stay clear.

    The robot drives at 2 m/s toward a wall at 60 deg while the goal lies almost straight
    ahead. The centre line along the goal heading is free, so legacy keeps full speed;
    turning at 1 rad/s at that speed swings the 1 m footprint into the wall.
    """
    wall_y = 2.2

    def wall(xs, ys):
        return (ys >= wall_y) & (ys <= wall_y + 0.2) & (xs >= -2.0) & (xs <= 14.0)

    heading = math.radians(60.0)
    obs = _observation(goal=(10.0, 0.5), heading=heading, wall=wall, speed=2.0)

    legacy_v, _ = _adapter().plan(obs)
    legacy_path = _drive(heading, 2.0, math.atan2(0.5, 10.0), legacy_v)
    assert legacy_path[:, 1].max() + ROBOT_RADIUS > wall_y  # the defect: footprint hits

    adapter = _bound(_adapter("bounded_v2"))
    v2_v, _ = adapter.plan(obs)
    path = _drive(heading, 2.0, adapter._last_sampling_v2["desired_heading"], v2_v)
    assert path[:, 1].max() + ROBOT_RADIUS < wall_y
    assert v2_v < legacy_v


def test_speed_near_wall_respects_drive_limit_and_braking_envelope() -> None:
    """Issue #9746: near a wall, speed must allow a stop before the swept surface."""

    def wall(xs, ys):
        return (xs >= 3.0) & (xs <= 3.2) & (np.abs(ys) <= 1.2)

    obs = _observation(goal=(20.0, 0.0), wall=wall)
    legacy_v, _ = _adapter().plan(obs)
    assert legacy_v > DRIVE["max_linear_speed"]  # the defect: 3 m/s, above the drive limit

    core = _bound(_adapter("bounded_v2"))
    core_v, _ = core.plan(obs)
    assert core_v <= DRIVE["max_linear_speed"]
    assert core_v < legacy_v

    braking = _bound(_adapter("bounded_v2", sampling_braking_envelope=True))
    brake_v, _ = braking.plan(obs)
    decision = braking._last_sampling_v2
    assert brake_v <= braking_speed_limit(decision["free_distance"], 1.0, 0.1, 0.1) + 1e-9
    if abs(decision["desired_heading"]) < 1e-6:
        # Straight at the wall: the stop must fit in the 2 m surface gap.
        assert stopping_distance(brake_v, 1.0, 0.1, 0.0) <= 2.0


def test_bound_drive_limit_caps_open_space_speed() -> None:
    """The cap comes from the bound drive, not the planner's 3 m/s default."""
    obs = _observation(goal=(30.0, 0.0))
    assert _adapter().plan(obs)[0] == pytest.approx(3.0)
    assert _adapter("bounded_v2").plan(obs)[0] == pytest.approx(3.0)  # unbound: config cap
    assert _bound(_adapter("bounded_v2")).plan(obs)[0] == pytest.approx(2.0)


def test_braking_helpers_are_consistent() -> None:
    """The braking bound inverts the stopping distance."""
    for free in (0.5, 1.0, 2.3, 4.0):
        speed = braking_speed_limit(free, 1.0, 0.1, 0.1)
        assert stopping_distance(speed, 1.0, 0.1, 0.1) == pytest.approx(free)
    assert braking_speed_limit(0.05, 1.0, 0.1, 0.1) == 0.0


@pytest.mark.slow
@pytest.mark.parametrize(
    "scenario_id", ["francis2023_crowd_navigation", "francis2023_robot_crowding"]
)
def test_release_cell_replay_reaches_goal_without_wall_contact(scenario_id: str) -> None:
    """Seed 111 of both cells ends in wall contact under legacy_v1 (#9727, #9746)."""
    from robot_sf.benchmark.classic_interactions_loader import (
        load_classic_matrix,
        select_scenario,
    )
    from robot_sf.benchmark.map_runner.map_runner import _run_map_episode

    scenario_path = Path(
        "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml"
    )
    scenario = select_scenario(load_classic_matrix(str(scenario_path)), scenario_id)
    record = _run_map_episode(
        scenario,
        111,
        horizon=600,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="socnav_sampling",
        scenario_path=scenario_path,
        algo_config_path=str(V2_CONFIG),
    )
    assert record["metrics"]["wall_collisions"] == 0
    assert record["termination_reason"] == "success"


def test_obstacle_clearance_handles_frames_and_missing_payloads() -> None:
    """Clearance is a lower bound in world and ego frames; off-grid counts as blocked."""
    adapter = _adapter("bounded_v2")
    assert not _ObstacleClearance(adapter, _observation()).available
    assert np.isinf(_ObstacleClearance(adapter, _observation())(np.zeros((1, 2))))

    obs = _observation(wall=lambda xs, ys: (xs >= 3.0) & (xs <= 3.2))
    clear = _ObstacleClearance(adapter, obs)
    values = clear(np.array([[0.0, 0.0], [2.0, 0.0], [40.0, 0.0]]))
    assert 2.7 <= values[0] <= 3.0
    assert values[1] < values[0]
    assert values[2] == 0.0  # off the grid

    empty = _ObstacleClearance(adapter, _observation(wall=lambda xs, ys: xs > 1e9))
    assert np.isinf(empty(np.zeros((1, 2))))[0]

    ego = _observation(wall=lambda xs, ys: (xs >= 3.0) & (xs <= 3.2), position=(5.0, 5.0))
    ego["occupancy_grid_meta"]["use_ego_frame"] = [1.0]
    ego["occupancy_grid_meta"]["robot_pose"] = [5.0, 5.0, math.pi / 2]
    # Ego x (forward) maps to world +y when the robot faces +y.
    ahead = _ObstacleClearance(adapter, ego)(np.array([[5.0, 7.0], [5.0, 5.0]]))
    assert ahead[0] < ahead[1]

    no_channel = _observation(wall=lambda xs, ys: xs > 0)
    no_channel["occupancy_grid_meta"]["channel_indices"] = [-1, 1, -1, -1]
    assert not _ObstacleClearance(adapter, no_channel).available
    bad_res = _observation(wall=lambda xs, ys: xs > 0)
    bad_res["occupancy_grid_meta"]["resolution"] = [0.0]
    assert not _ObstacleClearance(adapter, bad_res).available


def test_rollout_respects_drive_limits() -> None:
    """Speed changes by at most accel*dt per step; turning is rate limited."""
    points, travelled = _rollout(
        np.zeros(2), 0.0, 0.0, math.pi / 2, 5.0, 2.0, 0.1, (1.0, 1.0, 1.0, 1.0)
    )
    assert points.shape == (20, 2)
    steps = np.diff(np.concatenate(([0.0], travelled)))
    assert steps[0] == pytest.approx(0.01)
    assert np.all(np.diff(steps) <= 0.01 + 1e-9)
    _stopped, dist = _rollout(np.zeros(2), 0.0, 2.0, 0.0, 0.0, 3.0, 0.1, (1.0, 1.0, 1.0, 1.0))
    assert dist[-1] == pytest.approx(sum(max(0.0, 2.0 - 0.1 * k) * 0.1 for k in range(1, 31)))


def test_bind_env_edge_cases() -> None:
    """Binding tolerates missing configs, bad values and absent or broken geometry."""
    adapter = _adapter("bounded_v2")
    adapter.bind_env(SimpleNamespace())
    assert not hasattr(adapter, "_sampling_drive_limits")

    robot = SimpleNamespace(radius=1.0, max_linear_speed=True, max_linear_decel=-1.0)
    adapter.bind_env(SimpleNamespace(config=SimpleNamespace(robot_config=robot)))
    assert adapter._sampling_drive_limits == {"radius": 1.0}
    assert adapter._sampling_obstacle_segments is None

    def broken():
        return [("a", "b")]

    sim = SimpleNamespace(iter_obstacle_segments=broken)
    adapter.bind_env(SimpleNamespace(env_config=SimpleNamespace(robot_config=robot), simulator=sim))
    assert adapter._sampling_obstacle_segments is None

    sim = SimpleNamespace(iter_obstacle_segments=lambda: [((0.0, 0.0), (1.0, 0.0))])
    adapter.bind_env(SimpleNamespace(env_config=SimpleNamespace(robot_config=robot), simulator=sim))
    assert adapter._sampling_obstacle_segments.shape == (1, 4)

    legacy = _adapter()
    legacy.bind_env(SimpleNamespace(env_config=SimpleNamespace(robot_config=robot), simulator=sim))
    assert legacy._sampling_obstacle_segments is None  # legacy never reads map geometry
    diagnostics = adapter.diagnostics()
    assert diagnostics["drive_limits"] == {"radius": 1.0}
    assert diagnostics["braking_envelope"] is False
    assert diagnostics["pedestrian_prediction"] is False


def _wall_with_gap() -> np.ndarray:
    """A wall along x = 5 from y = -10 to 10 with a 3 m gap at y in [6, 9]."""
    return np.array(
        [
            [5.0, -10.0, 5.0, 6.0],
            [5.0, 9.0, 5.0, 10.0],
            [-5.0, -10.0, 15.0, -10.0],
            [-5.0, 10.0, 15.0, 10.0],
            [-5.0, -10.0, -5.0, 10.0],
            [15.0, -10.0, 15.0, 10.0],
        ]
    )


def test_goal_path_field_routes_through_the_gap() -> None:
    """Path distance goes around the wall, as the reference planner's FMM cost does."""
    field = GoalPathField(_wall_with_gap(), np.array([10.0, 0.0]), 1.0, 0.2, 1.5)
    waypoint = field.waypoint(np.array([0.0, 0.0]), 2.0)
    assert waypoint is not None
    assert waypoint[1] > 1.0  # heads up toward the gap, not straight into the wall
    assert field.waypoint(np.array([100.0, 0.0]), 2.0) is None  # off the field
    # Next to the wall the robot centre sits in the inflated band; it still gets a path.
    assert field.waypoint(np.array([4.5, 0.0]), 2.0) is not None
    blocked = GoalPathField(
        np.array([[4.0, -20.0, 4.0, 20.0]]), np.array([4.0, 0.0]), 1.0, 0.2, 0.5
    )
    assert not np.isfinite(blocked.distance).any()
    assert blocked.waypoint(np.array([0.0, 0.0]), 2.0) is None


def test_bound_map_steers_around_the_wall() -> None:
    """With map geometry bound, v2 aims for the gap; unbound, it aims straight."""
    robot = SimpleNamespace(radius=ROBOT_RADIUS, **DRIVE)
    sim = SimpleNamespace(
        iter_obstacle_segments=lambda: [((a, b), (c, d)) for a, b, c, d in _wall_with_gap()]
    )
    env = SimpleNamespace(env_config=SimpleNamespace(robot_config=robot), simulator=sim)
    obs = _observation(goal=(10.0, 0.0))
    bound = _adapter("bounded_v2")
    bound.bind_env(env)
    bound.plan(obs)
    assert bound._last_sampling_v2["path_distance"] is True
    assert bound._last_sampling_v2["base_heading"] > 0.3
    bound.plan(obs)  # cached field on the second call
    assert len(bound._sampling_path_fields) == 1
    off = _adapter("bounded_v2", sampling_path_distance=False)
    off.bind_env(env)
    off.plan(obs)
    assert off._last_sampling_v2["path_distance"] is False
    assert abs(off._last_sampling_v2["base_heading"]) < 1e-6


def _boxed(speed: float) -> dict:
    """Robot boxed in on three sides with the goal straight ahead through the wall."""

    def wall(xs, ys):
        return ((xs >= 1.1) & (xs <= 1.3) & (np.abs(ys) <= 3.0)) | (
            (np.abs(ys) >= 1.1) & (np.abs(ys) <= 1.3) & (xs >= -3.0) & (xs <= 1.3)
        )

    return _observation(goal=(10.0, 0.0), wall=wall, speed=speed)


def test_all_blocked_moving_robot_brakes_straight() -> None:
    """Every sample blocked while moving: brake without turning."""
    adapter = _bound(_adapter("bounded_v2"))
    assert adapter.plan(_boxed(1.0)) == (0.0, 0.0)
    assert adapter._last_sampling_v2["reason"] == "brake_straight"


def test_all_blocked_stopped_robot_turns_toward_the_open_side() -> None:
    """Every forward sample blocked while stopped: turn in place toward the exit."""
    adapter = _bound(_adapter("bounded_v2"))
    linear, angular = adapter.plan(_boxed(0.0))
    assert linear == 0.0
    assert adapter._last_sampling_v2["reason"] == "turn_in_place"
    assert abs(adapter._last_sampling_v2["desired_heading"]) > math.pi / 2
    assert abs(angular) == pytest.approx(1.0)


def test_all_blocked_stopped_robot_without_exit_holds() -> None:
    """Enclosed on every side: hold still."""

    def ring(xs, ys):
        r = np.hypot(xs, ys)
        return (r >= 1.05) & (r <= 1.3)

    adapter = _bound(_adapter("bounded_v2"))
    assert adapter.plan(_observation(goal=(10.0, 0.0), wall=ring)) == (0.0, 0.0)
    assert adapter._last_sampling_v2["reason"] == "hold"


def test_goal_reached_and_empty_sample_set() -> None:
    """Guards: at the goal, and with no speed fractions configured."""
    adapter = _adapter("bounded_v2")
    assert plan_bounded_v2(adapter, _observation(goal=(0.1, 0.0))) == (0.0, 0.0)
    assert adapter._last_sampling_v2 == {"reason": "goal_reached"}
    empty = _adapter("bounded_v2", sampling_speed_fractions=())
    assert empty.plan(_observation()) == (0.0, 0.0)
    assert empty._last_sampling_v2 == {"reason": "no_samples"}
    with pytest.raises(TypeError):
        resolve_socnav_sampling_version(2)


def test_approaching_pedestrian_blocks_earlier_with_prediction() -> None:
    """A pedestrian walking at the robot is swept where it will be, not where it is."""
    obs = _observation(peds=[[4.0, 0.0]], speed=1.5)
    obs["pedestrians"]["velocities"] = np.array([[-1.5, 0.0]])  # robot frame, heading 0
    predicted = _bound(_adapter("bounded_v2", sampling_pedestrian_prediction=True))
    static = _bound(_adapter("bounded_v2"))
    v_pred, _ = predicted.plan(obs)
    v_static, _ = static.plan(obs)
    assert v_pred < v_static
    obs["pedestrians"]["velocities"] = np.array([[np.nan, 0.0]])  # ignored when invalid
    assert predicted.plan(obs)[0] == pytest.approx(v_static)


def test_social_force_and_sampling_version_selectors_coexist() -> None:
    """One ``__setattr__`` resolves both the #9724 and the #9727/#9746 selectors."""
    from robot_sf.planner.socnav_base import (
        SOCIAL_FORCE_PLANNER_LEGACY_V1,
        SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2,
    )

    config = SocNavPlannerConfig(
        social_force_planner_version=f" {SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2} ",
        socnav_sampling_version=" bounded_v2 ",
    )
    assert config.social_force_planner_version == SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2
    assert config.socnav_sampling_version == SOCNAV_SAMPLING_BOUNDED_V2
    default = SocNavPlannerConfig()
    assert default.social_force_planner_version == SOCIAL_FORCE_PLANNER_LEGACY_V1
    assert default.socnav_sampling_version == SOCNAV_SAMPLING_LEGACY_V1
    with pytest.raises(ValueError, match="unsupported social-force planner version"):
        SocNavPlannerConfig(social_force_planner_version="grid_cell_sum_v9")
    with pytest.raises(ValueError, match="unsupported socnav sampling version"):
        SocNavPlannerConfig(socnav_sampling_version="bounded_v9")
    # The selectors take effect in their planners.
    from robot_sf.planner.socnav import SocialForcePlannerAdapter

    sf = SocialForcePlannerAdapter(config)
    assert sf.diagnostics()["planner_version"] == SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2
    sampler = SamplingPlannerAdapter(config)
    sampler.plan(_observation())
    assert sampler._last_sampling_v2["desired_heading"] == pytest.approx(0.0)
    assert sampler.diagnostics()["socnav_sampling_version"] == SOCNAV_SAMPLING_BOUNDED_V2
