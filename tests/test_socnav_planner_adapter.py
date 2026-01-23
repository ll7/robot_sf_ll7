"""Tests for SocNavBench-inspired planner adapters."""

import numpy as np

from robot_sf.planner.socnav import (
    ORCAPlannerAdapter,
    SACADRLPlannerAdapter,
    SamplingPlannerAdapter,
    SocialForcePlannerAdapter,
    SocNavBenchComplexPolicy,
    SocNavBenchSamplingAdapter,
    SocNavPlannerConfig,
    SocNavPlannerPolicy,
    make_orca_policy,
    make_sacadrl_policy,
    make_social_force_policy,
)


def _make_obs(goal=(5.0, 0.0), heading=0.0):
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([heading], dtype=np.float32),
            "speed": np.array([0.0, 0.0], dtype=np.float32),
            "radius": np.array([0.5], dtype=np.float32),
        },
        "goal": {
            "current": np.array(goal, dtype=np.float32),
            "next": np.array([0.0, 0.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": np.zeros((1, 2), dtype=np.float32),
            "radius": np.array([0.4], dtype=np.float32),
            "count": np.array([0.0], dtype=np.float32),
        },
        "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
        "sim": {"timestep": np.array([0.1], dtype=np.float32)},
    }


def _with_occupancy_grid(
    obs: dict,
    *,
    obstacle_cells: list[tuple[int, int]] | None = None,
    resolution: float = 1.0,
    origin: tuple[float, float] = (-2.0, -2.0),
    size: tuple[float, float] = (4.0, 4.0),
):
    """Attach a minimal occupancy grid to the observation."""
    grid = np.zeros((4, 4, 4), dtype=np.float32)
    for row, col in obstacle_cells or []:
        if 0 <= row < grid.shape[1] and 0 <= col < grid.shape[2]:
            grid[0, row, col] = 1.0  # obstacles channel
    obs["occupancy_grid"] = grid
    obs["occupancy_grid_meta_origin"] = np.array(origin, dtype=np.float32)
    obs["occupancy_grid_meta_resolution"] = np.array([resolution], dtype=np.float32)
    obs["occupancy_grid_meta_size"] = np.array(size, dtype=np.float32)
    obs["occupancy_grid_meta_use_ego_frame"] = np.array([1.0], dtype=np.float32)
    obs["occupancy_grid_meta_channel_indices"] = np.array([0, 1, 2, 3], dtype=np.float32)
    return obs


def _make_obs_with_peds(
    ped_positions: list[tuple[float, float]],
    *,
    goal: tuple[float, float] = (5.0, 0.0),
    heading: float = 0.0,
):
    """Build an observation that includes the requested pedestrians."""
    obs = _make_obs(goal=goal, heading=heading)
    max_peds = max(1, len(ped_positions))
    positions = np.zeros((max_peds, 2), dtype=np.float32)
    if ped_positions:
        positions[: len(ped_positions)] = np.array(ped_positions, dtype=np.float32)
    obs["pedestrians"]["positions"] = positions
    obs["pedestrians"]["count"] = np.array([float(len(ped_positions))], dtype=np.float32)
    return obs


def test_sampling_adapter_moves_toward_goal():
    """Adapter moves forward when aligned with goal."""
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig(max_linear_speed=1.0, angular_gain=2.0))
    obs = _make_obs(goal=(5.0, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert v > 0.0
    assert abs(w) < 1e-6  # aligned with heading


def test_sampling_adapter_stops_within_tolerance():
    """Adapter stops when within goal tolerance."""
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig(goal_tolerance=0.5))
    obs = _make_obs(goal=(0.2, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert v == 0.0
    assert w == 0.0


def test_sampling_adapter_turns_toward_goal():
    """Adapter turns left when goal is on the left."""
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig(max_linear_speed=1.0, angular_gain=1.0))
    obs = _make_obs(goal=(0.0, 5.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert w > 0.0  # needs to turn left
    assert v >= 0.0


def test_policy_wrapper_calls_adapter():
    """Policy delegates to the underlying adapter."""
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig(max_linear_speed=1.0))
    policy = SocNavPlannerPolicy(adapter)
    obs = _make_obs(goal=(1.0, 0.0), heading=0.0)
    v, _w = policy.act(obs)
    assert v >= 0.0


def test_socnavbench_adapter_fallbacks():
    """SocNavBench adapter should fall back gracefully when upstream is unavailable."""
    adapter = SocNavBenchSamplingAdapter(SocNavPlannerConfig(max_linear_speed=0.5))
    obs = _make_obs(goal=(1.0, 0.0), heading=0.0)
    v, _w = adapter.plan(obs)
    assert v >= 0.0


def test_socnavbench_complex_policy_fallback():
    """Complex policy should still return an action even without upstream deps."""
    policy = SocNavBenchComplexPolicy()
    obs = _make_obs(goal=(1.0, 0.0), heading=0.0)
    v, _w = policy.act(obs)
    assert v >= 0.0


def test_social_force_adapter():
    """Social-force heuristic returns finite action."""
    adapter = SocialForcePlannerAdapter(SocNavPlannerConfig())
    obs = _make_obs(goal=(2.0, 0.0), heading=0.0)
    v, _w = adapter.plan(obs)
    assert v >= 0.0


def test_orca_adapter():
    """ORCA-like heuristic returns finite action."""
    adapter = ORCAPlannerAdapter(SocNavPlannerConfig())
    obs = _make_obs(goal=(2.0, 0.0), heading=0.0)
    v, _w = adapter.plan(obs)
    assert v >= 0.0


def test_orca_slowdown_with_head_on_pedestrian():
    """ORCA-like heuristic reduces speed for a head-on pedestrian."""
    adapter = ORCAPlannerAdapter(SocNavPlannerConfig())
    obs_free = _make_obs(goal=(5.0, 0.0), heading=0.0)
    v_free, _w_free = adapter.plan(obs_free)
    obs = _make_obs_with_peds([(2.0, 0.0)], goal=(5.0, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert v < v_free
    assert np.isfinite(w)


def test_orca_ignores_far_pedestrian():
    """ORCA-like heuristic keeps heading when pedestrians are outside the avoidance radius."""
    adapter = ORCAPlannerAdapter(SocNavPlannerConfig())
    obs = _make_obs_with_peds([(6.0, 0.0)], goal=(5.0, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert v > 0.0
    assert abs(w) < 1e-3


def test_orca_with_lateral_pedestrian_returns_bounded_action():
    """ORCA-like heuristic returns bounded action with a lateral pedestrian."""
    adapter = ORCAPlannerAdapter(SocNavPlannerConfig())
    obs = _make_obs_with_peds([(0.0, 2.0)], goal=(5.0, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert v <= adapter.config.max_linear_speed + 1e-6
    assert abs(w) <= adapter.config.max_angular_speed + 1e-6


def test_orca_responds_to_static_obstacle_in_grid():
    """ORCA should reduce speed or steer when a grid obstacle blocks the path."""
    adapter = ORCAPlannerAdapter(SocNavPlannerConfig(orca_obstacle_range=4.0))
    obs_free = _with_occupancy_grid(_make_obs(goal=(5.0, 0.0), heading=0.0))
    v_free, w_free = adapter.plan(obs_free)
    obs_blocked = _with_occupancy_grid(
        _make_obs(goal=(5.0, 0.0), heading=0.0),
        obstacle_cells=[(2, 3)],
    )
    v_blocked, w_blocked = adapter.plan(obs_blocked)
    assert v_blocked < v_free or abs(w_blocked) > abs(w_free) + 1e-3


def test_sacadrl_adapter():
    """SA-CADRL-like heuristic returns finite action."""
    adapter = SACADRLPlannerAdapter(SocNavPlannerConfig())
    obs = _make_obs(goal=(2.0, 0.0), heading=0.0)
    v, _w = adapter.plan(obs)
    assert v >= 0.0


def test_policy_constructors():
    """Factory helpers build policies without error."""
    obs = _make_obs(goal=(1.0, 0.0), heading=0.0)
    for factory in (make_social_force_policy, make_orca_policy, make_sacadrl_policy):
        policy = factory()
        v, _w = policy.act(obs)
        assert v >= 0.0
