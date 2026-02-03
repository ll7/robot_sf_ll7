"""Tests for SocNavBench-inspired planner adapters."""

from pathlib import Path

import numpy as np
import pytest

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
    velocities = np.zeros((max_peds, 2), dtype=np.float32)
    if ped_positions:
        positions[: len(ped_positions)] = np.array(ped_positions, dtype=np.float32)
    obs["pedestrians"]["positions"] = positions
    obs["pedestrians"]["velocities"] = velocities
    obs["pedestrians"]["count"] = np.array([float(len(ped_positions))], dtype=np.float32)
    return obs


def _orca_fallback_adapter(
    monkeypatch, config: SocNavPlannerConfig | None = None
) -> ORCAPlannerAdapter:
    """Create an ORCA adapter forced into heuristic fallback for deterministic tests."""
    from robot_sf.planner import socnav

    monkeypatch.setattr(socnav, "rvo2", None)
    return ORCAPlannerAdapter(config or SocNavPlannerConfig(), allow_fallback=True)


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
    adapter = SocNavBenchSamplingAdapter(
        SocNavPlannerConfig(max_linear_speed=0.5),
        socnav_root=Path("does_not_exist"),
        allow_fallback=True,
    )
    obs = _make_obs(goal=(1.0, 0.0), heading=0.0)
    v, _w = adapter.plan(obs)
    assert v >= 0.0


def test_socnavbench_complex_policy_fallback():
    """Complex policy should still return an action even without upstream deps."""
    policy = SocNavBenchComplexPolicy(
        socnav_root=Path("does_not_exist"),
        allow_fallback=True,
    )
    obs = _make_obs(goal=(1.0, 0.0), heading=0.0)
    v, _w = policy.act(obs)
    assert v >= 0.0


def test_socnavbench_adapter_requires_upstream():
    """Adapter should raise when upstream planner is required but missing."""
    with pytest.raises(FileNotFoundError):
        SocNavBenchSamplingAdapter(socnav_root=Path("does_not_exist"))


def test_social_force_adapter():
    """Social-force heuristic returns finite action."""
    adapter = SocialForcePlannerAdapter(SocNavPlannerConfig())
    obs = _make_obs(goal=(2.0, 0.0), heading=0.0)
    v, _w = adapter.plan(obs)
    assert v >= 0.0


def test_social_force_adapter_responds_to_pedestrian():
    """Social-force adapter slows or turns when pedestrians are in the path."""
    cfg = SocNavPlannerConfig(social_force_repulsion_weight=2.0)
    adapter = SocialForcePlannerAdapter(cfg)
    obs_free = _make_obs(goal=(5.0, 0.0), heading=0.0)
    v_free, w_free = adapter.plan(obs_free)
    obs_ped = _make_obs_with_peds([(1.0, 0.0)], goal=(5.0, 0.0), heading=0.0)
    v_ped, w_ped = adapter.plan(obs_ped)
    assert v_ped < v_free or abs(w_ped) > abs(w_free) + 1e-3


def test_social_force_adapter_responds_to_obstacle_in_grid():
    """Social-force adapter reacts to nearby occupancy-grid obstacles."""
    cfg = SocNavPlannerConfig(social_force_obstacle_range=4.0)
    adapter = SocialForcePlannerAdapter(cfg)
    obs_free = _with_occupancy_grid(_make_obs(goal=(5.0, 0.0), heading=0.0))
    v_free, w_free = adapter.plan(obs_free)
    obs_blocked = _with_occupancy_grid(
        _make_obs(goal=(5.0, 0.0), heading=0.0),
        obstacle_cells=[(2, 3)],
    )
    v_blocked, w_blocked = adapter.plan(obs_blocked)
    assert v_blocked < v_free or abs(w_blocked) > abs(w_free) + 1e-3


def test_orca_adapter(monkeypatch):
    """ORCA-like heuristic returns finite action."""
    adapter = _orca_fallback_adapter(monkeypatch)
    obs = _make_obs(goal=(2.0, 0.0), heading=0.0)
    v, _w = adapter.plan(obs)
    assert v >= 0.0


def test_orca_slowdown_with_head_on_pedestrian(monkeypatch):
    """ORCA-like heuristic reduces speed for a head-on pedestrian."""
    adapter = _orca_fallback_adapter(monkeypatch)
    obs_free = _make_obs(goal=(5.0, 0.0), heading=0.0)
    v_free, _w_free = adapter.plan(obs_free)
    obs = _make_obs_with_peds([(2.0, 0.0)], goal=(5.0, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert v < v_free
    assert np.isfinite(w)


def test_orca_ignores_far_pedestrian(monkeypatch):
    """ORCA-like heuristic keeps heading when pedestrians are outside the avoidance radius."""
    adapter = _orca_fallback_adapter(monkeypatch)
    obs = _make_obs_with_peds([(6.0, 0.0)], goal=(5.0, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert v > 0.0
    assert abs(w) < 1e-3


def test_orca_with_lateral_pedestrian_returns_bounded_action(monkeypatch):
    """ORCA-like heuristic returns bounded action with a lateral pedestrian."""
    adapter = _orca_fallback_adapter(monkeypatch)
    obs = _make_obs_with_peds([(0.0, 2.0)], goal=(5.0, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert v <= adapter.config.max_linear_speed + 1e-6
    assert abs(w) <= adapter.config.max_angular_speed + 1e-6


def test_orca_responds_to_static_obstacle_in_grid(monkeypatch):
    """ORCA should reduce speed or steer when a grid obstacle blocks the path."""
    adapter = _orca_fallback_adapter(monkeypatch, SocNavPlannerConfig(orca_obstacle_range=4.0))
    obs_free = _with_occupancy_grid(_make_obs(goal=(5.0, 0.0), heading=0.0))
    v_free, w_free = adapter.plan(obs_free)
    obs_blocked = _with_occupancy_grid(
        _make_obs(goal=(5.0, 0.0), heading=0.0),
        obstacle_cells=[(2, 3)],
    )
    v_blocked, w_blocked = adapter.plan(obs_blocked)
    assert v_blocked < v_free or abs(w_blocked) > abs(w_free) + 1e-3


def test_orca_adapter_requires_rvo2_when_fallback_disabled(monkeypatch):
    """ORCA adapter should fail fast without fallback if rvo2 is missing."""
    from robot_sf.planner import socnav

    monkeypatch.setattr(socnav, "rvo2", None)
    adapter = ORCAPlannerAdapter(SocNavPlannerConfig(), allow_fallback=False)
    obs = _make_obs(goal=(2.0, 0.0), heading=0.0)
    with pytest.raises(RuntimeError, match="rvo2"):
        adapter.plan(obs)


def test_sacadrl_adapter(monkeypatch):
    """SA-CADRL adapter can fall back when the model is unavailable (guards tests without TF)."""

    def _boom(self):
        raise RuntimeError("missing model")

    monkeypatch.setattr(SACADRLPlannerAdapter, "_build_model", _boom)
    adapter = SACADRLPlannerAdapter(SocNavPlannerConfig(), allow_fallback=True)
    obs = _make_obs(goal=(2.0, 0.0), heading=0.0)
    v, _w = adapter.plan(obs)
    assert v >= 0.0


def test_sacadrl_adapter_requires_model_when_fallback_disabled(monkeypatch):
    """SA-CADRL adapter fails fast without fallback to prevent silent heuristic use."""
    adapter = SACADRLPlannerAdapter(SocNavPlannerConfig(), allow_fallback=False)

    def _boom(self):
        raise RuntimeError("missing model")

    monkeypatch.setattr(SACADRLPlannerAdapter, "_build_model", _boom)
    obs = _make_obs(goal=(2.0, 0.0), heading=0.0)
    with pytest.raises(RuntimeError, match="missing model"):
        adapter.plan(obs)


def test_policy_constructors():
    """Factory helpers build policies without error."""
    obs = _make_obs(goal=(1.0, 0.0), heading=0.0)
    for factory in (make_social_force_policy,):
        policy = factory()
        v, _w = policy.act(obs)
        assert v >= 0.0
    orca_policy = make_orca_policy(allow_fallback=True)
    v, _w = orca_policy.act(obs)
    assert v >= 0.0
    sacadrl_policy = make_sacadrl_policy(allow_fallback=True)
    assert isinstance(sacadrl_policy.adapter, SACADRLPlannerAdapter)
