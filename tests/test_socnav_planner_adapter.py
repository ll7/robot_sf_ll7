"""Tests for SocNavBench-inspired planner adapters."""

from pathlib import Path

import numpy as np
import pytest

from robot_sf.planner import socnav as _socnav_module
from robot_sf.planner.socnav import (
    HRVOPlannerAdapter,
    ORCAPlannerAdapter,
    PredictionPlannerAdapter,
    SACADRLPlannerAdapter,
    SamplingPlannerAdapter,
    SocialForcePlannerAdapter,
    SocNavBenchComplexPolicy,
    SocNavBenchSamplingAdapter,
    SocNavPlannerConfig,
    SocNavPlannerPolicy,
    TrivialReferencePlannerAdapter,
    _sacadrl_session_config,
    make_hrvo_policy,
    make_orca_policy,
    make_prediction_policy,
    make_sacadrl_policy,
    make_social_force_policy,
)


def _make_obs(goal=(5.0, 0.0), heading=0.0):
    """Build a minimal SocNav observation with configurable goal and heading."""
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


def test_trivial_reference_adapter_is_deterministic_and_bounded():
    """Reference adapter should document the minimal deterministic planner contract."""
    adapter = TrivialReferencePlannerAdapter(
        SocNavPlannerConfig(max_linear_speed=0.75, max_angular_speed=0.5)
    )
    obs = _make_obs(goal=(5.0, 0.0), heading=0.0)

    assert adapter.plan(obs) == adapter.plan(obs)
    v, w = adapter.plan(_make_obs(goal=(0.0, 5.0), heading=0.0))
    assert 0.0 <= v <= 0.75
    assert -0.5 <= w <= 0.5

    adapter.reset(seed=123)
    assert adapter.diagnostics()["steps"] == 0
    assert TrivialReferencePlannerAdapter._wrap_angle(5 * np.pi) == pytest.approx(np.pi)


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
        obstacle_cells=[(1, 3), (2, 3)],
    )
    v_blocked, w_blocked = adapter.plan(obs_blocked)
    assert v_blocked < v_free or abs(w_blocked) > abs(w_free) + 1e-3


def test_hrvo_adapter_returns_finite_action():
    """HRVO adapter should emit a bounded unicycle command."""
    adapter = HRVOPlannerAdapter(SocNavPlannerConfig(max_linear_speed=1.0, max_angular_speed=1.2))
    obs = _make_obs_with_peds([(1.6, 0.0)], goal=(5.0, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    vx, vy = adapter.plan_velocity_world(obs)
    assert 0.0 <= v <= adapter.config.max_linear_speed + 1e-6
    assert abs(w) <= adapter.config.max_angular_speed + 1e-6
    assert np.isfinite(vx)
    assert np.isfinite(vy)
    assert abs(vy) > 1e-4


def test_hrvo_builds_hybrid_apex_distinct_from_vo_and_rvo():
    """HRVO should build an asymmetric apex instead of plain VO or midpoint RVO."""
    adapter = HRVOPlannerAdapter(SocNavPlannerConfig(max_linear_speed=1.0, hrvo_time_horizon=4.0))
    obs = _make_obs_with_peds([(1.5, 0.0)], goal=(5.0, 0.0), heading=0.0)
    obs["pedestrians"]["velocities"][0] = np.array([-0.8, 0.0], dtype=np.float32)
    robot_state, goal_state, ped_state = adapter._socnav_fields(obs)
    robot_pos = np.asarray(robot_state["position"], dtype=float)
    robot_heading = float(np.asarray(robot_state["heading"], dtype=float)[0])
    goal = np.asarray(goal_state["current"], dtype=float)
    preferred_velocity = adapter._ego_to_world(
        adapter._preferred_velocity(
            goal,
            robot_pos,
            robot_heading,
            float(adapter.config.max_linear_speed),
        ),
        robot_heading,
    )
    ped_positions, ped_velocities, _ped_count, ped_radius = adapter._extract_pedestrians(ped_state)
    ped_vel_world = ped_velocities.astype(float)
    obstacles = adapter._build_hrvo_obstacles(
        robot_velocity_world=np.zeros(2, dtype=float),
        preferred_velocity_world=preferred_velocity,
        other_positions=ped_positions - robot_pos[None, :],
        other_velocities_world=ped_vel_world,
        other_pref_velocities_world=ped_vel_world.copy(),
        robot_radius=float(np.asarray(robot_state.get("radius", [0.3]), dtype=float)[0]),
        other_radii=np.full((ped_positions.shape[0],), float(ped_radius), dtype=float),
        time_step=0.1,
    )
    assert len(obstacles) == 1
    vo_apex = ped_vel_world[0]
    rvo_apex = 0.5 * ped_vel_world[0]
    assert not np.allclose(obstacles[0].apex, vo_apex)
    assert not np.allclose(obstacles[0].apex, rvo_apex)
    assert abs(float(obstacles[0].apex[1])) > 1e-4


def test_make_hrvo_policy_wraps_hrvo_adapter():
    """Convenience constructor should expose the HRVO adapter."""
    policy = make_hrvo_policy(SocNavPlannerConfig(max_linear_speed=0.9))
    obs = _make_obs_with_peds([(1.2, 0.0)], goal=(4.0, 0.0), heading=0.0)
    v, w = policy.act(obs)
    assert v >= 0.0
    assert np.isfinite(w)


def test_hrvo_responds_to_static_obstacle_in_grid():
    """HRVO should extract occupied grid cells into static obstacle constraints."""
    adapter = HRVOPlannerAdapter(SocNavPlannerConfig(max_linear_speed=1.0, orca_obstacle_range=4.0))
    obs_blocked = _with_occupancy_grid(
        _make_obs(goal=(5.0, 0.0), heading=0.0),
        obstacle_cells=[(1, 3), (2, 3)],
    )
    centers, radii = adapter._extract_obstacles_from_grid(
        obs_blocked,
        np.array([0.0, 0.0], dtype=float),
        0.0,
    )
    assert centers.shape[0] > 0
    assert centers.shape[0] == radii.shape[0]


def test_hrvo_coalesces_adjacent_static_obstacle_cells():
    """Adjacent occupied cells should be reduced before entering the HRVO solve."""
    adapter = HRVOPlannerAdapter(
        SocNavPlannerConfig(
            max_linear_speed=1.0,
            orca_obstacle_range=4.0,
            orca_obstacle_max_points=10,
        )
    )
    obs = _with_occupancy_grid(
        _make_obs(goal=(5.0, 0.0), heading=0.0),
        obstacle_cells=[(2, 1), (2, 2), (2, 3)],
    )
    centers, radii = adapter._extract_obstacles_from_grid(
        obs,
        np.array([0.0, 0.0], dtype=float),
        0.0,
    )
    assert centers.shape[0] < 3
    assert centers.shape[0] == radii.shape[0]
    assert centers.shape[0] > 0


def test_hrvo_extracts_bound_exact_static_obstacle_points():
    """HRVO should use bound exact obstacle geometry even without an occupancy grid."""
    adapter = HRVOPlannerAdapter(
        SocNavPlannerConfig(
            max_linear_speed=1.0,
            orca_obstacle_range=4.0,
            orca_obstacle_max_points=10,
        )
    )
    obs = _make_obs(goal=(5.0, 0.0), heading=0.0)
    adapter.bind_static_obstacle_points(
        np.array([[1.5, 0.0], [1.75, 0.0], [2.0, 0.0]], dtype=float),
        spacing=0.25,
    )
    centers, radii = adapter._extract_obstacles_from_grid(
        obs,
        np.array([0.0, 0.0], dtype=float),
        0.0,
    )
    assert centers.shape[0] > 0
    assert centers.shape[0] == radii.shape[0]
    assert np.all(centers[:, 0] > 1.0)


def test_orca_head_on_bias_breaks_straight_symmetry(monkeypatch):
    """Head-on bias should inject a turn instead of preserving a straight collision course."""
    adapter = _orca_fallback_adapter(
        monkeypatch,
        SocNavPlannerConfig(
            orca_head_on_bias=0.5,
            orca_symmetry_bias=0.3,
            orca_commit_distance=2.5,
        ),
    )
    obs = _make_obs_with_peds([(1.1, 0.0)], goal=(5.0, 0.0), heading=0.0)
    _v, w = adapter.plan(obs)
    assert abs(w) > 1e-3


def test_orca_stall_commit_bias_turns_when_forward_corridor_is_blocked(monkeypatch):
    """Repeated low-progress blocked steps should trigger persistent side commitment."""
    adapter = _orca_fallback_adapter(
        monkeypatch,
        SocNavPlannerConfig(
            orca_stall_cycles_before_commit=1,
            orca_commit_persistence_steps=4,
            orca_commit_lateral_gain=0.7,
            orca_side_probe_offset=0.6,
            orca_forward_probe_distance=1.0,
        ),
    )
    obs = _with_occupancy_grid(
        _make_obs(goal=(5.0, 0.0), heading=0.0),
        obstacle_cells=[(2, 3), (2, 2)],
    )
    obs["robot"]["speed"] = np.array([0.0], dtype=np.float32)
    _v1, w1 = adapter.plan(obs)
    _v2, w2 = adapter.plan(obs)
    assert abs(w1) > 1e-3 or abs(w2) > 1e-3
    assert adapter._commit_side_ttl > 0


def test_orca_reset_clears_commit_and_stall_state(monkeypatch):
    """Episode reset should clear ORCA's sticky stall/commit state."""
    adapter = _orca_fallback_adapter(monkeypatch)
    adapter._stall_cycles = 3
    adapter._last_goal_distance = 1.4
    adapter._commit_side = -1
    adapter._commit_side_ttl = 7

    adapter.reset()

    assert adapter._stall_cycles == 0
    assert adapter._last_goal_distance is None
    assert adapter._commit_side == 0
    assert adapter._commit_side_ttl == 0


def test_orca_adapter_requires_rvo2_when_fallback_disabled(monkeypatch):
    """ORCA adapter should fail fast without fallback if rvo2 is missing."""
    from robot_sf.planner import socnav

    monkeypatch.setattr(socnav, "rvo2", None)
    adapter = ORCAPlannerAdapter(SocNavPlannerConfig(), allow_fallback=False)
    obs = _make_obs(goal=(2.0, 0.0), heading=0.0)
    with pytest.raises(RuntimeError, match="rvo2"):
        adapter.plan(obs)


def test_orca_adapter_uses_rvo2_when_available():
    """ORCA adapter should execute the rvo2 path when the dependency is installed."""
    from robot_sf.planner import socnav

    if socnav.rvo2 is None:
        pytest.skip("rvo2 not installed; skipping real ORCA path check.")

    adapter = ORCAPlannerAdapter(SocNavPlannerConfig(), allow_fallback=False)
    obs = _make_obs_with_peds([(2.0, 0.0)], goal=(5.0, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert 0.0 <= v <= adapter.config.max_linear_speed + 1e-6
    assert abs(w) <= adapter.config.max_angular_speed + 1e-6


def test_sacadrl_adapter(monkeypatch):
    """SA-CADRL adapter can fall back when the model is unavailable (guards tests without TF)."""

    def _boom(self):
        """Simulate a missing SA-CADRL model."""
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
        """Simulate a missing SA-CADRL model when fallback is disabled."""
        raise RuntimeError("missing model")

    monkeypatch.setattr(SACADRLPlannerAdapter, "_build_model", _boom)
    obs = _make_obs(goal=(2.0, 0.0), heading=0.0)
    with pytest.raises(RuntimeError, match="missing model"):
        adapter.plan(obs)


def test_sacadrl_cpu_session_config_disables_gpu_devices():
    """SA-CADRL CPU inference should not ask TensorFlow to initialize CUDA devices."""
    from robot_sf.planner import socnav

    if socnav.tf is None:
        pytest.skip("TensorFlow not installed; skipping SA-CADRL TensorFlow config check.")

    config = _sacadrl_session_config(socnav.tf, device="/cpu:0")
    assert config.device_count["GPU"] == 0
    assert config.allow_soft_placement is True

    device_config = _sacadrl_session_config(socnav.tf, device=" /device:CPU:0 ")
    assert device_config.device_count["GPU"] == 0


def test_sacadrl_adapter_runs_model_when_available():
    """SA-CADRL adapter should run inference when TensorFlow and checkpoint are available."""
    from robot_sf.planner import socnav

    if socnav.tf is None:
        pytest.skip("TensorFlow not installed; skipping SA-CADRL inference check.")

    adapter = SACADRLPlannerAdapter(SocNavPlannerConfig(), allow_fallback=False)
    obs = _make_obs_with_peds([(1.5, 0.0)], goal=(4.0, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert np.isfinite(v)
    assert np.isfinite(w)


def test_prediction_adapter_fallback_when_model_missing(monkeypatch):
    """Predictive adapter can fall back to constant-velocity prediction when model is unavailable."""

    def _boom(self):
        """Simulate a missing predictive model for fallback planning."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    adapter = PredictionPlannerAdapter(SocNavPlannerConfig(), allow_fallback=True)
    obs = _make_obs_with_peds([(1.0, 0.2), (1.6, -0.1)], goal=(4.0, 0.0), heading=0.0)
    v, w = adapter.plan(obs)
    assert np.isfinite(v)
    assert np.isfinite(w)


def test_prediction_adapter_consumes_configured_forecast_variant() -> None:
    """Configured forecast variants should feed planner-consumed pedestrian futures."""
    cfg = SocNavPlannerConfig(
        forecast_variant="interaction_aware",
        forecast_variant_horizons_s=(0.5, 1.0, 1.5),
        forecast_variant_dt_s=0.5,
        predictive_horizon_steps=3,
        predictive_rollout_dt=0.5,
    )
    adapter = PredictionPlannerAdapter(cfg, allow_fallback=True)
    state = np.array(
        [
            [0.5, 0.0, 0.4, 0.0],
            [0.7, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mask = np.array([1.0, 1.0], dtype=np.float32)

    forecast_future = adapter._predict_trajectories(state, mask)
    cv_future = adapter._constant_velocity_prediction(state, mask)

    assert adapter.get_forecast_variant_execution_mode() == "native"
    assert forecast_future.shape == cv_future.shape == (2, 3, 2)
    assert not np.allclose(forecast_future, cv_future)


def test_prediction_adapter_reconfigures_forecast_variant_runtime_state() -> None:
    """Runtime configuration changes should not keep a stale baseline predictor."""
    adapter = PredictionPlannerAdapter(
        SocNavPlannerConfig(forecast_variant="interaction_aware"),
        allow_fallback=True,
    )
    assert adapter.get_forecast_variant_execution_mode() == "native"
    assert adapter._baseline_predictor is not None

    adapter.configure(SocNavPlannerConfig(forecast_variant=""))
    assert adapter.get_forecast_variant_execution_mode() == "native"
    assert adapter._baseline_predictor is None

    null_variant_config = SocNavPlannerConfig()
    null_variant_config.forecast_variant = None  # type: ignore[assignment]
    adapter.configure(null_variant_config)
    assert adapter.get_forecast_variant_execution_mode() == "native"
    assert adapter._baseline_predictor is None

    adapter._model = object()  # type: ignore[assignment]
    adapter._load_error = RuntimeError("stale")
    adapter._fallback_warned = True
    adapter.configure(SocNavPlannerConfig(forecast_variant="none"))
    assert adapter._model is None
    assert adapter._load_error is None
    assert adapter._fallback_warned is False


def test_prediction_adapter_invalid_forecast_variant_fails_closed() -> None:
    """Invalid configured forecast variants should fail closed when fallback is disabled."""
    with pytest.raises(RuntimeError, match="unsupported forecast_variant"):
        PredictionPlannerAdapter(
            SocNavPlannerConfig(forecast_variant="not-a-variant"),
            allow_fallback=False,
        )


def test_prediction_adapter_baseline_partial_miss_uses_constant_velocity_fallback() -> None:
    """Per-pedestrian forecast misses should not become robot-position futures."""

    class _Trajectory:
        def __init__(self, mean: np.ndarray) -> None:
            self.mean = mean

    class _Prediction:
        def __init__(self) -> None:
            self.predictions = [_Trajectory(np.empty((0, 2), dtype=np.float32))]

    class _PartialPredictor:
        def predict(self, observation):
            del observation
            return _Prediction()

    cfg = SocNavPlannerConfig(
        forecast_variant="interaction_aware",
        predictive_horizon_steps=3,
        predictive_rollout_dt=0.5,
    )
    adapter = PredictionPlannerAdapter(cfg, allow_fallback=True)
    adapter._baseline_predictor = _PartialPredictor()
    state = np.array([[1.0, 0.2, 0.4, 0.0]], dtype=np.float32)
    mask = np.array([1.0], dtype=np.float32)

    future = adapter._predict_with_baseline(state, mask)
    cv_future = adapter._constant_velocity_prediction(state, mask)

    assert future.shape == (1, 3, 2)
    assert np.allclose(future, cv_future)
    assert not np.allclose(future, np.zeros_like(future))


def test_prediction_adapter_requires_model_when_fallback_disabled(monkeypatch):
    """Predictive adapter fails fast when checkpoint/model loading fails and fallback is disabled."""

    def _boom(self):
        """Simulate a missing predictive model when fallback is disabled."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    adapter = PredictionPlannerAdapter(SocNavPlannerConfig(), allow_fallback=False)
    obs = _make_obs(goal=(2.0, 0.0), heading=0.0)
    with pytest.raises(RuntimeError, match="missing predictive model"):
        adapter.plan(obs)


def test_prediction_adapter_ttc_penalty_reduces_speed(monkeypatch):
    """High TTC weight should bias the predictive planner toward slower commands."""

    def _boom(self):
        """Force the predictive adapter onto heuristic prediction paths."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    obs = _make_obs_with_peds([(1.2, 0.0)], goal=(5.0, 0.0), heading=0.0)

    cfg_lo = SocNavPlannerConfig(
        predictive_ttc_weight=0.0,
        predictive_ttc_distance=0.9,
        predictive_collision_weight=0.2,
        predictive_candidate_speeds=(0.0, 0.5, 1.0),
    )
    cfg_hi = SocNavPlannerConfig(
        predictive_ttc_weight=0.8,
        predictive_ttc_distance=1.2,
        predictive_collision_weight=0.2,
        predictive_candidate_speeds=(0.0, 0.5, 1.0),
    )
    v_lo, _w_lo = PredictionPlannerAdapter(cfg_lo, allow_fallback=True).plan(obs)
    v_hi, _w_hi = PredictionPlannerAdapter(cfg_hi, allow_fallback=True).plan(obs)
    assert v_hi <= v_lo


def test_prediction_adapter_speed_clearance_gain_reduces_speed(monkeypatch):
    """Speed-adaptive safety margin should discourage aggressive forward speed."""

    def _boom(self):
        """Force the predictive adapter onto speed-clearance heuristic paths."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    obs = _make_obs_with_peds([(1.4, 0.0)], goal=(5.0, 0.0), heading=0.0)

    cfg_base = SocNavPlannerConfig(
        predictive_collision_weight=0.8,
        predictive_ttc_weight=0.6,
        predictive_ttc_distance=0.9,
        predictive_robot_radius=0.25,
        predictive_pedestrian_radius=0.25,
        predictive_speed_clearance_gain=0.0,
        predictive_candidate_speeds=(0.0, 0.5, 1.0),
    )
    cfg_gain = SocNavPlannerConfig(
        predictive_collision_weight=0.8,
        predictive_ttc_weight=0.6,
        predictive_ttc_distance=0.9,
        predictive_robot_radius=0.25,
        predictive_pedestrian_radius=0.25,
        predictive_speed_clearance_gain=0.6,
        predictive_candidate_speeds=(0.0, 0.5, 1.0),
    )

    v_base, _w_base = PredictionPlannerAdapter(cfg_base, allow_fallback=True).plan(obs)
    v_gain, _w_gain = PredictionPlannerAdapter(cfg_gain, allow_fallback=True).plan(obs)
    assert v_gain <= v_base


def test_prediction_adapter_progress_risk_penalty_reduces_speed(monkeypatch):
    """Progress-risk coupling should discourage fast progress through tight clearances."""

    def _boom(self):
        """Force the predictive adapter onto progress-risk heuristic paths."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    obs = _make_obs_with_peds([(1.1, 0.0)], goal=(4.0, 0.0), heading=0.0)

    cfg_lo = SocNavPlannerConfig(
        predictive_progress_risk_weight=0.0,
        predictive_progress_risk_distance=1.3,
        predictive_ttc_weight=0.1,
        predictive_collision_weight=0.2,
        predictive_candidate_speeds=(0.0, 0.5, 1.0),
    )
    cfg_hi = SocNavPlannerConfig(
        predictive_progress_risk_weight=4.0,
        predictive_progress_risk_distance=1.3,
        predictive_ttc_weight=0.1,
        predictive_collision_weight=0.2,
        predictive_candidate_speeds=(0.0, 0.5, 1.0),
    )

    v_lo, _w_lo = PredictionPlannerAdapter(cfg_lo, allow_fallback=True).plan(obs)
    v_hi, _w_hi = PredictionPlannerAdapter(cfg_hi, allow_fallback=True).plan(obs)
    assert v_hi <= v_lo


def test_prediction_adapter_adaptive_lattice_expands_near_field(monkeypatch):
    """Near-field context should produce an expanded candidate lattice."""

    def _boom(self):
        """Force candidate-set construction without a learned predictive model."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    cfg = SocNavPlannerConfig(
        predictive_near_field_distance=2.5,
        predictive_candidate_speeds=(0.0, 0.5, 1.0),
        predictive_candidate_heading_deltas=(-np.pi / 8, 0.0, np.pi / 8),
        predictive_near_field_speed_samples=(0.1, 0.2),
        predictive_near_field_heading_deltas=(-np.pi / 2, 0.0, np.pi / 2),
    )
    adapter = PredictionPlannerAdapter(cfg, allow_fallback=True)

    # One near pedestrian prediction triggers near-field expansion.
    future = np.zeros((1, 4, 2), dtype=np.float32)
    future[0, :, 0] = 0.8
    mask = np.array([1.0], dtype=np.float32)

    candidates = adapter._candidate_set(future_peds=future, mask=mask)
    base_count = len(cfg.predictive_candidate_speeds) * len(cfg.predictive_candidate_heading_deltas)
    assert len(candidates) > base_count


def test_prediction_adapter_candidate_set_computes_min_pred_dist_once(monkeypatch):
    """_candidate_set should call _min_predicted_distance exactly once for the shared near horizon."""

    def _boom(self):
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    cfg = SocNavPlannerConfig(
        predictive_near_field_distance=2.5,
        predictive_candidate_speeds=(0.0, 0.5, 1.0),
        predictive_candidate_heading_deltas=(-np.pi / 8, 0.0, np.pi / 8),
        predictive_near_field_speed_samples=(0.1, 0.2),
        predictive_near_field_heading_deltas=(-np.pi / 2, 0.0, np.pi / 2),
    )
    adapter = PredictionPlannerAdapter(cfg, allow_fallback=True)

    future = np.zeros((1, 4, 2), dtype=np.float32)
    future[0, :, 0] = 0.8
    mask = np.array([1.0], dtype=np.float32)

    call_count = 0
    original = PredictionPlannerAdapter._min_predicted_distance

    def counting_min_pred_dist(self, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(self, **kwargs)

    monkeypatch.setattr(PredictionPlannerAdapter, "_min_predicted_distance", counting_min_pred_dist)

    candidates = adapter._candidate_set(future_peds=future, mask=mask)
    assert call_count == 1, f"Expected 1 call to _min_predicted_distance, got {call_count}"

    base_count = len(cfg.predictive_candidate_speeds) * len(cfg.predictive_candidate_heading_deltas)
    assert len(candidates) > base_count


def test_prediction_adapter_reverse_candidates_appear_in_near_field(monkeypatch):
    """Reverse candidates should be added when explicitly enabled in close-contact regimes."""

    def _boom(self):
        """Force reverse-candidate construction without a learned model."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    cfg = SocNavPlannerConfig(
        predictive_allow_reverse_candidates=True,
        predictive_reverse_candidate_speeds=(-0.15, -0.3),
        predictive_reverse_near_field_only=True,
        predictive_near_field_distance=2.5,
        predictive_candidate_speeds=(0.0, 0.5, 1.0),
        predictive_candidate_heading_deltas=(-np.pi / 8, 0.0, np.pi / 8),
    )
    adapter = PredictionPlannerAdapter(cfg, allow_fallback=True)

    future = np.zeros((1, 4, 2), dtype=np.float32)
    future[0, :, 0] = 0.6
    mask = np.array([1.0], dtype=np.float32)

    candidates = adapter._candidate_set(future_peds=future, mask=mask)
    assert any(v < 0.0 for v, _ in candidates)


def test_prediction_adapter_progress_escape_injects_motion_in_clear_space(monkeypatch):
    """Progress-escape should avoid stationary commands when far from goal and safe."""

    def _boom(self):
        """Force progress-escape planning without a learned model."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    cfg = SocNavPlannerConfig(
        max_linear_speed=1.0,
        predictive_candidate_speeds=(0.0,),
        predictive_candidate_heading_deltas=(0.0,),
        predictive_progress_escape_enabled=True,
        predictive_progress_escape_distance=1.0,
        predictive_progress_escape_min_speed_ratio=0.4,
        predictive_progress_escape_clearance_margin=0.1,
    )
    obs = _make_obs(goal=(4.0, 0.0), heading=0.0)
    v, _w = PredictionPlannerAdapter(cfg, allow_fallback=True).plan(obs)
    assert v >= 0.39


def test_prediction_adapter_progress_escape_respects_clearance_gate(monkeypatch):
    """Progress-escape should not force motion when predicted clearance is too low."""

    def _boom(self):
        """Force the progress-escape clearance gate onto heuristic prediction."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    cfg = SocNavPlannerConfig(
        max_linear_speed=1.0,
        predictive_candidate_speeds=(0.0,),
        predictive_candidate_heading_deltas=(0.0,),
        predictive_progress_escape_enabled=True,
        predictive_progress_escape_distance=1.0,
        predictive_progress_escape_min_speed_ratio=0.5,
        predictive_hard_clearance_distance=0.75,
        predictive_progress_escape_clearance_margin=0.2,
        predictive_near_field_distance=0.0,
        predictive_near_field_speed_samples=(),
        predictive_near_field_heading_deltas=(0.0,),
    )
    obs = _make_obs_with_peds([(0.2, 0.0)], goal=(4.0, 0.0), heading=0.0)
    monkeypatch.setattr(
        PredictionPlannerAdapter,
        "_min_predicted_distance",
        lambda self, **_kwargs: 0.1,
    )
    v, _w = PredictionPlannerAdapter(cfg, allow_fallback=True).plan(obs)
    assert v <= 1e-6


def test_prediction_adapter_progress_escape_keeps_lower_cost_rollout(monkeypatch):
    """Progress-escape should not replace a safer rollout with a worse scored command."""

    def _boom(self):
        """Force lower-cost rollout selection without learned predictions."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    cfg = SocNavPlannerConfig(
        max_linear_speed=1.0,
        predictive_candidate_speeds=(0.0, 0.2),
        predictive_candidate_heading_deltas=(0.0,),
        predictive_progress_escape_enabled=True,
        predictive_progress_escape_distance=1.0,
        predictive_progress_escape_min_speed_ratio=0.5,
    )
    adapter = PredictionPlannerAdapter(cfg, allow_fallback=True)
    monkeypatch.setattr(
        adapter,
        "_score_action",
        lambda **kwargs: 0.1 if kwargs["v"] == 0.2 else 1.0,
    )
    v, w = adapter.plan(_make_obs(goal=(4.0, 0.0), heading=0.0))
    assert (v, w) == (0.2, 0.0)


def test_prediction_adapter_sequence_search_is_deterministic(monkeypatch):
    """Sequence search should stay deterministic for the same observation and config."""

    def _boom(self):
        """Force deterministic sequence search without learned predictions."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    cfg = SocNavPlannerConfig(
        predictive_sequence_search_enabled=True,
        predictive_sequence_segments=3,
        predictive_sequence_branch_factor=4,
        predictive_sequence_beam_width=6,
        predictive_candidate_speeds=(0.0, 0.4, 0.8),
        predictive_candidate_heading_deltas=(-np.pi / 6, 0.0, np.pi / 6),
    )
    obs = _make_obs_with_peds([(1.2, 0.3), (2.0, -0.4)], goal=(4.0, 0.0), heading=0.0)
    planner_a = PredictionPlannerAdapter(cfg, allow_fallback=True)
    planner_b = PredictionPlannerAdapter(cfg, allow_fallback=True)
    assert planner_a.plan(obs) == planner_b.plan(obs)


def test_prediction_adapter_sequence_search_keeps_progress_escape(monkeypatch):
    """Sequence search should still allow progress-escape recovery in clear space."""

    def _boom(self):
        """Force sequence-search progress escape without learned predictions."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    cfg = SocNavPlannerConfig(
        max_linear_speed=1.0,
        predictive_sequence_search_enabled=True,
        predictive_candidate_speeds=(0.0,),
        predictive_candidate_heading_deltas=(0.0,),
        predictive_progress_escape_enabled=True,
        predictive_progress_escape_distance=1.0,
        predictive_progress_escape_min_speed_ratio=0.4,
        predictive_progress_escape_clearance_margin=0.1,
    )
    obs = _make_obs(goal=(4.0, 0.0), heading=0.0)
    v, _w = PredictionPlannerAdapter(cfg, allow_fallback=True).plan(obs)
    assert v >= 0.39


def test_prediction_adapter_probabilistic_risk_mode_is_deterministic(monkeypatch):
    """Probabilistic rollout scoring should stay deterministic for a fixed seed."""

    def _boom(self):
        """Force probabilistic risk scoring onto heuristic predictions."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    cfg = SocNavPlannerConfig(
        predictive_uncertainty_mode="heuristic_gaussian",
        predictive_risk_sample_count=4,
        predictive_risk_seed=13,
        predictive_risk_objective="cvar",
        predictive_risk_cvar_alpha=0.5,
        predictive_candidate_speeds=(0.0, 0.3, 0.6),
        predictive_candidate_heading_deltas=(-np.pi / 8, 0.0, np.pi / 8),
    )
    obs = _make_obs_with_peds([(1.3, 0.4), (2.1, -0.2)], goal=(4.0, 0.0), heading=0.0)
    planner_a = PredictionPlannerAdapter(cfg, allow_fallback=True)
    planner_b = PredictionPlannerAdapter(cfg, allow_fallback=True)
    assert planner_a.plan(obs) == planner_b.plan(obs)


def test_prediction_adapter_cvar_objective_penalizes_worse_tail() -> None:
    """CVaR aggregation should be strictly more conservative than the mean on skewed samples."""
    cvar_cfg = SocNavPlannerConfig(
        predictive_risk_objective="cvar",
        predictive_risk_cvar_alpha=0.5,
    )
    mean_cfg = SocNavPlannerConfig(predictive_risk_objective="mean")
    cvar_adapter = PredictionPlannerAdapter(cvar_cfg, allow_fallback=True)
    mean_adapter = PredictionPlannerAdapter(mean_cfg, allow_fallback=True)
    costs = [1.0, 2.0, 10.0, 20.0]
    cvar = cvar_adapter._aggregate_risk_costs(costs)
    mean = mean_adapter._aggregate_risk_costs(costs)
    assert cvar > mean


def test_prediction_adapter_mcts_mode_is_deterministic(monkeypatch):
    """MCTS-lite search should stay deterministic under a fixed planner seed."""

    def _boom(self):
        """Force MCTS-lite search onto heuristic predictions."""
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)
    cfg = SocNavPlannerConfig(
        predictive_mcts_enabled=True,
        predictive_sequence_segments=3,
        predictive_mcts_iterations=24,
        predictive_mcts_branch_factor=3,
        predictive_mcts_rollout_count=2,
        predictive_risk_seed=5,
        predictive_candidate_speeds=(0.0, 0.35, 0.7),
        predictive_candidate_heading_deltas=(-np.pi / 6, 0.0, np.pi / 6),
    )
    obs = _make_obs_with_peds([(1.0, 0.25), (1.8, -0.3)], goal=(4.0, 0.0), heading=0.0)
    planner_a = PredictionPlannerAdapter(cfg, allow_fallback=True)
    planner_b = PredictionPlannerAdapter(cfg, allow_fallback=True)
    assert planner_a.plan(obs) == planner_b.plan(obs)


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
    prediction_policy = make_prediction_policy(allow_fallback=True)
    assert isinstance(prediction_policy.adapter, PredictionPlannerAdapter)


def test_prediction_planner_caching_rollout_in_score_action(monkeypatch):
    """PredictionPlannerAdapter must only call _rollout_robot once per _score_action invocation."""
    cfg = SocNavPlannerConfig(
        predictive_candidate_speeds=(0.5,),
        predictive_candidate_heading_deltas=(0.0,),
    )
    obs = _make_obs_with_peds([(1.0, 0.25)], goal=(4.0, 0.0), heading=0.0)

    def _boom(self):
        raise RuntimeError("missing predictive model")

    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", _boom)

    adapter = PredictionPlannerAdapter(cfg, allow_fallback=True)
    rollouts = 0
    original_rollout = adapter._rollout_robot

    def _tracked_rollout(*args, **kwargs):
        nonlocal rollouts
        rollouts += 1
        return original_rollout(*args, **kwargs)

    monkeypatch.setattr(adapter, "_rollout_robot", _tracked_rollout)

    future_peds = np.zeros((1, 5, 2))
    mask = np.ones(1)
    adapter._score_action(
        observation=obs,
        future_peds=future_peds,
        mask=mask,
        v=0.5,
        w=0.0,
        steps=5,
    )
    assert rollouts == 1


def _scalar_rollout_robot(v, w, dt, steps):
    """Reference sequential scalar recurrence for issue #5412 parity."""
    pos = np.zeros(2, dtype=float)
    heading = 0.0
    traj = np.zeros((steps, 2), dtype=float)
    for i in range(steps):
        heading += float(w) * dt
        pos[0] += float(v) * np.cos(heading) * dt
        pos[1] += float(v) * np.sin(heading) * dt
        traj[i] = pos
    return traj


def test_prediction_rollout_robot_vectorized_parity():
    """Vectorized _rollout_robot must match the scalar recurrence exactly (#5412)."""
    adapter = PredictionPlannerAdapter(SocNavPlannerConfig(), allow_fallback=True)
    rng = np.random.default_rng(20260716)
    for _ in range(25):
        v = float(rng.uniform(-1.2, 1.2))
        w = float(rng.uniform(-1.2, 1.2))
        dt = float(rng.uniform(0.05, 0.3))
        steps = int(rng.integers(1, 16))
        got = adapter._rollout_robot(v=v, w=w, dt=dt, steps=steps)
        ref = _scalar_rollout_robot(v, w, dt, steps)
        assert got.shape == (steps, 2)
        # Closed-form cumsum reorders float additions vs the sequential scalar
        # recurrence, so the residual is last-ULP drift (<1e-15). Per issue #5412
        # the parity gate tolerates this: atol=1e-12 is sub-nanometer and
        # benchmark-irrelevant, so the vectorization is accepted (no version bump).
        assert np.allclose(got, ref, atol=1e-12, rtol=0.0)


def test_prediction_rollout_robot_boundary_steps_match_scalar_reference():
    """Vectorization must preserve zero- and negative-step scalar behavior (#5412)."""
    adapter = PredictionPlannerAdapter(SocNavPlannerConfig(), allow_fallback=True)
    got = adapter._rollout_robot(v=0.5, w=0.2, dt=0.1, steps=0)
    assert got.shape == (0, 2)
    assert np.array_equal(got, _scalar_rollout_robot(0.5, 0.2, 0.1, 0))
    with pytest.raises(ValueError, match="negative dimensions"):
        adapter._rollout_robot(v=0.5, w=0.2, dt=0.1, steps=-1)


# --- SocialForce force-broadcast parity (issue #5412) ---------------------
# The scalar reference loops below reproduce the pre-vectorization
# ``_compute_social_force`` / ``_compute_obstacle_force`` kernels exactly: the
# per-pedestrian ``sf_forces.social_force_ped_ped`` call and the per-obstacle
# ``sf_forces.obstacle_force`` call on a degenerate single-point line. They are
# the numeric-parity reference for the vectorized broadcast; the determinism
# policy in #5412 requires this gate before a behavior-changing vectorization
# is accepted.


def _scalar_compute_social_force(adapter, robot_pos, robot_vel, ped_state, robot_heading):
    """Reference scalar ped-ped social-force loop (pre-#5412-vectorization)."""
    sf_forces = _socnav_module.sf_forces
    ped_positions = np.asarray(ped_state.get("positions", []), dtype=float)
    if ped_positions.ndim == 1:
        ped_positions = ped_positions.reshape(-1, 2)
    ped_count = int(adapter._as_1d_float(ped_state.get("count", [0]), pad=1)[0])
    ped_positions = ped_positions[:ped_count]
    if ped_positions.size == 0:
        return np.zeros(2, dtype=float)
    ped_velocities = np.asarray(ped_state.get("velocities", []), dtype=float)
    if ped_velocities.size == 0:
        ped_velocities = np.zeros_like(ped_positions, dtype=float)
    elif ped_velocities.ndim == 1:
        ped_velocities = ped_velocities.reshape(-1, 2)
    ped_velocities = ped_velocities[:ped_count]
    ped_vel_world = adapter._rotate_velocities_to_world(ped_velocities, robot_heading)

    total = np.zeros(2, dtype=float)
    for ped_pos, ped_vel in zip(ped_positions, ped_vel_world, strict=False):
        pos_diff = (robot_pos - ped_pos).astype(float)
        vel_diff = (robot_vel - ped_vel).astype(float)
        try:
            fx, fy = sf_forces.social_force_ped_ped(
                pos_diff,
                vel_diff,
                int(adapter.config.social_force_n),
                int(adapter.config.social_force_n_prime),
                float(adapter.config.social_force_lambda_importance),
                float(adapter.config.social_force_gamma),
            )
            total += np.array([fx, fy], dtype=float)
        except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
            continue
    return total * float(adapter.config.social_force_factor)


def _scalar_compute_obstacle_force(adapter, observation, robot_pos, robot_heading, robot_vel):
    """Reference scalar point-obstacle force loop (pre-#5412-vectorization)."""
    sf_forces = _socnav_module.sf_forces
    centers, radii = adapter._extract_obstacles_from_grid(observation, robot_pos, robot_heading)
    if centers.size == 0:
        return np.zeros(2, dtype=float)
    if np.linalg.norm(robot_vel) > adapter._EPS:
        ortho = np.array([-robot_vel[1], robot_vel[0]], dtype=float)
    else:
        ortho = np.array([-np.sin(robot_heading), np.cos(robot_heading)], dtype=float)
    robot_state = observation["robot"]
    robot_radius = float(adapter._as_1d_float(robot_state.get("radius", [0.0]), pad=1)[0])
    total = np.zeros(2, dtype=float)
    for center, radius in zip(centers, radii, strict=False):
        line = (float(center[0]), float(center[1]), float(center[0]), float(center[1]))
        try:
            fx, fy = sf_forces.obstacle_force(line, ortho, robot_pos, robot_radius + radius)
            total += np.array([fx, fy], dtype=float)
        except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
            continue
    return total * float(adapter.config.social_force_obstacle_factor)


_sf_available = pytest.mark.skipif(
    _socnav_module.sf_forces is None,
    reason="pysocialforce (fast-pysf) is required for the SocialForce parity gate",
)


@_sf_available
def test_social_force_compute_vectorized_parity():
    """Vectorized ped-ped social force must match the scalar loop within the parity gate."""
    adapter = SocialForcePlannerAdapter(SocNavPlannerConfig())
    rng = np.random.default_rng(20260717)
    for _ in range(50):
        m = int(rng.integers(0, 30))
        robot_pos = rng.uniform(-6.0, 6.0, 2)
        robot_vel = rng.uniform(-2.0, 2.0, 2)
        heading = float(rng.uniform(-np.pi, np.pi))
        positions = rng.uniform(-6.0, 6.0, (max(m, 1), 2))
        velocities = rng.uniform(-1.5, 1.5, (max(m, 1), 2))
        ped_state = {
            "positions": positions,
            "velocities": velocities,
            "count": np.array([float(m)]),
        }
        got = adapter._compute_social_force(robot_pos, robot_vel, ped_state, heading)
        ref = _scalar_compute_social_force(adapter, robot_pos, robot_vel, ped_state, heading)
        assert got.shape == (2,)
        # The vectorized broadcast reorders the float reduction versus the
        # sequential scalar accumulation. The residual is last-ULP drift; the
        # force magnitude here is bounded (no near-singular inputs in this
        # draw), so atol=1e-12 plus a tight rtol=1e-9 is benchmark-irrelevant and
        # accepts the vectorization (no version bump) per issue #5412.
        assert np.allclose(got, ref, atol=1e-12, rtol=1e-9)


@_sf_available
def test_social_force_compute_empty_peds_returns_zero():
    """An empty / zero-count ped population must yield a zero social force."""
    adapter = SocialForcePlannerAdapter(SocNavPlannerConfig())
    robot_pos = np.array([0.0, 0.0])
    robot_vel = np.array([1.0, 0.0])
    ped_state = {"positions": np.zeros((0, 2)), "velocities": np.zeros((0, 2)), "count": [0.0]}
    got = adapter._compute_social_force(robot_pos, robot_vel, ped_state, 0.0)
    assert np.array_equal(got, np.zeros(2))


@_sf_available
def test_social_force_obstacle_vectorized_parity():
    """Vectorized point-obstacle force must match the scalar loop within the parity gate."""
    cfg = SocNavPlannerConfig(social_force_obstacle_range=12.0)
    adapter = SocialForcePlannerAdapter(cfg)
    rng = np.random.default_rng(20260717)
    for trial in range(40):
        robot_pos = rng.uniform(-3.0, 3.0, 2)
        robot_vel = rng.uniform(-1.0, 1.0, 2)
        heading = float(rng.uniform(-np.pi, np.pi))
        cells = [(int(r), int(c)) for r, c in rng.integers(0, 6, size=(int(rng.integers(0, 8)), 2))]
        obs = _with_occupancy_grid(
            _make_obs(goal=(8.0, 0.0), heading=heading),
            obstacle_cells=cells or None,
            resolution=1.0,
            origin=(-3.0, -3.0),
        )
        obs["robot"]["position"] = robot_pos.astype(np.float32)
        got = adapter._compute_obstacle_force(obs, robot_pos, heading, robot_vel, obs["robot"])
        ref = _scalar_compute_obstacle_force(adapter, obs, robot_pos, heading, robot_vel)
        assert got.shape == (2,)
        # The point-obstacle potential ``1/obst_dist**3`` is near-singular when an
        # obstacle sits inside the combined robot+obstacle radius (both paths then
        # clamp obst_dist to 1e-5 and agree to machine-epsilon *relative* error,
        # while the *absolute* force is ~1e19). The parity gate is therefore on
        # relative error: rtol=1e-6 (six significant digits, far above the
        # observed ~1e-13 residual) is benchmark-irrelevant and accepts the
        # vectorization (no version bump) per issue #5412.
        scale = np.maximum(np.abs(ref), 1e-30)
        assert np.all(np.abs(got - ref) <= 1e-6 * scale + 1e-12)


@_sf_available
def test_social_force_obstacle_no_grid_returns_zero():
    """With no occupancy grid the vectorized obstacle force must be zero."""
    adapter = SocialForcePlannerAdapter(SocNavPlannerConfig())
    obs = _make_obs(goal=(5.0, 0.0))
    robot_pos = np.array([0.0, 0.0])
    got = adapter._compute_obstacle_force(obs, robot_pos, 0.0, np.array([1.0, 0.0]), obs["robot"])
    assert np.array_equal(got, np.zeros(2))
