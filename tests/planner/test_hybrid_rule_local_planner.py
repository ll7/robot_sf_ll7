"""Tests for the deterministic hybrid-rule local planner."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleCandidate,
    HybridRuleLocalPlannerAdapter,
    HybridRuleLocalPlannerConfig,
    build_hybrid_rule_local_planner_config,
)


def _obs(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    goal: tuple[float, float] = (2.0, 0.0),
    speed: float = 0.0,
    ped_positions: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
) -> dict:
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
            "radius": 0.25,
        },
        "sim": {"timestep": 0.1},
    }


def test_hybrid_rule_v0_returns_diagnostics_for_open_space() -> None:
    """V0 should choose a bounded forward command and expose score terms."""
    planner = HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())

    linear, angular = planner.plan(_obs())

    assert 0.0 <= linear <= planner.config.max_linear_speed
    assert abs(angular) <= planner.config.max_angular_speed
    diagnostics = planner.diagnostics()
    last = diagnostics["last_decision"]
    assert last["planner_variant"] == "hybrid_rule_v0_minimal"
    assert last["planner_mode"] == "NORMAL"
    assert last["top_k"]
    assert "goal_progress" in last["selected_terms"]
    assert diagnostics["selected_source_counts"]


def test_hybrid_rule_last_decision_returns_copy() -> None:
    """Step-level diagnostics should not expose mutable planner internals."""
    planner = HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())
    planner.plan(_obs())

    last = planner.last_decision()

    assert last is not None
    assert last["planner_variant"] == "hybrid_rule_v0_minimal"
    last["planner_variant"] = "mutated"
    assert planner.last_decision()["planner_variant"] == "hybrid_rule_v0_minimal"


def test_hybrid_rule_v0_speed_cap_near_humans() -> None:
    """The documented near-human speed cap should limit selected commands."""
    planner = HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())

    linear, _angular = planner.plan(_obs(ped_positions=[(0.8, 0.3)], ped_velocities=[(0.0, 0.0)]))

    assert linear <= planner.config.very_slow_speed + 1e-9
    last = planner.diagnostics()["last_decision"]
    assert last["nearest_pedestrian_distance"] < planner.config.slow_distance_human


def test_hybrid_rule_structured_pedestrian_velocities_convert_to_world_frame() -> None:
    """Structured SocNav pedestrian velocities are ego-frame and must be rotated."""
    planner = HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())

    state = planner._extract_state(
        _obs(
            heading=np.pi / 2.0,
            ped_positions=[(2.0, 2.0)],
            ped_velocities=[(1.0, 0.0)],
        )
    )

    assert np.allclose(state["ped_vel"][0], np.array([0.0, 1.0]), atol=1e-9)


def test_hybrid_rule_commits_to_current_waypoint_until_switch_distance() -> None:
    """Planner should not skip the active route waypoint just because next is closer."""
    planner = HybridRuleLocalPlannerAdapter(
        HybridRuleLocalPlannerConfig(waypoint_switch_distance=0.9)
    )
    obs = _obs(robot=(0.0, 0.0), goal=(2.0, 0.0))
    obs["goal"]["next"] = np.asarray((1.5, 0.0), dtype=float)

    state = planner._extract_state(obs)

    assert np.allclose(state["goal"], np.asarray((2.0, 0.0), dtype=float))


def test_hybrid_rule_switches_to_next_waypoint_near_current() -> None:
    """Planner should hand off to the next waypoint after reaching the active one."""
    planner = HybridRuleLocalPlannerAdapter(
        HybridRuleLocalPlannerConfig(waypoint_switch_distance=0.9)
    )
    obs = _obs(robot=(0.0, 0.0), goal=(0.5, 0.0))
    obs["goal"]["next"] = np.asarray((2.0, 0.0), dtype=float)

    state = planner._extract_state(obs)

    assert np.allclose(state["goal"], np.asarray((2.0, 0.0), dtype=float))


def test_hybrid_rule_v0_emergency_stop_when_all_candidates_rejected() -> None:
    """Hard dynamic collision filtering should fail closed to an emergency stop."""
    planner = HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())

    command = planner.plan(_obs(ped_positions=[(0.3, 0.0)], ped_velocities=[(0.0, 0.0)]))

    assert command == (0.0, 0.0)
    diagnostics = planner.diagnostics()
    assert diagnostics["fallback_count"] == 1
    assert diagnostics["last_decision"]["planner_mode"] == "EMERGENCY_STOP"
    assert diagnostics["rejection_counts"]["dynamic_collision"] > 0


def test_hybrid_rule_v0_rejects_static_footprint_clearance(monkeypatch) -> None:
    """Static filtering should account for robot radius, not just occupied center cells."""
    cfg = HybridRuleLocalPlannerConfig(
        linear_samples=2,
        angular_samples=3,
        max_linear_speed=0.4,
        hard_safety_margin=0.1,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    state = planner._extract_state(_obs(speed=0.2))
    candidate = next(
        candidate
        for candidate in planner._generate_candidates(state, speed_cap=cfg.max_linear_speed)
        if candidate.linear > planner.config.freezing_speed_threshold
    )

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 0.2)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=_obs(speed=0.2),
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 1.0},
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "static_clearance"
    assert evaluation["candidate"] == candidate
    assert evaluation["min_static_clearance"] == pytest.approx(0.2)
    assert evaluation["hard_static_clearance"] == pytest.approx(0.35)


def test_hybrid_rule_static_clearance_escape_allows_slow_nonworsening_motion(
    monkeypatch,
) -> None:
    """Static escape should allow only slow commands that do not reduce clearance."""
    cfg = HybridRuleLocalPlannerConfig(
        linear_samples=2,
        angular_samples=3,
        max_linear_speed=0.4,
        hard_safety_margin=0.1,
        rollout_horizon=0.2,
        static_clearance_escape_enabled=True,
        static_clearance_escape_min_clearance=0.1,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(speed=0.2)
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "test_escape")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 0.3)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 1.0},
    )

    assert evaluation["accepted"] is True
    assert evaluation["terms"]["static_clearance_escape"] == 1.0


def test_hybrid_rule_static_clearance_escape_rejects_worsening_motion(monkeypatch) -> None:
    """Static escape must still reject candidates that move deeper into the clearance band."""
    cfg = HybridRuleLocalPlannerConfig(
        linear_samples=2,
        angular_samples=3,
        max_linear_speed=0.4,
        hard_safety_margin=0.1,
        rollout_horizon=0.2,
        static_clearance_escape_enabled=True,
        static_clearance_escape_min_clearance=0.1,
        static_clearance_escape_tolerance=0.0,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(speed=0.2)
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "test_escape")
    clearances = iter([0.3, 0.25])

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(
        planner,
        "_min_obstacle_clearance",
        lambda point, observation: next(clearances),
    )

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 1.0},
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "static_clearance"


def test_hybrid_rule_static_clearance_escape_rejects_below_minimum(monkeypatch) -> None:
    """Tolerance must not allow escape commands below the configured minimum clearance."""
    cfg = HybridRuleLocalPlannerConfig(
        linear_samples=2,
        angular_samples=3,
        max_linear_speed=0.4,
        hard_safety_margin=0.1,
        rollout_horizon=0.2,
        static_clearance_escape_enabled=True,
        static_clearance_escape_min_clearance=0.1,
        static_clearance_escape_tolerance=0.05,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(speed=0.2)
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "test_escape")
    clearances = iter([0.12, 0.09])

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(
        planner,
        "_min_obstacle_clearance",
        lambda point, observation: next(clearances),
    )

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 1.0},
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "static_clearance"


def test_hybrid_rule_static_clearance_escape_rejects_bounded_corridor_transit(
    monkeypatch,
) -> None:
    """Static escape should not enter the hard band from a currently safe pose."""
    cfg = HybridRuleLocalPlannerConfig(
        linear_samples=2,
        angular_samples=3,
        max_linear_speed=0.4,
        hard_safety_margin=0.1,
        rollout_horizon=0.2,
        static_clearance_escape_enabled=True,
        static_clearance_escape_min_clearance=0.25,
        static_clearance_escape_max_speed=0.3,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(speed=0.2)
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "test_corridor_transit")
    clearances = iter([0.42, 0.32])

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(
        planner,
        "_min_obstacle_clearance",
        lambda point, observation: next(clearances),
    )

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 1.0},
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "static_clearance"


def test_hybrid_rule_route_guide_commitment_bonus_only_when_stalled(monkeypatch) -> None:
    """Route-guide commitment should only boost route candidates under stalled progress."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        route_guide_commitment_weight=1.0,
        route_guide_commitment_progress_threshold=0.5,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    route_candidate = HybridRuleCandidate(0.4, 0.0, "route_guide")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)

    stalled_eval = planner._evaluate_candidate(
        candidate=route_candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.2},
    )
    moving_eval = planner._evaluate_candidate(
        candidate=route_candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 2.0},
    )

    assert stalled_eval["terms"]["route_guide_commitment"] == 1.0
    assert moving_eval["terms"]["route_guide_commitment"] == 0.0
    assert stalled_eval["score"] > moving_eval["score"]


def test_hybrid_rule_route_guide_adds_candidate_source(monkeypatch) -> None:
    """Route guidance should contribute an explicit candidate when enabled."""
    cfg = HybridRuleLocalPlannerConfig(route_guide_enabled=True)
    planner = HybridRuleLocalPlannerAdapter(cfg)
    monkeypatch.setattr(planner._route_guide, "plan", lambda observation: (0.4, 0.2))
    state = planner._extract_state(_obs())

    candidates = planner._generate_candidates(state, speed_cap=cfg.max_linear_speed)

    assert any(candidate.source == "route_guide" for candidate in candidates)


def test_hybrid_rule_static_recovery_reorients_when_safe() -> None:
    """Recovery mode should rotate instead of freezing when static rejection dominates."""
    cfg = HybridRuleLocalPlannerConfig(recovery_enabled=True, recovery_reorient_angular_speed=0.5)
    planner = HybridRuleLocalPlannerAdapter(cfg)

    allowed = planner._static_recovery_allowed(
        {"static_clearance": 5, "dynamic_collision": 1},
        nearest_ped=float("inf"),
    )

    assert allowed is True


def test_hybrid_rule_deadlock_escape_bonus_prefers_rotation_when_stalled() -> None:
    """Recovery scoring should favor rotation over full stop after stalled progress."""
    cfg = HybridRuleLocalPlannerConfig(
        recovery_enabled=True,
        deadlock_escape_weight=1.0,
        linear_samples=2,
        angular_samples=3,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    state = planner._extract_state(_obs(goal=(4.0, 0.0)))
    candidates = planner._generate_candidates(state, speed_cap=cfg.max_linear_speed)
    stop_eval = planner._evaluate_candidate(
        candidate=next(
            candidate
            for candidate in candidates
            if candidate.linear == 0.0 and candidate.angular == 0.0
        ),
        observation=_obs(goal=(4.0, 0.0)),
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
    )
    rotate_eval = planner._evaluate_candidate(
        candidate=next(candidate for candidate in candidates if candidate.source == "rotate_left"),
        observation=_obs(goal=(4.0, 0.0)),
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
    )

    assert stop_eval["terms"]["deadlock_escape"] == 0.0
    assert rotate_eval["terms"]["deadlock_escape"] == 1.0
    assert rotate_eval["score"] > stop_eval["score"]


def test_hybrid_rule_static_recenter_probe_prefers_safe_rotation(monkeypatch) -> None:
    """Static recentering should prefer rotations that make the next forward rollout safe."""
    cfg = HybridRuleLocalPlannerConfig(
        linear_samples=2,
        angular_samples=5,
        max_linear_speed=0.0,
        static_recenter_enabled=True,
        static_recenter_weight=2.0,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    planner._progress_history.extend([(0.0, 4.0), (1.0, 4.0), (2.0, 4.0)])
    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 2.0)
    monkeypatch.setattr(
        planner,
        "_static_recenter_probe_score",
        lambda *, candidate, observation, state, hard_static_clearance: (
            1.0 if candidate.angular <= -0.6 else 0.0
        ),
    )

    command = planner.plan(_obs(goal=(4.0, 0.0)))

    assert command[0] == pytest.approx(0.0)
    assert command[1] <= -0.6
    last = planner.last_decision()
    assert last is not None
    assert last["selected_terms"]["static_recenter"] == pytest.approx(1.0)


def test_hybrid_rule_static_recovery_blocks_near_pedestrian() -> None:
    """Recovery should not rotate aggressively when a pedestrian is very close."""
    cfg = HybridRuleLocalPlannerConfig(recovery_enabled=True)
    planner = HybridRuleLocalPlannerAdapter(cfg)

    allowed = planner._static_recovery_allowed(
        {"static_clearance": 5, "dynamic_collision": 0},
        nearest_ped=0.7,
    )

    assert allowed is False


def test_hybrid_rule_config_builder_and_variant_guard() -> None:
    """Config parsing should preserve manual constants and reject unknown variants."""
    cfg = build_hybrid_rule_local_planner_config(
        {
            "max_linear_speed": "1.4",
            "linear_samples": "5",
            "lookahead_distances": [0.4, 0.8],
            "static_hard_safety_margin": "0.0",
        }
    )
    assert cfg.max_linear_speed == pytest.approx(1.4)
    assert cfg.linear_samples == 5
    assert cfg.lookahead_distances == (0.4, 0.8)
    assert cfg.static_hard_safety_margin == pytest.approx(0.0)

    v3_cfg = build_hybrid_rule_local_planner_config(
        {"planner_variant": "hybrid_rule_v3_teb_like_rollout"}
    )
    assert v3_cfg.planner_variant == "hybrid_rule_v3_teb_like_rollout"
    HybridRuleLocalPlannerAdapter(v3_cfg)

    with pytest.raises(ValueError, match="Unsupported hybrid rule planner variant"):
        HybridRuleLocalPlannerAdapter(
            HybridRuleLocalPlannerConfig(planner_variant="hybrid_rule_v9_unknown")
        )


def test_hybrid_rule_reset_clears_episode_diagnostics() -> None:
    """Reset should clear per-episode planner state for reproducible runs."""
    planner = HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())
    planner.plan(_obs())
    assert planner.diagnostics()["steps"] == 1

    planner.reset(seed=123)

    diagnostics = planner.diagnostics()
    assert diagnostics["steps"] == 0
    assert diagnostics["last_decision"] is None
