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
    """Build a compact observation payload for hybrid-rule planner tests."""
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


def _obs_with_grid(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    goal: tuple[float, float] = (2.0, 0.0),
    occupied_cells: list[tuple[int, int]] | None = None,
) -> dict:
    """Build an observation with occupancy-grid metadata attached."""
    obs = _obs(robot=robot, heading=heading, goal=goal)
    grid = np.zeros((3, 21, 21), dtype=float)
    for row, col in occupied_cells or []:
        grid[0, row, col] = 1.0
        grid[2, row, col] = 1.0
    obs["occupancy_grid"] = grid
    obs["occupancy_grid_meta"] = {
        "origin": [-2.0, -2.0],
        "resolution": [0.2],
        "size": [4.2, 4.2],
        "use_ego_frame": [0.0],
        "center_on_robot": [0.0],
        "channel_indices": [0, 1, 2],
        "robot_pose": [robot[0], robot[1], heading],
    }
    return obs


def _route_corridor_payload(
    *,
    tangent_heading: float = 0.0,
    route_progress_1s: float = 0.0,
    route_progress_3s: float = 0.0,
    lateral_offset: float = 0.0,
) -> dict:
    """Build route-corridor diagnostics for guide and subgoal tests."""
    return {
        "route_start_world": [0.0, 0.0],
        "route_next_world": [1.0, 0.0],
        "route_goal_world": [4.0, 0.0],
        "route_waypoint_world": [1.0, 0.0],
        "route_waypoint_index": 1,
        "route_path_cell_count": 5,
        "route_remaining_distance": 4.0,
        "route_distance_to_waypoint": 1.0,
        "route_corner_distance": None,
        "route_tangent_heading": tangent_heading,
        "route_heading_error": tangent_heading,
        "corridor_center_clearance": 1.0,
        "corridor_width_estimate": 2.0,
        "robot_lateral_offset_to_corridor": lateral_offset,
        "route_arc_progress_windows": {
            "1s": route_progress_1s,
            "3s": route_progress_3s,
            "5s": route_progress_3s,
        },
    }


def _bind_continuous_static_env(
    planner: HybridRuleLocalPlannerAdapter,
    *,
    obstacle_segments: np.ndarray | None = None,
    width: float = 4.0,
    height: float = 4.0,
) -> None:
    """Bind a minimal environment-shaped continuous obstacle surface."""

    class _MapDef:
        """Map shell carrying dimensions for static-obstacle queries."""

    class _Simulator:
        """Simulator shell exposing map definition and obstacle lines."""

    class _Env:
        """Environment shell exposing the simulator to the planner."""

    map_def = _MapDef()
    map_def.width = width
    map_def.height = height
    simulator = _Simulator()
    simulator.map_def = map_def
    simulator.get_obstacle_lines = lambda: (
        np.empty((0, 4), dtype=float) if obstacle_segments is None else obstacle_segments
    )
    env = _Env()
    env.simulator = simulator
    planner.bind_env(env)


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


def test_goal_posterior_channel_can_change_selected_command_from_info() -> None:
    """Opt-in issue #4164 path consumes info channel during action selection."""
    base = HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())
    obs = _obs(
        goal=(4.0, 0.0),
        ped_positions=[(1.0, 0.2)],
        ped_velocities=[(0.0, 0.8)],
    )
    base_command = base.plan(obs)

    config = HybridRuleLocalPlannerConfig(
        goal_posterior_avoidance_enabled=True,
        goal_posterior_min_confidence=0.5,
        goal_posterior_near_distance=2.0,
        goal_posterior_crossing_lateral_margin=0.5,
        goal_posterior_yield_speed=0.05,
        goal_posterior_turn_rate=0.7,
        goal_posterior_score_bonus=20.0,
    )
    planner = HybridRuleLocalPlannerAdapter(config)
    posterior_obs = _obs(
        goal=(4.0, 0.0),
        ped_positions=[(1.0, 0.2)],
        ped_velocities=[(0.0, 0.8)],
    )
    posterior_obs["info"] = {
        "planner_goal_posterior_channel": {
            "enabled": True,
            "pedestrian_goal_posteriors": {
                "crossing_ped_0": {
                    "pedestrian_id": "crossing_ped_0",
                    "top_goal_id": "crossing_ped_0_route_goal",
                    "top_goal_confidence": 0.9,
                    "blocker": None,
                }
            },
        }
    }

    posterior_command = planner.plan(posterior_obs)
    diagnostics = planner.diagnostics()
    last = diagnostics["last_decision"]

    assert posterior_command != base_command
    assert last["selected_source"].startswith("goal_posterior_yield_")
    assert last["goal_posterior_avoidance"]["consumed"] is True
    assert last["goal_posterior_avoidance"]["active"] is True
    assert last["selected_terms"]["goal_posterior_avoidance"] > 0.0
    assert diagnostics["goal_posterior_avoidance"]["enabled"] is True


def test_tentabot_value_scorer_v0_exposes_clean_room_diagnostics() -> None:
    """The Tentabot-style spike should be guarded and explicit about provenance."""
    config = build_hybrid_rule_local_planner_config(
        {
            "planner_variant": "tentabot_value_scorer_v0",
            "value_scorer_profile": "hand_scored_linear_teacher_v0",
            "value_scorer_training_source": "robot_sf_hybrid_rule_teacher_only",
        }
    )
    planner = HybridRuleLocalPlannerAdapter(config)

    linear, angular = planner.plan(_obs())

    assert 0.0 <= linear <= planner.config.max_linear_speed
    assert abs(angular) <= planner.config.max_angular_speed
    diagnostics = planner.diagnostics()
    assert diagnostics["planner_variant"] == "tentabot_value_scorer_v0"
    scorer = diagnostics["value_scorer"]
    assert scorer["profile"] == "hand_scored_linear_teacher_v0"
    assert scorer["training_source"] == "robot_sf_hybrid_rule_teacher_only"
    assert scorer["upstream_code_used"] is False
    assert scorer["source_parity_claim"] is False
    assert diagnostics["last_decision"]["top_k"]
    assert diagnostics["last_decision"]["rejected_examples"] is not None
    assert diagnostics["unavailable_counts"] == {"corridor_subgoal": 1}
    assert diagnostics["last_decision"]["unavailable_counts"] == {"corridor_subgoal": 1}
    assert diagnostics["last_decision"]["unavailable_examples"] == [
        {"source": "corridor_subgoal", "reason": "disabled"}
    ]


def test_tentabot_value_scorer_v1_static_gate_demotes_unsafe_route_candidates(
    monkeypatch,
) -> None:
    """V1 should keep raw value diagnostics while demoting low-clearance route commands."""
    config = build_hybrid_rule_local_planner_config(
        {
            "planner_variant": "tentabot_value_scorer_v1_static_gated",
            "route_guide_enabled": True,
            "static_safety_gate_enabled": True,
            "static_safety_gate_min_clearance": 0.55,
            "static_safety_gate_penalty": 12.0,
            "static_safety_gate_all_sources": True,
        }
    )
    planner = HybridRuleLocalPlannerAdapter(config)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.0, 0.0, "route_guide")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 0.4)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=config.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
    )

    assert evaluation["accepted"] is True
    assert evaluation["raw_value_score"] > evaluation["score"]
    assert evaluation["terms"]["static_safety_gate_penalty"] == pytest.approx(12.0)
    assert evaluation["static_safety_gate"]["tier"] == "low_clearance_demoted"
    row = planner._candidate_diagnostic(evaluation)
    assert row["raw_value_score"] == pytest.approx(evaluation["raw_value_score"])
    assert row["static_safety_gate"]["reason"] == "low_clearance_without_safe_progress"


def test_tentabot_value_scorer_v1_static_gate_allows_low_clearance_progress(
    monkeypatch,
) -> None:
    """Low-clearance route candidates can remain eligible when they progress safely."""
    config = build_hybrid_rule_local_planner_config(
        {
            "planner_variant": "tentabot_value_scorer_v1_static_gated",
            "route_guide_enabled": True,
            "static_safety_gate_enabled": True,
            "static_safety_gate_min_clearance": 0.55,
            "static_safety_gate_progress_threshold": 0.05,
        }
    )
    planner = HybridRuleLocalPlannerAdapter(config)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.25, 0.0, "route_guide")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 0.4)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=config.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
    )

    assert evaluation["accepted"] is True
    assert evaluation["score"] == pytest.approx(evaluation["raw_value_score"])
    assert evaluation["terms"]["static_safety_gate_penalty"] == 0.0
    assert evaluation["static_safety_gate"]["tier"] == "guarded_progress"
    assert evaluation["static_safety_gate"]["progress_metric"] == "goal_distance"


def test_tentabot_value_scorer_v1_static_gate_uses_route_local_progress(
    monkeypatch,
) -> None:
    """Route-aware gate progress should follow corridor direction, not Euclidean goal progress."""
    config = build_hybrid_rule_local_planner_config(
        {
            "planner_variant": "tentabot_value_scorer_v1_static_gated",
            "route_guide_enabled": True,
            "static_safety_gate_enabled": True,
            "static_safety_gate_min_clearance": 0.55,
            "static_safety_gate_progress_threshold": 0.05,
        }
    )
    planner = HybridRuleLocalPlannerAdapter(config)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.25, 0.0, "route_guide")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 0.4)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=config.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
        route_corridor=_route_corridor_payload(tangent_heading=np.pi),
    )

    assert evaluation["accepted"] is True
    assert evaluation["terms"]["goal_progress"] > 0.0
    assert evaluation["terms"]["static_safety_gate_penalty"] == pytest.approx(12.0)
    assert evaluation["static_safety_gate"]["tier"] == "low_clearance_demoted"
    assert evaluation["static_safety_gate"]["progress_metric"] == "route_local"
    assert evaluation["static_safety_gate"]["progress"] < 0.0


def test_tentabot_value_scorer_v1_static_gate_can_cover_all_sources(monkeypatch) -> None:
    """The exploratory v1 config can apply the static gate beyond route-only primitives."""
    config = build_hybrid_rule_local_planner_config(
        {
            "planner_variant": "tentabot_value_scorer_v1_static_gated",
            "static_safety_gate_enabled": True,
            "static_safety_gate_all_sources": True,
            "static_safety_gate_min_clearance": 0.55,
        }
    )
    planner = HybridRuleLocalPlannerAdapter(config)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.0, 0.0, "dynamic_window")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 0.4)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=config.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
    )

    assert evaluation["static_safety_gate"]["source_gated"] is True
    assert evaluation["static_safety_gate"]["tier"] == "low_clearance_demoted"


def test_tentabot_value_scorer_v2_scores_route_arc_progress_over_goal_shortcut(
    monkeypatch,
) -> None:
    """V2 should reward route-local progress even when Euclidean goal distance regresses."""
    config = build_hybrid_rule_local_planner_config(
        {
            "planner_variant": "tentabot_value_scorer_v2_route_arc",
            "goal_progress_weight": 1.0,
            "route_arc_progress_weight": 4.0,
            "path_alignment_weight": 0.0,
            "speed_preference_weight": 0.0,
            "static_clearance_weight": 0.0,
            "dynamic_clearance_weight": 0.0,
            "ttc_weight": 0.0,
            "heading_smoothness_weight": 0.0,
            "velocity_smoothness_weight": 0.0,
            "control_effort_weight": 0.0,
            "freezing_weight": 0.0,
            "oscillation_weight": 0.0,
        }
    )
    planner = HybridRuleLocalPlannerAdapter(config)
    observation = _obs(heading=np.pi, goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    route_corridor = _route_corridor_payload(tangent_heading=np.pi)

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)

    forward_eval = planner._evaluate_candidate(
        candidate=HybridRuleCandidate(0.25, 0.0, "route_guide"),
        observation=observation,
        state=state,
        speed_cap=config.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
        route_corridor=route_corridor,
    )
    stop_eval = planner._evaluate_candidate(
        candidate=HybridRuleCandidate(0.0, 0.0, "stop"),
        observation=observation,
        state=state,
        speed_cap=config.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
        route_corridor=route_corridor,
    )

    assert forward_eval["accepted"] is True
    assert stop_eval["accepted"] is True
    assert forward_eval["terms"]["goal_progress"] < 0.0
    assert forward_eval["terms"]["route_arc_progress"] > 0.0
    assert forward_eval["score"] > stop_eval["score"]


def test_tentabot_trace_recovery_activates_on_route_regression_despite_goal_progress() -> None:
    """Trace recovery should notice route regression even if goal distance improved."""
    config = build_hybrid_rule_local_planner_config(
        {
            "planner_variant": "tentabot_value_scorer_v3_trace_recovery",
            "route_guide_enabled": True,
            "route_trace_recovery_enabled": True,
        }
    )
    planner = HybridRuleLocalPlannerAdapter(config)

    signal = planner._route_trace_recovery_signal(
        route_corridor=_route_corridor_payload(route_progress_1s=-0.1, route_progress_3s=0.4),
        progress_windows={"1s": 0.1, "3s": 0.4},
        nearest_ped=float("inf"),
    )

    assert signal["active"] is True
    assert signal["reason"] == "route_regressing"
    assert signal["goal_stalled"] is False
    assert signal["route_regressing"] is True


def test_tentabot_trace_recovery_selects_accepted_corridor_subgoal() -> None:
    """Trace recovery may only override with an already accepted recovery candidate."""
    planner = HybridRuleLocalPlannerAdapter(
        HybridRuleLocalPlannerConfig(route_trace_recovery_enabled=True)
    )
    signal = {"active": True, "selected": False}
    accepted = [
        {
            "candidate": HybridRuleCandidate(0.0, 0.0, "stop"),
            "score": 99.0,
            "terms": {"route_arc_progress": 0.0},
        },
        {
            "candidate": HybridRuleCandidate(0.2, 0.0, "corridor_subgoal"),
            "score": 1.0,
            "terms": {"route_arc_progress": 0.2},
        },
        {
            "candidate": HybridRuleCandidate(0.2, 0.0, "route_guide"),
            "score": 2.0,
            "terms": {"route_arc_progress": 0.3},
        },
    ]

    selected = planner._select_route_trace_recovery_evaluation(accepted, signal)

    assert selected is accepted[1]
    assert signal["selected"] is True
    assert signal["selected_source"] == "corridor_subgoal"


def test_tentabot_trace_recovery_does_not_select_rejected_recovery_candidate() -> None:
    """A trace signal must not convert rejected recovery candidates into commands."""
    planner = HybridRuleLocalPlannerAdapter(
        HybridRuleLocalPlannerConfig(route_trace_recovery_enabled=True)
    )
    signal = {"active": True, "selected": False}
    accepted = [
        {
            "candidate": HybridRuleCandidate(0.0, 0.0, "stop"),
            "score": 99.0,
            "terms": {"route_arc_progress": 0.0},
        }
    ]

    selected = planner._select_route_trace_recovery_evaluation(accepted, signal)

    assert selected is None
    assert signal["selected"] is False
    assert signal["selected_reason"] == "no_accepted_recovery_candidate"


def test_tentabot_trace_recovery_records_blocked_diagnostics() -> None:
    """Decision diagnostics should explain why trace recovery was unavailable."""
    config = build_hybrid_rule_local_planner_config(
        {
            "planner_variant": "tentabot_value_scorer_v3_trace_recovery",
            "route_guide_enabled": True,
            "route_trace_recovery_enabled": True,
        }
    )
    planner = HybridRuleLocalPlannerAdapter(config)

    planner.plan(_obs())

    last = planner.last_decision()
    assert last is not None
    assert last["route_trace_recovery"]["enabled"] is True
    assert last["route_trace_recovery"]["active"] is False
    assert last["route_trace_recovery"]["reason"] == "missing_route_geometry"


def test_tentabot_trace_recovery_reset_clears_hold_state() -> None:
    """Episode reset should clear any active trace-recovery hold window."""
    config = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        route_trace_recovery_enabled=True,
        route_trace_recovery_hold_steps=3,
    )
    planner = HybridRuleLocalPlannerAdapter(config)
    planner._route_trace_recovery_signal(
        route_corridor=_route_corridor_payload(route_progress_1s=-0.1),
        progress_windows={"3s": 0.5},
        nearest_ped=float("inf"),
    )

    assert planner._route_trace_recovery_hold_remaining == 3
    planner.reset()
    assert planner._route_trace_recovery_hold_remaining == 0


def test_actuation_aware_variant_penalizes_synthetic_clip_risk() -> None:
    """Actuation-aware scoring should expose and penalize synthetic envelope risk."""
    config = build_hybrid_rule_local_planner_config(
        {
            "planner_variant": "actuation_aware_hybrid_rule_v0",
            "actuation_score_enabled": True,
            "actuation_profile_name": "amv-actuation-stress-v0",
            "actuation_max_linear_accel": 1.0,
            "actuation_max_linear_decel": 1.0,
            "actuation_max_yaw_rate": 0.6,
            "actuation_max_angular_accel": 1.0,
            "actuation_clip_risk_weight": 4.0,
        }
    )
    planner = HybridRuleLocalPlannerAdapter(config)
    state = planner._extract_state(_obs(speed=0.0))
    smooth = planner._evaluate_candidate(
        candidate=HybridRuleCandidate(0.1, 0.0, "smooth"),
        observation=_obs(speed=0.0),
        state=state,
        speed_cap=config.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 1.0},
    )
    abrupt = planner._evaluate_candidate(
        candidate=HybridRuleCandidate(1.2, 1.2, "abrupt"),
        observation=_obs(speed=0.0),
        state=state,
        speed_cap=config.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 1.0},
    )

    assert smooth["accepted"] is True
    assert abrupt["accepted"] is True
    assert smooth["terms"]["actuation_clip_risk"] == pytest.approx(0.0)
    assert abrupt["terms"]["actuation_clip_risk"] > 0.0
    assert abrupt["terms"]["actuation_feasibility"] < smooth["terms"]["actuation_feasibility"]
    assert abrupt["actuation_diagnostics"]["profile"] == "amv-actuation-stress-v0"
    assert abrupt["actuation_diagnostics"]["projected_command"] != [1.2, 1.2]
    assert smooth["score"] > abrupt["score"]


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
    """Static escape alone should not enter the hard band from a currently safe pose."""
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


def test_hybrid_rule_static_corridor_transit_allows_slow_progress(
    monkeypatch,
) -> None:
    """Corridor transit should permit bounded slow progress through the conservative band."""
    cfg = HybridRuleLocalPlannerConfig(
        linear_samples=2,
        angular_samples=3,
        max_linear_speed=0.4,
        hard_safety_margin=0.1,
        rollout_horizon=0.2,
        static_clearance_escape_min_clearance=0.3,
        static_clearance_escape_max_speed=0.3,
        static_corridor_transit_enabled=True,
        static_corridor_transit_initial_band=0.05,
        static_corridor_transit_tolerance=0.05,
        static_corridor_transit_min_progress_3s=0.1,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(speed=0.2, goal=(2.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "test_corridor_transit")
    clearances = iter([0.37, 0.33])

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
        progress_windows={"3s": 0.2},
    )

    assert evaluation["accepted"] is True
    assert evaluation["terms"]["static_corridor_transit"] == 1.0


def test_hybrid_rule_static_corridor_transit_requires_recent_progress(
    monkeypatch,
) -> None:
    """Corridor transit should not keep creeping after recent progress has collapsed."""
    cfg = HybridRuleLocalPlannerConfig(
        linear_samples=2,
        angular_samples=3,
        max_linear_speed=0.4,
        hard_safety_margin=0.1,
        rollout_horizon=0.2,
        static_clearance_escape_min_clearance=0.3,
        static_clearance_escape_max_speed=0.3,
        static_corridor_transit_enabled=True,
        static_corridor_transit_initial_band=0.05,
        static_corridor_transit_tolerance=0.05,
        static_corridor_transit_min_progress_3s=0.1,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(speed=0.2, goal=(2.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "test_corridor_transit")
    clearances = iter([0.37, 0.33])

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
        progress_windows={"3s": 0.05},
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


def test_hybrid_rule_corridor_subgoal_disabled_by_default() -> None:
    """Corridor-subgoal candidates should be absent until explicitly enabled."""
    cfg = HybridRuleLocalPlannerConfig(route_guide_enabled=True)
    planner = HybridRuleLocalPlannerAdapter(cfg)
    state = planner._extract_state(_obs(goal=(4.0, 0.0)))
    activation = planner._corridor_subgoal_activation(
        route_corridor=_route_corridor_payload(route_progress_3s=0.0),
        progress_windows={"3s": 0.0},
        nearest_ped=float("inf"),
    )

    candidates = planner._generate_candidates(
        state,
        speed_cap=cfg.max_linear_speed,
        route_corridor=_route_corridor_payload(route_progress_3s=0.0),
        corridor_subgoal=activation,
    )

    assert activation["active"] is False
    assert activation["reason"] == "disabled"
    assert all(candidate.source != "corridor_subgoal" for candidate in candidates)


def test_hybrid_rule_corridor_subgoal_activation_requires_route_and_no_near_pedestrian() -> None:
    """Subgoal recovery should fail closed without geometry or near pedestrians."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        corridor_subgoal_min_nearest_ped_distance=1.0,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)

    missing = planner._corridor_subgoal_activation(
        route_corridor=None,
        progress_windows={"3s": 0.0},
        nearest_ped=float("inf"),
    )
    near_ped = planner._corridor_subgoal_activation(
        route_corridor=_route_corridor_payload(route_progress_3s=0.0),
        progress_windows={"3s": 0.0},
        nearest_ped=0.8,
    )
    route_without_progress = _route_corridor_payload(route_progress_3s=0.0)
    route_without_progress.pop("route_arc_progress_windows")
    missing_progress = planner._corridor_subgoal_activation(
        route_corridor=route_without_progress,
        progress_windows={"3s": 0.0},
        nearest_ped=1.2,
    )
    active = planner._corridor_subgoal_activation(
        route_corridor=_route_corridor_payload(route_progress_1s=-0.1, route_progress_3s=0.0),
        progress_windows={"3s": 0.0},
        nearest_ped=1.2,
    )

    assert missing["active"] is False
    assert missing["reason"] == "missing_route_geometry"
    assert near_ped["active"] is False
    assert near_ped["reason"] == "near_pedestrian"
    assert missing_progress["active"] is False
    assert missing_progress["reason"] == "missing_route_progress"
    assert active["active"] is True
    assert active["route_regressing"] is True


def test_hybrid_rule_corridor_subgoal_uses_scaled_one_second_stall_threshold() -> None:
    """One-second goal progress should use a tighter threshold than the 3s window."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        corridor_subgoal_goal_stall_progress_3s=0.06,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)

    moving_recently = planner._corridor_subgoal_activation(
        route_corridor=_route_corridor_payload(route_progress_3s=0.0),
        progress_windows={"1s": 0.03, "3s": 0.03},
        nearest_ped=float("inf"),
    )
    stalled = planner._corridor_subgoal_activation(
        route_corridor=_route_corridor_payload(route_progress_3s=0.0),
        progress_windows={"1s": 0.02, "3s": 0.03},
        nearest_ped=float("inf"),
    )
    regressing = planner._corridor_subgoal_activation(
        route_corridor=_route_corridor_payload(route_progress_1s=-0.1, route_progress_3s=0.0),
        progress_windows={"1s": -0.02, "3s": -0.03},
        nearest_ped=float("inf"),
    )

    assert moving_recently["active"] is False
    assert moving_recently["reason"] == "progress_not_stalled"
    assert stalled["active"] is True
    assert regressing["active"] is True


def test_hybrid_rule_corridor_subgoal_turns_in_place_for_large_tangent_error() -> None:
    """Large route-tangent error should not mix turning with forward subgoal motion."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        corridor_subgoal_speed=0.3,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    route_corridor = _route_corridor_payload(tangent_heading=np.pi / 4.0, route_progress_3s=0.0)
    activation = planner._corridor_subgoal_activation(
        route_corridor=route_corridor,
        progress_windows={"3s": 0.0},
        nearest_ped=float("inf"),
    )
    state = planner._extract_state(_obs(goal=(4.0, 0.0)))

    candidates = planner._generate_candidates(
        state,
        speed_cap=cfg.max_linear_speed,
        route_corridor=route_corridor,
        corridor_subgoal=activation,
    )

    subgoals = [candidate for candidate in candidates if candidate.source == "corridor_subgoal"]
    assert activation["candidate_count"] == len(subgoals)
    assert len(subgoals) == 1
    assert subgoals[0].linear == pytest.approx(0.0)
    assert subgoals[0].angular > 0.0
    assert subgoals[0].rollout_sequence
    assert subgoals[0].rollout_sequence[0][1] == pytest.approx(0.0)
    assert any(
        segment_linear > cfg.freezing_speed_threshold
        for _duration, segment_linear, _segment_angular in subgoals[0].rollout_sequence
    )


def test_hybrid_rule_corridor_subgoal_adds_forward_candidate_when_aligned() -> None:
    """Aligned route-corridor recovery should add a bounded forward subgoal."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        corridor_subgoal_speed=0.3,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    route_corridor = _route_corridor_payload(tangent_heading=0.0, route_progress_3s=0.0)
    activation = planner._corridor_subgoal_activation(
        route_corridor=route_corridor,
        progress_windows={"3s": 0.0},
        nearest_ped=float("inf"),
    )
    state = planner._extract_state(_obs(goal=(4.0, 0.0)))

    candidates = planner._generate_candidates(
        state,
        speed_cap=cfg.max_linear_speed,
        route_corridor=route_corridor,
        corridor_subgoal=activation,
    )

    subgoals = [candidate for candidate in candidates if candidate.source == "corridor_subgoal"]
    assert activation["candidate_count"] == len(subgoals)
    assert len(subgoals) == 1
    assert subgoals[0].linear > cfg.freezing_speed_threshold
    assert subgoals[0].angular == pytest.approx(0.0)


def test_hybrid_rule_corridor_subgoal_forward_speed_tracks_alignment() -> None:
    """Poorly aligned forward subgoals should not use an artificial speed floor."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        corridor_subgoal_speed=0.3,
        corridor_subgoal_turn_in_place_error=1.6,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    route_corridor = _route_corridor_payload(tangent_heading=0.0, route_progress_3s=0.0)
    route_corridor["route_waypoint_world"] = [-1.0, 0.1]
    activation = planner._corridor_subgoal_activation(
        route_corridor=route_corridor,
        progress_windows={"3s": 0.0},
        nearest_ped=float("inf"),
    )
    state = planner._extract_state(_obs(goal=(4.0, 0.0)))

    candidates = planner._generate_candidates(
        state,
        speed_cap=cfg.max_linear_speed,
        route_corridor=route_corridor,
        corridor_subgoal=activation,
    )

    assert not [candidate for candidate in candidates if candidate.source == "corridor_subgoal"]


def test_hybrid_rule_corridor_subgoal_rejects_occupied_rollout(monkeypatch) -> None:
    """Subgoal candidates must still fail closed on occupied grid cells."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        rollout_horizon=0.2,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "corridor_subgoal")

    grid = np.zeros((1, 3, 3), dtype=float)
    monkeypatch.setattr(
        planner,
        "_obstacle_grid_payload",
        lambda observation: (grid, {}, 0, 0.2),
    )
    monkeypatch.setattr(planner, "_grid_value", lambda point, grid, meta, channel: 1.0)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
        route_corridor=_route_corridor_payload(route_progress_3s=0.0),
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "static_collision"
    assert evaluation["candidate"] == candidate


def test_hybrid_rule_corridor_subgoal_rejects_static_clearance_band(monkeypatch) -> None:
    """Subgoal recovery must not reuse static-clearance escape exceptions."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        rollout_horizon=0.2,
        static_clearance_escape_enabled=True,
        static_clearance_escape_min_clearance=0.1,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "corridor_subgoal")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 0.4)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
        route_corridor=_route_corridor_payload(route_progress_3s=0.0),
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "static_clearance"
    assert evaluation["min_static_clearance"] == pytest.approx(0.4)
    assert evaluation["hard_static_clearance"] == pytest.approx(0.3)
    assert evaluation["required_static_clearance"] == pytest.approx(0.5)
    assert planner._rejection_diagnostic(evaluation)["required_static_clearance"] == pytest.approx(
        0.5
    )


def test_hybrid_rule_corridor_subgoal_rejects_continuous_static_collision(monkeypatch) -> None:
    """Bound environment geometry should catch line collisions missed by grid payloads."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        rollout_horizon=0.2,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    obstacle_segments = np.asarray([[0.2, -1.0, 0.2, 1.0]], dtype=float)
    _bind_continuous_static_env(planner, obstacle_segments=obstacle_segments)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "corridor_subgoal")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 2.0)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
        route_corridor=_route_corridor_payload(route_progress_3s=0.0),
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "static_collision"
    assert evaluation["continuous_static_collision"] is True
    assert planner._rejection_diagnostic(evaluation)["continuous_static_collision"] is True


def test_hybrid_rule_corridor_subgoal_rejects_sequence_static_collision(
    monkeypatch,
) -> None:
    """Continuous checks should cover later forward segments after an initial turn."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        rollout_dt=0.2,
        rollout_horizon=0.6,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    obstacle_segments = np.asarray([[0.45, -1.0, 0.45, 1.0]], dtype=float)
    _bind_continuous_static_env(planner, obstacle_segments=obstacle_segments)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(
        0.0,
        0.5,
        "corridor_subgoal",
        ((0.2, 0.0, 0.5), (0.4, 0.4, 0.0)),
    )

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 2.0)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
        route_corridor=_route_corridor_payload(route_progress_3s=0.0),
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "static_collision"
    assert evaluation["continuous_static_collision"] is True
    assert evaluation["time"] == pytest.approx(0.6)


def test_hybrid_rule_corridor_subgoal_uses_continuous_static_check_over_grid_band(
    monkeypatch,
) -> None:
    """Exact continuous checks may allow close but non-colliding corridor commands."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        rollout_horizon=0.2,
        static_clearance_escape_enabled=True,
        static_clearance_escape_min_clearance=0.1,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    _bind_continuous_static_env(planner)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "corridor_subgoal")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 0.4)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
        route_corridor=_route_corridor_payload(route_progress_3s=0.0),
    )

    assert evaluation["accepted"] is True
    assert evaluation["continuous_static_checked"] is True
    assert evaluation["min_static_clearance"] == pytest.approx(0.4)


def test_hybrid_rule_continuous_static_clearance_opt_in_allows_dynamic_grid_band(
    monkeypatch,
) -> None:
    """Opt-in exact geometry checks may replace conservative grid clearance for any source."""
    cfg = HybridRuleLocalPlannerConfig(
        rollout_horizon=0.2,
        continuous_static_clearance_enabled=True,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    _bind_continuous_static_env(planner)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "dynamic_window")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 0.2)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
    )

    assert evaluation["accepted"] is True
    assert evaluation["continuous_static_checked"] is True
    assert evaluation["min_static_clearance"] == pytest.approx(0.2)


def test_hybrid_rule_continuous_static_clearance_opt_in_rejects_dynamic_collision(
    monkeypatch,
) -> None:
    """Opt-in exact geometry checks should still fail closed on hard static collision."""
    cfg = HybridRuleLocalPlannerConfig(
        rollout_horizon=0.2,
        continuous_static_clearance_enabled=True,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    obstacle_segments = np.asarray([[0.2, -1.0, 0.2, 1.0]], dtype=float)
    _bind_continuous_static_env(planner, obstacle_segments=obstacle_segments)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "dynamic_window")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 2.0)

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "static_collision"
    assert evaluation["continuous_static_collision"] is True


def test_hybrid_rule_corridor_subgoal_strict_lock_blocks_escape_candidates(monkeypatch) -> None:
    """Active subgoal recovery should make every candidate obey hard static clearance."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        rollout_horizon=0.2,
        static_clearance_escape_enabled=True,
        static_clearance_escape_min_clearance=0.1,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "dynamic_window")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)
    monkeypatch.setattr(planner, "_min_obstacle_clearance", lambda point, observation: 0.4)

    relaxed = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
    )
    strict = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
        strict_static_clearance=True,
    )

    assert relaxed["accepted"] is True
    assert relaxed["terms"]["static_clearance_escape"] == 0.0
    assert strict["accepted"] is False
    assert strict["reason"] == "static_clearance"
    assert strict["required_static_clearance"] == pytest.approx(0.5)


def test_hybrid_rule_corridor_subgoal_rejects_hard_dynamic_collision() -> None:
    """Subgoal candidates should use the same hard pedestrian collision gate."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        rollout_horizon=0.2,
        hard_collision_horizon=0.2,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(goal=(4.0, 0.0), ped_positions=[(0.25, 0.0)], ped_velocities=[(0.0, 0.0)])
    state = planner._extract_state(observation)
    candidate = HybridRuleCandidate(0.2, 0.0, "corridor_subgoal")

    evaluation = planner._evaluate_candidate(
        candidate=candidate,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=0.25,
        progress_windows={"3s": 0.0},
        route_corridor=_route_corridor_payload(route_progress_3s=0.0),
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "dynamic_collision"


def test_hybrid_rule_corridor_subgoal_route_terms_score_route_arc_progress(monkeypatch) -> None:
    """Route-corridor scoring should prefer progress along the route tangent."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        corridor_subgoal_route_progress_weight=2.0,
        corridor_subgoal_tangent_alignment_weight=1.0,
        rollout_horizon=0.4,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    observation = _obs(goal=(4.0, 0.0))
    state = planner._extract_state(observation)
    route_corridor = _route_corridor_payload(tangent_heading=0.0, route_progress_3s=0.0)
    forward = HybridRuleCandidate(0.4, 0.0, "corridor_subgoal")
    sideways = HybridRuleCandidate(0.4, 1.2, "corridor_subgoal")

    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)

    forward_eval = planner._evaluate_candidate(
        candidate=forward,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
        route_corridor=route_corridor,
    )
    sideways_eval = planner._evaluate_candidate(
        candidate=sideways,
        observation=observation,
        state=state,
        speed_cap=cfg.max_linear_speed,
        nearest_ped=float("inf"),
        progress_windows={"3s": 0.0},
        route_corridor=route_corridor,
    )

    assert forward_eval["accepted"] is True
    assert sideways_eval["accepted"] is True
    assert forward_eval["terms"]["corridor_subgoal_route_progress"] > 0.0
    assert (
        forward_eval["terms"]["corridor_subgoal_tangent_alignment"]
        > sideways_eval["terms"]["corridor_subgoal_tangent_alignment"]
    )
    assert forward_eval["score"] > sideways_eval["score"]


def test_hybrid_rule_last_decision_includes_corridor_subgoal_diagnostics(monkeypatch) -> None:
    """Decision diagnostics should report activation and selected subgoal terms."""
    cfg = HybridRuleLocalPlannerConfig(
        route_guide_enabled=True,
        corridor_subgoal_enabled=True,
        corridor_subgoal_tangent_alignment_weight=3.0,
        top_k_diagnostics=50,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    route_corridor = _route_corridor_payload(tangent_heading=-np.pi / 2.0, route_progress_3s=0.0)
    monkeypatch.setattr(
        planner,
        "_route_corridor_diagnostics",
        lambda observation, current_time: route_corridor,
    )
    monkeypatch.setattr(planner._route_guide, "plan", lambda observation: (0.0, 0.0))
    monkeypatch.setattr(planner, "_obstacle_grid_payload", lambda observation: None)

    planner.plan(_obs(goal=(4.0, 0.0)))

    last = planner.last_decision()
    assert last is not None
    assert last["corridor_subgoal"]["active"] is True
    assert last["corridor_subgoal"]["candidate_count"] > 0
    assert any(row["source"] == "corridor_subgoal" for row in last["top_k"])


def test_hybrid_rule_last_decision_includes_route_corridor_diagnostics() -> None:
    """Route-guide diagnostics should include route-corridor geometry when available."""
    cfg = HybridRuleLocalPlannerConfig(route_guide_enabled=True)
    planner = HybridRuleLocalPlannerAdapter(cfg)

    planner.plan(_obs_with_grid(goal=(2.0, 0.0)))

    last = planner.last_decision()
    assert last is not None
    route_corridor = last["route_corridor"]
    assert route_corridor is not None
    assert route_corridor["route_path_cell_count"] > 1
    assert len(route_corridor["route_waypoint_world"]) == 2
    assert route_corridor["route_remaining_distance"] > 0.0
    assert route_corridor["route_tangent_heading"] is not None
    assert route_corridor["route_heading_error"] is not None
    assert route_corridor["route_arc_progress_windows"] == {
        "1s": 0.0,
        "3s": 0.0,
        "5s": 0.0,
    }

    planner.plan(_obs_with_grid(robot=(0.2, 0.0), goal=(2.0, 0.0)))
    last = planner.last_decision()
    assert last is not None
    route_corridor = last["route_corridor"]
    assert route_corridor is not None
    assert any(value != 0.0 for value in route_corridor["route_arc_progress_windows"].values())


def test_hybrid_rule_route_corridor_diagnostics_fail_closed_without_grid() -> None:
    """Missing route geometry should leave diagnostics empty without affecting planning."""
    cfg = HybridRuleLocalPlannerConfig(route_guide_enabled=True)
    planner = HybridRuleLocalPlannerAdapter(cfg)

    command = planner.plan(_obs(goal=(2.0, 0.0)))

    assert command[0] >= 0.0
    last = planner.last_decision()
    assert last is not None
    assert last["route_corridor"] is None


def test_hybrid_rule_goal_stop_skips_route_corridor_diagnostics(monkeypatch) -> None:
    """Goal-stop decisions should not spend work on route-corridor diagnostics."""
    cfg = HybridRuleLocalPlannerConfig(route_guide_enabled=True)
    planner = HybridRuleLocalPlannerAdapter(cfg)

    def fail_route_geometry(_observation: dict) -> dict:
        """Fail if route geometry is evaluated during goal-stop handling."""
        raise AssertionError("route geometry should not run for goal stop")

    monkeypatch.setattr(planner._route_guide, "route_geometry", fail_route_geometry)

    command = planner.plan(_obs_with_grid(goal=(0.0, 0.0)))

    assert command == (0.0, 0.0)
    last = planner.last_decision()
    assert last is not None
    assert last["route_corridor"] is None


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


def test_hybrid_rule_rejection_diagnostics_include_moving_and_source_counts(monkeypatch) -> None:
    """Decision diagnostics should attribute static rejection sources for trace analysis."""
    cfg = HybridRuleLocalPlannerConfig(
        linear_samples=2,
        angular_samples=3,
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    planner._progress_history.extend([(0.0, 4.0), (1.0, 4.0), (2.0, 4.0)])
    stop = HybridRuleCandidate(0.0, 0.0, "stop")
    rotate = HybridRuleCandidate(0.0, 0.45, "rotate_left")
    blocked_forward = HybridRuleCandidate(0.3, 0.0, "dynamic_window")

    monkeypatch.setattr(
        planner,
        "_generate_candidates",
        lambda state, speed_cap, **kwargs: [stop, rotate, blocked_forward],
    )

    def evaluate_candidate(  # noqa: PLR0913
        *,
        candidate,
        observation,
        state,
        speed_cap,
        nearest_ped,
        progress_windows,
        route_corridor=None,
        strict_static_clearance=False,
        goal_posterior=None,
    ):
        """Return controlled candidate evaluations for rejection diagnostics."""
        if candidate == blocked_forward:
            return {
                "accepted": False,
                "reason": "static_clearance",
                "candidate": candidate,
                "min_static_clearance": 1.0,
                "hard_static_clearance": 1.05,
            }
        terms = {
            "goal_progress": 0.0,
            "path_alignment": 0.9 if candidate == rotate else 0.8,
            "speed_preference": 0.0,
            "static_clearance": 1.0,
            "dynamic_clearance": 1.0,
            "time_to_collision_margin": 1.0,
            "heading_smoothness": 0.8 if candidate == rotate else 1.0,
            "velocity_smoothness": 1.0,
            "control_effort": 0.8 if candidate == rotate else 1.0,
            "freezing_penalty": 1.0,
            "oscillation_penalty": 0.0,
            "deadlock_escape": 0.0,
            "static_recenter": 0.0,
            "static_clearance_escape": 0.0,
            "route_guide_commitment": 0.0,
        }
        return {
            "accepted": True,
            "candidate": candidate,
            "score": 5.0 if candidate == stop else 4.8,
            "terms": terms,
            "min_static_clearance": 1.2,
            "min_dynamic_clearance": float("inf"),
            "predicted_ttc": float("inf"),
        }

    monkeypatch.setattr(planner, "_evaluate_candidate", evaluate_candidate)

    command = planner.plan(_obs(goal=(4.0, 0.0)))

    assert command == (0.0, 0.0)
    last = planner.last_decision()
    assert last is not None
    assert last["planner_mode"] == "NORMAL"
    assert last["selected_source"] == "stop"
    assert last["moving_rejection_counts"] == {"static_clearance": 1}
    assert last["rejection_counts_by_source"] == {"dynamic_window": {"static_clearance": 1}}


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
            "route_guide_enabled": "false",
        }
    )
    assert cfg.max_linear_speed == pytest.approx(1.4)
    assert cfg.linear_samples == 5
    assert cfg.lookahead_distances == (0.4, 0.8)
    assert cfg.static_hard_safety_margin == pytest.approx(0.0)
    assert cfg.route_guide_enabled is False

    v3_cfg = build_hybrid_rule_local_planner_config(
        {"planner_variant": "hybrid_rule_v3_teb_like_rollout"}
    )
    assert v3_cfg.planner_variant == "hybrid_rule_v3_teb_like_rollout"
    HybridRuleLocalPlannerAdapter(v3_cfg)

    with pytest.raises(ValueError, match="Unsupported hybrid rule planner variant"):
        HybridRuleLocalPlannerAdapter(
            HybridRuleLocalPlannerConfig(planner_variant="hybrid_rule_v9_unknown")
        )

    with pytest.raises(ValueError, match="route_guide_enabled"):
        build_hybrid_rule_local_planner_config({"route_guide_enabled": "sometimes"})


def test_hybrid_rule_reset_clears_episode_diagnostics() -> None:
    """Reset should clear per-episode planner state for reproducible runs."""
    planner = HybridRuleLocalPlannerAdapter(HybridRuleLocalPlannerConfig())
    planner.plan(_obs())
    assert planner.diagnostics()["steps"] == 1

    planner.reset(seed=123)

    diagnostics = planner.diagnostics()
    assert diagnostics["steps"] == 0
    assert diagnostics["last_decision"] is None
