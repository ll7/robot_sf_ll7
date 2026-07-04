"""Tests for the diagnostic topology-guided local policy adapter."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.topology_guided_local_policy import (
    TopologyGuidedHybridRulePlannerAdapter,
    _apply_primary_route_reuse_penalty,
    _finalize_near_parity_gate_diagnostic,
    _recent_primary_route_progress,
    blend_topology_command,
    build_topology_guided_local_policy_config,
)


def _obs(
    *,
    radius: float = 0.1,
    occupied_cells: list[tuple[int, int]] | None = None,
) -> dict[str, object]:
    """Build a compact occupancy-grid observation with optional obstacles."""
    grid = np.zeros((3, 21, 21), dtype=float)
    for row, col in occupied_cells or []:
        grid[0, row, col] = 1.0
        grid[2, row, col] = 1.0
    return {
        "robot": {
            "position": [-1.6, 0.0],
            "heading": [0.0],
            "speed": [0.0],
            "radius": [radius],
        },
        "goal": {"current": [1.6, 0.0], "next": [1.6, 0.0]},
        "pedestrians": {"positions": [], "velocities": [], "count": [0], "radius": [0.3]},
        "sim": {"timestep": 0.1},
        "occupancy_grid": grid,
        "occupancy_grid_meta": {
            "origin": [-2.0, -2.0],
            "resolution": [0.2],
            "size": [4.2, 4.2],
            "use_ego_frame": [0.0],
            "center_on_robot": [0.0],
            "channel_indices": [0, 1, 2],
            "robot_pose": [0.0, 0.0, 0.0],
        },
    }


def _two_gap_wall() -> list[tuple[int, int]]:
    """Return a vertical wall with two separated passable gaps."""
    return [(row, 10) for row in range(21) if row not in {5, 15}]


def _config(**overrides):
    """Build a permissive test config for the diagnostic adapter."""
    raw = {
        "allow_testing_algorithms": True,
        "route_guide_enabled": True,
        "corridor_subgoal_enabled": True,
        "min_hypotheses": 2,
        "max_hypotheses": 2,
        "block_radius_cells": 2,
        "block_stride_cells": 1,
        "max_path_overlap": 0.95,
        "route_hypothesis": {
            "obstacle_inflation_cells": 0,
            "clearance_penalty_weight": 0.5,
            "waypoint_lookahead_cells": 3,
        },
    }
    raw.update(overrides)
    return build_topology_guided_local_policy_config(raw)


def test_nested_topology_guided_config_overrides_flat_defaults() -> None:
    """The explicit issue-3463 config block should normalize into runtime fields."""
    config = build_topology_guided_local_policy_config(
        {
            "allow_testing_algorithms": True,
            "topology_guided": {
                "schema_version": "topology_guided_hybrid_rule.v1",
                "enabled": True,
                "diagnostic_only": True,
                "candidate_required": True,
                "fallback_on_no_candidate": False,
                "arbitration_weight": 0.35,
                "near_parity_margin": 0.07,
                "min_route_progress_delta_m": 0.08,
                "stall_window_steps": 12,
            },
        }
    )

    assert config.schema_version == "topology_guided_hybrid_rule.v1"
    assert config.enabled is True
    assert config.diagnostic_only is True
    assert config.candidate_required is True
    assert config.fallback_on_no_candidate is False
    assert config.arbitration_weight == pytest.approx(0.35)
    assert config.near_parity_route_distance_slack_ratio == pytest.approx(0.07)
    assert config.min_route_progress_delta_m == pytest.approx(0.08)
    assert config.stall_window_steps == 12


def test_nested_topology_guided_config_does_not_mutate_caller_mapping() -> None:
    """Nested topology overrides should be copied before local compatibility rewrites."""
    cfg = {
        "topology_guided": {
            "near_parity_margin": 0.07,
            "arbitration_weight": 0.35,
        }
    }

    build_topology_guided_local_policy_config(cfg)

    assert cfg == {
        "topology_guided": {
            "near_parity_margin": 0.07,
            "arbitration_weight": 0.35,
        }
    }


def test_topology_arbitration_default_is_explicit_issue_3463_value() -> None:
    """Missing legacy config uses the explicit topology arbitration default."""
    config = build_topology_guided_local_policy_config({})

    assert config.arbitration_weight == pytest.approx(0.35)


def test_blend_topology_command_respects_weight_and_limits() -> None:
    """Explicit arbitration should be finite, monotone, and clipped to command limits."""
    limits = {"max_linear": 1.0, "max_angular": 0.5}

    assert blend_topology_command((0.2, -0.2), (0.8, 0.4), weight=0.0, command_limits=limits) == (
        pytest.approx(0.2),
        pytest.approx(-0.2),
    )
    assert blend_topology_command((0.2, -0.2), (0.8, 0.4), weight=1.0, command_limits=limits) == (
        pytest.approx(0.8),
        pytest.approx(0.4),
    )
    assert blend_topology_command((0.2, -0.2), (2.0, 2.0), weight=0.5, command_limits=limits) == (
        pytest.approx(1.0),
        pytest.approx(0.5),
    )


def test_blend_topology_command_rejects_invalid_weight() -> None:
    """Invalid arbitration weights fail closed instead silently changing behavior."""
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        blend_topology_command(
            (0.2, 0.0),
            (0.8, 0.0),
            weight=1.5,
            command_limits={"max_linear": 1.0, "max_angular": 1.0},
        )


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan")])
def test_topology_arbitration_config_rejects_invalid_weight(weight: float) -> None:
    """The config parser validates topology arbitration strength fail-closed."""
    with pytest.raises(ValueError, match="arbitration_weight"):
        _config(arbitration_weight=weight)


def test_topology_guided_selector_finds_two_distinct_route_hypotheses() -> None:
    """A two-gap wall should expose two selectable route hypotheses."""
    planner = TopologyGuidedHybridRulePlannerAdapter(_config())

    decision = planner._hypotheses_for_observation(_obs(occupied_cells=_two_gap_wall()))

    assert decision["status"] == "ok"
    assert decision["hypothesis_count"] == 2
    assert {item["hypothesis_id"] for item in decision["hypotheses"]} >= {"primary_route"}
    selected = next(
        item
        for item in decision["hypotheses"]
        if item["hypothesis_id"] == decision["selected_hypothesis_id"]
    )
    rejected = [
        item
        for item in decision["hypotheses"]
        if item["hypothesis_id"] != decision["selected_hypothesis_id"]
    ]
    assert selected["route_corridor"]["route_path_cell_count"] > 1
    assert selected["route_corridor"]["route_waypoint_world"]
    assert selected["static_clearance_min_m"] is not None
    assert decision["selected_score"] == selected["score"]
    assert decision["selection_score"] == selected["selection_score"]
    assert decision["near_parity_gate_enabled"] is False
    assert decision["near_parity_gate_reason"] == "disabled"
    assert decision["primary_vs_best_alternative_route_distance"] is not None
    assert decision["selected_static_clearance_min_m"] is not None
    assert decision["best_alternative_static_clearance_min_m"] is not None
    assert selected["selection_outcome"] == "selected"
    assert selected["rejection_reason"] is None
    assert selected["score_components"].keys() == {
        "length_penalty",
        "static_clearance_bonus",
    }
    assert all(item["selection_outcome"] == "rejected" for item in rejected)
    assert all(item["rejection_reason"] == "lower_topology_selection_score" for item in rejected)
    assert all(item["score_margin_to_selected"] >= 0.0 for item in decision["hypotheses"])


def test_near_parity_gate_can_select_non_primary_hypothesis() -> None:
    """The diagnostic gate should be able to promote a near-parity alternative route."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(
            near_parity_diversity_gate_enabled=True,
            near_parity_route_distance_slack_m=100.0,
            near_parity_route_distance_slack_ratio=100.0,
            near_parity_static_clearance_floor_m=100.0,
            near_parity_diversity_bonus=100.0,
        )
    )

    decision = planner._hypotheses_for_observation(_obs(occupied_cells=_two_gap_wall()))

    assert decision["status"] == "ok"
    assert decision["near_parity_gate_enabled"] is True
    assert decision["selected_hypothesis_id"] != "primary_route"
    assert decision["near_parity_gate_reason"] == "selected_non_primary_near_parity"
    assert decision["primary_vs_best_alternative_route_distance"] is not None
    assert decision["selected_static_clearance_min_m"] is not None
    assert decision["best_alternative_static_clearance_min_m"] is not None
    selected = next(
        item
        for item in decision["hypotheses"]
        if item["hypothesis_id"] == decision["selected_hypothesis_id"]
    )
    primary = next(
        item for item in decision["hypotheses"] if item["hypothesis_id"] == "primary_route"
    )
    assert selected["selection_score"] > selected["score"]
    assert primary["selection_outcome"] == "rejected"
    assert primary["rejection_reason"] == "near_parity_diversity_gate"


def test_near_parity_finalize_preserves_blocker_for_raw_non_primary_winner() -> None:
    """A raw non-primary winner must not look like an eligible near-parity gate activation."""
    selected = {
        "hypothesis_id": "masked_cell_1_1",
        "static_clearance_min_m": 0.1,
        "near_parity_gate_reason": "disabled",
    }
    primary = {
        "hypothesis_id": "primary_route",
        "static_clearance_min_m": 0.2,
        "near_parity_gate_reason": "disabled",
    }
    diagnostic = {
        "near_parity_gate_reason": "route_distance_exceeds_slack",
        "selected_static_clearance_min_m": None,
    }

    result = _finalize_near_parity_gate_diagnostic(
        _config(near_parity_diversity_gate_enabled=True),
        [primary, selected],
        selected,
        diagnostic,
    )

    assert result["near_parity_gate_reason"] == "route_distance_exceeds_slack"
    assert selected["near_parity_gate_reason"] == "route_distance_exceeds_slack"
    assert primary["near_parity_gate_reason"] == "disabled"


def test_near_parity_finalize_requires_boost_for_selected_non_primary_reason() -> None:
    """An eligible raw winner without a score boost should keep the eligibility reason."""
    selected = {
        "hypothesis_id": "masked_cell_1_1",
        "score": -1.0,
        "selection_score": -1.0,
        "static_clearance_min_m": 0.1,
        "near_parity_gate_reason": "eligible_near_parity_alternative",
    }
    primary = {
        "hypothesis_id": "primary_route",
        "score": -1.2,
        "selection_score": -1.2,
        "static_clearance_min_m": 0.2,
        "near_parity_gate_reason": "disabled",
    }
    diagnostic = {
        "near_parity_gate_reason": "eligible_near_parity_alternative",
        "selected_static_clearance_min_m": None,
    }

    result = _finalize_near_parity_gate_diagnostic(
        _config(near_parity_diversity_gate_enabled=True),
        [primary, selected],
        selected,
        diagnostic,
    )

    assert result["near_parity_gate_reason"] == "eligible_near_parity_alternative"
    assert selected["near_parity_gate_reason"] == "eligible_near_parity_alternative"


def test_topology_guided_policy_fails_closed_without_required_inputs() -> None:
    """Missing topology inputs should stop instead of silently claiming success."""
    planner = TopologyGuidedHybridRulePlannerAdapter(_config())
    observation = _obs(occupied_cells=[])
    observation.pop("occupancy_grid")

    command = planner.plan(observation)

    assert command == (0.0, 0.0)
    diagnostics = planner.diagnostics()
    assert diagnostics["topology_guided"]["status_counts"] == {"not_available": 1}
    assert diagnostics["last_decision"]["planner_mode"] == "TOPOLOGY_FAIL_CLOSED"
    assert diagnostics["last_decision"]["selected_source"] == "topology_fail_closed"
    assert diagnostics["last_decision"]["topology_lane_status"] == "failed"


def test_topology_guided_policy_records_fallback_only_status() -> None:
    """Diagnostic fallback remains visible and is not counted as topology success."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(fail_closed_on_missing_inputs=False, fallback_on_no_candidate=True)
    )
    observation = _obs(occupied_cells=[])
    observation.pop("occupancy_grid")

    linear, angular = planner.plan(observation)

    assert np.isfinite(linear)
    assert np.isfinite(angular)
    diagnostics = planner.diagnostics()
    assert diagnostics["topology_guided"]["status_counts"] == {"not_available": 1}
    assert diagnostics["last_decision"]["topology_lane_status"] == "fallback_only"
    assert diagnostics["last_decision"]["topology_fallback_status"] == "not_available"
    assert (
        diagnostics["last_decision"]["topology_guided_config"]["fallback_on_no_candidate"] is True
    )


def test_topology_guided_policy_candidate_required_can_fail_closed() -> None:
    """Explicit candidate-required mode stops when no topology candidate is available."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(
            fail_closed_on_missing_inputs=False,
            candidate_required=True,
            fallback_on_no_candidate=False,
        )
    )
    observation = _obs(occupied_cells=[])
    observation.pop("occupancy_grid")

    command = planner.plan(observation)

    assert command == (0.0, 0.0)
    diagnostics = planner.diagnostics()
    assert diagnostics["last_decision"]["planner_mode"] == "TOPOLOGY_FAIL_CLOSED"
    assert diagnostics["last_decision"]["topology_lane_status"] == "failed"


def test_topology_guided_policy_records_selected_hypothesis_diagnostics() -> None:
    """Successful topology selection should be visible in planner diagnostics."""
    planner = TopologyGuidedHybridRulePlannerAdapter(_config())

    linear, angular = planner.plan(_obs(occupied_cells=_two_gap_wall()))

    assert np.isfinite(linear)
    assert np.isfinite(angular)
    diagnostics = planner.diagnostics()
    assert diagnostics["topology_guided"]["status_counts"] == {"ok": 1}
    assert diagnostics["topology_guided"]["selected_hypothesis_counts"]
    route_corridor = diagnostics["last_decision"]["route_corridor"]
    assert route_corridor["topology_status"] == "ok"
    assert route_corridor["topology_guided_config"]["schema_version"] == (
        "topology_guided_hybrid_rule.v1"
    )
    assert route_corridor["topology_guided_config"]["arbitration_weight"] == pytest.approx(0.35)
    assert len(route_corridor["topology_hypotheses"]) == 2
    assert all("score_components" in item for item in route_corridor["topology_hypotheses"])
    assert any(
        item["rejection_reason"] == "lower_topology_selection_score"
        for item in route_corridor["topology_hypotheses"]
        if item["selection_outcome"] == "rejected"
    )


def test_topology_hypothesis_can_select_local_command_source() -> None:
    """Available topology hypotheses should materially affect command selection."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(
            route_guide_enabled=False,
            corridor_subgoal_enabled=False,
            goal_progress_weight=0.0,
            path_alignment_weight=0.0,
            speed_preference_weight=0.0,
            static_clearance_weight=0.0,
            dynamic_clearance_weight=0.0,
            ttc_weight=0.0,
            heading_smoothness_weight=0.0,
            velocity_smoothness_weight=0.0,
            control_effort_weight=0.0,
            freezing_weight=0.0,
            oscillation_weight=0.0,
            corridor_subgoal_route_progress_weight=8.0,
            corridor_subgoal_tangent_alignment_weight=4.0,
        )
    )

    linear, angular = planner.plan(_obs(occupied_cells=_two_gap_wall()))

    diagnostics = planner.diagnostics()
    last_decision = diagnostics["last_decision"]
    assert np.isfinite(linear)
    assert np.isfinite(angular)
    assert last_decision["selected_source"] == "topology_hypothesis"
    assert last_decision["route_corridor"]["topology_status"] == "ok"
    assert last_decision["route_corridor"]["topology_hypothesis"]["hypothesis_id"]
    assert last_decision["selected_terms"]["corridor_subgoal_tangent_alignment"] > 0.0
    assert (
        last_decision["topology_command_influence"]["selected_hypothesis_id"]
        == last_decision["route_corridor"]["topology_hypothesis"]["hypothesis_id"]
    )
    influence = last_decision["topology_command_influence"]
    assert influence["schema_version"] == "topology-command-influence.v1"
    assert influence["arbitration_weight"] == pytest.approx(0.35)
    assert influence["projected_command"] == last_decision["selected_command"]
    assert isinstance(influence["projection_applied"], bool)
    assert influence["command_limits"]["max_linear"] >= abs(influence["projected_command"][0])
    assert influence["command_limits"]["max_angular"] >= abs(influence["projected_command"][1])
    assert last_decision["topology_guided_config"]["arbitration_weight"] == pytest.approx(0.35)


def test_topology_hypothesis_command_blends_headings_across_pi_boundary() -> None:
    """Route tangent and waypoint headings should not cancel across the wrap boundary."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(topology_command_turn_in_place_error=4.0)
    )
    tangent_heading = float(np.pi - 0.1)
    waypoint_heading = float(-np.pi + 0.1)
    waypoint = np.array([np.cos(waypoint_heading), np.sin(waypoint_heading)], dtype=float)

    candidate = planner._topology_hypothesis_candidate(
        state={
            "robot_pos": np.zeros(2, dtype=float),
            "heading": 0.0,
            "current_speed": 0.0,
        },
        speed_cap=1.0,
        route_corridor={
            "topology_status": "ok",
            "route_waypoint_world": waypoint,
            "route_tangent_heading": tangent_heading,
        },
        bounds=(-1.0, 1.0, -10.0, 10.0),
    )

    assert candidate is not None
    assert candidate.source == "topology_hypothesis"
    assert candidate.linear == 0.0
    assert abs(candidate.angular) > 1.0


def test_topology_guided_policy_resets_stale_topology_decision() -> None:
    """A later plan step with missing inputs must not reuse the prior topology result."""
    planner = TopologyGuidedHybridRulePlannerAdapter(_config())
    planner.plan(_obs(occupied_cells=_two_gap_wall()))
    observation = _obs(occupied_cells=[])
    observation.pop("occupancy_grid")

    command = planner.plan(observation)

    assert command == (0.0, 0.0)
    diagnostics = planner.diagnostics()
    assert diagnostics["topology_guided"]["last_topology_decision"]["status"] == "not_available"
    assert diagnostics["last_decision"]["selected_source"] == "topology_fail_closed"


def test_topology_guided_policy_handles_zero_min_without_hypotheses() -> None:
    """Empty route-hypothesis lists should return diagnostics instead of crashing."""
    planner = TopologyGuidedHybridRulePlannerAdapter(_config(min_hypotheses=0))
    observation = _obs(occupied_cells=[])
    observation["goal"]["current"] = [-1.6, 0.0]
    observation["goal"]["next"] = [-1.6, 0.0]

    decision = planner._hypotheses_for_observation(observation)

    assert decision["status"] == "insufficient_hypotheses"
    assert decision["reason"] in {"no_hypotheses_available", "fewer_than_min_distinct_routes"}
    assert decision["hypothesis_count"] == 0


def test_default_no_reuse_penalty_fields_in_decision() -> None:
    """Default config should expose reuse-penalty fields but not apply the penalty."""
    planner = TopologyGuidedHybridRulePlannerAdapter(_config())
    decision = planner._hypotheses_for_observation(_obs(occupied_cells=_two_gap_wall()))

    assert decision["status"] == "ok"
    assert decision["reuse_penalty_applied"] is False
    assert decision["reuse_penalty_reason"] is None
    assert decision["recent_primary_selection_count"] == 0
    assert decision["eligible_near_parity_alternative_exists"] is False


def test_reuse_penalty_applied_after_repeated_primary_selections() -> None:
    """When enabled, repeated primary selections with eligible alternatives should trigger penalty."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(
            primary_route_reuse_penalty_enabled=True,
            primary_route_reuse_penalty_weight=0.1,
            primary_route_reuse_penalty_cooldown_steps=5,
            primary_route_reuse_penalty_min_prior_primary_selections=1,
            near_parity_diversity_gate_enabled=True,
            near_parity_route_distance_slack_m=100.0,
            near_parity_route_distance_slack_ratio=100.0,
            near_parity_static_clearance_floor_m=100.0,
            near_parity_diversity_bonus=0.0,
        )
    )
    obs = _obs(occupied_cells=_two_gap_wall())

    planner._hypotheses_for_observation(obs)
    decision = planner._hypotheses_for_observation(obs)

    assert decision["status"] == "ok"
    assert decision["recent_primary_selection_count"] == 1
    assert decision["reuse_penalty_applied"] is True
    assert decision["reuse_penalty_reason"]


def test_reuse_penalty_not_applied_when_disabled() -> None:
    """Penalty should not apply when primary_route_reuse_penalty_enabled is False."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(
            primary_route_reuse_penalty_enabled=False,
            primary_route_reuse_penalty_weight=10.0,
        )
    )
    obs = _obs(occupied_cells=_two_gap_wall())

    for _ in range(5):
        decision = planner._hypotheses_for_observation(obs)

    assert decision["reuse_penalty_applied"] is False


def test_reuse_penalty_allows_alternative_to_win() -> None:
    """When penalty is large enough, a near-parity alternative should win over primary_route."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(
            primary_route_reuse_penalty_enabled=True,
            primary_route_reuse_penalty_weight=200.0,
            primary_route_reuse_penalty_cooldown_steps=10,
            primary_route_reuse_penalty_min_prior_primary_selections=3,
            near_parity_diversity_gate_enabled=True,
            near_parity_route_distance_slack_m=100.0,
            near_parity_route_distance_slack_ratio=100.0,
            near_parity_static_clearance_floor_m=100.0,
            near_parity_diversity_bonus=0.0,
        )
    )
    obs = _obs(occupied_cells=_two_gap_wall())

    for _ in range(6):
        decision = planner._hypotheses_for_observation(obs)

    assert decision["status"] == "ok"
    assert decision["recent_primary_selection_count"] >= 3
    assert decision["selected_hypothesis_id"] != "primary_route"


def test_reuse_penalty_diagnostics_fields_present() -> None:
    """Reuse-penalty diagnostics should be present in decision and route output."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(
            primary_route_reuse_penalty_enabled=True,
            primary_route_reuse_penalty_weight=0.1,
            primary_route_reuse_penalty_cooldown_steps=2,
            primary_route_reuse_penalty_min_prior_primary_selections=1,
            near_parity_diversity_gate_enabled=True,
            near_parity_route_distance_slack_m=100.0,
            near_parity_route_distance_slack_ratio=100.0,
            near_parity_static_clearance_floor_m=100.0,
            near_parity_diversity_bonus=0.0,
        )
    )
    obs = _obs(occupied_cells=_two_gap_wall())
    planner._hypotheses_for_observation(obs)

    decision = planner._hypotheses_for_observation(obs)

    assert "reuse_penalty_applied" in decision
    assert "reuse_penalty_reason" in decision
    assert "recent_primary_selection_count" in decision
    assert "eligible_near_parity_alternative_exists" in decision

    route_corridor = planner._route_corridor_diagnostics(obs, current_time=1.0)
    assert route_corridor is not None
    reuse_penalty = route_corridor["topology_reuse_penalty"]
    assert reuse_penalty["reuse_penalty_applied"] is True
    assert reuse_penalty["reuse_penalty_reason"]
    assert reuse_penalty["recent_primary_selection_count"] >= 1
    assert reuse_penalty["eligible_near_parity_alternative_exists"] is True


def test_reuse_penalty_resets_after_reset() -> None:
    """The reuse-penalty state should be cleared when reset is called."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(
            primary_route_reuse_penalty_enabled=True,
            primary_route_reuse_penalty_cooldown_steps=5,
            primary_route_reuse_penalty_min_prior_primary_selections=1,
        )
    )
    obs = _obs(occupied_cells=_two_gap_wall())
    for _ in range(3):
        planner._hypotheses_for_observation(obs)
    assert planner._total_primary_selections > 0

    planner.reset(seed=0)

    assert planner._total_primary_selections == 0
    assert len(planner._recent_primary_selections) == 0


def test_apply_primary_route_reuse_penalty_no_primary_hypothesis() -> None:
    """Penalty function should not crash when primary_route hypothesis is absent."""
    from collections import deque

    config = _config(primary_route_reuse_penalty_enabled=True)
    hypotheses = [{"hypothesis_id": "masked_cell_1_1", "score": -5.0, "selection_score": -5.0}]
    recent = deque([("masked_cell_1_1", 10.0)], maxlen=3)

    result = _apply_primary_route_reuse_penalty(config, hypotheses, recent)

    assert result["reuse_penalty_applied"] is False
    assert result["recent_primary_selection_count"] == 0


def test_apply_primary_route_reuse_penalty_not_eligible_when_no_near_parity_alternative() -> None:
    """Penalty should not apply when no alternative has eligible near-parity gate reason."""
    from collections import deque

    config = _config(
        primary_route_reuse_penalty_enabled=True,
        primary_route_reuse_penalty_min_prior_primary_selections=1,
    )
    primary = {
        "hypothesis_id": "primary_route",
        "score": -5.0,
        "selection_score": -5.0,
    }
    alt = {
        "hypothesis_id": "masked_cell_1_1",
        "score": -6.0,
        "selection_score": -6.0,
        "near_parity_gate_reason": "route_distance_exceeds_slack",
    }
    recent = deque([("primary_route", 10.0), ("primary_route", 9.0)], maxlen=3)

    result = _apply_primary_route_reuse_penalty(
        config,
        [primary, alt],
        recent,
    )

    assert result["reuse_penalty_applied"] is False
    assert result["eligible_near_parity_alternative_exists"] is False


def test_progress_gate_suppresses_penalty_when_threshold_met() -> None:
    """Progress gate should suppress reuse penalty when recent progress meets threshold."""
    from collections import deque

    config = _config(
        primary_route_reuse_penalty_enabled=True,
        primary_route_reuse_penalty_weight=10.0,
        primary_route_reuse_penalty_cooldown_steps=5,
        primary_route_reuse_penalty_min_prior_primary_selections=1,
        near_parity_diversity_gate_enabled=True,
        near_parity_route_distance_slack_m=100.0,
        near_parity_route_distance_slack_ratio=100.0,
        near_parity_static_clearance_floor_m=100.0,
        near_parity_diversity_bonus=0.0,
        primary_route_progress_gate_enabled=True,
        primary_route_progress_gate_threshold_m=0.1,
    )
    primary = {
        "hypothesis_id": "primary_route",
        "score": -5.0,
        "selection_score": -5.0,
    }
    alt = {
        "hypothesis_id": "masked_cell_1_1",
        "score": -6.0,
        "selection_score": -6.0,
        "near_parity_gate_reason": "eligible_near_parity_alternative",
    }
    recent = deque([("primary_route", 1.5), ("primary_route", 1.3)], maxlen=5)

    result = _apply_primary_route_reuse_penalty(config, [primary, alt], recent)

    assert result["reuse_penalty_applied"] is False
    assert result["primary_route_progress_gate_satisfied"] is True
    assert result["reuse_penalty_suppressed_by_progress"] is True
    assert result["primary_route_recent_progress_m"] == pytest.approx(0.2)
    assert result["primary_route_recent_progress_sample_count"] == 2
    assert "progress_gate_suppressed" in result["reuse_penalty_reason"]
    assert float(primary["selection_score"]) == -5.0


def test_progress_gate_does_not_suppress_when_threshold_not_met() -> None:
    """Progress gate should not suppress penalty when recent progress is below threshold."""
    from collections import deque

    config = _config(
        primary_route_reuse_penalty_enabled=True,
        primary_route_reuse_penalty_weight=10.0,
        primary_route_reuse_penalty_cooldown_steps=5,
        primary_route_reuse_penalty_min_prior_primary_selections=1,
        near_parity_diversity_gate_enabled=True,
        near_parity_route_distance_slack_m=100.0,
        near_parity_route_distance_slack_ratio=100.0,
        near_parity_static_clearance_floor_m=100.0,
        near_parity_diversity_bonus=0.0,
        primary_route_progress_gate_enabled=True,
        primary_route_progress_gate_threshold_m=0.5,
    )
    primary = {
        "hypothesis_id": "primary_route",
        "score": -5.0,
        "selection_score": -5.0,
    }
    alt = {
        "hypothesis_id": "masked_cell_1_1",
        "score": -6.0,
        "selection_score": -6.0,
        "near_parity_gate_reason": "eligible_near_parity_alternative",
    }
    recent = deque([("primary_route", 1.1), ("primary_route", 1.0)], maxlen=5)

    result = _apply_primary_route_reuse_penalty(config, [primary, alt], recent)

    assert result["reuse_penalty_applied"] is True
    assert result["primary_route_progress_gate_satisfied"] is False
    assert result["reuse_penalty_suppressed_by_progress"] is False
    assert result["primary_route_recent_progress_m"] == pytest.approx(0.1)
    assert float(primary["selection_score"]) < -5.0


def test_progress_gate_disabled_does_not_suppress() -> None:
    """When progress gate is disabled, penalty should apply normally regardless of progress."""
    from collections import deque

    config = _config(
        primary_route_reuse_penalty_enabled=True,
        primary_route_reuse_penalty_weight=10.0,
        primary_route_reuse_penalty_cooldown_steps=5,
        primary_route_reuse_penalty_min_prior_primary_selections=1,
        near_parity_diversity_gate_enabled=True,
        near_parity_route_distance_slack_m=100.0,
        near_parity_route_distance_slack_ratio=100.0,
        near_parity_static_clearance_floor_m=100.0,
        near_parity_diversity_bonus=0.0,
        primary_route_progress_gate_enabled=False,
        primary_route_progress_gate_threshold_m=0.0,
    )
    primary = {
        "hypothesis_id": "primary_route",
        "score": -5.0,
        "selection_score": -5.0,
    }
    alt = {
        "hypothesis_id": "masked_cell_1_1",
        "score": -6.0,
        "selection_score": -6.0,
        "near_parity_gate_reason": "eligible_near_parity_alternative",
    }
    recent = deque([("primary_route", 1.0), ("primary_route", 1.0)], maxlen=5)

    result = _apply_primary_route_reuse_penalty(config, [primary, alt], recent)

    assert result["reuse_penalty_applied"] is True
    assert result["reuse_penalty_suppressed_by_progress"] is False


def test_progress_gate_diagnostic_fields_present_in_decision() -> None:
    """Progress gate diagnostic fields should be present in topology decision output."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(
            primary_route_reuse_penalty_enabled=True,
            primary_route_reuse_penalty_weight=0.1,
            primary_route_reuse_penalty_cooldown_steps=5,
            primary_route_reuse_penalty_min_prior_primary_selections=1,
            near_parity_diversity_gate_enabled=True,
            near_parity_route_distance_slack_m=100.0,
            near_parity_route_distance_slack_ratio=100.0,
            near_parity_static_clearance_floor_m=100.0,
            near_parity_diversity_bonus=0.0,
            primary_route_progress_gate_enabled=True,
            primary_route_progress_gate_threshold_m=0.1,
        )
    )
    obs = _obs(occupied_cells=_two_gap_wall())
    planner._hypotheses_for_observation(obs)

    decision = planner._hypotheses_for_observation(obs)

    assert "primary_route_recent_progress_m" in decision
    assert "primary_route_recent_progress_sample_count" in decision
    assert "primary_route_progress_gate_satisfied" in decision
    assert "reuse_penalty_suppressed_by_progress" in decision


def test_progress_gate_diagnostic_fields_in_route_corridor() -> None:
    """Progress gate diagnostic fields should be exposed in route corridor output."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(
            primary_route_reuse_penalty_enabled=True,
            primary_route_reuse_penalty_weight=0.1,
            primary_route_reuse_penalty_cooldown_steps=5,
            primary_route_reuse_penalty_min_prior_primary_selections=1,
            near_parity_diversity_gate_enabled=True,
            near_parity_route_distance_slack_m=100.0,
            near_parity_route_distance_slack_ratio=100.0,
            near_parity_static_clearance_floor_m=100.0,
            near_parity_diversity_bonus=0.0,
            primary_route_progress_gate_enabled=True,
            primary_route_progress_gate_threshold_m=0.1,
        )
    )
    obs = _obs(occupied_cells=_two_gap_wall())
    planner._hypotheses_for_observation(obs)
    planner._hypotheses_for_observation(obs)

    route_corridor = planner._route_corridor_diagnostics(obs, current_time=1.0)
    assert route_corridor is not None
    reuse_penalty = route_corridor["topology_reuse_penalty"]
    assert "primary_route_recent_progress_m" in reuse_penalty
    assert "primary_route_recent_progress_sample_count" in reuse_penalty
    assert "primary_route_progress_gate_satisfied" in reuse_penalty
    assert "reuse_penalty_suppressed_by_progress" in reuse_penalty


def test_progress_gate_no_primary_selections_yields_zero_progress() -> None:
    """When no primary selections exist, recent progress should be zero and gate unsatisfied."""
    from collections import deque

    config = _config(
        primary_route_reuse_penalty_enabled=True,
        primary_route_reuse_penalty_cooldown_steps=5,
        primary_route_reuse_penalty_min_prior_primary_selections=1,
        primary_route_progress_gate_enabled=True,
        primary_route_progress_gate_threshold_m=0.1,
    )
    primary = {
        "hypothesis_id": "primary_route",
        "score": -5.0,
        "selection_score": -5.0,
    }
    alt = {
        "hypothesis_id": "masked_cell_1_1",
        "score": -6.0,
        "selection_score": -6.0,
        "near_parity_gate_reason": "eligible_near_parity_alternative",
    }
    recent = deque([("masked_cell_1_1", 10.0)], maxlen=3)

    result = _apply_primary_route_reuse_penalty(config, [primary, alt], recent)

    assert result["primary_route_recent_progress_m"] == 0.0
    assert result["primary_route_recent_progress_sample_count"] == 0
    assert result["primary_route_progress_gate_satisfied"] is False
    assert result["reuse_penalty_suppressed_by_progress"] is False


def test_progress_gate_single_primary_selection_does_not_suppress() -> None:
    """One primary-route sample is not enough to prove recent route progress."""
    from collections import deque

    config = _config(
        primary_route_reuse_penalty_enabled=True,
        primary_route_reuse_penalty_weight=10.0,
        primary_route_reuse_penalty_cooldown_steps=5,
        primary_route_reuse_penalty_min_prior_primary_selections=1,
        primary_route_progress_gate_enabled=True,
        primary_route_progress_gate_threshold_m=0.1,
    )
    primary = {
        "hypothesis_id": "primary_route",
        "score": -5.0,
        "selection_score": -5.0,
    }
    alt = {
        "hypothesis_id": "masked_cell_1_1",
        "score": -6.0,
        "selection_score": -6.0,
        "near_parity_gate_reason": "eligible_near_parity_alternative",
    }
    recent = deque([("primary_route", 1.0)], maxlen=3)

    result = _apply_primary_route_reuse_penalty(config, [primary, alt], recent)

    assert result["reuse_penalty_applied"] is True
    assert result["primary_route_recent_progress_m"] == 0.0
    assert result["primary_route_recent_progress_sample_count"] == 1
    assert result["primary_route_progress_gate_satisfied"] is False


# ---------------------------------------------------------------------------
# Issue #3463: route-progress stall hardening + explicit gate-threshold config
# ---------------------------------------------------------------------------


def test_legacy_progress_accounting_uses_oldest_minus_newest() -> None:
    """Default (legacy) accounting compares the oldest and newest recent samples."""
    config = _config()

    progress = _recent_primary_route_progress(config, [3.0, 10.0, 2.0])

    # Oldest (3.0) - newest (2.0) = 1.0; the transient 10.0 spike is ignored.
    assert config.primary_route_progress_gate_use_monotone_accounting is False
    assert progress == pytest.approx(1.0)


def test_legacy_progress_accounting_clamps_replan_bump_to_zero() -> None:
    """A single re-plan that raises the newest remaining masks progress under legacy mode."""
    config = _config()

    # Robot advanced from 3.0 -> 2.0 then a re-plan bumped the latest remaining to 3.5.
    progress = _recent_primary_route_progress(config, [3.0, 2.0, 3.5])

    # Legacy oldest-minus-newest = 3.0 - 3.5 < 0, clamped to 0.0: the stall failure mode.
    assert progress == 0.0


def test_monotone_progress_accounting_survives_replan_bump() -> None:
    """Monotone accounting uses the max baseline so a re-plan bump does not mask progress."""
    config = _config(primary_route_progress_gate_use_monotone_accounting=True)

    progress = _recent_primary_route_progress(config, [3.0, 2.0, 3.5])

    # max(3.5, 3.0, 2.0) - newest(3.5) = 0.0 here because the bump *is* the newest.
    assert progress == 0.0
    # But when the newest is the smallest, monotone recovers the full advance.
    progress2 = _recent_primary_route_progress(config, [3.0, 4.0, 1.0])
    assert progress2 == pytest.approx(3.0)


def test_monotone_accounting_does_not_change_simple_monotone_progress() -> None:
    """On a cleanly decreasing window both accounting modes agree (regression safety)."""
    samples = [5.0, 4.0, 3.0, 2.0]
    legacy = _recent_primary_route_progress(_config(), samples)
    monotone = _recent_primary_route_progress(
        _config(primary_route_progress_gate_use_monotone_accounting=True), samples
    )

    assert legacy == pytest.approx(3.0)
    assert monotone == pytest.approx(3.0)


def test_monotone_accounting_suppresses_penalty_through_replan_noise() -> None:
    """Monotone accounting can prove progress and suppress the penalty despite a re-plan bump."""
    from collections import deque

    config = _config(
        primary_route_reuse_penalty_enabled=True,
        primary_route_reuse_penalty_weight=10.0,
        primary_route_reuse_penalty_cooldown_steps=5,
        primary_route_reuse_penalty_min_prior_primary_selections=1,
        near_parity_diversity_gate_enabled=True,
        near_parity_route_distance_slack_m=100.0,
        near_parity_route_distance_slack_ratio=100.0,
        near_parity_static_clearance_floor_m=100.0,
        near_parity_diversity_bonus=0.0,
        primary_route_progress_gate_enabled=True,
        primary_route_progress_gate_threshold_m=0.3,
        primary_route_progress_gate_use_monotone_accounting=True,
    )
    primary = {"hypothesis_id": "primary_route", "score": -5.0, "selection_score": -5.0}
    alt = {
        "hypothesis_id": "masked_cell_1_1",
        "score": -6.0,
        "selection_score": -6.0,
        "near_parity_gate_reason": "eligible_near_parity_alternative",
    }
    # Real progress 4.0 -> 3.0 with a transient re-plan bump to 4.2 in the middle.
    recent = deque(
        [("primary_route", 4.0), ("primary_route", 4.2), ("primary_route", 3.0)], maxlen=5
    )

    result = _apply_primary_route_reuse_penalty(config, [primary, alt], recent)

    assert result["primary_route_progress_accounting_mode"] == "monotone"
    assert result["primary_route_recent_progress_m"] == pytest.approx(1.2)
    assert result["primary_route_progress_gate_satisfied"] is True
    assert result["reuse_penalty_suppressed_by_progress"] is True
    assert result["reuse_penalty_applied"] is False
    assert float(primary["selection_score"]) == -5.0


def test_legacy_accounting_fires_penalty_through_replan_noise() -> None:
    """The same re-plan noise under legacy accounting fails to prove progress (baseline)."""
    from collections import deque

    config = _config(
        primary_route_reuse_penalty_enabled=True,
        primary_route_reuse_penalty_weight=10.0,
        primary_route_reuse_penalty_cooldown_steps=5,
        primary_route_reuse_penalty_min_prior_primary_selections=1,
        near_parity_diversity_gate_enabled=True,
        near_parity_route_distance_slack_m=100.0,
        near_parity_route_distance_slack_ratio=100.0,
        near_parity_static_clearance_floor_m=100.0,
        near_parity_diversity_bonus=0.0,
        primary_route_progress_gate_enabled=True,
        primary_route_progress_gate_threshold_m=0.3,
    )
    primary = {"hypothesis_id": "primary_route", "score": -5.0, "selection_score": -5.0}
    alt = {
        "hypothesis_id": "masked_cell_1_1",
        "score": -6.0,
        "selection_score": -6.0,
        "near_parity_gate_reason": "eligible_near_parity_alternative",
    }
    recent = deque(
        [("primary_route", 3.2), ("primary_route", 3.0), ("primary_route", 3.4)], maxlen=5
    )

    result = _apply_primary_route_reuse_penalty(config, [primary, alt], recent)

    # Legacy oldest(3.2) - newest(3.4) clamps to 0.0; gate not satisfied; penalty fires.
    assert result["primary_route_progress_accounting_mode"] == "legacy"
    assert result["primary_route_recent_progress_m"] == 0.0
    assert result["primary_route_progress_gate_satisfied"] is False
    assert result["reuse_penalty_applied"] is True


def test_progress_gate_min_samples_config_parses_and_applies() -> None:
    """Explicit min-samples threshold should parse and gate progress evaluation."""
    config = _config(
        primary_route_reuse_penalty_enabled=True,
        primary_route_reuse_penalty_weight=10.0,
        primary_route_reuse_penalty_cooldown_steps=5,
        primary_route_reuse_penalty_min_prior_primary_selections=1,
        near_parity_diversity_gate_enabled=True,
        near_parity_route_distance_slack_m=100.0,
        near_parity_route_distance_slack_ratio=100.0,
        near_parity_static_clearance_floor_m=100.0,
        near_parity_diversity_bonus=0.0,
        primary_route_progress_gate_enabled=True,
        primary_route_progress_gate_threshold_m=0.1,
        primary_route_progress_gate_min_samples=3,
    )
    assert config.primary_route_progress_gate_min_samples == 3

    from collections import deque

    primary = {"hypothesis_id": "primary_route", "score": -5.0, "selection_score": -5.0}
    alt = {
        "hypothesis_id": "masked_cell_1_1",
        "score": -6.0,
        "selection_score": -6.0,
        "near_parity_gate_reason": "eligible_near_parity_alternative",
    }
    # Two samples with clear progress, but the threshold now demands three samples.
    recent = deque([("primary_route", 2.0), ("primary_route", 1.0)], maxlen=5)

    result = _apply_primary_route_reuse_penalty(config, [primary, alt], recent)

    assert result["primary_route_recent_progress_min_samples"] == 3
    assert result["primary_route_progress_gate_satisfied"] is False
    assert result["reuse_penalty_applied"] is True


def test_progress_gate_default_min_samples_preserves_two() -> None:
    """The default min-samples threshold must remain the historical value of 2."""
    config = _config()

    assert config.primary_route_progress_gate_min_samples == 2
    assert config.primary_route_progress_gate_use_monotone_accounting is False


def test_progress_gate_min_samples_below_two_fails_closed() -> None:
    """A min-samples threshold below 2 cannot form a finite difference and must raise."""
    with pytest.raises(ValueError, match="must be >= 2"):
        _config(primary_route_progress_gate_min_samples=1)


def test_progress_gate_min_samples_non_integer_fails_closed() -> None:
    """A non-integer min-samples value must fail closed instead of silently defaulting."""
    with pytest.raises(ValueError, match="must be an integer"):
        _config(primary_route_progress_gate_min_samples="not-a-number")


def test_progress_gate_threshold_non_finite_fails_closed() -> None:
    """A non-finite progress threshold must raise rather than silently default."""
    with pytest.raises(ValueError, match="must be finite"):
        _config(primary_route_progress_gate_threshold_m=float("inf"))


def test_progress_gate_threshold_nan_fails_closed() -> None:
    """A NaN progress threshold must raise rather than silently default."""
    with pytest.raises(ValueError, match="must be finite"):
        _config(primary_route_progress_gate_threshold_m=float("nan"))


def test_progress_accounting_mode_visible_in_decision_and_route_corridor() -> None:
    """The accounting mode and min-samples threshold should be observable diagnostics."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(
            primary_route_reuse_penalty_enabled=True,
            primary_route_reuse_penalty_weight=0.1,
            primary_route_reuse_penalty_cooldown_steps=5,
            primary_route_reuse_penalty_min_prior_primary_selections=1,
            near_parity_diversity_gate_enabled=True,
            near_parity_route_distance_slack_m=100.0,
            near_parity_route_distance_slack_ratio=100.0,
            near_parity_static_clearance_floor_m=100.0,
            near_parity_diversity_bonus=0.0,
            primary_route_progress_gate_enabled=True,
            primary_route_progress_gate_threshold_m=0.1,
            primary_route_progress_gate_use_monotone_accounting=True,
            primary_route_progress_gate_min_samples=2,
        )
    )
    obs = _obs(occupied_cells=_two_gap_wall())
    planner._hypotheses_for_observation(obs)

    decision = planner._hypotheses_for_observation(obs)
    assert decision["primary_route_progress_accounting_mode"] == "monotone"
    assert decision["primary_route_recent_progress_min_samples"] == 2

    route_corridor = planner._route_corridor_diagnostics(obs, current_time=1.0)
    assert route_corridor is not None
    reuse_penalty = route_corridor["topology_reuse_penalty"]
    assert reuse_penalty["primary_route_progress_accounting_mode"] == "monotone"
    assert reuse_penalty["primary_route_recent_progress_min_samples"] == 2


def test_topology_route_progress_distinguishes_churn_from_true_stall() -> None:
    """Candidate switches under low progress classify as churn, not true stall."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(min_route_progress_delta_m=0.1, stall_window_steps=2)
    )
    first = planner._update_topology_route_progress(
        selected_id="primary_route",
        route_remaining_m=10.0,
    )
    churn = planner._update_topology_route_progress(
        selected_id="masked_cell_5_12",
        route_remaining_m=9.99,
    )
    stagnant = planner._update_topology_route_progress(
        selected_id="masked_cell_5_12",
        route_remaining_m=9.98,
    )
    stall = planner._update_topology_route_progress(
        selected_id="masked_cell_5_12",
        route_remaining_m=9.97,
    )

    assert first["terminal_reason"] == "insufficient_samples"
    assert churn["terminal_reason"] == "near_parity_churn"
    assert churn["candidate_switch_count"] == 1
    assert churn["stagnant_steps"] == 0
    assert stagnant["terminal_reason"] == "route_stagnant"
    assert stall["terminal_reason"] == "true_stall"


def test_topology_guided_diagnostics_expose_threshold_and_progress_metadata() -> None:
    """Aggregate diagnostics carry explicit arbitration and near-parity thresholds."""
    planner = TopologyGuidedHybridRulePlannerAdapter(
        _config(
            arbitration_weight=0.35,
            near_parity_diversity_gate_enabled=True,
            near_parity_route_distance_slack_m=0.7,
            near_parity_route_distance_slack_ratio=0.05,
            near_parity_static_clearance_floor_m=0.04,
            near_parity_diversity_bonus=0.2,
        )
    )
    planner._update_topology_route_progress(
        selected_id="primary_route",
        route_remaining_m=10.0,
    )

    diagnostics = planner.diagnostics()["topology_guided"]
    thresholds = diagnostics["near_parity_thresholds"]

    assert diagnostics["arbitration_weight"] == pytest.approx(0.35)
    assert diagnostics["route_progress_state"]["last_selected_candidate"] == "primary_route"
    assert thresholds == {
        "schema_version": "topology_near_parity_thresholds.v1",
        "enabled": True,
        "route_distance_slack_m": pytest.approx(0.7),
        "route_distance_slack_ratio": pytest.approx(0.05),
        "static_clearance_floor_m": pytest.approx(0.04),
        "diversity_bonus": pytest.approx(0.2),
        "deterministic_tie_policy": "stable_first_max_score",
    }
