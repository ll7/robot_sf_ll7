"""Tests for the diagnostic topology-guided local policy adapter."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.topology_guided_local_policy import (
    TopologyGuidedHybridRulePlannerAdapter,
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
