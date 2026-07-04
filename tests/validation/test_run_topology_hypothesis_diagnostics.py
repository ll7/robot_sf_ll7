"""Tests for topology-hypothesis diagnostic helpers."""

from __future__ import annotations

from collections import Counter

import numpy as np

from robot_sf.planner.grid_route import GridRoutePlannerAdapter, GridRoutePlannerConfig
from scripts.validation.run_topology_hypothesis_diagnostics import (
    _corrective_behavior_summary,
    _distinct_path,
    _extract_pedestrians,
    _find_alternative_paths,
    _first_float,
    _path_dynamic_clearance,
    _RouteHypothesisPath,
    _summarize_hypotheses,
    _terminal_outcome,
    _topology_signature,
)


def _adapter() -> GridRoutePlannerAdapter:
    """Return a deterministic grid-route adapter for synthetic fixtures."""
    return GridRoutePlannerAdapter(
        GridRoutePlannerConfig(
            obstacle_inflation_cells=0,
            clearance_penalty_weight=0.0,
        )
    )


def test_first_float_handles_none_and_non_finite_values() -> None:
    """_first_float should fall back to defaults for None/non-finite inputs."""
    assert _first_float(None, 0.35) == 0.35
    assert _first_float(float("nan"), 0.5) == 0.5
    assert _first_float(float("inf"), 1.0) == 1.0
    assert _first_float([2.5], 0.0) == 2.5
    assert _first_float(3.0, 0.0) == 3.0


def test_extract_pedestrians_handles_none_count_metadata() -> None:
    """_extract_pedestrians should not crash when count metadata is None."""
    obs = {
        "pedestrians": {
            "positions": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "count": None,
            "radius": 0.3,
        }
    }

    class _FailingAdapter:
        def _socnav_fields(self, obs):
            raise AttributeError

    positions, radius = _extract_pedestrians(_FailingAdapter(), obs)
    assert positions.shape == (2, 2)
    assert radius == 0.3


def test_path_dynamic_clearance_works_without_inner_asarray() -> None:
    """_path_dynamic_clearance should accept plain ndarray pedestrians without redundant conversion."""
    path = [np.array([0.0, 0.0]), np.array([10.0, 0.0])]
    pedestrians = np.array([[5.0, 2.0]])
    assert _path_dynamic_clearance(path, pedestrians, ped_radius=0.5) == 1.5


def test_distinct_path_uses_frozenset_signatures_without_list_conversion() -> None:
    """_distinct_path should accept frozenset topology_signatures without converting to lists."""
    route = _RouteHypothesisPath(
        hypothesis_id="primary",
        path=[(0, 0), (0, 1)],
        clearance_map=np.zeros((1, 1), dtype=bool),
        topology_signature=frozenset({(0, 1)}),
    )
    assert (
        _distinct_path(
            path=[(0, 0), (0, 1), (0, 2)],
            topology_signature=frozenset({(0, 1)}),
            accepted=[route],
        )
        is False
    )


def test_find_alternative_paths_recovers_two_wall_gaps() -> None:
    """Masking the primary path should expose a second distinct corridor."""
    blocked = np.zeros((24, 24), dtype=bool)
    blocked[:, 12] = True
    blocked[5, 12] = False
    blocked[18, 12] = False

    paths = _find_alternative_paths(
        _adapter(),
        blocked,
        start=(12, 2),
        goal=(12, 21),
        max_hypotheses=3,
        block_radius_cells=3,
        block_stride_cells=3,
    )

    assert len(paths) >= 2
    assert all(
        route.hypothesis_id == "primary_route" or route.hypothesis_id.startswith("masked_cell_")
        for route in paths
    )
    gap_rows = {
        next(cell[0] for cell in route.path if cell[1] == 12)
        for route in paths
        if any(cell[1] == 12 for cell in route.path)
    }
    assert {5, 18}.issubset(gap_rows)


def test_find_alternative_paths_fails_closed_with_single_gap() -> None:
    """A one-corridor grid should not invent an extra topology hypothesis."""
    blocked = np.zeros((24, 24), dtype=bool)
    blocked[:, 12] = True
    blocked[12, 12] = False

    paths = _find_alternative_paths(
        _adapter(),
        blocked,
        start=(12, 2),
        goal=(12, 21),
        max_hypotheses=3,
        block_radius_cells=3,
        block_stride_cells=3,
    )

    assert [route.hypothesis_id for route in paths] == ["primary_route"]


def test_topology_signature_prefers_choke_cells_over_same_gap_wiggles() -> None:
    """Same-gap detours should share the same compact bottleneck signature."""
    blocked = np.zeros((16, 16), dtype=bool)
    blocked[:, 8] = True
    blocked[7, 8] = False
    clearance = _adapter()._compute_clearance_map(blocked)

    direct = [(7, col) for col in range(4, 12)]
    wiggle = [(7, 4), (6, 5), (6, 6), (7, 7), (7, 8), (7, 9), (6, 10), (7, 11)]

    assert _topology_signature(
        direct,
        blocked,
        clearance,
        clearance_threshold_cells=3,
    ) == frozenset({(7, 8)})
    assert _topology_signature(
        wiggle,
        blocked,
        clearance,
        clearance_threshold_cells=3,
    ) == frozenset({(7, 8)})


def test_path_dynamic_clearance_reports_pedestrian_clearance_to_polyline() -> None:
    """Dynamic clearance should subtract pedestrian radius from nearest route segment."""
    path = [np.array([0.0, 0.0]), np.array([4.0, 0.0]), np.array([4.0, 3.0])]
    pedestrians = np.array(
        [
            [2.0, 1.0],
            [5.0, 3.0],
        ]
    )

    assert _path_dynamic_clearance(path, pedestrians, ped_radius=0.3) == 0.7


def test_terminal_outcome_classifies_success_and_collisions() -> None:
    """Terminal summaries should preserve success and collision outcomes."""
    assert _terminal_outcome(
        {"step": 17, "success": True, "terminated": True, "meta": {}},
        [],
    ) == {
        "outcome": "success",
        "step": 17,
        "terminated": True,
        "truncated": False,
        "success": True,
        "is_pedestrian_collision": False,
        "is_obstacle_collision": False,
        "is_robot_collision": False,
    }

    assert (
        _terminal_outcome(
            {
                "step": 5,
                "success": False,
                "terminated": True,
                "meta": {"is_obstacle_collision": True},
            },
            [],
        )["outcome"]
        == "obstacle_collision"
    )


def test_terminal_outcome_classifies_horizon_exhaustion_from_last_step() -> None:
    """Missing done_info should fall back to the last recorded step."""
    assert _terminal_outcome(
        {},
        [
            {
                "step": 159,
                "is_success": False,
                "terminated": False,
                "truncated": False,
                "meta": {},
            }
        ],
    ) == {
        "outcome": "horizon_exhausted",
        "step": 159,
        "terminated": False,
        "truncated": False,
        "success": False,
        "is_pedestrian_collision": False,
        "is_obstacle_collision": False,
        "is_robot_collision": False,
    }


def test_corrective_behavior_summary_classifies_continue() -> None:
    """A successful non-primary influenced slice should classify as continue."""
    summary = _corrective_behavior_summary(
        [],
        selected_source_counts=Counter({"topology_hypothesis": 4, "dynamic_window": 1}),
        influence_counts=Counter({"primary_route": 1, "masked_cell_5_12": 3}),
        progress_by_rank={"0": {"progress_delta_m": 2.0}},
        hypothesis_switch_count=1,
        terminal_outcome={"success": True, "outcome": "success"},
    )

    assert summary["decision"] == "continue"
    assert summary["non_primary_topology_command_steps"] == 3
    assert summary["positive_route_progress"] is True
    assert summary["claim_boundary"] == "diagnostic_only_not_benchmark_success"


def test_corrective_behavior_summary_revises_primary_only_influence() -> None:
    """Primary-only influence is useful but not non-primary corrective evidence."""
    summary = _corrective_behavior_summary(
        [],
        selected_source_counts=Counter({"topology_hypothesis": 2}),
        influence_counts=Counter({"primary_route": 2}),
        progress_by_rank={"0": {"progress_delta_m": 1.0}},
        hypothesis_switch_count=0,
        terminal_outcome={"success": True, "outcome": "success"},
    )

    assert summary["decision"] == "revise"
    assert summary["non_primary_topology_command_steps"] == 0


def test_corrective_behavior_summary_revises_without_terminal_success() -> None:
    """Command influence without terminal success should classify as revise."""
    summary = _corrective_behavior_summary(
        [],
        selected_source_counts=Counter({"topology_hypothesis": 2}),
        influence_counts=Counter({"masked_cell_84_103": 2}),
        progress_by_rank={"0": {"progress_delta_m": 0.2}},
        hypothesis_switch_count=0,
        terminal_outcome={"success": False, "outcome": "horizon_exhausted"},
    )

    assert summary["decision"] == "revise"
    assert summary["non_primary_topology_command_steps"] == 2
    assert summary["positive_route_progress"] is True


def test_corrective_behavior_summary_stops_without_command_influence() -> None:
    """Selection-only diversity should not count as corrective behavior."""
    summary = _corrective_behavior_summary(
        [],
        selected_source_counts=Counter({"dynamic_window": 3}),
        influence_counts=Counter(),
        progress_by_rank={"0": {"progress_delta_m": 1.0}},
        hypothesis_switch_count=0,
        terminal_outcome={"success": True, "outcome": "success"},
    )

    assert summary["decision"] == "stop"
    assert summary["topology_command_steps"] == 0


def test_summarize_hypotheses_counts_sources_progress_and_corrective_behavior() -> None:
    """The summary should preserve selected-source counts and route-progress deltas."""
    summary = _summarize_hypotheses(
        [
            {
                "topology_status": "ok",
                "selected_local_command_source": "dynamic_window",
                "topology_command_influence": {"selected_hypothesis_id": "primary_route"},
                "planner_route_corridor": {
                    "topology_reuse_penalty": {
                        "reuse_penalty_applied": True,
                        "reuse_penalty_reason": "primary_route_selected_2_times_in_last_3_steps",
                        "recent_primary_selection_count": 2,
                        "eligible_near_parity_alternative_exists": True,
                        "primary_route_recent_progress_m": 0.2,
                        "primary_route_recent_progress_sample_count": 2,
                        "primary_route_progress_gate_satisfied": True,
                        "reuse_penalty_suppressed_by_progress": False,
                    },
                    "topology_hypotheses": [
                        {
                            "hypothesis_id": "primary_route",
                            "near_parity_gate_reason": "selected_primary_route",
                            "score": -8.0,
                            "score_rank": 0,
                            "score_margin_to_selected": 0.0,
                            "score_components": {
                                "length_penalty": -10.0,
                                "static_clearance_bonus": 2.0,
                            },
                            "selection_outcome": "selected",
                            "rejection_reason": None,
                        },
                        {
                            "hypothesis_id": "masked_cell_5_12",
                            "near_parity_gate_reason": "eligible_near_parity_alternative",
                            "score": -9.5,
                            "score_rank": 1,
                            "score_margin_to_selected": 1.5,
                            "score_components": {
                                "length_penalty": -11.0,
                                "static_clearance_bonus": 1.5,
                            },
                            "selection_outcome": "rejected",
                            "rejection_reason": "lower_topology_selection_score",
                        },
                    ],
                },
                "topology_hypotheses": [
                    {
                        "rank": 0,
                        "corridor_name": "left_corridor_0",
                        "route_remaining_distance_m": 10.0,
                        "static_clearance_min_m": 0.2,
                        "dynamic_clearance_min_m": 1.5,
                    }
                ],
            },
            {
                "topology_status": "ok",
                "selected_local_command_source": "path_follow_0.5m",
                "topology_command_influence": {"selected_hypothesis_id": "masked_cell_5_12"},
                "planner_route_corridor": {
                    "topology_hypothesis": {
                        "hypothesis_id": "masked_cell_5_12",
                        "near_parity_gate_reason": "eligible_near_parity_alternative",
                    }
                },
                "topology_hypotheses": [
                    {
                        "rank": 0,
                        "corridor_name": "left_corridor_0",
                        "route_remaining_distance_m": 8.25,
                        "static_clearance_min_m": 0.3,
                        "dynamic_clearance_min_m": 1.0,
                    }
                ],
            },
        ]
        + [
            {
                "topology_status": "ok",
                "selected_local_command_source": "topology_hypothesis",
                "topology_command_influence": {"selected_hypothesis_id": "masked_cell_5_12"},
                "planner_route_corridor": {
                    "topology_hypothesis": {
                        "hypothesis_id": "masked_cell_7_14",
                        "near_parity_gate_reason": None,
                    }
                },
                "topology_hypotheses": [
                    {
                        "rank": 0,
                        "corridor_name": "left_corridor_0",
                        "route_remaining_distance_m": 7.0,
                        "static_clearance_min_m": 0.25,
                        "dynamic_clearance_min_m": 1.2,
                    }
                ],
            }
        ]
        + [
            {
                "topology_status": "ok",
                "selected_local_command_source": "dynamic_window",
                "planner_route_corridor": {"topology_hypothesis": {"hypothesis_id": ""}},
                "topology_hypotheses": [
                    {
                        "rank": 0,
                        "corridor_name": "left_corridor_0",
                        "route_remaining_distance_m": 6.0,
                        "static_clearance_min_m": 0.25,
                        "dynamic_clearance_min_m": 1.2,
                    }
                ],
            }
        ],
        {"step": 2, "terminated": True, "success": True, "meta": {}},
    )

    assert summary["topology_status_counts"] == {"ok": 4}
    assert summary["selected_source_counts"] == {
        "dynamic_window": 2,
        "path_follow_0.5m": 1,
        "topology_hypothesis": 1,
    }
    assert summary["route_selector_selected_hypothesis_counts"] == {
        "": 1,
        "masked_cell_5_12": 1,
        "masked_cell_7_14": 1,
        "primary_route": 1,
    }
    assert summary["selected_row_near_parity_gate_reasons"] == {
        "eligible_near_parity_alternative": 1,
        "selected_primary_route": 1,
    }
    assert summary["hypothesis_progress_by_rank"]["0"] == {
        "samples": 4,
        "first_corridor_name": "left_corridor_0",
        "last_corridor_name": "left_corridor_0",
        "first_remaining_distance_m": 10.0,
        "last_remaining_distance_m": 6.0,
        "progress_delta_m": 4.0,
        "min_static_clearance_m": 0.2,
        "min_dynamic_clearance_m": 1.0,
    }
    assert summary["topology_command_influence_counts"] == {
        "masked_cell_5_12": 2,
        "primary_route": 1,
    }
    assert summary["hypothesis_switch_count"] == 1
    assert summary["corrective_behavior"]["decision"] == "continue"
    assert summary["corrective_behavior"]["terminal_outcome"]["outcome"] == "success"
    assert summary["topology_selection_score_examples"] == [
        {
            "step": -1,
            "hypotheses": [
                {
                    "hypothesis_id": "primary_route",
                    "score": -8.0,
                    "score_rank": 0,
                    "score_margin_to_selected": 0.0,
                    "score_components": {
                        "length_penalty": -10.0,
                        "static_clearance_bonus": 2.0,
                    },
                    "selection_outcome": "selected",
                    "rejection_reason": None,
                },
                {
                    "hypothesis_id": "masked_cell_5_12",
                    "score": -9.5,
                    "score_rank": 1,
                    "score_margin_to_selected": 1.5,
                    "score_components": {
                        "length_penalty": -11.0,
                        "static_clearance_bonus": 1.5,
                    },
                    "selection_outcome": "rejected",
                    "rejection_reason": "lower_topology_selection_score",
                },
            ],
        }
    ]
    assert summary["topology_reuse_penalty"] == {
        "applied_steps": 1,
        "eligible_near_parity_alternative_steps": 1,
        "max_recent_primary_selection_count": 2,
        "progress_gate_satisfied_steps": 1,
        "progress_suppressed_steps": 0,
        "max_primary_route_recent_progress_m": 0.2,
        "max_primary_route_recent_progress_sample_count": 2,
        "reason_counts": {"primary_route_selected_2_times_in_last_3_steps": 1},
    }


def test_summarize_hypotheses_reports_route_progress_terminal_reasons() -> None:
    """Diagnostic summary distinguishes route progress, churn, and true stalls."""
    steps = [
        {
            "step": 0,
            "topology_status": "ok",
            "selected_local_command_source": "topology_hypothesis",
            "planner_route_corridor": {
                "topology_hypothesis": {"hypothesis_id": "primary_route"},
                "topology_route_progress": {
                    "terminal_reason": "goal_progress",
                    "selected_hypothesis_id": "primary_route",
                    "route_progress_delta_m": 0.2,
                    "stagnant_steps": 0,
                    "candidate_switch_count": 0,
                },
            },
            "topology_hypotheses": [],
            "topology_command_influence": {"selected_hypothesis_id": "primary_route"},
        },
        {
            "step": 1,
            "topology_status": "ok",
            "selected_local_command_source": "topology_hypothesis",
            "planner_route_corridor": {
                "topology_hypothesis": {"hypothesis_id": "masked_cell_5_12"},
                "topology_route_progress": {
                    "terminal_reason": "near_parity_churn",
                    "selected_hypothesis_id": "masked_cell_5_12",
                    "previous_selected_hypothesis_id": "primary_route",
                    "route_progress_delta_m": 0.01,
                    "stagnant_steps": 0,
                    "candidate_switch_count": 1,
                },
            },
            "topology_hypotheses": [],
            "topology_command_influence": {"selected_hypothesis_id": "masked_cell_5_12"},
        },
        {
            "step": 2,
            "topology_status": "ok",
            "selected_local_command_source": "topology_hypothesis",
            "planner_route_corridor": {
                "topology_hypothesis": {"hypothesis_id": "masked_cell_5_12"},
                "topology_route_progress": {
                    "terminal_reason": "true_stall",
                    "selected_hypothesis_id": "masked_cell_5_12",
                    "route_progress_delta_m": 0.0,
                    "stagnant_steps": 2,
                    "candidate_switch_count": 1,
                },
            },
            "topology_hypotheses": [],
            "topology_command_influence": {"selected_hypothesis_id": "masked_cell_5_12"},
        },
    ]

    summary = _summarize_hypotheses(steps, {"step": 2, "truncated": True})

    assert summary["topology_route_progress"]["terminal_reason_counts"] == {
        "goal_progress": 1,
        "near_parity_churn": 1,
        "true_stall": 1,
    }
    assert summary["topology_route_progress"]["near_parity_churn_steps"] == 1
    assert summary["topology_route_progress"]["true_stall_steps"] == 1
    assert summary["topology_route_progress"]["max_candidate_switch_count"] == 1
    assert summary["topology_route_progress"]["max_stagnant_steps"] == 2
    assert summary["corrective_behavior"]["route_progress_terminal_reason_counts"] == {
        "goal_progress": 1,
        "near_parity_churn": 1,
        "true_stall": 1,
    }
