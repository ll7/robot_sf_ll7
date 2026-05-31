"""Tests for topology-hypothesis diagnostic helpers."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.grid_route import GridRoutePlannerAdapter, GridRoutePlannerConfig
from scripts.validation.run_topology_hypothesis_diagnostics import (
    _distinct_path,
    _extract_pedestrians,
    _find_alternative_paths,
    _first_float,
    _path_dynamic_clearance,
    _RouteHypothesisPath,
    _summarize_hypotheses,
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


def test_summarize_hypotheses_counts_sources_and_progress() -> None:
    """The summary should preserve selected-source counts and route-progress deltas."""
    summary = _summarize_hypotheses(
        [
            {
                "topology_status": "ok",
                "selected_local_command_source": "dynamic_window",
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
    )

    assert summary["topology_status_counts"] == {"ok": 2}
    assert summary["selected_source_counts"] == {
        "dynamic_window": 1,
        "path_follow_0.5m": 1,
    }
    assert summary["hypothesis_progress_by_rank"]["0"] == {
        "samples": 2,
        "first_corridor_name": "left_corridor_0",
        "last_corridor_name": "left_corridor_0",
        "first_remaining_distance_m": 10.0,
        "last_remaining_distance_m": 8.25,
        "progress_delta_m": 1.75,
        "min_static_clearance_m": 0.2,
        "min_dynamic_clearance_m": 1.0,
    }
