"""Tests for the route-side and homotopy observability contract (issue #7890)."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.route_choice_observability import (
    HomotopyObservation,
    classify_route_side,
    diagnostic_record,
    homotopy_identity,
    temporal_consistency,
)
from robot_sf.planner.grid_route import GridRoutePlannerAdapter

#: Symmetric corridor map: two feasible corridors around a static barrier
#: (rows 1-7, cols 6-8 blocked).  Row 0 and row 8 are the open corridors.
SYMMETRIC_BLOCKED = np.zeros((9, 15), dtype=bool)
SYMMETRIC_BLOCKED[1:8, 6:9] = True

#: Horizontal directed axis from start to goal (row 4); left is positive y.
START = (0.0, 4.0)
GOAL = (8.0, 4.0)


def _left_route() -> list[tuple[float, float]]:
    """Route on the standard counter-clockwise left side (positive y)."""
    return [(0.0, 8.0), (2.0, 8.0), (4.0, 8.0), (6.0, 8.0), (8.0, 8.0)]


def _right_route() -> list[tuple[float, float]]:
    """Route on the standard clockwise right side (negative y)."""
    return [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0), (8.0, 0.0)]


def _neutral_route() -> list[tuple[float, float]]:
    """Direct route along the axis (row 4)."""
    return [(0.0, 4.0), (2.0, 4.0), (4.0, 4.0), (6.0, 4.0), (8.0, 4.0)]


def _mixed_route() -> list[tuple[float, float]]:
    """Route that starts left (row 0) and ends right (row 8)."""
    return [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (4.0, 8.0), (6.0, 8.0), (8.0, 8.0)]


def _left_route_grid() -> list[tuple[float, float]]:
    """Left corridor in grid ``(row, col)`` order (positive-y row 8)."""
    return [(8.0, float(column)) for column in range(9)]


def _right_route_grid() -> list[tuple[float, float]]:
    """Right corridor in grid ``(row, col)`` order (negative-y row 0)."""
    return [(0.0, float(column)) for column in range(9)]


def _resampled(path: list[tuple[float, float]], factor: int = 3) -> list[tuple[float, float]]:
    """Deterministically resample a grid path by repeating each cell.

    Grid-cell paths are sequences of traversed cells; resampling repeats each
    cell (and its edge) rather than interpolating between non-adjacent cells,
    so the traversed cell set is unchanged.
    """
    out: list[tuple[float, float]] = []
    for a, b in pairwise(path):
        for step in range(factor):
            out.append(a)
        out.append(b)
    return out


def _translate(path: list[tuple[float, float]], dx: float, dy: float) -> list[tuple[float, float]]:
    return [(x + dx, y + dy) for x, y in path]


def _rotate(path: list[tuple[float, float]], angle_rad: float) -> list[tuple[float, float]]:
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return [(cosine * x - sine * y, sine * x + cosine * y) for x, y in path]


def _mirror_x(path: list[tuple[float, float]], axis_y: float) -> list[tuple[float, float]]:
    return [(x, 2 * axis_y - y) for x, y in path]


def test_left_route_classifies_left() -> None:
    report = classify_route_side(_left_route(), start=START, goal=GOAL)
    assert report.side == "left"
    assert report.reason is None


def test_right_route_classifies_right() -> None:
    report = classify_route_side(_right_route(), start=START, goal=GOAL)
    assert report.side == "right"
    assert report.reason is None


def test_neutral_route_classifies_neutral() -> None:
    report = classify_route_side(_neutral_route(), start=START, goal=GOAL)
    assert report.side == "neutral"


def test_mixed_route_classifies_mixed_not_last_sample() -> None:
    report = classify_route_side(_mixed_route(), start=START, goal=GOAL)
    assert report.side == "mixed"


def test_route_side_invariant_to_deterministic_resampling() -> None:
    for route in (_left_route(), _right_route(), _neutral_route(), _mixed_route()):
        original = classify_route_side(route, start=START, goal=GOAL)
        resampled = classify_route_side(_resampled(route), start=START, goal=GOAL)
        assert original.side == resampled.side


def test_route_side_invariant_to_nonuniform_segment_resampling() -> None:
    coarse = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
    densely_resampled_tail = coarse[:3] + [
        (2.0 + 2.0 * index / 100.0, 0.0) for index in range(1, 101)
    ]
    original = classify_route_side(coarse, start=(0.0, 0.0), goal=(4.0, 0.0))
    resampled = classify_route_side(
        densely_resampled_tail,
        start=(0.0, 0.0),
        goal=(4.0, 0.0),
    )
    assert original.side == "left"
    assert resampled.side == original.side


def test_route_side_invariant_to_rigid_translation() -> None:
    dx, dy = 3.0, -2.0
    shifted_start = (START[0] + dx, START[1] + dy)
    shifted_goal = (GOAL[0] + dx, GOAL[1] + dy)
    for route in (_left_route(), _right_route(), _mixed_route()):
        original = classify_route_side(route, start=START, goal=GOAL)
        shifted = classify_route_side(
            _translate(route, dx, dy), start=shifted_start, goal=shifted_goal
        )
        assert original.side == shifted.side


def test_route_side_invariant_to_rigid_rotation() -> None:
    for angle_rad in (np.pi / 2.0, 0.37):
        rotated_start = _rotate([START], angle_rad)[0]
        rotated_goal = _rotate([GOAL], angle_rad)[0]
        for route in (_left_route(), _right_route(), _neutral_route(), _mixed_route()):
            original = classify_route_side(route, start=START, goal=GOAL)
            rotated = classify_route_side(
                _rotate(route, angle_rad),
                start=rotated_start,
                goal=rotated_goal,
            )
            assert rotated.side == original.side


def test_mirroring_swaps_left_and_right() -> None:
    axis_y = 4.0
    left = classify_route_side(_left_route(), start=START, goal=GOAL)
    right_mirror = classify_route_side(_mirror_x(_left_route(), axis_y), start=START, goal=GOAL)
    assert left.side == "left"
    assert right_mirror.side == "right"


def test_reversing_reference_swaps_direction_documented() -> None:
    # Reversing start/goal flips the directed axis; a geometrically left route
    # relative to the original axis becomes right relative to the reversed one.
    reversed_left = classify_route_side(list(reversed(_left_route())), start=GOAL, goal=START)
    assert reversed_left.side == "right"


def test_empty_path_fails_closed_unavailable() -> None:
    report = classify_route_side([], start=START, goal=GOAL)
    assert report.side == "unavailable"
    assert report.reason == "empty_path"


def test_single_point_fails_closed_unavailable() -> None:
    report = classify_route_side([(4.0, 4.0)], start=START, goal=GOAL)
    assert report.side == "unavailable"
    assert report.reason == "single_point"


def test_zero_length_path_fails_closed_unavailable() -> None:
    report = classify_route_side([(4.0, 4.0), (4.0, 4.0)], start=START, goal=GOAL)
    assert report.side == "unavailable"
    assert report.reason == "zero_length"


def test_non_finite_path_fails_closed_unavailable() -> None:
    report = classify_route_side(
        [(0.0, 4.0), (float("nan"), 4.0), (8.0, 4.0)], start=START, goal=GOAL
    )
    assert report.side == "unavailable"
    assert report.reason == "non_finite"


@pytest.mark.parametrize("malformed_point", [None, (1.0,)])
def test_malformed_route_path_fails_closed_unavailable(malformed_point: object) -> None:
    report = classify_route_side(
        [(0.0, 4.0), malformed_point],  # type: ignore[list-item]
        start=START,
        goal=GOAL,
    )
    assert report.side == "unavailable"
    assert report.reason == "invalid_path"


def test_degenerate_reference_fails_closed_unavailable() -> None:
    report = classify_route_side(_neutral_route(), start=(5.0, 5.0), goal=(5.0, 5.0))
    assert report.side == "unavailable"
    assert report.reason == "degenerate_reference"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"tolerance_m": -1.0}, "invalid_tolerance"),
        ({"neutral_band_m": float("nan")}, "invalid_neutral_band"),
        ({"progress_interval": (0.9, 0.1)}, "invalid_progress_interval"),
    ],
)
def test_route_side_rejects_invalid_numeric_contract(
    kwargs: dict[str, object], reason: str
) -> None:
    report = classify_route_side(_neutral_route(), start=START, goal=GOAL, **kwargs)
    assert report.side == "unavailable"
    assert report.reason == reason


def test_route_side_tolerance_expands_the_neutral_boundary() -> None:
    """Numerical tolerance must affect the signed-distance decision it documents."""
    near_axis = [(0.0, 0.0), (1.0, 0.01), (2.0, 0.0)]

    exact = classify_route_side(
        near_axis,
        start=(0.0, 0.0),
        goal=(2.0, 0.0),
        tolerance_m=0.0,
        neutral_band_m=0.0,
        progress_interval=(0.0, 1.0),
    )
    tolerant = classify_route_side(
        near_axis,
        start=(0.0, 0.0),
        goal=(2.0, 0.0),
        tolerance_m=0.05,
        neutral_band_m=0.0,
        progress_interval=(0.0, 1.0),
    )

    assert exact.side == "left"
    assert tolerant.side == "neutral"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": (0.0, 0.0, 1.0), "goal": GOAL},
        {"start": START, "goal": (8.0, 4.0, 1.0)},
        {"start": START, "goal": GOAL, "coordinate_frame": ""},
        {"start": START, "goal": GOAL, "units": ""},
    ],
)
def test_route_side_rejects_ambiguous_reference_metadata(kwargs: dict[str, object]) -> None:
    """Reference points, frame, and units remain explicit instead of being truncated."""
    report = classify_route_side(_neutral_route(), **kwargs)  # type: ignore[arg-type]

    assert report.side == "unavailable"
    assert report.reason == "invalid_reference"


def test_homotopy_identity_is_stable_across_discovery_order() -> None:
    left = homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED)
    right = homotopy_identity(_right_route_grid(), SYMMETRIC_BLOCKED)
    assert left.identity is not None
    assert right.identity is not None
    assert left.identity != right.identity
    # Stable under resampling.
    assert homotopy_identity(_resampled(_left_route_grid()), SYMMETRIC_BLOCKED).identity == (
        left.identity
    )


def test_homotopy_identity_survives_ego_grid_motion() -> None:
    """The same world route must not switch topology when its ego-grid cells move."""
    adapter = GridRoutePlannerAdapter()
    blocked_t0 = np.zeros((11, 17), dtype=bool)
    blocked_t0[1:9, 7:10] = True
    path_t0 = [(9, col) for col in range(2, 15)]

    blocked_t1 = np.zeros_like(blocked_t0)
    blocked_t1[1:9, 6:9] = True
    path_t1 = [(9, col) for col in range(1, 14)]
    meta_t0 = {
        "origin": [0.0, 0.0],
        "resolution": [1.0],
        "use_ego_frame": [1.0],
        "robot_pose": [0.0, 0.0, 0.0],
    }
    meta_t1 = {
        "origin": [0.0, 0.0],
        "resolution": [1.0],
        "use_ego_frame": [1.0],
        "robot_pose": [1.0, 0.0, 0.0],
    }
    world_t0 = [tuple(adapter._grid_to_world(cell, meta_t0)) for cell in path_t0]
    world_t1 = [tuple(adapter._grid_to_world(cell, meta_t1)) for cell in path_t1]
    assert np.allclose(world_t0, world_t1)

    topology_t0 = homotopy_identity(
        path_t0,
        blocked_t0,
        identity_coordinates=world_t0,
        identity_coordinate_frame="global_xy",
        identity_units="m",
    )
    topology_t1 = homotopy_identity(
        path_t1,
        blocked_t1,
        identity_coordinates=world_t1,
        identity_coordinate_frame="global_xy",
        identity_units="m",
    )

    assert topology_t0.identity == topology_t1.identity
    assert topology_t0.identity_coordinate_frame == "global_xy"
    assert topology_t0.identity_units == "m"


def test_temporal_consistency_matches_fractional_ego_lattice_motion() -> None:
    """Sub-cell movement remains one topology under bounded point-set matching."""
    adapter = GridRoutePlannerAdapter()
    blocked = np.zeros((11, 17), dtype=bool)
    blocked[1:9, 7:10] = True
    path = [(9, col) for col in range(2, 15)]
    meta_t0 = {
        "origin": [0.0, 0.0],
        "resolution": [1.0],
        "use_ego_frame": [1.0],
        "robot_pose": [0.4, 0.0, 0.0],
    }
    meta_t1 = {**meta_t0, "robot_pose": [0.5, 0.0, 0.0]}
    world_t0 = [tuple(adapter._grid_to_world(cell, meta_t0)) for cell in path]
    world_t1 = [tuple(adapter._grid_to_world(cell, meta_t1)) for cell in path]

    topology_t0 = homotopy_identity(
        path,
        blocked,
        identity_coordinates=world_t0,
        identity_coordinate_frame="global_xy",
        identity_units="m",
        identity_match_tolerance=1.0,
    )
    topology_t1 = homotopy_identity(
        path,
        blocked,
        identity_coordinates=world_t1,
        identity_coordinate_frame="global_xy",
        identity_units="m",
        identity_match_tolerance=1.0,
    )

    side = classify_route_side(world_t0, start=(0.0, 0.0), goal=(20.0, 0.0))
    temporal = temporal_consistency([side, side], [topology_t0, topology_t1])
    assert topology_t0.identity != topology_t1.identity
    assert topology_t0.identity_match_tolerance == 1.0
    assert temporal.topology_transition_count == 0
    assert temporal.consistency_fraction == 1.0
    assert temporal.first_stable_step == 0


def test_temporal_consistency_keeps_full_cell_topologies_distinct() -> None:
    """Strict tolerance does not merge choke-point sets one complete cell apart."""
    path = _left_route_grid()
    first_points = [(row, col) for row, col in path]
    second_points = [(row + 1.0, col) for row, col in path]
    first = homotopy_identity(
        path,
        SYMMETRIC_BLOCKED,
        identity_coordinates=first_points,
        identity_coordinate_frame="global_xy",
        identity_units="m",
        identity_match_tolerance=1.0,
    )
    second = homotopy_identity(
        path,
        SYMMETRIC_BLOCKED,
        identity_coordinates=second_points,
        identity_coordinate_frame="global_xy",
        identity_units="m",
        identity_match_tolerance=1.0,
    )
    side = classify_route_side(_left_route(), start=START, goal=GOAL)

    report = temporal_consistency([side, side], [first, second])

    assert report.topology_transition_count == 1
    assert report.first_stable_step is None


def test_temporal_consistency_complete_link_rejects_tolerance_bridge() -> None:
    """A non-transitive tolerance bridge cannot be stable and transitional simultaneously."""
    observations = [
        HomotopyObservation(
            identity=str(value),
            unavailable_reason=None,
            identity_coordinate_frame="global_xy",
            identity_units="m",
            identity_points=((value, 0.0),),
            identity_match_tolerance=1.0,
        )
        for value in (0.6, 0.0, 1.2)
    ]
    side = classify_route_side(_left_route(), start=START, goal=GOAL)

    report = temporal_consistency([side, side, side], observations)

    assert report.topology_transition_count == 1
    assert report.consistency_fraction == pytest.approx(2 / 3)
    assert report.first_stable_step is None


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan")])
def test_homotopy_identity_rejects_invalid_match_tolerance(tolerance: float) -> None:
    observation = homotopy_identity(
        _left_route_grid(),
        SYMMETRIC_BLOCKED,
        identity_match_tolerance=tolerance,
    )
    assert observation.identity is None
    assert observation.unavailable_reason == "invalid_identity_match_tolerance"


def test_homotopy_identity_rejects_misaligned_identity_coordinates() -> None:
    """An immutable-frame identity path must align one-for-one with grid cells."""
    observation = homotopy_identity(
        _left_route_grid(),
        SYMMETRIC_BLOCKED,
        identity_coordinates=[(0.0, 0.0)],
        identity_coordinate_frame="global_xy",
        identity_units="m",
    )

    assert observation.identity is None
    assert observation.unavailable_reason == "invalid_identity_coordinates"


def test_homotopy_identity_does_not_depend_on_ephemeral_names() -> None:
    left_a = homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED)
    left_b = homotopy_identity(list(reversed(_left_route_grid())), SYMMETRIC_BLOCKED)
    # Choke cells are order-independent (set-based), so reversal keeps identity.
    assert left_a.identity == left_b.identity


def test_homotopy_identity_fails_closed() -> None:
    assert homotopy_identity([], SYMMETRIC_BLOCKED).unavailable_reason == "empty_path"
    assert homotopy_identity([(1.0, 1.0)], SYMMETRIC_BLOCKED).unavailable_reason == ("single_point")
    assert homotopy_identity(
        _left_route_grid(), np.zeros((0, 0), dtype=bool)
    ).unavailable_reason == ("missing_blocked_map")


def test_homotopy_identity_uses_canonical_clearance_fallback() -> None:
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[2, 2] = True
    path = [(1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.0, 3.0)]
    observation = homotopy_identity(path, blocked, clearance_threshold_cells=1)
    assert observation.identity == "1,2"
    assert observation.unavailable_reason is None


@pytest.mark.parametrize(
    ("blocked", "reason"),
    [
        (np.zeros((5,), dtype=bool), "malformed_blocked_map"),
        (np.array([[0.0, 2.0]], dtype=float), "invalid_blocked_map"),
    ],
)
def test_homotopy_identity_rejects_malformed_blocked_map(blocked: np.ndarray, reason: str) -> None:
    assert homotopy_identity(_left_route_grid(), blocked).unavailable_reason == reason


def test_homotopy_identity_rejects_invalid_threshold() -> None:
    observation = homotopy_identity(
        _left_route_grid(), SYMMETRIC_BLOCKED, clearance_threshold_cells=0
    )
    assert observation.unavailable_reason == "invalid_clearance_threshold"


def test_homotopy_identity_rejects_non_integral_grid_cells() -> None:
    observation = homotopy_identity([(0.0, 0.0), (0.5, 1.0)], SYMMETRIC_BLOCKED)
    assert observation.unavailable_reason == "non_integral_grid_cell"


def test_homotopy_identity_rejects_non_adjacent_grid_steps() -> None:
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[1, 2] = True
    observation = homotopy_identity([(1.0, 1.0), (1.0, 3.0)], blocked)
    assert observation.identity is None
    assert observation.unavailable_reason == "non_adjacent_grid_step"


@pytest.mark.parametrize("step", [(0, 0), (1, 0), (0, 1), (1, 1), (-1, -1)])
def test_homotopy_identity_accepts_duplicate_and_8_connected_steps(
    step: tuple[int, int],
) -> None:
    blocked = np.zeros((5, 5), dtype=bool)
    path = [(1.0, 1.0), (1.0 + step[0], 1.0 + step[1])]
    observation = homotopy_identity(path, blocked)
    assert observation.unavailable_reason in {None, "no_choke_cells"}


@pytest.mark.parametrize("malformed_point", [None, (1.0,)])
def test_homotopy_identity_rejects_malformed_path_points(malformed_point: object) -> None:
    observation = homotopy_identity(
        [(0.0, 0.0), malformed_point],  # type: ignore[list-item]
        SYMMETRIC_BLOCKED,
    )
    assert observation.unavailable_reason == "invalid_path"


def test_homotopy_identity_rejects_paths_through_blocked_cells() -> None:
    observation = homotopy_identity([(4.0, 5.0), (4.0, 6.0)], SYMMETRIC_BLOCKED)
    assert observation.unavailable_reason == "path_intersects_blocked"


def test_temporal_consistency_separates_valid_and_unavailable() -> None:
    sides = [
        classify_route_side(_left_route(), start=START, goal=GOAL),
        classify_route_side(_right_route(), start=START, goal=GOAL),
        classify_route_side([], start=START, goal=GOAL),
    ]
    homotopies = [
        homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED),
        homotopy_identity(_right_route_grid(), SYMMETRIC_BLOCKED),
        _unavailable_homotopy(),
    ]
    report = temporal_consistency(sides, homotopies)
    assert report.valid_count == 2
    assert report.unavailable_count == 1
    assert report.denominator == 2
    assert report.side_transition_count == 1
    assert report.topology_transition_count == 1
    assert report.consistency_fraction == pytest.approx(1 / 2)
    assert report.availability_fraction == pytest.approx(2 / 3)
    assert report.availability_denominator == 3
    assert report.side_valid_count == 2
    assert report.topology_valid_count == 2
    assert report.aligned_count == 3
    assert report.alignment_valid is True


def test_temporal_consistency_does_not_bridge_unavailable_samples() -> None:
    sides = [
        classify_route_side(_left_route(), start=START, goal=GOAL),
        classify_route_side([], start=START, goal=GOAL),
        classify_route_side(_right_route(), start=START, goal=GOAL),
    ]
    homotopies = [
        homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED),
        _unavailable_homotopy(),
        homotopy_identity(_right_route_grid(), SYMMETRIC_BLOCKED),
    ]
    report = temporal_consistency(sides, homotopies)
    assert report.valid_count == 2
    assert report.side_transition_count == 0
    assert report.topology_transition_count == 0
    assert report.consistency_fraction == pytest.approx(1 / 2)
    assert report.availability_fraction == pytest.approx(2 / 3)


def test_temporal_consistency_rejects_unaligned_observations() -> None:
    report = temporal_consistency(
        [classify_route_side(_left_route(), start=START, goal=GOAL)],
        [],
    )
    assert report.alignment_valid is False
    assert report.alignment_reason == "length_mismatch"
    assert report.aligned_count == 0
    assert report.denominator == 0
    assert report.valid_count == 0
    assert report.consistency_fraction == 0.0


def test_temporal_consistency_rejects_route_reference_changes() -> None:
    """Side labels from different declared references are not comparable over time."""
    first = classify_route_side(_left_route(), start=START, goal=GOAL)
    second = classify_route_side(
        _left_route(),
        start=START,
        goal=GOAL,
        coordinate_frame="shifted_global_xy",
    )
    topology = homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED)

    report = temporal_consistency([first, second], [topology, topology])

    assert report.alignment_valid is False
    assert report.alignment_reason == "route_reference_mismatch"
    assert report.valid_count == 0


def test_temporal_consistency_rejects_homotopy_frame_changes() -> None:
    """Equal coordinate strings from different topology frames are not the same identity."""
    side = classify_route_side(_left_route(), start=START, goal=GOAL)
    grid_topology = homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED)
    world_topology = homotopy_identity(
        _left_route_grid(),
        SYMMETRIC_BLOCKED,
        identity_coordinates=_left_route_grid(),
        identity_coordinate_frame="global_xy",
        identity_units="m",
    )
    assert grid_topology.identity == world_topology.identity

    report = temporal_consistency([side, side], [grid_topology, world_topology])

    assert report.alignment_valid is False
    assert report.alignment_reason == "identity_reference_mismatch"
    assert report.valid_count == 0


def test_temporal_consistency_rejects_homotopy_match_tolerance_changes() -> None:
    side = classify_route_side(_left_route(), start=START, goal=GOAL)
    first = homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED, identity_match_tolerance=1.0)
    second = homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED, identity_match_tolerance=0.5)

    report = temporal_consistency([side, side], [first, second])

    assert report.alignment_valid is False
    assert report.alignment_reason == "identity_reference_mismatch"


def test_temporal_consistency_consistent_sequence() -> None:
    sides = [
        classify_route_side(_left_route(), start=START, goal=GOAL),
        classify_route_side(_left_route(), start=START, goal=GOAL),
    ]
    homotopies = [
        homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED),
        homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED),
    ]
    report = temporal_consistency(sides, homotopies)
    assert report.side_transition_count == 0
    assert report.topology_transition_count == 0
    assert report.dominant_side == "left"
    assert report.first_stable_step == 0
    assert report.consistency_fraction == 1.0


def test_temporal_consistency_detects_stability_after_an_initial_switch() -> None:
    sides = [
        classify_route_side(_left_route(), start=START, goal=GOAL),
        classify_route_side(_right_route(), start=START, goal=GOAL),
        classify_route_side(_right_route(), start=START, goal=GOAL),
    ]
    homotopies = [
        homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED),
        homotopy_identity(_right_route_grid(), SYMMETRIC_BLOCKED),
        homotopy_identity(_right_route_grid(), SYMMETRIC_BLOCKED),
    ]
    report = temporal_consistency(sides, homotopies)
    assert report.first_stable_step == 1
    assert report.consistency_fraction == pytest.approx(2 / 3)


def test_temporal_consistency_does_not_bridge_unavailable_stability() -> None:
    sides = [
        classify_route_side(_left_route(), start=START, goal=GOAL),
        classify_route_side([], start=START, goal=GOAL),
        classify_route_side(_left_route(), start=START, goal=GOAL),
    ]
    homotopies = [
        homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED),
        _unavailable_homotopy(),
        homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED),
    ]
    assert temporal_consistency(sides, homotopies).first_stable_step is None


def test_temporal_consistency_reports_original_index_after_unavailable_prefix() -> None:
    sides = [
        classify_route_side([], start=START, goal=GOAL),
        classify_route_side(_left_route(), start=START, goal=GOAL),
        classify_route_side(_left_route(), start=START, goal=GOAL),
    ]
    homotopies = [
        _unavailable_homotopy(),
        homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED),
        homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED),
    ]
    report = temporal_consistency(sides, homotopies)
    assert report.first_stable_step == 1
    assert report.availability_fraction == pytest.approx(2 / 3)
    assert report.consistency_fraction == 1.0


def test_temporal_consistency_detects_topology_only_transition() -> None:
    left = classify_route_side(_left_route(), start=START, goal=GOAL)
    left_topology = homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED)
    right_topology = homotopy_identity(_right_route_grid(), SYMMETRIC_BLOCKED)
    report = temporal_consistency(
        [left, left, left],
        [left_topology, right_topology, right_topology],
    )
    assert report.side_transition_count == 0
    assert report.topology_transition_count == 1
    assert report.first_stable_step == 1
    assert report.consistency_fraction == pytest.approx(2 / 3)


def test_diagnostic_record_is_versioned_json_ready_and_claim_bounded() -> None:
    record = diagnostic_record(
        [classify_route_side(_left_route(), start=START, goal=GOAL)],
        [homotopy_identity(_left_route_grid(), SYMMETRIC_BLOCKED)],
    )
    assert record["schema_version"] == "route_choice_observability.v1"
    assert record["evidence_tier"] == "analysis-only"
    assert record["result_classification"] == "diagnostic-only"
    assert record["status"] == "available"
    assert "not pedestrian preference" in record["claim_boundary"]
    assert json.loads(json.dumps(record)) == record


def test_diagnostic_record_fails_closed_on_alignment_mismatch() -> None:
    record = diagnostic_record(
        [classify_route_side(_left_route(), start=START, goal=GOAL)],
        [],
    )
    assert record["status"] == "not_available"
    assert record["temporal_consistency"]["alignment_valid"] is False


def test_tracked_diagnostic_receipt_matches_production_record() -> None:
    side_paths = [_left_route(), _right_route(), []]
    grid_paths = [_left_route_grid(), _right_route_grid(), []]
    expected = diagnostic_record(
        [classify_route_side(path, start=START, goal=GOAL) for path in side_paths],
        [homotopy_identity(path, SYMMETRIC_BLOCKED) for path in grid_paths],
    )
    expected["fixture"] = {
        "name": "symmetric_two_corridor_with_unavailable_step",
        "blocked_map_shape": [9, 15],
        "blocked_slice": {"rows": [1, 7], "cols": [6, 8]},
        "source": (
            "deterministic synthetic fixture in tests/benchmark/test_route_choice_observability.py"
        ),
    }
    receipt_path = (
        Path(__file__).parents[2]
        / "docs/context/evidence/issue_7890_route_choice_observability/receipt.json"
    )
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == expected


def _unavailable_homotopy():
    """Return an unavailable homotopy observation matching the dataclass."""
    from robot_sf.benchmark.route_choice_observability import HomotopyObservation

    return HomotopyObservation(identity=None, unavailable_reason="empty_path")
