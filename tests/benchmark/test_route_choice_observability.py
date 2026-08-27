"""Tests for the route-side and homotopy observability contract (issue #7890)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from robot_sf.benchmark.route_choice_observability import (
    classify_route_side,
    homotopy_identity,
    temporal_consistency,
)

#: Symmetric corridor map: two feasible corridors around a static barrier
#: (rows 1-7, cols 6-8 blocked).  Row 0 and row 8 are the open corridors.
SYMMETRIC_BLOCKED = np.zeros((9, 15), dtype=bool)
SYMMETRIC_BLOCKED[1:8, 6:9] = True

#: Horizontal directed axis from start to goal (row 4); left is negative y.
START = (0.0, 4.0)
GOAL = (8.0, 4.0)


def _left_route() -> list[tuple[float, float]]:
    """Route above the axis (negative y, grid row 0 corridor)."""
    return [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0), (8.0, 0.0)]


def _right_route() -> list[tuple[float, float]]:
    """Route below the axis (positive y, grid row 8 corridor)."""
    return [(0.0, 8.0), (2.0, 8.0), (4.0, 8.0), (6.0, 8.0), (8.0, 8.0)]


def _neutral_route() -> list[tuple[float, float]]:
    """Direct route along the axis (row 4)."""
    return [(0.0, 4.0), (2.0, 4.0), (4.0, 4.0), (6.0, 4.0), (8.0, 4.0)]


def _mixed_route() -> list[tuple[float, float]]:
    """Route that starts left (row 0) and ends right (row 8)."""
    return [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (4.0, 8.0), (6.0, 8.0), (8.0, 8.0)]


def _left_route_grid() -> list[tuple[float, float]]:
    """Left corridor in grid ``(row, col)`` order (row 0, cols 0-8)."""
    return [(0.0, 0.0), (0.0, 2.0), (0.0, 4.0), (0.0, 6.0), (0.0, 8.0)]


def _right_route_grid() -> list[tuple[float, float]]:
    """Right corridor in grid ``(row, col)`` order (row 8, cols 0-8)."""
    return [(8.0, 0.0), (8.0, 2.0), (8.0, 4.0), (8.0, 6.0), (8.0, 8.0)]


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
    assert report.denominator == 3
    assert report.side_transition_count == 1
    assert report.topology_transition_count == 1
    assert report.consistency_fraction == pytest.approx(2 / 3)
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
    assert report.consistency_fraction == pytest.approx(2 / 3)


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


def _unavailable_homotopy():
    """Return an unavailable homotopy observation matching the dataclass."""
    from robot_sf.benchmark.route_choice_observability import HomotopyObservation

    return HomotopyObservation(identity=None, unavailable_reason="empty_path")
