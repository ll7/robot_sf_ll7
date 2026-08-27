"""Deterministic route-side and homotopy observability contract (issue #7890).

This module owns pure, typed helpers that describe which side and which
route/corridor hypothesis a planned path uses, relative to a declared
directed reference axis from scenario start to goal.  It separates:

1. route-side classification;
2. topological/homotopy identity;
3. temporal consistency across replans;
4. unavailable or ambiguous observations.

The contract measures *planner-route observability* only.  It does not claim
pedestrian preference, understanding, response, comfort, or general human
predictability, and it never merges outputs into a single social-compliance
score.

The homotopy identity reuses the compact corridor-signature idea from
``scripts/validation/run_topology_hypothesis_diagnostics.py``
(``_topology_signature``): a stable identity derives from route geometry and
blocked-cell topology, not from discovery order or ephemeral names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.planner.grid_route import GridRoutePlannerAdapter

#: Bounded route-side vocabulary.
ROUTE_SIDES = frozenset({"left", "right", "neutral", "mixed", "unavailable"})

#: Fail-closed reasons for an unavailable route-side classification.
UNAVAILABLE_REASONS = frozenset(
    {
        "empty_path",
        "single_point",
        "zero_length",
        "non_finite",
        "degenerate_reference",
        "insufficient_progress",
        "invalid_tolerance",
        "invalid_neutral_band",
        "invalid_progress_interval",
        "invalid_reference",
        "unknown",
    }
)

#: Default numerical tolerance for the signed cross-product side test.
DEFAULT_SIDE_TOLERANCE_M = 0.05
#: Default neutral-band half-width around the reference axis.
DEFAULT_NEUTRAL_BAND_M = 0.2


@dataclass(frozen=True)
class RouteSideReport:
    """Deterministic route-side classification for one path."""

    side: str
    reason: str | None
    coordinate_frame: str
    start: tuple[float, float]
    goal: tuple[float, float]
    units: str
    tolerance_m: float
    neutral_band_m: float
    progress_interval: tuple[float, float]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "side": self.side,
            "reason": self.reason,
            "coordinate_frame": self.coordinate_frame,
            "start": list(self.start),
            "goal": list(self.goal),
            "units": self.units,
            "tolerance_m": self.tolerance_m,
            "neutral_band_m": self.neutral_band_m,
            "progress_interval": list(self.progress_interval),
        }


@dataclass(frozen=True)
class HomotopyObservation:
    """One homotopy identity observation for a planned path."""

    identity: str | None
    unavailable_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "identity": self.identity,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class TemporalConsistencyReport:
    """Temporal consistency summary across a sequence of replanned paths."""

    valid_count: int
    unavailable_count: int
    side_transition_count: int
    topology_transition_count: int
    dominant_side: str | None
    dominant_topology: str | None
    consistency_fraction: float
    denominator: int
    side_valid_count: int
    topology_valid_count: int
    side_denominator: int
    topology_denominator: int
    aligned_count: int
    alignment_valid: bool
    alignment_reason: str | None
    first_stable_step: int | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "valid_count": self.valid_count,
            "unavailable_count": self.unavailable_count,
            "side_transition_count": self.side_transition_count,
            "topology_transition_count": self.topology_transition_count,
            "dominant_side": self.dominant_side,
            "dominant_topology": self.dominant_topology,
            "consistency_fraction": self.consistency_fraction,
            "denominator": self.denominator,
            "side_valid_count": self.side_valid_count,
            "topology_valid_count": self.topology_valid_count,
            "side_denominator": self.side_denominator,
            "topology_denominator": self.topology_denominator,
            "aligned_count": self.aligned_count,
            "alignment_valid": self.alignment_valid,
            "alignment_reason": self.alignment_reason,
            "first_stable_step": self.first_stable_step,
        }


def _reference_axis(
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    tolerance_m: float,
) -> tuple[float, float] | None:
    """Return the directed reference axis ``(dx, dy)`` or ``None`` if degenerate."""
    dx = float(goal[0]) - float(start[0])
    dy = float(goal[1]) - float(start[1])
    if not np.isfinite(dx) or not np.isfinite(dy):
        return None
    length = float(np.hypot(dx, dy))
    if length <= tolerance_m:
        return None
    return (dx / length, dy / length)


def _cross_axis(axis: tuple[float, float]) -> tuple[float, float]:
    """Return the left-hand perpendicular of the directed axis.

    Facing the goal along ``axis``, the left-hand side is the clockwise
    perpendicular.  For the unit axis ``(1, 0)`` this returns ``(0, -1)`` so
    negative y is left.
    """
    return (axis[1], -axis[0])


def _finite_path(path: list[tuple[float, float]]) -> bool:
    """Return whether every path point is finite."""
    return all(np.isfinite(float(point[0])) and np.isfinite(float(point[1])) for point in path)


def _path_geometry(
    path: list[tuple[float, float]],
    *,
    tolerance_m: float,
) -> tuple[float, float] | None:
    """Return the directed path axis or ``None`` when geometry is degenerate."""
    if len(path) < 2:
        return None
    if not _finite_path(path):
        return None
    dx = float(path[-1][0]) - float(path[0][0])
    dy = float(path[-1][1]) - float(path[0][1])
    length = float(np.hypot(dx, dy))
    if length <= tolerance_m:
        return None
    return (dx / length, dy / length)


def _unavailable_report(
    reason: str,
    *,
    coordinate_frame: str,
    start: tuple[float, float],
    goal: tuple[float, float],
    units: str,
    tolerance_m: float,
    neutral_band_m: float,
    progress_interval: tuple[float, float],
) -> RouteSideReport:
    """Build a fail-closed unavailable route-side report.

    Returns:
        A :class:`RouteSideReport` with ``side=unavailable`` and ``reason``.
    """
    return RouteSideReport(
        side="unavailable",
        reason=reason,
        coordinate_frame=coordinate_frame,
        start=start,
        goal=goal,
        units=units,
        tolerance_m=tolerance_m,
        neutral_band_m=neutral_band_m,
        progress_interval=progress_interval,
    )


def _finite_nonnegative(value: Any) -> float | None:
    """Return a finite non-negative float, or ``None`` for invalid input."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def _finite_point(value: Any) -> tuple[float, float] | None:
    """Return a finite two-dimensional point, or ``None`` for malformed input."""
    try:
        point = (float(value[0]), float(value[1]))
    except (IndexError, TypeError, ValueError):
        return None
    if not np.isfinite(point[0]) or not np.isfinite(point[1]):
        return None
    return point


def _valid_progress_interval(value: Any) -> tuple[float, float] | None:
    """Return a strictly increasing normalized progress interval, or ``None``."""
    try:
        lo, hi = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(lo) or not np.isfinite(hi) or not 0.0 <= lo < hi <= 1.0:
        return None
    return (lo, hi)


def _bucket_signed(
    signed: float,
    *,
    neutral_band_m: float,
    left_seen: bool,
    right_seen: bool,
    neutral_seen: bool,
) -> tuple[bool, bool, bool]:
    """Bucket one signed perpendicular distance into side flags.

    Returns:
        Updated ``(left_seen, right_seen, neutral_seen)`` flags.
    """
    if abs(signed) <= neutral_band_m:
        return left_seen, right_seen, True
    if signed > 0:
        return True, right_seen, neutral_seen
    return left_seen, True, neutral_seen


def classify_route_side(  # noqa: C901 - fail-closed parameter and geometry gates
    path: list[tuple[float, float]],
    *,
    start: tuple[float, float],
    goal: tuple[float, float],
    coordinate_frame: str = "global_xy",
    units: str = "m",
    tolerance_m: float = DEFAULT_SIDE_TOLERANCE_M,
    neutral_band_m: float = DEFAULT_NEUTRAL_BAND_M,
    progress_interval: tuple[float, float] = (0.1, 0.9),
) -> RouteSideReport:
    """Classify which side of the directed start-to-goal axis ``path`` uses.

    The signed perpendicular distance of each path point from the reference
    axis is computed with a left-hand positive convention.  Points inside the
    neutral band count as ``neutral``; a path that visits both strict sides is
    ``mixed``; missing or degenerate geometry fails closed as ``unavailable``
    with an explicit reason.

    Returns:
        A :class:`RouteSideReport` with the bounded side vocabulary.
    """

    normalized_tolerance = _finite_nonnegative(tolerance_m)
    normalized_band = _finite_nonnegative(neutral_band_m)
    normalized_interval = _valid_progress_interval(progress_interval)
    normalized_start = _finite_point(start)
    normalized_goal = _finite_point(goal)
    invalid_reason: str | None = None
    if normalized_tolerance is None:
        invalid_reason = "invalid_tolerance"
    elif normalized_band is None:
        invalid_reason = "invalid_neutral_band"
    elif normalized_interval is None:
        invalid_reason = "invalid_progress_interval"
    elif normalized_start is None or normalized_goal is None:
        invalid_reason = "invalid_reference"
    if invalid_reason is not None:
        return _unavailable_report(
            invalid_reason,
            coordinate_frame=coordinate_frame,
            start=normalized_start or (0.0, 0.0),
            goal=normalized_goal or (0.0, 0.0),
            units=units,
            tolerance_m=normalized_tolerance or 0.0,
            neutral_band_m=normalized_band or 0.0,
            progress_interval=normalized_interval or (0.0, 1.0),
        )

    assert normalized_tolerance is not None
    assert normalized_band is not None
    assert normalized_interval is not None
    assert normalized_start is not None
    assert normalized_goal is not None
    tolerance_m = normalized_tolerance
    neutral_band_m = normalized_band
    progress_interval = normalized_interval
    start = normalized_start
    goal = normalized_goal

    axis = _reference_axis(start, goal, tolerance_m=tolerance_m)
    if not path:
        reason = "empty_path"
    elif len(path) == 1:
        reason = "single_point"
    elif not _finite_path(path):
        reason = "non_finite"
    elif axis is None:
        reason = "degenerate_reference"
    elif _path_geometry(path, tolerance_m=tolerance_m) is None:
        reason = "zero_length"
    else:
        reason = None
    if reason is not None:
        return _unavailable_report(
            reason,
            coordinate_frame=coordinate_frame,
            start=start,
            goal=goal,
            units=units,
            tolerance_m=tolerance_m,
            neutral_band_m=neutral_band_m,
            progress_interval=progress_interval,
        )

    normal = _cross_axis(axis)
    start_arr = np.asarray(start, dtype=float)
    left_seen = False
    right_seen = False
    neutral_seen = False
    classified = 0
    lo, hi = progress_interval
    total = max(len(path) - 1, 1)
    for index, point in enumerate(path):
        progress = index / total if total else 0.0
        if progress < lo or progress > hi:
            continue
        offset = np.asarray(point, dtype=float) - start_arr
        signed = float(np.dot(offset, normal))
        left_seen, right_seen, neutral_seen = _bucket_signed(
            signed,
            neutral_band_m=neutral_band_m,
            left_seen=left_seen,
            right_seen=right_seen,
            neutral_seen=neutral_seen,
        )
        classified += 1

    if classified == 0:
        return _unavailable_report(
            "insufficient_progress",
            coordinate_frame=coordinate_frame,
            start=start,
            goal=goal,
            units=units,
            tolerance_m=tolerance_m,
            neutral_band_m=neutral_band_m,
            progress_interval=progress_interval,
        )

    return RouteSideReport(
        side=_side_from_flags(left_seen, right_seen, neutral_seen),
        reason=None,
        coordinate_frame=coordinate_frame,
        start=start,
        goal=goal,
        units=units,
        tolerance_m=tolerance_m,
        neutral_band_m=neutral_band_m,
        progress_interval=progress_interval,
    )


def _side_from_flags(left_seen: bool, right_seen: bool, neutral_seen: bool) -> str:
    """Resolve the bounded side vocabulary from the bucket flags.

    Returns:
        One of ``mixed``, ``left``, ``right``, ``neutral``.
    """
    if left_seen and right_seen:
        return "mixed"
    if left_seen:
        return "left"
    if right_seen:
        return "right"
    return "neutral"


def topology_signature(
    path: list[tuple[int, int]],
    blocked: np.ndarray,
    clearance_map: np.ndarray,
    *,
    clearance_threshold_cells: int,
) -> frozenset[tuple[int, int]]:
    """Return the canonical low-clearance corridor signature for a grid path.

    Choke cells take precedence, matching the original topology diagnostic.  If
    no choke cell is present, the same diagnostic falls back to path cells whose
    finite clearance is within ``clearance_threshold_cells``.  Keeping this
    implementation shared prevents route observability and topology diagnostics
    from silently acquiring different identity semantics.
    """

    if blocked.ndim != 2 or clearance_map.shape != blocked.shape:
        return frozenset()
    choke_cells: set[tuple[int, int]] = set()
    rows, cols = blocked.shape
    for row, col in path:
        up_blocked = row <= 0 or bool(blocked[row - 1, col])
        down_blocked = row >= rows - 1 or bool(blocked[row + 1, col])
        left_blocked = col <= 0 or bool(blocked[row, col - 1])
        right_blocked = col >= cols - 1 or bool(blocked[row, col + 1])
        if (up_blocked and down_blocked) or (left_blocked and right_blocked):
            choke_cells.add((row, col))
    if choke_cells:
        return frozenset(choke_cells)

    threshold = max(int(clearance_threshold_cells), 1)
    return frozenset(
        cell
        for cell in path
        if np.isfinite(float(clearance_map[cell])) and float(clearance_map[cell]) <= threshold
    )


def homotopy_identity(  # noqa: C901 - fail-closed map, path, and threshold gates
    path: list[tuple[float, float]],
    blocked: np.ndarray,
    *,
    clearance_threshold_cells: int = 2,
) -> HomotopyObservation:
    """Return a stable compact corridor identity for a grid-cell path.

    The identity is derived from low-clearance (choke) cells of the path
    relative to the blocked map, mirroring the ``_topology_signature`` idea
    from the topology-hypothesis diagnostics.  It is stable across discovery
    order and does not depend on ephemeral route names.

    ``path`` points use the grid convention ``(row, col)`` matching the
    blocked map's index order (row 0 is the top edge).

    Returns:
        A :class:`HomotopyObservation` with the identity string or an
        unavailable reason.
    """

    if not path:
        return HomotopyObservation(identity=None, unavailable_reason="empty_path")
    if len(path) == 1:
        return HomotopyObservation(identity=None, unavailable_reason="single_point")
    if not _finite_path(path):
        return HomotopyObservation(identity=None, unavailable_reason="non_finite")
    try:
        blocked_map = np.asarray(blocked)
    except (TypeError, ValueError):
        return HomotopyObservation(identity=None, unavailable_reason="malformed_blocked_map")
    if blocked_map.ndim != 2:
        return HomotopyObservation(identity=None, unavailable_reason="malformed_blocked_map")
    if blocked_map.size == 0:
        return HomotopyObservation(identity=None, unavailable_reason="missing_blocked_map")
    if np.issubdtype(blocked_map.dtype, np.number):
        if not np.isfinite(blocked_map).all() or not np.isin(blocked_map, [0, 1]).all():
            return HomotopyObservation(identity=None, unavailable_reason="invalid_blocked_map")
    elif blocked_map.dtype != np.dtype(bool):
        return HomotopyObservation(identity=None, unavailable_reason="invalid_blocked_map")

    try:
        threshold_value = float(clearance_threshold_cells)
    except (TypeError, ValueError):
        return HomotopyObservation(identity=None, unavailable_reason="invalid_clearance_threshold")
    if (
        not np.isfinite(threshold_value)
        or threshold_value < 1.0
        or not threshold_value.is_integer()
    ):
        return HomotopyObservation(identity=None, unavailable_reason="invalid_clearance_threshold")

    blocked_map = blocked_map.astype(bool, copy=False)

    rows, cols = blocked_map.shape
    grid_path: list[tuple[int, int]] = []
    for point in path:
        row = round(float(point[0]))
        col = round(float(point[1]))
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return HomotopyObservation(identity=None, unavailable_reason="out_of_bounds")
        grid_path.append((row, col))

    clearance_map = GridRoutePlannerAdapter._compute_clearance_map(blocked_map)
    signature = topology_signature(
        grid_path,
        blocked_map,
        clearance_map,
        clearance_threshold_cells=int(threshold_value),
    )
    if not signature:
        return HomotopyObservation(identity=None, unavailable_reason="no_choke_cells")
    # Canonical, order-independent identity: sorted choke cells joined by '|'.
    identity = ";".join(f"{row},{col}" for row, col in sorted(signature))
    return HomotopyObservation(identity=identity, unavailable_reason=None)


def temporal_consistency(
    side_reports: list[RouteSideReport],
    homotopy_observations: list[HomotopyObservation],
) -> TemporalConsistencyReport:
    """Summarize route-side and topology consistency across replans.

    Valid and unavailable observations are counted separately; side and
    topology transition counts are computed over consecutive valid entries;
    the consistency fraction uses the valid count as denominator.  Outputs are
    never merged into a single social-compliance score.

    Returns:
        A :class:`TemporalConsistencyReport`.
    """

    side_denominator = len(side_reports)
    topology_denominator = len(homotopy_observations)
    alignment_valid = side_denominator == topology_denominator
    if not alignment_valid:
        return TemporalConsistencyReport(
            valid_count=0,
            unavailable_count=max(side_denominator, topology_denominator),
            side_transition_count=0,
            topology_transition_count=0,
            dominant_side=None,
            dominant_topology=None,
            consistency_fraction=0.0,
            denominator=0,
            side_valid_count=sum(report.side != "unavailable" for report in side_reports),
            topology_valid_count=sum(obs.identity is not None for obs in homotopy_observations),
            side_denominator=side_denominator,
            topology_denominator=topology_denominator,
            aligned_count=0,
            alignment_valid=False,
            alignment_reason="length_mismatch",
            first_stable_step=None,
        )

    aligned_count = side_denominator
    side_valid_indices = [
        index for index, report in enumerate(side_reports) if report.side != "unavailable"
    ]
    topology_valid_indices = [
        index for index, observation in enumerate(homotopy_observations) if observation.identity
    ]
    valid_pair_indices = [
        index
        for index in range(aligned_count)
        if index in side_valid_indices and index in topology_valid_indices
    ]
    valid_sides = [side_reports[index].side for index in side_valid_indices]
    valid_topologies = [homotopy_observations[index].identity for index in topology_valid_indices]
    valid_count = len(valid_pair_indices)
    unavailable_count = aligned_count - valid_count

    side_transitions = sum(
        1
        for index in range(1, aligned_count)
        if index in side_valid_indices
        and index - 1 in side_valid_indices
        and side_reports[index].side != side_reports[index - 1].side
    )
    topology_transitions = sum(
        1
        for index in range(1, aligned_count)
        if index in topology_valid_indices
        and index - 1 in topology_valid_indices
        and homotopy_observations[index].identity != homotopy_observations[index - 1].identity
    )

    dominant_side = _dominant(valid_sides)
    dominant_topology = _dominant(valid_topologies)
    consistency_fraction = float(valid_count / aligned_count) if aligned_count else 0.0

    first_stable_step: int | None = None
    if len(valid_sides) >= 2:
        first_stable = valid_sides[0]
        stable_run = 0
        for index, side in enumerate(valid_sides):
            if side == first_stable:
                stable_run += 1
            else:
                break
        if stable_run == len(valid_sides):
            first_stable_step = 0

    return TemporalConsistencyReport(
        valid_count=valid_count,
        unavailable_count=unavailable_count,
        side_transition_count=side_transitions,
        topology_transition_count=topology_transitions,
        dominant_side=dominant_side,
        dominant_topology=dominant_topology,
        consistency_fraction=consistency_fraction,
        denominator=aligned_count,
        side_valid_count=len(side_valid_indices),
        topology_valid_count=len(topology_valid_indices),
        side_denominator=side_denominator,
        topology_denominator=topology_denominator,
        aligned_count=aligned_count,
        alignment_valid=True,
        alignment_reason=None,
        first_stable_step=first_stable_step,
    )


def _dominant(values: list[str]) -> str | None:
    """Return the most frequent value, or ``None`` when empty or tied."""
    if not values:
        return None
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]
