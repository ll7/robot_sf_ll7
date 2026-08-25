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
from itertools import pairwise
from typing import Any

import numpy as np

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


def classify_route_side(
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


def homotopy_identity(
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
    if blocked.size == 0:
        return HomotopyObservation(identity=None, unavailable_reason="missing_blocked_map")

    rows, cols = blocked.shape
    choke_cells: set[tuple[int, int]] = set()
    for point in path:
        row = round(float(point[0]))
        col = round(float(point[1]))
        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue
        up_blocked = row <= 0 or bool(blocked[row - 1, col])
        down_blocked = row >= rows - 1 or bool(blocked[row + 1, col])
        left_blocked = col <= 0 or bool(blocked[row, col - 1])
        right_blocked = col >= cols - 1 or bool(blocked[row, col + 1])
        if (up_blocked and down_blocked) or (left_blocked and right_blocked):
            choke_cells.add((row, col))
    if not choke_cells:
        return HomotopyObservation(identity=None, unavailable_reason="no_choke_cells")
    # Canonical, order-independent identity: sorted choke cells joined by '|'.
    identity = ";".join(f"{row},{col}" for row, col in sorted(choke_cells))
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

    valid_sides = [report.side for report in side_reports if report.side != "unavailable"]
    valid_topologies = [obs.identity for obs in homotopy_observations if obs.identity is not None]
    unavailable_count = len(side_reports) - len(valid_sides)
    denominator = len(side_reports)
    valid_count = len(valid_sides)

    side_transitions = sum(1 for a, b in pairwise(valid_sides) if a != b)
    topology_transitions = sum(1 for a, b in pairwise(valid_topologies) if a != b)

    dominant_side = _dominant(valid_sides)
    dominant_topology = _dominant(valid_topologies)
    consistency_fraction = float(valid_count / denominator) if denominator else 0.0

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
        denominator=denominator,
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
