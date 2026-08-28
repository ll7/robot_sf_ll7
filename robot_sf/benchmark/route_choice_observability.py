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
from typing import Any, cast

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
        "invalid_path",
        "unknown",
    }
)

#: Default numerical tolerance for the signed cross-product side test.
DEFAULT_SIDE_TOLERANCE_M = 0.05
#: Default neutral-band half-width around the reference axis.
DEFAULT_NEUTRAL_BAND_M = 0.2
#: Versioned JSON-ready diagnostic record emitted by :func:`diagnostic_record`.
DIAGNOSTIC_SCHEMA_VERSION = "route_choice_observability.v1"
#: Conservative interpretation boundary carried by every diagnostic record.
DIAGNOSTIC_CLAIM_BOUNDARY = (
    "planner-route observability only; not pedestrian preference, response, comfort, "
    "or general human predictability"
)


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
    identity_coordinate_frame: str = "occupancy_grid_rc"
    identity_units: str = "cells"
    identity_points: tuple[tuple[float, float], ...] = ()
    identity_match_tolerance: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "identity": self.identity,
            "unavailable_reason": self.unavailable_reason,
            "identity_coordinate_frame": self.identity_coordinate_frame,
            "identity_units": self.identity_units,
            "identity_points": [list(point) for point in self.identity_points],
            "identity_match_tolerance": self.identity_match_tolerance,
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
    availability_fraction: float
    availability_denominator: int
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
            "availability_fraction": self.availability_fraction,
            "availability_denominator": self.availability_denominator,
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

    Robot SF's global and ego XY frames use the standard counter-clockwise
    convention: for the unit axis ``(1, 0)``, ``(0, 1)`` is left.
    """
    return (-axis[1], axis[0])


def _path_problem(path: list[tuple[float, float]]) -> str | None:
    """Return a fail-closed path-shape or finiteness reason, if any."""
    for point in path:
        try:
            x, y = point
            x_value = float(x)
            y_value = float(y)
        except (TypeError, ValueError):
            return "invalid_path"
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            return "non_finite"
    return None


def _path_geometry(
    path: list[tuple[float, float]],
    *,
    tolerance_m: float,
) -> tuple[float, float] | None:
    """Return the directed path axis or ``None`` when geometry is degenerate."""
    if len(path) < 2:
        return None
    if _path_problem(path) is not None:
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
        x, y = value
        point = (float(x), float(y))
    except (TypeError, ValueError):
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


def _signed_range_over_progress_interval(
    path: list[tuple[float, float]],
    *,
    start: tuple[float, float],
    normal: tuple[float, float],
    progress_interval: tuple[float, float],
) -> tuple[float, float] | None:
    """Return signed-distance extrema over an arc-length-clipped path interval.

    Treating progress as normalized cumulative arc length makes the result
    invariant to repeated points and to inserting samples along existing
    segments.  Evaluating the clipped segment endpoints is sufficient because
    signed distance varies linearly along each segment.
    """
    points = np.asarray(path, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    if not np.isfinite(total_length) or total_length <= 0.0:
        return None

    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths))) / total_length
    start_arr = np.asarray(start, dtype=float)
    normal_arr = np.asarray(normal, dtype=float)
    lo, hi = progress_interval
    signed_values: list[float] = []
    for index, segment_length in enumerate(segment_lengths):
        if segment_length <= 0.0:
            continue
        segment_lo = float(cumulative[index])
        segment_hi = float(cumulative[index + 1])
        overlap_lo = max(lo, segment_lo)
        overlap_hi = min(hi, segment_hi)
        if overlap_lo > overlap_hi:
            continue
        denominator = segment_hi - segment_lo
        for progress in (overlap_lo, overlap_hi):
            fraction = (progress - segment_lo) / denominator
            point = points[index] + fraction * (points[index + 1] - points[index])
            signed_values.append(float(np.dot(point - start_arr, normal_arr)))
    if not signed_values:
        return None
    return min(signed_values), max(signed_values)


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
    elif (
        normalized_start is None
        or normalized_goal is None
        or not isinstance(coordinate_frame, str)
        or not coordinate_frame.strip()
        or not isinstance(units, str)
        or not units.strip()
    ):
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

    tolerance_m = cast("float", normalized_tolerance)
    neutral_band_m = cast("float", normalized_band)
    progress_interval = cast("tuple[float, float]", normalized_interval)
    start = cast("tuple[float, float]", normalized_start)
    goal = cast("tuple[float, float]", normalized_goal)

    axis = _reference_axis(start, goal, tolerance_m=tolerance_m)
    if not path:
        reason = "empty_path"
    elif len(path) == 1:
        reason = "single_point"
    elif (path_problem := _path_problem(path)) is not None:
        reason = path_problem
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
    signed_range = _signed_range_over_progress_interval(
        path,
        start=start,
        normal=normal,
        progress_interval=progress_interval,
    )
    if signed_range is None:
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

    minimum_signed, maximum_signed = signed_range
    side_threshold_m = neutral_band_m + tolerance_m
    left_seen = maximum_signed > side_threshold_m
    right_seen = minimum_signed < -side_threshold_m
    return RouteSideReport(
        side=_side_from_flags(left_seen, right_seen, not left_seen and not right_seen),
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


def _has_only_8_connected_steps(path: list[tuple[int, int]]) -> bool:
    """Return whether consecutive grid cells are duplicates or 8-connected."""
    return all(
        max(abs(current[0] - previous[0]), abs(current[1] - previous[1])) <= 1
        for previous, current in pairwise(path)
    )


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

    if (
        blocked.ndim != 2
        or clearance_map.shape != blocked.shape
        or not _has_only_8_connected_steps(path)
    ):
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


def homotopy_identity(  # noqa: C901, PLR0912 - fail-closed map, path, and threshold gates
    path: list[tuple[float, float]],
    blocked: np.ndarray,
    *,
    clearance_threshold_cells: int = 2,
    identity_coordinates: list[tuple[float, float]] | None = None,
    identity_coordinate_frame: str = "occupancy_grid_rc",
    identity_units: str = "cells",
    identity_match_tolerance: float | None = None,
) -> HomotopyObservation:
    """Return a stable compact corridor identity for a grid-cell path.

    The identity is derived from low-clearance (choke) cells of the path
    relative to the blocked map, mirroring the ``_topology_signature`` idea
    from the topology-hypothesis diagnostics.  It is stable across discovery
    order and does not depend on ephemeral route names.

    ``path`` points use the grid convention ``(row, col)`` matching the
    blocked map's index order (row 0 is the top edge). When
    ``identity_coordinates`` is supplied, it must align one-for-one with
    ``path`` and provides the immutable coordinate frame used to serialize the
    compact signature. Production ego-grid callers use global world points so
    grid motion cannot create false topology transitions.

    Returns:
        A :class:`HomotopyObservation` with the identity string or an
        unavailable reason.
    """

    def _unavailable(reason: str) -> HomotopyObservation:
        return HomotopyObservation(
            identity=None,
            unavailable_reason=reason,
            identity_coordinate_frame=identity_coordinate_frame,
            identity_units=identity_units,
            identity_match_tolerance=match_tolerance_value,
        )

    match_tolerance_value: float | None = None
    if identity_match_tolerance is not None:
        try:
            match_tolerance_value = float(identity_match_tolerance)
        except (TypeError, ValueError):
            return _unavailable("invalid_identity_match_tolerance")
        if not np.isfinite(match_tolerance_value) or match_tolerance_value <= 0.0:
            return _unavailable("invalid_identity_match_tolerance")

    if not path:
        return _unavailable("empty_path")
    if len(path) == 1:
        return _unavailable("single_point")
    if (path_problem := _path_problem(path)) is not None:
        return _unavailable(path_problem)
    try:
        blocked_map = np.asarray(blocked)
    except (TypeError, ValueError):
        return _unavailable("malformed_blocked_map")
    if blocked_map.ndim != 2:
        return _unavailable("malformed_blocked_map")
    if blocked_map.size == 0:
        return _unavailable("missing_blocked_map")
    if np.issubdtype(blocked_map.dtype, np.number):
        if not np.isfinite(blocked_map).all() or not np.isin(blocked_map, [0, 1]).all():
            return _unavailable("invalid_blocked_map")
    elif blocked_map.dtype != np.dtype(bool):
        return _unavailable("invalid_blocked_map")

    try:
        threshold_value = float(clearance_threshold_cells)
    except (TypeError, ValueError):
        return _unavailable("invalid_clearance_threshold")
    if (
        not np.isfinite(threshold_value)
        or threshold_value < 1.0
        or not threshold_value.is_integer()
    ):
        return _unavailable("invalid_clearance_threshold")

    blocked_map = blocked_map.astype(bool, copy=False)

    rows, cols = blocked_map.shape
    grid_path: list[tuple[int, int]] = []
    for point in path:
        row_value = float(point[0])
        col_value = float(point[1])
        if not row_value.is_integer() or not col_value.is_integer():
            return _unavailable("non_integral_grid_cell")
        row = int(row_value)
        col = int(col_value)
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return _unavailable("out_of_bounds")
        if blocked_map[row, col]:
            return _unavailable("path_intersects_blocked")
        grid_path.append((row, col))

    if not _has_only_8_connected_steps(grid_path):
        return _unavailable("non_adjacent_grid_step")

    if not isinstance(identity_coordinate_frame, str) or not identity_coordinate_frame.strip():
        return _unavailable("invalid_identity_coordinates")
    if not isinstance(identity_units, str) or not identity_units.strip():
        return _unavailable("invalid_identity_coordinates")
    if identity_coordinates is None:
        canonical_coordinates = [(float(row), float(col)) for row, col in grid_path]
    else:
        if len(identity_coordinates) != len(grid_path):
            return _unavailable("invalid_identity_coordinates")
        canonical_coordinates = []
        for point in identity_coordinates:
            normalized_point = _finite_point(point)
            if normalized_point is None:
                return _unavailable("invalid_identity_coordinates")
            canonical_coordinates.append(normalized_point)
    clearance_map = GridRoutePlannerAdapter._compute_clearance_map(blocked_map)
    signature = topology_signature(
        grid_path,
        blocked_map,
        clearance_map,
        clearance_threshold_cells=int(threshold_value),
    )
    if not signature:
        return _unavailable("no_choke_cells")

    coordinate_by_cell: dict[tuple[int, int], tuple[float, float]] = {}
    for cell, coordinate in zip(grid_path, canonical_coordinates, strict=True):
        previous = coordinate_by_cell.setdefault(cell, coordinate)
        if not np.allclose(previous, coordinate, rtol=0.0, atol=1e-9):
            return _unavailable("invalid_identity_coordinates")

    def _format_coordinate(value: float) -> str:
        normalized = 0.0 if abs(value) <= 1e-12 else value
        return format(normalized, ".12g")

    identity_points = sorted(coordinate_by_cell[cell] for cell in signature)
    identity = ";".join(
        f"{_format_coordinate(first)},{_format_coordinate(second)}"
        for first, second in identity_points
    )
    return HomotopyObservation(
        identity=identity,
        unavailable_reason=None,
        identity_coordinate_frame=identity_coordinate_frame,
        identity_units=identity_units,
        identity_points=tuple(identity_points),
        identity_match_tolerance=match_tolerance_value,
    )


def _homotopy_equivalent(left: HomotopyObservation, right: HomotopyObservation) -> bool:
    """Return whether two identities denote one topology under their declared tolerance."""
    if left.identity is None or right.identity is None:
        return False
    if (
        left.identity_coordinate_frame != right.identity_coordinate_frame
        or left.identity_units != right.identity_units
        or left.identity_match_tolerance != right.identity_match_tolerance
    ):
        return False
    tolerance = left.identity_match_tolerance
    if tolerance is None or not left.identity_points or not right.identity_points:
        return left.identity == right.identity
    left_points = np.asarray(left.identity_points, dtype=float)
    right_points = np.asarray(right.identity_points, dtype=float)
    distances = np.linalg.norm(left_points[:, None, :] - right_points[None, :, :], axis=2)
    symmetric_hausdorff = max(
        float(np.max(np.min(distances, axis=1))),
        float(np.max(np.min(distances, axis=0))),
    )
    # Strict comparison keeps corridors separated by one complete grid cell distinct. Exclude
    # floating-point values numerically equal to the boundary before applying ``<``.
    boundary_equal = bool(
        np.isclose(
            symmetric_hausdorff,
            tolerance,
            rtol=1e-9,
            atol=max(1e-12, tolerance * 1e-9),
        )
    )
    return symmetric_hausdorff < tolerance and not boundary_equal


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

    def _alignment_failure(reason: str) -> TemporalConsistencyReport:
        return TemporalConsistencyReport(
            valid_count=0,
            unavailable_count=max(side_denominator, topology_denominator),
            side_transition_count=0,
            topology_transition_count=0,
            dominant_side=None,
            dominant_topology=None,
            consistency_fraction=0.0,
            denominator=0,
            availability_fraction=0.0,
            availability_denominator=max(side_denominator, topology_denominator),
            side_valid_count=sum(report.side != "unavailable" for report in side_reports),
            topology_valid_count=sum(obs.identity is not None for obs in homotopy_observations),
            side_denominator=side_denominator,
            topology_denominator=topology_denominator,
            aligned_count=0,
            alignment_valid=False,
            alignment_reason=reason,
            first_stable_step=None,
        )

    if side_denominator != topology_denominator:
        return _alignment_failure("length_mismatch")

    route_references = {
        (
            report.coordinate_frame,
            report.units,
            report.start,
            report.goal,
            report.tolerance_m,
            report.neutral_band_m,
            report.progress_interval,
        )
        for report in side_reports
        if report.side != "unavailable"
    }
    if len(route_references) > 1:
        return _alignment_failure("route_reference_mismatch")

    identity_references = {
        (
            observation.identity_coordinate_frame,
            observation.identity_units,
            observation.identity_match_tolerance,
        )
        for observation in homotopy_observations
        if observation.identity is not None
    }
    if len(identity_references) > 1:
        return _alignment_failure("identity_reference_mismatch")

    aligned_count = side_denominator
    side_valid_indices = [
        index for index, report in enumerate(side_reports) if report.side != "unavailable"
    ]
    topology_valid_indices = [
        index for index, observation in enumerate(homotopy_observations) if observation.identity
    ]
    topology_cluster_labels: dict[int, str] = {}
    topology_clusters: list[list[int]] = []
    for index in topology_valid_indices:
        cluster = next(
            (
                candidate_cluster
                for candidate_cluster in topology_clusters
                if all(
                    _homotopy_equivalent(
                        homotopy_observations[index], homotopy_observations[candidate]
                    )
                    for candidate in candidate_cluster
                )
            ),
            None,
        )
        if cluster is None:
            cluster = []
            topology_clusters.append(cluster)
        cluster.append(index)
        topology_cluster_labels[index] = str(homotopy_observations[cluster[0]].identity)
    valid_pair_indices = [
        index
        for index in range(aligned_count)
        if index in side_valid_indices and index in topology_valid_indices
    ]
    valid_sides = [side_reports[index].side for index in side_valid_indices]
    valid_topologies = [topology_cluster_labels[index] for index in topology_valid_indices]
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
        and not _homotopy_equivalent(homotopy_observations[index], homotopy_observations[index - 1])
    )

    dominant_side = _dominant(valid_sides)
    dominant_topology = _dominant(valid_topologies)
    valid_pairs = [
        (side_reports[index].side, topology_cluster_labels[index]) for index in valid_pair_indices
    ]
    pair_counts: dict[tuple[str, str], int] = {}
    for pair in valid_pairs:
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    consistency_fraction = float(max(pair_counts.values()) / valid_count) if valid_count else 0.0
    availability_fraction = float(valid_count / aligned_count) if aligned_count else 0.0

    aligned_pairs: list[tuple[str, str] | None] = [None] * aligned_count
    for index, pair in zip(valid_pair_indices, valid_pairs, strict=True):
        aligned_pairs[index] = pair
    first_stable_step = _first_stable_pair_step(aligned_pairs)

    return TemporalConsistencyReport(
        valid_count=valid_count,
        unavailable_count=unavailable_count,
        side_transition_count=side_transitions,
        topology_transition_count=topology_transitions,
        dominant_side=dominant_side,
        dominant_topology=dominant_topology,
        consistency_fraction=consistency_fraction,
        denominator=valid_count,
        availability_fraction=availability_fraction,
        availability_denominator=aligned_count,
        side_valid_count=len(side_valid_indices),
        topology_valid_count=len(topology_valid_indices),
        side_denominator=side_denominator,
        topology_denominator=topology_denominator,
        aligned_count=aligned_count,
        alignment_valid=True,
        alignment_reason=None,
        first_stable_step=first_stable_step,
    )


def diagnostic_record(
    side_reports: list[RouteSideReport],
    homotopy_observations: list[HomotopyObservation],
) -> dict[str, Any]:
    """Return one versioned JSON-ready route-choice diagnostic record.

    The record intentionally carries observation and availability semantics,
    not a benchmark result or social-compliance score.
    """
    temporal = temporal_consistency(side_reports, homotopy_observations)
    status = (
        "available" if temporal.alignment_valid and temporal.valid_count > 0 else "not_available"
    )
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "evidence_tier": "analysis-only",
        "result_classification": "diagnostic-only",
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
        "status": status,
        "route_side_observations": [report.as_dict() for report in side_reports],
        "homotopy_observations": [observation.as_dict() for observation in homotopy_observations],
        "temporal_consistency": temporal.as_dict(),
    }


def _first_stable_pair_step(values: list[tuple[str, str] | None]) -> int | None:
    """Return the first step beginning a stable two-sample route-choice suffix.

    Unavailable samples invalidate the suffix instead of being bridged.  A
    single final observation is insufficient to establish stability.
    """
    for index in range(max(len(values) - 1, 0)):
        value = values[index]
        if value is None or values[index + 1] != value:
            continue
        if all(candidate == value for candidate in values[index:]):
            return index
    return None


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
