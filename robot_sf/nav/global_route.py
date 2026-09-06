"""Route utilities for global navigation."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from math import dist, isfinite
from typing import Literal

from robot_sf.common.types import Rect, Vec2D

_DEFAULT_DUPLICATE_TOLERANCE_M = 1e-9
_DEFAULT_TIE_TOLERANCE_M = 1e-9
RouteProjectionStatus = Literal["ok", "ambiguous", "invalid_query"]


@dataclass(frozen=True, slots=True)
class RouteProjection:
    """Result of projecting one point onto a route polyline.

    ``arc_length_m`` and the geometric fields are populated only for a valid,
    unambiguous projection.  An ``ambiguous`` result intentionally carries no
    route branch so callers cannot mistake a self-intersection tie for valid
    route progress.
    """

    status: RouteProjectionStatus
    query: Vec2D | None
    projected_point: Vec2D | None
    segment_index: int | None
    arc_length_m: float | None
    distance_m: float | None
    lateral_offset_m: float | None

    @property
    def is_valid(self) -> bool:
        """Whether the projection identifies one unambiguous route branch."""

        return self.status == "ok"


@dataclass(frozen=True, slots=True)
class RouteGeometry:
    """Deterministic geometry view over an ordered route polyline.

    Consecutive points no farther apart than ``duplicate_tolerance_m`` are
    collapsed.  This keeps zero-length sections out of projection and
    interpolation while preserving the route's order.  A route must retain at
    least two distinct finite points after normalization.
    """

    waypoints: tuple[Vec2D, ...]
    duplicate_tolerance_m: float = _DEFAULT_DUPLICATE_TOLERANCE_M
    tie_tolerance_m: float = _DEFAULT_TIE_TOLERANCE_M

    def __post_init__(self) -> None:
        """Normalize and validate route points and numeric tolerances."""

        _validate_tolerance(self.duplicate_tolerance_m, "duplicate_tolerance_m")
        _validate_tolerance(self.tie_tolerance_m, "tie_tolerance_m")

        normalized: list[Vec2D] = []
        for point in self.waypoints:
            candidate = _finite_point(point, "waypoint")
            if not normalized or dist(normalized[-1], candidate) > self.duplicate_tolerance_m:
                normalized.append(candidate)
        if len(normalized) < 2:
            raise ValueError("RouteGeometry needs at least two distinct finite waypoints.")
        object.__setattr__(self, "waypoints", tuple(normalized))

    @property
    def sections(self) -> tuple[tuple[Vec2D, Vec2D], ...]:
        """Consecutive non-zero-length route sections."""

        return tuple(zip(self.waypoints[:-1], self.waypoints[1:], strict=True))

    @property
    def section_lengths(self) -> tuple[float, ...]:
        """Euclidean length of each normalized route section."""

        return tuple(dist(start, end) for start, end in self.sections)

    @property
    def section_offsets(self) -> tuple[float, ...]:
        """Cumulative arc length at each section start."""

        offsets: list[float] = []
        offset = 0.0
        for length in self.section_lengths:
            offsets.append(offset)
            offset += length
        return tuple(offsets)

    @property
    def total_length_m(self) -> float:
        """Total normalized route arc length in metres."""

        return sum(self.section_lengths)

    @property
    def route_hash(self) -> str:
        """Stable SHA-256 identity of the normalized route geometry."""

        payload = dumps(self.waypoints, separators=(",", ":"), allow_nan=False)
        return sha256(payload.encode("utf-8")).hexdigest()

    def point_at_arc_length(self, arc_length_m: float) -> Vec2D:
        """Return the route point at clamped cumulative arc length.

        Args:
            arc_length_m: Requested distance from the first route point.

        Returns:
            Vec2D: Interpolated point on the normalized polyline.

        Raises:
            ValueError: If ``arc_length_m`` is not finite.
        """

        if not isfinite(arc_length_m):
            raise ValueError("arc_length_m must be finite.")
        if arc_length_m <= 0.0:
            return self.waypoints[0]
        total_length = self.total_length_m
        if arc_length_m >= total_length:
            return self.waypoints[-1]

        for (start, end), offset, length in zip(
            self.sections,
            self.section_offsets,
            self.section_lengths,
            strict=True,
        ):
            if arc_length_m <= offset + length:
                fraction = (arc_length_m - offset) / length
                return (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
        return self.waypoints[-1]

    def project(self, query: Sequence[float]) -> RouteProjection:
        """Project a query point onto the nearest route section.

        The signed lateral offset is positive to the left of the directed
        section.  If multiple distinct route branches are within the tie
        tolerance of the nearest distance, the result is ``ambiguous`` and no
        branch-specific progress is returned.

        Returns:
            RouteProjection: Valid projection data, or an explicit invalid or
                ambiguous status without route-relative progress.
        """

        try:
            point = _finite_point(query, "query")
        except ValueError:
            return RouteProjection(
                status="invalid_query",
                query=None,
                projected_point=None,
                segment_index=None,
                arc_length_m=None,
                distance_m=None,
                lateral_offset_m=None,
            )

        candidates: list[tuple[float, int, Vec2D, float, float]] = []
        for index, ((start, end), offset, length) in enumerate(
            zip(self.sections, self.section_offsets, self.section_lengths, strict=True)
        ):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            query_dx = point[0] - start[0]
            query_dy = point[1] - start[1]
            fraction = max(0.0, min(1.0, (query_dx * dx + query_dy * dy) / (length * length)))
            projected = (start[0] + fraction * dx, start[1] + fraction * dy)
            distance_m = dist(point, projected)
            arc_length_m = offset + fraction * length
            lateral_offset_m = (dx * query_dy - dy * query_dx) / length
            candidates.append((distance_m, index, projected, arc_length_m, lateral_offset_m))

        candidates.sort(key=lambda candidate: (candidate[0], candidate[3], candidate[1]))
        best_distance, best_index, best_point, best_arc, best_lateral = candidates[0]
        tied = [
            candidate
            for candidate in candidates
            if candidate[0] <= best_distance + self.tie_tolerance_m
        ]
        if any(
            dist(candidate[2], best_point) > self.tie_tolerance_m
            or abs(candidate[3] - best_arc) > self.tie_tolerance_m
            for candidate in tied[1:]
        ):
            return RouteProjection(
                status="ambiguous",
                query=point,
                projected_point=None,
                segment_index=None,
                arc_length_m=None,
                distance_m=best_distance,
                lateral_offset_m=None,
            )

        return RouteProjection(
            status="ok",
            query=point,
            projected_point=best_point,
            segment_index=best_index,
            arc_length_m=best_arc,
            distance_m=best_distance,
            lateral_offset_m=best_lateral,
        )


def _validate_tolerance(value: float, name: str) -> None:
    """Require a finite non-negative geometric tolerance."""

    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _finite_point(value: Sequence[float], name: str) -> Vec2D:
    """Normalize one finite two-dimensional point or raise a clear error.

    Returns:
        Vec2D: The normalized point.
    """

    try:
        x, y = value
        point = (float(x), float(y))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain exactly two numeric values.") from None
    if not isfinite(point[0]) or not isfinite(point[1]):
        raise ValueError(f"{name} must contain finite values.")
    return point


@dataclass
class GlobalRoute:
    """Global route from a spawn zone to a goal zone.

    Attributes:
        spawn_id: Identifier of the spawn zone.
        goal_id: Identifier of the goal zone.
        waypoints: Ordered list of 2D waypoints defining the path.
        spawn_zone: Polygon describing the spawn area.
        goal_zone: Polygon describing the goal area.
        source_path_id: Source SVG path id for traceability (empty for synthetic routes).
        source_label: Source SVG route label for traceability (empty for synthetic routes).
    """

    spawn_id: int
    goal_id: int
    waypoints: list[Vec2D]
    spawn_zone: Rect
    goal_zone: Rect
    source_path_id: str = ""
    source_label: str = ""

    def __post_init__(self):
        """Validate spawn/goal identifiers and waypoint list."""

        if self.spawn_id < 0:
            raise ValueError("Spawn id needs to be an integer >= 0!")
        if self.goal_id < 0:
            raise ValueError("Goal id needs to be an integer >= 0!")
        if len(self.waypoints) < 1:
            raise ValueError(f"Route {self.spawn_id} -> {self.goal_id} contains no waypoints!")

    @property
    def sections(self) -> list[tuple[Vec2D, Vec2D]]:
        """List consecutive waypoint pairs.

        Returns:
            list[tuple[Vec2D, Vec2D]]: Start/end points for each segment.
        """

        return (
            []
            if len(self.waypoints) < 2
            else list(zip(self.waypoints[:-1], self.waypoints[1:], strict=False))
        )

    @property
    def section_lengths(self) -> list[float]:
        """Compute length of each segment."""

        return [dist(p1, p2) for p1, p2 in self.sections]

    @property
    def section_offsets(self) -> list[float]:
        """Cumulative offsets for each segment start.

        Returns:
            list[float]: Distance from route start to each segment start.
        """

        lengths = self.section_lengths
        offsets = []
        temp_offset = 0.0
        for section_length in lengths:
            offsets.append(temp_offset)
            temp_offset += section_length
        return offsets

    @property
    def total_length(self) -> float:
        """Total path length."""

        return 0 if len(self.waypoints) < 2 else sum(self.section_lengths)


__all__ = ["GlobalRoute", "RouteGeometry", "RouteProjection", "RouteProjectionStatus"]
