"""Route utilities for global navigation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from math import dist, isfinite
from typing import Literal

from robot_sf.common.types import Rect, Vec2D  # noqa: TC001

_DEFAULT_DUPLICATE_TOLERANCE_M = 1e-9
_DEFAULT_TIE_TOLERANCE_M = 1e-9
_DEFAULT_MAX_FORWARD_JUMP_M = 5.0
_DEFAULT_MAX_BACKTRACK_M = 1.0
RouteProjectionStatus = Literal["ok", "ambiguous", "invalid_query", "discontinuous"]
RouteTrackerStatus = Literal[
    "ok",
    "ambiguous",
    "invalid_query",
    "discontinuous",
    "reset",
    "duplicate_step",
    "out_of_order",
]


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
class RouteProjectionHint:
    """Continuity context for one route projection.

    ``previous_s_m`` is deliberately optional so a caller can explicitly request
    an unconstrained first projection.  Once a prior projection exists, the
    forward and backward bounds define the only route arc-length window that a
    hinted projection may enter.
    """

    previous_s_m: float | None = None
    previous_segment_index: int | None = None
    max_forward_jump_m: float = _DEFAULT_MAX_FORWARD_JUMP_M
    max_backtrack_m: float = _DEFAULT_MAX_BACKTRACK_M

    def __post_init__(self) -> None:
        """Validate finite continuity bounds and optional prior position."""

        _validate_tolerance(self.max_forward_jump_m, "max_forward_jump_m")
        _validate_tolerance(self.max_backtrack_m, "max_backtrack_m")
        if self.previous_s_m is not None:
            _validate_tolerance(self.previous_s_m, "previous_s_m")
        if self.previous_segment_index is not None and (
            isinstance(self.previous_segment_index, bool)
            or not isinstance(self.previous_segment_index, int)
            or self.previous_segment_index < 0
        ):
            raise ValueError("previous_segment_index must be a non-negative integer.")
        if self.previous_s_m is None and self.previous_segment_index is not None:
            raise ValueError("previous_segment_index requires previous_s_m.")


@dataclass(frozen=True, slots=True)
class RouteTrackerResult:
    """One explicit route-tracker step outcome."""

    status: RouteTrackerStatus
    projection: RouteProjection | None
    step: int | None
    failure_count: int
    last_reset_reason: str | None

    @property
    def is_valid(self) -> bool:
        """Whether this step produced one valid route branch."""

        return self.status == "ok" and self.projection is not None


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

        try:
            finite_arc_length = isfinite(arc_length_m)
        except (TypeError, OverflowError):
            finite_arc_length = False
        if not finite_arc_length:
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

    def project(
        self, query: Sequence[float], *, hint: RouteProjectionHint | None = None
    ) -> RouteProjection:
        """Project a query point onto the nearest route section.

        The signed lateral offset is positive to the left of the directed
        section.  If multiple distinct route branches are within the tie
        tolerance of the nearest distance, the result is ``ambiguous`` and no
        branch-specific progress is returned.

        Returns:
            RouteProjection: Valid projection data, or an explicit invalid,
                ambiguous, or discontinuous status without route-relative
                progress.
        """

        if hint is not None and not isinstance(hint, RouteProjectionHint):
            raise TypeError("hint must be a RouteProjectionHint or None.")

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
        best_distance = candidates[0][0]
        if hint is not None and hint.previous_s_m is not None:
            lower_bound = hint.previous_s_m - hint.max_backtrack_m
            upper_bound = hint.previous_s_m + hint.max_forward_jump_m
            candidates = [
                candidate
                for candidate in candidates
                if lower_bound - self.tie_tolerance_m
                <= candidate[3]
                <= upper_bound + self.tie_tolerance_m
            ]
            if not candidates:
                return RouteProjection(
                    status="discontinuous",
                    query=point,
                    projected_point=None,
                    segment_index=None,
                    arc_length_m=None,
                    distance_m=best_distance,
                    lateral_offset_m=None,
                )
            if hint.previous_segment_index is not None:
                candidates.sort(
                    key=lambda candidate: (
                        candidate[0],
                        candidate[1] != hint.previous_segment_index,
                        candidate[3],
                        candidate[1],
                    )
                )

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


class RouteProjectionTracker:
    """Explicit continuity state for replayable route projections.

    The tracker owns only route identity, the previous valid projection, step
    ordering, failure count, and reset diagnostics.  Projection failures do
    not overwrite the previous valid position, and all state is instance-local.
    """

    def __init__(
        self,
        route: RouteGeometry,
        *,
        max_forward_jump_m: float = _DEFAULT_MAX_FORWARD_JUMP_M,
        max_backtrack_m: float = _DEFAULT_MAX_BACKTRACK_M,
    ) -> None:
        """Create a tracker for one route and its continuity bounds."""

        if not isinstance(route, RouteGeometry):
            raise TypeError("route must be a RouteGeometry.")
        _validate_tolerance(max_forward_jump_m, "max_forward_jump_m")
        _validate_tolerance(max_backtrack_m, "max_backtrack_m")
        self._route = route
        self._max_forward_jump_m = max_forward_jump_m
        self._max_backtrack_m = max_backtrack_m
        self.reset(reason="initial")

    @property
    def route_hash(self) -> str:
        """Stable identity of the route currently tracked."""

        return self._route.route_hash

    @property
    def previous_s_m(self) -> float | None:
        """Arc length of the last valid projection, if one exists."""

        return self._previous_s_m

    @property
    def previous_segment_index(self) -> int | None:
        """Segment of the last valid projection, if one exists."""

        return self._previous_segment_index

    @property
    def last_step(self) -> int | None:
        """Most recently accepted step-order token."""

        return self._last_step

    @property
    def failure_count(self) -> int:
        """Number of consecutive failed projection steps."""

        return self._failure_count

    @property
    def last_reset_reason(self) -> str | None:
        """Reason for the most recent explicit or route-change reset."""

        return self._last_reset_reason

    def reset(self, *, reason: str = "manual") -> RouteTrackerResult:
        """Clear continuity state and return an explicit reset status.

        Returns:
            RouteTrackerResult: Reset status with no associated projection step.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string.")
        self._previous_s_m = None
        self._previous_segment_index = None
        self._last_step = None
        self._failure_count = 0
        self._last_reset_reason = reason
        return self._result("reset", None)

    def update_route(self, route: RouteGeometry) -> RouteTrackerResult | None:
        """Switch routes and explicitly reset continuity when the hash changes.

        Returns:
            RouteTrackerResult | None: Reset status on a changed route, otherwise
                ``None`` when the route hash is unchanged.
        """

        if not isinstance(route, RouteGeometry):
            raise TypeError("route must be a RouteGeometry.")
        if route.route_hash == self._route.route_hash:
            self._route = route
            return None
        self._route = route
        return self.reset(reason="route_changed")

    def project(self, query: Sequence[float], *, step: int) -> RouteTrackerResult:
        """Project one ordered step while preserving the last valid state.

        Returns:
            RouteTrackerResult: Explicit projection or step-order status.
        """

        _validate_step(step)
        if self._last_step is not None:
            if step == self._last_step:
                return self._result("duplicate_step", step)
            if step < self._last_step:
                return self._result("out_of_order", step)

        hint = (
            None
            if self._previous_s_m is None
            else RouteProjectionHint(
                previous_s_m=self._previous_s_m,
                previous_segment_index=self._previous_segment_index,
                max_forward_jump_m=self._max_forward_jump_m,
                max_backtrack_m=self._max_backtrack_m,
            )
        )
        projection = self._route.project(query, hint=hint)
        self._last_step = step
        if projection.status == "ok":
            self._previous_s_m = projection.arc_length_m
            self._previous_segment_index = projection.segment_index
            self._failure_count = 0
        else:
            self._failure_count += 1
        return self._result(projection.status, step, projection)

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-safe immutable-state snapshot."""

        return {
            "schema_version": "route_projection_tracker.v1",
            "route_hash": self.route_hash,
            "max_forward_jump_m": self._max_forward_jump_m,
            "max_backtrack_m": self._max_backtrack_m,
            "previous_s_m": self._previous_s_m,
            "previous_segment_index": self._previous_segment_index,
            "last_step": self._last_step,
            "failure_count": self._failure_count,
            "last_reset_reason": self._last_reset_reason,
        }

    @classmethod
    def restore(
        cls, route: RouteGeometry, snapshot: Mapping[str, object]
    ) -> RouteProjectionTracker:
        """Restore a tracker after validating route identity and JSON-safe state.

        Returns:
            RouteProjectionTracker: Tracker restored to the captured state.
        """

        if not isinstance(route, RouteGeometry):
            raise TypeError("route must be a RouteGeometry.")
        if not isinstance(snapshot, Mapping):
            raise ValueError("snapshot must be a mapping.")
        if snapshot.get("schema_version") != "route_projection_tracker.v1":
            raise ValueError("snapshot schema_version is unsupported.")
        if snapshot.get("route_hash") != route.route_hash:
            raise ValueError("snapshot route_hash does not match the route.")

        tracker = cls(
            route,
            max_forward_jump_m=_finite_nonnegative_number(
                snapshot.get("max_forward_jump_m"), "max_forward_jump_m"
            ),
            max_backtrack_m=_finite_nonnegative_number(
                snapshot.get("max_backtrack_m"), "max_backtrack_m"
            ),
        )
        previous_s_m = _optional_nonnegative_number(snapshot.get("previous_s_m"), "previous_s_m")
        previous_segment_index = _optional_nonnegative_index(
            snapshot.get("previous_segment_index"), "previous_segment_index"
        )
        if (previous_s_m is None) != (previous_segment_index is None):
            raise ValueError("snapshot previous projection fields must be provided together.")
        if previous_s_m is not None and previous_s_m > route.total_length_m:
            raise ValueError("snapshot previous_s_m exceeds route length.")
        if previous_segment_index is not None and previous_segment_index >= len(route.sections):
            raise ValueError("snapshot previous_segment_index exceeds route sections.")
        last_step = _optional_nonnegative_index(snapshot.get("last_step"), "last_step")
        failure_count = _finite_nonnegative_index(snapshot.get("failure_count"), "failure_count")
        last_reset_reason = snapshot.get("last_reset_reason")
        if last_reset_reason is not None and (
            not isinstance(last_reset_reason, str) or not last_reset_reason.strip()
        ):
            raise ValueError("snapshot last_reset_reason must be a non-empty string or None.")

        tracker._previous_s_m = previous_s_m
        tracker._previous_segment_index = previous_segment_index
        tracker._last_step = last_step
        tracker._failure_count = failure_count
        tracker._last_reset_reason = last_reset_reason
        return tracker

    def _result(
        self,
        status: RouteTrackerStatus,
        step: int | None,
        projection: RouteProjection | None = None,
    ) -> RouteTrackerResult:
        """Build a result from the current explicit tracker state.

        Returns:
            RouteTrackerResult: Result carrying the current failure/reset state.
        """

        return RouteTrackerResult(
            status=status,
            projection=projection,
            step=step,
            failure_count=self._failure_count,
            last_reset_reason=self._last_reset_reason,
        )


def _validate_tolerance(value: float, name: str) -> None:
    """Require a finite non-negative geometric tolerance."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and non-negative.")
    try:
        finite_value = isfinite(value)
    except OverflowError:
        finite_value = False
    if not finite_value or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _validate_step(step: int) -> None:
    """Require a non-negative integer step token."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer.")


def _finite_nonnegative_number(value: object, name: str) -> float:
    """Validate one finite non-negative snapshot number.

    Returns:
        float: The validated number.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"snapshot {name} must be a finite non-negative number.")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"snapshot {name} must be a finite non-negative number.") from None
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"snapshot {name} must be a finite non-negative number.")
    return number


def _optional_nonnegative_number(value: object, name: str) -> float | None:
    """Validate an optional finite non-negative snapshot number.

    Returns:
        float | None: The validated number or ``None``.
    """

    return None if value is None else _finite_nonnegative_number(value, name)


def _optional_nonnegative_index(value: object, name: str) -> int | None:
    """Validate an optional non-negative snapshot integer.

    Returns:
        int | None: The validated index or ``None``.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"snapshot {name} must be a non-negative integer or None.")
    return value


def _finite_nonnegative_index(value: object, name: str) -> int:
    """Validate one required non-negative snapshot integer.

    Returns:
        int: The validated index.
    """

    result = _optional_nonnegative_index(value, name)
    if result is None:
        raise ValueError(f"snapshot {name} must be a non-negative integer.")
    return result


def _finite_point(value: Sequence[float], name: str) -> Vec2D:
    """Normalize one finite two-dimensional point or raise a clear error.

    Returns:
        Vec2D: The normalized point.
    """

    try:
        x, y = value
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must contain exactly two numeric values.") from None
    try:
        point = (float(x), float(y))
    except OverflowError:
        raise ValueError(f"{name} must contain finite values.") from None
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


__all__ = [
    "GlobalRoute",
    "RouteGeometry",
    "RouteProjection",
    "RouteProjectionHint",
    "RouteProjectionStatus",
    "RouteProjectionTracker",
    "RouteTrackerResult",
    "RouteTrackerStatus",
]
