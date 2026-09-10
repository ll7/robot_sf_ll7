"""Route utilities for global navigation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from math import dist, isfinite
from typing import Literal, cast

from robot_sf.common.types import Rect, Vec2D  # noqa: TC001

_DEFAULT_DUPLICATE_TOLERANCE_M = 1e-9
_DEFAULT_TIE_TOLERANCE_M = 1e-9
_DEFAULT_MAX_FORWARD_JUMP_M = 5.0
_DEFAULT_MAX_BACKTRACK_M = 1.0
_DEFAULT_MIN_LOOKAHEAD_M = 0.5
_DEFAULT_MAX_LOOKAHEAD_M = 5.0
_DEFAULT_BASE_LOOKAHEAD_M = 2.0
_DEFAULT_SPEED_LOOKAHEAD_GAIN_S = 1.0
_DEFAULT_CROWD_LOOKAHEAD_PENALTY_M_PER_DENSITY = 0.5
_DEFAULT_UNCERTAINTY_LOOKAHEAD_WEIGHT = 0.5
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
CandidateProgressStatus = Literal[
    "ok",
    "invalid_trace",
    "insufficient_trace",
    "ambiguous",
    "discontinuous",
]
AdaptiveLocalGoalStatus = Literal[
    "ok",
    "invalid_input",
    "invalid_query",
    "invalid_route",
    "ambiguous",
    "discontinuous",
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
class CandidateProgressConfig:
    """Continuity policy for route-relative candidate progress.

    The first candidate point is projected without a prior position. Each
    subsequent point is constrained to this forward/backward arc-length window
    through :class:`RouteProjectionTracker`. ``max_lateral_error_m`` is an
    optional diagnostic acceptance bound; when set, a point farther from its
    nearest route section makes the whole candidate invalid.
    """

    max_forward_jump_m: float = _DEFAULT_MAX_FORWARD_JUMP_M
    max_backtrack_m: float = _DEFAULT_MAX_BACKTRACK_M
    max_lateral_error_m: float | None = None

    def __post_init__(self) -> None:
        """Validate finite, non-negative continuity and lateral-error bounds."""

        _validate_tolerance(self.max_forward_jump_m, "max_forward_jump_m")
        _validate_tolerance(self.max_backtrack_m, "max_backtrack_m")
        if self.max_lateral_error_m is not None:
            _validate_tolerance(self.max_lateral_error_m, "max_lateral_error_m")


@dataclass(frozen=True, slots=True)
class CandidateProgressResult:
    """Route-relative progress and fail-closed projection diagnostics.

    ``signed_progress_m`` is the end arc position minus the start arc position
    and may be negative when a candidate backtracks. ``bounded_progress_m`` is
    clipped to ``[0, route.total_length_m]`` and is only populated for a fully
    valid trace. Any invalid, ambiguous, discontinuous, or off-route sample
    clears the progress fields so a malformed candidate cannot look optimistic.
    """

    status: CandidateProgressStatus
    route_hash: str
    start_arc_length_m: float | None
    end_arc_length_m: float | None
    signed_progress_m: float | None
    bounded_progress_m: float | None
    progress_ratio: float | None
    start_lateral_error_m: float | None
    end_lateral_error_m: float | None
    max_lateral_error_m: float | None
    mean_lateral_error_m: float | None
    start_lateral_offset_m: float | None
    end_lateral_offset_m: float | None
    sample_count: int
    valid_sample_count: int
    ambiguity_count: int
    discontinuity_count: int
    invalid_count: int
    failure_status: str | None

    @property
    def is_valid(self) -> bool:
        """Whether the complete candidate trace has usable route progress."""

        return self.status == "ok"

    def diagnostics(self) -> dict[str, object]:
        """Return a finite, JSON-safe diagnostic mapping for this result."""

        return {
            "schema_version": "route_candidate_progress.v1",
            "status": self.status,
            "route_hash": self.route_hash,
            "start_arc_length_m": self.start_arc_length_m,
            "end_arc_length_m": self.end_arc_length_m,
            "signed_progress_m": self.signed_progress_m,
            "bounded_progress_m": self.bounded_progress_m,
            "progress_ratio": self.progress_ratio,
            "start_lateral_error_m": self.start_lateral_error_m,
            "end_lateral_error_m": self.end_lateral_error_m,
            "max_lateral_error_m": self.max_lateral_error_m,
            "mean_lateral_error_m": self.mean_lateral_error_m,
            "start_lateral_offset_m": self.start_lateral_offset_m,
            "end_lateral_offset_m": self.end_lateral_offset_m,
            "sample_count": self.sample_count,
            "valid_sample_count": self.valid_sample_count,
            "ambiguity_count": self.ambiguity_count,
            "discontinuity_count": self.discontinuity_count,
            "invalid_count": self.invalid_count,
            "failure_status": self.failure_status,
        }

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-safe diagnostic representation of this result."""

        return self.diagnostics()


@dataclass(frozen=True, slots=True)
class AdaptiveLocalGoalConfig:
    """Bounds and gains for deterministic route-arc local-goal selection.

    The unclipped lookahead is
    ``base + speed_gain_s * speed - crowd_penalty * crowd_density -
    uncertainty_weight * uncertainty_m``. Higher speed permits a longer
    preview, while crowding and uncertainty shorten it. The result is clipped
    to the configured bounds and then to the remaining route arc; this is a
    guidance utility only and performs no collision or planner arbitration.
    """

    min_lookahead_m: float = _DEFAULT_MIN_LOOKAHEAD_M
    max_lookahead_m: float = _DEFAULT_MAX_LOOKAHEAD_M
    base_lookahead_m: float = _DEFAULT_BASE_LOOKAHEAD_M
    speed_gain_s: float = _DEFAULT_SPEED_LOOKAHEAD_GAIN_S
    crowd_penalty_m_per_density: float = _DEFAULT_CROWD_LOOKAHEAD_PENALTY_M_PER_DENSITY
    uncertainty_weight: float = _DEFAULT_UNCERTAINTY_LOOKAHEAD_WEIGHT

    def __post_init__(self) -> None:
        """Validate finite, non-negative lookahead bounds and gains."""

        _validate_tolerance(self.min_lookahead_m, "min_lookahead_m")
        _validate_tolerance(self.max_lookahead_m, "max_lookahead_m")
        _validate_tolerance(self.base_lookahead_m, "base_lookahead_m")
        _validate_tolerance(self.speed_gain_s, "speed_gain_s")
        _validate_tolerance(
            self.crowd_penalty_m_per_density,
            "crowd_penalty_m_per_density",
        )
        _validate_tolerance(self.uncertainty_weight, "uncertainty_weight")
        if self.min_lookahead_m > self.max_lookahead_m:
            raise ValueError("min_lookahead_m must be <= max_lookahead_m.")
        if not self.min_lookahead_m <= self.base_lookahead_m <= self.max_lookahead_m:
            raise ValueError("base_lookahead_m must be within the lookahead bounds.")


@dataclass(frozen=True, slots=True)
class AdaptiveLocalGoalResult:
    """Selected route point plus explicit lookahead and projection diagnostics."""

    status: AdaptiveLocalGoalStatus
    route_hash: str
    current_arc_length_m: float | None
    arc_length_m: float | None
    point: Vec2D | None
    lookahead_m: float | None
    effective_lookahead_m: float | None
    speed_m_s: float | None
    crowd_density: float | None
    uncertainty_m: float | None
    lateral_error_m: float | None
    lateral_offset_m: float | None
    projection_status: RouteProjectionStatus | None

    @property
    def is_valid(self) -> bool:
        """Whether a bounded route point was selected."""

        return self.status == "ok" and self.point is not None and self.arc_length_m is not None

    def diagnostics(self) -> dict[str, object]:
        """Return a finite, JSON-safe diagnostic mapping for this result."""

        return {
            "schema_version": "adaptive_local_goal.v1",
            "status": self.status,
            "route_hash": self.route_hash,
            "current_arc_length_m": self.current_arc_length_m,
            "arc_length_m": self.arc_length_m,
            "point": list(self.point) if self.point is not None else None,
            "lookahead_m": self.lookahead_m,
            "effective_lookahead_m": self.effective_lookahead_m,
            "speed_m_s": self.speed_m_s,
            "crowd_density": self.crowd_density,
            "uncertainty_m": self.uncertainty_m,
            "lateral_error_m": self.lateral_error_m,
            "lateral_offset_m": self.lateral_offset_m,
            "projection_status": self.projection_status,
        }

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-safe diagnostic representation of this result."""

        return self.diagnostics()


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

    def candidate_progress(
        self,
        candidate_trace: Sequence[Sequence[float]],
        *,
        config: CandidateProgressConfig | None = None,
    ) -> CandidateProgressResult:
        """Evaluate one ordered candidate trace in the route's arc-length frame.

        Returns:
            CandidateProgressResult: Route-relative progress and diagnostics.
        """

        return compute_candidate_progress(self, candidate_trace, config=config)

    def adaptive_local_goal(
        self,
        current_position: Sequence[float],
        *,
        speed_m_s: float,
        crowd_density: float,
        uncertainty_m: float,
        config: AdaptiveLocalGoalConfig | None = None,
        hint: RouteProjectionHint | None = None,
    ) -> AdaptiveLocalGoalResult:
        """Select a bounded local goal from explicit route and context inputs.

        Returns:
            AdaptiveLocalGoalResult: Selected route point and diagnostics.
        """

        return select_adaptive_local_goal(
            self,
            current_position,
            speed_m_s=speed_m_s,
            crowd_density=crowd_density,
            uncertainty_m=uncertainty_m,
            config=config,
            hint=hint,
        )

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
                candidate for candidate in candidates if lower_bound <= candidate[3] <= upper_bound
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
        if (
            route.route_hash == self._route.route_hash
            and route.duplicate_tolerance_m == self._route.duplicate_tolerance_m
            and route.tie_tolerance_m == self._route.tie_tolerance_m
        ):
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
            "duplicate_tolerance_m": self._route.duplicate_tolerance_m,
            "tie_tolerance_m": self._route.tie_tolerance_m,
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
        _validate_snapshot_route_policy(route, snapshot)

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
        last_step = _optional_nonnegative_index(snapshot.get("last_step"), "last_step")
        failure_count = _finite_nonnegative_index(snapshot.get("failure_count"), "failure_count")
        _validate_snapshot_projection_state(route, previous_s_m, previous_segment_index)
        _validate_snapshot_order_state(previous_s_m, last_step, failure_count)
        last_reset_reason = _validated_reset_reason(snapshot.get("last_reset_reason"))

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


def compute_candidate_progress(
    route: RouteGeometry,
    candidate_trace: Sequence[Sequence[float]],
    *,
    config: CandidateProgressConfig | None = None,
) -> CandidateProgressResult:
    """Compute route-arc progress for one ordered candidate trace.

    Every trace point is projected through :class:`RouteProjectionTracker`, so
    a nearer point on a later parallel branch cannot silently inflate progress.
    A single invalid, ambiguous, discontinuous, or over-lateral sample makes
    the result unusable and clears all progress values.

    Args:
        route: Immutable ordered route geometry.
        candidate_trace: Ordered candidate positions, including its start and
            end points.
        config: Optional continuity and lateral-error policy.

    Returns:
        CandidateProgressResult: Signed and non-negative-bounded route progress
        with finite diagnostics. Configuration errors raise ``ValueError``;
        malformed candidate samples are represented by an invalid result.
    """

    if not isinstance(route, RouteGeometry):
        raise TypeError("route must be a RouteGeometry.")
    resolved_config = _resolve_candidate_progress_config(config)

    try:
        points = tuple(candidate_trace)
    except (TypeError, ValueError, OverflowError):
        return _candidate_progress_failure(
            route,
            status="invalid_trace",
            sample_count=0,
            valid_sample_count=0,
            invalid_count=1,
            failure_status="invalid_trace",
        )

    summary = _project_candidate_trace(route, points, resolved_config)
    sample_count = len(points)
    if summary.failure_status is not None:
        return _candidate_progress_failure(
            route,
            status=_candidate_progress_status(summary.failure_status),
            sample_count=sample_count,
            valid_sample_count=len(summary.projections),
            ambiguity_count=summary.ambiguity_count,
            discontinuity_count=summary.discontinuity_count,
            invalid_count=summary.invalid_count,
            failure_status=summary.failure_status,
        )

    if sample_count < 2:
        return _candidate_progress_failure(
            route,
            status="insufficient_trace",
            sample_count=sample_count,
            valid_sample_count=len(summary.projections),
            failure_status="insufficient_trace",
        )

    return _build_candidate_progress_result(
        route,
        summary,
        sample_count=sample_count,
    )


@dataclass(frozen=True, slots=True)
class _CandidateProjectionSummary:
    """Private aggregate of ordered candidate projections and failure counts."""

    projections: tuple[RouteProjection, ...]
    ambiguity_count: int
    discontinuity_count: int
    invalid_count: int
    failure_status: str | None


def _resolve_candidate_progress_config(
    config: CandidateProgressConfig | None,
) -> CandidateProgressConfig:
    """Resolve and type-check the candidate progress policy.

    Returns:
        CandidateProgressConfig: Validated continuity policy.
    """

    if config is None:
        return CandidateProgressConfig()
    if not isinstance(config, CandidateProgressConfig):
        raise TypeError("config must be a CandidateProgressConfig or None.")
    return config


def _project_candidate_trace(
    route: RouteGeometry,
    points: tuple[object, ...],
    config: CandidateProgressConfig,
) -> _CandidateProjectionSummary:
    """Project every candidate sample with one continuity-aware tracker.

    Returns:
        _CandidateProjectionSummary: Valid projections and explicit failure counts.
    """

    tracker = RouteProjectionTracker(
        route,
        max_forward_jump_m=config.max_forward_jump_m,
        max_backtrack_m=config.max_backtrack_m,
    )
    projections: list[RouteProjection] = []
    ambiguity_count = 0
    discontinuity_count = 0
    invalid_count = 0
    failure_status: str | None = None
    for step, point in enumerate(points):
        projection, failure = _project_candidate_sample(
            tracker,
            point,
            step=step,
            max_lateral_error_m=config.max_lateral_error_m,
        )
        if projection is not None:
            projections.append(projection)
        ambiguity_count += int(failure == "ambiguous")
        discontinuity_count += int(failure == "discontinuous")
        invalid_count += int(failure is not None and failure not in {"ambiguous", "discontinuous"})
        failure_status = failure_status or failure
    return _CandidateProjectionSummary(
        projections=tuple(projections),
        ambiguity_count=ambiguity_count,
        discontinuity_count=discontinuity_count,
        invalid_count=invalid_count,
        failure_status=failure_status,
    )


def _project_candidate_sample(
    tracker: RouteProjectionTracker,
    point: object,
    *,
    step: int,
    max_lateral_error_m: float | None,
) -> tuple[RouteProjection | None, str | None]:
    """Project one sample and classify unusable route-frame outcomes.

    Returns:
        tuple[RouteProjection | None, str | None]: A validated projection and
        optional failure reason.
    """

    tracker_result = tracker.project(cast("Sequence[float]", point), step=step)
    if tracker_result.status in {"ambiguous", "discontinuous"}:
        return None, tracker_result.status
    if not tracker_result.is_valid or tracker_result.projection is None:
        return None, "invalid_query"
    projection = tracker_result.projection
    if not _projection_values_are_finite(projection):
        return None, "invalid_projection"
    assert projection.distance_m is not None
    if max_lateral_error_m is not None and projection.distance_m > max_lateral_error_m:
        return None, "lateral_error"
    return projection, None


def _projection_values_are_finite(projection: RouteProjection) -> bool:
    """Return whether all numeric fields needed for route progress are finite."""

    return (
        projection.arc_length_m is not None
        and projection.distance_m is not None
        and projection.lateral_offset_m is not None
        and isfinite(projection.arc_length_m)
        and isfinite(projection.distance_m)
        and isfinite(projection.lateral_offset_m)
    )


def _candidate_progress_status(failure_status: str) -> CandidateProgressStatus:
    """Map a projection failure reason to the public candidate status.

    Returns:
        CandidateProgressStatus: Public fail-closed status.
    """

    if failure_status == "ambiguous":
        return "ambiguous"
    if failure_status == "discontinuous":
        return "discontinuous"
    return "invalid_trace"


def _build_candidate_progress_result(
    route: RouteGeometry,
    summary: _CandidateProjectionSummary,
    *,
    sample_count: int,
) -> CandidateProgressResult:
    """Build a successful progress record or a fail-closed numeric result.

    Returns:
        CandidateProgressResult: Progress metrics and their diagnostics.
    """

    projections = summary.projections
    valid_sample_count = len(projections)
    total_length_m = route.total_length_m
    if not isfinite(total_length_m) or total_length_m < 0.0:
        return _candidate_progress_failure(
            route,
            status="invalid_trace",
            sample_count=sample_count,
            valid_sample_count=valid_sample_count,
            invalid_count=1,
            failure_status="invalid_route",
        )

    start_projection = projections[0]
    end_projection = projections[-1]
    if start_projection.arc_length_m is None or end_projection.arc_length_m is None:
        return _candidate_progress_failure(
            route,
            status="invalid_trace",
            sample_count=sample_count,
            valid_sample_count=valid_sample_count,
            invalid_count=1,
            failure_status="invalid_projection",
        )

    start_arc_length_m = start_projection.arc_length_m
    end_arc_length_m = end_projection.arc_length_m
    signed_progress_m = end_arc_length_m - start_arc_length_m
    bounded_progress_m = min(max(signed_progress_m, 0.0), total_length_m)
    progress_ratio = bounded_progress_m / total_length_m if total_length_m > 0.0 else 0.0
    lateral_errors = [projection.distance_m for projection in projections]
    lateral_offsets = [projection.lateral_offset_m for projection in projections]
    if (
        not isfinite(signed_progress_m)
        or not isfinite(bounded_progress_m)
        or not isfinite(progress_ratio)
        or any(value is None or not isfinite(value) for value in lateral_errors)
        or any(value is None or not isfinite(value) for value in lateral_offsets)
    ):
        return _candidate_progress_failure(
            route,
            status="invalid_trace",
            sample_count=sample_count,
            valid_sample_count=valid_sample_count,
            invalid_count=1,
            failure_status="nonfinite_result",
        )

    finite_lateral_errors = [float(value) for value in lateral_errors if value is not None]
    finite_lateral_offsets = [float(value) for value in lateral_offsets if value is not None]
    return CandidateProgressResult(
        status="ok",
        route_hash=route.route_hash,
        start_arc_length_m=float(start_arc_length_m),
        end_arc_length_m=float(end_arc_length_m),
        signed_progress_m=float(signed_progress_m),
        bounded_progress_m=float(bounded_progress_m),
        progress_ratio=float(progress_ratio),
        start_lateral_error_m=finite_lateral_errors[0],
        end_lateral_error_m=finite_lateral_errors[-1],
        max_lateral_error_m=float(max(finite_lateral_errors)),
        mean_lateral_error_m=float(sum(finite_lateral_errors) / len(finite_lateral_errors)),
        start_lateral_offset_m=finite_lateral_offsets[0],
        end_lateral_offset_m=finite_lateral_offsets[-1],
        sample_count=sample_count,
        valid_sample_count=valid_sample_count,
        ambiguity_count=summary.ambiguity_count,
        discontinuity_count=summary.discontinuity_count,
        invalid_count=summary.invalid_count,
        failure_status=None,
    )


def select_adaptive_local_goal(
    route: RouteGeometry,
    current_position: Sequence[float],
    *,
    speed_m_s: float,
    crowd_density: float,
    uncertainty_m: float,
    config: AdaptiveLocalGoalConfig | None = None,
    hint: RouteProjectionHint | None = None,
) -> AdaptiveLocalGoalResult:
    """Select a deterministic, bounded point ahead in the route arc frame.

    ``speed_m_s``, ``crowd_density`` (people per square metre), and
    ``uncertainty_m`` are explicit inputs. The validated configuration maps
    these inputs to a bounded lookahead, then the remaining route length caps
    the selected point. Projection failures are returned with no route point;
    this helper does not arbitrate planners or evaluate collisions.

    Raises:
        ValueError: If a context input is not finite and non-negative.
        TypeError: If ``route`` or ``config`` has the wrong type.

    Returns:
        AdaptiveLocalGoalResult: Selected route point and finite diagnostics,
            or an explicit failure status without a route point.
    """

    if not isinstance(route, RouteGeometry):
        raise TypeError("route must be a RouteGeometry.")
    if config is None:
        resolved_config = AdaptiveLocalGoalConfig()
    elif isinstance(config, AdaptiveLocalGoalConfig):
        resolved_config = config
    else:
        raise TypeError("config must be an AdaptiveLocalGoalConfig or None.")

    speed = _finite_nonnegative_input(speed_m_s, "speed_m_s")
    crowd = _finite_nonnegative_input(crowd_density, "crowd_density")
    uncertainty = _finite_nonnegative_input(uncertainty_m, "uncertainty_m")
    total_length_m = route.total_length_m
    if not isfinite(total_length_m) or total_length_m < 0.0:
        return _adaptive_local_goal_failure(
            route,
            status="invalid_route",
            speed_m_s=speed,
            crowd_density=crowd,
            uncertainty_m=uncertainty,
            projection_status=None,
        )

    projection = route.project(current_position, hint=hint)
    if projection.status != "ok":
        status: AdaptiveLocalGoalStatus = projection.status
        return _adaptive_local_goal_failure(
            route,
            status=status,
            speed_m_s=speed,
            crowd_density=crowd,
            uncertainty_m=uncertainty,
            projection_status=projection.status,
        )
    if (
        projection.arc_length_m is None
        or projection.distance_m is None
        or projection.lateral_offset_m is None
        or not isfinite(projection.arc_length_m)
        or not isfinite(projection.distance_m)
        or not isfinite(projection.lateral_offset_m)
    ):
        return _adaptive_local_goal_failure(
            route,
            status="invalid_route",
            speed_m_s=speed,
            crowd_density=crowd,
            uncertainty_m=uncertainty,
            projection_status="invalid_query",
        )

    raw_lookahead_m = (
        resolved_config.base_lookahead_m
        + resolved_config.speed_gain_s * speed
        - resolved_config.crowd_penalty_m_per_density * crowd
        - resolved_config.uncertainty_weight * uncertainty
    )
    if not isfinite(raw_lookahead_m):
        return _adaptive_local_goal_failure(
            route,
            status="invalid_input",
            speed_m_s=speed,
            crowd_density=crowd,
            uncertainty_m=uncertainty,
            projection_status=projection.status,
        )
    lookahead_m = min(
        max(raw_lookahead_m, resolved_config.min_lookahead_m),
        resolved_config.max_lookahead_m,
    )
    current_arc_length_m = projection.arc_length_m
    target_arc_length_m = min(current_arc_length_m + lookahead_m, total_length_m)
    effective_lookahead_m = target_arc_length_m - current_arc_length_m
    if not isfinite(target_arc_length_m) or not isfinite(effective_lookahead_m):
        return _adaptive_local_goal_failure(
            route,
            status="invalid_input",
            speed_m_s=speed,
            crowd_density=crowd,
            uncertainty_m=uncertainty,
            projection_status=projection.status,
        )
    point = route.point_at_arc_length(target_arc_length_m)
    return AdaptiveLocalGoalResult(
        status="ok",
        route_hash=route.route_hash,
        current_arc_length_m=float(current_arc_length_m),
        arc_length_m=float(target_arc_length_m),
        point=point,
        lookahead_m=float(lookahead_m),
        effective_lookahead_m=float(effective_lookahead_m),
        speed_m_s=speed,
        crowd_density=crowd,
        uncertainty_m=uncertainty,
        lateral_error_m=float(projection.distance_m),
        lateral_offset_m=float(projection.lateral_offset_m),
        projection_status=projection.status,
    )


def _candidate_progress_failure(
    route: RouteGeometry,
    *,
    status: CandidateProgressStatus,
    sample_count: int,
    valid_sample_count: int,
    ambiguity_count: int = 0,
    discontinuity_count: int = 0,
    invalid_count: int = 0,
    failure_status: str | None,
) -> CandidateProgressResult:
    """Build a progress result with all optimistic route values cleared.

    Returns:
        CandidateProgressResult: Failure result with route-relative values unset.
    """

    return CandidateProgressResult(
        status=status,
        route_hash=route.route_hash,
        start_arc_length_m=None,
        end_arc_length_m=None,
        signed_progress_m=None,
        bounded_progress_m=None,
        progress_ratio=None,
        start_lateral_error_m=None,
        end_lateral_error_m=None,
        max_lateral_error_m=None,
        mean_lateral_error_m=None,
        start_lateral_offset_m=None,
        end_lateral_offset_m=None,
        sample_count=sample_count,
        valid_sample_count=valid_sample_count,
        ambiguity_count=ambiguity_count,
        discontinuity_count=discontinuity_count,
        invalid_count=invalid_count,
        failure_status=failure_status,
    )


def _adaptive_local_goal_failure(
    route: RouteGeometry,
    *,
    status: AdaptiveLocalGoalStatus,
    speed_m_s: float | None,
    crowd_density: float | None,
    uncertainty_m: float | None,
    projection_status: RouteProjectionStatus | None,
) -> AdaptiveLocalGoalResult:
    """Build an adaptive-goal result without a potentially misleading point.

    Returns:
        AdaptiveLocalGoalResult: Failure result with the route point unset.
    """

    return AdaptiveLocalGoalResult(
        status=status,
        route_hash=route.route_hash,
        current_arc_length_m=None,
        arc_length_m=None,
        point=None,
        lookahead_m=None,
        effective_lookahead_m=None,
        speed_m_s=speed_m_s,
        crowd_density=crowd_density,
        uncertainty_m=uncertainty_m,
        lateral_error_m=None,
        lateral_offset_m=None,
        projection_status=projection_status,
    )


def _finite_nonnegative_input(value: object, name: str) -> float:
    """Return a finite non-negative runtime context value as a float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and non-negative.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite and non-negative.") from None
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return number


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


def _validate_snapshot_route_policy(route: RouteGeometry, snapshot: Mapping[str, object]) -> None:
    """Require snapshot projection tolerances to match the route exactly."""

    snapshot_duplicate_tolerance = _finite_nonnegative_number(
        snapshot.get("duplicate_tolerance_m"), "duplicate_tolerance_m"
    )
    snapshot_tie_tolerance = _finite_nonnegative_number(
        snapshot.get("tie_tolerance_m"), "tie_tolerance_m"
    )
    if snapshot_duplicate_tolerance != route.duplicate_tolerance_m:
        raise ValueError("snapshot duplicate_tolerance_m does not match the route.")
    if snapshot_tie_tolerance != route.tie_tolerance_m:
        raise ValueError("snapshot tie_tolerance_m does not match the route.")


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


def _validate_snapshot_projection_state(
    route: RouteGeometry,
    previous_s_m: float | None,
    previous_segment_index: int | None,
) -> None:
    """Validate the restored arc position and its owning route section."""

    if (previous_s_m is None) != (previous_segment_index is None):
        raise ValueError("snapshot previous projection fields must be provided together.")
    if previous_s_m is None:
        return
    if previous_s_m > route.total_length_m:
        raise ValueError("snapshot previous_s_m exceeds route length.")
    if previous_segment_index is None:
        raise ValueError("snapshot previous projection fields must be provided together.")
    if previous_segment_index >= len(route.sections):
        raise ValueError("snapshot previous_segment_index exceeds route sections.")
    section_start = route.section_offsets[previous_segment_index]
    section_end = section_start + route.section_lengths[previous_segment_index]
    if not section_start <= previous_s_m <= section_end:
        raise ValueError("snapshot previous_s_m does not belong to previous_segment_index.")


def _validate_snapshot_order_state(
    previous_s_m: float | None,
    last_step: int | None,
    failure_count: int,
) -> None:
    """Validate the restored step and failure counters."""

    if last_step is None:
        if failure_count != 0 or previous_s_m is not None:
            raise ValueError("snapshot last_step is required for prior projection or failures.")
    elif previous_s_m is None and failure_count == 0:
        raise ValueError("snapshot previous projection or failure state is required for last_step.")


def _validated_reset_reason(value: object) -> str | None:
    """Validate and return the restored reset reason.

    Returns:
        str | None: The validated reset reason.
    """

    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("snapshot last_reset_reason must be a non-empty string or None.")
    return value if isinstance(value, str) else None


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
    "AdaptiveLocalGoalConfig",
    "AdaptiveLocalGoalResult",
    "AdaptiveLocalGoalStatus",
    "CandidateProgressConfig",
    "CandidateProgressResult",
    "CandidateProgressStatus",
    "GlobalRoute",
    "RouteGeometry",
    "RouteProjection",
    "RouteProjectionHint",
    "RouteProjectionStatus",
    "RouteProjectionTracker",
    "RouteTrackerResult",
    "RouteTrackerStatus",
    "compute_candidate_progress",
    "select_adaptive_local_goal",
]
