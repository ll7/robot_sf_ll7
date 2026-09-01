# ruff: noqa: C901, DOC201, PLR0912, PLR0915

"""Observation-only public pedestrian goal-candidate generation.

The provider in this module is intentionally a geometry and provenance boundary.  It
consumes public map annotations, public route geometry, and causal tracked state only;
assigned pedestrian routes, true goals, waypoint indices, and future trajectories are
not accepted as inputs.  Candidate coverage is evaluated separately in
``goal_candidate_coverage`` and must not be read as posterior or behavior evidence.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any

from shapely.geometry import LineString, Polygon

from robot_sf.prediction._contract_utils import (
    require_finite,
    require_non_negative,
    require_text,
    stable_digest,
)
from robot_sf.prediction.goal_belief_contract import CoordinateFrame
from robot_sf.prediction.goal_intention import (
    GoalCandidate,
    GoalCandidateAvailability,
    GoalCandidateRole,
    GoalCandidateSet,
)

GOAL_CANDIDATE_PROVIDER_SCHEMA_VERSION = "goal_candidate_provider.v1"
CANDIDATE_ID_QUANTIZATION_M = 1e-3

Point = tuple[float, float]
PolygonPoints = tuple[Point, ...]


class GoalCandidateSource(StrEnum):
    """Public source families supported by the provider contract."""

    MAP_DESTINATION_ZONE = "map_destination_zone"
    PEDESTRIAN_ROUTE_TERMINAL = "pedestrian_route_terminal"
    NAVIGATION_GRAPH_TERMINAL = "navigation_graph_terminal"
    NAVIGATION_GRAPH_BRANCH = "navigation_graph_branch"
    DOOR_OR_EXIT = "door_or_exit"
    CROSSING_ENTRY_EXIT = "crossing_entry_exit"
    CORRIDOR_ENDPOINT = "corridor_endpoint"
    POINT_OF_INTEREST = "point_of_interest"
    FEASIBLE_PATH_WAYPOINT = "feasible_path_waypoint"
    OPEN_RAY = "open_ray"
    UNKNOWN = "unknown"


class CandidateFeasibilityStatus(StrEnum):
    """Explicit source/path feasibility state."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class CandidatePriorMode(StrEnum):
    """Permitted prior policies, all independent of target truth."""

    UNIFORM = "uniform"
    PUBLIC = "public"


class CandidatePathMode(StrEnum):
    """Provenance classification for a candidate's path geometry."""

    PLANNER_PATH = "planner_path"
    STRAIGHT_LINE_FALLBACK = "straight_line_fallback"
    NONE = "none"


_PUBLIC_SOURCES = tuple(
    source for source in GoalCandidateSource if source is not GoalCandidateSource.UNKNOWN
)
_ORACLE_SOURCE_NAMES = frozenset(
    {
        "scenario_assigned_route",
        "assigned_route",
        "true_goal",
        "goal_truth",
        "waypoint_truth",
        "future_trajectory",
        "simulator_goal",
        "simulator_route",
    }
)


def _parse_source(value: GoalCandidateSource | str, field_name: str) -> GoalCandidateSource:
    """Parse a source while rejecting oracle aliases at the contract boundary."""

    if isinstance(value, GoalCandidateSource):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a GoalCandidateSource or string")
    normalized = value.strip().lower()
    if normalized in _ORACLE_SOURCE_NAMES:
        raise ValueError(f"{field_name} requests forbidden oracle source: {value}")
    try:
        return GoalCandidateSource(normalized)
    except ValueError as exc:
        allowed = ", ".join(source.value for source in GoalCandidateSource)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _parse_role(value: GoalCandidateRole | str, field_name: str) -> GoalCandidateRole:
    """Parse a public candidate role."""

    if isinstance(value, GoalCandidateRole):
        return value
    try:
        return GoalCandidateRole(str(value).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(role.value for role in GoalCandidateRole)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _parse_feasibility(
    value: CandidateFeasibilityStatus | str, field_name: str
) -> CandidateFeasibilityStatus:
    """Parse a feasibility status."""

    if isinstance(value, CandidateFeasibilityStatus):
        return value
    try:
        return CandidateFeasibilityStatus(str(value).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(status.value for status in CandidateFeasibilityStatus)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _parse_path_mode(
    value: CandidatePathMode | str | None, field_name: str
) -> CandidatePathMode | None:
    """Parse an optional path provenance mode."""

    if value is None or isinstance(value, CandidatePathMode):
        return value
    try:
        return CandidatePathMode(str(value).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(mode.value for mode in CandidatePathMode)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _parse_coordinate_frame(value: CoordinateFrame | str) -> CoordinateFrame:
    """Parse the only actor-safe coordinate frame."""

    if isinstance(value, CoordinateFrame):
        frame = value
    else:
        try:
            frame = CoordinateFrame(str(value).strip().lower())
        except (TypeError, ValueError) as exc:
            raise ValueError("coordinate_frame must be global_xy") from exc
    if frame is not CoordinateFrame.GLOBAL_XY:
        raise ValueError("coordinate_frame must be global_xy")
    return frame


def _xy(value: Point | Sequence[float], field_name: str) -> Point:
    """Return a finite two-dimensional point."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    return (
        require_finite(value[0], f"{field_name}[0]"),
        require_finite(value[1], f"{field_name}[1]"),
    )


def _unit(value: Point | Sequence[float], field_name: str) -> Point:
    """Return a finite unit direction."""

    x, y = _xy(value, field_name)
    norm = math.hypot(x, y)
    if norm <= 0.0:
        raise ValueError(f"{field_name} must be non-zero")
    return (x / norm, y / norm)


def _quantized(value: float) -> int:
    """Quantize geometry for stable IDs at a documented one-millimetre scale."""

    return round(value / CANDIDATE_ID_QUANTIZATION_M)


def _quantized_point(point: Point | None) -> tuple[int, int] | None:
    """Return an integer quantized point for canonical identity."""

    return None if point is None else (_quantized(point[0]), _quantized(point[1]))


def _point_json(point: Point | None) -> list[float] | None:
    """Serialize an optional point without introducing tuples into JSON receipts."""

    return None if point is None else [point[0], point[1]]


def _distance(left: Point, right: Point) -> float:
    """Return Euclidean point distance."""

    return math.hypot(left[0] - right[0], left[1] - right[1])


def _angle_between(left: Point, right: Point) -> float:
    """Return the unsigned angle between two unit directions."""

    dot = max(-1.0, min(1.0, left[0] * right[0] + left[1] * right[1]))
    return math.acos(dot)


@dataclass(frozen=True, slots=True)
class GoalCandidateProviderConfig:
    """Strict immutable configuration for bounded candidate generation."""

    enabled_sources: tuple[GoalCandidateSource, ...] = _PUBLIC_SOURCES
    active_waypoint_cap: int = 16
    final_destination_cap: int = 32
    open_ray_cap: int = 8
    max_source_records: int = 4096
    homotopy_count: int = 2
    waypoint_lookahead_m: float = 2.0
    deduplication_tolerance_m: float = 0.05
    path_clearance_m: float = 0.0
    feasibility_policy: str = "reject_infeasible"
    prior_mode: CandidatePriorMode | str = CandidatePriorMode.UNIFORM
    open_ray_count: int = 4
    open_ray_angular_support_rad: float = math.pi / 8.0
    always_emit_open_rays: bool = False
    unknown_enabled: bool = True
    unknown_prior_floor: float = 0.1
    allow_straight_line_fallback: bool = True
    cache_policy: str = "map_static"
    map_semantic_requirements: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize every setting participating in provenance."""

        sources = tuple(_parse_source(value, "enabled_sources[]") for value in self.enabled_sources)
        if len(sources) != len(set(sources)):
            raise ValueError("enabled_sources must be unique")
        if GoalCandidateSource.UNKNOWN in sources:
            raise ValueError("unknown is emitted by unknown_enabled, not enabled_sources")
        object.__setattr__(self, "enabled_sources", sources)

        for field_name in (
            "active_waypoint_cap",
            "final_destination_cap",
            "open_ray_cap",
            "max_source_records",
            "homotopy_count",
            "open_ray_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in (
            "waypoint_lookahead_m",
            "deduplication_tolerance_m",
            "path_clearance_m",
            "open_ray_angular_support_rad",
            "unknown_prior_floor",
        ):
            value = require_non_negative(getattr(self, field_name), field_name)
            if field_name == "open_ray_angular_support_rad" and value > math.pi:
                raise ValueError("open_ray_angular_support_rad must be at most pi")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "always_emit_open_rays",
            "unknown_enabled",
            "allow_straight_line_fallback",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")
        try:
            prior_mode = CandidatePriorMode(self.prior_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("prior_mode must be uniform or public") from exc
        object.__setattr__(self, "prior_mode", prior_mode)
        if self.feasibility_policy not in {"reject_infeasible", "retain_unavailable"}:
            raise ValueError("feasibility_policy must be reject_infeasible or retain_unavailable")
        if self.cache_policy not in {"map_static", "off"}:
            raise ValueError("cache_policy must be map_static or off")
        requirements = tuple(
            require_text(value, "map_semantic_requirements[]")
            for value in self.map_semantic_requirements
        )
        if len(requirements) != len(set(requirements)):
            raise ValueError("map_semantic_requirements must be unique")
        object.__setattr__(self, "map_semantic_requirements", tuple(sorted(requirements)))

    def to_dict(self) -> dict[str, object]:
        """Return the full JSON-safe configuration provenance."""

        return {
            "schema_version": GOAL_CANDIDATE_PROVIDER_SCHEMA_VERSION,
            "enabled_sources": [source.value for source in self.enabled_sources],
            "active_waypoint_cap": self.active_waypoint_cap,
            "final_destination_cap": self.final_destination_cap,
            "open_ray_cap": self.open_ray_cap,
            "max_source_records": self.max_source_records,
            "homotopy_count": self.homotopy_count,
            "waypoint_lookahead_m": self.waypoint_lookahead_m,
            "deduplication_tolerance_m": self.deduplication_tolerance_m,
            "path_clearance_m": self.path_clearance_m,
            "feasibility_policy": self.feasibility_policy,
            "prior_mode": self.prior_mode.value,
            "open_ray_count": self.open_ray_count,
            "open_ray_angular_support_rad": self.open_ray_angular_support_rad,
            "always_emit_open_rays": self.always_emit_open_rays,
            "unknown_enabled": self.unknown_enabled,
            "unknown_prior_floor": self.unknown_prior_floor,
            "allow_straight_line_fallback": self.allow_straight_line_fallback,
            "cache_policy": self.cache_policy,
            "map_semantic_requirements": list(self.map_semantic_requirements),
            "candidate_id_quantization_m": CANDIDATE_ID_QUANTIZATION_M,
        }

    @property
    def config_hash(self) -> str:
        """Return the stable SHA-256 configuration digest."""

        return stable_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class PublicGoalCandidateRecord:
    """One public source annotation narrowed to actor-safe geometry."""

    source: GoalCandidateSource | str
    source_id: str
    role: GoalCandidateRole | str = GoalCandidateRole.FINAL_DESTINATION
    position: Point | None = None
    direction: Point | None = None
    angular_support_rad: float | None = None
    route_signature: str | None = None
    parent_destination_id: str | None = None
    path_points: tuple[Point, ...] = ()
    path_tangent: Point | None = None
    path_mode: CandidatePathMode | str | None = None
    feasibility_status: CandidateFeasibilityStatus | str = CandidateFeasibilityStatus.FEASIBLE
    feasibility_reason: str | None = None
    provenance_refs: tuple[str, ...] = ()
    prior_weight: float | None = None
    path_cost_m: float | None = None
    coordinate_frame: CoordinateFrame | str = CoordinateFrame.GLOBAL_XY

    def __post_init__(self) -> None:
        """Fail closed on non-public, non-finite, or semantically incomplete records."""

        object.__setattr__(self, "source", _parse_source(self.source, "record.source"))
        object.__setattr__(self, "source_id", require_text(self.source_id, "record.source_id"))
        role = _parse_role(self.role, "record.role")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "coordinate_frame", _parse_coordinate_frame(self.coordinate_frame))
        if self.position is not None:
            object.__setattr__(self, "position", _xy(self.position, "record.position"))
        if self.direction is not None:
            object.__setattr__(self, "direction", _unit(self.direction, "record.direction"))
        if self.path_tangent is not None:
            object.__setattr__(
                self, "path_tangent", _unit(self.path_tangent, "record.path_tangent")
            )
        if self.angular_support_rad is not None:
            support = require_finite(self.angular_support_rad, "record.angular_support_rad")
            if not 0.0 <= support <= math.pi:
                raise ValueError("record.angular_support_rad must be between 0 and pi")
            object.__setattr__(self, "angular_support_rad", support)
        if self.route_signature is not None:
            object.__setattr__(
                self,
                "route_signature",
                require_text(self.route_signature, "record.route_signature"),
            )
        if self.parent_destination_id is not None:
            object.__setattr__(
                self,
                "parent_destination_id",
                require_text(self.parent_destination_id, "record.parent_destination_id"),
            )
        points = tuple(_xy(point, "record.path_points[]") for point in self.path_points)
        object.__setattr__(self, "path_points", points)
        object.__setattr__(self, "path_mode", _parse_path_mode(self.path_mode, "record.path_mode"))
        status = _parse_feasibility(self.feasibility_status, "record.feasibility_status")
        object.__setattr__(self, "feasibility_status", status)
        if self.feasibility_reason is not None:
            object.__setattr__(
                self,
                "feasibility_reason",
                require_text(self.feasibility_reason, "record.feasibility_reason"),
            )
        refs = tuple(require_text(ref, "record.provenance_refs[]") for ref in self.provenance_refs)
        if len(refs) != len(set(refs)):
            raise ValueError("record.provenance_refs must be unique")
        object.__setattr__(self, "provenance_refs", tuple(sorted(refs)))
        if self.prior_weight is not None:
            object.__setattr__(
                self, "prior_weight", require_non_negative(self.prior_weight, "record.prior_weight")
            )
        if self.path_cost_m is not None:
            object.__setattr__(
                self, "path_cost_m", require_non_negative(self.path_cost_m, "record.path_cost_m")
            )

        if role is GoalCandidateRole.OPEN_RAY:
            if self.position is not None:
                raise ValueError("open-ray records must not contain a Cartesian position")
            if self.direction is None:
                raise ValueError("open-ray records require direction")
        elif (
            role not in {GoalCandidateRole.UNKNOWN}
            and status is CandidateFeasibilityStatus.FEASIBLE
        ):
            if self.position is None:
                raise ValueError("feasible point records require position")

    def to_dict(self) -> dict[str, object]:
        """Return deterministic public record bytes for map digests and receipts."""

        return {
            "source": self.source.value,
            "source_id": self.source_id,
            "role": self.role.value,
            "position": _point_json(self.position),
            "direction": _point_json(self.direction),
            "angular_support_rad": self.angular_support_rad,
            "route_signature": self.route_signature,
            "parent_destination_id": self.parent_destination_id,
            "path_points": [[x, y] for x, y in self.path_points],
            "path_tangent": _point_json(self.path_tangent),
            "path_mode": self.path_mode.value if self.path_mode is not None else None,
            "feasibility_status": self.feasibility_status.value,
            "feasibility_reason": self.feasibility_reason,
            "provenance_refs": list(self.provenance_refs),
            "prior_weight": self.prior_weight,
            "path_cost_m": self.path_cost_m,
            "coordinate_frame": self.coordinate_frame.value,
        }


@dataclass(frozen=True, slots=True)
class PublicGoalMapInputs:
    """A narrowed public map projection accepted by the actor provider."""

    records: tuple[PublicGoalCandidateRecord, ...] = ()
    obstacles: tuple[PolygonPoints, ...] = ()
    forbidden_zones: tuple[PolygonPoints, ...] = ()
    unavailable_sources: tuple[GoalCandidateSource | str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the map projection without retaining a simulator object."""

        records = tuple(self.records)
        if any(type(record) is not PublicGoalCandidateRecord for record in records):
            raise TypeError("records must contain PublicGoalCandidateRecord values")
        if len(records) > 4096:
            raise ValueError("records exceed the public map projection bound of 4096")
        object.__setattr__(self, "records", records)
        polygons = tuple(_polygon(polygon, "obstacles[]") for polygon in self.obstacles)
        object.__setattr__(self, "obstacles", polygons)
        forbidden = tuple(
            _polygon(polygon, "forbidden_zones[]") for polygon in self.forbidden_zones
        )
        object.__setattr__(self, "forbidden_zones", forbidden)
        unavailable = tuple(
            _parse_source(source, "unavailable_sources[]") for source in self.unavailable_sources
        )
        if len(unavailable) != len(set(unavailable)):
            raise ValueError("unavailable_sources must be unique")
        object.__setattr__(self, "unavailable_sources", unavailable)

    def to_dict(self) -> dict[str, object]:
        """Return an order-independent public map digest payload."""

        return {
            "records": [record.to_dict() for record in sorted(self.records, key=_record_sort_key)],
            "obstacles": [
                [[x, y] for x, y in polygon] for polygon in sorted(self.obstacles, key=tuple)
            ],
            "forbidden_zones": [
                [[x, y] for x, y in polygon] for polygon in sorted(self.forbidden_zones, key=tuple)
            ],
            "unavailable_sources": [
                source.value
                for source in sorted(self.unavailable_sources, key=lambda item: item.value)
            ],
        }

    @property
    def map_digest(self) -> str:
        """Return the digest of public map inputs only."""

        return stable_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class GoalCandidateSourceStatus:
    """Per-source availability and rejection accounting."""

    source: GoalCandidateSource
    status: str
    record_count: int = 0
    rejected_count: int = 0
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a compact source status receipt."""

        return {
            "source": self.source.value,
            "status": self.status,
            "record_count": self.record_count,
            "rejected_count": self.rejected_count,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RejectedGoalCandidate:
    """An explicit reason why a public record did not enter the candidate set."""

    source: GoalCandidateSource
    source_id: str
    role: GoalCandidateRole
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe rejection record."""

        return {
            "source": self.source.value,
            "source_id": self.source_id,
            "role": self.role.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GoalCandidateGenerationResult:
    """Bounded candidate set plus provenance and diagnostic accounting."""

    candidate_set: GoalCandidateSet
    candidate_set_digest: str
    config_hash: str
    map_digest: str
    source_statuses: tuple[GoalCandidateSourceStatus, ...]
    rejected_records: tuple[RejectedGoalCandidate, ...]
    runtime_ms: float
    cache_key: str
    cache_hit: bool = False
    schema_version: str = GOAL_CANDIDATE_PROVIDER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return the compact JSON receipt used by smoke and research workflows."""

        return {
            "schema_version": self.schema_version,
            "candidate_set": self.candidate_set.to_dict(),
            "candidate_set_digest": self.candidate_set_digest,
            "config_hash": self.config_hash,
            "map_digest": self.map_digest,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "runtime_ms": self.runtime_ms,
            "candidate_count": len(self.candidate_set.candidates),
            "source_statuses": [status.to_dict() for status in self.source_statuses],
            "rejected_records": [record.to_dict() for record in self.rejected_records],
            "claim_boundary": "candidate_generation_only",
        }


@dataclass(frozen=True, slots=True)
class _PreparedCandidate:
    """Internal normalized candidate before stable ID assignment."""

    role: GoalCandidateRole
    position: Point | None
    source: GoalCandidateSource
    direction: Point | None = None
    angular_support_rad: float | None = None
    route_signature: str | None = None
    parent_destination_id: str | None = None
    path_tangent: Point | None = None
    path_mode: CandidatePathMode | None = None
    feasibility_status: CandidateFeasibilityStatus = CandidateFeasibilityStatus.FEASIBLE
    availability: GoalCandidateAvailability = GoalCandidateAvailability.AVAILABLE
    provenance_refs: tuple[str, ...] = ()
    prior_weight: float | None = None
    path_cost_m: float | None = None
    source_id: str = ""


def _polygon(value: Sequence[Sequence[float]], field_name: str) -> PolygonPoints:
    """Validate a polygon ring used for static path rejection."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) < 3:
        raise ValueError(f"{field_name} must contain at least three points")
    points = tuple(_xy(point, f"{field_name}[]") for point in value)
    geometry = Polygon(points)
    if geometry.is_empty or not geometry.is_valid or geometry.area <= 0.0:
        raise ValueError(f"{field_name} must describe a valid non-empty polygon")
    return points


def _record_sort_key(record: PublicGoalCandidateRecord) -> tuple[object, ...]:
    """Return an order-independent source-record key."""

    return (
        record.source.value,
        record.source_id,
        record.role.value,
        record.route_signature or "",
        _quantized_point(record.position) or (),
        _quantized_point(record.direction) or (),
        tuple(_quantized_point(point) for point in record.path_points),
    )


def _candidate_identity(candidate: _PreparedCandidate) -> dict[str, object]:
    """Build stable identity fields, excluding priors and provenance order."""

    return {
        "role": candidate.role.value,
        "position_q": _quantized_point(candidate.position),
        "direction_q": _quantized_point(candidate.direction),
        "route_signature": candidate.route_signature,
        "parent_destination_id": candidate.parent_destination_id,
    }


def _candidate_id(candidate: _PreparedCandidate) -> str:
    """Return a stable candidate ID independent of input iteration order."""

    return f"goal-candidate-{stable_digest(_candidate_identity(candidate))[:20]}"


def _is_final_role(role: GoalCandidateRole) -> bool:
    """Return whether a role represents a final destination hypothesis."""

    return role in {
        GoalCandidateRole.FINAL_DESTINATION,
        GoalCandidateRole.ROUTE_ENDPOINT,
        GoalCandidateRole.BOTH,
    }


def _can_merge(left: _PreparedCandidate, right: _PreparedCandidate, tolerance: float) -> bool:
    """Merge geometric duplicates while retaining route/topology distinctions."""

    if left.role is not right.role:
        return False
    if (
        left.route_signature is not None
        and right.route_signature is not None
        and left.route_signature != right.route_signature
    ):
        return False
    if (
        left.parent_destination_id is not None
        and right.parent_destination_id is not None
        and left.parent_destination_id != right.parent_destination_id
    ):
        return False
    if left.position is None or right.position is None:
        if left.position is not right.position:
            return False
    elif _distance(left.position, right.position) > tolerance:
        return False
    if left.direction is not None and right.direction is not None:
        if _angle_between(left.direction, right.direction) > 1e-6:
            return False
    return True


def _prepared_sort_key(candidate: _PreparedCandidate) -> tuple[object, ...]:
    """Return deterministic ordering before and after deduplication."""

    return (
        candidate.role.value,
        candidate.route_signature or "",
        candidate.parent_destination_id or "",
        _quantized_point(candidate.position) or (),
        _quantized_point(candidate.direction) or (),
        candidate.source.value,
        candidate.source_id,
    )


def _merge_prepared(
    values: Sequence[_PreparedCandidate], tolerance: float
) -> tuple[_PreparedCandidate, ...]:
    """Merge candidates in a permutation-invariant sorted pass."""

    merged: list[_PreparedCandidate] = []
    for value in sorted(values, key=_prepared_sort_key):
        for index, existing in enumerate(merged):
            if _can_merge(existing, value, tolerance):
                merged[index] = _merge_pair(existing, value)
                break
        else:
            merged.append(value)
    return tuple(sorted(merged, key=_prepared_sort_key))


def _merge_pair(left: _PreparedCandidate, right: _PreparedCandidate) -> _PreparedCandidate:
    """Combine provenance and public prior metadata for one geometric candidate."""

    path_modes = [mode for mode in (left.path_mode, right.path_mode) if mode is not None]
    path_mode = (
        min(
            path_modes,
            key=lambda mode: {
                CandidatePathMode.PLANNER_PATH: 0,
                CandidatePathMode.STRAIGHT_LINE_FALLBACK: 1,
                CandidatePathMode.NONE: 2,
            }[mode],
        )
        if path_modes
        else None
    )
    tangents = [
        tangent for tangent in (left.path_tangent, right.path_tangent) if tangent is not None
    ]
    direction = left.direction if left.direction is not None else right.direction
    route_signature = left.route_signature or right.route_signature
    parent_destination_id = left.parent_destination_id or right.parent_destination_id
    prior_values = [value for value in (left.prior_weight, right.prior_weight) if value is not None]
    path_costs = [value for value in (left.path_cost_m, right.path_cost_m) if value is not None]
    refs = tuple(sorted(set(left.provenance_refs).union(right.provenance_refs)))
    return _PreparedCandidate(
        role=left.role,
        position=left.position if left.position is not None else right.position,
        source=min(left.source, right.source, key=lambda source: source.value),
        direction=direction,
        angular_support_rad=max(
            (
                value
                for value in (left.angular_support_rad, right.angular_support_rad)
                if value is not None
            ),
            default=None,
        ),
        route_signature=route_signature,
        parent_destination_id=parent_destination_id,
        path_tangent=tangents[0] if tangents else None,
        path_mode=path_mode,
        feasibility_status=left.feasibility_status,
        availability=min(left.availability, right.availability, key=lambda value: value.value),
        provenance_refs=refs,
        prior_weight=max(prior_values, default=None),
        path_cost_m=min(path_costs, default=None),
        source_id=min(left.source_id, right.source_id),
    )


def _path_length(path: Sequence[Point]) -> float:
    """Return polyline length."""

    return sum(_distance(left, right) for left, right in pairwise(path))


def _path_tangent(path: Sequence[Point], lookahead_m: float) -> Point | None:
    """Return the initial unit tangent of a feasible path.

    ``lookahead_m`` remains part of the helper signature because the active waypoint
    and tangent are emitted together by the provider.  The tangent itself is the
    path's initial direction, which is the geometry needed to compare immediate
    desired-force evidence with a detour around an obstacle.
    """

    del lookahead_m

    if len(path) < 2:
        return None
    for start, end in pairwise(path):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0.0:
            continue
        return (dx / length, dy / length)
    return None


def _interpolate_path(path: Sequence[Point], distance_m: float) -> Point | None:
    """Interpolate a path point at a bounded look-ahead distance."""

    if not path:
        return None
    remaining = max(0.0, distance_m)
    for start, end in pairwise(path):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0.0:
            continue
        if remaining <= length:
            fraction = remaining / length
            return (start[0] + fraction * dx, start[1] + fraction * dy)
        remaining -= length
    return path[-1]


def _path_intersects_obstacle(
    path: Sequence[Point], obstacles: Sequence[PolygonPoints], clearance_m: float
) -> bool:
    """Reject any path segment that enters or touches a declared static obstacle."""

    if len(path) < 2:
        return False
    line = LineString(path)
    return any(line.intersects(Polygon(obstacle).buffer(clearance_m)) for obstacle in obstacles)


def _record_provenance(record: PublicGoalCandidateRecord) -> tuple[str, ...]:
    """Add stable source identity to caller-provided provenance."""

    return tuple(
        sorted(
            set(record.provenance_refs).union(
                {f"source:{record.source.value}", f"source_id:{record.source_id}"}
            )
        )
    )


def _prepared_from_record(
    record: PublicGoalCandidateRecord,
    *,
    config: GoalCandidateProviderConfig,
    observed_position: Point | None,
    obstacles: Sequence[PolygonPoints],
) -> tuple[_PreparedCandidate | None, _PreparedCandidate | None, str | None]:
    """Validate one record and optionally derive its active-waypoint candidate."""

    if record.feasibility_status is CandidateFeasibilityStatus.INFEASIBLE:
        return None, None, record.feasibility_reason or "source_marked_infeasible"
    if record.feasibility_status in {
        CandidateFeasibilityStatus.UNAVAILABLE,
        CandidateFeasibilityStatus.UNKNOWN,
    }:
        if config.feasibility_policy != "retain_unavailable":
            return None, None, record.feasibility_reason or record.feasibility_status.value
        availability = (
            GoalCandidateAvailability.UNAVAILABLE
            if record.feasibility_status is CandidateFeasibilityStatus.UNAVAILABLE
            else GoalCandidateAvailability.UNKNOWN
        )
        return (
            _PreparedCandidate(
                role=record.role,
                position=record.position,
                source=record.source,
                direction=record.direction,
                angular_support_rad=record.angular_support_rad,
                route_signature=record.route_signature,
                parent_destination_id=record.parent_destination_id,
                feasibility_status=record.feasibility_status,
                availability=availability,
                provenance_refs=_record_provenance(record),
                prior_weight=record.prior_weight,
                source_id=record.source_id,
            ),
            None,
            None,
        )

    path = record.path_points
    path_mode = record.path_mode
    position = record.position
    if (
        position is not None
        and not path
        and observed_position is not None
        and config.allow_straight_line_fallback
    ):
        path = (observed_position, position)
        path_mode = CandidatePathMode.STRAIGHT_LINE_FALLBACK
    elif (
        path
        and position is not None
        and _distance(path[-1], position) > config.deduplication_tolerance_m
    ):
        path = (*path, position)
    if path and _path_intersects_obstacle(path, obstacles, config.path_clearance_m):
        return None, None, record.feasibility_reason or "path_intersects_declared_obstacle"

    tangent = record.path_tangent or _path_tangent(path, config.waypoint_lookahead_m)
    if path and path_mode is None:
        path_mode = CandidatePathMode.PLANNER_PATH
    if path_mode is None and position is not None:
        path_mode = CandidatePathMode.NONE
    path_cost = (
        record.path_cost_m
        if record.path_cost_m is not None
        else (_path_length(path) if path else None)
    )
    final = _PreparedCandidate(
        role=record.role,
        position=position,
        source=record.source,
        direction=record.direction,
        angular_support_rad=record.angular_support_rad,
        route_signature=record.route_signature,
        parent_destination_id=record.parent_destination_id,
        path_tangent=tangent,
        path_mode=path_mode,
        feasibility_status=record.feasibility_status,
        availability=GoalCandidateAvailability.AVAILABLE,
        provenance_refs=_record_provenance(record),
        prior_weight=record.prior_weight,
        path_cost_m=path_cost,
        source_id=record.source_id,
    )
    active: _PreparedCandidate | None = None
    if _is_final_role(record.role) and len(path) >= 2:
        active_position = _interpolate_path(path, config.waypoint_lookahead_m)
        final_id = _candidate_id(final)
        if active_position is not None and tangent is not None:
            active = _PreparedCandidate(
                role=GoalCandidateRole.ACTIVE_WAYPOINT,
                position=active_position,
                source=GoalCandidateSource.FEASIBLE_PATH_WAYPOINT,
                route_signature=record.route_signature,
                parent_destination_id=final_id,
                path_tangent=tangent,
                path_mode=path_mode,
                feasibility_status=record.feasibility_status,
                availability=GoalCandidateAvailability.AVAILABLE,
                provenance_refs=tuple(
                    sorted(set(_record_provenance(record)).union({f"derived_from:{final_id}"}))
                ),
                prior_weight=record.prior_weight,
                path_cost_m=path_cost,
                source_id=f"{record.source_id}:lookahead:{config.waypoint_lookahead_m:g}",
            )
    return final, active, None


def _rank_key(candidate: _PreparedCandidate) -> tuple[object, ...]:
    """Rank only on public prior, path cost, and stable identity."""

    return (
        -(candidate.prior_weight if candidate.prior_weight is not None else 1.0),
        candidate.path_cost_m if candidate.path_cost_m is not None else math.inf,
        _candidate_id(candidate),
    )


def _public_prior(
    record: PublicGoalCandidateRecord, config: GoalCandidateProviderConfig
) -> float | None:
    """Resolve a public prior without inspecting target state."""

    if config.prior_mode is CandidatePriorMode.UNIFORM:
        return 1.0
    return record.prior_weight if record.prior_weight is not None else 1.0


def _with_public_prior(
    candidate: _PreparedCandidate, config: GoalCandidateProviderConfig
) -> _PreparedCandidate:
    """Normalize internal priority policy for derived and source candidates."""

    if candidate.prior_weight is not None:
        prior = candidate.prior_weight if config.prior_mode is CandidatePriorMode.PUBLIC else 1.0
    else:
        prior = 1.0
    return _PreparedCandidate(
        role=candidate.role,
        position=candidate.position,
        source=candidate.source,
        direction=candidate.direction,
        angular_support_rad=candidate.angular_support_rad,
        route_signature=candidate.route_signature,
        parent_destination_id=candidate.parent_destination_id,
        path_tangent=candidate.path_tangent,
        path_mode=candidate.path_mode,
        feasibility_status=candidate.feasibility_status,
        availability=candidate.availability,
        provenance_refs=candidate.provenance_refs,
        prior_weight=prior,
        path_cost_m=candidate.path_cost_m,
        source_id=candidate.source_id,
    )


def _to_goal_candidate(candidate: _PreparedCandidate, config_hash: str) -> GoalCandidate:
    """Convert normalized internal data to the shared #8068 candidate schema."""

    return GoalCandidate(
        id=_candidate_id(candidate),
        position=candidate.position,
        source=candidate.source.value,
        role=candidate.role,
        route_signature=candidate.route_signature,
        availability=candidate.availability,
        prior_weight=candidate.prior_weight,
        coordinate_frame=CoordinateFrame.GLOBAL_XY,
        direction=candidate.direction,
        angular_support_rad=candidate.angular_support_rad,
        parent_destination_id=candidate.parent_destination_id,
        path_tangent=candidate.path_tangent,
        path_mode=candidate.path_mode.value if candidate.path_mode is not None else None,
        feasibility_status=candidate.feasibility_status.value,
        provenance_refs=candidate.provenance_refs,
        config_hash=config_hash,
    )


def _generated_open_rays(
    directions: Sequence[Point], config: GoalCandidateProviderConfig
) -> tuple[PublicGoalCandidateRecord, ...]:
    """Build direction-only records when the public map has no finite endpoint."""

    values = tuple(directions)
    if not values:
        values = tuple(
            (
                math.cos(2.0 * math.pi * index / config.open_ray_count),
                math.sin(2.0 * math.pi * index / config.open_ray_count),
            )
            for index in range(config.open_ray_count)
        )
    return tuple(
        PublicGoalCandidateRecord(
            source=GoalCandidateSource.OPEN_RAY,
            source_id=f"generated:{index}",
            role=GoalCandidateRole.OPEN_RAY,
            direction=direction,
            angular_support_rad=config.open_ray_angular_support_rad,
            path_mode=CandidatePathMode.NONE,
            provenance_refs=("open_ray:generated",),
        )
        for index, direction in enumerate(values)
    )


def _source_statuses(
    config: GoalCandidateProviderConfig,
    records: Sequence[PublicGoalCandidateRecord],
    rejected: Sequence[RejectedGoalCandidate],
    unavailable_sources: Sequence[GoalCandidateSource],
    *,
    emitted_open_rays: bool,
    emitted_unknown: bool,
    derived_active_count: int,
) -> tuple[GoalCandidateSourceStatus, ...]:
    """Build deterministic source availability accounting."""

    statuses: list[GoalCandidateSourceStatus] = []
    for source in (*config.enabled_sources, GoalCandidateSource.UNKNOWN):
        if source is GoalCandidateSource.UNKNOWN:
            statuses.append(
                GoalCandidateSourceStatus(
                    source=source,
                    status="available" if emitted_unknown else "unavailable",
                    reason=None if emitted_unknown else "disabled_by_config",
                )
            )
            continue
        source_records = [record for record in records if record.source is source]
        source_rejected = [record for record in rejected if record.source is source]
        if source in unavailable_sources:
            status, reason = "unavailable", "canonical_public_source_not_available"
        elif source is GoalCandidateSource.OPEN_RAY and emitted_open_rays:
            status, reason = "available", None
        elif source is GoalCandidateSource.FEASIBLE_PATH_WAYPOINT and derived_active_count:
            status, reason = "available", None
        elif source_records and len(source_rejected) < len(source_records):
            status, reason = "available", None
        elif source_rejected:
            status, reason = "unavailable", "all_records_rejected"
        else:
            status, reason = "unavailable", "no_public_records"
        statuses.append(
            GoalCandidateSourceStatus(
                source=source,
                status=status,
                record_count=len(source_records)
                + (
                    derived_active_count
                    if source is GoalCandidateSource.FEASIBLE_PATH_WAYPOINT
                    else 0
                ),
                rejected_count=len(source_rejected),
                reason=reason,
            )
        )
    return tuple(statuses)


def generate_goal_candidates(
    records: Sequence[PublicGoalCandidateRecord],
    *,
    config: GoalCandidateProviderConfig | None = None,
    observed_position_global: Point | Sequence[float] | None = None,
    obstacles: Sequence[PolygonPoints] = (),
    forbidden_zones: Sequence[PolygonPoints] = (),
    open_ray_directions: Sequence[Point] = (),
    unavailable_sources: Sequence[GoalCandidateSource | str] = (),
    _static_map: PublicGoalMapInputs | None = None,
) -> GoalCandidateGenerationResult:
    """Generate a deterministic observation-only candidate set.

    Args:
        records: Public map, route, topology, or semantic source records.
        config: Immutable provider settings.
        observed_position_global: Current causal tracked position used only for
            path tangent/fallback construction.
        obstacles: Public static obstacle polygons used to reject crossing paths.
        forbidden_zones: Public static forbidden polygons treated like obstacles.
        open_ray_directions: Optional public direction annotations; no endpoint is
            created for these records.
        unavailable_sources: Canonical sources known to be absent from this map.

    Returns:
        A bounded candidate set with separate source/rejection diagnostics.
    """

    started = time.perf_counter()
    cfg = config or GoalCandidateProviderConfig()
    if type(cfg) is not GoalCandidateProviderConfig:
        raise TypeError("config must be a GoalCandidateProviderConfig")
    values = tuple(records)
    if any(type(record) is not PublicGoalCandidateRecord for record in values):
        raise TypeError("records must contain PublicGoalCandidateRecord values")
    if len(values) > cfg.max_source_records:
        raise ValueError("records exceed config.max_source_records")
    if _static_map is None:
        static_map = PublicGoalMapInputs(
            records=values,
            obstacles=tuple(obstacles),
            forbidden_zones=tuple(forbidden_zones),
            unavailable_sources=tuple(unavailable_sources),
        )
    else:
        if type(_static_map) is not PublicGoalMapInputs:
            raise TypeError("_static_map must be a PublicGoalMapInputs")
        if _static_map.records != values:
            raise ValueError("_static_map.records must match records")
        static_map = _static_map
    normalized_open_ray_directions = tuple(
        _unit(direction, "open_ray_directions[]") for direction in open_ray_directions
    )
    observed = (
        None
        if observed_position_global is None
        else _xy(observed_position_global, "observed_position_global")
    )
    unavailable = static_map.unavailable_sources
    map_digest = stable_digest(
        {
            "public_map": static_map.to_dict(),
            "open_ray_directions": [
                _point_json(direction) for direction in normalized_open_ray_directions
            ],
        }
    )
    cache_key = stable_digest({"map_digest": map_digest, "config_hash": cfg.config_hash})
    enabled = set(cfg.enabled_sources)
    source_records = tuple(sorted(values, key=_record_sort_key))
    rejected: list[RejectedGoalCandidate] = []
    prepared_finals: list[_PreparedCandidate] = []
    prepared_actives: list[_PreparedCandidate] = []
    explicit_open_rays: list[PublicGoalCandidateRecord] = []

    for record in source_records:
        if record.source is GoalCandidateSource.UNKNOWN:
            continue
        if record.source not in enabled:
            rejected.append(
                RejectedGoalCandidate(
                    record.source, record.source_id, record.role, "source_disabled"
                )
            )
            continue
        if record.source in unavailable:
            rejected.append(
                RejectedGoalCandidate(
                    record.source, record.source_id, record.role, "source_unavailable"
                )
            )
            continue
        if record.source is GoalCandidateSource.OPEN_RAY:
            explicit_open_rays.append(record)
            continue
        final, active, reason = _prepared_from_record(
            record,
            config=cfg,
            observed_position=observed,
            obstacles=(*static_map.obstacles, *static_map.forbidden_zones),
        )
        if reason is not None:
            rejected.append(
                RejectedGoalCandidate(record.source, record.source_id, record.role, reason)
            )
            continue
        if final is not None:
            prepared = _with_public_prior(final, cfg)
            if _is_final_role(final.role):
                prepared_finals.append(prepared)
            else:
                prepared_actives.append(prepared)
        if active is not None and GoalCandidateSource.FEASIBLE_PATH_WAYPOINT in enabled:
            prepared_actives.append(_with_public_prior(active, cfg))

    merged_finals = list(_merge_prepared(prepared_finals, cfg.deduplication_tolerance_m))
    destination_groups: dict[
        tuple[tuple[int, int] | None, str | None], list[_PreparedCandidate]
    ] = {}
    for candidate in merged_finals:
        destination_groups.setdefault(
            (_quantized_point(candidate.position), candidate.parent_destination_id), []
        ).append(candidate)
    retained_finals: list[_PreparedCandidate] = []
    for group in sorted(
        destination_groups.values(), key=lambda values: _prepared_sort_key(values[0])
    ):
        ranked = sorted(group, key=_rank_key)
        retained_finals.extend(ranked[: cfg.homotopy_count])
        for candidate in ranked[cfg.homotopy_count :]:
            rejected.append(
                RejectedGoalCandidate(
                    candidate.source, candidate.source_id, candidate.role, "homotopy_cap"
                )
            )
    retained_finals = sorted(retained_finals, key=_rank_key)[: cfg.final_destination_cap]
    merged_actives = list(_merge_prepared(prepared_actives, cfg.deduplication_tolerance_m))
    retained_final_ids = {_candidate_id(final) for final in retained_finals}
    merged_actives = [
        active
        for active in merged_actives
        if active.parent_destination_id is None
        or active.parent_destination_id in retained_final_ids
    ]
    retained_actives = sorted(merged_actives, key=_rank_key)[: cfg.active_waypoint_cap]

    open_records = list(explicit_open_rays)
    finite_available = bool(retained_finals or retained_actives)
    should_generate_rays = bool(GoalCandidateSource.OPEN_RAY in enabled) and (
        cfg.always_emit_open_rays or not finite_available
    )
    if should_generate_rays and not open_records:
        open_records.extend(_generated_open_rays(normalized_open_ray_directions, cfg))
    prepared_open = []
    for record in open_records:
        final, _active, reason = _prepared_from_record(
            record,
            config=cfg,
            observed_position=observed,
            obstacles=(*static_map.obstacles, *static_map.forbidden_zones),
        )
        if reason is not None:
            rejected.append(
                RejectedGoalCandidate(record.source, record.source_id, record.role, reason)
            )
        elif final is not None:
            prepared_open.append(_with_public_prior(final, cfg))
    retained_open = sorted(
        _merge_prepared(prepared_open, cfg.deduplication_tolerance_m), key=_rank_key
    )[: cfg.open_ray_cap]

    output_candidates: list[GoalCandidate] = []
    output_candidates.extend(
        _to_goal_candidate(candidate, cfg.config_hash) for candidate in retained_finals
    )
    output_candidates.extend(
        _to_goal_candidate(candidate, cfg.config_hash) for candidate in retained_actives
    )
    output_candidates.extend(
        _to_goal_candidate(candidate, cfg.config_hash) for candidate in retained_open
    )
    if cfg.unknown_enabled:
        unknown = _PreparedCandidate(
            role=GoalCandidateRole.UNKNOWN,
            position=None,
            source=GoalCandidateSource.UNKNOWN,
            path_mode=CandidatePathMode.NONE,
            feasibility_status=CandidateFeasibilityStatus.UNKNOWN,
            availability=GoalCandidateAvailability.AVAILABLE,
            provenance_refs=("source:unknown", "unknown:unconditional"),
            prior_weight=cfg.unknown_prior_floor,
            source_id="unknown",
        )
        output_candidates.append(_to_goal_candidate(unknown, cfg.config_hash))
    output_candidates.sort(key=lambda candidate: (candidate.role.value, candidate.id))
    candidate_set = GoalCandidateSet(
        candidates=tuple(output_candidates),
        source="goal_candidate_provider",
        availability=GoalCandidateAvailability.AVAILABLE
        if output_candidates
        else GoalCandidateAvailability.UNAVAILABLE,
    )
    candidate_set_digest = stable_digest(candidate_set.to_dict())
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    statuses = _source_statuses(
        cfg,
        source_records,
        rejected,
        unavailable,
        emitted_open_rays=bool(retained_open),
        emitted_unknown=cfg.unknown_enabled,
        derived_active_count=len(retained_actives),
    )
    return GoalCandidateGenerationResult(
        candidate_set=candidate_set,
        candidate_set_digest=candidate_set_digest,
        config_hash=cfg.config_hash,
        map_digest=map_digest,
        source_statuses=statuses,
        rejected_records=tuple(
            sorted(rejected, key=lambda item: (item.source.value, item.source_id, item.reason))
        ),
        runtime_ms=elapsed_ms,
        cache_key=cache_key,
    )


def public_goal_map_inputs_from_definition(map_definition: Any) -> PublicGoalMapInputs:
    """Project canonical public map fields without reading pedestrian assignments.

    The adapter intentionally reads only ``ped_goal_zones``, ``ped_routes``,
    ``poi_positions``, ``poi_labels``, and static ``obstacles``.  It never accesses
    ``single_pedestrians``, assigned routes, true goals, waypoint indices, or runtime
    actor state.  Maps without reliable semantic fields remain explicitly unavailable
    through the provider's source status report.
    """

    records: list[PublicGoalCandidateRecord] = []
    for zone_index, zone in enumerate(getattr(map_definition, "ped_goal_zones", ())):
        corners = tuple(_xy(point, f"ped_goal_zones[{zone_index}][]") for point in zone)
        if len(corners) != 3:
            raise ValueError("ped_goal_zones entries must contain three corners")
        fourth = (
            corners[0][0] + corners[2][0] - corners[1][0],
            corners[0][1] + corners[2][1] - corners[1][1],
        )
        center = (
            sum(point[0] for point in (*corners, fourth)) / 4.0,
            sum(point[1] for point in (*corners, fourth)) / 4.0,
        )
        center_id = stable_digest({"center": [center[0], center[1]]})[:20]
        records.append(
            PublicGoalCandidateRecord(
                source=GoalCandidateSource.MAP_DESTINATION_ZONE,
                source_id=f"ped_goal_zone:{center_id}",
                position=center,
                provenance_refs=("map:ped_goal_zones", f"zone_center:{center_id}"),
            )
        )

    routes = tuple(getattr(map_definition, "ped_routes", ()))
    for route_index, route in enumerate(routes):
        waypoints = tuple(
            _xy(point, f"ped_routes[{route_index}].waypoints[]")
            for point in getattr(route, "waypoints", ())
        )
        if not waypoints:
            continue
        spawn_id = getattr(route, "spawn_id", None)
        goal_id = getattr(route, "goal_id", None)
        route_name = (
            getattr(route, "source_path_id", "")
            or getattr(route, "source_label", "")
            or (
                f"{spawn_id}:{goal_id}"
                if spawn_id is not None and goal_id is not None
                else "anonymous"
            )
        )
        route_identity = {
            "route": [[x, y] for x, y in waypoints],
            "name": str(route_name),
            "spawn_id": spawn_id,
            "goal_id": goal_id,
        }
        route_signature = f"ped-route:{stable_digest(route_identity)[:20]}"
        source_id = f"ped_route:{stable_digest(route_identity)[:20]}"
        records.append(
            PublicGoalCandidateRecord(
                source=GoalCandidateSource.PEDESTRIAN_ROUTE_TERMINAL,
                source_id=source_id,
                position=waypoints[-1],
                route_signature=route_signature,
                path_points=waypoints,
                path_mode=CandidatePathMode.PLANNER_PATH,
                provenance_refs=("map:ped_routes", f"route_signature:{route_signature}"),
            )
        )

    poi_positions = tuple(getattr(map_definition, "poi_positions", ()))
    poi_labels = getattr(map_definition, "poi_labels", {})
    if not isinstance(poi_labels, Mapping):
        raise TypeError("map_definition.poi_labels must be a mapping")
    if poi_labels and len(poi_positions) != len(poi_labels):
        raise ValueError("map_definition POI positions and labels must have equal lengths")
    if poi_labels:
        poi_ids = list(poi_labels)
        poi_entries = ((str(poi_id), poi_positions[index]) for index, poi_id in enumerate(poi_ids))
    else:
        poi_entries = (
            (stable_digest({"position": list(position)})[:20], position)
            for position in poi_positions
        )
    for poi_id, position in sorted(poi_entries, key=lambda item: item[0]):
        records.append(
            PublicGoalCandidateRecord(
                source=GoalCandidateSource.POINT_OF_INTEREST,
                source_id=f"poi:{poi_id}",
                position=_xy(position, f"poi_positions[{poi_id}]"),
                provenance_refs=("map:poi_positions", f"poi_id:{poi_id}"),
            )
        )

    polygons: list[PolygonPoints] = []
    for obstacle_index, obstacle in enumerate(getattr(map_definition, "obstacles", ())):
        iter_polygons = getattr(obstacle, "iter_polygons", None)
        if callable(iter_polygons):
            geometries = iter_polygons()
            for geometry in geometries:
                coordinates = tuple(geometry.exterior.coords)[:-1]
                polygons.append(_polygon(coordinates, f"obstacles[{obstacle_index}]"))
            continue
        vertices = getattr(obstacle, "vertices", ())
        if vertices:
            polygons.append(_polygon(vertices, f"obstacles[{obstacle_index}]"))

    return PublicGoalMapInputs(records=tuple(records), obstacles=tuple(polygons))


def generate_goal_candidates_from_map(
    public_map: PublicGoalMapInputs,
    *,
    config: GoalCandidateProviderConfig | None = None,
    observed_position_global: Point | Sequence[float] | None = None,
    open_ray_directions: Sequence[Point] = (),
) -> GoalCandidateGenerationResult:
    """Generate candidates from a narrowed public map projection."""

    if type(public_map) is not PublicGoalMapInputs:
        raise TypeError("public_map must be a PublicGoalMapInputs")
    return generate_goal_candidates(
        public_map.records,
        config=config,
        observed_position_global=observed_position_global,
        obstacles=public_map.obstacles,
        forbidden_zones=public_map.forbidden_zones,
        open_ray_directions=open_ray_directions,
        unavailable_sources=public_map.unavailable_sources,
    )


class GoalCandidateProvider:
    """State-light provider with a bounded map-static projection cache."""

    __slots__ = ("_last_cache_key", "_last_static_map", "config")

    def __init__(self, config: GoalCandidateProviderConfig | None = None) -> None:
        """Create a provider with immutable configuration and no actor state."""

        self.config = config or GoalCandidateProviderConfig()
        if type(self.config) is not GoalCandidateProviderConfig:
            raise TypeError("config must be a GoalCandidateProviderConfig")
        self._last_cache_key: str | None = None
        self._last_static_map: PublicGoalMapInputs | None = None

    def generate(
        self,
        records: Sequence[PublicGoalCandidateRecord],
        *,
        observed_position_global: Point | Sequence[float] | None = None,
        obstacles: Sequence[PolygonPoints] = (),
        forbidden_zones: Sequence[PolygonPoints] = (),
        open_ray_directions: Sequence[Point] = (),
        unavailable_sources: Sequence[GoalCandidateSource | str] = (),
    ) -> GoalCandidateGenerationResult:
        """Generate candidates and report whether the map-static cache key was reused."""

        static_map = PublicGoalMapInputs(
            records=tuple(records),
            obstacles=tuple(obstacles),
            forbidden_zones=tuple(forbidden_zones),
            unavailable_sources=tuple(unavailable_sources),
        )
        normalized_open_ray_directions = tuple(
            _unit(direction, "open_ray_directions[]") for direction in open_ray_directions
        )
        cache_key = stable_digest(
            {
                "map_digest": stable_digest(
                    {
                        "public_map": static_map.to_dict(),
                        "open_ray_directions": [
                            _point_json(direction) for direction in normalized_open_ray_directions
                        ],
                    }
                ),
                "config_hash": self.config.config_hash,
            }
        )
        cache_hit = (
            self.config.cache_policy == "map_static"
            and cache_key == self._last_cache_key
            and self._last_static_map is not None
        )
        cached_static_map = self._last_static_map if cache_hit else None
        if cached_static_map is not None:
            static_map = cached_static_map
        result = generate_goal_candidates(
            static_map.records,
            config=self.config,
            observed_position_global=observed_position_global,
            obstacles=static_map.obstacles,
            forbidden_zones=static_map.forbidden_zones,
            open_ray_directions=normalized_open_ray_directions,
            unavailable_sources=static_map.unavailable_sources,
            _static_map=static_map,
        )
        if self.config.cache_policy == "map_static":
            self._last_cache_key = cache_key
            self._last_static_map = static_map
        return GoalCandidateGenerationResult(
            candidate_set=result.candidate_set,
            candidate_set_digest=result.candidate_set_digest,
            config_hash=result.config_hash,
            map_digest=result.map_digest,
            source_statuses=result.source_statuses,
            rejected_records=result.rejected_records,
            runtime_ms=result.runtime_ms,
            cache_key=result.cache_key,
            cache_hit=cache_hit,
        )


__all__ = [
    "CANDIDATE_ID_QUANTIZATION_M",
    "GOAL_CANDIDATE_PROVIDER_SCHEMA_VERSION",
    "CandidateFeasibilityStatus",
    "CandidatePathMode",
    "CandidatePriorMode",
    "GoalCandidateGenerationResult",
    "GoalCandidateProvider",
    "GoalCandidateProviderConfig",
    "GoalCandidateSource",
    "GoalCandidateSourceStatus",
    "PublicGoalCandidateRecord",
    "PublicGoalMapInputs",
    "RejectedGoalCandidate",
    "generate_goal_candidates",
    "generate_goal_candidates_from_map",
    "public_goal_map_inputs_from_definition",
]
