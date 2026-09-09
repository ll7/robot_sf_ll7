"""Typed pedestrian-response observations for biased-route encounters.

This module provides an additive, deterministic observation contract for one
encounter in a structured indoor fixture. It composes the existing
``RouteSideReport`` rather than reimplementing route-side classification.

The contract is diagnostic-only. It does not change planner behavior, metric
semantics, campaigns, preregistration, social-compliance scalars, or any
paper-facing claim. In particular, ``response_present=False`` is an observed
absence of a response; ``None`` is unavailable and is listed explicitly in
``missing_fields`` or ``unavailable_fields``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from robot_sf.benchmark.passing_clearance import DISTANCE_BASIS_SURFACE_CLEARANCE
from robot_sf.benchmark.route_choice_observability import (
    DIAGNOSTIC_SCHEMA_VERSION,
    ROUTE_SIDES,
    RouteSideReport,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

PEDESTRIAN_RESPONSE_SCHEMA_VERSION = "pedestrian_response_observation.v1"
PEDESTRIAN_RESPONSE_CLAIM_BOUNDARY = (
    "diagnostic-only structured-indoor encounter observation; not AMV evidence, "
    "a planner metric, a social-compliance scalar, or a human-predictability claim"
)

ResponseStatus = Literal["available", "not_available"]

_REQUIRED_FIELDS = (
    "minimum_passing_clearance_m",
    "offered_side",
    "route_reference",
    "taken_side",
    "response_present",
)
_REQUIRED_FIELD_SET = frozenset(_REQUIRED_FIELDS)
_RESPONSE_STATUSES = frozenset({"available", "not_available"})


@dataclass(frozen=True)
class RouteReference:
    """Reference metadata carried by a route-side observation."""

    coordinate_frame: str
    start: tuple[float, float]
    goal: tuple[float, float]
    units: str
    tolerance_m: float
    neutral_band_m: float
    progress_interval: tuple[float, float]

    def __post_init__(self) -> None:
        """Normalize and validate the declared route reference."""
        for field_name in ("coordinate_frame", "units"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())
        object.__setattr__(self, "start", _normalize_reference_point(self.start, "start"))
        object.__setattr__(self, "goal", _normalize_reference_point(self.goal, "goal"))
        object.__setattr__(
            self,
            "tolerance_m",
            _normalize_reference_scalar(self.tolerance_m, "tolerance_m"),
        )
        object.__setattr__(
            self,
            "neutral_band_m",
            _normalize_reference_scalar(self.neutral_band_m, "neutral_band_m"),
        )
        object.__setattr__(
            self,
            "progress_interval",
            _normalize_reference_interval(self.progress_interval),
        )

    @classmethod
    def from_report(cls, report: RouteSideReport) -> RouteReference:
        """Extract the canonical reference metadata from a route-side report.

        Returns:
            The normalized reference metadata declared by ``report``.
        """
        return cls(
            coordinate_frame=report.coordinate_frame,
            start=report.start,
            goal=report.goal,
            units=report.units,
            tolerance_m=report.tolerance_m,
            neutral_band_m=report.neutral_band_m,
            progress_interval=report.progress_interval,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-ready reference metadata without duplicating the side label."""
        return {
            "coordinate_frame": self.coordinate_frame,
            "start": list(self.start),
            "goal": list(self.goal),
            "units": self.units,
            "tolerance_m": self.tolerance_m,
            "neutral_band_m": self.neutral_band_m,
            "progress_interval": list(self.progress_interval),
        }


@dataclass(frozen=True)
class PedestrianResponseObservation:
    """One typed, per-encounter pedestrian-response observation.

    ``offered_side`` is the side classified for the biased/offered route and
    ``taken_side`` is the side classified for the observed taken route. Their
    values use :data:`ROUTE_SIDES`; an unavailable route report is represented
    by ``"unavailable"`` and listed in ``unavailable_fields``. A missing route
    report is represented by ``None`` and listed in ``missing_fields``.

    ``minimum_passing_clearance_m`` is a caller-supplied observed minimum in
    metres. This record does not introduce a threshold or redefine any
    existing benchmark metric.
    """

    encounter_id: str
    minimum_passing_clearance_m: float | None = None
    offered_side: str | None = None
    taken_side: str | None = None
    response_present: bool | None = None
    route_reference: RouteReference | None = None
    status: ResponseStatus | None = None
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    unavailable_fields: tuple[str, ...] = field(default_factory=tuple)
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        """Normalize and validate the fail-closed observation state."""
        _normalize_encounter_id(self)
        missing = set(_field_names(self.missing_fields, "missing_fields"))
        unavailable = set(_field_names(self.unavailable_fields, "unavailable_fields"))
        if overlap := sorted(missing & unavailable):
            raise ValueError(f"fields cannot be both missing and unavailable: {overlap}")
        _validate_sides(self, unavailable)
        _validate_route_reference(self)
        _normalize_clearance(self)
        _validate_response_flag(self)
        _complete_field_state(self, missing, unavailable)
        normalized_missing = tuple(sorted(missing))
        normalized_unavailable = tuple(sorted(unavailable))
        object.__setattr__(self, "missing_fields", normalized_missing)
        object.__setattr__(self, "unavailable_fields", normalized_unavailable)
        _set_status(self, normalized_missing, normalized_unavailable)
        _set_unavailable_reason(self, normalized_missing, normalized_unavailable)

    @property
    def biased_side(self) -> str | None:
        """Return the canonical ``offered_side`` under biased-route wording."""
        return self.offered_side

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready, versioned diagnostic record."""
        return {
            "schema_version": PEDESTRIAN_RESPONSE_SCHEMA_VERSION,
            "encounter_id": self.encounter_id,
            "status": self.status,
            "evidence_tier": "analysis-only",
            "result_classification": "diagnostic-only",
            "claim_boundary": PEDESTRIAN_RESPONSE_CLAIM_BOUNDARY,
            "minimum_passing_clearance_m": self.minimum_passing_clearance_m,
            "offered_side": self.offered_side,
            "taken_side": self.taken_side,
            "response_present": self.response_present,
            "route_reference": (
                self.route_reference.as_dict() if self.route_reference is not None else None
            ),
            "missing_fields": list(self.missing_fields),
            "unavailable_fields": list(self.unavailable_fields),
            "unavailable_reason": self.unavailable_reason,
            "distance_basis": DISTANCE_BASIS_SURFACE_CLEARANCE,
            "units": {"distance": "m"},
            "route_side_observability_schema": DIAGNOSTIC_SCHEMA_VERSION,
        }


def build_pedestrian_response_observation(
    *,
    encounter_id: str,
    offered_route: RouteSideReport | None = None,
    taken_route: RouteSideReport | None = None,
    minimum_passing_clearance_m: float | None = None,
    response_present: bool | None = None,
    unavailable_fields: Iterable[str] = (),
) -> PedestrianResponseObservation:
    """Build one observation from existing route-side reports and encounter data.

    ``offered_route`` and ``taken_route`` must be reports from the existing
    route-choice observability contract. A ``None`` input is missing; a report
    whose side is ``"unavailable"`` is unavailable. Invalid scalar encounter
    values are retained as unavailable rather than converted into a plausible
    value.

    Returns:
        A typed observation with explicit missing or unavailable fields.
    """
    missing: set[str] = set()
    unavailable = set(_field_names(unavailable_fields, "unavailable_fields"))
    reasons: list[str] = []

    (
        offered_side,
        taken_side,
        offered_reference,
        taken_reference,
    ) = _extract_route_inputs(
        offered_route,
        taken_route,
        missing=missing,
        unavailable=unavailable,
        reasons=reasons,
    )
    route_reference, offered_side, taken_side = _resolve_route_reference(
        offered_reference,
        taken_reference,
        offered_side=offered_side,
        taken_side=taken_side,
        unavailable=unavailable,
        reasons=reasons,
    )
    normalized_clearance = _normalize_builder_clearance(
        minimum_passing_clearance_m,
        missing=missing,
        unavailable=unavailable,
        reasons=reasons,
    )
    normalized_response = _normalize_builder_response(
        response_present,
        missing=missing,
        unavailable=unavailable,
        reasons=reasons,
    )

    reason = ";".join(sorted(set(reasons))) or None
    return PedestrianResponseObservation(
        encounter_id=encounter_id,
        minimum_passing_clearance_m=normalized_clearance,
        offered_side=offered_side,
        taken_side=taken_side,
        response_present=normalized_response,
        route_reference=route_reference,
        missing_fields=tuple(sorted(missing)),
        unavailable_fields=tuple(sorted(unavailable)),
        unavailable_reason=reason,
    )


def _extract_route_inputs(
    offered_route: RouteSideReport | None,
    taken_route: RouteSideReport | None,
    *,
    missing: set[str],
    unavailable: set[str],
    reasons: list[str],
) -> tuple[
    str | None,
    str | None,
    RouteReference | None,
    RouteReference | None,
]:
    """Extract both route sides and their provenance metadata.

    Returns:
        Offered side, taken side, and each report's route reference.
    """
    offered_side, offered_reference, offered_reason = _extract_route_side(
        offered_route, field_name="offered_side", missing=missing, unavailable=unavailable
    )
    if offered_reason is not None:
        reasons.append(offered_reason)
    taken_side, taken_reference, taken_reason = _extract_route_side(
        taken_route, field_name="taken_side", missing=missing, unavailable=unavailable
    )
    if taken_reason is not None:
        reasons.append(taken_reason)
    return offered_side, taken_side, offered_reference, taken_reference


def _resolve_route_reference(
    offered_reference: RouteReference | None,
    taken_reference: RouteReference | None,
    *,
    offered_side: str | None,
    taken_side: str | None,
    unavailable: set[str],
    reasons: list[str],
) -> tuple[RouteReference | None, str | None, str | None]:
    """Retain shared route provenance or fail closed on a mismatch.

    Returns:
        Shared route reference and possibly updated route-side values.
    """
    if "route_reference" in unavailable:
        reasons.append("route_reference:explicitly_unavailable")
        return None, offered_side, taken_side

    route_references = [
        reference for reference in (offered_reference, taken_reference) if reference is not None
    ]
    if not route_references:
        return None, offered_side, taken_side
    route_reference = route_references[0]
    if all(reference == route_reference for reference in route_references[1:]):
        return route_reference, offered_side, taken_side

    unavailable.update({"offered_side", "route_reference", "taken_side"})
    reasons.append("route_reference:mismatch")
    return None, "unavailable", "unavailable"


def _normalize_builder_clearance(
    clearance: float | None,
    *,
    missing: set[str],
    unavailable: set[str],
    reasons: list[str],
) -> float | None:
    """Normalize builder clearance while preserving invalid-value provenance.

    Returns:
        A finite non-negative clearance, or ``None`` when unavailable/missing.
    """
    field_name = "minimum_passing_clearance_m"
    if field_name in unavailable:
        return None
    if clearance is None:
        missing.add(field_name)
        return None
    try:
        if isinstance(clearance, bool):
            raise TypeError("boolean clearance is not a distance")
        normalized = float(clearance)
    except (OverflowError, TypeError, ValueError):
        normalized = None
    if normalized is None or not math.isfinite(normalized) or normalized < 0.0:
        unavailable.add(field_name)
        reasons.append(f"{field_name}:invalid_value")
        return None
    return normalized


def _normalize_builder_response(
    response_present: bool | None,
    *,
    missing: set[str],
    unavailable: set[str],
    reasons: list[str],
) -> bool | None:
    """Normalize the builder response flag while preserving its state.

    Returns:
        The observed boolean, or ``None`` when the value is missing/unavailable.
    """
    field_name = "response_present"
    if field_name in unavailable:
        return None
    if response_present is None:
        missing.add(field_name)
        return None
    if type(response_present) is bool:
        return response_present
    unavailable.add(field_name)
    reasons.append(f"{field_name}:invalid_value")
    return None


def _extract_route_side(
    report: RouteSideReport | None,
    *,
    field_name: str,
    missing: set[str],
    unavailable: set[str],
) -> tuple[str | None, RouteReference | None, str | None]:
    """Extract one route side while retaining route-contract availability.

    Returns:
        The side value, its reference metadata, and an optional field-level
        unavailability reason.
    """
    if report is None:
        if field_name not in unavailable:
            missing.add(field_name)
        return None, None, None
    if not isinstance(report, RouteSideReport):
        raise TypeError(f"{field_name} must be a RouteSideReport or None")
    if report.side not in ROUTE_SIDES:
        raise ValueError(f"{field_name} report uses an unknown route-side value")
    reference = RouteReference.from_report(report)
    if report.side == "unavailable":
        unavailable.add(field_name)
        return "unavailable", reference, f"{field_name}:{report.reason or 'unknown'}"
    if field_name in unavailable:
        return None, reference, f"{field_name}:explicitly_unavailable"
    return report.side, reference, None


def _field_names(value: Iterable[str], field_name: str) -> tuple[str, ...]:
    """Validate and normalize explicit field-name lists.

    Returns:
        Sorted, duplicate-free required field names.
    """
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be an iterable of field names") from exc
    if any(not isinstance(item, str) or item not in _REQUIRED_FIELD_SET for item in values):
        raise ValueError(f"{field_name} contains an unknown required field")
    return tuple(sorted(set(values)))


def _normalize_encounter_id(observation: PedestrianResponseObservation) -> None:
    """Validate and normalize the stable encounter identity."""
    if not isinstance(observation.encounter_id, str) or not observation.encounter_id.strip():
        raise ValueError("encounter_id must be a non-empty string")
    object.__setattr__(observation, "encounter_id", observation.encounter_id.strip())


def _validate_sides(observation: PedestrianResponseObservation, unavailable: set[str]) -> None:
    """Validate route-side values and retain explicit unavailable sides."""
    for field_name in ("offered_side", "taken_side"):
        value = getattr(observation, field_name)
        if value is not None and value not in ROUTE_SIDES:
            raise ValueError(f"{field_name} must use the route-side vocabulary")
        if value == "unavailable":
            unavailable.add(field_name)


def _validate_route_reference(observation: PedestrianResponseObservation) -> None:
    """Validate the typed route-reference provenance field."""
    if observation.route_reference is not None and not isinstance(
        observation.route_reference, RouteReference
    ):
        raise ValueError("route_reference must be a RouteReference or None")


def _normalize_clearance(observation: PedestrianResponseObservation) -> None:
    """Normalize and validate a supplied minimum clearance."""
    clearance = observation.minimum_passing_clearance_m
    if clearance is None:
        return
    if isinstance(clearance, bool):
        raise ValueError("minimum_passing_clearance_m must be finite and non-negative")
    try:
        normalized = float(clearance)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_passing_clearance_m must be finite and non-negative") from exc
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError("minimum_passing_clearance_m must be finite and non-negative")
    object.__setattr__(observation, "minimum_passing_clearance_m", normalized)


def _validate_response_flag(observation: PedestrianResponseObservation) -> None:
    """Validate the tri-state response-presence flag."""
    if observation.response_present is not None and type(observation.response_present) is not bool:
        raise ValueError("response_present must be a bool or None")


def _complete_field_state(
    observation: PedestrianResponseObservation,
    missing: set[str],
    unavailable: set[str],
) -> None:
    """Add implicit missing fields and reject contradictory field states."""
    for field_name in _REQUIRED_FIELDS:
        value = getattr(observation, field_name)
        if value is None:
            if field_name not in unavailable:
                missing.add(field_name)
        elif field_name in missing or field_name in unavailable:
            if (
                field_name in {"offered_side", "taken_side"}
                and value == "unavailable"
                and field_name in unavailable
            ):
                continue
            raise ValueError(f"{field_name} cannot be present and unavailable")


def _set_status(
    observation: PedestrianResponseObservation,
    missing: tuple[str, ...],
    unavailable: tuple[str, ...],
) -> None:
    """Set and validate the derived availability status."""
    expected: ResponseStatus = "available" if not missing and not unavailable else "not_available"
    if observation.status is not None and observation.status not in _RESPONSE_STATUSES:
        raise ValueError("status must be available or not_available")
    if observation.status is not None and observation.status != expected:
        raise ValueError(f"status must be {expected!r} for the supplied fields")
    object.__setattr__(observation, "status", expected)


def _set_unavailable_reason(
    observation: PedestrianResponseObservation,
    missing: tuple[str, ...],
    unavailable: tuple[str, ...],
) -> None:
    """Set and validate the record-level unavailable reason."""
    reason = observation.unavailable_reason
    if reason is not None and not isinstance(reason, str):
        raise ValueError("unavailable_reason must be a string or None")
    if observation.status == "available":
        if reason is not None and reason.strip():
            raise ValueError("available observations cannot carry unavailable_reason")
        object.__setattr__(observation, "unavailable_reason", None)
        return
    if reason is None or not reason.strip():
        if missing and unavailable:
            reason = "missing_and_unavailable_fields"
        elif missing:
            reason = "missing_fields"
        else:
            reason = "unavailable_fields"
    object.__setattr__(observation, "unavailable_reason", reason.strip())


__all__ = [
    "PEDESTRIAN_RESPONSE_CLAIM_BOUNDARY",
    "PEDESTRIAN_RESPONSE_SCHEMA_VERSION",
    "PedestrianResponseObservation",
    "ResponseStatus",
    "RouteReference",
    "build_pedestrian_response_observation",
]


def _normalize_reference_point(value: Any, field_name: str) -> tuple[float, float]:
    """Normalize one finite two-dimensional route-reference point.

    Returns:
        A finite two-dimensional point.
    """
    try:
        x, y = value
        point = (float(x), float(y))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite two-dimensional point") from exc
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{field_name} must be a finite two-dimensional point")
    return point


def _normalize_reference_scalar(value: Any, field_name: str) -> float:
    """Normalize one finite non-negative route-reference scalar.

    Returns:
        A finite non-negative scalar.
    """
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite and non-negative")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite and non-negative") from exc
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return normalized


def _normalize_reference_interval(value: Any) -> tuple[float, float]:
    """Normalize one finite strictly increasing progress interval.

    Returns:
        A finite interval bounded by zero and one.
    """
    try:
        lo, hi = value
        interval = (float(lo), float(hi))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("progress_interval must be finite and strictly increasing") from exc
    if not (
        all(math.isfinite(item) for item in interval) and 0.0 <= interval[0] < interval[1] <= 1.0
    ):
        raise ValueError("progress_interval must be finite and strictly increasing")
    return interval
