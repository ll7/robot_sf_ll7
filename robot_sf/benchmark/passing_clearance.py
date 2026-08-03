"""Versioned, footprint-aware passing-clearance contract.

The contract keeps center distance, proxy radii, and surface clearance
distinct. It is opt-in: existing planner and metric defaults do not acquire a
new threshold or a new radius meaning unless a caller supplies a contract.
Source-derived values are transfer priors, not validated AMMV requirements.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PASSING_CLEARANCE_SCHEMA_VERSION = "PassingClearanceContract.v1"
DISTANCE_BASIS_SURFACE_CLEARANCE = "surface_clearance_m"
_ENCOUNTER_TYPES = frozenset({"passing", "overtaking", "unspecified"})
_EVIDENCE_CLASSES = frozenset(
    {"source_observation", "derived_transfer_prior", "author_defined", "legacy_unverified"}
)


class PassingClearanceContractError(ValueError):
    """Raised when a passing-clearance contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class PassingClearanceContract:
    """Typed passing-clearance profile for circular proxy geometry.

    ``robot_radius_m`` and ``pedestrian_radius_m`` are circular proxy radii,
    not a statement about the physical AMMV footprint. The profile must carry
    an explicit evidence class and source limitation before it can be used in
    a new opt-in path.
    """

    profile_id: str
    robot_radius_m: float
    pedestrian_radius_m: float
    encounter_type: str = "unspecified"
    speed_range_mps: tuple[float, float] | None = None
    desired_clearance_m: float | None = None
    minimum_clearance_m: float | None = None
    source_citation: str = ""
    source_platform_geometry: str = ""
    evidence_class: str = "legacy_unverified"
    limitation: str = ""
    distance_basis: str = DISTANCE_BASIS_SURFACE_CLEARANCE

    def __post_init__(self) -> None:
        """Validate units, geometry, evidence class, and threshold ordering."""
        if not str(self.profile_id).strip():
            raise PassingClearanceContractError("profile_id must be non-empty")
        if self.distance_basis != DISTANCE_BASIS_SURFACE_CLEARANCE:
            raise PassingClearanceContractError(
                f"distance_basis must be {DISTANCE_BASIS_SURFACE_CLEARANCE!r}"
            )
        if self.encounter_type not in _ENCOUNTER_TYPES:
            raise PassingClearanceContractError(
                "encounter_type must be passing, overtaking, or unspecified"
            )
        if self.evidence_class not in _EVIDENCE_CLASSES:
            raise PassingClearanceContractError(
                "evidence_class is not a recognized PassingClearanceContract.v1 class"
            )
        _require_finite_nonnegative(self.robot_radius_m, "robot_radius_m")
        _require_finite_nonnegative(self.pedestrian_radius_m, "pedestrian_radius_m")
        if self.robot_radius_m + self.pedestrian_radius_m <= 0.0:
            raise PassingClearanceContractError("combined proxy radii must be positive")
        _validate_optional_clearance(self.desired_clearance_m, "desired_clearance_m")
        _validate_optional_clearance(self.minimum_clearance_m, "minimum_clearance_m")
        if (
            self.desired_clearance_m is not None
            and self.minimum_clearance_m is not None
            and self.minimum_clearance_m > self.desired_clearance_m
        ):
            raise PassingClearanceContractError(
                "minimum_clearance_m must not exceed desired_clearance_m"
            )
        _validate_speed_range(self.speed_range_mps)
        if self.evidence_class != "legacy_unverified" and not str(self.limitation).strip():
            raise PassingClearanceContractError(
                "non-legacy evidence classes require an explicit limitation"
            )

    @property
    def combined_radius_m(self) -> float:
        """Return the sum of the circular proxy radii in metres."""
        return float(self.robot_radius_m + self.pedestrian_radius_m)

    @property
    def profile_hash(self) -> str:
        """Return the stable SHA-256 hash of the canonical profile payload."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def center_distance_from_surface_clearance(self, clearance_m: float) -> float:
        """Convert surface clearance to center distance using this profile.

        Returns:
            Center-to-center distance in metres.
        """
        value = _finite_number(clearance_m, "surface_clearance_m")
        return value + self.combined_radius_m

    def surface_clearance_from_center_distance(self, center_distance_m: float) -> float:
        """Convert center distance to surface clearance using this profile.

        Returns:
            Surface clearance in metres.
        """
        value = _finite_number(center_distance_m, "center_distance_m")
        return value - self.combined_radius_m

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        """Return a JSON-safe contract mapping with units and provenance."""
        payload: dict[str, Any] = {
            "schema_version": PASSING_CLEARANCE_SCHEMA_VERSION,
            "profile_id": str(self.profile_id).strip(),
            "distance_basis": self.distance_basis,
            "robot_radius_m": float(self.robot_radius_m),
            "pedestrian_radius_m": float(self.pedestrian_radius_m),
            "combined_radius_m": self.combined_radius_m,
            "encounter_type": self.encounter_type,
            "speed_conditioning": ("unavailable" if self.speed_range_mps is None else "range_mps"),
            "speed_range_mps": (
                list(self.speed_range_mps) if self.speed_range_mps is not None else None
            ),
            "desired_clearance_m": self.desired_clearance_m,
            "minimum_clearance_m": self.minimum_clearance_m,
            "source_citation": str(self.source_citation).strip(),
            "source_platform_geometry": str(self.source_platform_geometry).strip(),
            "evidence_class": self.evidence_class,
            "limitation": str(self.limitation).strip(),
            "units": {
                "distance": "m",
                "speed": "m/s",
            },
        }
        if include_hash:
            payload["profile_hash"] = self.profile_hash
        return payload

    def canonical_json(self) -> str:
        """Return canonical JSON without the self-referential profile hash."""
        return json.dumps(self.to_dict(include_hash=False), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PassingClearanceContract:
        """Parse a strict mapping and verify an optional supplied hash.

        Returns:
            A validated contract.
        """
        fields = _mapping_constructor_fields(value)
        contract = cls(**fields)
        _validate_serialized_derived_fields(value, contract)
        return contract


def _mapping_constructor_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate serialized structure and return constructor fields.

    Returns:
        Constructor keyword arguments for :class:`PassingClearanceContract`.
    """
    _validate_mapping_header(value)
    speed_range = _mapping_speed_range(value.get("speed_range_mps"))
    try:
        return {
            "profile_id": str(value.get("profile_id", "")),
            "robot_radius_m": float(value.get("robot_radius_m")),
            "pedestrian_radius_m": float(value.get("pedestrian_radius_m")),
            "encounter_type": str(value.get("encounter_type", "unspecified")),
            "speed_range_mps": speed_range,
            "desired_clearance_m": _optional_float(value.get("desired_clearance_m")),
            "minimum_clearance_m": _optional_float(value.get("minimum_clearance_m")),
            "source_citation": str(value.get("source_citation", "")),
            "source_platform_geometry": str(value.get("source_platform_geometry", "")),
            "evidence_class": str(value.get("evidence_class", "legacy_unverified")),
            "limitation": str(value.get("limitation", "")),
            "distance_basis": str(value.get("distance_basis", "")),
        }
    except (TypeError, ValueError) as exc:
        if isinstance(exc, PassingClearanceContractError):
            raise
        raise PassingClearanceContractError("profile numeric fields must be valid") from exc


def _validate_mapping_header(value: Mapping[str, Any]) -> None:
    """Validate mapping identity, known fields, and units."""
    if not isinstance(value, Mapping):
        raise PassingClearanceContractError("passing-clearance profile must be a mapping")
    if value.get("schema_version") != PASSING_CLEARANCE_SCHEMA_VERSION:
        raise PassingClearanceContractError(
            f"schema_version must be {PASSING_CLEARANCE_SCHEMA_VERSION!r}"
        )
    allowed = {
        "schema_version",
        "profile_id",
        "distance_basis",
        "robot_radius_m",
        "pedestrian_radius_m",
        "combined_radius_m",
        "encounter_type",
        "speed_conditioning",
        "speed_range_mps",
        "desired_clearance_m",
        "minimum_clearance_m",
        "source_citation",
        "source_platform_geometry",
        "evidence_class",
        "limitation",
        "units",
        "profile_hash",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise PassingClearanceContractError(f"unknown profile fields: {unknown}")
    units = value.get("units")
    if (
        not isinstance(units, Mapping)
        or units.get("distance") != "m"
        or units.get("speed") != "m/s"
    ):
        raise PassingClearanceContractError("profile units must declare metres and m/s")


def _mapping_speed_range(value: object) -> tuple[float, float] | None:
    """Parse an optional serialized speed range.

    Returns:
        An ordered speed range, or ``None`` when unavailable.
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise PassingClearanceContractError("speed_range_mps must contain two values")
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise PassingClearanceContractError("speed_range_mps must be numeric") from exc


def _validate_serialized_derived_fields(
    value: Mapping[str, Any], contract: PassingClearanceContract
) -> None:
    """Check serialized fields derived from the validated contract."""
    expected_speed_conditioning = "unavailable" if contract.speed_range_mps is None else "range_mps"
    if value.get("speed_conditioning") != expected_speed_conditioning:
        raise PassingClearanceContractError("speed_conditioning does not match speed_range_mps")
    supplied_combined = value.get("combined_radius_m")
    if supplied_combined is not None and not math.isclose(
        float(supplied_combined), contract.combined_radius_m, rel_tol=0.0, abs_tol=1e-12
    ):
        raise PassingClearanceContractError("combined_radius_m does not match proxy radii")
    supplied_hash = value.get("profile_hash")
    if supplied_hash is not None and supplied_hash != contract.profile_hash:
        raise PassingClearanceContractError("profile_hash does not match canonical profile")


def neggers_source_transfer_prior() -> PassingClearanceContract:
    """Return the source-specific passing-distance transfer prior.

    The source range is represented as 0.36--0.56 m surface clearance after
    subtracting an approximately 0.44 m combined circular proxy radius from
    the reported 0.80--1.00 m center-to-center range. This is not a universal
    active default and is not human-subject validated for an AMMV.

    Returns:
        A source-specific, explicitly non-universal transfer-prior profile.
    """
    return PassingClearanceContract(
        profile_id="neggers-2022-passing-distance-transfer-prior-v1",
        robot_radius_m=0.22,
        pedestrian_radius_m=0.22,
        encounter_type="passing",
        speed_range_mps=None,
        desired_clearance_m=0.56,
        minimum_clearance_m=0.36,
        source_citation="Neggers et al. 2022, DOI:10.3389/frobt.2022.915972",
        source_platform_geometry="Source robot 0.425 m x 0.480 m; combined circular proxy approximately 0.44 m.",
        evidence_class="derived_transfer_prior",
        limitation="Edge-clearance transfer to a different platform size is not human-subject validated.",
    )


def resolve_passing_clearance_contract(
    value: PassingClearanceContract | Mapping[str, Any] | None,
) -> PassingClearanceContract | None:
    """Resolve an optional explicit contract without changing legacy behavior.

    Returns:
        A validated contract, or ``None`` when the caller did not opt in.
    """
    if value is None:
        return None
    if isinstance(value, PassingClearanceContract):
        return value
    return PassingClearanceContract.from_mapping(value)


def _finite_number(value: object, field_name: str) -> float:
    """Require a finite numeric conversion.

    Returns:
        The finite float value.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PassingClearanceContractError(f"{field_name} must be finite") from exc
    if not math.isfinite(number):
        raise PassingClearanceContractError(f"{field_name} must be finite")
    return number


def _require_finite_nonnegative(value: object, field_name: str) -> None:
    """Require a finite non-negative quantity."""
    number = _finite_number(value, field_name)
    if number < 0.0:
        raise PassingClearanceContractError(f"{field_name} must be non-negative")


def _validate_optional_clearance(value: float | None, field_name: str) -> None:
    """Validate an optional non-negative clearance threshold."""
    if value is not None:
        _require_finite_nonnegative(value, field_name)


def _validate_speed_range(value: tuple[float, float] | None) -> None:
    """Validate an optional inclusive speed range."""
    if value is None:
        return
    if len(value) != 2:
        raise PassingClearanceContractError("speed_range_mps must contain two values")
    low = _finite_number(value[0], "speed_range_mps[0]")
    high = _finite_number(value[1], "speed_range_mps[1]")
    if low < 0.0 or high < low:
        raise PassingClearanceContractError("speed_range_mps must be ordered and non-negative")


def _optional_float(value: object) -> float | None:
    """Convert an optional numeric field without accepting non-finite values.

    Returns:
        A finite float or ``None``.
    """
    if value is None:
        return None
    return _finite_number(value, "optional clearance field")


__all__ = [
    "DISTANCE_BASIS_SURFACE_CLEARANCE",
    "PASSING_CLEARANCE_SCHEMA_VERSION",
    "PassingClearanceContract",
    "PassingClearanceContractError",
    "neggers_source_transfer_prior",
    "resolve_passing_clearance_contract",
]
