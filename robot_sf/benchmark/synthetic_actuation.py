"""Synthetic differential-drive actuation-envelope helpers for diagnostic benchmark slices."""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SYNTHETIC_ACTUATION_CLAIM_SCOPE = "synthetic-only"
SYNTHETIC_ACTUATION_CLAIM_BOUNDARY = "diagnostic-only"
CALIBRATED_ACTUATION_REQUIRED_PROVENANCE_FIELDS = (
    "source_id",
    "source_uri",
    "source_type",
    "profile_version",
    "measurement_date",
    "supported_actuation_fields",
    "units",
    "claim_boundary",
)

_LATENCY_MODE_TO_STEPS = {
    "zero-step-delay": 0,
    "one-step-delay": 1,
    "two-step-delay": 2,
}
_UPDATE_MODE_TO_STEPS = {
    "10hz-matched": 1,
    "5hz-hold": 2,
    "2.5hz-hold": 4,
}
_SATURATION_TOL = 1e-9


@dataclass(frozen=True)
class SyntheticActuationProfile:
    """Serializable synthetic actuation-envelope profile for one diagnostic campaign."""

    name: str
    max_linear_accel_m_s2: float
    max_linear_decel_m_s2: float
    max_yaw_rate_rad_s: float
    max_angular_accel_rad_s2: float
    latency_mode: str
    update_mode: str
    profile_version: str = "v0"
    claim_scope: str = SYNTHETIC_ACTUATION_CLAIM_SCOPE
    claim_boundary: str = SYNTHETIC_ACTUATION_CLAIM_BOUNDARY

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-safe metadata payload."""
        return {
            "name": self.name,
            "profile_version": self.profile_version,
            "claim_scope": self.claim_scope,
            "claim_boundary": self.claim_boundary,
            "max_linear_accel_m_s2": float(self.max_linear_accel_m_s2),
            "max_linear_decel_m_s2": float(self.max_linear_decel_m_s2),
            "max_yaw_rate_rad_s": float(self.max_yaw_rate_rad_s),
            "max_angular_accel_rad_s2": float(self.max_angular_accel_rad_s2),
            "latency_mode": self.latency_mode,
            "update_mode": self.update_mode,
        }


def known_latency_modes() -> tuple[str, ...]:
    """Return supported synthetic latency-mode labels."""
    return tuple(_LATENCY_MODE_TO_STEPS)


def known_update_modes() -> tuple[str, ...]:
    """Return supported synthetic update-mode labels."""
    return tuple(_UPDATE_MODE_TO_STEPS)


def _non_empty_string(value: Any) -> bool:
    """Return whether a value is a non-empty string after stripping whitespace."""
    return isinstance(value, str) and bool(value.strip())


def _non_empty_sequence(value: Any) -> bool:
    """Return whether a value is a non-empty list/tuple of non-empty strings."""
    return (
        isinstance(value, list | tuple)
        and bool(value)
        and all(_non_empty_string(item) for item in value)
    )


def _looks_calibrated_actuation_profile(payload: Mapping[str, Any]) -> bool:
    """Return whether profile metadata is labeled as calibrated or hardware-aligned."""
    calibrated_markers = ("calibrated", "hardware", "measured")
    fields = (
        payload.get("name"),
        payload.get("claim_scope"),
        payload.get("claim_boundary"),
        payload.get("calibration_status"),
        payload.get("profile_family"),
    )
    return any(
        token in calibrated_markers
        for value in fields
        for token in re.split(r"[^a-z]+", str(value).strip().lower())
        if token
    )


def _validate_calibrated_actuation_provenance(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Validate required provenance fields for calibrated-labeled actuation profiles."""
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        fields = ", ".join(CALIBRATED_ACTUATION_REQUIRED_PROVENANCE_FIELDS)
        raise ValueError(
            f"{label} calibrated actuation profile requires provenance fields: {fields}"
        )

    missing: list[str] = []
    for field_name in CALIBRATED_ACTUATION_REQUIRED_PROVENANCE_FIELDS:
        value = provenance.get(field_name)
        if field_name == "supported_actuation_fields":
            if not _non_empty_sequence(value):
                missing.append(field_name)
        elif field_name == "units":
            if not isinstance(value, Mapping) or not value:
                missing.append(field_name)
        elif not _non_empty_string(value):
            missing.append(field_name)
    if missing:
        raise ValueError(
            f"{label} calibrated actuation profile missing provenance fields: {', '.join(missing)}"
        )


def validate_actuation_profile_claim_boundary(
    payload: Mapping[str, Any],
    *,
    label: str = "synthetic_actuation_profile",
) -> None:
    """Validate synthetic-vs-calibrated AMV actuation claim-boundary metadata.

    Synthetic profiles must explicitly remain diagnostic-only. Any profile labeled as calibrated,
    hardware-aligned, or measured must carry the minimum provenance fields needed for a future
    calibrated profile before downstream code may inspect numeric limits.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must be a mapping when provided")
    if _looks_calibrated_actuation_profile(payload):
        _validate_calibrated_actuation_provenance(payload, label=label)
        return

    claim_scope = str(payload.get("claim_scope", "")).strip()
    if claim_scope == SYNTHETIC_ACTUATION_CLAIM_SCOPE:
        claim_boundary = str(payload.get("claim_boundary", "")).strip()
        if claim_boundary != SYNTHETIC_ACTUATION_CLAIM_BOUNDARY:
            raise ValueError(
                f"{label}.claim_boundary must be '{SYNTHETIC_ACTUATION_CLAIM_BOUNDARY}'"
            )


def _validate_synthetic_positive_bounds(profile: SyntheticActuationProfile) -> None:
    """Validate positive numeric limits for a synthetic actuation profile."""
    bounds = {
        "max_linear_accel_m_s2": profile.max_linear_accel_m_s2,
        "max_linear_decel_m_s2": profile.max_linear_decel_m_s2,
        "max_yaw_rate_rad_s": profile.max_yaw_rate_rad_s,
        "max_angular_accel_rad_s2": profile.max_angular_accel_rad_s2,
    }
    for field_name, value in bounds.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"synthetic_actuation_profile.{field_name} must be > 0")


def _validate_mode(value: str, *, field_name: str, known: tuple[str, ...]) -> None:
    """Validate one synthetic profile mode label."""
    if value not in known:
        joined = ", ".join(known)
        raise ValueError(
            f"Unsupported synthetic_actuation_profile.{field_name} '{value}'. "
            f"Expected one of: {joined}"
        )


def validate_synthetic_actuation_profile(profile: SyntheticActuationProfile) -> None:
    """Validate that one synthetic profile is usable for differential-drive diagnostics."""
    if not profile.name.strip():
        raise ValueError("synthetic_actuation_profile.name must be non-empty")
    if not profile.profile_version.strip():
        raise ValueError("synthetic_actuation_profile.profile_version must be non-empty")
    profile_metadata = profile.to_metadata()
    if _looks_calibrated_actuation_profile(profile_metadata):
        _validate_calibrated_actuation_provenance(
            profile_metadata,
            label="synthetic_actuation_profile",
        )
    if profile.claim_scope.strip() != SYNTHETIC_ACTUATION_CLAIM_SCOPE:
        raise ValueError(
            f"synthetic_actuation_profile.claim_scope must be '{SYNTHETIC_ACTUATION_CLAIM_SCOPE}'"
        )
    if profile.claim_boundary.strip() != SYNTHETIC_ACTUATION_CLAIM_BOUNDARY:
        raise ValueError(
            "synthetic_actuation_profile.claim_boundary must be "
            f"'{SYNTHETIC_ACTUATION_CLAIM_BOUNDARY}'"
        )
    _validate_synthetic_positive_bounds(profile)
    _validate_mode(
        profile.latency_mode,
        field_name="latency_mode",
        known=known_latency_modes(),
    )
    _validate_mode(
        profile.update_mode,
        field_name="update_mode",
        known=known_update_modes(),
    )


@dataclass(frozen=True)
class SyntheticActuationStep:
    """One synthetic-actuation trace step."""

    requested_command: tuple[float, float]
    delayed_command: tuple[float, float]
    held_command: tuple[float, float]
    applied_command: tuple[float, float]
    command_clipped: bool
    yaw_rate_saturated: bool
    linear_accel_applied_m_s2: float
    angular_accel_applied_rad_s2: float


class SyntheticActuationController:
    """Apply a synthetic differential-drive envelope to absolute ``(v, omega)`` commands."""

    def __init__(self, profile: SyntheticActuationProfile, *, dt: float) -> None:
        """Initialize controller state for one episode."""
        validate_synthetic_actuation_profile(profile)
        if dt <= 0.0:
            raise ValueError("Synthetic actuation controller requires dt > 0")
        self.profile = profile
        self.dt = float(dt)
        self._latency_steps = _LATENCY_MODE_TO_STEPS[profile.latency_mode]
        self._hold_steps = _UPDATE_MODE_TO_STEPS[profile.update_mode]
        self._delay_buffer: deque[tuple[float, float]] = deque(
            [(0.0, 0.0)] * max(1, self._latency_steps + 1),
            maxlen=max(1, self._latency_steps + 1),
        )
        self._held_command = (0.0, 0.0)
        self._hold_counter = 0
        self._step_count = 0
        self._clip_count = 0
        self._yaw_saturation_count = 0
        self._signed_braking_peak = 0.0

    def apply(
        self,
        *,
        current_command: tuple[float, float],
        requested_command: tuple[float, float],
    ) -> SyntheticActuationStep:
        """Return the delayed, held, and clipped command for one benchmark step."""
        requested_linear = float(requested_command[0])
        requested_angular = float(requested_command[1])
        self._delay_buffer.append((requested_linear, requested_angular))
        delayed_command = self._delay_buffer[0]
        if self._step_count == 0 or self._hold_counter <= 0:
            self._held_command = delayed_command
            self._hold_counter = self._hold_steps
        self._hold_counter -= 1
        held_command = self._held_command

        current_linear = float(current_command[0])
        current_angular = float(current_command[1])
        target_linear = float(held_command[0])
        target_angular = float(held_command[1])

        allowed_linear_up = float(self.profile.max_linear_accel_m_s2) * self.dt
        allowed_linear_down = float(self.profile.max_linear_decel_m_s2) * self.dt
        requested_linear_delta = target_linear - current_linear
        if requested_linear_delta >= 0.0:
            applied_linear = current_linear + min(requested_linear_delta, allowed_linear_up)
        else:
            applied_linear = current_linear + max(requested_linear_delta, -allowed_linear_down)

        bounded_target_angular = max(
            -float(self.profile.max_yaw_rate_rad_s),
            min(float(self.profile.max_yaw_rate_rad_s), target_angular),
        )
        allowed_angular_delta = float(self.profile.max_angular_accel_rad_s2) * self.dt
        requested_angular_delta = bounded_target_angular - current_angular
        applied_angular = current_angular + max(
            -allowed_angular_delta,
            min(allowed_angular_delta, requested_angular_delta),
        )

        applied_command = (float(applied_linear), float(applied_angular))
        command_clipped = (
            abs(applied_linear - target_linear) > _SATURATION_TOL
            or abs(applied_angular - target_angular) > _SATURATION_TOL
        )
        yaw_rate_saturated = abs(bounded_target_angular - target_angular) > _SATURATION_TOL

        self._step_count += 1
        if command_clipped:
            self._clip_count += 1
        if yaw_rate_saturated:
            self._yaw_saturation_count += 1

        linear_accel = float((applied_linear - current_linear) / self.dt)
        angular_accel = float((applied_angular - current_angular) / self.dt)
        self._signed_braking_peak = min(self._signed_braking_peak, linear_accel)
        return SyntheticActuationStep(
            requested_command=(requested_linear, requested_angular),
            delayed_command=(float(delayed_command[0]), float(delayed_command[1])),
            held_command=(float(held_command[0]), float(held_command[1])),
            applied_command=applied_command,
            command_clipped=command_clipped,
            yaw_rate_saturated=yaw_rate_saturated,
            linear_accel_applied_m_s2=linear_accel,
            angular_accel_applied_rad_s2=angular_accel,
        )

    def summary(self) -> dict[str, Any]:
        """Return auditable per-episode saturation diagnostics."""
        if self._step_count <= 0:
            return {
                "schema_version": "synthetic-actuation-summary.v1",
                "status": "not_available",
                "command_clip_fraction": "not_available",
                "yaw_rate_saturation_fraction": "not_available",
                "signed_braking_peak_m_s2": "not_available",
                "step_count": 0,
            }
        return {
            "schema_version": "synthetic-actuation-summary.v1",
            "status": "ok",
            "command_clip_fraction": float(self._clip_count / self._step_count),
            "yaw_rate_saturation_fraction": float(self._yaw_saturation_count / self._step_count),
            "signed_braking_peak_m_s2": float(self._signed_braking_peak),
            "step_count": int(self._step_count),
            "command_clip_steps": int(self._clip_count),
            "yaw_rate_saturation_steps": int(self._yaw_saturation_count),
        }


def not_available_saturation_metrics() -> dict[str, str]:
    """Return explicit placeholders for unimplemented saturation metrics."""
    return {
        "command_clip_fraction": "not_available",
        "yaw_rate_saturation_fraction": "not_available",
        "signed_braking_peak_m_s2": "not_available",
    }
