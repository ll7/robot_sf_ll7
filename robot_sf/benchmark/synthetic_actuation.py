"""Synthetic differential-drive actuation-envelope helpers for diagnostic benchmark slices."""

from __future__ import annotations

import math
import random
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SYNTHETIC_ACTUATION_CLAIM_SCOPE = "synthetic-only"
SYNTHETIC_ACTUATION_CLAIM_BOUNDARY = "diagnostic-only"
CALIBRATED_ACTUATION_CLAIM_SCOPE = "hardware-calibrated"
CALIBRATED_ACTUATION_CLAIM_BOUNDARY = "calibrated-amv-actuation"
SYNTHETIC_ACTUATION_VARIABILITY_SCHEMA_VERSION = "synthetic-actuation-variability-distribution.v1"
SYNTHETIC_ACTUATION_VARIABILITY_SAMPLE_SCHEMA_VERSION = "synthetic-actuation-variability-sample.v1"
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
_NUMERIC_VARIABILITY_FIELDS = (
    "max_linear_accel_m_s2",
    "max_linear_decel_m_s2",
    "max_yaw_rate_rad_s",
    "max_angular_accel_rad_s2",
)
_CATEGORICAL_VARIABILITY_FIELDS = ("latency_mode", "update_mode")
_VARIABILITY_FIELDS = _NUMERIC_VARIABILITY_FIELDS + _CATEGORICAL_VARIABILITY_FIELDS
_VARIABILITY_FIELD_UNITS = {
    "max_linear_accel_m_s2": "m/s^2",
    "max_linear_decel_m_s2": "m/s^2",
    "max_yaw_rate_rad_s": "rad/s",
    "max_angular_accel_rad_s2": "rad/s^2",
    "latency_mode": "profile-label",
    "update_mode": "profile-label",
}


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
    variability_distribution: Mapping[str, Any] | None = None
    variability_sample: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-safe metadata payload."""
        payload = {
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
        if self.variability_distribution is not None:
            payload["variability_distribution"] = _json_safe_mapping(self.variability_distribution)
        if self.variability_sample is not None:
            payload["variability_sample"] = _json_safe_mapping(self.variability_sample)
        if self.provenance is not None:
            payload["provenance"] = _json_safe_mapping(self.provenance)
        return payload


def known_latency_modes() -> tuple[str, ...]:
    """Return supported synthetic latency-mode labels."""
    return tuple(_LATENCY_MODE_TO_STEPS)


def known_update_modes() -> tuple[str, ...]:
    """Return supported synthetic update-mode labels."""
    return tuple(_UPDATE_MODE_TO_STEPS)


def actuation_variability_fields() -> tuple[str, ...]:
    """Return profile fields that may be sampled by synthetic variability distributions."""
    return _VARIABILITY_FIELDS


def _json_safe_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of one metadata mapping."""
    return {
        str(key): _json_safe_value(value) for key, value in payload.items() if value is not None
    }


def _json_safe_value(value: Any) -> Any:
    """Return a JSON-safe copy of nested metadata values."""
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    return value


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


def missing_calibrated_provenance_fields(payload: Mapping[str, Any]) -> list[str]:
    """Return required calibrated-provenance fields that are absent or empty in ``payload``.

    A missing ``provenance`` mapping reports every required field as missing. This is the single
    owner of the field-emptiness rules so structural validation and the readiness/preflight checker
    in :mod:`robot_sf.benchmark.amv_calibration_readiness` stay consistent.
    """
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        return list(CALIBRATED_ACTUATION_REQUIRED_PROVENANCE_FIELDS)

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
    return missing


def looks_calibrated_actuation_profile(payload: Mapping[str, Any]) -> bool:
    """Public wrapper for calibrated-marker detection.

    Returns:
        True when ``payload`` is labeled calibrated/hardware-aligned/measured.
    """
    return _looks_calibrated_actuation_profile(payload)


def _validate_calibrated_actuation_provenance(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Validate required provenance fields for calibrated-labeled actuation profiles."""
    if not isinstance(payload.get("provenance"), Mapping):
        fields = ", ".join(CALIBRATED_ACTUATION_REQUIRED_PROVENANCE_FIELDS)
        raise ValueError(
            f"{label} calibrated actuation profile requires provenance fields: {fields}"
        )

    missing = missing_calibrated_provenance_fields(payload)
    if missing:
        raise ValueError(
            f"{label} calibrated actuation profile missing provenance fields: {', '.join(missing)}"
        )


def _reject_synthetic_calibrated_conflation(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Reject profiles that mix synthetic-only claim_scope with calibrated-looking markers."""
    claim_scope = str(payload.get("claim_scope", "")).strip()
    if claim_scope == SYNTHETIC_ACTUATION_CLAIM_SCOPE and _looks_calibrated_actuation_profile(
        payload
    ):
        raise ValueError(
            f"{label} has claim_scope='{SYNTHETIC_ACTUATION_CLAIM_SCOPE}' but contains "
            "calibrated-looking markers. A calibrated-actuation profile must use "
            f"claim_scope='{CALIBRATED_ACTUATION_CLAIM_SCOPE}' and provide provenance metadata."
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

    _reject_synthetic_calibrated_conflation(payload, label=label)

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


def _validate_variability_provenance(
    provenance: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    """Validate the source-status metadata attached to one variability distribution.

    Returns:
        The original provenance mapping after validation.
    """
    if not isinstance(provenance, Mapping):
        raise ValueError(
            f"synthetic_actuation_profile.variability_distribution.parameters."
            f"{field_name}.provenance must be a mapping"
        )
    required = ("source_status", "caveat", "units")
    missing = [name for name in required if not _non_empty_string(provenance.get(name))]
    expected_units = _VARIABILITY_FIELD_UNITS[field_name]
    if _non_empty_string(provenance.get("units")) and str(provenance["units"]) != expected_units:
        missing.append("units")
    if missing:
        raise ValueError(
            f"synthetic_actuation_profile.variability_distribution.parameters.{field_name}"
            f".provenance missing fields: {', '.join(sorted(set(missing)))}"
        )
    return provenance


def _validate_numeric_variability_spec(field_name: str, spec: Mapping[str, Any]) -> None:
    """Validate one numeric uniform distribution spec."""
    if str(spec.get("distribution", "")).strip() != "uniform":
        raise ValueError(
            "Numeric synthetic actuation variability fields currently support only "
            f"uniform distributions: {field_name}"
        )
    try:
        low = float(spec.get("low"))
        high = float(spec.get("high"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "synthetic_actuation_profile.variability_distribution.parameters."
            f"{field_name} requires numeric low/high bounds"
        ) from exc
    if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high < low:
        raise ValueError(
            "synthetic_actuation_profile.variability_distribution.parameters."
            f"{field_name} requires finite 0 < low <= high"
        )


def _validate_categorical_variability_spec(field_name: str, spec: Mapping[str, Any]) -> None:
    """Validate one categorical choice distribution spec."""
    if str(spec.get("distribution", "")).strip() != "choice":
        raise ValueError(
            "Categorical synthetic actuation variability fields currently support only "
            f"choice distributions: {field_name}"
        )
    choices = spec.get("choices")
    if not isinstance(choices, list | tuple) or not choices:
        raise ValueError(
            "synthetic_actuation_profile.variability_distribution.parameters."
            f"{field_name}.choices must be a non-empty sequence"
        )
    known = known_latency_modes() if field_name == "latency_mode" else known_update_modes()
    for choice in choices:
        _validate_mode(str(choice), field_name=field_name, known=known)


def validate_synthetic_actuation_variability_distribution(
    distribution: Mapping[str, Any],
) -> None:
    """Validate a synthetic/provisional distribution over actuation profile fields."""
    if not isinstance(distribution, Mapping):
        raise ValueError("synthetic_actuation_profile.variability_distribution must be a mapping")
    if distribution.get("schema_version") != SYNTHETIC_ACTUATION_VARIABILITY_SCHEMA_VERSION:
        raise ValueError(
            "synthetic_actuation_profile.variability_distribution.schema_version must be "
            f"'{SYNTHETIC_ACTUATION_VARIABILITY_SCHEMA_VERSION}'"
        )
    mode = str(distribution.get("mode", "")).strip()
    if mode != "synthetic-provisional":
        raise ValueError(
            "synthetic_actuation_profile.variability_distribution.mode must be "
            "'synthetic-provisional'"
        )
    claim_boundary = str(
        distribution.get("claim_boundary", SYNTHETIC_ACTUATION_CLAIM_BOUNDARY)
    ).strip()
    if claim_boundary != SYNTHETIC_ACTUATION_CLAIM_BOUNDARY:
        raise ValueError(
            "synthetic_actuation_profile.variability_distribution.claim_boundary must be "
            f"'{SYNTHETIC_ACTUATION_CLAIM_BOUNDARY}'"
        )
    parameters = distribution.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError(
            "synthetic_actuation_profile.variability_distribution.parameters must be a "
            "non-empty mapping"
        )
    for raw_field_name, raw_spec in parameters.items():
        field_name = str(raw_field_name)
        if field_name not in _VARIABILITY_FIELDS:
            known = ", ".join(_VARIABILITY_FIELDS)
            raise ValueError(
                "Unsupported synthetic_actuation_profile.variability_distribution "
                f"field '{field_name}'. Expected one of: {known}"
            )
        if not isinstance(raw_spec, Mapping):
            raise ValueError(
                "synthetic_actuation_profile.variability_distribution.parameters."
                f"{field_name} must be a mapping"
            )
        _validate_variability_provenance(raw_spec.get("provenance"), field_name=field_name)
        if field_name in _NUMERIC_VARIABILITY_FIELDS:
            _validate_numeric_variability_spec(field_name, raw_spec)
        else:
            _validate_categorical_variability_spec(field_name, raw_spec)


def _profile_field_value(profile: SyntheticActuationProfile, field_name: str) -> Any:
    """Return one supported field value from a synthetic actuation profile."""
    return getattr(profile, field_name)


def _sample_distribution_value(
    field_name: str,
    spec: Mapping[str, Any],
    *,
    rng: random.Random,
) -> Any:
    """Sample one concrete profile value from a validated field distribution.

    Returns:
        A sampled numeric or categorical profile value.
    """
    if field_name in _NUMERIC_VARIABILITY_FIELDS:
        return float(rng.uniform(float(spec["low"]), float(spec["high"])))
    choices = [str(choice) for choice in spec["choices"]]
    return choices[rng.randrange(len(choices))]


def sample_synthetic_actuation_profile(
    base_profile: SyntheticActuationProfile,
    distribution: Mapping[str, Any],
    *,
    seed: int,
    sample_index: int,
    name: str | None = None,
) -> SyntheticActuationProfile:
    """Materialize one deterministic synthetic variability sample as scalar profile values.

    Returns:
        A validated scalar synthetic actuation profile carrying sample metadata.
    """
    validate_synthetic_actuation_profile(base_profile)
    validate_synthetic_actuation_variability_distribution(distribution)
    if sample_index < 0:
        raise ValueError("sample_index must be >= 0")

    parameters = distribution["parameters"]
    rng = random.Random(int(seed) + sample_index * 1_000_003)
    sampled_values: dict[str, Any] = {}
    for field_name, raw_spec in sorted(parameters.items()):
        spec = raw_spec if isinstance(raw_spec, Mapping) else {}
        sampled_values[str(field_name)] = _sample_distribution_value(str(field_name), spec, rng=rng)

    sample_payload = {
        "schema_version": SYNTHETIC_ACTUATION_VARIABILITY_SAMPLE_SCHEMA_VERSION,
        "mode": "variability-sweep",
        "sample_index": int(sample_index),
        "sample_id": f"sample-{sample_index:03d}",
        "sampling_seed": int(seed),
        "sampled_parameters": _json_safe_mapping(sampled_values),
        "summary": {
            field_name: {
                "value": _json_safe_value(value),
                "units": _VARIABILITY_FIELD_UNITS[field_name],
            }
            for field_name, value in sampled_values.items()
        },
    }

    values = {
        field_name: _profile_field_value(base_profile, field_name)
        for field_name in _VARIABILITY_FIELDS
    }
    values.update(sampled_values)
    return SyntheticActuationProfile(
        name=name or f"{base_profile.name}-{sample_payload['sample_id']}",
        profile_version=base_profile.profile_version,
        claim_scope=base_profile.claim_scope,
        claim_boundary=base_profile.claim_boundary,
        max_linear_accel_m_s2=float(values["max_linear_accel_m_s2"]),
        max_linear_decel_m_s2=float(values["max_linear_decel_m_s2"]),
        max_yaw_rate_rad_s=float(values["max_yaw_rate_rad_s"]),
        max_angular_accel_rad_s2=float(values["max_angular_accel_rad_s2"]),
        latency_mode=str(values["latency_mode"]),
        update_mode=str(values["update_mode"]),
        variability_distribution=distribution,
        variability_sample=sample_payload,
    )


def summarize_synthetic_actuation_samples(
    profiles: list[SyntheticActuationProfile],
) -> dict[str, Any]:
    """Summarize sampled actuation parameters for materialized variability sweeps.

    Returns:
        JSON-safe sampled-parameter summary rows.
    """
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        metadata = profile.to_metadata()
        sample = metadata.get("variability_sample")
        if not isinstance(sample, Mapping):
            continue
        row = {
            "profile_name": profile.name,
            "sample_id": str(sample.get("sample_id", "")),
            "sample_index": int(sample.get("sample_index", -1)),
            "sampling_seed": int(sample.get("sampling_seed", 0)),
            "sampled_parameters": dict(sample.get("sampled_parameters", {})),
            "summary": dict(sample.get("summary", {})),
            "claim_boundary": profile.claim_boundary,
        }
        rows.append(row)
    return {
        "schema_version": "synthetic-actuation-sampled-parameter-summary.v1",
        "claim_boundary": SYNTHETIC_ACTUATION_CLAIM_BOUNDARY,
        "row_count": len(rows),
        "rows": rows,
    }


def _validate_variability_sample(sample: Any) -> None:
    """Validate optional sampled-parameter metadata on a synthetic profile."""
    if not isinstance(sample, Mapping):
        raise ValueError("synthetic_actuation_profile.variability_sample must be a mapping")
    if sample.get("schema_version") != SYNTHETIC_ACTUATION_VARIABILITY_SAMPLE_SCHEMA_VERSION:
        raise ValueError(
            "synthetic_actuation_profile.variability_sample.schema_version must be "
            f"'{SYNTHETIC_ACTUATION_VARIABILITY_SAMPLE_SCHEMA_VERSION}'"
        )


def validate_synthetic_actuation_profile(profile: SyntheticActuationProfile) -> None:
    """Validate that one synthetic profile is usable for differential-drive diagnostics."""
    if not profile.name.strip():
        raise ValueError("synthetic_actuation_profile.name must be non-empty")
    if not profile.profile_version.strip():
        raise ValueError("synthetic_actuation_profile.profile_version must be non-empty")
    if profile.variability_distribution is not None and not isinstance(
        profile.variability_distribution, Mapping
    ):
        raise ValueError("synthetic_actuation_profile.variability_distribution must be a mapping")
    sample = profile.variability_sample
    if sample is not None and not isinstance(sample, Mapping):
        raise ValueError("synthetic_actuation_profile.variability_sample must be a mapping")
    profile_metadata = profile.to_metadata()

    _reject_synthetic_calibrated_conflation(profile_metadata, label="synthetic_actuation_profile")

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
    if profile.variability_distribution is not None:
        validate_synthetic_actuation_variability_distribution(profile.variability_distribution)
    if sample is not None:
        _validate_variability_sample(sample)


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
