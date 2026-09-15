"""Diagnostic-only provenance receipt for versioned obstacle-force metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from numbers import Integral
from typing import Any

OBSTACLE_FORCE_DIAGNOSTIC_RECEIPT_SCHEMA_VERSION = "obstacle_force_law_diagnostic_receipt.v1"

_LAW_VERSIONS = frozenset({"legacy_shifted_gradient_v1", "surface_distance_unit_normal_v2"})
_SELECTION_SOURCES = frozenset({"defaulted_missing", "historical_unversioned", "explicit"})
_SHA256_HEX = frozenset("0123456789abcdef")
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "claim_boundary",
        "site",
        "law_version",
        "selection_source",
        "parameters_sha256",
        "enabled",
        "applied",
        "fallback",
        "input_identity",
    }
)
_INPUT_IDENTITY_FIELDS = frozenset({"config_hash", "source_commit", "input_sha256"})
_FALLBACK_FIELDS = frozenset(
    {
        "fallback",
        "fallback_triggered",
        "used",
        "fallback_count",
        "count",
        "fallback_reason",
        "first_reason",
        "fallback_reasons",
        "reasons",
    }
)


class ObstacleForceDiagnosticReceiptError(ValueError):
    """Raised when a diagnostic receipt cannot be made unambiguous and JSON-safe."""


def _non_empty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObstacleForceDiagnosticReceiptError(f"{field} must be a non-empty string")
    return value.strip()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _json_bytes(value: Any, *, field: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise ObstacleForceDiagnosticReceiptError(f"{field} must be JSON-safe") from exc


def _parameters_digest(parameters: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(dict(parameters), field="parameters")).hexdigest()


def _input_digest(
    *,
    config_hash: str,
    source_commit: str,
    site: str,
    law_version: str,
    selection_source: str,
    parameters_sha256: str | None,
) -> str:
    payload = {
        "config_hash": config_hash,
        "law_version": law_version,
        "parameters_sha256": parameters_sha256,
        "selection_source": selection_source,
        "site": site,
        "source_commit": source_commit,
    }
    return hashlib.sha256(_json_bytes(payload, field="input identity")).hexdigest()


def obstacle_force_fallback_from_mapping(
    payload: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Extract fallback fields from a metadata or planner-diagnostics mapping.

    The wrapper diagnostics use the ``fallback_*`` names while the receipt uses
    shorter names. Returning only the recognized fields keeps unrelated planner
    diagnostics out of the receipt and lets callers preserve an absent state as
    the explicit default ``used=false, count=0``.

    Returns:
        Recognized fallback fields, or ``None`` when no fallback fields are present.
    """

    if not isinstance(payload, Mapping):
        return None
    nested = payload.get("fallback")
    if isinstance(nested, Mapping):
        return dict(nested)
    if not any(key in payload for key in _FALLBACK_FIELDS):
        return None
    return {key: payload[key] for key in _FALLBACK_FIELDS if key in payload}


def _normalize_fallback(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    source = obstacle_force_fallback_from_mapping(payload) or {}
    raw_used = source.get("used", source.get("fallback", source.get("fallback_triggered", False)))
    if not isinstance(raw_used, bool):
        raise ObstacleForceDiagnosticReceiptError("fallback.used must be a boolean")

    raw_count = source.get("count", source.get("fallback_count", 0))
    if isinstance(raw_count, bool) or not isinstance(raw_count, Integral) or raw_count < 0:
        raise ObstacleForceDiagnosticReceiptError("fallback.count must be a non-negative integer")
    count = int(raw_count)

    raw_reason = source.get("first_reason", source.get("fallback_reason"))
    if raw_reason is not None:
        raw_reason = _non_empty_text(raw_reason, field="fallback.first_reason")

    raw_reasons = source.get("reasons", source.get("fallback_reasons", {}))
    if not isinstance(raw_reasons, Mapping):
        raise ObstacleForceDiagnosticReceiptError("fallback.reasons must be a mapping")
    reasons: dict[str, int] = {}
    for key, value in raw_reasons.items():
        reason = _non_empty_text(key, field="fallback.reasons key")
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ObstacleForceDiagnosticReceiptError(
                "fallback.reasons values must be non-negative integers"
            )
        reasons[reason] = int(value)

    if count > 0 and not raw_used:
        raise ObstacleForceDiagnosticReceiptError("fallback.count requires fallback.used=true")
    if not raw_used and (count != 0 or raw_reason is not None or reasons):
        raise ObstacleForceDiagnosticReceiptError(
            "fallback details cannot be present when fallback.used=false"
        )
    return {
        "used": raw_used,
        "count": count,
        "first_reason": raw_reason,
        "reasons": reasons,
    }


def _validate_receipt_shape(receipt: Mapping[str, Any]) -> None:
    """Validate receipt type, keys, schema version, and claim boundary."""
    if not isinstance(receipt, Mapping):
        raise ObstacleForceDiagnosticReceiptError("receipt must be a mapping")
    if set(receipt) != _RECEIPT_FIELDS:
        missing = sorted(_RECEIPT_FIELDS - set(receipt))
        extra = sorted(set(receipt) - _RECEIPT_FIELDS)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ObstacleForceDiagnosticReceiptError(", ".join(details))
    if receipt["schema_version"] != OBSTACLE_FORCE_DIAGNOSTIC_RECEIPT_SCHEMA_VERSION:
        raise ObstacleForceDiagnosticReceiptError("unsupported diagnostic receipt schema")
    if receipt["claim_boundary"] != "diagnostic_only":
        raise ObstacleForceDiagnosticReceiptError("claim_boundary must be diagnostic_only")


def _validate_receipt_selection(
    receipt: Mapping[str, Any],
) -> tuple[str, str, str, str | None]:
    """Validate selection/application fields and return normalized identity inputs.

    Returns:
        Site, resolved law, selection source, and optional parameter digest.
    """
    site = _non_empty_text(receipt["site"], field="site")
    law_version = _non_empty_text(receipt["law_version"], field="law_version")
    if law_version not in _LAW_VERSIONS:
        raise ObstacleForceDiagnosticReceiptError(f"unsupported law_version: {law_version!r}")
    selection_source = _non_empty_text(receipt["selection_source"], field="selection_source")
    if selection_source not in _SELECTION_SOURCES:
        raise ObstacleForceDiagnosticReceiptError(
            f"unsupported selection_source: {selection_source!r}"
        )

    parameters_sha256 = receipt["parameters_sha256"]
    if parameters_sha256 is not None and not _is_sha256(parameters_sha256):
        raise ObstacleForceDiagnosticReceiptError("parameters_sha256 must be a SHA-256 digest")
    for field in ("enabled", "applied"):
        if not isinstance(receipt[field], bool):
            raise ObstacleForceDiagnosticReceiptError(f"{field} must be a boolean")
    return site, law_version, selection_source, parameters_sha256


def _validate_receipt_input_identity(
    receipt: Mapping[str, Any],
    *,
    site: str,
    law_version: str,
    selection_source: str,
    parameters_sha256: str | None,
) -> dict[str, str]:
    """Validate input identity fields and their deterministic digest.

    Returns:
        Normalized configuration, source, and input digest fields.
    """
    input_identity = receipt["input_identity"]
    if not isinstance(input_identity, Mapping):
        raise ObstacleForceDiagnosticReceiptError("input_identity must be a mapping")
    if set(input_identity) != _INPUT_IDENTITY_FIELDS:
        raise ObstacleForceDiagnosticReceiptError("input_identity fields are incomplete or unknown")
    config_hash = _non_empty_text(input_identity["config_hash"], field="input_identity.config_hash")
    source_commit = _non_empty_text(
        input_identity["source_commit"], field="input_identity.source_commit"
    )
    input_sha256 = input_identity["input_sha256"]
    if not _is_sha256(input_sha256):
        raise ObstacleForceDiagnosticReceiptError(
            "input_identity.input_sha256 must be a SHA-256 digest"
        )
    expected_input_digest = _input_digest(
        config_hash=config_hash,
        source_commit=source_commit,
        site=site,
        law_version=law_version,
        selection_source=selection_source,
        parameters_sha256=parameters_sha256,
    )
    if input_sha256 != expected_input_digest:
        raise ObstacleForceDiagnosticReceiptError(
            "input_identity.input_sha256 does not match receipt"
        )
    return {
        "config_hash": config_hash,
        "source_commit": source_commit,
        "input_sha256": input_sha256,
    }


def validate_obstacle_force_diagnostic_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one receipt and return a shallow normalized copy.

    This validator intentionally checks only the diagnostic contract. It does not
    infer numerical correctness, physical suitability, safety, social behavior,
    benchmark status, or domain approval.

    Returns:
        A shallow normalized copy of the validated receipt.
    """

    _validate_receipt_shape(receipt)
    site, law_version, selection_source, parameters_sha256 = _validate_receipt_selection(receipt)
    fallback = _normalize_fallback(receipt["fallback"])
    input_identity = _validate_receipt_input_identity(
        receipt,
        site=site,
        law_version=law_version,
        selection_source=selection_source,
        parameters_sha256=parameters_sha256,
    )

    normalized = dict(receipt)
    normalized["site"] = site
    normalized["law_version"] = law_version
    normalized["selection_source"] = selection_source
    normalized["fallback"] = fallback
    normalized["input_identity"] = input_identity
    _json_bytes(normalized, field="receipt")
    return normalized


def build_obstacle_force_diagnostic_receipt(
    metadata: Mapping[str, Any],
    *,
    config_hash: str,
    source_commit: str,
    fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate a compact diagnostic receipt from law metadata.

    Returns:
        A JSON-safe diagnostic-only receipt.
    """

    if not isinstance(metadata, Mapping):
        raise ObstacleForceDiagnosticReceiptError("obstacle-force metadata must be a mapping")
    site = _non_empty_text(metadata.get("site"), field="site")
    law_version = _non_empty_text(metadata.get("law_version"), field="law_version")
    selection_source = _non_empty_text(metadata.get("resolution_mode"), field="resolution_mode")
    enabled = metadata.get("enabled")
    applied = metadata.get("applied")
    if not isinstance(enabled, bool) or not isinstance(applied, bool):
        raise ObstacleForceDiagnosticReceiptError("enabled and applied must be booleans")

    parameters_sha256 = metadata.get("parameters_sha256")
    parameters = metadata.get("parameters")
    if parameters is not None:
        if not isinstance(parameters, Mapping):
            raise ObstacleForceDiagnosticReceiptError("parameters must be a mapping")
        expected_parameters_digest = _parameters_digest(parameters)
        if parameters_sha256 != expected_parameters_digest:
            raise ObstacleForceDiagnosticReceiptError(
                "parameters_sha256 does not match the JSON-safe parameters"
            )
    elif parameters_sha256 is not None and not _is_sha256(parameters_sha256):
        raise ObstacleForceDiagnosticReceiptError("parameters_sha256 must be a SHA-256 digest")
    if parameters_sha256 is not None:
        parameters_sha256 = _non_empty_text(parameters_sha256, field="parameters_sha256")

    config_hash = _non_empty_text(config_hash, field="config_hash")
    source_commit = _non_empty_text(source_commit, field="source_commit")
    receipt = {
        "schema_version": OBSTACLE_FORCE_DIAGNOSTIC_RECEIPT_SCHEMA_VERSION,
        "claim_boundary": "diagnostic_only",
        "site": site,
        "law_version": law_version,
        "selection_source": selection_source,
        "parameters_sha256": parameters_sha256,
        "enabled": enabled,
        "applied": applied,
        "fallback": _normalize_fallback(fallback or metadata),
        "input_identity": {
            "config_hash": config_hash,
            "source_commit": source_commit,
            "input_sha256": _input_digest(
                config_hash=config_hash,
                source_commit=source_commit,
                site=site,
                law_version=law_version,
                selection_source=selection_source,
                parameters_sha256=parameters_sha256,
            ),
        },
    }
    return validate_obstacle_force_diagnostic_receipt(receipt)


def attach_obstacle_force_diagnostic_receipt(
    metadata: Mapping[str, Any],
    *,
    config_hash: str,
    source_commit: str,
    fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return law metadata with a validated diagnostic receipt attached."""

    result = dict(metadata)
    result["diagnostic_receipt"] = build_obstacle_force_diagnostic_receipt(
        metadata,
        config_hash=config_hash,
        source_commit=source_commit,
        fallback=fallback,
    )
    return result


__all__ = [
    "OBSTACLE_FORCE_DIAGNOSTIC_RECEIPT_SCHEMA_VERSION",
    "ObstacleForceDiagnosticReceiptError",
    "attach_obstacle_force_diagnostic_receipt",
    "build_obstacle_force_diagnostic_receipt",
    "obstacle_force_fallback_from_mapping",
    "validate_obstacle_force_diagnostic_receipt",
]
