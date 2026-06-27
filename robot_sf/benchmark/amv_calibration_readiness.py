"""Diagnostic readiness/preflight checker for AMV actuation-envelope calibration inputs.

Issue #1559 keeps hardware-calibrated and paper-facing AMV (autonomous micromobility vehicle)
actuation use **blocked** until a documented, provenance-backed source replaces the placeholder
calibration skeleton. The accepted proxy-source decision lives in #1585/#2001 (a non-hardware
e-scooter platform-class proxy); a real hardware trace is tracked in #2000.

This module does **not** calibrate from data, tune envelope values, or run campaigns. It only
*inspects* a candidate calibrated-actuation profile (such as the
``configs/benchmarks/issue_1586_calibrated_actuation_profile_skeleton_v0.yaml`` skeleton) and reports
whether its calibration inputs are ``ready`` or ``blocked``, fail-closed.

The gap this fills: :func:`robot_sf.benchmark.synthetic_actuation.validate_actuation_profile_claim_boundary`
checks that provenance fields are *present and non-empty*, but the skeleton fills them with
placeholder sentinels (``source_id: "pending-#1585"``, ``measurement_date: "pending"``, ...). Such a
profile passes structural validation while remaining unfit for calibrated or paper-facing use.
Readiness assessment additionally detects those placeholder sentinels and the proxy-vs-hardware
distinction, so blocked external artifacts are never mistaken for calibration evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.synthetic_actuation import (
    CALIBRATED_ACTUATION_REQUIRED_PROVENANCE_FIELDS,
    looks_calibrated_actuation_profile,
    missing_calibrated_provenance_fields,
    validate_actuation_profile_claim_boundary,
)

# Substrings that mark a provenance value as a placeholder / not-yet-staged external artifact.
# Matched case-insensitively against required provenance string values. These signal that the
# calibration source has not actually been staged, so the profile must fail closed.
PLACEHOLDER_PROVENANCE_MARKERS: tuple[str, ...] = (
    "pending",
    "placeholder",
    "tbd",
    "todo",
    "unknown",
    "n/a",
    "not-for-benchmark",
    "not_for_benchmark",
    "external-trace-collection-pending",
    "<record-id>",
    "blocked",
)

# Source-type / source-id substrings that indicate a non-hardware proxy. A proxy source (e.g. the
# accepted #1585 e-scooter platform-class proxy) may unblock calibrated *exploratory* work but is
# never sufficient for paper-facing claims, which require a real hardware trace (#2000) or an
# official platform spec.
PROXY_SOURCE_MARKERS: tuple[str, ...] = (
    "proxy",
    "platform-class",
    "platform_class",
    "synthetic",
    "estimate",
)

# Source-type substrings that indicate a genuine hardware-calibrated or official-spec source. Only
# these make a profile eligible for paper-facing discussion (still subject to maintainer review).
PAPER_FACING_SOURCE_MARKERS: tuple[str, ...] = (
    "hardware",
    "measured-trace",
    "measured_trace",
    "official-spec",
    "official_spec",
    "manufacturer-spec",
    "manufacturer_spec",
)


def _scan_placeholder_fields(provenance: Mapping[str, Any]) -> list[str]:
    """Return required provenance fields whose string value contains a placeholder marker."""
    placeholder_fields: list[str] = []
    for field_name in CALIBRATED_ACTUATION_REQUIRED_PROVENANCE_FIELDS:
        value = provenance.get(field_name)
        if not isinstance(value, str):
            continue
        lowered = value.strip().lower()
        if any(marker in lowered for marker in PLACEHOLDER_PROVENANCE_MARKERS):
            placeholder_fields.append(field_name)
    return placeholder_fields


def _source_uri_is_tracking_issue(provenance: Mapping[str, Any]) -> bool:
    """Return whether ``source_uri`` points at a GitHub issue rather than a durable artifact."""
    source_uri = provenance.get("source_uri")
    if not isinstance(source_uri, str):
        return False
    lowered = source_uri.strip().lower()
    return "github.com" in lowered and "/issues/" in lowered


def _classify_source(provenance: Mapping[str, Any]) -> str:
    """Classify the provenance source class.

    Returns:
        ``"hardware"``, ``"proxy"``, or ``"unknown"`` based on source-type/id/claim markers.
    """
    tokens = " ".join(
        str(provenance.get(key, "")).lower()
        for key in ("source_type", "source_id", "claim_boundary")
    )
    if any(marker in tokens for marker in PAPER_FACING_SOURCE_MARKERS):
        return "hardware"
    if any(marker in tokens for marker in PROXY_SOURCE_MARKERS):
        return "proxy"
    return "unknown"


@dataclass(frozen=True)
class AmvCalibrationReadiness:
    """Outcome of an AMV actuation-envelope calibration-input readiness check.

    ``status`` is ``"ready"`` only when the profile is calibrated-labeled, structurally valid, has
    every required provenance field populated, and carries no placeholder sentinel. Anything else is
    ``"blocked"`` (fail-closed). ``paper_facing_allowed`` is stricter still: it additionally requires
    a hardware/official-spec source class, because the accepted proxy source is not paper-facing.
    """

    status: str
    profile_name: str | None
    claim_scope: str | None
    source_class: str
    paper_facing_allowed: bool
    looks_calibrated: bool
    missing_provenance_fields: tuple[str, ...] = field(default_factory=tuple)
    placeholder_fields: tuple[str, ...] = field(default_factory=tuple)
    blocking_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_ready(self) -> bool:
        """Whether the calibration inputs are ready (non-blocked)."""
        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the readiness outcome."""
        return asdict(self)


def assess_amv_calibration_readiness(
    profile_metadata: Any,
    *,
    label: str = "synthetic_actuation_profile",
) -> AmvCalibrationReadiness:
    """Assess whether an actuation-profile's calibration inputs are ready (fail-closed).

    Args:
        profile_metadata: The ``synthetic_actuation_profile`` mapping (calibrated or synthetic).
        label: Diagnostic label used in structural-validation error messages.

    Returns:
        An :class:`AmvCalibrationReadiness`. The result is ``blocked`` for any of: non-mapping input,
        synthetic-only profiles (not calibration candidates), structural/conflation errors, missing
        required provenance fields, or placeholder/pending provenance values.
    """
    if not isinstance(profile_metadata, Mapping):
        return AmvCalibrationReadiness(
            status="blocked",
            profile_name=None,
            claim_scope=None,
            source_class="unknown",
            paper_facing_allowed=False,
            looks_calibrated=False,
            blocking_reasons=("profile metadata is missing or not a mapping",),
        )

    profile_name = profile_metadata.get("name")
    claim_scope = profile_metadata.get("claim_scope")
    profile_name = str(profile_name) if profile_name is not None else None
    claim_scope = str(claim_scope) if claim_scope is not None else None

    looks_calibrated = looks_calibrated_actuation_profile(profile_metadata)

    # Structural validation owns conflation detection and required-field error semantics. Capture
    # any failure but defer the verdict so the readiness report can also surface explicit
    # missing/placeholder field lists rather than only a generic error string.
    try:
        validate_actuation_profile_claim_boundary(profile_metadata, label=label)
        structural_error: str | None = None
    except (ValueError, TypeError) as exc:
        structural_error = str(exc)

    if not looks_calibrated:
        # A purely synthetic diagnostic profile is intentionally not a calibration candidate; report
        # blocked so it can never be conflated with calibrated evidence.
        reason = (
            f"structural validation failed: {structural_error}"
            if structural_error is not None
            else "profile is synthetic-only (not calibrated-labeled); no calibration source to assess"
        )
        return AmvCalibrationReadiness(
            status="blocked",
            profile_name=profile_name,
            claim_scope=claim_scope,
            source_class="unknown",
            paper_facing_allowed=False,
            looks_calibrated=False,
            blocking_reasons=(reason,),
        )

    provenance = profile_metadata.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}

    missing = tuple(missing_calibrated_provenance_fields(profile_metadata))
    placeholder_fields = tuple(_scan_placeholder_fields(provenance))
    source_class = _classify_source(provenance)

    reasons: list[str] = []
    # Surface non-missing structural failures (e.g. conflation) first; a pure missing-provenance
    # error is reported through the explicit ``missing`` list below to avoid duplicate messaging.
    if structural_error is not None and "provenance fields" not in structural_error:
        reasons.append(f"structural validation failed: {structural_error}")
    if missing:
        reasons.append(f"missing required provenance fields: {', '.join(missing)}")
    if placeholder_fields:
        reasons.append("placeholder/pending provenance values in: " + ", ".join(placeholder_fields))
    if _source_uri_is_tracking_issue(provenance):
        reasons.append("source_uri points at a tracking issue, not a durable artifact")

    status = "blocked" if reasons else "ready"
    paper_facing_allowed = status == "ready" and source_class == "hardware"

    return AmvCalibrationReadiness(
        status=status,
        profile_name=profile_name,
        claim_scope=claim_scope,
        source_class=source_class,
        paper_facing_allowed=paper_facing_allowed,
        looks_calibrated=True,
        missing_provenance_fields=missing,
        placeholder_fields=placeholder_fields,
        blocking_reasons=tuple(reasons),
    )


def assess_amv_calibration_readiness_from_config(
    config_path: str | Path,
    *,
    profile_key: str = "synthetic_actuation_profile",
) -> AmvCalibrationReadiness:
    """Load a benchmark config and assess the readiness of its actuation calibration profile.

    Args:
        config_path: Path to a YAML benchmark config (e.g. the #1586 calibrated-profile skeleton).
        profile_key: Top-level key holding the actuation profile mapping.

    Returns:
        An :class:`AmvCalibrationReadiness`. A missing config file or missing profile key is reported
        as ``blocked`` rather than raising, so the checker fails closed on incomplete inputs.
    """
    path = Path(config_path)
    if not path.is_file():
        return AmvCalibrationReadiness(
            status="blocked",
            profile_name=None,
            claim_scope=None,
            source_class="unknown",
            paper_facing_allowed=False,
            looks_calibrated=False,
            blocking_reasons=(f"config file not found: {path}",),
        )

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profile_metadata = payload.get(profile_key) if isinstance(payload, Mapping) else None
    if profile_metadata is None:
        return AmvCalibrationReadiness(
            status="blocked",
            profile_name=None,
            claim_scope=None,
            source_class="unknown",
            paper_facing_allowed=False,
            looks_calibrated=False,
            blocking_reasons=(f"config has no '{profile_key}' profile to assess",),
        )

    return assess_amv_calibration_readiness(profile_metadata, label=profile_key)
