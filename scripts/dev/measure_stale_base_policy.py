#!/usr/bin/env python3
"""Measure a frozen risk-tiered stale-base observation window.

The command consumes an explicit ``stale_base_observation_window.v1`` JSON
snapshot. It never queries GitHub, infers causal latency from a live queue, or
changes the stale-base policy. Missing or incompatible source evidence remains
visible as ``not_available`` rather than becoming a zero or a success claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WINDOW_SCHEMA_VERSION = "stale_base_observation_window.v1"
REPORT_SCHEMA_VERSION = "stale_base_measurement_report.v1"
CLAIM_BOUNDARY = (
    "Workflow observation only. This report does not establish benchmark, planner, "
    "scientific, paper-facing, or causal research evidence."
)
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EVIDENCE_STATUSES = frozenset({"workflow_observation", "workflow_fixture"})
SOURCE_KINDS = frozenset({"fixture", "repository_snapshot", "external_snapshot"})
RISK_TIERS = ("ordinary", "base_sensitive", "unknown")
WAIT_TYPES = frozenset({"ordinary_cas", "base_sensitive_refresh", "none", "unknown"})
ATTRIBUTIONS = frozenset({"stale_base", "not_attributable", "unknown"})
HOLD_WAIT_TYPES = frozenset({"ordinary_cas", "base_sensitive_refresh"})
RED_MAIN_COVERAGE_STATUSES = frozenset({"complete", "not_available", "unknown"})
SEMANTICS_ID = "risk-tiered-stale-base.v1"


def _mapping(value: Any) -> dict[str, Any] | None:
    """Return a mapping value or ``None``."""
    return value if isinstance(value, dict) else None


def _parse_timestamp(
    value: Any,
    *,
    field: str,
    errors: list[str],
    required: bool = False,
) -> datetime | None:
    """Parse an explicitly timezone-aware ISO-8601 timestamp."""
    if value in (None, ""):
        if required:
            errors.append(f"{field} is required")
        return None
    if not isinstance(value, str):
        errors.append(f"{field} must be an ISO-8601 string")
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        errors.append(f"{field} is not a valid ISO-8601 timestamp")
        return None
    if parsed.tzinfo is None:
        errors.append(f"{field} must include a timezone")
        return None
    return parsed.astimezone(UTC)


def _sha(
    value: Any,
    *,
    field: str,
    errors: list[str],
    length: int = 40,
    required: bool = False,
) -> str | None:
    """Validate a lowercase hexadecimal SHA value."""
    if value in (None, ""):
        if required:
            errors.append(f"{field} is required")
        return None
    if not isinstance(value, str):
        errors.append(f"{field} must be a lowercase hexadecimal SHA")
        return None
    pattern = SHA1_RE if length == 40 else SHA256_RE
    if pattern.fullmatch(value) is None:
        errors.append(f"{field} must be a {length}-character lowercase hexadecimal SHA")
        return None
    return value


def _file_sha256(path: Path) -> str | None:
    """Return a local file digest, or ``None`` when it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _content_addressed_digest(locator: str) -> str | None:
    """Extract a digest from a content-addressed external locator."""
    direct = re.fullmatch(r"sha256://([0-9a-f]{64})", locator)
    if direct is not None:
        return direct.group(1)
    suffix = re.search(r"(?:[?#])sha256=([0-9a-f]{64})$", locator)
    return suffix.group(1) if suffix is not None else None


def _verify_repository_snapshot(
    *, path: str, digest: str | None, prefix: str
) -> tuple[bool, str, list[str], list[str]]:
    """Verify a repository-local source snapshot."""
    local_path = Path(path)
    actual_digest = _file_sha256(local_path) if local_path.is_file() else None
    if actual_digest is None:
        return (
            False,
            "unverified",
            [],
            [f"{prefix} repository snapshot is not locally verifiable: {path}"],
        )
    if digest is not None and actual_digest != digest:
        return (
            False,
            "digest_mismatch",
            [f"{prefix}.sha256 does not match the repository snapshot contents"],
            [],
        )
    return True, "verified", [], []


def _verify_external_snapshot(
    *, path: str, digest: str | None, prefix: str
) -> tuple[bool, str, list[str], list[str]]:
    """Verify a content-addressed external source snapshot."""
    locator_digest = _content_addressed_digest(path)
    if locator_digest is None:
        return (
            False,
            "unverified",
            [],
            [f"{prefix} external snapshot lacks a content-addressed immutable locator"],
        )
    if digest is not None and locator_digest != digest:
        return (
            False,
            "digest_mismatch",
            [f"{prefix}.sha256 does not match its immutable locator"],
            [],
        )
    return True, "verified", [], []


def _verify_available_source(
    *,
    source_kind: str,
    path: str,
    digest: str | None,
    evidence_status: str,
    prefix: str,
) -> tuple[bool, str, list[str], list[str]]:
    """Verify an available source according to its declared source kind."""
    if source_kind == "fixture":
        errors = (
            []
            if path.startswith("fixture://")
            else [f"{prefix}.fixture source must use a fixture:// path"]
        )
        if evidence_status == "workflow_observation":
            errors.append(f"{prefix} fixture source cannot support workflow_observation")
        return path.startswith("fixture://"), "synthetic", errors, []
    if path.startswith("fixture://"):
        return False, "unverified", [f"{prefix}.fixture:// path must have source_kind=fixture"], []
    if source_kind == "repository_snapshot":
        return _verify_repository_snapshot(path=path, digest=digest, prefix=prefix)
    if source_kind == "external_snapshot":
        return _verify_external_snapshot(path=path, digest=digest, prefix=prefix)
    return False, "unverified", [], []


def _verify_source(
    *,
    source_kind: str,
    path: str,
    digest: str | None,
    available: bool,
    evidence_status: str,
    prefix: str,
) -> tuple[bool, str, list[str], list[str]]:
    """Verify one source locator and return status, errors, and unavailable reasons."""
    errors: list[str] = []
    unavailable: list[str] = []
    if not available:
        unavailable.append(f"{prefix} source is marked unavailable: {path or '<missing>'}")
        verified, verification_status = False, "unverified"
    else:
        verified, verification_status, errors, unavailable = _verify_available_source(
            source_kind=source_kind,
            path=path,
            digest=digest,
            evidence_status=evidence_status,
            prefix=prefix,
        )
    if available and not verified and not errors and not unavailable and source_kind != "fixture":
        unavailable.append(f"{prefix} source could not be independently verified")
    return verified, verification_status, errors, unavailable


def _validate_source_snapshots(
    value: Any,
    *,
    field: str,
    errors: list[str],
    unavailable: list[str],
    evidence_status: str,
) -> list[dict[str, Any]]:
    """Validate source kind, digest, and whether the source is independently verifiable."""
    if not isinstance(value, list) or not value:
        errors.append(f"{field} must contain at least one source snapshot")
        return []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        source = _mapping(raw)
        prefix = f"{field}[{index}]"
        if source is None:
            errors.append(f"{prefix} must be an object")
            continue
        source_kind = source.get("source_kind")
        if source_kind not in SOURCE_KINDS:
            errors.append(f"{prefix}.source_kind must be one of {sorted(SOURCE_KINDS)}")
            source_kind = "unknown"
        path = source.get("path")
        if not isinstance(path, str) or not path.strip():
            errors.append(f"{prefix}.path must be a non-empty string")
            path = ""
        digest = _sha(
            source.get("sha256"),
            field=f"{prefix}.sha256",
            errors=errors,
            length=64,
            required=True,
        )
        available = source.get("available", True)
        if not isinstance(available, bool):
            errors.append(f"{prefix}.available must be boolean")
            available = False
        verified, verification_status, source_errors, source_unavailable = _verify_source(
            source_kind=source_kind,
            path=path,
            digest=digest,
            available=available,
            evidence_status=evidence_status,
            prefix=prefix,
        )
        errors.extend(source_errors)
        unavailable.extend(source_unavailable)
        normalized.append(
            {
                "path": path,
                "sha256": digest,
                "source_kind": source_kind,
                "available": available,
                "verified": verified,
                "verification_status": verification_status,
                "role": str(source.get("role", "") or ""),
            }
        )
    return normalized


def _validate_evidence(
    value: Any,
    *,
    field: str,
    errors: list[str],
) -> dict[str, str]:
    """Validate optional exact-head/base evidence fields."""
    if value in (None, {}):
        return {}
    evidence = _mapping(value)
    if evidence is None:
        errors.append(f"{field} must be an object")
        return {}
    normalized: dict[str, str] = {}
    for key, raw in evidence.items():
        if not str(key).endswith("_sha"):
            continue
        parsed = _sha(raw, field=f"{field}.{key}", errors=errors, length=40)
        if parsed is not None:
            normalized[str(key)] = parsed
    return normalized


def _record_duration(
    record: dict[str, Any],
    *,
    field: str,
    errors: list[str],
) -> tuple[float | None, str | None, datetime | None, datetime | None]:
    """Return duration, missingness, and normalized hold boundaries."""
    started = _parse_timestamp(
        record.get("hold_started_at"),
        field=f"{field}.hold_started_at",
        errors=errors,
    )
    ended = _parse_timestamp(
        record.get("hold_ended_at"),
        field=f"{field}.hold_ended_at",
        errors=errors,
    )
    if started is None and ended is None:
        return None, "hold_timestamps_unavailable", None, None
    if started is None or ended is None:
        return None, "hold_timestamps_incomplete", started, ended
    duration = (ended - started).total_seconds()
    if duration < 0:
        errors.append(f"{field}.hold_ended_at must not precede hold_started_at")
        return None, "hold_timestamps_invalid", started, ended
    return round(duration, 6), None, started, ended


def _effective_attribution(
    declared: str,
    evidence: dict[str, str],
) -> tuple[str, str | None]:
    """Accept stale-base attribution only when exact evidence supports it."""
    if declared != "stale_base":
        return declared, None
    required = (
        "head_sha",
        "ci_head_sha",
        "reviewed_head_sha",
        "cas_head_sha",
        "ci_base_sha",
        "current_main_sha",
        "cas_main_sha",
    )
    if any(key not in evidence for key in required):
        return "unknown", "stale_base_attribution_missing_exact_evidence"
    if not (
        evidence["head_sha"]
        == evidence["ci_head_sha"]
        == evidence["reviewed_head_sha"]
        == evidence["cas_head_sha"]
    ):
        return "unknown", "stale_base_attribution_head_mismatch"
    if evidence["ci_base_sha"] == evidence["current_main_sha"]:
        return "unknown", "stale_base_attribution_base_not_stale"
    if evidence["cas_main_sha"] != evidence["current_main_sha"]:
        return "unknown", "stale_base_attribution_cas_main_mismatch"
    return "stale_base", None


def _normalize_record(
    raw: Any,
    *,
    field: str,
    errors: list[str],
) -> dict[str, Any] | None:
    """Validate and normalize one PR observation record."""
    record = _mapping(raw)
    if record is None:
        errors.append(f"{field} must be an object")
        return None
    pr_number = record.get("pr_number")
    if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number <= 0:
        errors.append(f"{field}.pr_number must be a positive integer")
        return None
    risk_tier = record.get("risk_tier")
    if risk_tier not in RISK_TIERS:
        errors.append(f"{field}.risk_tier must be one of {sorted(RISK_TIERS)}")
        risk_tier = "unknown"
    wait_type = record.get("wait_type")
    if wait_type not in WAIT_TYPES:
        errors.append(f"{field}.wait_type must be one of {sorted(WAIT_TYPES)}")
        wait_type = "unknown"
    attribution = record.get("attribution")
    if attribution not in ATTRIBUTIONS:
        errors.append(f"{field}.attribution must be one of {sorted(ATTRIBUTIONS)}")
        attribution = "unknown"
    if wait_type == "ordinary_cas" and risk_tier != "ordinary":
        errors.append(f"{field}.ordinary_cas requires risk_tier=ordinary")
    if wait_type == "base_sensitive_refresh" and risk_tier != "base_sensitive":
        errors.append(f"{field}.base_sensitive_refresh requires risk_tier=base_sensitive")
    if attribution == "stale_base" and wait_type not in HOLD_WAIT_TYPES:
        errors.append(f"{field}.stale_base attribution requires a hold wait type")
    evidence = _validate_evidence(record.get("evidence"), field=f"{field}.evidence", errors=errors)
    duration, duration_missing, started, ended = _record_duration(
        record, field=field, errors=errors
    )
    effective, attribution_reason = _effective_attribution(attribution, evidence)
    return {
        "pr_number": pr_number,
        "risk_tier": risk_tier,
        "wait_type": wait_type,
        "declared_attribution": attribution,
        "hold_started_at": started.isoformat().replace("+00:00", "Z") if started else None,
        "hold_ended_at": ended.isoformat().replace("+00:00", "Z") if ended else None,
        "duration": duration,
        "duration_missing": duration_missing,
        "effective_attribution": effective,
        "attribution_reason": attribution_reason,
        "evidence": evidence,
    }


def _validate_records(
    value: Any,
    *,
    field: str,
    errors: list[str],
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> list[dict[str, Any]]:
    """Validate and normalize PR observation records."""
    if not isinstance(value, list):
        errors.append(f"{field} must be a list")
        return []
    normalized: list[dict[str, Any]] = []
    seen_prs: set[int] = set()
    for index, raw in enumerate(value):
        record = _normalize_record(raw, field=f"{field}[{index}]", errors=errors)
        if record is None:
            continue
        pr_number = int(record["pr_number"])
        if pr_number in seen_prs:
            errors.append(f"{field}[{index}].pr_number duplicates {pr_number}")
            continue
        seen_prs.add(pr_number)
        if record["wait_type"] in HOLD_WAIT_TYPES:
            started = _parse_timestamp(
                record["hold_started_at"],
                field=f"{field}[{index}].hold_started_at",
                errors=[],
            )
            ended = _parse_timestamp(
                record["hold_ended_at"],
                field=f"{field}[{index}].hold_ended_at",
                errors=[],
            )
            if window_start is not None and started is not None and started < window_start:
                errors.append(f"{field}[{index}].hold_started_at falls outside the named window")
            if window_end is not None and ended is not None and ended > window_end:
                errors.append(f"{field}[{index}].hold_ended_at falls outside the named window")
        normalized.append(record)
    return normalized


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return a deterministic nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return ordered[rank - 1]


def _empty_tier_metrics() -> dict[str, Any]:
    return {
        "records": 0,
        "holds": 0,
        "ordinary_cas_waits": 0,
        "base_sensitive_refresh_waits": 0,
        "stale_base_holds": 0,
        "unknown_attribution": 0,
        "missing_wait_duration": 0,
        "wait_seconds_denominator": 0,
        "p50_wait_seconds": None,
        "p95_wait_seconds": None,
    }


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize normalized records without treating unknowns as successes."""
    by_tier = {tier: _empty_tier_metrics() for tier in RISK_TIERS}
    durations: dict[str, list[float]] = {tier: [] for tier in RISK_TIERS}
    for record in records:
        tier = str(record["risk_tier"])
        metrics = by_tier[tier]
        metrics["records"] += 1
        wait_type = str(record["wait_type"])
        if wait_type in HOLD_WAIT_TYPES:
            metrics["holds"] += 1
        if wait_type == "ordinary_cas":
            metrics["ordinary_cas_waits"] += 1
        elif wait_type == "base_sensitive_refresh":
            metrics["base_sensitive_refresh_waits"] += 1
        attribution = str(record["effective_attribution"])
        is_hold = wait_type in HOLD_WAIT_TYPES
        if is_hold and record["duration"] is None:
            metrics["missing_wait_duration"] += 1
        if is_hold and attribution == "stale_base":
            metrics["stale_base_holds"] += 1
            duration = record["duration"]
            if duration is not None:
                metrics["wait_seconds_denominator"] += 1
                durations[tier].append(float(duration))
        elif is_hold and attribution == "unknown":
            metrics["unknown_attribution"] += 1
    for tier, metrics in by_tier.items():
        metrics["p50_wait_seconds"] = _percentile(durations[tier], 50.0)
        metrics["p95_wait_seconds"] = _percentile(durations[tier], 95.0)
    return {
        "records_total": len(records),
        "by_risk_tier": by_tier,
        "wait_distribution": {
            "status": "available" if any(durations.values()) else "not_available",
            "method": "nearest_rank",
            "denominator": sum(len(values) for values in durations.values()),
        },
    }


def _observations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deterministic, missingness-preserving record audit trail."""
    return [
        {
            "pr_number": record["pr_number"],
            "risk_tier": record["risk_tier"],
            "wait_type": record["wait_type"],
            "hold_started_at": record["hold_started_at"],
            "hold_ended_at": record["hold_ended_at"],
            "duration_seconds": record["duration"],
            "duration_missing": record["duration_missing"],
            "declared_attribution": record["declared_attribution"],
            "effective_attribution": record["effective_attribution"],
            "attribution_reason": record["attribution_reason"],
            "evidence": record["evidence"],
        }
        for record in sorted(records, key=lambda item: int(item["pr_number"]))
    ]


def _classify_red_main_event(
    event: dict[str, Any],
    *,
    evidence: dict[str, str],
    occurred_at: datetime | None,
    in_window: bool,
) -> tuple[str, str]:
    """Classify a red-main event from exact head/base evidence only."""
    if occurred_at is None:
        return "unknown", "red-main event timestamp is missing"
    if not in_window:
        return "unknown", "red-main event falls outside the named window"
    if event.get("red_main_observed") is False:
        return "not_attributable", "red-main was not observed for this event"
    if event.get("red_main_observed") is not True:
        return "unknown", "red-main observation flag is missing or invalid"
    required = ("pr_head_sha", "ci_head_sha", "merge_head_sha", "ci_base_sha", "merge_base_sha")
    if any(key not in evidence for key in required):
        return "unknown", "red-main exact head/base evidence is missing"
    values = {key: evidence[key] for key in required}
    if any(SHA1_RE.fullmatch(value) is None for value in values.values()):
        return "unknown", "red-main exact head/base evidence is invalid"
    if not (values["pr_head_sha"] == values["ci_head_sha"] == values["merge_head_sha"]):
        return "unknown", "red-main PR head differs across CI or merge evidence"
    if values["ci_base_sha"] != values["merge_base_sha"]:
        return "stale_base_attributable", "CI base differs from the merge base"
    return "not_attributable", "CI base matches the merge base"


def _red_main_unavailable(reason: str) -> dict[str, Any]:
    """Return an explicit non-zero-looking red-main unavailable state."""
    return {
        "coverage_status": "not_available",
        "coverage_reason": reason,
        "source_snapshots": [],
        "events_total": 0,
        "by_classification": {
            "stale_base_attributable": 0,
            "not_attributable": 0,
            "unknown": 0,
        },
        "events": [],
        "rollback_condition": "unknown",
    }


def _prepare_red_main_coverage(
    coverage_value: Any,
    events_value: Any,
    *,
    evidence_status: str,
    errors: list[str],
    unavailable: list[str],
) -> tuple[str, str, list[dict[str, Any]], list[Any]]:
    """Validate red-main coverage metadata and return its event payload."""
    coverage = _mapping(coverage_value)
    if coverage is None:
        reason = "red-main coverage was not supplied"
        unavailable.append(reason)
        return "not_available", reason, [], []
    coverage_status = coverage.get("status")
    if coverage_status not in RED_MAIN_COVERAGE_STATUSES:
        errors.append(
            f"red_main_coverage.status must be one of {sorted(RED_MAIN_COVERAGE_STATUSES)}"
        )
        coverage_status = "unknown"
    coverage_sources: list[dict[str, Any]] = []
    if coverage_status == "complete":
        coverage_sources = _validate_source_snapshots(
            coverage.get("source_snapshots"),
            field="red_main_coverage.source_snapshots",
            errors=errors,
            unavailable=unavailable,
            evidence_status=evidence_status,
        )
        if events_value is None:
            errors.append("red_main_events is required when red-main coverage is complete")
            events_value = []
    elif coverage.get("source_snapshots") is not None:
        coverage_sources = _validate_source_snapshots(
            coverage.get("source_snapshots"),
            field="red_main_coverage.source_snapshots",
            errors=errors,
            unavailable=unavailable,
            evidence_status=evidence_status,
        )
    if coverage_status != "complete":
        reason = str(
            coverage.get(
                "reason",
                "red-main coverage is explicitly unavailable"
                if coverage_status == "not_available"
                else "red-main coverage is explicitly unknown",
            )
        )
        if coverage_status == "not_available":
            unavailable.append(reason)
        return coverage_status, reason, coverage_sources, []
    if not isinstance(events_value, list):
        errors.append("red_main_events must be a list when red-main coverage is complete")
        events: list[Any] = []
    else:
        events = events_value
    return "complete", "", coverage_sources, events


def _normalize_red_main_event(
    raw: Any,
    index: int,
    *,
    window_start: datetime | None,
    window_end: datetime | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize one event and retain exact evidence plus classification reasoning."""
    if not isinstance(raw, dict):
        return None, [f"red_main_events[{index}] must be an object"]
    errors: list[str] = []
    incident_id = str(raw.get("incident_id", "")).strip()
    if not incident_id:
        errors.append(f"red_main_events[{index}].incident_id must be non-empty")
        incident_id = f"event-{index}"
    evidence_errors: list[str] = []
    evidence = _validate_evidence(
        raw.get("evidence"),
        field=f"red_main_events[{index}].evidence",
        errors=evidence_errors,
    )
    occurred_at = _parse_timestamp(
        raw.get("occurred_at"),
        field=f"red_main_events[{index}].occurred_at",
        errors=errors,
    )
    in_window = (
        occurred_at is not None
        and (window_start is None or occurred_at >= window_start)
        and (window_end is None or occurred_at <= window_end)
    )
    classification, reason = _classify_red_main_event(
        raw,
        evidence=evidence,
        occurred_at=occurred_at,
        in_window=in_window,
    )
    if evidence_errors:
        classification = "unknown"
        reason = "red-main exact head/base evidence is invalid"
    errors.extend(evidence_errors)
    return {
        "incident_id": incident_id,
        "occurred_at": occurred_at.isoformat().replace("+00:00", "Z") if occurred_at else None,
        "classification": classification,
        "classification_reason": reason,
        "evidence": evidence,
    }, errors


def _summarize_red_main_events(
    events_value: Any,
    coverage_value: Any,
    *,
    evidence_status: str,
    window_start: datetime | None,
    window_end: datetime | None,
    errors: list[str],
    unavailable: list[str],
) -> dict[str, Any]:
    """Summarize red-main classifications while making coverage explicit."""
    coverage_status, coverage_reason, coverage_sources, events = _prepare_red_main_coverage(
        coverage_value,
        events_value,
        evidence_status=evidence_status,
        errors=errors,
        unavailable=unavailable,
    )
    if coverage_status != "complete":
        return {
            **_red_main_unavailable(coverage_reason),
            "coverage_status": coverage_status,
            "source_snapshots": coverage_sources,
        }
    counts = {
        "stale_base_attributable": 0,
        "not_attributable": 0,
        "unknown": 0,
    }
    classifications: list[dict[str, Any]] = []
    seen_incidents: set[str] = set()
    for index, raw in enumerate(events):
        normalized, event_errors = _normalize_red_main_event(
            raw,
            index,
            window_start=window_start,
            window_end=window_end,
        )
        if normalized is None:
            errors.extend(event_errors)
            continue
        errors.extend(event_errors)
        incident_id = normalized["incident_id"]
        if incident_id in seen_incidents:
            errors.append(f"red_main_events[{index}].incident_id duplicates {incident_id}")
        seen_incidents.add(incident_id)
        classification = normalized["classification"]
        counts[classification] += 1
        classifications.append(normalized)
    rollback_condition = (
        "unknown"
        if counts["unknown"]
        else "met"
        if counts["stale_base_attributable"]
        else "not_met"
    )
    return {
        "coverage_status": "complete",
        "coverage_reason": None,
        "source_snapshots": coverage_sources,
        "events_total": len(classifications),
        "by_classification": counts,
        "events": classifications,
        "rollback_condition": rollback_condition,
    }


def _comparison(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    current_evidence_status: str,
) -> dict[str, Any]:
    """Compare compatible aggregate counts and percentiles."""
    if current_evidence_status != "workflow_observation":
        return {
            "status": "fixture_only",
            "reason": "current window is synthetic workflow fixture evidence",
        }
    if baseline.get("status") == "fixture_only":
        return {
            "status": "fixture_only",
            "reason": "pre-rollout baseline is synthetic workflow fixture evidence",
        }
    if baseline.get("status") != "available":
        return {
            "status": "not_available",
            "reason": str(baseline.get("reason", "baseline is unavailable")),
        }
    deltas: dict[str, Any] = {}
    current_tiers = current["by_risk_tier"]
    baseline_tiers = baseline["metrics"]["by_risk_tier"]
    for tier in RISK_TIERS:
        deltas[tier] = {}
        for key in (
            "records",
            "holds",
            "stale_base_holds",
            "wait_seconds_denominator",
        ):
            deltas[tier][key] = current_tiers[tier][key] - baseline_tiers[tier][key]
        for key in ("p50_wait_seconds", "p95_wait_seconds"):
            current_value = current_tiers[tier][key]
            baseline_value = baseline_tiers[tier][key]
            deltas[tier][key] = (
                None
                if current_value is None or baseline_value is None
                else round(current_value - baseline_value, 6)
            )
    return {
        "status": "available",
        "method": "current_minus_pre_rollout_nearest_rank_summary",
        "by_risk_tier": deltas,
    }


def _unavailable_report(
    reason: str,
    *,
    input_path: str | None = None,
    input_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a machine-readable unavailable report for absent inputs."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_schema_version": None,
        "status": "not_available",
        "evidence_status": "workflow_observation",
        "evidence_class": "workflow_only",
        "input_path": input_path,
        "input_sha256": input_sha256,
        "claim_boundary": CLAIM_BOUNDARY,
        "blockers": [reason],
        "validation_errors": [],
        "limitations": [reason],
        "metrics": None,
        "baseline": {"status": "not_available", "reason": "current window is unavailable"},
        "comparison": {"status": "not_available", "reason": "current window is unavailable"},
        "observations": [],
        "red_main": _red_main_unavailable("current window is unavailable"),
    }


def _validate_policy(
    root: dict[str, Any], *, evidence_status: str, errors: list[str], unavailable: list[str]
) -> dict[str, Any]:
    """Validate policy rollout metadata and its immutable source pointer."""
    policy = _mapping(root.get("policy"))
    if policy is None:
        errors.append("policy must be an object")
        policy = {}
    rollout_at = _parse_timestamp(
        policy.get("rollout_at"), field="policy.rollout_at", errors=errors, required=True
    )
    rollout_commit = _sha(
        policy.get("rollout_commit"),
        field="policy.rollout_commit",
        errors=errors,
        length=40,
        required=True,
    )
    source_value = [policy["source_snapshot"]] if "source_snapshot" in policy else []
    sources = _validate_source_snapshots(
        source_value,
        field="policy.source_snapshots",
        errors=errors,
        unavailable=unavailable,
        evidence_status=evidence_status,
    )
    return {
        "raw": policy,
        "rollout_at": rollout_at,
        "rollout_commit": rollout_commit,
        "source_snapshots": sources,
    }


def _validate_window(
    value: Any,
    *,
    field: str,
    expected_kind: str,
    rollout_at: datetime | None,
    evidence_status: str,
    errors: list[str],
    unavailable: list[str],
) -> dict[str, Any]:
    """Validate a named current or pre-rollout observation window."""
    window = _mapping(value)
    if window is None:
        errors.append(f"{field} must be an object")
        window = {}
    window_id = window.get("window_id")
    if not isinstance(window_id, str) or not window_id.strip():
        errors.append(f"{field}.window_id must be a non-empty string")
    kind = window.get("kind")
    if kind != expected_kind:
        errors.append(f"{field}.kind must be {expected_kind}")
    start_at = _parse_timestamp(
        window.get("start_at"), field=f"{field}.start_at", errors=errors, required=True
    )
    end_at = _parse_timestamp(
        window.get("end_at"), field=f"{field}.end_at", errors=errors, required=True
    )
    captured_at = _parse_timestamp(
        window.get("captured_at"),
        field=f"{field}.captured_at",
        errors=errors,
        required=evidence_status == "workflow_observation",
    )
    if start_at is not None and end_at is not None:
        if end_at <= start_at:
            errors.append(f"{field}.end_at must follow {field}.start_at")
        if (
            expected_kind == "normal_throughput"
            and rollout_at is not None
            and start_at < rollout_at
        ):
            errors.append(f"{field}.start_at must not precede policy.rollout_at")
        if expected_kind == "pre_rollout" and rollout_at is not None and end_at > rollout_at:
            errors.append(f"{field}.end_at must not follow policy.rollout_at")
        if captured_at is not None and end_at > captured_at:
            errors.append(f"{field}.end_at must not follow {field}.captured_at")
    sources = _validate_source_snapshots(
        window.get("source_snapshots"),
        field=f"{field}.source_snapshots",
        errors=errors,
        unavailable=unavailable,
        evidence_status=evidence_status,
    )
    return {
        "raw": window,
        "window_id": window_id,
        "kind": kind,
        "start_at": start_at,
        "end_at": end_at,
        "captured_at": captured_at,
        "source_snapshots": sources,
    }


def _validate_contract(root: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate the current-window contract and return normalized components."""
    errors: list[str] = []
    unavailable: list[str] = []
    input_schema = root.get("schema_version")
    if input_schema != WINDOW_SCHEMA_VERSION:
        errors.append(f"schema_version must be {WINDOW_SCHEMA_VERSION}")
    repo = root.get("repo")
    if not isinstance(repo, str) or not repo.strip():
        errors.append("repo must be a non-empty string")
    semantics_id = root.get("semantics_id")
    if semantics_id != SEMANTICS_ID:
        errors.append(f"semantics_id must be {SEMANTICS_ID}")
    evidence_status = root.get("evidence_status")
    if evidence_status not in EVIDENCE_STATUSES:
        errors.append(f"evidence_status must be one of {sorted(EVIDENCE_STATUSES)}")
        evidence_status = "workflow_observation"
    policy = _validate_policy(
        root, evidence_status=evidence_status, errors=errors, unavailable=unavailable
    )
    window = _validate_window(
        root.get("window"),
        field="window",
        expected_kind="normal_throughput",
        rollout_at=policy["rollout_at"],
        evidence_status=evidence_status,
        errors=errors,
        unavailable=unavailable,
    )
    records = _validate_records(
        root.get("records"),
        field="records",
        errors=errors,
        window_start=window["start_at"],
        window_end=window["end_at"],
    )
    red_main = _summarize_red_main_events(
        root.get("red_main_events"),
        root.get("red_main_coverage"),
        evidence_status=evidence_status,
        window_start=window["start_at"],
        window_end=window["end_at"],
        errors=errors,
        unavailable=unavailable,
    )
    return (
        {
            "input_schema": input_schema,
            "repo": repo,
            "semantics_id": semantics_id,
            "evidence_status": evidence_status,
            "policy": policy,
            "window": window,
            "records": records,
            "red_main": red_main,
        },
        errors,
        unavailable,
    )


def _baseline_report(
    value: Any,
    *,
    current_repo: str,
    current_semantics_id: str,
    rollout_at: datetime | None,
) -> dict[str, Any]:
    """Validate and summarize an independent compatible pre-rollout baseline."""
    if value is None:
        return {
            "status": "not_available",
            "reason": "pre-rollout baseline source was not supplied",
        }
    baseline = _mapping(value)
    if baseline is None:
        return {"status": "incompatible", "reason": "baseline must be an object"}
    if baseline.get("schema_version") != WINDOW_SCHEMA_VERSION:
        return {
            "status": "incompatible",
            "reason": f"baseline schema must be {WINDOW_SCHEMA_VERSION}",
        }
    baseline_repo = baseline.get("repo")
    if baseline_repo != current_repo:
        return {
            "status": "incompatible",
            "reason": "baseline repo does not match the current observation repo",
        }
    baseline_semantics_id = baseline.get("semantics_id")
    if baseline_semantics_id != current_semantics_id:
        return {
            "status": "incompatible",
            "reason": "baseline semantics do not match the current observation",
        }
    baseline_evidence_status = baseline.get("evidence_status")
    if baseline_evidence_status not in EVIDENCE_STATUSES:
        return {
            "status": "incompatible",
            "reason": f"baseline evidence_status must be one of {sorted(EVIDENCE_STATUSES)}",
        }
    errors: list[str] = []
    unavailable: list[str] = []
    window = _validate_window(
        baseline.get("window"),
        field="baseline.window",
        expected_kind="pre_rollout",
        rollout_at=rollout_at,
        evidence_status=baseline_evidence_status,
        errors=errors,
        unavailable=unavailable,
    )
    records = _validate_records(
        baseline.get("records"),
        field="baseline.records",
        errors=errors,
        window_start=window["start_at"],
        window_end=window["end_at"],
    )
    if errors:
        return {
            "status": "incompatible",
            "reason": "baseline contract is invalid",
            "validation_errors": errors,
        }
    if unavailable or not records:
        return {
            "status": "not_available",
            "reason": unavailable[0]
            if unavailable
            else "pre-rollout baseline contains no PR records",
            "evidence_status": baseline_evidence_status,
            "source_snapshots": window["source_snapshots"],
        }
    status = "fixture_only" if baseline_evidence_status == "workflow_fixture" else "available"
    return {
        "status": status,
        "evidence_status": baseline_evidence_status,
        "repo": baseline_repo,
        "semantics_id": baseline_semantics_id,
        "window": {
            "window_id": window["window_id"],
            "kind": window["kind"],
            "start_at": window["start_at"].isoformat().replace("+00:00", "Z")
            if window["start_at"]
            else None,
            "end_at": window["end_at"].isoformat().replace("+00:00", "Z")
            if window["end_at"]
            else None,
            "captured_at": window["captured_at"].isoformat().replace("+00:00", "Z")
            if window["captured_at"]
            else None,
            "source_snapshots": window["source_snapshots"],
        },
        "source_snapshots": window["source_snapshots"],
        "metrics": _summarize_records(records),
        "observations": _observations(records),
    }


def _invalid_contract_report(
    *,
    input_schema: Any,
    evidence_status: str,
    input_path: str | None,
    input_sha256: str | None,
    errors: list[str],
    unavailable: list[str],
    observations: list[dict[str, Any]],
    red_main: dict[str, Any],
) -> dict[str, Any]:
    """Build a consistent invalid-contract report."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_schema_version": input_schema,
        "status": "invalid_contract",
        "evidence_status": evidence_status,
        "evidence_class": "workflow_only",
        "input_path": input_path,
        "input_sha256": input_sha256,
        "claim_boundary": CLAIM_BOUNDARY,
        "blockers": ["observation input contract is invalid"],
        "validation_errors": errors,
        "limitations": unavailable,
        "metrics": None,
        "observations": observations,
        "baseline": {"status": "not_available", "reason": "current contract is invalid"},
        "comparison": {"status": "not_available", "reason": "current contract is invalid"},
        "red_main": red_main,
    }


def _path_sha256(input_path: str | None) -> str | None:
    """Compute a digest only for a caller-provided local input path."""
    if not input_path:
        return None
    return _file_sha256(Path(input_path))


def analyze_observation(
    payload: Any,
    *,
    input_path: str | None = None,
    input_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and summarize an observation-window payload."""
    if input_sha256 is None:
        input_sha256 = _path_sha256(input_path)
    root = _mapping(payload)
    if root is None:
        report = _unavailable_report(
            "observation input must be a JSON object",
            input_path=input_path,
            input_sha256=input_sha256,
        )
        report["status"] = "invalid_contract"
        report["validation_errors"] = ["observation input must be a JSON object"]
        return report
    contract, errors, unavailable = _validate_contract(root)
    if errors:
        return _invalid_contract_report(
            input_schema=contract["input_schema"],
            evidence_status=contract["evidence_status"],
            input_path=input_path,
            input_sha256=input_sha256,
            errors=errors,
            unavailable=unavailable,
            observations=_observations(contract["records"]),
            red_main=contract["red_main"],
        )
    metrics = _summarize_records(contract["records"])
    limitations = list(unavailable)
    if not contract["records"]:
        limitations.append("normal-throughput window contains no PR records")
    if contract["red_main"]["coverage_status"] != "complete":
        limitations.append("red-main attribution coverage is not available")
    elif contract["red_main"]["by_classification"].get("unknown", 0):
        limitations.append("red-main events with unknown attribution remain excluded")
    baseline = _baseline_report(
        root.get("baseline"),
        current_repo=contract["repo"],
        current_semantics_id=contract["semantics_id"],
        rollout_at=contract["policy"]["rollout_at"],
    )
    if baseline.get("status") == "not_available":
        limitations.append(str(baseline.get("reason", "pre-rollout baseline unavailable")))
    status = "fixture_only" if contract["evidence_status"] == "workflow_fixture" else "available"
    if unavailable or not contract["records"]:
        status = "not_available"
    policy = contract["policy"]
    window = contract["window"]
    rollout_at = policy["rollout_at"]
    start_at = window["start_at"]
    end_at = window["end_at"]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_schema_version": contract["input_schema"],
        "status": status,
        "evidence_status": contract["evidence_status"],
        "evidence_class": "workflow_only",
        "input_path": input_path,
        "input_sha256": input_sha256,
        "repo": contract["repo"],
        "semantics_id": contract["semantics_id"],
        "policy": {
            "policy_id": str(policy["raw"].get("policy_id", "risk-tiered-stale-base")),
            "rollout_at": rollout_at.isoformat().replace("+00:00", "Z") if rollout_at else None,
            "rollout_commit": policy["rollout_commit"],
            "source_snapshots": policy["source_snapshots"],
        },
        "window": {
            "window_id": window["window_id"],
            "kind": window["kind"],
            "start_at": start_at.isoformat().replace("+00:00", "Z") if start_at else None,
            "end_at": end_at.isoformat().replace("+00:00", "Z") if end_at else None,
            "captured_at": window["captured_at"].isoformat().replace("+00:00", "Z")
            if window["captured_at"]
            else None,
            "source_snapshots": window["source_snapshots"],
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "blockers": [],
        "validation_errors": [],
        "limitations": limitations,
        "metrics": metrics,
        "observations": _observations(contract["records"]),
        "baseline": baseline,
        "comparison": _comparison(
            metrics,
            baseline,
            current_evidence_status=contract["evidence_status"],
        ),
        "red_main": contract["red_main"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-facing report without strengthening claims."""
    lines = [
        "# Risk-tiered stale-base observation report",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Evidence status: `{report.get('evidence_status')}`",
        f"- Claim boundary: {report.get('claim_boundary')}",
        "",
    ]
    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        lines.extend(
            [
                "## Wait summary",
                "",
                "| Risk tier | Records | Holds | Stale-base holds | P50 (s) | P95 (s) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for tier, values in metrics.get("by_risk_tier", {}).items():
            lines.append(
                f"| `{tier}` | {values['records']} | {values['holds']} | "
                f"{values['stale_base_holds']} | {values['p50_wait_seconds']} | "
                f"{values['p95_wait_seconds']} |"
            )
    else:
        lines.extend(["## Wait summary", "", "Current-window metrics are `not_available`."])
    baseline = report.get("baseline")
    if isinstance(baseline, dict):
        lines.extend(["", "## Baseline", "", f"- Status: `{baseline.get('status')}`"])
        if baseline.get("reason"):
            lines.append(f"- Reason: {baseline['reason']}")
    red_main = report.get("red_main")
    if isinstance(red_main, dict):
        lines.extend(
            [
                "",
                "## Red-main classification",
                "",
                f"- Events: {red_main.get('events_total', 0)}",
                f"- Counts: `{json.dumps(red_main.get('by_classification', {}), sort_keys=True)}`",
            ]
        )
    limitations = report.get("limitations") or []
    if limitations:
        lines.extend(["", "## Missingness and limitations", ""])
        lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines) + "\n"


def _write(path: Path, content: str) -> None:
    """Write a generated report to an explicit caller-provided path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="stale_base_observation_window.v1 JSON path")
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    parser.add_argument(
        "--markdown-output", type=Path, help="write a Markdown summary to this path"
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        report = _unavailable_report(
            f"observation input is unavailable: {args.input}", input_path=str(args.input)
        )
    else:
        try:
            input_bytes = args.input.read_bytes()
            input_sha256 = hashlib.sha256(input_bytes).hexdigest()
            payload = json.loads(input_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            report = _unavailable_report(
                f"observation input could not be read: {exc}", input_path=str(args.input)
            )
            report["status"] = "invalid_contract"
            report["validation_errors"] = [str(exc)]
        else:
            report = analyze_observation(
                payload,
                input_path=str(args.input),
                input_sha256=input_sha256,
            )

    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        _write(args.output, serialized)
    if args.markdown_output:
        _write(args.markdown_output, render_markdown(report))
    print(serialized, end="")
    return 0 if report["status"] in {"available", "fixture_only"} else 2


if __name__ == "__main__":
    sys.exit(main())
