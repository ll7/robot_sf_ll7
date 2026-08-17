#!/usr/bin/env python3
"""Measure a frozen risk-tiered stale-base observation window.

The command consumes an explicit ``stale_base_observation_window.v1`` JSON
snapshot. It never queries GitHub, infers causal latency from a live queue, or
changes the stale-base policy. Missing or incompatible source evidence remains
visible as ``not_available`` rather than becoming a zero or a success claim.
"""

from __future__ import annotations

import argparse
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
RISK_TIERS = ("ordinary", "base_sensitive", "unknown")
WAIT_TYPES = frozenset({"ordinary_cas", "base_sensitive_refresh", "none", "unknown"})
ATTRIBUTIONS = frozenset({"stale_base", "not_attributable", "unknown"})
HOLD_WAIT_TYPES = frozenset({"ordinary_cas", "base_sensitive_refresh"})


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


def _validate_source_snapshots(
    value: Any,
    *,
    field: str,
    errors: list[str],
    unavailable: list[str],
) -> list[dict[str, Any]]:
    """Validate source pointers without reading or promoting their contents."""
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
        if not available:
            unavailable.append(f"{prefix} source is marked unavailable: {path or '<missing>'}")
        normalized.append(
            {
                "path": path,
                "sha256": digest,
                "available": available,
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
) -> tuple[float | None, str | None]:
    """Return a hold duration and an explicit missingness reason."""
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
        return None, "hold_timestamps_unavailable"
    if started is None or ended is None:
        return None, "hold_timestamps_incomplete"
    duration = (ended - started).total_seconds()
    if duration < 0:
        errors.append(f"{field}.hold_ended_at must not precede hold_started_at")
        return None, "hold_timestamps_invalid"
    return round(duration, 6), None


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
    evidence = _validate_evidence(record.get("evidence"), field=f"{field}.evidence", errors=errors)
    duration, duration_missing = _record_duration(record, field=field, errors=errors)
    effective, attribution_reason = _effective_attribution(attribution, evidence)
    return {
        "pr_number": pr_number,
        "risk_tier": risk_tier,
        "wait_type": wait_type,
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
        if wait_type in HOLD_WAIT_TYPES and record["duration"] is None:
            metrics["missing_wait_duration"] += 1
        if attribution == "stale_base":
            metrics["stale_base_holds"] += 1
            duration = record["duration"]
            if duration is not None:
                metrics["wait_seconds_denominator"] += 1
                durations[tier].append(float(duration))
        elif attribution == "unknown":
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


def _classify_red_main_event(event: dict[str, Any]) -> str:
    """Classify a red-main event from exact head/base evidence only."""
    if event.get("red_main_observed") is False:
        return "not_attributable"
    if event.get("red_main_observed") is not True:
        return "unknown"
    evidence = event.get("evidence")
    if not isinstance(evidence, dict):
        return "unknown"
    required = ("pr_head_sha", "ci_head_sha", "merge_head_sha", "ci_base_sha", "merge_base_sha")
    if any(not isinstance(evidence.get(key), str) for key in required):
        return "unknown"
    values = {key: evidence[key] for key in required}
    if any(SHA1_RE.fullmatch(value) is None for value in values.values()):
        return "unknown"
    if not (values["pr_head_sha"] == values["ci_head_sha"] == values["merge_head_sha"]):
        return "unknown"
    if values["ci_base_sha"] != values["merge_base_sha"]:
        return "stale_base_attributable"
    return "not_attributable"


def _summarize_red_main_events(value: Any, *, errors: list[str]) -> dict[str, Any]:
    """Summarize red-main classifications with unknowns retained."""
    if value is None:
        events: list[Any] = []
    elif isinstance(value, list):
        events = value
    else:
        errors.append("red_main_events must be a list when supplied")
        events = []
    counts = {
        "stale_base_attributable": 0,
        "not_attributable": 0,
        "unknown": 0,
    }
    classifications: list[dict[str, Any]] = []
    for index, raw in enumerate(events):
        event = _mapping(raw)
        if event is None:
            errors.append(f"red_main_events[{index}] must be an object")
            continue
        classification = _classify_red_main_event(event)
        counts[classification] += 1
        classifications.append(
            {
                "incident_id": str(event.get("incident_id", f"event-{index}")),
                "classification": classification,
            }
        )
    return {
        "events_total": len(classifications),
        "by_classification": counts,
        "events": classifications,
    }


def _comparison(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compare compatible aggregate counts and percentiles."""
    if baseline.get("status") not in {"available", "fixture_only"}:
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


def _unavailable_report(reason: str, *, input_path: str | None = None) -> dict[str, Any]:
    """Build a machine-readable unavailable report for absent inputs."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_schema_version": None,
        "status": "not_available",
        "evidence_status": "workflow_observation",
        "input_path": input_path,
        "claim_boundary": CLAIM_BOUNDARY,
        "blockers": [reason],
        "validation_errors": [],
        "limitations": [reason],
        "metrics": None,
        "baseline": {"status": "not_available", "reason": "current window is unavailable"},
        "comparison": {"status": "not_available", "reason": "current window is unavailable"},
        "red_main": {"events_total": 0, "by_classification": {}, "events": []},
    }


def _validate_policy(
    root: dict[str, Any], *, errors: list[str], unavailable: list[str]
) -> dict[str, Any]:
    """Validate policy rollout metadata and its immutable source pointer."""
    policy = _mapping(root.get("policy"))
    if policy is None:
        errors.append("policy must be an object")
        policy = {}
    rollout_at = _parse_timestamp(
        policy.get("rollout_at"), field="policy.rollout_at", errors=errors, required=True
    )
    source_value = [policy["source_snapshot"]] if "source_snapshot" in policy else []
    sources = _validate_source_snapshots(
        source_value,
        field="policy.source_snapshots",
        errors=errors,
        unavailable=unavailable,
    )
    return {
        "raw": policy,
        "rollout_at": rollout_at,
        "source_snapshots": sources,
    }


def _validate_window(
    root: dict[str, Any],
    *,
    rollout_at: datetime | None,
    errors: list[str],
    unavailable: list[str],
) -> dict[str, Any]:
    """Validate one named normal-throughput window."""
    window = _mapping(root.get("window"))
    if window is None:
        errors.append("window must be an object")
        window = {}
    kind = window.get("kind")
    if kind != "normal_throughput":
        errors.append("window.kind must be normal_throughput")
    start_at = _parse_timestamp(
        window.get("start_at"), field="window.start_at", errors=errors, required=True
    )
    end_at = _parse_timestamp(
        window.get("end_at"), field="window.end_at", errors=errors, required=True
    )
    if start_at is not None and end_at is not None:
        if end_at <= start_at:
            errors.append("window.end_at must follow window.start_at")
        if rollout_at is not None and start_at < rollout_at:
            errors.append("window.start_at must not precede policy.rollout_at")
    sources = _validate_source_snapshots(
        window.get("source_snapshots"),
        field="window.source_snapshots",
        errors=errors,
        unavailable=unavailable,
    )
    return {
        "raw": window,
        "kind": kind,
        "start_at": start_at,
        "end_at": end_at,
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
    evidence_status = root.get("evidence_status")
    if evidence_status not in EVIDENCE_STATUSES:
        errors.append(f"evidence_status must be one of {sorted(EVIDENCE_STATUSES)}")
        evidence_status = "workflow_observation"
    policy = _validate_policy(root, errors=errors, unavailable=unavailable)
    window = _validate_window(
        root,
        rollout_at=policy["rollout_at"],
        errors=errors,
        unavailable=unavailable,
    )
    records = _validate_records(root.get("records"), field="records", errors=errors)
    red_main = _summarize_red_main_events(root.get("red_main_events"), errors=errors)
    return (
        {
            "input_schema": input_schema,
            "repo": repo,
            "evidence_status": evidence_status,
            "policy": policy,
            "window": window,
            "records": records,
            "red_main": red_main,
        },
        errors,
        unavailable,
    )


def _baseline_report(value: Any, *, evidence_status: str) -> dict[str, Any]:
    """Validate and summarize an optional compatible pre-rollout baseline."""
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
    errors: list[str] = []
    unavailable: list[str] = []
    sources = _validate_source_snapshots(
        baseline.get("source_snapshots"),
        field="baseline.source_snapshots",
        errors=errors,
        unavailable=unavailable,
    )
    records = _validate_records(baseline.get("records"), field="baseline.records", errors=errors)
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
            "source_snapshots": sources,
        }
    status = "fixture_only" if evidence_status == "workflow_fixture" else "available"
    return {
        "status": status,
        "source_snapshots": sources,
        "metrics": _summarize_records(records),
    }


def _invalid_contract_report(
    *,
    input_schema: Any,
    evidence_status: str,
    input_path: str | None,
    errors: list[str],
    unavailable: list[str],
    red_main: dict[str, Any],
) -> dict[str, Any]:
    """Build a consistent invalid-contract report."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input_schema_version": input_schema,
        "status": "invalid_contract",
        "evidence_status": evidence_status,
        "input_path": input_path,
        "claim_boundary": CLAIM_BOUNDARY,
        "blockers": ["observation input contract is invalid"],
        "validation_errors": errors,
        "limitations": unavailable,
        "metrics": None,
        "baseline": {"status": "not_available", "reason": "current contract is invalid"},
        "comparison": {"status": "not_available", "reason": "current contract is invalid"},
        "red_main": red_main,
    }


def analyze_observation(payload: Any, *, input_path: str | None = None) -> dict[str, Any]:
    """Validate and summarize an observation-window payload."""
    root = _mapping(payload)
    if root is None:
        report = _unavailable_report(
            "observation input must be a JSON object", input_path=input_path
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
            errors=errors,
            unavailable=unavailable,
            red_main=contract["red_main"],
        )
    metrics = _summarize_records(contract["records"])
    limitations = list(unavailable)
    if not contract["records"]:
        limitations.append("normal-throughput window contains no PR records")
    if contract["red_main"]["by_classification"].get("unknown", 0):
        limitations.append("red-main events with unknown attribution remain excluded")
    baseline = _baseline_report(root.get("baseline"), evidence_status=contract["evidence_status"])
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
        "input_path": input_path,
        "repo": contract["repo"],
        "policy": {
            "policy_id": str(policy["raw"].get("policy_id", "risk-tiered-stale-base")),
            "rollout_at": rollout_at.isoformat().replace("+00:00", "Z") if rollout_at else None,
            "source_snapshots": policy["source_snapshots"],
        },
        "window": {
            "window_id": str(window["raw"].get("window_id", "")),
            "kind": window["kind"],
            "start_at": start_at.isoformat().replace("+00:00", "Z") if start_at else None,
            "end_at": end_at.isoformat().replace("+00:00", "Z") if end_at else None,
            "source_snapshots": window["source_snapshots"],
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "blockers": [],
        "validation_errors": [],
        "limitations": limitations,
        "metrics": metrics,
        "baseline": baseline,
        "comparison": _comparison(metrics, baseline),
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
            payload = json.loads(args.input.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report = _unavailable_report(
                f"observation input could not be read: {exc}", input_path=str(args.input)
            )
            report["status"] = "invalid_contract"
            report["validation_errors"] = [str(exc)]
        else:
            report = analyze_observation(payload, input_path=str(args.input))

    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        _write(args.output, serialized)
    if args.markdown_output:
        _write(args.markdown_output, render_markdown(report))
    print(serialized, end="")
    return 0 if report["status"] in {"available", "fixture_only"} else 2


if __name__ == "__main__":
    sys.exit(main())
